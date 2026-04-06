"""
omr_reader.py  –  Leitor de Cartão-Resposta (90 questões, 6 painéis × 15 × 5)
==============================================================================

ESTRATÉGIA (v3 – robusta para fotos de celular)
------------------------------------------------
1. Warp inicial  →  detecta o maior quadrilátero da folha; se não achar,
   redimensiona sem correção (melhor que travar).
2. Detecção dos 6 painéis  →  cada painel tem uma caixa retangular com borda
   preta, proporção ~0.40 e área ~84 k px². Usamos essas caixas como âncoras,
   eliminando o erro residual do warp.
3. Scoring por painel  →  posições dos círculos codificadas como frações
   normalizadas (calibradas no template de referência) e projetadas na caixa
   real detectada.
4. Local search ±8 px  →  micro-ajuste fino ao redor do centro estimado.
5. Threshold adaptativo  →  blank_threshold calculado por percentil dos scores
   do painel inteiro (imune à iluminação e tipo de caneta).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Posições normalizadas dos círculos dentro de cada caixa de painel
# Calibradas no template de referência (template_warped.png, 1200×1700).
# xs[p][col]  →  fração da LARGURA  do bounding-box do painel (col 0-4 = A-E)
# ys[p][row]  →  fração da ALTURA   do bounding-box do painel (row 0-14)
# ---------------------------------------------------------------------------
PANEL_X_NORM = [
    [0.1129, 0.2796, 0.4462, 0.6129, 0.8710],  # painel 1
    [0.2646, 0.4180, 0.5556, 0.7249, 0.8571],  # painel 2
    [0.2634, 0.4086, 0.5591, 0.7043, 0.8817],  # painel 3
    [0.2460, 0.4225, 0.5668, 0.6952, 0.8342],  # painel 4
    [0.2620, 0.4171, 0.5561, 0.7112, 0.8556],  # painel 5
    [0.1141, 0.2663, 0.4185, 0.7120, 0.8641],  # painel 6
]

PANEL_Y_NORM = [
    [0.1099, 0.1746, 0.2392, 0.3017, 0.3685, 0.4332, 0.4978,
     0.5647, 0.6272, 0.6940, 0.7565, 0.8211, 0.8858, 0.9526, 0.9800],
    [0.1080, 0.1728, 0.2376, 0.3024, 0.3672, 0.4320, 0.4968,
     0.5616, 0.6263, 0.6890, 0.7538, 0.8207, 0.8855, 0.9503, 0.9800],
    [0.0540, 0.1102, 0.1749, 0.2419, 0.2981, 0.3629, 0.4276,
     0.4924, 0.5572, 0.6220, 0.6868, 0.7516, 0.8164, 0.8834, 0.9503],
    [0.0540, 0.1080, 0.1728, 0.2376, 0.3024, 0.3672, 0.4298,
     0.4946, 0.5616, 0.6263, 0.6911, 0.7559, 0.8207, 0.8855, 0.9525],
    [0.0499, 0.1085, 0.1735, 0.2451, 0.3037, 0.3666, 0.4317,
     0.4989, 0.5640, 0.6291, 0.6941, 0.7570, 0.8221, 0.8872, 0.9610],
    [0.0459, 0.1092, 0.1812, 0.2489, 0.3035, 0.3690, 0.4345,
     0.5022, 0.5742, 0.6397, 0.6987, 0.7598, 0.8319, 0.8865, 0.9607],
]

# Raio de referência por painel (pixels, no warp 1200×1700)
PANEL_R_REF  = [10.23, 9.08, 9.08, 8.75, 9.24, 9.24]
REF_PANEL_H  = 464.0   # altura de referência dos painéis no warp 1200×1700

# Posições fixas dos painéis no warp 1200×1700 (fallback)
PANELS_REF = [
    (36,  1040, 186, 464),
    (227, 1042, 189, 463),
    (421, 1044, 186, 463),
    (611, 1044, 187, 463),
    (801, 1046, 187, 461),
    (991, 1048, 184, 458),
]

WARP_W        = 1200
WARP_H        = 1700
FILL_FRAC     = 0.60   # fração do raio para o círculo de scoring
LOCAL_SEARCH  = 6      # pixels de micro-busca ao redor do centro
THICK_MIN     = 1.3    # distanceTransform mínimo para "tinta grossa"
BLANK_PCTILE  = 65     # percentil abaixo do qual a bolha é "blank"
AMBIG_RATIO   = 0.55   # 3º melhor / 1º melhor → ambiguous (tolera 1 coluna elevada)


# ---------------------------------------------------------------------------
# I/O unicode-safe (Windows)
# ---------------------------------------------------------------------------
def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não consegui abrir: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = Path(path).suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Falha ao salvar imagem ({ext})")
    buf.tofile(path)


# ---------------------------------------------------------------------------
# Warp de perspectiva
# ---------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s    = pts.sum(axis=1)
    d    = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # topo-esq
    rect[1] = pts[np.argmin(d)]     # topo-dir
    rect[2] = pts[np.argmax(s)]     # baixo-dir
    rect[3] = pts[np.argmax(d)]     # baixo-esq
    return rect


def _find_sheet_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w    = gray.shape
    min_a   = w * h * 0.20

    for blur_k, lo, hi, dil in [
        (5, 50, 150, 5),
        (5, 30, 120, 7),
        (9, 20, 80,  9),
    ]:
        blr   = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blr, lo, hi)
        edges = cv2.dilate(edges, np.ones((dil, dil), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
            if cv2.contourArea(c) < min_a:
                break
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype("float32")
    return None


def warp_sheet(image_bgr: np.ndarray,
               out_w: int = WARP_W,
               out_h: int = WARP_H) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = _find_sheet_quad(gray)
    dst  = np.array([[0, 0], [out_w-1, 0],
                     [out_w-1, out_h-1], [0, out_h-1]], dtype="float32")
    if quad is not None:
        rect = _order_points(quad)
        M    = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image_bgr, M, (out_w, out_h))
    # Fallback: só redimensiona
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Detecção dos painéis
# ---------------------------------------------------------------------------
def _iou(a: Tuple, b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax2,bx2) - max(ax1,bx1))
    ih = max(0, min(ay2,by2) - max(ay1,by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[Tuple]) -> List[Tuple]:
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep  = []
    for b in boxes:
        if not any(_iou(b, k) > 0.5 for k in keep):
            keep.append(b)
    return keep


def _detect_panel_boxes(gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Detecta as 6 caixas de painel de resposta."""
    h, w   = gray.shape
    min_h  = h * 0.18
    min_a  = h * w * 0.025
    max_a  = h * w * 0.12

    all_cands: List[Tuple] = []

    for lo, hi, dil in [(40, 140, 5), (30, 100, 7), (60, 180, 5)]:
        blr   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blr, lo, hi)
        edges = cv2.dilate(edges, np.ones((dil, dil), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (min_a < a < max_a):
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)
            if len(approx) != 4:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            asp = bw / bh if bh > 0 else 0
            if 0.28 < asp < 0.60 and bh >= min_h:
                all_cands.append((bx, by, bw, bh))

    cands = _nms(all_cands)
    cands.sort(key=lambda b: b[0])

    if len(cands) >= 6:
        return cands[:6]

    # Completa faltantes por interpolação
    return _fill_missing(cands, gray.shape)


def _fill_missing(found: List[Tuple], shape: Tuple) -> List[Tuple]:
    if not found:
        return _scale_ref(shape)

    med_w = int(np.median([b[2] for b in found]))
    med_h = int(np.median([b[3] for b in found]))
    med_y = int(np.median([b[1] for b in found]))
    iw    = shape[1]

    # X esperados (normalizados) dos 6 painéis
    x_fracs = [0.040, 0.222, 0.388, 0.556, 0.722, 0.888]
    result  = list(found)

    for xf in x_fracs:
        x_est = int(iw * xf)
        if not any(abs(b[0] - x_est) < med_w * 0.4 for b in result):
            result.append((x_est, med_y, med_w, med_h))

    result.sort(key=lambda b: b[0])
    return result[:6]


def _scale_ref(shape: Tuple) -> List[Tuple[int,int,int,int]]:
    h, w  = shape
    sx, sy = w / WARP_W, h / WARP_H
    return [(int(x*sx), int(y*sy), int(bw*sx), int(bh*sy))
            for x, y, bw, bh in PANELS_REF]


# ---------------------------------------------------------------------------
# Pré-processamento de painel
# ---------------------------------------------------------------------------
def _preprocess(gray_panel: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray_panel)
    blur  = cv2.GaussianBlur(eq, (5, 5), 0)
    th    = cv2.adaptiveThreshold(blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 51, 7)
    th    = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    ph, pw = th.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, pw//8), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, ph//8)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(th, cv2.MORPH_OPEN, hk),
        cv2.morphologyEx(th, cv2.MORPH_OPEN, vk)
    )
    ink   = cv2.bitwise_and(th, cv2.bitwise_not(lines))
    dist  = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thick = (dist >= THICK_MIN).astype(np.uint8) * 255
    return ink, thick


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score(mask: np.ndarray, cx: float, cy: float, r: int) -> float:
    h, w  = mask.shape
    x, y  = int(round(cx)), int(round(cy))
    if not (0 <= x < w and 0 <= y < h):
        return 0.0
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (x, y), r, 255, -1)
    return float(cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=m)))


def _best(mask: np.ndarray, cx: float, cy: float,
          r: int, search: int) -> Tuple[float, float, float]:
    bs, bx, by = -1.0, cx, cy
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            s = _score(mask, cx+dx, cy+dy, r)
            if s > bs:
                bs, bx, by = s, cx+dx, cy+dy
    return bs, bx, by


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    question: int
    answer:   str
    status:   str
    score:    float


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def read_omr_answers(
    image_path: str,
    config=None,
    debug_dir: Optional[str] = None,
) -> List[Answer]:

    img    = imread_unicode(image_path)
    warped = warp_sheet(img)
    gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    boxes  = _detect_panel_boxes(gray)
    while len(boxes) < 6:
        boxes = _fill_missing(boxes, gray.shape)
    boxes = boxes[:6]

    letters = ["A", "B", "C", "D", "E"]
    out: List[Answer] = []
    overlay = warped.copy()

    for p, (bx, by, bw, bh) in enumerate(boxes):
        bx = max(0, min(bx, WARP_W - 1))
        by = max(0, min(by, WARP_H - 1))
        bw = min(bw, WARP_W - bx)
        bh = min(bh, WARP_H - by)

        if bw < 20 or bh < 20:
            for row in range(15):
                out.append(Answer(1 + p*15 + row, "", "blank", 0.0))
            continue

        panel_gray  = gray[by:by+bh, bx:bx+bw]
        ink, thick  = _preprocess(panel_gray)

        r_px = max(3, int(round(PANEL_R_REF[p] * bh / REF_PANEL_H)))
        rr   = max(3, int(round(r_px * FILL_FRAC)))

        # Primeira passagem: coleta scores brutos (max de thick e ink por bolha)
        # cache[row][col] = (raw_score, best_cx, best_cy)
        all_raw: List[float] = []
        cache: List[List[Tuple[float, float, float]]] = []

        for row in range(15):
            row_cache = []
            for col in range(5):
                cx = PANEL_X_NORM[p][col] * bw
                cy = PANEL_Y_NORM[p][row] * bh
                sT, bx2, by2 = _best(thick, cx, cy, rr, LOCAL_SEARCH)
                sI, bx3, by3 = _best(ink,   cx, cy, rr, LOCAL_SEARCH)
                # Usa o maior dos dois; preserva coordenadas do vencedor
                if sT >= sI:
                    raw_s, best_cx, best_cy = sT, bx2, by2
                else:
                    raw_s, best_cx, best_cy = sI, bx3, by3
                row_cache.append((raw_s, best_cx, best_cy))
                all_raw.append(raw_s)
            cache.append(row_cache)

        # Threshold adaptativo por painel (separa marcadas de vazias)
        blank_thr = max(float(np.percentile(all_raw, BLANK_PCTILE)), 8.0)

        for row in range(15):
            q  = 1 + p * 15 + row

            # Scores brutos da linha
            sc_raw = np.array([cache[row][col][0] for col in range(5)],
                               dtype=np.float32)

            # Correção de baseline por questão:
            # subtrai o mínimo da linha para anular o "chão" uniforme
            # causado pelas letras impressas dentro das bolhas vazias.
            sc = sc_raw - sc_raw.min()

            order  = np.argsort(sc)[::-1]
            best_c = int(order[0])

            best_raw = float(sc_raw[best_c])
            best_adj = float(sc[order[0]])
            third_adj = float(sc[order[2]])  # 3º melhor (tolerante a 1 col. elevada)

            if best_raw < blank_thr:
                out.append(Answer(q, "", "blank", best_raw))
                continue

            status = "ok"
            if best_adj > 0 and third_adj / best_adj > AMBIG_RATIO:
                status = "ambiguous"

            out.append(Answer(q, letters[best_c], status, best_raw))

            if debug_dir is not None:
                abs_x = int(cache[row][best_c][1]) + bx
                abs_y = int(cache[row][best_c][2]) + by
                color = (0, 255, 0) if status == "ok" else (0, 165, 255)
                cv2.circle(overlay, (abs_x, abs_y), rr + 4, color, 2)
                cv2.putText(overlay, str(q), (abs_x - rr - 2, abs_y - rr - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    if debug_dir is not None:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        imwrite_unicode(str(d / "01_warped.png"),      warped)
        imwrite_unicode(str(d / "02_overlay.png"),     overlay)

        box_vis = warped.copy()
        for i, (bx, by, bw, bh) in enumerate(boxes):
            cv2.rectangle(box_vis, (bx, by), (bx+bw, by+bh), (255, 0, 0), 3)
            cv2.putText(box_vis, f"P{i+1}", (bx+5, by+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        imwrite_unicode(str(d / "03_panel_boxes.png"), box_vis)

    return out


# ---------------------------------------------------------------------------
# Compatibilidade retroativa
# ---------------------------------------------------------------------------
@dataclass
class OMRConfig:
    """Aceito por app.py / calibrate.py – parâmetros ignorados na v3."""
    use_clahe:             bool  = True
    fill_frac:             float = FILL_FRAC
    local_search_px:       int   = LOCAL_SEARCH
    enable_fine_alignment: bool  = False
