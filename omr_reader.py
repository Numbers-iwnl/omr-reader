from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


# ---------------------------
# Template (fallback / modelo antigo)
# ---------------------------
DEFAULT_TEMPLATE: Dict = {
    "warp_width": 1200,
    "warp_height": 1700,
    "answers_y0_frac": 0.62,
    "answers_y1_frac": 0.99,
    "panels": [
        {"x":[72.88,114.36,154.79,189.73,217.33],
         "y":[1075.0,1121.38,1155.33,1167.17,1195.33,1256.0,1285.0,1315.0,1346.8,1362.8,1375.5,1405.6,1435.0,1480.5,1587.5],
         "r":14.0},
        {"x":[259.88,291.64,320.21,361.74,404.33],
         "y":[1074.75,1119.88,1164.83,1193.67,1195.5,1224.4,1254.83,1284.6,1314.6,1357.57,1394.0,1435.86,1467.0,1496.6,1575.0],
         "r":12.0},
        {"x":[456.53,494.67,518.55,556.67,594.5],
         "y":[1092.0,1132.17,1163.6,1192.6,1224.0,1252.33,1283.0,1314.25,1364.86,1404.8,1440.75,1495.5,1552.0,1577.0,1600.2],
         "r":15.0},
        {"x":[639.12,678.58,719.71,753.36,782.24],
         "y":[1073.25,1122.2,1163.71,1198.6,1244.88,1281.83,1316.0,1342.75,1373.75,1404.5,1450.86,1535.6,1561.29,1578.67,1595.33],
         "r":15.0},
        {"x":[828.59,861.31,899.75,936.17,972.6],
         "y":[1096.0,1134.0,1163.0,1165.8,1192.33,1222.25,1252.0,1282.5,1312.4,1360.71,1404.4,1434.17,1463.8,1493.75,1567.75],
         "r":15.0},
        {"x":[1020.07,1041.6,1060.33,1093.25,1148.29],
         "y":[1075.0,1104.4,1133.0,1163.0,1192.33,1224.0,1264.67,1312.33,1343.75,1371.5,1401.75,1431.6,1461.33,1490.33,1567.5],
         "r":15.0},
    ],
}

TEMPLATE_PATH = Path(__file__).with_name("template.json")


def load_template() -> Dict:
    if TEMPLATE_PATH.exists():
        tpl = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
        for k in ("warp_width", "warp_height", "answers_y0_frac", "answers_y1_frac", "panels"):
            if k not in tpl:
                raise ValueError(f"template.json sem chave obrigatória: {k}")
        if len(tpl["panels"]) != 6:
            raise ValueError("template.json deve ter 6 painéis.")
        return tpl
    return DEFAULT_TEMPLATE


# ---------------------------
# Unicode-safe IO (Windows)
# ---------------------------
def imread_unicode(path: str) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não consegui abrir a imagem.")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """
    cv2.imwrite pode falhar com paths Unicode no Windows.
    Workaround: imencode + tofile. :contentReference[oaicite:3]{index=3}
    """
    p = Path(path)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Falha ao codificar imagem para salvar ({ext}).")
    buf.tofile(str(p))


# ---------------------------
# Perspectiva (warp)
# ---------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_sheet(image_bgr: np.ndarray, out_w: int = 1200, out_h: int = 1700) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        edges = cv2.Canny(blur, 30, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        raise ValueError("Não encontrei contornos suficientes. Pegue a folha inteira na foto.")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    sheet = None
    for c in cnts[:12]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet = approx.reshape(4, 2).astype("float32")
            break

    if sheet is None:
        raise ValueError("Não consegui achar o retângulo da folha (4 cantos).")

    rect = _order_points(sheet)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, M, (out_w, out_h))


# ---------------------------
# Preprocessamento
# ---------------------------
def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(gray)


def binarize_for_marks(gray: np.ndarray, block_size: int = 51, C: int = 7) -> np.ndarray:
    """
    adaptiveThreshold precisa de blockSize ímpar. :contentReference[oaicite:4]{index=4}
    """
    if block_size % 2 == 0:
        block_size += 1
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    return th


def remove_long_lines(th: np.ndarray) -> np.ndarray:
    """
    Remove grades/caixas por morfologia. :contentReference[oaicite:5]{index=5}
    """
    h, w = th.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 18)))
    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
    lines = cv2.bitwise_or(h_lines, v_lines)
    return cv2.bitwise_and(th, cv2.bitwise_not(lines))


# ---------------------------
# Fine alignment (mantido, mas caro; Hough pode ser lento) :contentReference[oaicite:6]{index=6}
# ---------------------------
def _detect_circles(gray_crop: np.ndarray, y_min: int, y_max: int) -> Optional[np.ndarray]:
    roi = gray_crop[y_min:y_max, :]
    eq = apply_clahe(roi)
    blur = cv2.GaussianBlur(eq, (5, 5), 0)

    best = None
    best_count = 0
    for param2 in (22, 21, 20, 19, 18, 17, 16, 15, 14, 13):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.1, minDist=16,
            param1=120, param2=param2,
            minRadius=7, maxRadius=22
        )
        if circles is None:
            continue
        c = circles[0]
        if c.shape[0] > best_count:
            best = c
            best_count = c.shape[0]
        if best_count >= 80:
            break
    if best is None:
        return None
    best[:, 1] += y_min
    return best


def _nearest_idx(val: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - val)))


def _estimate_panel_affine(
    circles_xy: List[Tuple[float, float]],
    x_template: List[float],
    y_template: List[float],
    min_samples: int = 12,
    ransac_thresh: float = 3.0,
) -> Optional[np.ndarray]:
    if len(circles_xy) < min_samples:
        return None

    x_tpl = np.array(x_template, dtype=np.float32)
    y_tpl = np.array(y_template, dtype=np.float32)

    src_pts, dst_pts = [], []
    for x, y in circles_xy:
        col = _nearest_idx(x, x_tpl)
        row = _nearest_idx(y, y_tpl)
        src_pts.append([x_tpl[col], y_tpl[row]])
        dst_pts.append([x, y])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )
    return M


def _apply_affine(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(nx), float(ny)


# ---------------------------
# Score
# ---------------------------
def _fill_score(mask_bin: np.ndarray, cx: float, cy: float, rr: int) -> float:
    h, w = mask_bin.shape
    x = int(round(cx))
    y = int(round(cy))
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), rr, 255, -1)
    return float(cv2.countNonZero(cv2.bitwise_and(mask_bin, mask_bin, mask=mask)))


def _best_local_score(mask_bin: np.ndarray, cx: float, cy: float, rr: int, search_px: int) -> Tuple[float, float, float]:
    if search_px <= 0:
        return _fill_score(mask_bin, cx, cy, rr), cx, cy

    best_s = -1.0
    best_x, best_y = cx, cy
    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            s = _fill_score(mask_bin, cx + dx, cy + dy, rr)
            if s > best_s:
                best_s = s
                best_x, best_y = cx + dx, cy + dy
    return best_s, best_x, best_y


@dataclass
class Answer:
    question: int
    answer: str
    status: str
    score: float


@dataclass
class OMRConfig:
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    adaptive_block_size: int = 51
    adaptive_C: int = 7

    fill_frac: float = 0.35

    ink_erode_kernel: int = 1
    ink_erode_iterations: int = 0

    use_thick_mask: bool = True
    thick_min_dist: float = 1.3

    # decisão (normalizado por questão)
    blank_min_score: float = 3.0
    ambig_gap: float = 1.0
    multi_second_min_score: float = 0.0
    normalize_per_question: bool = True

    # NEW: micro-ajuste de centro
    local_search_px: int = 2  # 0 desliga; 2–3 costuma fechar os últimos erros

    # NEW: fallback quando thick falha na questão
    fallback_to_ink: bool = True
    thick_fallback_max: float = 2.0  # se max(thick) <= isso, usa ink nessa questão

    enable_fine_alignment: bool = False
    fine_alignment_min_samples: int = 12
    fine_y0_frac: float = 0.15
    fine_y1_frac: float = 0.80
    fine_ransac_thresh: float = 3.0


def read_omr_answers(image_path: str, config: Optional[OMRConfig] = None, debug_dir: Optional[str] = None) -> List[Answer]:
    cfg = config or OMRConfig()
    tpl = load_template()

    warp_w = int(tpl["warp_width"])
    warp_h = int(tpl["warp_height"])
    y0_frac = float(tpl["answers_y0_frac"])
    y1_frac = float(tpl["answers_y1_frac"])

    warped = warp_sheet(imread_unicode(image_path), out_w=warp_w, out_h=warp_h)
    gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    y0_abs = int(warp_h * y0_frac)
    y1_abs = int(warp_h * y1_frac)
    gray = gray_full[y0_abs:y1_abs, :]

    if cfg.use_clahe:
        gray = apply_clahe(gray, cfg.clahe_clip_limit, cfg.clahe_grid)

    th_inv = binarize_for_marks(gray, cfg.adaptive_block_size, cfg.adaptive_C)
    th_clean = remove_long_lines(th_inv)

    k = max(1, int(cfg.ink_erode_kernel))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    ink = cv2.erode(th_clean, kernel, iterations=int(cfg.ink_erode_iterations))

    if cfg.use_thick_mask:
        # distanceTransform para manter “tinta grossa” (remove anel/letra) :contentReference[oaicite:7]{index=7}
        dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
        thick = (dist >= float(cfg.thick_min_dist)).astype(np.uint8) * 255
    else:
        thick = ink

    tpl_panels = tpl["panels"]
    tpl_y_crop = [[float(y) - y0_abs for y in p["y"]] for p in tpl_panels]

    panel_affines: List[Optional[np.ndarray]] = [None] * 6
    if cfg.enable_fine_alignment:
        H = gray.shape[0]
        y_min = int(H * cfg.fine_y0_frac)
        y_max = int(H * cfg.fine_y1_frac)

        circles = _detect_circles(gray, y_min=y_min, y_max=y_max)
        if circles is not None:
            circles = np.round(circles).astype(int)
            circles_xy_all = [(float(x), float(y)) for x, y, _r in circles]

            panel_x_bounds_frac = (0.00, 0.20, 0.36, 0.52, 0.67, 0.82, 1.00)
            bounds = [int(warp_w * f) for f in panel_x_bounds_frac]

            for p in range(6):
                xL, xR = bounds[p], bounds[p + 1]
                pts = [(x, y) for (x, y) in circles_xy_all if xL <= x < xR]
                M = _estimate_panel_affine(
                    pts,
                    tpl_panels[p]["x"],
                    tpl_y_crop[p],
                    min_samples=cfg.fine_alignment_min_samples,
                    ransac_thresh=cfg.fine_ransac_thresh,
                )
                panel_affines[p] = M

    letters = ["A", "B", "C", "D", "E"]
    out: List[Answer] = []
    overlay = warped.copy()

    for p in range(6):
        x_centers = tpl_panels[p]["x"]
        y_centers = tpl_y_crop[p]
        r = float(tpl_panels[p]["r"])
        rr = max(2, int(r * cfg.fill_frac))
        M = panel_affines[p]

        for row in range(15):
            q = 1 + p * 15 + row

            # calcula scores no thick e no ink (pra fallback)
            scores_thick = []
            scores_ink = []
            coords_best = []  # (x,y) do melhor local para overlay

            for col in range(5):
                cx = float(x_centers[col])
                cy = float(y_centers[row])
                if M is not None:
                    cx, cy = _apply_affine(M, cx, cy)

                sT, bx, by = _best_local_score(thick, cx, cy, rr, cfg.local_search_px)
                sI, _, _ = _best_local_score(ink, cx, cy, rr, cfg.local_search_px)

                scores_thick.append(sT)
                scores_ink.append(sI)
                coords_best.append((bx, by))

            scT = np.array(scores_thick, dtype=np.float32)
            scI = np.array(scores_ink, dtype=np.float32)

            # fallback por questão
            sc = scT
            coords_use = coords_best
            if cfg.fallback_to_ink and float(np.max(scT)) <= cfg.thick_fallback_max:
                sc = scI  # usa ink quando thick não captou nada relevante

            # normaliza por questão
            if cfg.normalize_per_question:
                sc_adj = sc - float(np.min(sc))
            else:
                sc_adj = sc

            order = np.argsort(sc_adj)[::-1]
            best = int(order[0])
            second = int(order[1])

            best_adj = float(sc_adj[best])
            second_adj = float(sc_adj[second])

            if best_adj < cfg.blank_min_score:
                out.append(Answer(q, "", "blank", float(sc[best])))
                continue

            status = "ok"
            if (best_adj - second_adj) < cfg.ambig_gap and second_adj >= cfg.multi_second_min_score:
                status = "ambiguous"

            out.append(Answer(q, letters[best], status, float(sc[best])))

            if debug_dir is not None:
                cx, cy = coords_use[best]
                cv2.circle(overlay, (int(cx), int(cy + y0_abs)), 10, (0, 255, 0), 2)

    if debug_dir is not None:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)

        imwrite_unicode(str(d / "01_warped.png"), warped)
        imwrite_unicode(str(d / "02_gray_answers.png"), gray)
        imwrite_unicode(str(d / "03_threshold_inv.png"), th_inv)
        imwrite_unicode(str(d / "04_threshold_clean.png"), th_clean)
        imwrite_unicode(str(d / "05_ink.png"), ink)
        imwrite_unicode(str(d / "06_thick.png"), thick)
        imwrite_unicode(str(d / "07_overlay.png"), overlay)

    return out