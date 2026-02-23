\
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np


# ---------------------------
# Util: robust image reading on Windows (paths with accents, etc.)
# ---------------------------
def imread_unicode(path: str) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não consegui abrir a imagem. Verifique se o arquivo existe e é um JPG/PNG válido.")
    return img


# ---------------------------
# Perspectiva: achar a folha e "endireitar"
# ---------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def warp_sheet(image_bgr: np.ndarray, out_w: int = 1200, out_h: int = 1700) -> np.ndarray:
    """
    Detecta o contorno principal (a folha) e aplica perspective transform para um tamanho fixo.
    Isso dá robustez contra rotação, inclinação e foto descentralizada.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("Não encontrei contornos suficientes. Tente uma foto com mais contraste (folha inteira aparecendo).")

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    sheet = None
    for c in cnts[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            sheet = approx.reshape(4, 2).astype("float32")
            break

    if sheet is None:
        raise ValueError("Não consegui achar o retângulo da folha. Tente afastar um pouco a câmera e pegar as 4 bordas.")

    rect = _order_points(sheet)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_bgr, M, (out_w, out_h))
    return warped


# ---------------------------
# OMR: detectar bolhas (círculos), montar grade e ler preenchimento
# ---------------------------
def _kmeans_1d(data: np.ndarray, k: int, iters: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    data = data.astype(np.float32)
    qs = np.linspace(0, 1, k, endpoint=False) + 0.5 / k
    centers = np.quantile(data, qs)

    for _ in range(iters):
        labels = np.argmin(np.abs(data[:, None] - centers[None, :]), axis=1)
        new_centers = np.array(
            [data[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)],
            dtype=np.float32,
        )
        if np.allclose(new_centers, centers, atol=0.05):
            centers = new_centers
            break
        centers = new_centers

    return centers, labels


def _detect_circles(gray_crop: np.ndarray) -> np.ndarray:
    """
    Detecta círculos via Hough. Tentamos algumas sensibilidades e escolhemos a primeira aceitável.
    """
    blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)

    # Param2 menor => mais sensível (mais falsos positivos). Vamos do mais "conservador" pro mais sensível.
    for param2 in (20, 19, 18, 17):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=18,
            param1=100,
            param2=param2,
            minRadius=7,
            maxRadius=18,
        )
        if circles is not None and circles.shape[1] >= 320:
            return circles[0]

    # último chute
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=16,
        param1=90,
        param2=17,
        minRadius=7,
        maxRadius=18,
    )
    if circles is None:
        raise ValueError("Não consegui detectar as bolhas. Tente uma foto mais nítida e com menos sombra.")
    return circles[0]


def _mean_intensity(gray: np.ndarray, cx: float, cy: float, r: float, frac: float = 0.35) -> float:
    rr = max(2, int(r * frac))
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), rr, 255, -1)
    return float(cv2.mean(gray, mask=mask)[0])


@dataclass
class Answer:
    question: int
    answer: str          # "A".."E" ou "" (em branco) ou "MULTI"
    status: str          # "ok" | "blank" | "ambiguous" | "panel_missing"
    score: float         # métrica (aqui: média de cinza do marcado -> menor é mais escuro)


@dataclass
class OMRConfig:
    warp_width: int = 1200
    warp_height: int = 1700

    # Região (em fração da altura) onde ficam as respostas 1..90
    answers_y0_frac: float = 0.62
    answers_y1_frac: float = 0.99

    # Fronteiras (em fração da largura) que separam os 6 painéis (1-15, 16-30, ..., 76-90)
    # Baseado no template da foto enviada.
    panel_x_bounds_frac: Tuple[float, ...] = (0.00, 0.20, 0.36, 0.52, 0.67, 0.82, 1.00)

    # Heurísticas de decisão
    blank_delta: float = 12.0       # quão mais escuro que a mediana precisa ser pra considerar marcado
    ambiguous_delta: float = 6.0    # diferença mínima entre 1º e 2º mais escuro para não ser "MULTI"

    # Para evitar que círculos do topo (cor do caderno / língua) estraguem a grade,
    # usamos um offset extra dentro da região de respostas.
    # (Se precisar, ajustamos isso com mais amostras.)
    inner_crop_pad_px: int = 0


def read_omr_answers(image_path: str, config: Optional[OMRConfig] = None, debug_dir: Optional[str] = None) -> List[Answer]:
    """
    Lê uma imagem de cartão-resposta (modelo fixo) e devolve 90 respostas.
    Se debug_dir for informado, salva imagens de debug (warp e overlay).
    """
    cfg = config or OMRConfig()
    img = imread_unicode(image_path)
    warped = warp_sheet(img, out_w=cfg.warp_width, out_h=cfg.warp_height)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    y0 = int(cfg.warp_height * cfg.answers_y0_frac) + cfg.inner_crop_pad_px
    y1 = int(cfg.warp_height * cfg.answers_y1_frac)
    crop = gray[y0:y1, :]

    circles = _detect_circles(crop)
    circles = np.round(circles).astype(int)
    circles_xy = [(int(x), int(y + y0), int(r)) for x, y, r in circles]

    # Painéis
    bounds = [int(cfg.warp_width * f) for f in cfg.panel_x_bounds_frac]

    letters = ["A", "B", "C", "D", "E"]
    out: List[Answer] = []

    # opcional: overlay de debug
    overlay = warped.copy()

    for p in range(6):
        xL, xR = bounds[p], bounds[p + 1]
        pts = [(x, y, r) for (x, y, r) in circles_xy if xL <= x < xR]

        if len(pts) < 40:
            # Sem círculos suficientes nesse painel
            for row in range(15):
                q = 1 + p * 15 + row
                out.append(Answer(q, "", "panel_missing", 0.0))
            continue

        xs = np.array([t[0] for t in pts], dtype=np.float32)
        ys = np.array([t[1] for t in pts], dtype=np.float32)
        rs = np.array([t[2] for t in pts], dtype=np.float32)

        x_centers, _ = _kmeans_1d(xs, 5)
        y_centers, _ = _kmeans_1d(ys, 15)

        x_centers = np.sort(x_centers)
        y_centers = np.sort(y_centers)
        r_med = float(np.median(rs))

        # mapeia cada célula (row,col) para o círculo mais próximo do centro da célula
        cell_map: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
        for x, y, r in pts:
            col = int(np.argmin(np.abs(x_centers - x)))
            row = int(np.argmin(np.abs(y_centers - y)))
            key = (row, col)

            dist = float((x - x_centers[col]) ** 2 + (y - y_centers[row]) ** 2)
            if key not in cell_map or dist < cell_map[key][0]:
                cell_map[key] = (dist, float(x), float(y))

        for row in range(15):
            q = 1 + p * 15 + row

            means = []
            coords = []
            for col in range(5):
                key = (row, col)
                if key in cell_map:
                    _, cx, cy = cell_map[key]
                else:
                    cx, cy = float(x_centers[col]), float(y_centers[row])

                m = _mean_intensity(gray, cx, cy, r_med, frac=0.35)
                means.append(m)
                coords.append((cx, cy))

            means_np = np.array(means, dtype=np.float32)
            best = int(np.argmin(means_np))  # menor média = mais escuro
            sorted_means = np.sort(means_np)
            best_m = float(sorted_means[0])
            second_m = float(sorted_means[1])
            med = float(np.median(means_np))

            if (med - best_m) < cfg.blank_delta:
                out.append(Answer(q, "", "blank", best_m))
            elif (second_m - best_m) < cfg.ambiguous_delta:
                out.append(Answer(q, "MULTI", "ambiguous", best_m))
            else:
                out.append(Answer(q, letters[best], "ok", best_m))

                # debug: desenha um pontinho na opção marcada
                if debug_dir is not None:
                    cx, cy = coords[best]
                    cv2.circle(overlay, (int(cx), int(cy)), 10, (0, 255, 0), 2)

    if debug_dir is not None:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "01_warped.png"), warped)
        cv2.imwrite(str(d / "02_overlay.png"), overlay)

    return out
