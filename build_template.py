from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ---------- Unicode-safe IO (Windows) ----------
def imread_unicode(path: str) -> np.ndarray:
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não consegui abrir a imagem.")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    p = Path(path)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError("Falha ao codificar imagem para salvar.")
    buf.tofile(str(p))


# ---------- Warp ----------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_sheet(image_bgr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("Não achei contornos da folha. Pegue as 4 bordas na foto.")

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


# ---------- Preprocess ----------
def apply_clahe(gray: np.ndarray, clip_limit=2.0, grid=(8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(gray)


def binarize(gray: np.ndarray, block_size: int = 51, C: int = 7) -> np.ndarray:
    # block_size precisa ser ímpar
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
    Extrai linhas longas (horiz/vert) com morfologia e remove do binário,
    pra projeções ficarem dominadas pelas bolhas (não pelas caixas/grades).
    """
    h, w = th.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 18)))

    h_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(th, cv2.MORPH_OPEN, v_kernel, iterations=1)
    lines = cv2.bitwise_or(h_lines, v_lines)

    cleaned = cv2.bitwise_and(th, cv2.bitwise_not(lines))
    return cleaned


# ---------- Peak picking ----------
def smooth_1d(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x.astype(np.float32), k, mode="same")


def pick_peaks(arr: np.ndarray, n: int, min_dist: int, margin: int) -> List[int]:
    """
    Escolhe n picos por greedy max, com distância mínima entre picos.
    """
    a = arr.astype(np.float32).copy()
    L = len(a)

    # ignora margens (bordas das caixas)
    margin = max(0, int(margin))
    if margin > 0:
        a[:margin] = -1e18
        a[L - margin:] = -1e18

    peaks = []
    for _ in range(n):
        i = int(np.argmax(a))
        if a[i] < 0:
            break
        peaks.append(i)
        lo = max(0, i - min_dist)
        hi = min(L, i + min_dist + 1)
        a[lo:hi] = -1e18

    peaks.sort()
    return peaks


def main(img_path: str):
    # fixos
    warp_w, warp_h = 1200, 1700

    # recorte das respostas (corta antes do barcode!)
    answers_y0_frac = 0.62
    answers_y1_frac = 0.93

    panel_x_bounds_frac = (0.00, 0.20, 0.36, 0.52, 0.67, 0.82, 1.00)
    bounds = [int(warp_w * f) for f in panel_x_bounds_frac]

    img = imread_unicode(img_path)
    warped = warp_sheet(img, warp_w, warp_h)
    gray_full = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    y0 = int(warp_h * answers_y0_frac)
    y1 = int(warp_h * answers_y1_frac)
    crop = gray_full[y0:y1, :]

    crop = apply_clahe(crop)
    th = binarize(crop, block_size=51, C=7)
    th_clean = remove_long_lines(th)

    H, W = th_clean.shape

    panels_out = []
    debug = cv2.cvtColor(th_clean, cv2.COLOR_GRAY2BGR)

    for p in range(6):
        xL, xR = bounds[p], bounds[p + 1]
        panel = th_clean[:, xL:xR]

        # Projeções
        col_proj = panel.sum(axis=0)  # tamanho = panel_w
        row_proj = panel.sum(axis=1)  # tamanho = H

        # Suaviza
        col_s = smooth_1d(col_proj, win=max(9, (xR - xL) // 25))
        row_s = smooth_1d(row_proj, win=max(9, H // 40))

        # Picos esperados
        col_min_dist = max(8, (xR - xL) // 8)
        row_min_dist = max(6, H // 22)

        col_margin = int((xR - xL) * 0.08)
        row_margin = int(H * 0.03)

        col_peaks = pick_peaks(col_s, n=5, min_dist=col_min_dist, margin=col_margin)
        row_peaks = pick_peaks(row_s, n=15, min_dist=row_min_dist, margin=row_margin)

        if len(col_peaks) != 5 or len(row_peaks) != 15:
            raise ValueError(
                f"Painel {p+1}: não consegui achar picos suficientes "
                f"(cols={len(col_peaks)}, rows={len(row_peaks)}). "
                f"Tente ajustar answers_y1_frac ou block_size/C."
            )

        x_centers = [float(xL + i) for i in col_peaks]
        y_centers_full = [float(y0 + j) for j in row_peaks]

        # raio estimado pela distância entre colunas (bem estável)
        dxs = [x_centers[i + 1] - x_centers[i] for i in range(4)]
        r = float(np.median(dxs) * 0.33)  # ~1/3 do espaçamento costuma bater bem

        panels_out.append({"x": x_centers, "y": y_centers_full, "r": r})

        # debug: desenha uma coluna de círculos por painel
        for cx in x_centers:
            for cy in y_centers_full:
                cv2.circle(debug, (int(cx), int(cy - y0)), int(max(8, r)), (0, 255, 0), 1)

    template = {
        "warp_width": warp_w,
        "warp_height": warp_h,
        "answers_y0_frac": answers_y0_frac,
        "answers_y1_frac": answers_y1_frac,
        "panels": panels_out,
    }

    out_json = Path(__file__).with_name("template.json")
    out_dbg = Path(__file__).with_name("template_debug.png")
    out_th = Path(__file__).with_name("template_threshold_clean.png")
    out_warp = Path(__file__).with_name("template_warped.png")

    out_json.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    imwrite_unicode(str(out_dbg), debug)
    imwrite_unicode(str(out_th), th_clean)
    imwrite_unicode(str(out_warp), warped)

    print("OK! template.json gerado em:", out_json)
    print("Debug:", out_dbg)
    print("Threshold clean:", out_th)
    print("Warp:", out_warp)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("uso: python build_template.py <imagem.jpg>")
        raise SystemExit(2)
    main(sys.argv[1])