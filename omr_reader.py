"""
omr_reader.py  –  Leitor de Cartão-Resposta (6 painéis × 15 questões × 5 opções)
==================================================================================

ESTRATÉGIA (v4.1 – grade auto-detectada com ancoragem na coluna mais à direita)
---------------------------------------------------------------------------
1. Detecta os 6 painéis como caixas retangulares na imagem bruta (sem warp).
2. Dentro de cada painel, usa projeções do thick-mask para localizar
   automaticamente as 15 linhas e 5 colunas de bolhas.
   • Colunas: exige que o pico mais à direita selecionado coincida com o pico
     global mais à direita (elimina falsos picos do número da questão à esquerda).
   • Linhas: picos por amplitude, ordenados por posição.
3. Scoring: max(thick, ink) com local search ±LOCAL_SEARCH px.
4. Threshold adaptativo por painel (percentil BLANK_PCTILE dos scores).
5. Decisão de ambíguo: baseline por questão + ratio 3º/1º.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import combinations

import cv2
import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PANEL_R_REF  = [10.23, 9.08, 9.08, 8.75, 9.24, 9.24]   # raios de ref. (px, warp 1200×1700)
REF_PANEL_H  = 464.0
WARP_W, WARP_H = 1200, 1700
PANELS_REF = [                                             # posições de fallback
    (36,  1040, 186, 464), (227, 1042, 189, 463),
    (421, 1044, 186, 463), (611, 1044, 187, 463),
    (801, 1046, 187, 461), (991, 1048, 184, 458),
]
FILL_FRAC    = 0.60
LOCAL_SEARCH = 6
THICK_MIN    = 1.3
BLANK_PCTILE = 65
AMBIG_RATIO  = 0.55   # 3º / 1º → ambiguous


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
        raise ValueError(f"Falha ao salvar ({ext})")
    buf.tofile(path)


# ---------------------------------------------------------------------------
# Warp (fallback quando a detecção de painéis falha)
# ---------------------------------------------------------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0]=pts[np.argmin(s)]; rect[1]=pts[np.argmin(d)]
    rect[2]=pts[np.argmax(s)]; rect[3]=pts[np.argmax(d)]
    return rect

def _find_sheet_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape; min_a = w*h*0.20
    for bk, lo, hi, dil in [(5,50,150,5),(5,30,120,7),(9,20,80,9)]:
        edges = cv2.Canny(cv2.GaussianBlur(gray,(bk,bk),0), lo, hi)
        edges = cv2.dilate(edges, np.ones((dil,dil),np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:15]:
            if cv2.contourArea(c) < min_a: break
            ap = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c,True), True)
            if len(ap) == 4:
                return ap.reshape(4,2).astype("float32")
    return None

def warp_sheet(image_bgr: np.ndarray, out_w=WARP_W, out_h=WARP_H) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = _find_sheet_quad(gray)
    dst  = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype="float32")
    if quad is not None:
        return cv2.warpPerspective(image_bgr,
                                   cv2.getPerspectiveTransform(_order_points(quad), dst),
                                   (out_w, out_h))
    return cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Detecção dos 6 painéis
# ---------------------------------------------------------------------------
def _iou(a,b):
    ax2,ay2=a[0]+a[2],a[1]+a[3]; bx2,by2=b[0]+b[2],b[1]+b[3]
    iw=max(0,min(ax2,bx2)-max(a[0],b[0])); ih=max(0,min(ay2,by2)-max(a[1],b[1]))
    inter=iw*ih; union=a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union>0 else 0.0

def _nms(boxes):
    boxes=sorted(boxes,key=lambda b:b[2]*b[3],reverse=True); keep=[]
    for b in boxes:
        if not any(_iou(b,k)>0.5 for k in keep): keep.append(b)
    return keep

def _detect_panel_boxes(gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
    h,w = gray.shape; min_h=h*0.18; min_a=h*w*0.025; max_a=h*w*0.12
    cands=[]
    for lo,hi,dil in [(40,140,5),(30,100,7),(60,180,5)]:
        edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),lo,hi)
        edges=cv2.dilate(edges,np.ones((dil,dil),np.uint8),iterations=1)
        cnts,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a=cv2.contourArea(c)
            if not (min_a<a<max_a): continue
            ap=cv2.approxPolyDP(c,0.025*cv2.arcLength(c,True),True)
            if len(ap)!=4: continue
            bx,by,bw,bh=cv2.boundingRect(c)
            if 0.28<bw/bh<0.60 and bh>=min_h: cands.append((bx,by,bw,bh))
    cands=_nms(cands); cands.sort(key=lambda b:b[0])
    return cands[:6] if len(cands)>=6 else _fill_missing(cands, gray.shape)

def _fill_missing(found,shape):
    if not found: return _scale_ref(shape)
    mw=int(np.median([b[2] for b in found])); mh=int(np.median([b[3] for b in found]))
    my=int(np.median([b[1] for b in found])); iw=shape[1]
    result=list(found)
    for xf in [0.040,0.222,0.388,0.556,0.722,0.888]:
        xe=int(iw*xf)
        if not any(abs(b[0]-xe)<mw*0.4 for b in result):
            result.append((xe,my,mw,mh))
    result.sort(key=lambda b:b[0]); return result[:6]

def _scale_ref(shape):
    h,w=shape; sx,sy=w/WARP_W,h/WARP_H
    return [(int(x*sx),int(y*sy),int(bw*sx),int(bh*sy)) for x,y,bw,bh in PANELS_REF]


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
def _preprocess(gray_panel: np.ndarray):
    eq  = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray_panel)
    th  = cv2.adaptiveThreshold(cv2.GaussianBlur(eq,(5,5),0),255,
              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,7)
    th  = cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    ph,pw=th.shape
    lines=cv2.bitwise_or(
        cv2.morphologyEx(th,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(20,pw//8),1))),
        cv2.morphologyEx(th,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,ph//8)))))
    ink  = cv2.bitwise_and(th,cv2.bitwise_not(lines))
    dist = cv2.distanceTransform(ink,cv2.DIST_L2,3)
    thick= (dist>=THICK_MIN).astype(np.uint8)*255
    return ink, thick


# ---------------------------------------------------------------------------
# Auto-detecção da grade de bolhas por projeção
# ---------------------------------------------------------------------------
def _detect_row_y(thick: np.ndarray, n: int = 15) -> List[int]:
    bh = thick.shape[0]
    proj = np.convolve(thick.sum(axis=1).astype(float), np.ones(5)/5, mode='same')
    pks, _ = find_peaks(proj, height=proj.max()*0.15, distance=max(10,bh//(n+3)))
    top = sorted(sorted(pks, key=lambda p: -proj[p])[:n])
    return [int(y) for y in top]


def _detect_col_x(thick: np.ndarray, n: int = 5) -> List[int]:
    """
    Detecta as 5 posições X das colunas de bolhas.
    Âncora: o pico mais à direita selecionado deve ser o pico global mais à direita
    (exclui automaticamente picos espúrios do número da questão à esquerda).
    """
    bw = thick.shape[1]
    proj = np.convolve(thick.sum(axis=0).astype(float), np.ones(3)/3, mode='same')
    min_dist = max(8, bw//(n+3))
    pks, _ = find_peaks(proj, height=proj.max()*0.10, distance=min_dist)
    pks = sorted(pks)

    if not pks:
        return [int(bw*(i+1)/(n+1)) for i in range(n)]

    rightmost = int(pks[-1])   # pico mais à direita = coluna E (ancora)

    if len(pks) == n:
        return [int(x) for x in pks]

    if len(pks) < n:
        # Interpola picos faltantes nos maiores gaps
        xs = list(pks)
        while len(xs) < n:
            sp = np.diff(sorted(xs))
            gi = int(np.argmax(sp))
            xs = sorted(xs + [int(sorted(xs)[gi] + sp[gi]//2)])
        return [int(x) for x in xs[:n]]

    # Mais picos que o necessário: escolhe combinação mais regular
    # que inclua obrigatoriamente o pico mais à direita
    best_xs, best_score = None, float('inf')
    for combo in combinations(pks, n):
        combo = sorted(combo)
        if combo[-1] < rightmost - 5:          # deve incluir pico mais à direita
            continue
        sp  = np.diff(combo)
        med = float(np.median(sp))
        cv  = float(np.std(sp)) / med if med > 0 else 999
        gap = sum(1 for s in sp if s > med*1.8)
        score = cv + gap*2.0
        if score < best_score:
            best_score, best_xs = score, combo

    return [int(x) for x in best_xs] if best_xs else [int(x) for x in pks[-n:]]


def _fallback_grid(bh, bw, p):
    """Grade de fallback baseada em frações calibradas no Dia 1."""
    XN = [[0.1129,0.2796,0.4462,0.6129,0.8710],[0.2646,0.4180,0.5556,0.7249,0.8571],
          [0.2634,0.4086,0.5591,0.7043,0.8817],[0.2460,0.4225,0.5668,0.6952,0.8342],
          [0.2620,0.4171,0.5561,0.7112,0.8556],[0.1141,0.2663,0.4185,0.7120,0.8641]]
    YN = [0.1099,0.1746,0.2392,0.3017,0.3685,0.4332,0.4978,
          0.5647,0.6272,0.6940,0.7565,0.8211,0.8858,0.9526,0.9800]
    return [int(f*bh) for f in YN], [int(f*bw) for f in XN[p]]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score(mask, cx, cy, r):
    h,w=mask.shape; x,y=int(round(cx)),int(round(cy))
    if not(0<=x<w and 0<=y<h): return 0.0
    m=np.zeros((h,w),np.uint8); cv2.circle(m,(x,y),r,255,-1)
    return float(cv2.countNonZero(cv2.bitwise_and(mask,mask,mask=m)))

def _best_score(mask, cx, cy, r, search):
    bs,bx,by=-1.0,cx,cy
    for dy in range(-search,search+1):
        for dx in range(-search,search+1):
            s=_score(mask,cx+dx,cy+dy,r)
            if s>bs: bs,bx,by=s,cx+dx,cy+dy
    return bs,bx,by


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
    q_start: int = 1,
) -> List[Answer]:
    """
    Lê as respostas de um cartão-resposta.

    Parâmetros
    ----------
    image_path : caminho para a foto do cartão
    config     : ignorado (mantido para compatibilidade)
    debug_dir  : se fornecido, salva imagens de debug nesta pasta
    q_start    : número da primeira questão (1 para Dia 1, 91 para Dia 2)
    """
    img  = imread_unicode(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    boxes = _detect_panel_boxes(gray)
    if len(boxes) < 6:
        try:
            warped = warp_sheet(img)
            gw     = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            bw     = _detect_panel_boxes(gw)
            if len(bw) > len(boxes):
                img, gray, boxes = warped, gw, bw
        except Exception:
            pass

    while len(boxes) < 6:
        boxes = _fill_missing(boxes, gray.shape)
    boxes = boxes[:6]

    letters = ["A","B","C","D","E"]
    out: List[Answer] = []
    overlay = img.copy()

    for p, (bx,by,bw,bh) in enumerate(boxes):
        bx=max(0,min(bx,gray.shape[1]-1)); by=max(0,min(by,gray.shape[0]-1))
        bw=min(bw,gray.shape[1]-bx);       bh=min(bh,gray.shape[0]-by)
        if bw<20 or bh<20:
            for r in range(15): out.append(Answer(q_start+p*15+r,"","blank",0.0))
            continue

        panel_gray = gray[by:by+bh, bx:bx+bw]
        ink, thick = _preprocess(panel_gray)

        ys = _detect_row_y(thick)
        xs = _detect_col_x(thick)

        if len(ys) < 15 or len(xs) < 5:
            ys_fb, xs_fb = _fallback_grid(bh, bw, p)
            if len(ys) < 15: ys = ys_fb
            if len(xs) < 5:  xs = xs_fb
        ys, xs = ys[:15], xs[:5]

        r_px = max(3, int(round(PANEL_R_REF[p]*bh/REF_PANEL_H)))
        rr   = max(3, int(round(r_px*FILL_FRAC)))

        all_raw: List[float] = []
        cache: List[List[Tuple]] = []

        for row in range(len(ys)):
            rc = []
            for col in range(len(xs)):
                cxf,cyf = float(xs[col]), float(ys[row])
                sT,bx2,by2 = _best_score(thick, cxf, cyf, rr, LOCAL_SEARCH)
                sI,bx3,by3 = _best_score(ink,   cxf, cyf, rr, LOCAL_SEARCH)
                if sT>=sI: s,rx,ry=sT,bx2,by2
                else:      s,rx,ry=sI,bx3,by3
                rc.append((s,rx,ry)); all_raw.append(s)
            cache.append(rc)

        blank_thr = max(float(np.percentile(all_raw, BLANK_PCTILE)), 8.0)

        for row in range(len(ys)):
            q = q_start + p*15 + row
            sc_raw = np.array([cache[row][c][0] for c in range(len(xs))], dtype=np.float32)
            sc     = sc_raw - sc_raw.min()          # baseline por questão
            order  = np.argsort(sc)[::-1]
            best_c = int(order[0])
            best_raw  = float(sc_raw[best_c])
            best_adj  = float(sc[order[0]])
            third_adj = float(sc[order[2]]) if len(order)>2 else 0.0

            if best_raw < blank_thr:
                out.append(Answer(q,"","blank",best_raw)); continue

            status = "ok"
            if best_adj>0 and third_adj/best_adj > AMBIG_RATIO:
                status = "ambiguous"

            out.append(Answer(q, letters[best_c], status, best_raw))

            if debug_dir is not None:
                ax=int(cache[row][best_c][1])+bx; ay=int(cache[row][best_c][2])+by
                col=(0,255,0) if status=="ok" else (0,165,255)
                cv2.circle(overlay,(ax,ay),rr+4,col,2)
                cv2.putText(overlay,str(q),(ax-rr-2,ay-rr-4),
                            cv2.FONT_HERSHEY_SIMPLEX,0.28,col,1)

    if debug_dir is not None:
        d=Path(debug_dir); d.mkdir(parents=True,exist_ok=True)
        imwrite_unicode(str(d/"01_source.png"),  img)
        imwrite_unicode(str(d/"02_overlay.png"), overlay)
        bv=img.copy()
        for i,(bx,by,bw,bh) in enumerate(boxes):
            cv2.rectangle(bv,(bx,by),(bx+bw,by+bh),(255,0,0),3)
            cv2.putText(bv,f"P{i+1}",(bx+5,by+28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        imwrite_unicode(str(d/"03_panels.png"), bv)

    return out


# ---------------------------------------------------------------------------
# Compatibilidade retroativa
# ---------------------------------------------------------------------------
@dataclass
class OMRConfig:
    use_clahe: bool=True; fill_frac: float=FILL_FRAC
    local_search_px: int=LOCAL_SEARCH; enable_fine_alignment: bool=False
