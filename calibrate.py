import csv
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Set

from omr_reader import read_omr_answers, OMRConfig


def load_truth(csv_path: str) -> Tuple[str, Dict[int, str]]:
    truth: Dict[int, str] = {}
    img_name = None
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img_name = row["arquivo"]
            q = int(row["questao"])
            a = row["resposta"].strip().upper()
            truth[q] = a
    if img_name is None:
        raise ValueError("CSV vazio.")
    return img_name, truth


def accuracy(pred, truth: Dict[int, str], only_questions: Optional[Set[int]] = None) -> float:
    ok = 0
    total = 0
    for a in pred:
        if only_questions is not None and a.question not in only_questions:
            continue
        t = truth.get(a.question)
        if t is None:
            continue
        total += 1
        if a.answer == t:
            ok += 1
    return ok / total if total else 0.0


def build_cfg(**kwargs) -> OMRConfig:
    cfg_fields = {f.name for f in fields(OMRConfig)}
    clean = {k: v for k, v in kwargs.items() if k in cfg_fields and v is not None}
    return OMRConfig(**clean)


def make_subset_questions() -> Set[int]:
    # 30 questões bem distribuídas (5 por painel)
    picks = []
    for base in (0, 15, 30, 45, 60, 75):
        picks += [base + 1, base + 4, base + 8, base + 12, base + 15]
    return set(picks)


def make_debug_dir(img_path: str, suffix: str) -> Path:
    p = Path(img_path)
    base = p.with_suffix("")
    return base.parent / f"{base.name}{suffix}"


def cfg_with(cfg: OMRConfig, **patch) -> OMRConfig:
    d = dict(cfg.__dict__)
    d.update(patch)
    return build_cfg(**d)


def main(img_path: str, truth_csv: str):
    img_name, truth = load_truth(truth_csv)
    if Path(img_path).name != img_name:
        print(f"[AVISO] CSV diz arquivo='{img_name}', mas você passou '{Path(img_path).name}'.")

    cfg_fields = {f.name for f in fields(OMRConfig)}
    if not (("blank_min_score" in cfg_fields) and ("ambig_gap" in cfg_fields)):
        raise RuntimeError("Esse calibrate.py assume modo absoluto (blank_min_score + ambig_gap).")

    # Hard rule: keep Hough OFF during calibration (slow + unstable in noisy images)
    # Also: local_search_px OFF during calibration (speed)
    subset_q = make_subset_questions()

    START = time.time()
    MAX_SECONDS = 180  # 3 min

    # Smaller, high-yield grids
    block_sizes = [31, 41, 51]                 # adaptiveThreshold blockSize must be odd :contentReference[oaicite:4]{index=4}
    Cs = [5, 7, 9]
    fill_fracs = [0.30, 0.35, 0.40, 0.45]

    thick_dists = [1.2, 1.3, 1.4, 1.5, 1.7] if "thick_min_dist" in cfg_fields else [None]
    use_thick = True if "use_thick_mask" in cfg_fields else None

    # IMPORTANT: base thresholds for Phase A must be SMALL now
    base_blank = 2.0
    base_gap = 1.0
    base_ms = 0.0 if "multi_second_min_score" in cfg_fields else None
    base_norm = True if "normalize_per_question" in cfg_fields else None

    TOP_K = 6
    top: List[Tuple[float, OMRConfig]] = []

    def push_top(acc: float, cfg: OMRConfig):
        nonlocal top
        top.append((acc, cfg))
        top.sort(key=lambda x: x[0], reverse=True)
        top = top[:TOP_K]

    evals = 0

    # ---------- Phase A (subset ranking) ----------
    for bs in (block_sizes if "adaptive_block_size" in cfg_fields else [None]):
        for C in (Cs if "adaptive_C" in cfg_fields else [None]):
            for ff in (fill_fracs if "fill_frac" in cfg_fields else [None]):
                for td in thick_dists:
                    if time.time() - START > MAX_SECONDS:
                        print("[AVISO] Tempo limite da calibração (fase A) atingido.")
                        break

                    cfg = build_cfg(
                        adaptive_block_size=bs,
                        adaptive_C=C,
                        fill_frac=ff,
                        use_thick_mask=use_thick,
                        thick_min_dist=td,
                        blank_min_score=base_blank,
                        ambig_gap=base_gap,
                        multi_second_min_score=base_ms,
                        normalize_per_question=base_norm,
                        enable_fine_alignment=False,   # keep OFF
                        local_search_px=0,              # keep OFF
                    )

                    pred = read_omr_answers(img_path, cfg, debug_dir=None)
                    acc_sub = accuracy(pred, truth, only_questions=subset_q)

                    evals += 1
                    if evals % 25 == 0:
                        best = top[0][0] if top else 0.0
                        print(f"[fase A] evals={evals} best_sub={best:.4f}")

                    if len(top) < TOP_K or acc_sub >= top[-1][0]:
                        push_top(acc_sub, cfg)

    if not top:
        raise RuntimeError("Top vazio: nada foi avaliado.")

    print("\nTop configs (fase A) por SUBSET:")
    for i, (acc, cfg) in enumerate(top, start=1):
        print(f"  {i:02d}) acc_sub={acc:.4f} cfg={cfg}")

    # Debug of best Phase A, but with local_search ON (so you can *see* it)
    debug_A = make_debug_dir(img_path, "_debug_A")
    debug_A.mkdir(parents=True, exist_ok=True)
    cfgA_dbg = cfg_with(top[0][1], local_search_px=2)
    read_omr_answers(img_path, cfgA_dbg, debug_dir=str(debug_A))
    print("Debug (fase A) salvo em:", debug_A)

    # ---------- Phase A full validation ----------
    top_full: List[Tuple[float, OMRConfig]] = []
    for acc_sub, cfg in top:
        pred = read_omr_answers(img_path, cfg, debug_dir=None)
        acc_full = accuracy(pred, truth)
        top_full.append((acc_full, cfg))
    top_full.sort(key=lambda x: x[0], reverse=True)

    print("\nTop configs (fase A) por FULL(90):")
    for i, (acc, cfg) in enumerate(top_full, start=1):
        print(f"  {i:02d}) acc_full={acc:.4f} cfg={cfg}")

    # ---------- Phase B thresholds (fast grid + progress) ----------
    blank_mins = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
    ambig_gaps = [0.5, 1.0, 2.0, 3.0, 4.0]
    ms_grid = [0.0, 3.0, 5.0] if "multi_second_min_score" in cfg_fields else [None]

    best_acc = -1.0
    best_cfg = None
    tries = 0

    for base_acc, base_cfg in top_full[:TOP_K]:
        for b in blank_mins:
            for g in ambig_gaps:
                for ms in ms_grid:
                    tries += 1
                    if tries % 50 == 0:
                        print(f"[fase B] tries={tries} best={best_acc:.4f}")

                    cfg = cfg_with(
                        base_cfg,
                        blank_min_score=float(b),
                        ambig_gap=float(g),
                        multi_second_min_score=(ms if ms is not None else base_cfg.__dict__.get("multi_second_min_score")),
                        local_search_px=0,              # keep OFF while scoring fast
                    )
                    pred = read_omr_answers(img_path, cfg, debug_dir=None)
                    acc = accuracy(pred, truth)
                    if acc > best_acc:
                        best_acc, best_cfg = acc, cfg

    print("\nMelhor acc:", best_acc)
    print("cfg:", best_cfg)

    # Final debug with local_search ON
    debug_final = make_debug_dir(img_path, "_debug")
    debug_final.mkdir(parents=True, exist_ok=True)
    cfg_dbg = cfg_with(best_cfg, local_search_px=2)
    read_omr_answers(img_path, cfg_dbg, debug_dir=str(debug_final))
    print("Debug final salvo em:", debug_final)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("uso: python calibrate.py <imagem> <truth.csv>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])