# omr-reader

**Reads phone photos of 90-question answer sheets and exports the answers to Excel.** Built to replace manual grading of practice exams: staff photograph each sheet, select the photos, and get an `.xlsx` with every answer flagged as `ok`, `blank` or `ambiguous` for a human to double-check.

> 🇧🇷 Leitor de gabarito: lê fotos de cartões-resposta (6 painéis × 15 questões × A–E) e exporta as respostas para `.xlsx`. Interface em português.

## How it works

Phone photos are skewed, shadowed and in perspective, so the reader never trusts a single global alignment:

1. **Find the six answer panels directly** on the raw photo (Canny edges in three passes with different parameters → rectangular contours filtered by size and aspect ratio → non-max suppression). Missing panels are inferred from the ones that were found.
2. **Fallback warp:** only if fewer than six panels are found, look for the sheet's four corners and correct perspective, then try again.
3. **Per-panel grid detection:** inside each panel, remove printed lines (morphological opening), keep only thick ink (distance transform), and find the 15 rows and 5 columns from projection profiles. Rows hidden by consecutive blank answers are interpolated; the column picker anchors on the right-most column so question numbers can't be mistaken for bubbles.
4. **Scoring:** each bubble is scored by filled pixels inside a circle, searching ±6 px around its expected centre to absorb residual distortion.
5. **Decision:** the blank threshold adapts per panel (65th percentile of that panel's scores); each question's scores are baselined against its own weakest bubble, which cancels the printed letters inside the circles; the 3rd-best/best ratio flags double marks as `ambiguous`.

## Evolution

The repository history keeps each version as a commit, dated when it was written. Each one was driven by failures on real photos:

| Version | Problem it fixed | Change |
|---|---|---|
| v1 | — | Fixed template, global warp |
| v2 | Positions drifted between photos | Template builder and calibration tool |
| v3 | The warp failed on 9 of 10 photos (it grabbed the table, not the sheet) | Panels detected as independent anchors; adaptive threshold; per-question baseline |
| v3.1 | Photos where the sheet's corners weren't visible | Panels detected on the raw photo, warp only as fallback; day-2 support (Q91–180); batch processing; UI work moved to a thread |
| v4 | Fixed bubble positions broke on slightly different prints | Bubble grid found per panel from projection profiles, with a calibrated fallback |
| v4.1 | Empty rows confused row detection; heavy SciPy dependency | Missing rows interpolated; own peak finder (SciPy dropped); PyInstaller spec |

## Use

```bash
pip install -r requirements.txt
python app.py
```

Select one or more photos (and *Dia 2* for questions 91–180). The `.xlsx` is saved next to each image.

```python
from omr_reader import read_omr_answers

for a in read_omr_answers("sheet.jpg", q_start=1, debug_dir="debug/"):
    print(a.question, a.answer, a.status)
```

`debug_dir` saves the detected panels and an overlay of every reading — useful when tuning for a new form.

| File | Role |
|---|---|
| `omr_reader.py` | Engine: panel detection, warp fallback, grid detection, scoring |
| `app.py` | PySide6 interface (multi-photo, threaded) |
| `exporter.py` | `.xlsx` export |
| `build_template.py`, `calibrate.py` | Maintenance tools used in earlier versions to measure a reference sheet |
| `LeitorGabarito.spec` | PyInstaller build |

**Photo tips:** keep all four edges of the sheet in frame, avoid hard shadows over the answer panels, and use even light.

### About sample images

There are none in this repository on purpose: every photo used during development was a real student's answer sheet, with their name and signature. The reader is tuned to one specific form layout (six bordered panels of 15 questions); adapting it to another form means re-measuring that layout.

## Stack

Python · OpenCV · NumPy · PySide6 · openpyxl · PyInstaller

---

Built by [João Barbosa](https://joaobarbosa.pages.dev) at his employer and published here **with the employer's permission**, without any student data.

**© João Barbosa. All rights reserved.** No open-source license is granted — you're welcome to read the code, but please don't reuse it without permission.
