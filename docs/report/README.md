# MovieMind course report (LaTeX)

Source for the **PDF project report** required by the capstone rubric (Abstract through References).

## Prerequisites

| Tool | Purpose | Install (macOS) |
|------|---------|-----------------|
| **tectonic** | Build PDF (no sudo) | `brew install tectonic` |
| **pandoc** | Export Word (optional) | `brew install pandoc` (already on many systems) |
| **BasicTeX** (optional) | `pdflatex` / `latexmk` alternative | `brew install --cask basictex` (requires password) |

## Build

```bash
cd docs/report
make figures  # optional: regenerate PNGs from data/ml-1m/
make pdf      # -> moviemind_report.pdf (runs figures first)
make docx     # -> moviemind_report.docx (via pandoc; best-effort)
make clean
```

## Edit

| File | Content |
|------|---------|
| `moviemind_report.tex` | Title, preamble, `\input` sections |
| `sections/*.tex` | One file per rubric section |
| `references.bib` | Citations |
| `figures/` | Export plots from `notebooks/MovieMind_capstone.ipynb` here |

Update `\author{Your Name ...}` in `moviemind_report.tex` before submission.

## Content sources

- **Notebook:** `notebooks/MovieMind_capstone.ipynb`
- **Metrics:** `evidence/phase9_split_eval/evaluation_master_summary_70_10_20.md`
- **EDA plots:** `python3 docs/report/generate_figures.py` (writes `figures/*.png`)

## Git

**Not in git** (see root `.gitignore` and this folder’s `.gitignore`):

- `moviemind_report.pdf`, `moviemind_report.docx`
- LaTeX build junk (`*.aux`, `*.log`, …)

**In git:** `*.tex`, `Makefile`, `README.md`, `REFERENCES_VERIFY.md`, `figures/` (optional plot PNGs you add).

Rebuild before submission: `make pdf`.

## Word editing workflow

1. `make pdf` for the submission-quality PDF.
2. `make docx` for an editable Word copy (review formatting; complex tables may need tweaks).
3. Prefer editing `.tex` and rebuilding PDF when layout matters.
