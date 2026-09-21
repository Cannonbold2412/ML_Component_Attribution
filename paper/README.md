# Paper sources (Version 5)

* `MBCA_paper.pdf` -- the manuscript.
* `MBCA_paper.tex` + `sections/*.tex` -- hand-written prose. **Every result number is a macro** from
  `tex/numbers.tex`. Every result table (`tex/s1_*.tex`, `tex/s2_*.tex`) and figure (`figures/s1_*`, `figures/s2_*`) is
  generated. Only `figures/p36_fig_mbca_pipeline.png` (a schematic) is a static asset.
* Regenerate everything: `python -m mbca study paper` (from the repository root).
* Build: `python paper/build_pdf.py` (pandoc -> docx -> Word export; needs pypandoc-binary, python-docx, pywin32, and
  MS Word), or compile `MBCA_paper.tex` with any LaTeX distribution or Overleaf.
* `tests/test_paper.py` fails if the manuscript uses an undefined macro, a missing table or figure, or a hand-typed
  decimal in a results section.

See `../ERRATA.md` for what changed relative to earlier drafts.
