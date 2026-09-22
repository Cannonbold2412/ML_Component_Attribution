# Paper sources (Version 6, IEEE two-column)

* `MBCA_paper.pdf`: the manuscript, compiled with the real `IEEEtran` class (journal mode, Times New Roman).
  `MBCA_paper.docx` is the same manuscript as an IEEE-styled Word file (generated, not committed).
* `MBCA_paper.tex` + `sections/*.tex`: hand-written prose. **Every result number is a macro** from
  `tex/numbers.tex`. Every table (`tex/m_*.tex`, `tex/s2_*.tex`, except Table I, which is design text) and every figure
  (`figures/m_*`, `figures/s1_*`, `figures/s2_*`, drawn by `mbca/paper_figures.py`) is generated.
* Figures are drawn at their exact print width (3.487 in column / 7.14 in page) at 600 dpi in Times New Roman, so no
  figure is rescaled and all figure text is 7-8 pt. Colours follow one validated, colour-blind-safe palette:
  blue = significant gain, red = significant loss, open marker = not significant.
* The full per-configuration numbers behind the figures are supplementary data in `../results/` (the long tables of
  earlier drafts were removed from the manuscript).
* Regenerate every table, figure and number: `python -m mbca study paper` (from the repository root).
* Build: `python paper/build_pdf.py` (`--pdf` / `--docx` for one of them).
  * PDF: [Tectonic](https://tectonic-typesetting.github.io) (found on PATH, via `$TECTONIC`, or in
    `%LOCALAPPDATA%\Programs\tectonic`). The same `.tex` also compiles unchanged on Overleaf (pdfLaTeX falls back to
    `newtxtext`/`newtxmath` Times).
  * DOCX: pandoc (`pypandoc-binary`) + `python-docx`; no Microsoft Word needed.
* `tests/test_paper.py` fails if the manuscript uses an undefined macro, a missing table or figure, a figure whose
  pixel width does not match its printed width, or a hand-typed decimal in a results section.

See `../ERRATA.md` for what changed relative to earlier drafts.
