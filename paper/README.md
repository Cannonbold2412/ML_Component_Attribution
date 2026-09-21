# Paper sources

* `MBCA_paper.pdf` — the manuscript (Version 4).
* `MBCA_paper.tex` + `tex/` + `figures/` + `references.bib` — LaTeX source. Compiles with any LaTeX
  distribution or Overleaf (`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`).
* `build_pdf.py` — the route actually used to build the shipped PDF: pandoc (tex → docx, citeproc +
  `assets/ieee.csl`), a python-docx two-column pass, then an MS Word export. Needs `pypandoc-binary`,
  `python-docx`, `pywin32`, and Word.

Regenerating inputs:

| Artifact | Command |
|---|---|
| Figures 1–2 (forest plots, unit-corrected) | `python examples/04_paper_results.py` |
| Replication table (`tex/replication_table.tex`) | `python examples/03_reproduce_paper.py` then `python paper/make_replication_table.py` |
| Tables 1–2, Opportunity Matrix, pipeline/flowchart figures | produced by the original study pipeline from `results/paper/*.csv` (unchanged from Draft v3) |

See `../ERRATA.md` for what changed relative to Draft v3.
