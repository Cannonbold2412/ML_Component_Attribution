"""Compile docs/papers/MBCA_paper.tex into an IEEE-style
DOCX and PDF. Rerunnable: `python scripts/paper_p36_build_pdf.py`.

Route: pandoc (.tex -> .docx, citeproc + IEEE CSL, embeds tables/figures/math)
against a restyled reference.docx, then a python-docx post-pass splits the
body into a 1-column title block + 2-column body (IEEE two-column look),
then Word COM exports the final PDF.

Requires: pypandoc-binary, python-docx, pywin32 (win32com), and MS Word
installed for the PDF export step.
"""
import re
import zipfile
import shutil
from pathlib import Path

import pypandoc
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PAPERS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PAPERS_DIR / "assets"
TEX_SRC = PAPERS_DIR / "MBCA_paper.tex"
BIB_SRC = PAPERS_DIR / "references.bib"
CSL = ASSETS_DIR / "ieee.csl"
REF_DOCX = ASSETS_DIR / "ieee_ref.docx"
OUT_DOCX = PAPERS_DIR / "MBCA_paper.docx"
OUT_PDF = PAPERS_DIR / "MBCA_paper.pdf"


def restyle_reference_doc():
    """Edit the handful of styles pandoc actually emits so the output reads
    IEEE-ish: Times New Roman body, centered Title/Author, bold Heading1,
    italic Heading2, A4 page with narrow margins."""
    doc = Document(str(REF_DOCX))

    def set_font(style_name, name="Times New Roman", size=10, bold=None, italic=None):
        style = doc.styles[style_name]
        f = style.font
        f.name = name
        f.size = Pt(size)
        if bold is not None:
            f.bold = bold
        if italic is not None:
            f.italic = italic
        # ensure east-asian font fallback doesn't override
        rpr = style.element.get_or_add_rPr()
        rFonts = rpr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rpr.append(rFonts)
        rFonts.set(qn("w:ascii"), name)
        rFonts.set(qn("w:hAnsi"), name)
        # Title/Heading1/Heading2 in the reference doc are theme-linked (asciiTheme=
        # "majorHAnsi" etc alongside the explicit ascii font); Word's export path
        # resolves the theme font over the explicit override for these built-in
        # styles unless the theme attributes are removed outright.
        for theme_attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rFonts.get(qn(theme_attr)) is not None:
                del rFonts.attrib[qn(theme_attr)]

    set_font("Normal", size=10)
    set_font("BodyText", size=10)
    set_font("FirstParagraph", size=10)

    set_font("Title", size=22, bold=True)
    doc.styles["Title"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_font("Author", size=11)
    doc.styles["Author"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    set_font("AbstractTitle", size=10, bold=True, italic=True)
    set_font("Abstract", size=9, italic=True)

    set_font("Heading1", size=11, bold=True)
    set_font("Heading2", size=10, bold=True, italic=True)
    set_font("Heading3", size=10, italic=True)

    set_font("Table", size=8)
    set_font("Caption", size=8, italic=True)
    set_font("TableCaption", size=8, italic=True)
    set_font("ImageCaption", size=8, italic=True)
    set_font("Bibliography", size=9)

    # Page setup: A4, narrow-ish margins, single column at this stage
    # (2-column split happens in the post-pass on the OUTPUT doc, not here).
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(1.9)
    section.left_margin = Cm(1.9)
    section.right_margin = Cm(1.9)

    doc.save(str(REF_DOCX))


def convert_tex_to_docx():
    extra_args = [
        f"--reference-doc={REF_DOCX}",
        "--citeproc",
        f"--bibliography={BIB_SRC}",
        f"--csl={CSL}",
        f"--resource-path={PAPERS_DIR}:{PAPERS_DIR / 'tex'}:{PAPERS_DIR / 'figures'}",
        "--mathml",
    ]
    pypandoc.convert_file(
        str(TEX_SRC),
        to="docx",
        format="latex",
        outputfile=str(OUT_DOCX),
        extra_args=extra_args,
        cworkdir=str(PAPERS_DIR),
    )


def _make_cols_element(num_cols):
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), "432" if num_cols == 2 else "0")  # 18pt/0 in twips (720 twips/in)
    return cols


def split_into_two_column_body():
    """Insert a continuous section break right after the Keywords/Abstract
    block so the title/author/abstract stay single-column and everything
    from the first Heading1 (Introduction) onward runs two-column, matching
    the IEEE template's own `w:cols w:num="2"` body convention."""
    doc = Document(str(OUT_DOCX))
    body = doc.element.body

    # Find the first Heading1 paragraph -> that's where two-column starts.
    paragraphs = doc.paragraphs
    split_index = None
    for i, p in enumerate(paragraphs):
        if p.style is not None and p.style.name == "Heading 1":
            split_index = i
            break
    if split_index is None:
        print("WARNING: no Heading 1 found; skipping two-column split.")
        return

    split_para = paragraphs[split_index]

    # A section break is expressed as a sectPr inside the pPr of the LAST
    # paragraph of the section it closes. Insert a new sectPr into the
    # paragraph immediately BEFORE split_para, cloning the body's final
    # sectPr (page size/margins) but forcing num=1 (title block stays
    # single-column). The body's trailing sectPr (num=2) governs everything
    # after that, i.e. the two-column region.
    final_sectPr = body.find(qn("w:sectPr"))
    if final_sectPr is None:
        print("WARNING: no body sectPr found; skipping two-column split.")
        return

    prev_para = paragraphs[split_index - 1]
    new_sectPr = OxmlElement("w:sectPr")
    for child in final_sectPr:
        if child.tag == qn("w:cols"):
            continue
        new_sectPr.append(_clone(child))
    new_sectPr.append(_make_cols_element(1))

    pPr = prev_para._p.get_or_add_pPr()
    pPr.append(new_sectPr)

    # Force the body's final (already-existing) sectPr to two columns.
    existing_cols = final_sectPr.find(qn("w:cols"))
    if existing_cols is not None:
        final_sectPr.remove(existing_cols)
    final_sectPr.append(_make_cols_element(2))

    doc.save(str(OUT_DOCX))


def _clone(element):
    from copy import deepcopy
    return deepcopy(element)


def _final_body_sectpr(doc):
    return doc.element.body.find(qn("w:sectPr"))


def _build_sectpr(template_sectpr, num_cols):
    """Clone the body sectPr (page size/margins), force a *continuous* break
    with the given column count. Element order per OOXML: type before pgSz,
    cols after pgMar."""
    sect = _clone(template_sectpr)
    for tag in ("w:type", "w:cols"):
        el = sect.find(qn(tag))
        if el is not None:
            sect.remove(el)

    wtype = OxmlElement("w:type")
    wtype.set(qn("w:val"), "continuous")
    pgSz = sect.find(qn("w:pgSz"))
    if pgSz is not None:
        pgSz.addprevious(wtype)
    else:
        sect.insert(0, wtype)

    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), "432" if num_cols == 2 else "0")
    pgMar = sect.find(qn("w:pgMar"))
    if pgMar is not None:
        pgMar.addnext(cols)
    else:
        sect.append(cols)
    return sect


def _boundary_paragraph(sectpr):
    """An (empty, thin) paragraph whose only job is to carry a section break."""
    p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "0")
    spacing.set(qn("w:after"), "0")
    spacing.set(qn("w:line"), "20")
    spacing.set(qn("w:lineRule"), "exact")
    pPr.append(spacing)
    pPr.append(sectpr)
    p.append(pPr)
    return p


def isolate_wide_blocks():
    """Wrap wide result tables and figures in a single-column continuous
    section so each spans the full page width, with the 2-column body flow
    resuming afterwards. Word renders a continuous section break as a
    column-span, not a page break.

    Two things this has to get right or Word produces large dead-space gaps:
    1. Blocks that already live in the single-column front matter (before the
       first Heading 1 -- title/abstract/graphical-abstract) must NOT be
       wrapped again; forcing a 2-column boundary in front of them there
       hijacks the front matter itself into 2 columns (nothing follows the
       first Heading 1's own 1-column->2-column boundary until then).
    2. Consecutive wide blocks (e.g. a figure immediately followed by a wide
       table) must share ONE single-column region, not each get their own
       1-col/2-col pair -- an empty 2-column region between two back-to-back
       wide blocks is what produces the near-blank-page gaps."""
    doc = Document(str(OUT_DOCX))
    body = doc.element.body
    tmpl = _final_body_sectpr(doc)

    kids = list(body)

    def is_p(el):
        return el.tag == qn("w:p")

    def is_tbl(el):
        return el.tag == qn("w:tbl")

    def style_of(el):
        if not is_p(el):
            return None
        pPr = el.find(qn("w:pPr"))
        if pPr is None:
            return None
        ps = pPr.find(qn("w:pStyle"))
        return ps.get(qn("w:val")) if ps is not None else None

    def has_drawing(el):
        return is_p(el) and el.find(".//" + qn("w:drawing")) is not None

    def is_heading1(el):
        return is_p(el) and style_of(el) == "Heading1"

    first_heading_idx = next((i for i, el in enumerate(kids) if is_heading1(el)), 0)

    # Collect blocks: (start_idx, end_idx, start_el, end_el). Caption travels
    # with its table/figure. Blocks before the first Heading 1 are already
    # single-column (front matter) and are skipped entirely.
    blocks = []
    for i, el in enumerate(kids):
        if i < first_heading_idx:
            continue
        if is_tbl(el):
            start_idx, start = i, el
            prev = kids[i - 1] if i > 0 else None
            if prev is not None and style_of(prev) in ("TableCaption", "Caption"):
                start_idx, start = i - 1, prev
            blocks.append((start_idx, i, start, el))
        elif has_drawing(el):
            start_idx, end_idx, end = i, i, el
            nxt = kids[i + 1] if i + 1 < len(kids) else None
            if nxt is not None and style_of(nxt) in ("ImageCaption", "Caption"):
                end_idx, end = i + 1, nxt
            blocks.append((i, end_idx, el, end))

    # Merge blocks that are adjacent (nothing but whitespace/empty paragraphs
    # between one block's end and the next block's start) so they share one
    # continuous single-column region instead of a spurious empty 2-column
    # gap between them.
    def only_blank_between(end_idx, start_idx):
        return all(is_p(kids[j]) and not kids[j].findall(".//" + qn("w:t"))
                   for j in range(end_idx + 1, start_idx))

    merged = []
    for start_idx, end_idx, start_el, end_el in blocks:
        if merged and only_blank_between(merged[-1][1], start_idx):
            prev_start_idx, _, prev_start, _ = merged[-1]
            merged[-1] = (prev_start_idx, end_idx, prev_start, end_el)
        else:
            merged.append((start_idx, end_idx, start_el, end_el))

    # Process bottom-to-top so element refs stay valid.
    for _, _, start_el, end_el in reversed(merged):
        # Close the preceding 2-column region.
        start_el.addprevious(_boundary_paragraph(_build_sectpr(tmpl, 2)))
        # Close the full-width block region (1 column).
        end_el.addnext(_boundary_paragraph(_build_sectpr(tmpl, 1)))

    doc.save(str(OUT_DOCX))


def export_pdf_via_word():
    import win32com.client

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        wdoc = word.Documents.Open(str(OUT_DOCX))
        wdoc.ExportAsFixedFormat(str(OUT_PDF), ExportFormat=17)  # 17 = wdExportFormatPDF
        wdoc.Close(False)
    finally:
        word.Quit()


def main():
    assert TEX_SRC.exists(), f"missing {TEX_SRC}"
    assert BIB_SRC.exists(), f"missing {BIB_SRC}"
    assert CSL.exists(), f"missing {CSL} -- fetch ieee.csl first"

    print("1/4 restyling reference doc...")
    restyle_reference_doc()

    print("2/4 pandoc: .tex -> .docx...")
    convert_tex_to_docx()

    print("3/5 splitting into 1-col title / 2-col body...")
    split_into_two_column_body()

    print("4/5 making tables + figures full-width...")
    isolate_wide_blocks()

    print("5/5 exporting PDF via Word...")
    export_pdf_via_word()

    print(f"done: {OUT_DOCX}")
    print(f"done: {OUT_PDF}")


if __name__ == "__main__":
    main()
