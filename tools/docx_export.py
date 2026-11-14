"""Conversion Markdown -> DOCX minimale (titres, paragraphes, listes, citations, tableaux, gras/italique, code)."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|_[^_]+_|`[^`]+`)")


def _add_inline(par, text: str) -> None:
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif (part.startswith("*") and part.endswith("*")) or (part.startswith("_") and part.endswith("_") and len(part) > 2):
            par.add_run(part[1:-1]).italic = True
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Consolas"
        else:
            par.add_run(part)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def markdown_to_docx(md: str, path: Path, base_dir: Path | None = None) -> Path:
    """`base_dir` : dossier de référence des images « ![légende](fichier.png) » (par défaut celui du .docx)."""
    base_dir = base_dir or Path(path).parent
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):  # bloc de code
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            par = doc.add_paragraph()
            run = par.add_run("\n".join(code))
            run.font.name = "Consolas"
            run.font.size = Pt(8)
            i += 1
            continue
        img = re.match(r"^!\[(.*)\]\((.+)\)$", s)
        if img:  # figure : image pleine largeur + légende en italique
            src = Path(base_dir) / img.group(2)
            if src.exists():
                doc.add_picture(str(src), width=Inches(6.3))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph()
            _add_inline(cap, img.group(1) if src.exists() else f"{img.group(1)} (image not found: {img.group(2)})")
            for run in cap.runs:
                run.italic = True
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            doc.add_heading(m.group(2), level=len(m.group(1)) - 1)  # « # » = titre du document, « ## » = niveau 1...
            i += 1
            continue
        if s.startswith("|"):  # tableau
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\|\s*:?-{3,}", lines[i].strip()):
                    rows.append(_cells(lines[i]))
                i += 1
            ncol = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=ncol)
            table.style = "Table Grid"
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    cell = table.cell(r, c)
                    cell.text = ""
                    par = cell.paragraphs[0]
                    _add_inline(par, val)
                    for run in par.runs:
                        run.font.size = Pt(8)
                        if r == 0:
                            run.bold = True
            doc.add_paragraph()
            continue
        if s.startswith("- "):
            _add_inline(doc.add_paragraph(style="List Bullet"), s[2:])
        elif s.startswith("> "):
            par = doc.add_paragraph()
            _add_inline(par, s[2:])
            for run in par.runs:
                run.italic = True
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        else:
            _add_inline(doc.add_paragraph(), s)
        i += 1
    doc.save(path)
    return path
