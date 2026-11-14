"""Données et diagramme de flux PRISMA 2020, construits à partir des fichiers produits à chaque étape."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape


def _load(path: Path, step: str):
    if not path.exists():
        raise SystemExit(f"{path} introuvable : lancez d'abord l'étape « {step} ».")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_flow(run_dir: Path) -> dict:
    """Rassemble les effectifs de chaque phase (identification, screening, éligibilité, inclusion)."""
    ident_file = _load(run_dir / "prisma_identification.json", "récupération des articles")
    ident = ident_file["identification"]
    screen = _load(run_dir / "screening" / "prisma_screening.json", "screening (--screen-only)")
    elig = _load(run_dir / "fulltext" / "prisma_eligibility.json", "éligibilité (--eligibility-only)")
    included = _load(run_dir / "fulltext" / "included_for_extraction.json", "éligibilité (--eligibility-only)")

    n_include = sum(1 for p in included if p.get("eligibility") == "include")
    n_full = sum(1 for p in included if p.get("evidence_level") == "full_text")
    flow = {
        "identification": {
            "databases": {s: v["retrieved"] for s, v in ident["per_source"].items()},
            "databases_available": {s: v["hits_total"] for s, v in ident["per_source"].items()},
            "total_retrieved": ident["total_retrieved"],
            "duplicates_removed": ident["duplicates_removed"],
            "unique_records": ident["records_unique"],
        },
        "screening": {
            "screened": screen["records_screened"],
            "excluded": screen["excluded"],
            "passed": screen["passed_to_fulltext"],
        },
        "eligibility": {
            "sought": elig["reports_sought"],
            "not_retrieved": elig["reports_not_retrieved"],
            "assessed": elig["reports_assessed"],
            "excluded": elig["excluded"],
            "reasons": elig["exclusion_reasons"],
        },
        "included": {
            "studies": len(included),
            "include": n_include,
            "provisional": len(included) - n_include,  # « maybe » : décision humaine encore attendue
            "full_text": n_full,
            "abstract_only": len(included) - n_full,
        },
        "warnings": [],
    }

    w = flow["warnings"]
    if flow["identification"]["unique_records"] != flow["screening"]["screened"]:
        w.append(
            f"{flow['identification']['unique_records']} articles uniques mais {flow['screening']['screened']} screenés "
            "(screening partiel ? relancez --screen-only sans --screen-limit)."
        )
    if flow["screening"]["passed"] != flow["eligibility"]["sought"]:
        w.append("Le nombre d'articles passés au texte intégral diffère des rapports recherchés (relancez --fulltext-only puis --eligibility-only).")
    if flow["eligibility"]["assessed"] - flow["eligibility"]["excluded"] != flow["included"]["studies"]:
        w.append("Études incluses incohérentes avec évalués - exclus (relancez --eligibility-only).")
    return flow


# --------------------------------------------------------------------------- SVG
SOURCE_NAMES = {"pubmed": "PubMed", "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv", "openalex": "OpenAlex"}
LINE_H, PAD = 18, 12
X_LEFT, X_RIGHT, BOX_W, GAP = 30, 500, 420, 38


def _wrap(text: str, width: int = 52) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _box(x: int, y: int, lines: list[str]) -> tuple[str, int]:
    h = PAD * 2 + LINE_H * len(lines)
    parts = [f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{h}" rx="6" fill="#ffffff" stroke="#444444" stroke-width="1.5"/>']
    for i, line in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        parts.append(
            f'<text x="{x + PAD}" y="{y + PAD + LINE_H * (i + 1) - 5}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="13" font-weight="{weight}" fill="#111111">{escape(line)}</text>'
        )
    return "".join(parts), h


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#444444" stroke-width="1.5" marker-end="url(#arrow)"/>'


def flow_rows(flow: dict) -> list[tuple[list[str], list[str] | None]]:
    """Contenu des boîtes (gauche, droite) de chaque ligne du diagramme PRISMA 2020 ; 1re ligne de chaque boîte = titre."""
    i, s, e, inc = flow["identification"], flow["screening"], flow["eligibility"], flow["included"]

    left1 = ["Identification of studies via databases", f"Records identified (n = {i['total_retrieved']})"]
    left1 += [f"  {SOURCE_NAMES.get(src, src)}: n = {n}" for src, n in i["databases"].items()]
    reasons: list[str] = []
    for reason, n in sorted(e["reasons"].items(), key=lambda kv: -kv[1]):
        wrapped = _wrap(f"{reason} (n = {n})", 50)
        reasons += [f"- {wrapped[0]}"] + [f"  {w}" for w in wrapped[1:]]

    return [
        (left1, ["Records removed before screening", f"Duplicate records (n = {i['duplicates_removed']})"]),
        ([f"Records screened (n = {s['screened']})"], [f"Records excluded (n = {s['excluded']})"]),
        ([f"Reports sought for retrieval (n = {e['sought']})"], [f"Reports not retrieved (n = {e['not_retrieved']})"]),
        ([f"Reports assessed for eligibility (n = {e['assessed']})"], [f"Reports excluded (n = {e['excluded']})"] + reasons),
        (
            [
                f"Studies included in review (n = {inc['studies']})",
                f"Full text: {inc['full_text']} ; abstract only: {inc['abstract_only']}",
                f"Included: {inc['include']} ; pending human decision: {inc['provisional']}",
            ],
            None,
        ),
    ]


def render_svg(flow: dict) -> str:
    """Diagramme de flux PRISMA 2020 en SVG autonome (aucune dépendance)."""
    rows = flow_rows(flow)
    body: list[str] = []
    y = 20
    prev_bottom: int | None = None
    for left, right in rows:
        lsvg, lh = _box(X_LEFT, y, left)
        body.append(lsvg)
        rh = 0
        if right is not None:
            rsvg, rh = _box(X_RIGHT, y, right)
            body.append(rsvg)
            body.append(_arrow(X_LEFT + BOX_W, y + 22, X_RIGHT, y + 22))
        if prev_bottom is not None:
            body.append(_arrow(X_LEFT + BOX_W // 2, prev_bottom, X_LEFT + BOX_W // 2, y))
        prev_bottom = y + lh
        y += max(lh, rh) + GAP

    width, height = X_RIGHT + BOX_W + 30, y
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">'
        '<path d="M0,0 L10,4 L0,8 z" fill="#444444"/></marker></defs>'
        '<rect width="100%" height="100%" fill="#ffffff"/>' + "".join(body) + "</svg>"
    )


# --------------------------------------------------------------------------- PNG (pour Word et le rapport)
def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_png(flow: dict, path: Path, scale: int = 2) -> Path:
    """Diagramme de flux PRISMA 2020 en PNG (Pillow), même contenu et même mise en page que le SVG.

    `scale` = résolution (2 -> environ 1900 px de large, net à l'impression et dans Word).
    """
    from PIL import Image, ImageDraw

    rows = flow_rows(flow)
    s = scale
    line_h, pad, gap = 20 * s, 14 * s, 40 * s
    box_w, x_left, x_right, top = 440 * s, 30 * s, 530 * s, 20 * s
    regular, bold = _font(14 * s), _font(14 * s, bold=True)

    def height(lines: list[str]) -> int:
        return pad * 2 + line_h * len(lines)

    total = top + sum(max(height(l), height(r) if r else 0) + gap for l, r in rows)
    img = Image.new("RGB", (x_right + box_w + 30 * s, total), "white")
    d = ImageDraw.Draw(img)

    def box(x: int, y: int, lines: list[str]) -> int:
        h = height(lines)
        d.rounded_rectangle([x, y, x + box_w, y + h], radius=6 * s, outline="#444444", width=2 * s, fill="white")
        for k, line in enumerate(lines):
            d.text((x + pad, y + pad + line_h * k), line, fill="#111111", font=bold if k == 0 else regular)
        return h

    def arrow(x1: int, y1: int, x2: int, y2: int) -> None:
        d.line([x1, y1, x2, y2], fill="#444444", width=2 * s)
        a = 9 * s
        if x1 == x2:  # flèche verticale vers le bas
            d.polygon([(x2, y2), (x2 - a // 2, y2 - a), (x2 + a // 2, y2 - a)], fill="#444444")
        else:  # flèche horizontale vers la droite
            d.polygon([(x2, y2), (x2 - a, y2 - a // 2), (x2 - a, y2 + a // 2)], fill="#444444")

    y, prev_bottom = top, None
    for left, right in rows:
        lh = box(x_left, y, left)
        rh = 0
        if right is not None:
            rh = box(x_right, y, right)
            arrow(x_left + box_w, y + 24 * s, x_right, y + 24 * s)
        if prev_bottom is not None:
            arrow(x_left + box_w // 2, prev_bottom, x_left + box_w // 2, y)
        prev_bottom = y + lh
        y += max(lh, rh) + gap
    img.save(path, "PNG", dpi=(150 * s, 150 * s))
    return path
