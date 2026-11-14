"""Vérification qu'une citation existe réellement dans un texte (garde-fou anti-hallucination)."""
from __future__ import annotations

import re
import unicodedata

_ELLIPSIS = re.compile(r"\[\s*\.\.\.\s*\]|\.\.\.|…")
# Point médian utilisé comme séparateur décimal (style The Lancet : « 84·2% ») -> point.
_MIDDOT_DECIMAL = re.compile(r"(?<=\d)[·∙⋅](?=\d)")


def _decimal_points(s: str) -> str:
    return _MIDDOT_DECIMAL.sub(".", s)


def _norm(s: str) -> str:
    s = _decimal_points(unicodedata.normalize("NFKC", s))
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("–", "-"), ("—", "-"), ("−", "-")):
        s = s.replace(a, b)
    s = re.sub(r"-\s*\n\s*", "", s)  # mots coupés en fin de ligne
    return re.sub(r"\s+", " ", s).lower().strip()


def _trim(s: str) -> str:
    return s.strip(" .,;:\"'()[]")


def verify_quote(quote: str, source: str, min_words: int = 2) -> bool:
    """Vrai si la citation (ou chacun de ses fragments séparés par « ... ») figure mot pour mot dans la source.

    La comparaison ignore la casse, les espaces multiples, les apostrophes/guillemets typographiques et la
    ponctuation aux extrémités. Une citation trop courte (moins de `min_words` mots) n'est jamais acceptée.
    """
    if not quote or not source:
        return False
    q, src = _norm(quote), _norm(source)
    parts = [_trim(p) for p in _ELLIPSIS.split(q)]
    parts = [p for p in parts if p]
    if not parts or any(len(p.split()) < min_words for p in parts):
        return False
    if all(p in src for p in parts):
        return True
    # Repli : l'extraction des PDF coupe parfois des mots par une espace (« diag nosis », « eli gible »).
    # On compare alors sans aucune espace : la suite exacte de caractères reste exigée.
    compact = src.replace(" ", "")
    return all(p.replace(" ", "") in compact for p in parts)


_NUM = re.compile(r"\d+(?:\.\d+)?|\.\d+")
_THOUSANDS = re.compile(r"(?<=\d)[,\u00a0\u202f ](?=\d{3}(?!\d))")


def significant_numbers(text: str) -> set[str]:
    """Nombres d'un texte, hors petits entiers (« 3 », « 95 ») trop fréquents pour prouver quoi que ce soit.

    Les séparateurs de milliers sont retirés : « 78 849 » et « 78,849 » donnent le même nombre.
    """
    text = _THOUSANDS.sub("", _decimal_points(unicodedata.normalize("NFKC", text)))
    return {n for n in _NUM.findall(text) if "." in n or len(n) >= 3}


def numbers_in_source(value: str, source: str) -> bool | None:
    """Vrai si tous les nombres significatifs de `value` figurent dans la source ; None s'il n'y en a aucun."""
    nums = significant_numbers(value)
    if not nums:
        return None
    return nums <= significant_numbers(source)
