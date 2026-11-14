"""Découpage d'un texte intégral en passages (chunks) pour le RAG."""
from __future__ import annotations

import re


def _split_long(piece: str, max_chars: int) -> list[str]:
    """Coupe un paragraphe trop long aux fins de phrases (coupe dure en dernier recours)."""
    if len(piece) <= max_chars:
        return [piece]
    out: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?])\s+", piece):
        while len(sentence) > max_chars:  # phrase interminable ou texte sans ponctuation
            if cur:
                out.append(cur)
                cur = ""
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if not sentence:
            continue
        if cur and len(cur) + 1 + len(sentence) > max_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Regroupe les paragraphes en passages d'environ `max_chars` caractères, avec un léger recouvrement.

    Un titre de section ("## Methods") est un paragraphe à part : il se retrouve donc collé au texte qui suit.
    Longueur maximale d'un passage : max_chars + overlap + 2.
    """
    pieces: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if para:
            pieces.extend(_split_long(para, max_chars))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 2 + len(piece) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail[tail.find(" ") + 1 :]  # le recouvrement commence sur un mot entier
            cur = f"{tail}\n\n{piece}" if tail else piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks
