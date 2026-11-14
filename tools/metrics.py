"""Métriques d'évaluation (accord inter-relecteurs)."""
from __future__ import annotations


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Kappa de Cohen pour deux classements binaires. None si la liste est vide."""
    if len(a) != len(b):
        raise ValueError("Les deux listes doivent avoir la même longueur")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:  # les deux relecteurs répondent toujours la même chose : accord parfait
        return 1.0
    return (po - pe) / (1 - pe)
