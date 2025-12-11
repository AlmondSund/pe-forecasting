"""Permutation entropy utilities (pure Python).

Implements:
- `ordinal_pattern`: stable ordinal pattern from a window.
- `permutation_entropy` (PE): normalized Shannon entropy of ordinal patterns.
- `weighted_permutation_entropy` (WPE): PE weighted by local variance.
- `multiscale_pe`: PE/WPE over multiple delays.
- `sliding_windows`: helper to segment a series into overlapping windows.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple


def ordinal_pattern(window: Sequence[float]) -> Tuple[int, ...]:
    """Return the ordinal pattern indices with deterministic tie-breaking."""
    return tuple(
        idx for idx, _ in sorted(enumerate(window), key=lambda p: (p[1], p[0]))
    )


def _window(series: Sequence[float], start: int, m: int, tau: int) -> List[float]:
    return [series[start + k * tau] for k in range(m)]


def permutation_entropy(
    series: Sequence[float], m: int = 3, tau: int = 1, normalize: bool = True
) -> float:
    n = len(series)
    usable = n - (m - 1) * tau
    if usable <= 0:
        return float("nan")
    counts: Counter[Tuple[int, ...]] = Counter()
    for start in range(usable):
        pat = ordinal_pattern(_window(series, start, m, tau))
        counts[pat] += 1
    total = float(sum(counts.values()))
    probs = (c / total for c in counts.values())
    ent = -sum(p * math.log(p) for p in probs)
    return ent / math.log(math.factorial(m)) if normalize else ent


def weighted_permutation_entropy(
    series: Sequence[float], m: int = 3, tau: int = 1, normalize: bool = True
) -> float:
    n = len(series)
    usable = n - (m - 1) * tau
    if usable <= 0:
        return float("nan")
    counts: Counter[Tuple[int, ...]] = Counter()
    total_weight = 0.0
    for start in range(usable):
        win = _window(series, start, m, tau)
        mean = sum(win) / m
        var = sum((v - mean) ** 2 for v in win) / m
        weight = var
        counts[ordinal_pattern(win)] += weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    probs = (c / total_weight for c in counts.values())
    ent = -sum(p * math.log(p) for p in probs)
    return ent / math.log(math.factorial(m)) if normalize else ent


def multiscale_pe(
    series: Sequence[float],
    m: int = 3,
    taus: Iterable[int] = (1, 2, 3, 4),
    weighted: bool = False,
) -> List[Tuple[int, float]]:
    """Compute PE (or WPE if weighted=True) for multiple delays."""
    out: List[Tuple[int, float]] = []
    for tau in taus:
        val = (
            weighted_permutation_entropy(series, m=m, tau=tau)
            if weighted
            else permutation_entropy(series, m=m, tau=tau)
        )
        out.append((tau, val))
    return out


def sliding_windows(
    series: Sequence[float], window: int, step: int
) -> List[Sequence[float]]:
    """Return overlapping fixed-size windows from a 1D sequence."""
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    frames: List[Sequence[float]] = []
    for start in range(0, max(0, len(series) - window + 1), step):
        frames.append(series[start : start + window])
    return frames
