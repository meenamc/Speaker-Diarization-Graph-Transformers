"""Multi-segmentation refinement utilities.

The refinement step compares short speech windows around word-boundary candidate
change points. A change is emitted when the most similar before/after window pair
falls below a similarity threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class WordSegment:
    word: str
    start: float
    end: float


def detect_change_points(
    word_segments: Sequence[WordSegment],
    similarity_fn: Callable[[float, float, float, float], float],
    threshold: float,
    window: float = 0.75,
) -> list[int]:
    """Detect speaker changes at word boundaries.

    Parameters
    ----------
    word_segments:
        Word-level segments with start/end times.
    similarity_fn:
        Function called as ``similarity_fn(start_a, end_a, start_b, end_b)``.
        It should return a larger value for more similar acoustic windows.
    threshold:
        Candidate boundary is accepted when the best local similarity is below
        this value.
    window:
        Maximum window duration around each candidate boundary.
    """
    change_points: list[int] = []
    if len(word_segments) < 2:
        return change_points

    for idx in range(1, len(word_segments)):
        left = word_segments[idx - 1]
        right = word_segments[idx]
        boundary = right.start

        before_start = max(left.start, boundary - window)
        before_end = boundary
        after_start = boundary
        after_end = min(right.end, boundary + window)

        similarity = similarity_fn(before_start, before_end, after_start, after_end)
        if np.isfinite(similarity) and similarity < threshold:
            change_points.append(idx)

    return change_points
