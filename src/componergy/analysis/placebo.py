"""Placebo (permutation) test for confounding in event-composite results."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_PERM = 2000
DEFAULT_BLOCK_MONTHS = 12


def circular_shift_mask(mask: np.ndarray, shift: int) -> np.ndarray:
    """Roll an event mask through the record, wrapping at the ends."""
    return np.roll(np.asarray(mask, dtype=bool), shift)
