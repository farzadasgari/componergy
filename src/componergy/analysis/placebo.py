"""Placebo (permutation) test for confounding in event-composite results."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_N_PERM = 2000
DEFAULT_BLOCK_MONTHS = 12
