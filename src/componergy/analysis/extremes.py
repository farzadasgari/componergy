"""
Extreme-value analysis of the systemic risk series: empirical plotting
positions plus two fitted extreme-value models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MONTHS_PER_YEAR = 12
