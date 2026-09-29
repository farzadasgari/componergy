"""
Hovmoller diagrams of STI, SAPEI and SCDHI.

Four variants of one plot, matching the published captions:
    1. annual  x latitude
    2. annual  x longitude
    3. monthly x latitude
    4. monthly x longitude
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 600,
})

import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
