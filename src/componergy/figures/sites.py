"""climate-extreme indices at six representative California locations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Six locations spanning California's distinct climate zones.
# Coordinates are approximate city/area centroids.
REPRESENTATIVE_SITES = [
    ("Eureka (North Coast)", 40.80, -124.16),
    ("Sacramento (Sacramento Valley)", 38.58, -121.49),
    ("Fresno (San Joaquin Valley)", 36.75, -119.77),
    ("San Luis Obispo (Central Coast)", 35.28, -120.66),
    ("Los Angeles (South Coast)", 34.05, -118.24),
    ("Barstow (Mojave Desert)", 34.90, -117.02),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 600,
})

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
