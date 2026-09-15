import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from componergy.analysis.regions import build_county_to_region_map, compute_regional_mean
