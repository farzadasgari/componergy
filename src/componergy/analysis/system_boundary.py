"""Validate that (demand - generation) actually behaves like net imports."""

from __future__ import annotations

import numpy as np
import pandas as pd

CA_NET_IMPORT_FRACTION_RANGE = (0.15, 0.40)
