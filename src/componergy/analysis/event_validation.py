"""Out-of-sample sanity check"""

from __future__ import annotations

import numpy as np
import pandas as pd

KNOWN_GRID_EVENTS = {
    "2020-08": "CAISO rotating outages during the August 2020 West-wide heat wave",
    "2020-09": "September 2020 record heat and statewide Flex Alerts",
    "2022-09": "September 2022 heat dome; CAISO record demand and emergency alerts",
    "2021-07": "July 2021 Bootleg Fire / drought-constrained hydro and transfer limits",
    "2024-07": "July 2024 sustained statewide heat wave",
}
