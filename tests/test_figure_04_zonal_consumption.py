import numpy as np
import pandas as pd

from componergy.figures.zonal_consumption import (
    region_abbrev, text_color_for_bg, compute_zonal_composites,
)


def test_region_abbrev_uses_first_letters_of_up_to_three_words():
    assert region_abbrev("San Joaquin Valley") == "SJV"
    assert region_abbrev("North Coast") == "NC"
    assert region_abbrev("Sacramento-Delta") == "SD"
