import numpy as np
import pandas as pd

from componergy.figures.zonal_consumption import (
    region_abbrev, text_color_for_bg, compute_zonal_composites,
)


def test_region_abbrev_uses_first_letters_of_up_to_three_words():
    assert region_abbrev("San Joaquin Valley") == "SJV"
    assert region_abbrev("North Coast") == "NC"
    assert region_abbrev("Sacramento-Delta") == "SD"


def test_text_color_for_bg_picks_black_on_light_white_on_dark():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-1, vmax=1)

    assert text_color_for_bg(0.0, norm, cmap) == "black"
    assert text_color_for_bg(-1.0, norm, cmap) == "white"
    assert text_color_for_bg(np.nan, norm, cmap) == "black"
