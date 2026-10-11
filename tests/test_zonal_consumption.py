import numpy as np
import pandas as pd

from tests.synthetic_inputs import make_no_event_effect_series
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


def test_compute_zonal_composites_fdr_corrects_across_all_region_event_tests():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2008-01-01", periods=n, freq="MS")

    event_mask = pd.Series([True] * 20 + [False] * (n - 20), index=idx)
    normal_mask = pd.Series([False] * 20 + [True] * (n - 20), index=idx)

    raw_a = pd.Series(rng.normal(1000, 20, n), index=idx)
    raw_a.iloc[:20] += 150
    raw_b = make_no_event_effect_series(rng.normal(500, 20, n), idx, event_mask)

    electricity_raw = pd.DataFrame({"RegionA": raw_a, "RegionB": raw_b})
    electricity_z = electricity_raw.apply(lambda s: (s - s.mean()) / s.std())

    data = {
        "electricity_raw": electricity_raw,
        "electricity_z": electricity_z,
        "event_masks": {"Heatwave": event_mask},
        "normal": normal_mask,
    }

    comp = compute_zonal_composites(data)
    assert len(comp) == 2
    assert "q_value" in comp.columns

    assert np.all(comp["q_value"].values >= comp["p_value"].values - 1e-12)

    row_a = comp[comp["Region"] == "RegionA"].iloc[0]
    row_b = comp[comp["Region"] == "RegionB"].iloc[0]
    assert abs(row_a["q_value"] - len(comp) * row_a["p_value"]) < 1e-12
    assert row_a["q_value"] < 0.05
    assert row_b["q_value"] > 0.05
    assert row_a["Delta_pct"] > 10
    assert abs(row_b["Delta_pct"]) < 5
