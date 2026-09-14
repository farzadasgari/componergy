from componergy.figures.print_style import SOURCE_STYLE, EVENT_STYLE


def test_no_two_sources_share_the_same_linestyle_and_marker_combination():
    combos = [(s["linestyle"], s["marker"]) for s in SOURCE_STYLE.values()]
    assert len(combos) == len(
        set(combos)), "two sources share a (linestyle, marker) pair -- would be indistinguishable in grayscale if colors also converge"


def test_no_two_sources_share_the_same_hatch():
    hatches = [s["hatch"] for s in SOURCE_STYLE.values()]
    assert len(hatches) == len(set(hatches))


def test_no_two_events_share_the_same_hatch():
    hatches = [s["hatch"] for s in EVENT_STYLE.values()]
    assert len(hatches) == len(set(hatches))
