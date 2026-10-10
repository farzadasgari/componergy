import pytest

from componergy.figures.labels import (
    REGION_ABBREV,
    region_abbrev,
    region_label_table,
    duplicate_labels,
    format_region_key,
)

ELEVEN = list(REGION_ABBREV)


def test_the_eleven_real_regions_get_the_manuscript_labels_and_all_labels_are_unique():
    want = {"North Central": "N-C", "Sonoran Desert": "SoD", "Northeast": "NE", "Sierra": "SR",
            "San Joaquin Valley": "SJV", "Sacramento-Delta": "SD", "North Coast": "NC"}
    for name, lab in want.items():
        assert region_abbrev(name) == lab
    assert len(ELEVEN) == 11
    assert duplicate_labels(ELEVEN) == []
    assert region_label_table(ELEVEN)["Abbreviation"].is_unique


def test_regression_the_generic_initials_rule_collided_on_these_regions():
    def initials(n):
        return "".join(p[0].upper() for p in [p for p in n.replace("-", " ").split() if p][:3])

    assert initials("North Coast") == initials("North Central") == "NC"
    assert initials("Sacramento-Delta") == initials("Sonoran Desert") == "SD"
    assert region_abbrev("North Coast") != region_abbrev("North Central")
    assert region_abbrev("Sacramento-Delta") != region_abbrev("Sonoran Desert")


def test_an_unknown_region_falls_back_to_initials_without_colliding_with_a_known_one():
    t = region_label_table(["North Coast", "Northern Coastal Cliffs"]).set_index("Region")
    assert t.loc["North Coast", "Abbreviation"] == "NC"
    assert t.loc["Northern Coastal Cliffs", "Abbreviation"] != "NC"
    assert t["Abbreviation"].is_unique
    assert t.loc["Northern Coastal Cliffs", "Source"].startswith("fallback")


def test_two_unknown_regions_with_the_same_initials_are_disambiguated():
    t = region_label_table(["Alpha Beta", "Alpha Bravo"])
    assert t["Abbreviation"].is_unique


def test_the_key_is_caption_ready_and_lists_every_region():
    key = format_region_key(["North Coast", "North Central"])
    assert key == "NC, North Coast; N-C, North Central"
