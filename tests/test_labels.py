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
