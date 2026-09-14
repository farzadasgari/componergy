from pathlib import Path
import tempfile
from unittest.mock import patch

import componergy.paths as paths


def test_pipeline_dirs_contains_no_file_path_constants():
    file_constants = {
        paths.NOAA_MONTHLY_FILE,
        paths.INDICES_FILE,
        paths.EIA_GENERATION_RAW_FILE,
        paths.GENERATION_MONTHLY_FILE,
        paths.MANIFEST_PATH,
    }
    for d in paths._PIPELINE_DIRS:
        assert d not in file_constants, f"{d} is a file path and must not be in _PIPELINE_DIRS"
