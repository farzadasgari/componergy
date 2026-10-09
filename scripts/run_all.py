"""Run the whole study, or any contiguous/selected part of it, in the documented order."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    from componergy.workflow import STAGE_NAMES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=STAGE_NAMES)
    p.add_argument("--from", dest="start", choices=STAGE_NAMES)
    p.add_argument("--to", dest="end", choices=STAGE_NAMES)
    p.add_argument("--skip-missing", action="store_true", help="skip stages whose inputs are not on disk")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--baseline", choices=["full_record", "window"], default=None)
    p.add_argument("--demand-source", choices=["cec", "eia861m"], default=None)
    return p.parse_args()