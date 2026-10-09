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


def main() -> int:
    from componergy.workflow import plan
    a = parse_args()
    steps = plan(only=a.only, start=a.start, end=a.end, skip_missing=a.skip_missing,
                 baseline=a.baseline, demand_source=a.demand_source)
    for i, s in enumerate(steps, 1):
        print(f"[{i:>2}/{len(steps)}] {s['stage']:<14} {s['status']:<48} {s['description']}")
    if a.dry_run:
        return 0
    blocked = [s for s in steps if s["status"].startswith("BLOCKED")]
    if blocked:
        print("\nstopping: inputs missing for", ", ".join(s["stage"] for s in blocked),
              "-- supply them, or use --skip-missing")
        return 2
    failures = []
    for s in steps:
        if not s["status"] == "run":
            continue
        print(f"\n=== {s['stage']} ===\n$ {' '.join(s['command'])}", flush=True)
        if subprocess.run(s["command"]).returncode != 0:
            failures.append(s["stage"])
            if not a.continue_on_error:
                break
    print("\nfailed stages:", failures or "none")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
