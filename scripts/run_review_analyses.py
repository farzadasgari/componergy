"""CLI entry point for the review-response analyses added after peer review."""

from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the post-review analyses on the processed system-risk series.")
    parser.add_argument("--risk-csv", default=None,
                        help="Path to figure_07_monthly_system_risk.csv. Defaults to the standard figures location.")
    parser.add_argument("--threshold-q", type=float, default=0.90,
                        help="High-risk quantile for validation and POT threshold.")
    return parser.parse_args()
