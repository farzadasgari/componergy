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


def main() -> None:
    from componergy.analysis.extremes import fit_gev_annual_maxima, fit_pot_gpd, return_period_of_value
    from componergy.analysis.event_validation import validate_against_known_events, summarize_validation
    from componergy.paths import FIGURES_DIR

    args = parse_args()
    risk_csv = args.risk_csv or (FIGURES_DIR / "figure_07" / "figure_07_monthly_system_risk.csv")

    df = pd.read_csv(risk_csv, parse_dates=[0], index_col=0)
    df.index = pd.DatetimeIndex(df.index.values, name=None)
    risk = df["System_Risk"]

    gev = fit_gev_annual_maxima(risk)
    pot = fit_pot_gpd(risk, threshold_q=args.threshold_q)
    obs_max = float(risk.max())
    rp = return_period_of_value(obs_max, gev, pot)

    print(f"observed max system risk: {obs_max:.4f}")
    print(f"  empirical return period (record-length-bound): {((len(risk) + 1) / 12.0):.1f} yr")
    print(f"  GEV-fitted return period: {rp['gev_return_period_years']:.1f} yr")
    print(f"  POT-fitted return period: {rp['pot_return_period_years']:.1f} yr")
    print()

    validation = validate_against_known_events(risk, high_risk_q=args.threshold_q)
    print("validation against documented grid emergencies:")
    print(validation.to_string(index=False))
    print()
    print("summary:", summarize_validation(validation, high_risk_q=args.threshold_q))
    print()

    from componergy.analysis.study_periods import collect_study_periods
    periods = collect_study_periods(FIGURES_DIR)
    if len(periods):
        print("study period actually used by each figure:")
        print(periods.to_string(index=False))
    else:
        print("no figure logs found; run the figure scripts first to populate study periods.")


if __name__ == "__main__":
    main()
