""""Generation-mix-by-source figure: statewide timeseries and composite event-anomaly bars, one output file per source."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from componergy.figures.trend_stats import compute_area_weights, compute_statewide_mean
from componergy.figures.print_style import SOURCE_STYLE, EVENT_STYLE
from componergy.figures.composite_stats import series_to_data_array, compute_composite_table, best_lag_slope
from componergy.figures.stats_log import write_figure_log
from componergy.analysis.events import (
    classify_heatwave, classify_drought, classify_compound,
)
from componergy.paths import INDICES_FILE, GENERATION_MONTHLY_FILE, FIGURES_DIR

SMOOTH_MONTHS = 3
LAGS = range(-6, 7)
SOURCE_COLUMNS = ["Fossil", "Hydro", "Solar", "Wind", "Nuclear", "Other Renewable"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not std or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def deseasonalize_monthly(series: pd.Series) -> pd.Series:
    clim = series.groupby(series.index.month).mean()
    return series - series.index.month.map(clim)


def style_timeseries_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))


def style_bar_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def add_letter(ax, letter, x=0.01, y=0.98, fs=19):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=fs, clip_on=False)


def prepare_data(indices_ds: xr.Dataset, generation_df: pd.DataFrame, sapei_var="sapei_3m", scdhi_var="scdhi_3m"):
    """
    Align statewide climate signals with generation shares and classify events.
    """
    weights = compute_area_weights(indices_ds["lat"])
    sti = zscore(compute_statewide_mean(indices_ds["sti"], weights).to_series())
    sapei = zscore(compute_statewide_mean(indices_ds[sapei_var], weights).to_series())
    scdhi = zscore(compute_statewide_mean(indices_ds[scdhi_var], weights).to_series())

    mix_pct = generation_df[SOURCE_COLUMNS].div(generation_df["Total"], axis=0) * 100.0
    mix_pct_sm = mix_pct.rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()

    common = mix_pct_sm.index.intersection(sti.index).intersection(sapei.index).intersection(scdhi.index)
    mix_pct_sm = mix_pct_sm.loc[common]
    sti, sapei, scdhi = sti.loc[common], sapei.loc[common], scdhi.loc[common]

    mix_z_total = mix_pct_sm.apply(zscore, axis=0)
    mix_z_local = mix_pct_sm.apply(deseasonalize_monthly, axis=0).apply(zscore, axis=0)

    heat = classify_heatwave(series_to_data_array(sti)).to_series()
    drought = classify_drought(series_to_data_array(sapei)).to_series()
    compound = classify_compound(series_to_data_array(scdhi)).to_series()

    heat_only = heat & (~drought)
    drought_only = drought & (~heat)
    normal = ~(heat | drought | compound)

    event_masks = {"Heatwave": heat_only, "Drought": drought_only, "Compound": compound}

    return {
        "drivers": {"STI": sti, "SAPEI": sapei, "SCDHI": scdhi},
        "mix_pct_sm": mix_pct_sm,
        "mix_z_total": mix_z_total,
        "mix_z_local": mix_z_local,
        "event_masks": event_masks,
        "normal": normal,
        "compound_months": mix_pct_sm.index[compound.reindex(mix_pct_sm.index).fillna(False).values],
    }


def shade_compound_months(ax, compound_months):
    for d in compound_months:
        ax.axvspan(d, d + pd.offsets.MonthEnd(0), color="#d73027", alpha=0.14, linewidth=0)


def plot_mix_timeseries(ax, data):
    for col in SOURCE_COLUMNS:
        style = SOURCE_STYLE[col]
        ax.plot(data["mix_pct_sm"].index, data["mix_pct_sm"][col].values,
                color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
                markevery=24, markersize=6, lw=2.2, label=col)
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Time (Year)")
    shade_compound_months(ax, data["compound_months"])
    style_timeseries_axis(ax)

    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.34),
              borderaxespad=0.0, handlelength=2.6, columnspacing=1.2)


def plot_single_source_timeseries(ax, data, src):
    style = SOURCE_STYLE[src]
    s = data["mix_pct_sm"][src]
    ax.plot(s.index, s.values, color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markevery=24, markersize=6, lw=2.4)
    ax.set_ylabel(f"{src} share (%)")
    ax.set_xlabel("Time (Year)")
    shade_compound_months(ax, data["compound_months"])
    style_timeseries_axis(ax)


def plot_mix_composite_bars(ax, comp_table):
    order_src = SOURCE_COLUMNS
    events_plot = ["Heatwave", "Drought", "Compound"]
    x = np.arange(len(order_src))
    w = 0.26

    for j, ev in enumerate(events_plot):
        ev_tbl = comp_table[comp_table["Event"] == ev].set_index("Source")
        vals = ev_tbl.reindex(order_src)["Difference"].to_numpy(dtype=float)
        style = EVENT_STYLE[ev]
        ax.bar(x + (j - 1) * w, vals, width=w, color=style["color"], hatch=style["hatch"],
               alpha=0.9, edgecolor="black", linewidth=1, label=ev)

    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order_src, rotation=40, ha="right")
    ax.set_ylabel("Generation share anomaly ($\\Delta z$)")
    style_bar_axis(ax)

    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.30), borderaxespad=0.0)


def plot_single_source_composite_bars(ax, comp_table, src):
    events_plot = ["Heatwave", "Drought", "Compound"]
    tbl = comp_table[comp_table["Source"] == src].set_index("Event")
    x = np.arange(len(events_plot))

    for i, ev in enumerate(events_plot):
        val = tbl.loc[ev, "Difference"] if ev in tbl.index else np.nan
        style = EVENT_STYLE[ev]
        ax.bar(x[i], val, width=0.6, color=style["color"], hatch=style["hatch"],
               alpha=0.9, edgecolor="black", linewidth=1, label=ev)

    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(events_plot)
    ax.set_ylabel("Share anomaly ($\\Delta z$)")
    style_bar_axis(ax)

    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.30), borderaxespad=0.0)


def save_two_panel_figure(fig_path, left_label, right_label, left_plot_fn, right_plot_fn):
    """
    Render a timeseries + composite-bars panel pair and save to fig_path.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.28})
    fig.subplots_adjust(top=0.72, bottom=0.16, left=0.06, right=0.98)

    left_plot_fn(axL)
    right_plot_fn(axR)

    add_letter(axL, left_label, x=-0.08, y=1.32)
    add_letter(axR, right_label, x=-0.12, y=1.32)

    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main(sapei_var: str = "sapei_3m", scdhi_var: str = "scdhi_3m") -> None:
    """Build all 6 output figures, the composite/lag-scan CSVs, and the log/json summary."""
    from componergy.netcdf_io import atomic_to_csv

    out_dir = FIGURES_DIR / "figure_02"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {INDICES_FILE} and {GENERATION_MONTHLY_FILE}...")
    indices_ds = xr.open_dataset(INDICES_FILE)
    generation_df = pd.read_csv(GENERATION_MONTHLY_FILE, parse_dates=[0], index_col=0)
    generation_df.index = pd.DatetimeIndex(generation_df.index.values, name=None)

    data = prepare_data(indices_ds, generation_df, sapei_var=sapei_var, scdhi_var=scdhi_var)

    comp_total = compute_composite_table(data["mix_z_total"], data["event_masks"], data["normal"])
    comp_total.insert(0, "Mode", "total_share")
    comp_local = compute_composite_table(data["mix_z_local"], data["event_masks"], data["normal"])
    comp_local.insert(0, "Mode", "deseasonalized_share")
    comp_all = pd.concat([comp_total, comp_local], ignore_index=True)

    lag_rows = []
    for mode_name, mix_z in [("total_share", data["mix_z_total"]), ("deseasonalized_share", data["mix_z_local"])]:
        for src in mix_z.columns:
            y = mix_z[src].dropna()
            for driver_name, driver_series in data["drivers"].items():
                x = driver_series.reindex(y.index)
                best = best_lag_slope(x, y, lags=LAGS, min_n=60)
                if best is None:
                    continue
                lag_rows.append({"Mode": mode_name, "Source": src, "Driver": driver_name, **best})

    slope_df = pd.DataFrame(lag_rows)

    print("saving figures...")
    save_two_panel_figure(
        out_dir / "figure_02A_mix_composite.jpg", "A", "B",
        left_plot_fn=lambda ax: plot_mix_timeseries(ax, data),
        right_plot_fn=lambda ax: plot_mix_composite_bars(ax, comp_total),
    )
    single_source_panels = [
        ("figure_02B_fossil.jpg", "C", "D", "Fossil"),
        ("figure_02C_hydro.jpg", "E", "F", "Hydro"),
        ("figure_02D_solar.jpg", "G", "H", "Solar"),
        ("figure_02E_wind.jpg", "I", "J", "Wind"),
        ("figure_02F_nuclear.jpg", "K", "L", "Nuclear"),
    ]
    for fname, left_letter, right_letter, src in single_source_panels:
        save_two_panel_figure(
            out_dir / fname, left_letter, right_letter,
            left_plot_fn=lambda ax, s=src: plot_single_source_timeseries(ax, data, s),
            right_plot_fn=lambda ax, s=src: plot_single_source_composite_bars(ax, comp_total, s),
        )

    atomic_to_csv(comp_all, out_dir / "figure_02_composite.csv", index=False)
    atomic_to_csv(slope_df, out_dir / "figure_02_lag_scan.csv", index=False)

    log_sections = {
        "Overview": {
            "sapei_variable": sapei_var, "scdhi_variable": scdhi_var,
            "smoothing_months": SMOOTH_MONTHS,
            "year_start": int(data["mix_pct_sm"].index.year.min()),
            "year_end": int(data["mix_pct_sm"].index.year.max()),
            "n_months": len(data["mix_pct_sm"]),
        },
        "Event counts": {k: int(v.sum()) for k, v in data["event_masks"].items()},
        "Note": (
            "Generation shares reflect both climate-driven anomalies and secular "
            "changes in installed capacity over the record; the latter is not "
            "controlled for here (would require an installed-capacity dataset)."
        ),
    }
    write_figure_log(out_dir / "figure_02_generation_response", log_sections)
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
