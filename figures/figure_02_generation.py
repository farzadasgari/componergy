from componergy.paths import CALIFORNIA_MONTHLY_FILE, GENERATION_MONTHLY_FILE, FIGURES_DIR
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NC_PATH = str(CALIFORNIA_MONTHLY_FILE)
GEN_PATH = str(GENERATION_MONTHLY_FILE)

OUT_DIR = str(FIGURES_DIR / "Figure 2")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_A = os.path.join(OUT_DIR, "figure 2A - mix + composite.jpg")
FIG_B = os.path.join(OUT_DIR, "figure 2B - fossil + composite.jpg")
FIG_C = os.path.join(OUT_DIR, "figure 2C - hydro + composite.jpg")
FIG_D = os.path.join(OUT_DIR, "figure 2D - solar + composite.jpg")
FIG_E = os.path.join(OUT_DIR, "figure 2E - wind + composite.jpg")
FIG_F = os.path.join(OUT_DIR, "figure 2F - nuclear + composite.jpg")

LOG_PATH = os.path.join(OUT_DIR, "figure 2 - generation response.log")
CSV_EVENTS = os.path.join(OUT_DIR, "figure 2 - event_counts.csv")
CSV_COMPOS = os.path.join(OUT_DIR, "figure 2 - composite_generation.csv")
CSV_SLOPES = os.path.join(OUT_DIR, "figure 2 - sensitivity_slopes.csv")
CSV_LAGSCAN = os.path.join(OUT_DIR, "figure 2 - lag_scan.csv")

SMOOTH_MONTHS = 3
ALPHA_SIG = 0.05
LAGS = np.arange(-6, 7)

P_HEAT = 0.90
P_DROUGHT = 0.10
P_COMP = 0.10
NORMAL_Q = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
    "savefig.dpi": 600,
})

C_STI = "crimson"
C_SAPEI = "deepskyblue"
C_SCDHI = "purple"
C_SHADE = "#d73027"

C_FOSSIL = "#fc8c23"
C_HYDRO = "#1f78b4"
C_SOLAR = "#33a02c"
C_WIND = "#6a3d9a"
C_NUC = "#e31a1c"
C_OTHER = "#777777"


def add_letter(ax, letter, x=0.01, y=0.98, fs=18):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fs, fontweight="normal",
        clip_on=False
    )


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


def guess_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "y", "YLAT", "LAT"]
    lon_candidates = ["lon", "longitude", "x", "XLONG", "LON"]
    lat = next((n for n in lat_candidates if n in ds.coords), None)
    lon = next((n for n in lon_candidates if n in ds.coords), None)
    if lat is None or lon is None:
        raise ValueError(
            f"Could not find lat/lon in coords: {list(ds.coords)}")
    return lat, lon


def year_month_to_index(years, months):
    return pd.to_datetime([f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(years, months)])


def stack_year_month_to_series(da_ym):
    da_stk = da_ym.stack(time=("year", "month")).sortby("time")
    idx = year_month_to_index(da_stk["year"].values, da_stk["month"].values)
    return pd.Series(da_stk.values.astype(float), index=idx)


def rolling_mean(s, w):
    return s.rolling(w, center=True, min_periods=1).mean()


def zscore(s):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if (sd and np.isfinite(sd)) else s * 0.0


def deseasonalize_monthly(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    clim = s.groupby(s.index.month).mean()
    return s - s.index.month.map(clim)


def area_weights_2d(ds, lat_name, lon_name):
    lat = ds[lat_name]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w2d = w_lat.broadcast_like(ds[lat_name] * 0 + ds[lon_name] * 0 + 1)
    return w2d


def area_weighted_mean_monthly(da, w2d, lat_name, lon_name):
    num = (da * w2d).sum(dim=[lat_name, lon_name], skipna=True)
    den = (xr.where(np.isfinite(da), 1.0, np.nan) *
           w2d).sum(dim=[lat_name, lon_name], skipna=True)
    out = num / den
    return stack_year_month_to_series(out)


def best_lag_slope(x, y, lags=LAGS, min_n=60):
    best = None
    for L in lags:
        if L < 0:
            xL = x.iloc[-L:].values
            yL = y.iloc[:len(xL)].values
        elif L > 0:
            xL = x.iloc[:-L].values
            yL = y.iloc[L:].values
        else:
            xL = x.values
            yL = y.values

        m = np.isfinite(xL) & np.isfinite(yL)
        if m.sum() < min_n:
            continue

        A = np.vstack([xL[m], np.ones(m.sum())]).T
        slope, intercept = np.linalg.lstsq(A, yL[m], rcond=None)[0]
        yhat = slope * xL[m] + intercept
        ssr = np.sum((yL[m] - yhat) ** 2)
        sst = np.sum((yL[m] - np.mean(yL[m])) ** 2)
        r2 = 1.0 - ssr / sst if sst > 0 else np.nan

        cand = {"R2": float(r2), "Lag": int(L), "Slope": float(slope)}
        if (best is None) or (np.nan_to_num(cand["R2"], nan=-np.inf) > np.nan_to_num(best["R2"], nan=-np.inf)):
            best = cand
    return best


def welch_ttest(a, b):
    a = pd.to_numeric(a, errors="coerce").astype(float).dropna()
    b = pd.to_numeric(b, errors="coerce").astype(float).dropna()
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    tstat, pval = stats.ttest_ind(a, b, equal_var=False)
    return float(tstat), float(pval)


def summarize_mask(mask, name):
    return {"name": name, "n": int(mask.sum()), "frac": float(mask.mean())}


ds = xr.open_dataset(NC_PATH)
LAT, LON = guess_lat_lon_names(ds)
w2d = area_weights_2d(ds, LAT, LON)

sti = ds["sti_scdhi"]
sapei = ds["sapei_scdhi"]
scdhi = ds["scdhi"]

sti_mean = area_weighted_mean_monthly(sti, w2d, LAT, LON)
sapei_mean = area_weighted_mean_monthly(sapei, w2d, LAT, LON)
scdhi_mean = area_weighted_mean_monthly(scdhi, w2d, LAT, LON)

sti_z = rolling_mean(zscore(sti_mean), SMOOTH_MONTHS)
sapei_z = rolling_mean(zscore(sapei_mean), SMOOTH_MONTHS)
scdhi_z = rolling_mean(zscore(scdhi_mean), SMOOTH_MONTHS)

thr_sti_hi = sti_z.quantile(P_HEAT)
thr_sapei_lo = sapei_z.quantile(P_DROUGHT)
thr_scdhi_lo = scdhi_z.quantile(P_COMP)

df = pd.read_csv(GEN_PATH)

if "TYPE OF PRODUCER" in df.columns:
    df = df[df["TYPE OF PRODUCER"] == "Total Electric Power Industry"].copy()

df["Date"] = pd.to_datetime(df["YEAR"].astype(int).astype(
    str) + "-" + df["MONTH"].astype(int).astype(str) + "-01")

gen_wide = (
    df.pivot_table(index="Date", columns="ENERGY SOURCE",
                   values="GENERATION (Megawatthours)", aggfunc="sum")
    .fillna(0.0)
)

gen_wide["Fossil"] = (
    gen_wide.get("Coal", 0) +
    gen_wide.get("Natural Gas", 0) +
    gen_wide.get("Petroleum", 0) +
    gen_wide.get("Other Gases", 0)
)
gen_wide["Hydro"] = gen_wide.get("Hydroelectric Conventional", 0)
gen_wide["Solar"] = gen_wide.get("Solar Thermal and Photovoltaic", 0)
gen_wide["Wind"] = gen_wide.get("Wind", 0)
gen_wide["Nuclear"] = gen_wide.get("Nuclear", 0)
gen_wide["Other Renew"] = (
    gen_wide.get("Geothermal", 0) +
    gen_wide.get("Wood and Wood Derived Fuels", 0) +
    gen_wide.get("Other Biomass", 0)
)

gen_wide["Renewables"] = gen_wide["Hydro"] + \
    gen_wide["Solar"] + gen_wide["Wind"] + gen_wide["Other Renew"]
gen_wide["Total"] = gen_wide["Fossil"] + \
    gen_wide["Renewables"] + gen_wide["Nuclear"]
gen_wide.loc[gen_wide["Total"] <= 0, "Total"] = np.nan

gen_total_twh = gen_wide["Total"] / 1e6  # MWh -> TWh
gen_total_twh_sm = gen_total_twh.rolling(
    SMOOTH_MONTHS, center=True, min_periods=1).mean()

mix_cols = ["Fossil", "Hydro", "Solar", "Wind", "Nuclear", "Other Renew"]
mix_pct = gen_wide[mix_cols].div(gen_wide["Total"], axis=0) * 100.0
mix_pct_sm = mix_pct.rolling(SMOOTH_MONTHS, center=True, min_periods=1).mean()

common = mix_pct_sm.index.intersection(sti_z.index).intersection(
    sapei_z.index).intersection(scdhi_z.index)
mix_pct_sm = mix_pct_sm.loc[common]
sti_z = sti_z.loc[common]
sapei_z = sapei_z.loc[common]
scdhi_z = scdhi_z.loc[common]
gen_total_twh_sm = gen_total_twh_sm.loc[common]

mix_z_total_shares = mix_pct_sm.apply(zscore, axis=0)
mix_ds = mix_pct_sm.apply(deseasonalize_monthly, axis=0)
mix_z_local_shares = mix_ds.apply(zscore, axis=0)

TOTAL_LABEL = "Total"
totalgen_z_total = zscore(gen_total_twh_sm)
totalgen_z_local = zscore(deseasonalize_monthly(gen_total_twh_sm))

mix_z_total = pd.concat([totalgen_z_total.rename(
    TOTAL_LABEL), mix_z_total_shares], axis=1)
mix_z_local = pd.concat([totalgen_z_local.rename(
    TOTAL_LABEL), mix_z_local_shares], axis=1)

heat = sti_z >= thr_sti_hi
drought = sapei_z <= thr_sapei_lo
compound = scdhi_z <= thr_scdhi_lo

events_over = {
    "Heatwave (STI ≥ 90p)": heat,
    "Drought (SAPEI ≤ 10p)": drought,
    "Compound (SCDHI ≤ 10p)": compound,
}

pure_heat = heat & (~drought)
pure_drought = drought  # & (~heat)

events_pure = {
    "Heatwave-only": pure_heat,
    "Drought-only": pure_drought,
    "Compound (SCDHI ≤ 10p)": compound,
}

normal = scdhi_z >= scdhi_z.quantile(NORMAL_Q)

compound_mask = compound.reindex(mix_pct_sm.index).fillna(False)
compound_months = mix_pct_sm.index[compound_mask.values]


def shade_compound_months(ax):
    for d in compound_months:
        ax.axvspan(d, d + pd.offsets.MonthEnd(0),
                   color=C_SHADE, alpha=0.16, linewidth=0)


def composite_table(mix_z, events_dict, normal_mask, label):
    rows = []
    for ev_name, ev_mask in events_dict.items():
        for src in mix_z.columns:
            ev_vals = mix_z.loc[ev_mask, src]
            nm_vals = mix_z.loc[normal_mask, src]
            diff = float(ev_vals.mean() - nm_vals.mean())
            tstat, pval = welch_ttest(ev_vals, nm_vals)
            rows.append({
                "Mode": label,
                "Event": ev_name,
                "Source": src,
                "N_event": int(ev_mask.sum()),
                "N_normal": int(normal_mask.sum()),
                "Mean_event": float(ev_vals.mean()),
                "Mean_normal": float(nm_vals.mean()),
                "Difference": diff,
                "tstat": tstat,
                "p_value": pval
            })
    return pd.DataFrame(rows)


comp_over_total = composite_table(
    mix_z_total, events_over, normal, "OVER (total)")
comp_pure_total = composite_table(
    mix_z_total, events_pure, normal, "PURE (total)")
comp_over_local = composite_table(
    mix_z_local, events_over, normal, "OVER (local)")
comp_pure_local = composite_table(
    mix_z_local, events_pure, normal, "PURE (local)")

comp_all = pd.concat([comp_over_total, comp_pure_total,
                     comp_over_local, comp_pure_local], ignore_index=True)
comp_all.to_csv(CSV_COMPOS, index=False)

drivers = {"STI": sti_z, "SAPEI": sapei_z, "SCDHI": scdhi_z}
lag_rows = []
slope_rows = []

for mode_name, mix_z in [("total", mix_z_total), ("local", mix_z_local)]:
    for src in mix_z.columns:
        y = mix_z[src].dropna()
        for drv_name, x0 in drivers.items():
            x = x0.reindex(y.index)
            yy = y.reindex(x.index)
            xx = x.reindex(yy.index)

            best = best_lag_slope(xx, yy, lags=LAGS, min_n=60)
            if best is None:
                continue

            for L in LAGS:
                if L < 0:
                    xL = xx.iloc[-L:]
                    yL = yy.iloc[:len(xL)]
                elif L > 0:
                    xL = xx.iloc[:-L]
                    yL = yy.iloc[L:]
                else:
                    xL = xx
                    yL = yy

                m = np.isfinite(xL.values) & np.isfinite(yL.values)
                if m.sum() < 60:
                    continue

                A = np.vstack([xL.values[m], np.ones(m.sum())]).T
                slope, intercept = np.linalg.lstsq(
                    A, yL.values[m], rcond=None)[0]
                yhat = slope * xL.values[m] + intercept
                ssr = np.sum((yL.values[m] - yhat) ** 2)
                sst = np.sum((yL.values[m] - np.mean(yL.values[m])) ** 2)
                r2 = 1.0 - ssr / sst if sst > 0 else np.nan

                lag_rows.append({
                    "Mode": mode_name,
                    "Source": src,
                    "Driver": drv_name,
                    "Lag": int(L),
                    "Slope": float(slope),
                    "R2": float(r2),
                    "N": int(m.sum())
                })

            slope_rows.append({
                "Mode": mode_name,
                "Source": src,
                "Driver": drv_name,
                "BestLag": best["Lag"],
                "BestR2": best["R2"],
                "Slope_at_BestLag": best["Slope"]
            })

lag_df = pd.DataFrame(lag_rows)
slope_df = pd.DataFrame(slope_rows)
lag_df.to_csv(CSV_LAGSCAN, index=False)
slope_df.to_csv(CSV_SLOPES, index=False)

events_plot = ["Heatwave-only", "Drought-only", "Compound (SCDHI ≤ 10p)"]
colors_ev = {"Heatwave-only": C_STI, "Drought-only": C_SAPEI,
             "Compound (SCDHI ≤ 10p)": C_SCDHI}
legend_label = {
    "Heatwave-only": "Heatwave",
    "Drought-only": "Drought",
    "Compound (SCDHI ≤ 10p)": "Compound",
}

order_src = [TOTAL_LABEL, "Fossil", "Hydro",
             "Solar", "Wind", "Nuclear", "Other Renew"]

sub_pure_total = comp_all[(comp_all["Mode"] == "PURE (total)") & (
    comp_all["Event"].isin(events_plot))].copy()
sub_pure_total["Source"] = pd.Categorical(
    sub_pure_total["Source"], categories=order_src, ordered=True)


def plot_mix_timeseries(ax):
    plot_items = [
        ("Hydro", C_HYDRO),
        ("Solar", C_SOLAR),
        ("Wind", C_WIND),
        ("Nuclear", C_NUC),
        ("Other Renew", C_OTHER),
        ("Fossil", C_FOSSIL),
    ]

    for col, c in plot_items:
        ax.plot(mix_pct_sm.index,
                mix_pct_sm[col].values, lw=2.4, label=col, color=c)
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Time (Year)")

    # ax2 = ax.twinx()
    # htot = ax2.plot(gen_total_twh_sm.index, gen_total_twh_sm.values, color="black", lw=2.2, alpha=0.85)[0]
    # ax2.set_ylabel("Total generation (TWh)")
    # ax2.spines["top"].set_visible(False)
    # ax2.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ["Fossil", "Hydro", "Solar",
                                       "Wind", "Nuclear", "Other Renew"] if k in labels]
    leg = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        borderaxespad=0.0
    )
    for line in leg.get_lines():
        line.set_linewidth(4)

    # ax2.legend([htot], ["Total (TWh)"], frameon=False, loc="upper right")

    shade_compound_months(ax)

    style_timeseries_axis(ax)


def plot_single_source_timeseries(ax, src, color):
    s = mix_pct_sm[src]
    ax.plot(s.index, s.values, color=color, lw=2.4)
    ax.set_ylabel(f"{src} share (%)")
    ax.set_xlabel(f"Time (Year)")
    shade_compound_months(ax)
    style_timeseries_axis(ax)


def plot_mix_composite_bars(ax):
    subC = sub_pure_total.copy().sort_values(["Event", "Source"])
    x = np.arange(len(order_src))
    w = 0.26

    for j, ev in enumerate(events_plot):
        ev_tbl = subC[subC["Event"] == ev].set_index("Source")
        vals = ev_tbl.reindex(order_src)["Difference"].to_numpy(dtype=float)

        ax.bar(
            x + (j - 1) * w,
            vals,
            width=w,
            color=colors_ev[ev],
            alpha=0.90,
            edgecolor="black",
            linewidth=1,
            label=legend_label[ev]
        )

    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order_src, rotation=90)  # ha="right")
    ax.set_ylabel("Generation anomaly (Δz)")

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0.0
    )
    style_bar_axis(ax)


def plot_single_source_composite_bars(ax, src):
    tbl = sub_pure_total[sub_pure_total["Source"]
                         == src].copy().set_index("Event")
    x = np.arange(len(events_plot))
    vals = np.array([tbl.loc[ev, "Difference"]
                    if ev in tbl.index else np.nan for ev in events_plot], dtype=float)

    for i, ev in enumerate(events_plot):
        ax.bar(
            x[i],
            vals[i],
            width=0.65,
            color=colors_ev[ev],
            alpha=0.90,
            label=legend_label[ev],
            edgecolor="black",
            linewidth=1,
        )

    ax.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Heatwave", "Drought", "Compound"])
    ax.set_ylabel("Generation anomaly (Δz)")

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0.0
    )
    style_bar_axis(ax)


def save_two_panel_figure(fig_path, left_label, right_label, left_plot_fn, right_plot_fn):
    fig, (axL, axR) = plt.subplots(
        1, 2,
        figsize=(16.5, 4.8),
        gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.12}
    )
    left_plot_fn(axL)
    right_plot_fn(axR)

    add_letter(axL, left_label, x=0.01, y=0.98, fs=18)
    add_letter(axR, right_label, x=0.01, y=0.98, fs=18)

    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Save figures (each has 2 labeled subplots)
# -----------------------------
save_two_panel_figure(
    FIG_A, "A", "B",
    left_plot_fn=plot_mix_timeseries,
    right_plot_fn=plot_mix_composite_bars
)

save_two_panel_figure(
    FIG_B, "C", "D",
    left_plot_fn=lambda ax: plot_single_source_timeseries(
        ax, "Fossil", C_FOSSIL),
    right_plot_fn=lambda ax: plot_single_source_composite_bars(ax, "Fossil")
)

save_two_panel_figure(
    FIG_C, "E", "F",
    left_plot_fn=lambda ax: plot_single_source_timeseries(
        ax, "Hydro", C_HYDRO),
    right_plot_fn=lambda ax: plot_single_source_composite_bars(ax, "Hydro")
)

save_two_panel_figure(
    FIG_D, "G", "H",
    left_plot_fn=lambda ax: plot_single_source_timeseries(
        ax, "Solar", C_SOLAR),
    right_plot_fn=lambda ax: plot_single_source_composite_bars(ax, "Solar")
)

save_two_panel_figure(
    FIG_E, "I", "J",
    left_plot_fn=lambda ax: plot_single_source_timeseries(ax, "Wind", C_WIND),
    right_plot_fn=lambda ax: plot_single_source_composite_bars(ax, "Wind")
)

save_two_panel_figure(
    FIG_F, "K", "L",
    left_plot_fn=lambda ax: plot_single_source_timeseries(
        ax, "Nuclear", C_NUC),
    right_plot_fn=lambda ax: plot_single_source_composite_bars(ax, "Nuclear")
)

event_counts = []
for k, m in events_over.items():
    event_counts.append({"Mode": "OVER", **summarize_mask(m, k)})
for k, m in events_pure.items():
    event_counts.append({"Mode": "PURE", **summarize_mask(m, k)})
event_counts.append({"Mode": "BASE", **summarize_mask(normal,
                    f"Normal (SCDHI ≥ {int(NORMAL_Q*100)}p)")})

event_counts_df = pd.DataFrame(event_counts)
event_counts_df.to_csv(CSV_EVENTS, index=False)

with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=== Figure 2: Statewide generation response to climate extremes (California) ===\n\n")
    f.write(f"NetCDF: {NC_PATH}\n")
    f.write(f"Generation CSV: {GEN_PATH}\n")
    f.write(f"Smoothing: {SMOOTH_MONTHS}-month centered rolling mean\n")
    f.write("Drivers: statewide area-weighted STI/SAPEI/SCDHI (cos(lat) weights), then z-scored.\n")
    f.write(
        f"Event thresholds (statewide signal): STI ≥ {P_HEAT:.2f}q ; SAPEI ≤ {P_DROUGHT:.2f}q ; SCDHI ≤ {P_COMP:.2f}q\n")
    f.write(f"Normal months: SCDHI ≥ {NORMAL_Q:.2f}q\n\n")

    f.write("---- Saved figures (each has 2 labeled subplots) ----\n")
    f.write(f"{FIG_A}  (A,B)\n")
    f.write(f"{FIG_B}  (C,D)\n")
    f.write(f"{FIG_C}  (E,F)\n")
    f.write(f"{FIG_D}  (G,H)\n")
    f.write(f"{FIG_E}  (I,J)\n")
    f.write(f"{FIG_F}  (K,L)\n\n")

    f.write("---- Event counts ----\n")
    f.write(event_counts_df.to_string(index=False))
    f.write("\n\n")
