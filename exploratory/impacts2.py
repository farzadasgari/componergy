import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

CSV_PATH = Path("../dataset/impact_analysis_results_per_region.csv")
OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_DIR = OUT_DIR / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

value_col = "Mean_AnomPct_vsMeanBase"

colors = {
    "Normal": "#666666",
    "Heatwave_only": "#d73027",
    "Heatwave": "#d73027",
    "Drought_only": "#4575b4",
    "Drought": "#4575b4",
    "Compound": "#7b3294",
}

df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)].copy()
df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0).clip(-100, 100)


def drop_deserts(frame):
    return frame[~frame["Region"].astype(str).str.upper().str.contains("MOJAVE|SONORA", regex=True)].copy()


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, linewidth=1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def vertical_colored_boxplot(case, events, fname, title):
    d = df[(df["Case"] == case) & (df["Event"].isin(events))].copy()
    data = [d.loc[d["Event"] == e, value_col].dropna().to_numpy()
            for e in events]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bp = ax.boxplot(
        data,
        labels=events,
        patch_artist=True,
        showfliers=True,
        notch=False,
        whis=1.5,
        widths=0.6,
    )
    for patch, e in zip(bp["boxes"], events):
        patch.set_facecolor(colors.get(e, "#aaaaaa"))
        patch.set_alpha(0.35)
        patch.set_linewidth(1.2)
    for k in ["whiskers", "caps", "medians"]:
        for artist in bp.get(k, []):
            artist.set_linewidth(1.2)
    for fl in bp.get("fliers", []):
        fl.set_markersize(3)
        fl.set_alpha(0.35)
    ax.set_ylabel("Mean anomaly vs MOY mean baseline (%)")
    ax.set_ylim(-10, 110)
    style_axes(ax)
    # ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=300)
    fig.savefig((OUT_DIR / fname).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


vertical_colored_boxplot("Pure", ["Normal", "Heatwave_only", "Drought_only",
                         "Compound"], "figA_box_pure_vertical.png", "Pure definitions")
vertical_colored_boxplot("Overlapping", ["Normal", "Heatwave", "Drought", "Compound"],
                         "figB_box_overlapping_vertical.png", "Overlapping definitions")


def region_grid_bars(case, events, fname, title):
    d = drop_deserts(df[(df["Case"] == case) & (
        df["Event"].isin(events))].copy())
    regions = sorted(d["Region"].dropna().unique().tolist())

    preferred_class_order = ["Residential",
                             "Commercial", "Industrial", "Agricultural"]
    classes_all = [
        c for c in preferred_class_order if c in d["Class"].dropna().unique()]
    classes_rest = sorted(
        [c for c in d["Class"].dropna().unique() if c not in classes_all])
    classes = classes_all + classes_rest

    n = len(regions)
    ncols = 3 if n >= 7 else 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols * 5.0, nrows * 3.4), sharey=True)
    axes = np.array(axes).reshape(-1)

    width = 0.22
    x = np.arange(len(classes))

    for i, region in enumerate(regions):
        ax = axes[i]
        sub = d[d["Region"] == region].copy()
        pivot = sub.pivot_table(index="Class", columns="Event",
                                values=value_col, aggfunc="mean").reindex(classes).fillna(0)
        for j, ev in enumerate(events):
            vals = pivot[ev].to_numpy() if ev in pivot.columns else np.full(
                len(classes), np.nan)
            ax.bar(
                x + (j - (len(events) - 1) / 2) * width,
                vals,
                width=width,
                label=ev,
                color=colors.get(ev, None),
                alpha=0.9,
            )
        ax.set_xticks(x, labels=classes, rotation=25, ha="right")
        ax.set_title(region, fontsize=11)
        ax.set_ylim(-10, 110)
        style_axes(ax)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    handles = [plt.Rectangle(
        (0, 0), 1, 1, color=colors.get(e, "#aaaaaa")) for e in events]
    fig.legend(handles, events, loc="lower center", ncols=len(
        events), frameon=False, bbox_to_anchor=(0.5, -0.01))
    # fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    fig.savefig((OUT_DIR / fname).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


region_grid_bars("Pure", ["Heatwave_only", "Drought_only", "Compound"],
                 "figC_grid_regionbars_pure.png", "Pure: mean anomaly (%) by class within each region")
region_grid_bars("Overlapping", ["Heatwave", "Drought", "Compound"],
                 "figD_grid_regionbars_overlapping.png", "Overlapping: mean anomaly (%) by class within each region")


def make_summary_table(frame, group_cols):
    t = (
        frame.groupby(group_cols)[value_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            sd="std",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    for c in ["mean", "median", "sd", "p25", "p75"]:
        t[c] = t[c].round(1)
    return t


def export_tables(case, events, prefix, include_normal=False):
    evs = (["Normal"] + events) if include_normal else events
    d = drop_deserts(df[(df["Case"] == case) & (df["Event"].isin(evs))].copy())

    tbl_event_class = make_summary_table(
        d, ["Event", "Class"]).sort_values(["Event", "Class"])
    tbl_event_class.to_csv(
        TABLE_DIR / f"{prefix}_event_by_class.csv", index=False)
    tbl_event_class.to_excel(
        TABLE_DIR / f"{prefix}_event_by_class.xlsx", index=False)
    (TABLE_DIR / f"{prefix}_event_by_class.tex").write_text(
        tbl_event_class.to_latex(
            index=False,
            float_format="%.1f",
            caption=f"Mean anomaly (%) vs MOY mean baseline by event and sector class ({case}).",
            label=f"tab:{prefix}_event_by_class",
            escape=True,
        )
    )

    tbl_event_region = make_summary_table(
        d, ["Event", "Region"]).sort_values(["Event", "Region"])
    tbl_event_region.to_csv(
        TABLE_DIR / f"{prefix}_event_by_region.csv", index=False)
    tbl_event_region.to_excel(
        TABLE_DIR / f"{prefix}_event_by_region.xlsx", index=False)

    tbl_counts = (
        d.groupby(["Event", "Class", "Region"])[value_col]
        .count()
        .reset_index(name="n_obs")
        .sort_values(["Event", "Class", "Region"])
    )
    tbl_counts.to_csv(TABLE_DIR / f"{prefix}_sample_sizes.csv", index=False)
    tbl_counts.to_excel(TABLE_DIR / f"{prefix}_sample_sizes.xlsx", index=False)

    with pd.ExcelWriter(TABLE_DIR / f"{prefix}_region_by_class_per_event.xlsx", engine="openpyxl") as xw:
        for ev in events:
            pv = (
                d[d["Event"] == ev]
                .pivot_table(index="Region", columns="Class", values=value_col, aggfunc="mean")
                .sort_index()
                .round(1)
            )
            pv.to_excel(xw, sheet_name=ev[:31])


export_tables("Pure", ["Heatwave_only", "Drought_only",
              "Compound"], "pure", include_normal=True)
export_tables("Overlapping", ["Heatwave", "Drought",
              "Compound"], "overlapping", include_normal=True)


def heatmap_region_class(case, events, fname_prefix, title_prefix):
    d = drop_deserts(df[(df["Case"] == case) & (
        df["Event"].isin(events))].copy())
    preferred_class_order = ["Residential",
                             "Commercial", "Industrial", "Agricultural"]
    classes_all = [
        c for c in preferred_class_order if c in d["Class"].dropna().unique()]
    classes_rest = sorted(
        [c for c in d["Class"].dropna().unique() if c not in classes_all])
    classes = classes_all + classes_rest
    regions = sorted(d["Region"].dropna().unique().tolist())

    for ev in events:
        sub = d[d["Event"] == ev].copy()
        pv = sub.pivot_table(index="Region", columns="Class", values=value_col,
                             aggfunc="mean").reindex(index=regions, columns=classes)
        mat = pv.to_numpy()

        fig_w = max(3.8, 0.35 * len(classes) + 2.2)
        fig_h = max(3.0, 0.22 * len(regions) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat, aspect="auto", vmin=-10, vmax=100, cmap="RdBu_r")
        ax.set_xticks(np.arange(len(classes)),
                      labels=classes, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(regions)), labels=regions)
        # ax.set_title(f"{title_prefix}: {ev}")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label("Mean anomaly (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{fname_prefix}_{ev}.png",
                    dpi=300, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{fname_prefix}_{ev}.pdf", bbox_inches="tight")
        plt.close(fig)


heatmap_region_class("Pure", ["Heatwave_only", "Drought_only", "Compound"],
                     "figE_heatmap_pure", "Pure heatmap (region × class)")
heatmap_region_class("Overlapping", ["Heatwave", "Drought", "Compound"],
                     "figF_heatmap_overlapping", "Overlapping heatmap (region × class)")


def forest_region_effect(case, events, fname, title):
    d = drop_deserts(df[(df["Case"] == case) & (
        df["Event"].isin(events))].copy())
    regions = sorted(d["Region"].dropna().unique().tolist())
    offsets = np.linspace(-0.18, 0.18, num=len(events)
                          ) if len(events) > 1 else np.array([0.0])
    y = np.arange(len(regions))

    fig_h = max(3.2, 0.22 * len(regions) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    for ev, off in zip(events, offsets):
        sub = d[d["Event"] == ev]
        g = sub.groupby("Region")[value_col]
        mean = g.mean().reindex(regions)
        n = g.count().reindex(regions)
        sd = g.std().reindex(regions)
        se = sd / np.sqrt(n.replace(0, np.nan))
        ci = 1.96 * se
        ax.errorbar(
            mean.to_numpy(),
            y + off,
            xerr=ci.to_numpy(),
            fmt="o",
            markersize=3.5,
            linewidth=1.0,
            capsize=2.2,
            label=ev,
            color=colors.get(ev, None),
        )

    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, labels=regions)
    ax.set_xlabel("Mean anomaly vs MOY mean baseline (%)")
    # ax.set_title(title)
    ax.set_xlim(-10, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    fig.savefig((OUT_DIR / fname).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


forest_region_effect("Pure", ["Heatwave_only", "Drought_only", "Compound"],
                     "figG_forest_pure.png", "Pure: region-level mean effect (±95% CI)")
forest_region_effect("Overlapping", ["Heatwave", "Drought", "Compound"],
                     "figH_forest_overlapping.png", "Overlapping: region-level mean effect (±95% CI)")

from matplotlib.lines import Line2D

def region_grid_bars(case, events, fname, title):
    d = drop_deserts(df[(df["Case"] == case) & (df["Event"].isin(events))].copy())
    regions = sorted(d["Region"].dropna().unique().tolist())

    preferred_class_order = ["Residential", "Commercial", "Industrial", "Agricultural"]
    classes_all = [c for c in preferred_class_order if c in d["Class"].dropna().unique()]
    classes_rest = sorted([c for c in d["Class"].dropna().unique() if c not in classes_all])
    classes = classes_all + classes_rest

    n = len(regions)
    ncols = 3 if n >= 7 else 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 3.4), sharey=True)
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(classes))

    for i, region in enumerate(regions):
        ax = axes[i]
        sub = d[d["Region"] == region].copy()
        pivot = sub.pivot_table(index="Class", columns="Event", values=value_col, aggfunc="mean").reindex(classes)

        for ev in events:
            vals = pivot[ev].to_numpy() if ev in pivot.columns else np.full(len(classes), np.nan)
            ax.plot(
                x,
                vals,
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                color=colors.get(ev, None),
                alpha=0.95,
                solid_capstyle="round",
            )

        ax.set_xticks(x, labels=classes, rotation=25, ha="right")
        ax.set_ylim(-10, 110)
        style_axes(ax)
        ax.set_title(region, fontsize=10, pad=6)
        
    for k in range(n, len(axes)):
        axes[k].axis("off")

    handles = [
        Line2D([0], [0], color=colors.get(e, "#aaaaaa"), marker="o", linewidth=1.4, markersize=4)
        for e in events
    ]
    fig.legend(handles, events, loc="lower center", ncols=len(events), frameon=False, bbox_to_anchor=(0.5, -0.01))
    # fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=300, bbox_inches="tight")
    fig.savefig((OUT_DIR / fname).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

region_grid_bars("Pure", ["Heatwave_only", "Drought_only", "Compound"], "figI_grid_regionbars_pure.png", "Pure: mean anomaly (%) by class within each region")
region_grid_bars("Overlapping", ["Heatwave", "Drought", "Compound"], "figJ_grid_regionbars_overlapping.png", "Overlapping: mean anomaly (%) by class within each region")
