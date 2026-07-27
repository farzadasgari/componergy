import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import warnings
from datetime import datetime
from pathlib import Path
import re

warnings.filterwarnings("ignore")

sns.set_style("white")
plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 18

df = pd.read_csv("../dataset/california_all.csv")
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values('date').reset_index(drop=True)

df['has_eia_gen'] = df['hydro_gen'].notna() & df['gas_gen'].notna()
df['has_iso'] = df['total_supply_iso'].notna()

df_eia = df[df['has_eia_gen']].copy()
df_iso = df[df['has_iso']].copy()

df_eia['fossil_gen'] = df_eia[['gas_gen', 'coal_gen', 'petroleum']].sum(axis=1)
df_eia['vre_gen'] = df_eia['solar_gen'] + df_eia['wind_gen']
df_eia['renewable_gen'] = (
    df_eia['vre_gen']
    + df_eia['geothermal']
    + df_eia.get('other_biomass', 0)
    + df_eia.get('wood_and_wood_derived_fuels', 0)
)
df_eia['clean_gen'] = df_eia['renewable_gen'] + df_eia['hydro_gen'] + df_eia.get('nuclear', 0)

df_iso['fossil_iso'] = df_iso['gas_iso'] + df_iso['coal_iso']
df_iso['hydro_iso_total'] = df_iso['large_hydro_iso'] + df_iso['small_hydro_iso']
df_iso['vre_iso'] = df_iso['solar_iso'] + df_iso['wind_iso']
df_iso['renewable_iso'] = df_iso['vre_iso'] + df_iso['geothermal_iso'] + df_iso['biomass_iso'] + df_iso['biogas_iso']
df_iso['firm_lowcarbon_iso'] = df_iso['nuclear_iso'] + df_iso['geothermal_iso'] + df_iso['biomass_iso'] + df_iso['biogas_iso']

compound_thresh, drought_thresh, heat_thresh = -0.8, -0.5, 0.5

CASE_SPECS = {
    "case1_pure_only": {
        "drought_mask": lambda d, h, c: d & (~h) & (~c),
        "heat_mask":    lambda d, h, c: h & (~d) & (~c),
        "label": "Case 1: Pure only (exclude other hazard + exclude compound)"
    },
    "case2_inclusive": {
        "drought_mask": lambda d, h, c: d,
        "heat_mask":    lambda d, h, c: h,
        "label": "Case 2: Inclusive (no exclusions)"
    },
    "case3_exclusive": {
        "drought_mask": lambda d, h, c: d & (~c),
        "heat_mask":    lambda d, h, c: h & (~c),
        "label": "Case 3: Exclusive (no compound)"
    }
}

order = ['Normal', 'Drought', 'Heatwave', 'CDHW']

def define_events_case(df_subset: pd.DataFrame, case_key: str):
    spec = CASE_SPECS[case_key]
    is_compound = df_subset['scdhi'] <= compound_thresh
    is_drought  = df_subset['drought'] <= drought_thresh
    is_heat     = df_subset['heatwave'] >= heat_thresh
    normal_mask   = (~is_compound) & (~is_drought) & (~is_heat)
    compound_mask = is_compound
    drought_mask = spec["drought_mask"](is_drought, is_heat, is_compound)
    heat_mask    = spec["heat_mask"](is_drought, is_heat, is_compound)
    return {
        "Normal":  df_subset[normal_mask].copy(),
        "Drought": df_subset[drought_mask].copy(),
        "Heatwave": df_subset[heat_mask].copy(),
        "CDHW":    df_subset[compound_mask].copy(),
    }

eia_gen_vars = {
    'Gas': 'gas_gen', 'Coal': 'coal_gen', 'Petroleum': 'petroleum',
    'Fossil Total': 'fossil_gen', 'Hydropower': 'hydro_gen',
    'Nuclear': 'nuclear', 'Solar': 'solar_gen', 'Wind': 'wind_gen',
    'Geothermal': 'geothermal', 'Biomass/Wood': 'other_biomass',
    'VRE Total': 'vre_gen', 'Renewable Total': 'renewable_gen',
    'Clean Total': 'clean_gen'
}

eia_emissions_vars = {
    'CO₂ Emissions': 'co2_emissions',
    'NOx Emissions': 'nox_emissions',
    'SO₂ Emissions': 'so2_emissions'
}

iso_gen_vars = {
    'Gas': 'gas_iso', 'Coal': 'coal_iso', 'Fossil Total': 'fossil_iso',
    'Large Hydro': 'large_hydro_iso', 'Small Hydro': 'small_hydro_iso',
    'Hydro Total': 'hydro_iso_total', 'Solar': 'solar_iso', 'Wind': 'wind_iso',
    'VRE Total': 'vre_iso', 'Nuclear': 'nuclear_iso', 'Geothermal': 'geothermal_iso',
    'Biomass': 'biomass_iso', 'Renewable Total': 'renewable_iso',
    'Firm Low-Carbon': 'firm_lowcarbon_iso', 'Imports': 'cimports_iso',
    'Batteries': 'batteries_iso', 'Demand (Supply)': 'total_supply_iso'
}

demand_vars = {
    'Total Sales': 'total_sales',
    'Residential Sales': 'residential_sales',
    'Commercial Sales': 'commercial_sales'
}

units_by_col = {
    'co2_emissions': 'tons',
    'nox_emissions': 'tons',
    'so2_emissions': 'tons'
}

def col_unit(col: str):
    return units_by_col.get(col, 'MWh')

def analyze_group(dataset_name, events_dict, var_dict, group_name, case_key, log_lines, results_rows):
    normal = events_dict['Normal']
    log_lines.append(f"\n=== {dataset_name} | {group_name} | {case_key} ===")
    print(f"\n=== {dataset_name} | {group_name} | {case_key} ===")

    for event_name in ['Drought', 'Heatwave', 'CDHW']:
        event_df = events_dict[event_name]
        n_event = len(event_df)
        n_normal = len(normal)
        log_lines.append(f"\n{event_name} vs Normal ({n_event} vs {n_normal} months)")
        print(f"\n{event_name} vs Normal ({n_event} vs {n_normal} months)")

        for display, col in var_dict.items():
            if col not in event_df.columns or col not in normal.columns:
                continue

            ev = event_df[col].dropna()
            no = normal[col].dropna()
            if len(ev) == 0 or len(no) == 0:
                continue

            mean_ev, mean_no = ev.mean(), no.mean()
            perc = (mean_ev - mean_no) / mean_no * 100 if mean_no != 0 else np.nan

            _, p = mannwhitneyu(ev, no, alternative='two-sided')
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            u = col_unit(col)
            line = f"{display}: {mean_ev:,.0f} {u} ({perc:+.1f}%) {sig}"
            print(line)
            log_lines.append(line)

            results_rows.append({
                'Case': case_key,
                'Dataset': dataset_name,
                'Group': group_name,
                'Event': event_name,
                'Variable': display,
                'Column': col,
                'Unit': u,
                'Mean Event': mean_ev,
                'Mean Normal': mean_no,
                'Percent Change': perc,
                'P-value': p,
                'Significance': sig,
                'N Event': n_event,
                'N Normal': n_normal
            })

palette = sns.color_palette("Set2", 4)

def make_boxplot_figure(events_dict, vars_cols, var_names, filename, ylabel_first):
    n_vars = len(vars_cols)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.6 * n_vars, 5.5), constrained_layout=True)
    if n_vars == 1:
        axes = [axes]

    for ax, col, name in zip(axes, vars_cols, var_names):
        data = []
        for label in order:
            if col not in events_dict[label].columns:
                continue
            vals = events_dict[label][col].dropna() / 1e6
            for v in vals:
                data.append({'Event': label, 'Value': v})
        plot_df = pd.DataFrame(data)

        sns.boxplot(x='Event', y='Value', data=plot_df, order=order, ax=ax,
                    palette=palette, linewidth=1.8, fliersize=3)
        sns.stripplot(x='Event', y='Value', data=plot_df, order=order, ax=ax,
                      color='black', alpha=0.4, size=2.5, jitter=True)

        ax.set_title(name, fontsize=14, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel_first if ax == axes[0] else '')
        ax.tick_params(axis='x', rotation=90)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def bootstrap_ci_median(x, n_boot=5000, ci=0.95, seed=123):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        meds.append(np.median(samp))
    meds = np.array(meds)
    alpha = (1 - ci) / 2
    lo = np.quantile(meds, alpha)
    hi = np.quantile(meds, 1 - alpha)
    return np.median(x), lo, hi

def make_stress_panel_caiso(events_dict, filename):
    vars_cols = ['hydro_iso_total', 'gas_iso', 'cimports_iso', 'total_supply_iso']
    var_names = ['Total Hydro', 'Gas', 'Imports', 'Demand']
    n_vars = len(vars_cols)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.8 * n_vars, 4.8), constrained_layout=True)
    if n_vars == 1:
        axes = [axes]

    for ax, col, title in zip(axes, vars_cols, var_names):
        meds, los, his = [], [], []
        for label in order:
            if col not in events_dict[label].columns:
                meds.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            vals = (events_dict[label][col].dropna() / 1e6).values
            m, lo, hi = bootstrap_ci_median(vals)
            meds.append(m); los.append(lo); his.append(hi)

        x = np.arange(len(order))
        yerr = np.vstack([np.array(meds) - np.array(los), np.array(his) - np.array(meds)])
        ax.errorbar(x, meds, yerr=yerr, fmt='o', capsize=4, linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_ylabel("Monthly (GWh)" if ax == axes[0] else "")
        ax.grid(True, axis='y', alpha=0.25)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()

def _safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\\/*?:\[\]]+", "", name)
    return name[:31] if len(name) > 31 else name

def build_explainable_tables(results_df: pd.DataFrame) -> dict:
    df = results_df.copy()

    df["Unit"] = df["Unit"].fillna("")

    def fmt_int(x):
        return "" if pd.isna(x) else f"{x:,.0f}"

    def fmt_pct(x):
        return "" if pd.isna(x) else f"{x:+.1f}%"

    df["Event Mean"]  = df.apply(lambda r: f"{fmt_int(r['Mean Event'])} {r['Unit']}".strip(), axis=1)
    df["Normal Mean"] = df.apply(lambda r: f"{fmt_int(r['Mean Normal'])} {r['Unit']}".strip(), axis=1)
    df["Δ%"] = df["Percent Change"].apply(fmt_pct)
    df["N (event vs normal)"] = df.apply(
        lambda r: f"{int(r['N Event'])} vs {int(r['N Normal'])}"
        if pd.notna(r["N Event"]) and pd.notna(r["N Normal"]) else "",
        axis=1
    )

    pretty_long = df[[
        "Case", "Dataset", "Group", "Event", "Variable",
        "Event Mean", "Normal Mean",
        "Δ%", "N (event vs normal)"
    ]].copy()

    summary_blocks = {}
    detail_blocks = {}

    for (dataset, group), sub in pretty_long.groupby(["Dataset", "Group"], sort=False):
        tmp = sub.copy()

        pivot = (tmp.pivot(index="Variable", columns="Event", values="Δ%")
                   .reindex(columns=["Drought", "Heatwave", "CDHW"], fill_value=""))
        pivot = pivot.reset_index()

        detail = (sub.sort_values(["Event", "Variable"])
                    .reset_index(drop=True))

        summary_blocks[(dataset, group)] = pivot
        detail_blocks[(dataset, group)] = detail

    return {
        "pretty_long": pretty_long,
        "summary_blocks": summary_blocks,
        "detail_blocks": detail_blocks
    }

def write_explainable_outputs(results_df: pd.DataFrame, case_key: str, out_dir: str = "../results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = build_explainable_tables(results_df)
    summary_blocks = built["summary_blocks"]
    detail_blocks = built["detail_blocks"]

    html_path = out_dir / f"event_impacts_{case_key}_tables.html"
    css = """
    <style>
      body { font-family: Cambria, Arial, sans-serif; margin: 24px; }
      h1 { margin-bottom: 6px; }
      .note { color: #333; margin-bottom: 18px; }
      h2 { margin-top: 26px; }
      h3 { margin-top: 18px; }
      table { border-collapse: collapse; width: 100%; margin: 10px 0 18px 0; }
      th, td { border: 1px solid #ddd; padding: 7px 9px; font-size: 13px; }
      th { background: #f4f4f4; text-align: left; }
      .small { font-size: 12px; color: #444; }
    </style>
    """

    parts = []
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append(css)
    parts.append(f"</head><body><h1>Explainable Tables — {case_key}</h1>")
    parts.append(
        "<div class='note'>"
        "<b>How to read:</b> Δ% is mean(Event) vs mean(Normal). "
        "N shows months (event vs normal)."
        "</div>"
    )

    for (dataset, group), pivot in summary_blocks.items():
        parts.append(f"<h2>{dataset} — {group}</h2>")
        parts.append("<h3>Summary (Δ% + significance)</h3>")
        parts.append(pivot.to_html(index=False, escape=False))

        parts.append("<h3>Details (means, Δ%, p, N)</h3>")
        detail = detail_blocks[(dataset, group)]
        parts.append(detail.to_html(index=False, escape=False))

    parts.append("</body></html>")

    html_path.write_text("\n".join(parts), encoding="utf-8")

    xlsx_path = out_dir / f"event_impacts_{case_key}_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for (dataset, group), pivot in summary_blocks.items():
            sheet = _safe_sheet_name(f"{dataset} - {group}")
            pivot.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

            detail = detail_blocks[(dataset, group)]
            startrow = len(pivot) + 3
            detail.to_excel(writer, sheet_name=sheet, index=False, startrow=startrow)

    return html_path, xlsx_path

for case_key, spec in CASE_SPECS.items():
    os.makedirs("../results", exist_ok=True)
    os.makedirs("../plots/results", exist_ok=True)

    log_lines = []
    log_lines.append("CALIFORNIA COMPOUND DROUGHT-HEATWAVE IMPACT ANALYSIS")
    # log_lines.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}")
    log_lines.append(f"Definition: {spec['label']}")
    log_lines.append(f"Thresholds: scdhi<= {compound_thresh}, drought<= {drought_thresh}, heatwave>= {heat_thresh}")
    log_lines.append("")

    results_rows = []

    events_eia = define_events_case(df_eia, case_key)
    events_iso = define_events_case(df_iso, case_key)

    analyze_group("EIA Long-Term", events_eia, eia_gen_vars, "Generation Sources", case_key, log_lines, results_rows)
    analyze_group("EIA Long-Term", events_eia, eia_emissions_vars, "Emissions", case_key, log_lines, results_rows)
    analyze_group("EIA Long-Term", events_eia, demand_vars, "Electricity Demand", case_key, log_lines, results_rows)

    analyze_group("CAISO Recent", events_iso, iso_gen_vars, "Generation & Supply", case_key, log_lines, results_rows)
    analyze_group("CAISO Recent", events_iso, demand_vars, "Electricity Demand", case_key, log_lines, results_rows)

    log_path = f"../results/event_impacts_{case_key}.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\nLog saved: {log_path}")

    results_df = pd.DataFrame(results_rows)
    csv_path = f"../results/event_impacts_{case_key}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
    '''
    make_boxplot_figure(
        events_eia,
        ['gas_gen', 'fossil_gen', 'hydro_gen', 'clean_gen'],
        ['Gas', 'Fossil Fuels', 'Hydropower', 'Clean Energy'],
        f"../plots/results/eia_impacts_boxplots_{case_key}.png",
        ylabel_first="Monthly Generation (GWh)"
    )

    make_boxplot_figure(
        events_iso,
        ['gas_iso', 'fossil_iso', 'hydro_iso_total', 'cimports_iso', 'total_supply_iso'],
        ['Gas', 'Fossil Fuels', 'Total Hydro', 'Imports', 'Demand'],
        f"../plots/results/caiso_impacts_boxplots_{case_key}.png",
        ylabel_first="Monthly Generation (GWh)"
    )

    make_stress_panel_caiso(
        events_iso,
        f"../plots/results/caiso_stress_panel_{case_key}.png"
    )

    print(f"Plots saved for {case_key}.\n")
    '''
    html_path, xlsx_path = write_explainable_outputs(results_df, case_key, out_dir="../results/impacts")
    print(f"Explainable tables saved:\n- {html_path}\n- {xlsx_path}")
