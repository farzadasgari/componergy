import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Cambria'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

df = pd.read_csv("../dataset/california_all.csv")

df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values('date').reset_index(drop=True)

df['has_eia_gen'] = df['hydro_gen'].notna() & df['gas_gen'].notna()
df['has_iso'] = df['total_supply_iso'].notna()

df_eia = df[df['has_eia_gen']].copy()
df_iso = df[df['has_iso']].copy()

print(f"EIA analysis period: {df_eia['date'].min()} to {df_eia['date'].max()} ({len(df_eia)} months)")
print(f"CAISO analysis period: {df_iso['date'].min()} to {df_iso['date'].max()} ({len(df_iso)} months)")

df_eia['fossil_gen'] = df_eia['gas_gen'] + df_eia['coal_gen'] + df_eia.get('petroleum', 0)
df_eia['clean_gen'] = (df_eia['hydro_gen'] + df_eia['solar_gen'] + df_eia['wind_gen'] +
                      df_eia.get('nuclear', 0) + df_eia.get('geothermal', 0) +
                      df_eia.get('other_biomass', 0) + df_eia.get('wood_and_wood_derived_fuels', 0))

df_iso['fossil_iso'] = df_iso['gas_iso'] + df_iso['coal_iso']
df_iso['clean_iso'] = (df_iso['large_hydro_iso'] + df_iso['small_hydro_iso'] +
                      df_iso['solar_iso'] + df_iso['wind_iso'] + df_iso['nuclear_iso'] +
                      df_iso['geothermal_iso'] + df_iso['biomass_iso'] + df_iso['biogas_iso'])

drought_thresh = -0.5
heat_thresh = 0.5
compound_thresh = -0.8

drought_only_eia = df_eia[(df_eia['drought'] <= drought_thresh)]
heat_only_eia = df_eia[(df_eia['heatwave'] >= heat_thresh)]
compound_eia = df_eia[(df_eia['scdhi'] <= compound_thresh)]
normal_eia = df_eia[(df_eia['scdhi'] > compound_thresh) & (df_eia['heatwave'] < heat_thresh) & (df_eia['drought'] > drought_thresh)]

drought_only_iso = df_iso[(df_iso['drought'] <= drought_thresh)]
heat_only_iso = df_iso[(df_iso['heatwave'] >= heat_thresh)]
compound_iso = df_iso[(df_iso['scdhi'] <= compound_thresh)]
normal_iso = df_iso[(df_iso['scdhi'] > compound_thresh) & (df_iso['heatwave'] < heat_thresh) & (df_iso['drought'] > drought_thresh)]

scdhi_severe_eia = df_eia[df_eia['scdhi'] <= compound_thresh]
scdhi_severe_iso = df_iso[df_iso['scdhi'] <= compound_thresh]

def print_event_summary(name, event_df, normal_df, vars_dict, dataset_name="EIA"):
    print(f"\n=== {name} vs Normal ({dataset_name}) ===")
    print(f"Months: {len(event_df)} (vs {len(normal_df)} normal)")
    log_lines = [f"{name} vs Normal ({dataset_name})", f"Months: {len(event_df)}"]
    
    for var_name, var_col in vars_dict.items():
        if var_col not in event_df.columns or var_col not in normal_df.columns:
            continue
        mean_event = event_df[var_col].mean()
        mean_normal = normal_df[var_col].mean()
        perc = (mean_event - mean_normal) / mean_normal * 100 if mean_normal != 0 else np.nan
        print(f"{var_name}: {mean_event:,.0f} MWh → {perc:+.1f}% vs normal ({mean_normal:,.0f} MWh)")
        log_lines.append(f"{var_name}: {mean_event:,.0f} MWh ({perc:+.1f}%)")
    
    return log_lines

eia_vars = {
    'Gas Generation': 'gas_gen',
    'Fossil Total': 'fossil_gen',
    'Hydro': 'hydro_gen',
    'Clean Total': 'clean_gen'
}

iso_vars = {
    'Gas': 'gas_iso',
    'Fossil Total': 'fossil_iso',
    'Large Hydro': 'large_hydro_iso',
    'Imports': 'cimports_iso',
    'Demand (Supply)': 'total_supply_iso'
}

log_output = []
log_output.append("CALIFORNIA EXTREME EVENTS IMPACT ANALYSIS")
log_output.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
log_output.append("")

log_output.extend(print_event_summary("Severe Drought Only", drought_only_eia, normal_eia, eia_vars, "EIA Long-Term"))
log_output.extend(print_event_summary("Severe Heatwave Only", heat_only_eia, normal_eia, eia_vars, "EIA Long-Term"))
log_output.extend(print_event_summary("Compound (Drought + Heatwave)", compound_eia, normal_eia, eia_vars, "EIA Long-Term"))

log_output.append("")

log_output.extend(print_event_summary("Severe Drought Only", drought_only_iso, normal_iso, iso_vars, "CAISO Recent"))
log_output.extend(print_event_summary("Severe Heatwave Only", heat_only_iso, normal_iso, iso_vars, "CAISO Recent"))
log_output.extend(print_event_summary("Compound (Drought + Heatwave)", compound_iso, normal_iso, iso_vars, "CAISO Recent"))

log_path = '../results/event_impacts_log.txt'
with open(log_path, 'w') as f:
    f.write('\n'.join(log_output))

print(f"\nNumerical summary saved to: {log_path}")

def get_event_spans(dates, condition):
    mask = condition.reset_index(drop=True)
    dates = dates.reset_index(drop=True)
    starts = dates[mask & ~mask.shift(1, fill_value=False)]
    ends = dates[mask & ~mask.shift(-1, fill_value=False)]
    spans = []
    for s, e in zip(starts, ends):
        spans.append((s, e + pd.offsets.MonthBegin(1)))
    return spans

def shade_events(ax, spans, color='brown', alpha=0.15, label=None):
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=0,
                   label=label if i == 0 else None)

def add_year_grid(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.35)
    ax.tick_params(axis='x', labelrotation=90)

fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(14, 12),
                         gridspec_kw={'height_ratios': [1, 2, 2, 2]}, constrained_layout=True)

ax_indices, ax_fossil, ax_clean, ax_gas = axes

drought_spans = get_event_spans(df_eia['date'], df_eia['drought'] < drought_thresh)
heat_spans = get_event_spans(df_eia['date'], df_eia['heatwave'] > heat_thresh)
compound_spans = get_event_spans(df_eia['date'], df_eia['scdhi'] > compound_thresh)

for ax in axes:
    shade_events(ax, drought_spans, 'brown', 0.15, 'Drought Only')
    shade_events(ax, heat_spans, 'red', 0.15, 'Heatwave Only')
    shade_events(ax, compound_spans, 'purple', 0.25, 'Compound CDHW')

ax_indices.plot(df_eia['date'], df_eia['drought'], color='brown', label='Drought')
ax_indices.plot(df_eia['date'], df_eia['heatwave'], color='red', label='Heatwave')
ax_indices.plot(df_eia['date'], df_eia['scdhi'], color='purple', label='SCDHI')
ax_indices.set_ylabel('Indices')
ax_indices.legend(fontsize=12)

ax_fossil.plot(df_eia['date'], df_eia['fossil_gen']/1e6, color='darkred')
ax_fossil.set_ylabel('Fossil (GWh)')

ax_clean.plot(df_eia['date'], df_eia['clean_gen']/1e6, color='darkgreen')
ax_clean.set_ylabel('Clean (GWh)')

ax_gas.plot(df_eia['date'], df_eia['gas_gen']/1e6, color='orange')
ax_gas.set_ylabel('Gas (GWh)')

add_year_grid(axes[-1])
axes[-1].set_xlim(pd.Timestamp('2000-07-01'), pd.Timestamp('2026-01-01'))
plt.savefig('../plots/results/full_events_eia_final.png', dpi=300)
plt.close()

fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(14, 14),
                         gridspec_kw={'height_ratios': [1, 2, 2, 2, 2]}, constrained_layout=True)

ax_indices, ax_imports, ax_fossil, ax_clean, ax_gas = axes

for ax in axes:
    shade_events(ax, drought_spans[-len(df_iso):], 'brown', label='Drought Only')
    shade_events(ax, heat_spans[-len(df_iso):], 'red', label='Heatwave Only')
    shade_events(ax, compound_spans[-len(df_iso):], 'purple', label='Compound CDHW')

ax_indices.plot(df_iso['date'], df_iso['drought'], color='brown')
ax_indices.plot(df_iso['date'], df_iso['heatwave'], color='red')
ax_indices.plot(df_iso['date'], df_iso['scdhi'], color='purple')
ax_indices.set_ylabel('Indices')
ax_indices.legend(fontsize=12)

ax_imports.plot(df_iso['date'], df_iso['cimports_iso']/1e6, color='gray')
ax_imports.set_ylabel('Imports (GWh)')

ax_fossil.plot(df_iso['date'], df_iso['fossil_iso']/1e6, color='darkred')
ax_fossil.set_ylabel('Fossil (GWh)')

ax_clean.plot(df_iso['date'], df_iso['clean_iso']/1e6, color='darkgreen')
ax_clean.set_ylabel('Clean (GWh)')

ax_gas.plot(df_iso['date'], df_iso['gas_iso']/1e6, color='orange')
ax_gas.set_ylabel('Gas (GWh)')

add_year_grid(axes[-1])
axes[-1].set_xlim(pd.Timestamp('2018-01-01'), pd.Timestamp('2026-01-01'))
plt.savefig('../plots/results/full_events_caiso_final.png', dpi=300)
plt.close()
