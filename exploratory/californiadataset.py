import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ds = xr.open_dataset("../dataset/california_monthly.nc")
elec_gen = pd.read_csv("../dataset/california_electricity_generation_monthly.csv")
elec_dem = pd.read_csv("../dataset/california_electricity_demand_monthly.csv")
elec_iso = pd.read_csv("../dataset/california_electricity_caiso_monthly.csv")
emission = pd.read_csv("../dataset/california_emission_monthly.csv")

state = ds[['scdhi', 'sapei_scdhi', 'sti_scdhi', 'tmin', 'tmax', 'tmean', 'prcp']].mean(dim=['lat', 'lon']).to_dataframe().reset_index(drop=True)

time = pd.date_range(start='1951-01-01', end='2025-12-01', freq='MS')

state['year'] = time.year
state['month'] = time.month

climate = state.iloc[0:896]

elec_gen.columns = [col.replace(' ', '_').lower() for col in elec_gen.columns]

gen_pivot = elec_gen.pivot_table(
    index=['year', 'month'],
    columns='energy_source',
    values='generation_(megawatthours)',
    aggfunc='sum'
).reset_index().fillna(np.nan)

gen_pivot.columns = gen_pivot.columns.str.replace(' ', '_').str.lower()
gen_pivot = gen_pivot.rename(columns={
    'hydroelectric_conventional': 'hydro_gen',
    'natural_gas': 'gas_gen',
    'solar_thermal_and_photovoltaic': 'solar_gen',
    'wind': 'wind_gen',
    'coal': 'coal_gen'
})

dem_cols = ['year', 'month', 'total_sales', 'residential_sales', 'commercial_sales']
elec_dem = elec_dem[dem_cols].copy()

emission = emission[['year', 'month', 'co2Mass', 'noxMass', 'so2Mass']].rename(
    columns={'co2Mass': 'co2_emissions', 'noxMass': 'nox_emissions', 'so2Mass': 'so2_emissions'}
)

climate = climate.rename(columns={
    'scdhi': 'scdhi',
    'sapei_scdhi': 'drought',
    'sti_scdhi': 'heatwave'
})

elec_iso.drop(['Day ahead forecast', 'Hour ahead forecast'], axis=1, errors='ignore', inplace=True)

elec_iso = elec_iso.rename(columns={
    'Current demand': 'total_supply_iso',
    'Avg_Demand_MW': 'avg_demand_iso',
    'Solar': 'solar_iso',
    'Wind': 'wind_iso',
    'Natural gas': 'gas_iso',
    'Large hydro': 'large_hydro_iso',
    'Small hydro': 'small_hydro_iso',
    'Coal': 'coal_iso',
    'Nuclear': 'nuclear_iso',
    'Geothermal': 'geothermal_iso',
    'Biomass': 'biomass_iso',
    'Biogas': 'biogas_iso',
    'Batteries': 'batteries_iso',
    'Imports': 'cimports_iso',
    'Other': 'other_iso'
})

caiso_cols = ['year', 'month'] + [col for col in elec_iso.columns if col.endswith('_iso')]
elec_iso = elec_iso[caiso_cols]

master = pd.merge(climate, elec_dem, on=['year', 'month'], how='outer')
master = pd.merge(master, gen_pivot, on=['year', 'month'], how='outer')
master = pd.merge(master, emission, on=['year', 'month'], how='outer')
master = pd.merge(master, elec_iso, on=['year', 'month'], how='outer')

master['date'] = pd.to_datetime(master['year'].astype(str) + '-' + master['month'].astype(str))
master = master.sort_values('date').reset_index(drop=True)
gen_cols = [col for col in master.columns if '_gen' in col]
dem_cols = [col for col in master.columns if '_sales' in col]
emis_cols = [col for col in master.columns if '_emissions' in col]
iso_cols = [col for col in master.columns if '_iso' in col]

master[gen_cols + dem_cols + emis_cols + iso_cols] = master[gen_cols + dem_cols + emis_cols + iso_cols].fillna(np.nan)
master = master[['year', 'month'] + [col for col in master.columns if col not in ['year', 'month']]]
master.drop("date", axis=1).to_csv('../dataset/california_all.csv', index=False)
