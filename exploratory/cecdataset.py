import pandas as pd
import numpy as np

INPUT_FILE = "../dataset/2026-01-14 Farzad Asgari - attach - County-Level Electricity Consumption 2008-2024.xlsx"
OUTPUT_FILE = "../dataset/california_county_electricity_consumption_2008_2024_clean.csv"
SHEET_NAME = "Electricity Consumption"

KEY_COLS = ['Year', 'Month', 'CountyNum', 'CountyName', 'Sector']
VALUE_COL = 'Consumption (MWh)'

df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, dtype={'Year': int, 'Month': int})
df = df[['Year', 'Month', 'CountyNum', 'CountyName', 'Sector', VALUE_COL]].copy()

negatives = df[VALUE_COL] < 0
if negatives.any():
    print(f"Found {negatives.sum()} negative consumption values "
          f"({negatives.mean():.2%} of rows)")

county_agg = (
    df.groupby(['Year', 'Month', 'CountyNum', 'CountyName', 'Sector'], as_index=False)
      [VALUE_COL]
      .sum()
      .rename(columns={VALUE_COL: 'Consumption_MWh'})
)

wide = county_agg.pivot_table(
    index=['Year', 'Month', 'CountyNum', 'CountyName'],
    columns='Sector',
    values='Consumption_MWh',
    aggfunc='sum'
).reset_index()

wide = wide.fillna(0)

sector_columns = [c for c in wide.columns if c not in ['Year', 'Month', 'CountyNum', 'CountyName']]
wide['Total_MWh'] = wide[sector_columns].sum(axis=1)
county_agg.to_csv(OUTPUT_FILE, index=False)
wide.to_csv(OUTPUT_FILE.replace('.csv', '_wide.csv'), index=False)

print("\nBasic summary:")
print(county_agg.groupby('Sector')['Consumption_MWh'].agg(['sum', 'mean', 'count']).round(1))
