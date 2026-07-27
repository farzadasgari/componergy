import pandas as pd

df = pd.read_csv('../dataset/california_electricity_caiso_raw.csv')

df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time')

num_cols = df.select_dtypes("number").columns

monthly_avg = df[num_cols].resample("MS").mean().reset_index()
monthly_avg["year"] = monthly_avg["Time"].dt.year
monthly_avg["month"] = monthly_avg["Time"].dt.month

df_mwh = df[num_cols] * (5/60)

monthly_mwh = df_mwh.resample("MS").sum().reset_index()
monthly_mwh["year"] = monthly_mwh["Time"].dt.year
monthly_mwh["month"] = monthly_mwh["Time"].dt.month
monthly_mwh['Avg_Demand_MW'] = df['Current demand'].resample('ME').mean()

monthly_mwh.to_csv('../dataset/california_electricity_caiso_monthly.csv'index=False)
