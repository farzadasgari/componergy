import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import pgeocode
import warnings

warnings.filterwarnings("ignore")

ELEC_CSV_PATH = "../dataset/electricity/electricity_by_county.csv"
CLIMATE_NC_PATH = "../dataset/california_regional_timeseries.nc"
REGIONS_URL = "https://api.cal-adapt.org/api/climregions/"
RESULTS_CSV_PATH = "../dataset/impact_analysis_results_per_region.csv"

# -----------------------------
# 1) ZIP -> Climate Region map
# -----------------------------
def get_zipcode_region_map(unique_zipcodes):
    nomi = pgeocode.Nominatim("us")
    rows = []
    for z in unique_zipcodes:
        try:
            r = nomi.query_postal_code(str(int(z)))
            if pd.notna(r.latitude) and pd.notna(r.longitude):
                rows.append({"ZipCode": z, "Latitude": r.latitude, "Longitude": r.longitude})
        except Exception:
            pass

    stations_df = pd.DataFrame(rows)
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df["Longitude"], stations_df["Latitude"]),
        crs="EPSG:4326",
    )

    regions_gdf = gpd.read_file(REGIONS_URL)
    if regions_gdf.crs is None:
        regions_gdf = regions_gdf.set_crs("EPSG:4326")
    regions_gdf = regions_gdf.to_crs("EPSG:4326")

    # pick a reasonable name column
    candidate_cols = ["name", "title", "label", "region", "abbr", "code"]
    region_col = next((c for c in candidate_cols if c in regions_gdf.columns),
                      [c for c in regions_gdf.columns if c != "geometry"][0])
    regions_gdf["ClimateRegion"] = regions_gdf[region_col].astype("string").str.strip()

    # Use intersects (more forgiving than within for boundary points)
    joined = gpd.sjoin(stations_gdf, regions_gdf[["ClimateRegion", "geometry"]],
                       how="left", predicate="intersects")

    # IMPORTANT: do NOT default missing to South Coast; drop them later instead
    return joined.set_index("ZipCode")["ClimateRegion"].to_dict()


def load_electricity_data():
    df = pd.read_csv(ELEC_CSV_PATH)

    # optional: drop obvious junk
    df = df[df["TotalkWh"].fillna(0) > 0].copy()

    zip_map = get_zipcode_region_map(df["ZipCode"].dropna().unique())
    df["Region"] = df["ZipCode"].map(zip_map)
    df = df.dropna(subset=["Region"])

    elec_agg = (
        df.groupby(["Region", "Year", "Month", "CustomerClassClean"], as_index=False)["TotalkWh"]
          .sum()
    )
    return elec_agg


# --------------------------------------
# 2) Climate flags (MONTH-SPECIFIC q's)
# --------------------------------------
def analyze_climate_extremes_monthly(
    ds,
    baseline_years=(1951, 2025),
    heat_var="sti_scdhi",
    drought_var="sapei_scdhi",
    heat_q=0.90,
    drought_q=0.10,
):
    y0, y1 = baseline_years
    base = ds.sel(year=slice(y0, y1))

    # month-specific thresholds across years (region x month)
    heat_thr = base[heat_var].quantile(heat_q, dim="year").reset_coords(drop=True)
    drought_thr = base[drought_var].quantile(drought_q, dim="year").reset_coords(drop=True)

    # flags (region x year x month)
    is_heat = (ds[heat_var] > heat_thr).reset_coords(drop=True)
    is_drought = (ds[drought_var] < drought_thr).reset_coords(drop=True)

    # compound MUST be intersection for clean comparison
    is_compound = (is_heat & is_drought).reset_coords(drop=True)

    flags = xr.Dataset(
        {
            "Is_Heatwave": is_heat.astype(bool),
            "Is_Drought": is_drought.astype(bool),
            "Is_Compound": is_compound.astype(bool),
        }
    )

    df_flags = (
        flags.to_dataframe()
        .reset_index()
        .rename(columns={"region": "Region", "year": "Year", "month": "Month"})
    )

    return df_flags[["Region", "Year", "Month", "Is_Heatwave", "Is_Drought", "Is_Compound"]]



# ----------------------------------------------------
# 3) Event comparison using seasonally-adjusted anomaly
# ----------------------------------------------------
def summarize_impacts_monthly_anomaly(merged):
    out = []

    for (region, cust_class), g in merged.groupby(["Region", "CustomerClassClean"]):
        g = g.copy()

        # mutually exclusive buckets
        g["Bucket"] = np.select(
            [
                (~g["Is_Heatwave"]) & (~g["Is_Drought"]),
                (g["Is_Heatwave"]) & (~g["Is_Drought"]),
                (~g["Is_Heatwave"]) & (g["Is_Drought"]),
                (g["Is_Heatwave"]) & (g["Is_Drought"]),
            ],
            ["Normal", "Heat_only", "Drought_only", "Compound"],
            default="Other",
        )

        # month-of-year baseline using NORMAL months only (best practice)
        normal = g[g["Bucket"] == "Normal"]
        base_moy = normal.groupby("Month")["TotalkWh"].median()

        # fallback if a month has no normal observations
        fallback = g.groupby("Month")["TotalkWh"].median()
        base_moy = base_moy.combine_first(fallback).replace(0, np.nan)

        g["Baseline_moy"] = g["Month"].map(base_moy)
        g["AnomPct"] = 100.0 * (g["TotalkWh"] - g["Baseline_moy"]) / g["Baseline_moy"]

        for b, gb in g.groupby("Bucket"):
            out.append({
                "Region": region,
                "Class": cust_class,
                "Bucket": b,
                "Count_Months": int(len(gb)),
                "Mean_kWh": float(gb["TotalkWh"].mean()),
                "Mean_AnomPct": float(gb["AnomPct"].mean()),
                "Median_AnomPct": float(gb["AnomPct"].median()),
            })

    return pd.DataFrame(out)


import numpy as np
import pandas as pd

def summarize_impacts_two_cases(merged, ignore_zero_for_baseline=True):
    """
    Returns results for:
      - Pure: Heat_only, Drought_only, Compound (intersection), Normal
      - Overlapping: Heatwave (incl compound), Drought (incl compound), Compound, Normal
    Includes mean + median of kWh and anomaly% vs MOY baselines (mean+median baselines).
    """

    results = []

    def build_moy_baselines(g):
        # Use normal months if possible
        normal = g[(~g["Is_Heatwave"]) & (~g["Is_Drought"])].copy()
        allm = g.copy()

        if ignore_zero_for_baseline:
            normal["TotalkWh"] = normal["TotalkWh"].replace(0, np.nan)
            allm["TotalkWh"] = allm["TotalkWh"].replace(0, np.nan)

        base_mean = normal.groupby("Month")["TotalkWh"].mean()
        base_median = normal.groupby("Month")["TotalkWh"].median()

        # fallback if normal is missing for a month
        fb_mean = allm.groupby("Month")["TotalkWh"].mean()
        fb_median = allm.groupby("Month")["TotalkWh"].median()

        base_mean = base_mean.combine_first(fb_mean)
        base_median = base_median.combine_first(fb_median)

        # final fallback (global)
        global_mean = allm["TotalkWh"].mean()
        global_median = allm["TotalkWh"].median()

        base_mean = base_mean.fillna(global_mean)
        base_median = base_median.fillna(global_median)

        # avoid divide-by-zero
        base_mean = base_mean.replace(0, np.nan)
        base_median = base_median.replace(0, np.nan)

        return base_mean, base_median

    def add_result(region, cust_class, case, event, df_event):
        n = int(len(df_event))
        if n == 0:
            results.append({
                "Region": region, "Class": cust_class, "Case": case, "Event": event,
                "Count_Months": 0,
                "Mean_kWh": np.nan, "Median_kWh": np.nan,
                "Mean_BaselineMean": np.nan, "Mean_BaselineMedian": np.nan,
                "Mean_AnomPct_vsMeanBase": np.nan, "Median_AnomPct_vsMeanBase": np.nan,
                "Mean_AnomPct_vsMedianBase": np.nan, "Median_AnomPct_vsMedianBase": np.nan,
            })
            return

        results.append({
            "Region": region, "Class": cust_class, "Case": case, "Event": event,
            "Count_Months": n,
            "Mean_kWh": float(df_event["TotalkWh"].mean()),
            "Median_kWh": float(df_event["TotalkWh"].median()),
            "Mean_BaselineMean": float(df_event["BaselineMean_moy"].mean()),
            "Mean_BaselineMedian": float(df_event["BaselineMedian_moy"].mean()),
            "Mean_AnomPct_vsMeanBase": float(df_event["AnomPct_vsMeanBase"].mean()),
            "Median_AnomPct_vsMeanBase": float(df_event["AnomPct_vsMeanBase"].median()),
            "Mean_AnomPct_vsMedianBase": float(df_event["AnomPct_vsMedianBase"].mean()),
            "Median_AnomPct_vsMedianBase": float(df_event["AnomPct_vsMedianBase"].median()),
        })

    for (region, cust_class), g in merged.groupby(["Region", "CustomerClassClean"]):
        g = g.copy()

        # month-of-year baselines
        base_mean_moy, base_median_moy = build_moy_baselines(g)
        g["BaselineMean_moy"] = g["Month"].map(base_mean_moy)
        g["BaselineMedian_moy"] = g["Month"].map(base_median_moy)

        # anomaly % vs baselines
        g["AnomPct_vsMeanBase"] = 100.0 * (g["TotalkWh"] - g["BaselineMean_moy"]) / g["BaselineMean_moy"]
        g["AnomPct_vsMedianBase"] = 100.0 * (g["TotalkWh"] - g["BaselineMedian_moy"]) / g["BaselineMedian_moy"]

        # baseline (normal) subset
        normal = g[(~g["Is_Heatwave"]) & (~g["Is_Drought"])]

        # ----------------
        # Case A: PURE
        # ----------------
        pure_hw = g[(g["Is_Heatwave"]) & (~g["Is_Drought"])]
        pure_dr = g[(g["Is_Drought"]) & (~g["Is_Heatwave"])]
        pure_comp = g[(g["Is_Heatwave"]) & (g["Is_Drought"])]

        add_result(region, cust_class, "Pure", "Normal", normal)
        add_result(region, cust_class, "Pure", "Heatwave_only", pure_hw)
        add_result(region, cust_class, "Pure", "Drought_only", pure_dr)
        add_result(region, cust_class, "Pure", "Compound", pure_comp)

        # ----------------------
        # Case B: OVERLAPPING
        # ----------------------
        ov_hw = g[g["Is_Heatwave"]]     # includes compound
        ov_dr = g[g["Is_Drought"]]      # includes compound
        ov_comp = pure_comp             # same intersection definition

        add_result(region, cust_class, "Overlapping", "Normal", normal)
        add_result(region, cust_class, "Overlapping", "Heatwave", ov_hw)
        add_result(region, cust_class, "Overlapping", "Drought", ov_dr)
        add_result(region, cust_class, "Overlapping", "Compound", ov_comp)

    return pd.DataFrame(results)


def main():
    elec_df = load_electricity_data()
    ds = xr.open_dataset(CLIMATE_NC_PATH)

    climate_flags = analyze_climate_extremes_monthly(
        ds,
        baseline_years=(1951, 2025),
        heat_var="sti_scdhi",       # consider trying sti_sdhi if that’s your pure-heat index
        drought_var="sapei_scdhi",  # consider spei_sdhi/spi_sdhi depending on your drought definition
        heat_q=0.90,
        drought_q=0.50
    )

    merged = pd.merge(elec_df, climate_flags, on=["Region", "Year", "Month"], how="inner")

    results_df = summarize_impacts_two_cases(merged, ignore_zero_for_baseline=True)
    event_order = {
        "Normal": 0,
        "Heatwave_only": 1,
        "Drought_only": 2,
        "Compound": 3,
        "Heatwave": 1,
        "Drought": 2,
    }

    results_df["EventOrder"] = results_df["Event"].map(event_order).fillna(99).astype(int)

    print(
        results_df
        .sort_values(["Region", "Class", "Case", "EventOrder", "Event"])
        .drop(columns=["EventOrder"])
        .head(40)
    )
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    # Example: inspect one region/class
    mask = (results_df["Region"] == "South Coast") & (results_df["Class"] == "Residential")
    print(results_df[mask].sort_values(["Case", "Event"]))

    # quick view
    # print(results_df.sort_values(["Region", "Class", "Bucket"]).head(40))
    print(f"\nSaved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
