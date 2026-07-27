import json
import math
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import shape
from shapely import wkt

# =============================
# CONFIG
# =============================
ELEC_CSV_PATH = "../dataset/electricity/electricity_by_county.csv"
CLIMATE_NC_PATH = "../dataset/california_regional_timeseries.nc"
RESULTS_CSV_PATH = "../dataset/impact_analysis_results_per_region.csv"

# Cal-Adapt WRCC climate regions
REGIONS_URL = "https://api.cal-adapt.org/api/climregions/"

# Cache for ZIP->Region weights (so you don’t recompute spatial overlay every run)
ZIP_REGION_WEIGHTS_CACHE = "../dataset/zip_to_climregion_weights.csv"

# TIGER/Line ZCTA (ZIP Code Tabulation Area) polygons (BIG download)
TIGER_ZCTA_URL = "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
TIGER_DIR = Path("../dataset/boundaries/tiger_zcta520_2020")
TIGER_DIR.mkdir(parents=True, exist_ok=True)

# CRS for area computations in California (CA Albers)
AREA_CRS = "EPSG:3310"


# =============================
# Helpers
# =============================
def clean_zip5(x) -> str:
    """
    Return a 5-digit ZIP string or None.
    Handles floats/ints/strings; keeps leading zeros.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    # common case: "94107.0"
    if s.endswith(".0"):
        s = s[:-2]
    # drop non-digits
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) == 0:
        return None
    if len(s) > 5:
        s = s[:5]
    return s.zfill(5)


def _fix_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[gdf.geometry.notnull()].copy()
    # try make_valid when available; fallback to buffer(0)
    try:
        gdf["geometry"] = gdf["geometry"].make_valid()
    except Exception:
        gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf[gdf.geometry.notnull()].copy()


# =============================
# 1) Fetch Cal-Adapt polygons (ALL pages)
# =============================
def fetch_caladapt_features(url: str, pagesize: int = 5000) -> dict:
    """
    Fetch ALL pages from a Cal-Adapt vector endpoint and return a combined FeatureCollection-like dict.
    Works with the default 'json' representation (FeatureCollection + pagination fields).
    """
    sep = "&" if "?" in url else "?"
    next_url = f"{url}{sep}pagesize={pagesize}"
    all_features = []
    meta = None

    while next_url:
        with urlopen(next_url) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Cal-Adapt returns:
        #  { "type": "FeatureCollection", "features": [...], "count": ..., "next": ..., "previous": ..., "crs": ... }
        feats = data.get("features", [])
        all_features.extend(feats)

        meta = data  # keep last meta (crs/count)
        next_url = data.get("next", None)

    out = {
        "type": "FeatureCollection",
        "features": all_features,
        "crs": (meta.get("crs") if meta else None),
    }
    return out


def caladapt_to_gdf(fc: dict) -> gpd.GeoDataFrame:
    """
    Convert Cal-Adapt FeatureCollection to GeoDataFrame.
    Cal-Adapt sometimes encodes geometry as WKT string; handle both WKT and GeoJSON dict.
    CRS in response often indicates EPSG:3857.
    """
    features = fc.get("features", [])
    rows = []
    for f in features:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", None)

        if geom is None:
            continue

        if isinstance(geom, str):
            # WKT (as seen in Cal-Adapt API examples)
            geom_obj = wkt.loads(geom)
        elif isinstance(geom, dict):
            # GeoJSON geometry
            geom_obj = shape(geom)
        else:
            continue

        rows.append({**props, "geometry": geom_obj})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")

    # Try to read CRS from the payload; default to EPSG:3857 if absent (Cal-Adapt commonly uses 3857)
    crs = None
    try:
        crs_name = (fc.get("crs") or {}).get("properties", {}).get("name", "")
        if "EPSG" in crs_name:
            # e.g. "urn:ogc:def:crs:EPSG::3857"
            epsg = crs_name.split("EPSG::")[-1]
            if epsg.isdigit():
                crs = f"EPSG:{epsg}"
    except Exception:
        crs = None

    gdf = gdf.set_crs(crs or "EPSG:3857")
    return _fix_geoms(gdf)


def load_climate_regions_gdf(regions_url: str = REGIONS_URL) -> gpd.GeoDataFrame:
    fc = fetch_caladapt_features(regions_url, pagesize=2000)
    gdf = caladapt_to_gdf(fc)

    # Pick region name field
    candidate_cols = ["name", "title", "label", "region", "abbr", "code"]
    name_col = next((c for c in candidate_cols if c in gdf.columns), None)
    if name_col is None:
        # fallback: first non-geometry column
        name_col = [c for c in gdf.columns if c != "geometry"][0]

    gdf["Region"] = gdf[name_col].astype("string").str.strip()
    return gdf[["Region", "geometry"]].copy()


# =============================
# 2) Download/Load ZCTA polygons
# =============================
def ensure_zcta_shapefile(tiger_zip_url: str = TIGER_ZCTA_URL, out_dir: Path = TIGER_DIR) -> Path:
    """
    Download + extract TIGER ZCTA zip if missing. Returns path to extracted .shp.
    """
    zip_path = out_dir / Path(tiger_zip_url).name
    shp_path = None

    # already extracted?
    for p in out_dir.glob("*.shp"):
        shp_path = p
        break
    if shp_path and shp_path.exists():
        return shp_path

    # download if missing
    if not zip_path.exists():
        print(f"[INFO] Downloading ZCTA polygons (large): {tiger_zip_url}")
        urlretrieve(tiger_zip_url, zip_path)

    print(f"[INFO] Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    # find shp
    for p in out_dir.glob("*.shp"):
        shp_path = p
        break
    if shp_path is None:
        raise FileNotFoundError(f"Could not find .shp after extracting {zip_path}")

    return shp_path


def load_zcta_gdf(zcta_shp: Path, keep_zips: set[str]) -> gpd.GeoDataFrame:
    """
    Load ZCTA GeoDataFrame and filter to ZIPs used in your electricity dataset.
    """
    zcta = gpd.read_file(zcta_shp)

    # Detect ZCTA code column
    zcta_cols = [c for c in zcta.columns if "ZCTA" in c.upper() and "CE" in c.upper()]
    if len(zcta_cols) == 0:
        # common fallback is ZCTA5CE20
        raise ValueError(f"Could not detect ZCTA id column in {list(zcta.columns)}")
    zcta_id_col = zcta_cols[0]

    zcta["ZipCode"] = zcta[zcta_id_col].astype(str).str.zfill(5)
    zcta = zcta[zcta["ZipCode"].isin(keep_zips)].copy()

    if zcta.crs is None:
        # TIGER often comes with a CRS; if missing, assume NAD83 (EPSG:4269)
        zcta = zcta.set_crs("EPSG:4269")

    return _fix_geoms(zcta[["ZipCode", "geometry"]].copy())


# =============================
# 3) ZIP -> Region weights (fractional by area)
# =============================
def build_zip_region_weights(
    unique_zipcodes: np.ndarray,
    cache_path: str = ZIP_REGION_WEIGHTS_CACHE,
) -> pd.DataFrame:
    """
    Returns a dataframe: ZipCode, Region, Weight (area fraction within ZIP/ZCTA).
    Cached to disk to avoid repeating expensive spatial overlay.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        w = pd.read_csv(cache_path, dtype={"ZipCode": str, "Region": str})
        # quick sanity
        if {"ZipCode", "Region", "Weight"}.issubset(w.columns):
            return w

    zips = sorted({clean_zip5(z) for z in unique_zipcodes if clean_zip5(z) is not None})
    keep = set(zips)

    print(f"[INFO] Building ZIP->Region weights for {len(keep):,} ZIPs...")

    # Load polygons
    regions = load_climate_regions_gdf(REGIONS_URL)
    zcta_shp = ensure_zcta_shapefile()
    zcta = load_zcta_gdf(zcta_shp, keep)

    # Project to equal-area CRS for area fractions
    regions = regions.to_crs(AREA_CRS)
    zcta = zcta.to_crs(AREA_CRS)

    # Spatial intersection
    inter = gpd.overlay(zcta, regions, how="intersection", keep_geom_type=False)
    inter["Area"] = inter.geometry.area

    # Sum overlap area by ZIP-region
    w = (
        inter.groupby(["ZipCode", "Region"], as_index=False)["Area"]
        .sum()
        .sort_values(["ZipCode", "Region"])
    )
    total = w.groupby("ZipCode")["Area"].transform("sum")
    w["Weight"] = w["Area"] / total
    w = w.drop(columns=["Area"])

    # Normalize tiny numeric drift
    w["Weight"] = w["Weight"].clip(lower=0)
    # optional: renormalize per Zip
    w["Weight"] = w["Weight"] / w.groupby("ZipCode")["Weight"].transform("sum")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(cache_path, index=False)
    print(f"[INFO] Saved ZIP->Region weights: {cache_path}")

    # Report missing ZIPs
    mapped = set(w["ZipCode"].unique())
    missing = sorted(keep - mapped)
    if missing:
        print(f"[WARN] {len(missing)} ZIPs not found in ZCTA shapefile (will be dropped). Example: {missing[:10]}")

    return w


def load_electricity_data() -> pd.DataFrame:
    df = pd.read_csv(ELEC_CSV_PATH)

    # Clean ZIP, types
    df["ZipCode"] = df["ZipCode"].apply(clean_zip5)
    df = df.dropna(subset=["ZipCode"]).copy()

    # Drop obvious junk
    df["TotalkWh"] = pd.to_numeric(df["TotalkWh"], errors="coerce")
    df = df[df["TotalkWh"].fillna(0) > 0].copy()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Month", "CustomerClassClean"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    # Build/Load ZIP->Region weights and allocate kWh fractionally
    weights = build_zip_region_weights(df["ZipCode"].unique(), cache_path=ZIP_REGION_WEIGHTS_CACHE)
    merged = df.merge(weights, on="ZipCode", how="inner")
    merged["TotalkWh"] = merged["TotalkWh"] * merged["Weight"]

    # Aggregate to region-month-class
    elec_agg = (
        merged.groupby(["Region", "Year", "Month", "CustomerClassClean"], as_index=False)["TotalkWh"]
        .sum()
    )
    return elec_agg


# =============================
# 4) Climate flags (month-specific quantiles)
# =============================
def analyze_climate_extremes_monthly(
    ds: xr.Dataset,
    baseline_years=(1951, 2025),
    heat_var="sti_scdhi",
    drought_var="sapei_scdhi",
    heat_q=0.90,
    drought_q=0.10,
) -> pd.DataFrame:
    # Make sure coords exist
    if "region" not in ds.coords and "region" not in ds.dims:
        raise ValueError("Dataset must have a 'region' dimension/coord.")
    if "year" not in ds.coords and "year" not in ds.dims:
        raise ValueError("Dataset must have a 'year' dimension/coord.")
    if "month" not in ds.coords and "month" not in ds.dims:
        raise ValueError("Dataset must have a 'month' dimension/coord.")

    # Clean region labels to match electricity mapping
    ds = ds.assign_coords(region=[str(r).strip() for r in ds["region"].values])

    # Clip baseline to what’s actually available in ds
    y0, y1 = baseline_years
    y_min = int(ds["year"].min().values)
    y_max = int(ds["year"].max().values)
    y0c, y1c = max(y0, y_min), min(y1, y_max)

    if y0c > y1c:
        raise ValueError(f"Baseline years {baseline_years} not compatible with ds years [{y_min}, {y_max}].")

    base = ds.sel(year=slice(y0c, y1c))

    # month-specific thresholds across years (region x month)
    heat_thr = base[heat_var].quantile(heat_q, dim="year").reset_coords(drop=True)
    drought_thr = base[drought_var].quantile(drought_q, dim="year").reset_coords(drop=True)

    # flags (region x year x month)
    is_heat = (ds[heat_var] > heat_thr).reset_coords(drop=True)
    is_drought = (ds[drought_var] < drought_thr).reset_coords(drop=True)
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

    # Ensure ints
    df_flags["Year"] = df_flags["Year"].astype(int)
    df_flags["Month"] = df_flags["Month"].astype(int)

    return df_flags[["Region", "Year", "Month", "Is_Heatwave", "Is_Drought", "Is_Compound"]]


# =============================
# 5) Impact summaries
#    - Seasonal anomaly vs "Normal" baseline (your approach)
#    - TWO-WAY FE de-season + de-trend residuals (recommended)
# =============================
def add_buckets(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
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
    return g


def add_moy_anomalies(g: pd.DataFrame, ignore_zero=True) -> pd.DataFrame:
    """
    Month-of-year baseline computed from NORMAL months (fallback to all months if needed).
    Adds AnomPct_vsMedianBase (median baseline) and AnomPct_vsMeanBase.
    """
    g = g.copy()
    y = g["TotalkWh"].astype(float)
    if ignore_zero:
        y = y.replace(0, np.nan)
    g["_y"] = y

    normal = g[g["Bucket"] == "Normal"].copy()
    base_mean = normal.groupby("Month")["_y"].mean()
    base_median = normal.groupby("Month")["_y"].median()

    fb_mean = g.groupby("Month")["_y"].mean()
    fb_median = g.groupby("Month")["_y"].median()

    base_mean = base_mean.combine_first(fb_mean)
    base_median = base_median.combine_first(fb_median)

    # final fallback (global)
    base_mean = base_mean.fillna(g["_y"].mean())
    base_median = base_median.fillna(g["_y"].median())

    g["BaselineMean_moy"] = g["Month"].map(base_mean).replace(0, np.nan)
    g["BaselineMedian_moy"] = g["Month"].map(base_median).replace(0, np.nan)

    g["AnomPct_vsMeanBase"] = 100.0 * (g["_y"] - g["BaselineMean_moy"]) / g["BaselineMean_moy"]
    g["AnomPct_vsMedianBase"] = 100.0 * (g["_y"] - g["BaselineMedian_moy"]) / g["BaselineMedian_moy"]
    g = g.drop(columns=["_y"])
    return g


def add_twfe_residuals(g: pd.DataFrame, log_outcome=True) -> pd.DataFrame:
    """
    Two-way FE transform within (Region, Class):
      y_tilde = y - mean_month(y) - mean_year(y) + mean_all(y)
    Using log(TotalkWh) makes it interpretable ~ percent.
    Adds:
      - TWFE_LogResid
      - TWFE_PctApprox (100*(exp(resid)-1))
    """
    g = g.copy()
    y = g["TotalkWh"].astype(float)

    if log_outcome:
        y = y.where(y > 0, np.nan)
        y = np.log(y)

    # If too many NaNs, residuals become useless; still compute
    m_mean = y.groupby(g["Month"]).transform("mean")
    y_mean = y.groupby(g["Year"]).transform("mean")
    overall = np.nanmean(y)

    resid = y - m_mean - y_mean + overall
    g["TWFE_LogResid"] = resid

    # Convert log residual to percent effect
    g["TWFE_PctApprox"] = 100.0 * (np.exp(resid) - 1.0)
    return g


def summarize_impacts_two_cases(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Reports BOTH:
      - Pure buckets (Heat_only, Drought_only, Compound, Normal)
      - Overlapping (Heatwave includes compound, Drought includes compound, plus Compound, Normal)

    And includes:
      - mean/median kWh
      - mean/median MOY anomaly %
      - mean/median TWFE percent (de-season + de-trend)
    """
    results = []

    def add_result(region, cust_class, case, event, df_event):
        n = int(len(df_event))
        if n == 0:
            results.append({
                "Region": region, "Class": cust_class, "Case": case, "Event": event,
                "Count_Months": 0,
                "Mean_kWh": np.nan, "Median_kWh": np.nan,
                "Mean_AnomPct_vsMeanBase": np.nan, "Median_AnomPct_vsMeanBase": np.nan,
                "Mean_AnomPct_vsMedianBase": np.nan, "Median_AnomPct_vsMedianBase": np.nan,
                "Mean_TWFE_Pct": np.nan, "Median_TWFE_Pct": np.nan,
            })
            return

        results.append({
            "Region": region, "Class": cust_class, "Case": case, "Event": event,
            "Count_Months": n,
            "Mean_kWh": float(df_event["TotalkWh"].mean()),
            "Median_kWh": float(df_event["TotalkWh"].median()),
            "Mean_AnomPct_vsMeanBase": float(df_event["AnomPct_vsMeanBase"].mean()),
            "Median_AnomPct_vsMeanBase": float(df_event["AnomPct_vsMeanBase"].median()),
            "Mean_AnomPct_vsMedianBase": float(df_event["AnomPct_vsMedianBase"].mean()),
            "Median_AnomPct_vsMedianBase": float(df_event["AnomPct_vsMedianBase"].median()),
            "Mean_TWFE_Pct": float(df_event["TWFE_PctApprox"].mean()),
            "Median_TWFE_Pct": float(df_event["TWFE_PctApprox"].median()),
        })

    for (region, cust_class), g in merged.groupby(["Region", "CustomerClassClean"]):
        g = add_buckets(g)
        g = add_moy_anomalies(g, ignore_zero=True)
        g = add_twfe_residuals(g, log_outcome=True)

        # baseline (normal)
        normal = g[g["Bucket"] == "Normal"]

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
        ov_comp = pure_comp             # intersection

        add_result(region, cust_class, "Overlapping", "Normal", normal)
        add_result(region, cust_class, "Overlapping", "Heatwave", ov_hw)
        add_result(region, cust_class, "Overlapping", "Drought", ov_dr)
        add_result(region, cust_class, "Overlapping", "Compound", ov_comp)

    return pd.DataFrame(results)


# =============================
# MAIN
# =============================
def main():
    elec_df = load_electricity_data()
    ds = xr.open_dataset(CLIMATE_NC_PATH)

    climate_flags = analyze_climate_extremes_monthly(
        ds,
        baseline_years=(1951, 2025),
        heat_var="sti_scdhi",
        drought_var="sapei_scdhi",
        heat_q=0.90,
        drought_q=0.10,
    )

    # Merge
    merged = pd.merge(elec_df, climate_flags, on=["Region", "Year", "Month"], how="inner")

    # Summaries
    results_df = summarize_impacts_two_cases(merged)

    event_order = {
        "Normal": 0,
        "Heatwave_only": 1,
        "Heatwave": 1,
        "Drought_only": 2,
        "Drought": 2,
        "Compound": 3,
    }
    results_df["EventOrder"] = results_df["Event"].map(event_order).fillna(99).astype(int)

    # Preview
    print(
        results_df
        .sort_values(["Region", "Class", "Case", "EventOrder", "Event"])
        .drop(columns=["EventOrder"])
        .head(40)
    )

    # Save
    Path(RESULTS_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    results_df.drop(columns=["EventOrder"]).to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
