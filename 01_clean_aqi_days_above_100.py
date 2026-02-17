"""
01_clean_aqi_days_above_100.py
Clean AQI data and calculate days above AQI 100 per monitor per year (2020-2025).
Python equivalent of 01_us_calculate_days_above_aqi.R.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"
YEARS = list(range(2020, 2026))  # 2020 through 2025
AQI_THRESHOLD = 100

# EPA AirData daily files: one CSV per year per pollutant.
# Download from: https://aqs.epa.gov/aqsweb/airdata/download_files.html
# e.g. daily_44201_2020.zip (PM2.5), daily_42401_2020.zip (Ozone)
# This script expects either:
#   - raw/daily_44201_YYYY.csv (PM2.5) and/or daily_42401_YYYY.csv (Ozone)
#   - or a single combined daily file with columns: Date, AQI, State Code, County Code, Site Num, Lat, Lon
# We use "daily_aqi_by_cbsa" or "daily_44201" (PM2.5) if available; otherwise document manual download.
# -----------------------------------------------------------------------------


def ensure_dirs():
    """Create data directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_daily_aqi_for_year(year: int):
    """
    Load daily AQI data for one year.
    Expects EPA AirData format: daily_44201_YYYY.csv (PM2.5) or similar.
    Columns typically: State Code, County Code, Site Num, Parameter Code,
    POC, Lat, Lon, Date, Daily Mean PM2.5 Concentration, AQI, etc.
    """
    # Try PM2.5 daily (parameter 44201)
    path_pm25 = RAW_DIR / f"daily_44201_{year}.csv"
    if path_pm25.exists():
        df = pd.read_csv(path_pm25)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        return df

    # Try generic daily AQI file name
    path_gen = RAW_DIR / f"daily_aqi_{year}.csv"
    if path_gen.exists():
        df = pd.read_csv(path_gen)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df.dropna(subset=["Date"]) if "Date" in df.columns else df

    return None


def get_aqi_column(df: pd.DataFrame) -> str:
    """Identify the AQI column (EPA uses 'AQI' or 'DAILY_AQI_VALUE')."""
    for col in ["AQI", "DAILY_AQI_VALUE", "daily_aqi"]:
        if col in df.columns:
            return col
    # Some files have "Arithmetic Mean" for concentration; we need AQI. If only concentration, we could convert.
    return ""


def site_id(df: pd.DataFrame) -> pd.Series:
    """Create a unique site identifier from State, County, Site Number (and Lat/Lon if present)."""
    state = df["State Code"].astype(str).str.zfill(2) if "State Code" in df.columns else pd.Series(["00"] * len(df), index=df.index)
    county = df["County Code"].astype(str).str.zfill(3) if "County Code" in df.columns else pd.Series(["000"] * len(df), index=df.index)
    site = df["Site Num"].astype(str) if "Site Num" in df.columns else df.index.astype(str)
    return state + "-" + county + "-" + site


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean daily AQI data: drop missing AQI, ensure numeric, optional bounds.
    """
    aqi_col = get_aqi_column(df)
    if not aqi_col:
        raise ValueError(
            "No AQI column found. Expected one of: AQI, DAILY_AQI_VALUE, daily_aqi. "
            f"Columns: {list(df.columns)}"
        )
    out = df[["Date", aqi_col]].copy()
    out = out.rename(columns={aqi_col: "AQI"})
    out["AQI"] = pd.to_numeric(out["AQI"], errors="coerce")
    out = out.dropna(subset=["AQI"])
    # Optional: clamp to valid AQI range 0–500
    out["AQI"] = out["AQI"].clip(0, 500)
    # Keep site and location for merging
    for col in ["State Code", "County Code", "Site Num", "Latitude", "Longitude", "Lat", "Lon"]:
        if col in df.columns:
            out[col] = df[col].values
    if "Latitude" not in out.columns and "Lat" in df.columns:
        out["Latitude"] = df["Lat"]
        out["Longitude"] = df["Lon"]
    out["Site_ID"] = site_id(df) if "State Code" in df.columns else df.index.astype(str)
    return out


def days_above_aqi_per_site_year(daily_clean: pd.DataFrame, threshold: int = AQI_THRESHOLD) -> pd.DataFrame:
    """
    For each site and year, count days where AQI > threshold.
    """
    daily_clean = daily_clean.copy()
    daily_clean["Year"] = daily_clean["Date"].dt.year
    above = daily_clean[daily_clean["AQI"] > threshold]
    counts = (
        above.groupby(["Site_ID", "Year"], dropna=False)
        .size()
        .reset_index(name="days_above_100")
    )
    # One row per site with lat/lon (take first occurrence)
    geo_cols = {}
    if "Latitude" in daily_clean.columns:
        geo_cols["Latitude"] = ("Latitude", "first")
    elif "Lat" in daily_clean.columns:
        geo_cols["Latitude"] = ("Lat", "first")
    if "Longitude" in daily_clean.columns:
        geo_cols["Longitude"] = ("Longitude", "first")
    elif "Lon" in daily_clean.columns:
        geo_cols["Longitude"] = ("Lon", "first")
    if geo_cols:
        site_geo = daily_clean.groupby("Site_ID").agg(**geo_cols).reset_index()
    else:
        site_geo = counts[["Site_ID"]].drop_duplicates()
    result = counts.merge(site_geo, on="Site_ID", how="left")
    return result


def run_all_years() -> pd.DataFrame:
    """Load, clean, and compute days above AQI 100 for all years; combine into one long table."""
    ensure_dirs()
    all_site_years = []

    for year in YEARS:
        df = load_daily_aqi_for_year(year)
        if df is None:
            print(f"No data found for {year}. Place EPA AirData daily CSV in {RAW_DIR} (e.g. daily_44201_{year}.csv).")
            continue
        cleaned = clean_daily(df)
        site_year = days_above_aqi_per_site_year(cleaned)
        all_site_years.append(site_year)

    if not all_site_years:
        # Create synthetic example so IDW and viz scripts can run
        print("No EPA data found. Writing synthetic example data for 2020-2025.")
        rng = np.random.default_rng(42)
        n_sites = 50
        lats = rng.uniform(25, 49, n_sites)
        lons = rng.uniform(-125, -66, n_sites)
        site_ids = [f"site_{i:03d}" for i in range(n_sites)]
        rows = []
        for year in YEARS:
            for i in range(n_sites):
                rows.append({
                    "Site_ID": site_ids[i],
                    "Year": year,
                    "days_above_100": int(rng.integers(0, 60)),
                    "Latitude": lats[i],
                    "Longitude": lons[i],
                })
        combined = pd.DataFrame(rows)
    else:
        combined = pd.concat(all_site_years, ignore_index=True)

    out_path = OUT_DIR / "days_above_aqi100_by_site_year.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(combined)} rows.")
    return combined


if __name__ == "__main__":
    run_all_years()
