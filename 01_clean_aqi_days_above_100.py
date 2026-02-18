"""
01_clean_aqi_days_above_100.py
Build a site-level yearly table of days above AQI 100 (2020-2025).
"""

from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "processed"
YEARS = list(range(2020, 2026))
AQI_THRESHOLD = 100


def snake_case_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def _compose_defining_site(
    state_code: pd.Series,
    county_code: pd.Series,
    site_code: pd.Series,
) -> pd.Series:
    return (
        state_code.astype(str).str.strip().str.zfill(2)
        + "-"
        + county_code.astype(str).str.strip().str.zfill(3)
        + "-"
        + site_code.astype(str).str.strip().str.zfill(4)
    )


def _normalize_defining_site(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    split = s.str.split("-", expand=True)
    if split.shape[1] == 3:
        return _compose_defining_site(split[0], split[1], split[2])
    return s


def process_aqi_data_left(aqi_data: pd.DataFrame, aqi_threshold: int = 100) -> pd.DataFrame:
    days_col = f"days_above_{aqi_threshold}"

    aqi_days_count = (
        aqi_data.loc[aqi_data["aqi"] > aqi_threshold]
        .groupby("defining_site", as_index=False)
        .size()
        .rename(columns={"size": "days"})
    )

    site_info = (
        aqi_data.sort_values("defining_site")
        .drop_duplicates(subset="defining_site", keep="first")
        .copy()
    )

    out = site_info.merge(aqi_days_count, on="defining_site", how="left")
    out["days"] = out["days"].fillna(0).astype(int)

    drop_cols = ["date", "aqi", "category", "number_of_sites_reporting", "defining_parameter"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out.rename(columns={"days": days_col})


def _load_daily_csv(raw_data_dir: Path, year: int, daily_filename_template: str) -> pd.DataFrame:
    candidates = [
        raw_data_dir / daily_filename_template.format(year=year),
        raw_data_dir / f"daily_aqi_by_county_{year}.csv",
        raw_data_dir / f"daily_aqi_by_site_{year}.csv",
        raw_data_dir / f"daily_44201_{year}.csv",
        raw_data_dir / f"daily_aqi_{year}.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using daily file: {path.name}")
            return pd.read_csv(path).pipe(snake_case_cols)

    raise FileNotFoundError(
        "No daily AQI file found for year "
        f"{year}. Looked for {[p.name for p in candidates]} in {raw_data_dir}."
    )


def run_days_above_100_workflow(
    year: int,
    raw_data_dir: Path,
    intermediate_dir: Path,
    daily_filename_template: str = "daily_aqi_by_site_{year}.csv",
    sites_filename: str = "aqs_sites.csv",
    threshold: int = 100,
    max_missing_coord_frac: float = 0.01,   # fail if >1% missing coords
) -> pd.DataFrame:
    daily = _load_daily_csv(raw_data_dir, year, daily_filename_template)

    if "aqi" not in daily.columns:
        raise ValueError(f"Missing required column 'aqi' in daily data for {year}.")

    if "defining_site" not in daily.columns:
        required = {"state_code", "county_code", "site_num"}
        if not required.issubset(set(daily.columns)):
            raise ValueError(
                "Missing 'defining_site' and missing components to build it. "
                "Expected either defining_site or state_code/county_code/site_num."
            )
        daily["defining_site"] = _compose_defining_site(
            daily["state_code"], daily["county_code"], daily["site_num"]
        )
    else:
        daily["defining_site"] = _normalize_defining_site(daily["defining_site"])

    sites_path = raw_data_dir / sites_filename
    if sites_path.exists():
        sites = pd.read_csv(sites_path).pipe(snake_case_cols)
        if "defining_site" not in sites.columns:
            needed = {"state_code", "county_code", "site_number"}
            if not needed.issubset(set(sites.columns)):
                raise ValueError(
                    "Site file is missing 'defining_site' and missing components "
                    "state_code/county_code/site_number."
                )
            sites["defining_site"] = _compose_defining_site(
                sites["state_code"], sites["county_code"], sites["site_number"]
            )
        else:
            sites["defining_site"] = _normalize_defining_site(sites["defining_site"])

        daily["defining_site"] = daily["defining_site"].astype(str)
        sites["defining_site"] = sites["defining_site"].astype(str)

        daily_coords = daily.merge(
            sites[["defining_site", "latitude", "longitude"]],
            on="defining_site",
            how="left",
        )
    else:
        # If no site table exists, use coordinates already present in daily file.
        daily_coords = daily.copy()
        if "latitude" not in daily_coords.columns and "lat" in daily_coords.columns:
            daily_coords["latitude"] = daily_coords["lat"]
        if "longitude" not in daily_coords.columns and "lon" in daily_coords.columns:
            daily_coords["longitude"] = daily_coords["lon"]

    if "latitude" not in daily_coords.columns or "longitude" not in daily_coords.columns:
        raise ValueError(
            f"No usable latitude/longitude columns for year {year}. "
            "Provide aqs_sites.csv or include lat/lon in daily file."
        )

    missing_frac = daily_coords["latitude"].isna().mean()
    if missing_frac > max_missing_coord_frac:
        raise ValueError(
            f"Too many missing coordinates after join: {missing_frac:.2%}. "
            "Check defining_site formatting / site table."
        )

    days_above = process_aqi_data_left(daily_coords, aqi_threshold=threshold)
    days_above["year"] = year

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    out_csv = intermediate_dir / f"us_days_above_{threshold}_{year}.csv"
    days_above.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return days_above


def run_all_years_workflow() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    yearly_frames = []
    for year in YEARS:
        try:
            yearly = run_days_above_100_workflow(
                year=year,
                raw_data_dir=RAW_DIR,
                intermediate_dir=INTERMEDIATE_DIR,
                threshold=AQI_THRESHOLD,
            )
            yearly_frames.append(yearly)
        except FileNotFoundError as exc:
            print(exc)
        except ValueError as exc:
            print(f"Skipping {year}: {exc}")

    if not yearly_frames:
        raise RuntimeError(
            "No yearly outputs were created. Add raw files in data/raw and re-run."
        )

    combined = pd.concat(yearly_frames, ignore_index=True)
    combined = combined.rename(
        columns={
            "year": "Year",
            "latitude": "Latitude",
            "longitude": "Longitude",
        }
    )

    out_combined = INTERMEDIATE_DIR / "days_above_aqi100_by_site_year.csv"
    combined.to_csv(out_combined, index=False)
    print(f"Wrote {out_combined} with {len(combined)} rows.")
    return combined


if __name__ == "__main__":
    run_all_years_workflow()
