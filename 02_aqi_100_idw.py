"""
02_aqi_100_idw.py
Inverse Distance Weighted (IDW) interpolation of days above AQI 100.
Produces a grid/raster per year for mapping. Python equivalent of 02_aqi_100_idw.R.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "idw"
YEARS = list(range(2020, 2026))
# IDW power (higher = more local)
IDW_POWER = 2
# Grid resolution (degrees). Finer = slower.
GRID_RES = 0.25


def idw_interpolate(points: np.ndarray, values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, power: float = 2):
    """
    Inverse Distance Weighted interpolation onto a regular grid.
    points: (n, 2) array of (x, y) = (lon, lat)
    values: (n,) array of z values
    grid_x, grid_y: 1D arrays of coordinates (e.g. lon and lat)
    Returns: 2D array of shape (len(grid_y), len(grid_x))
    """
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    dists = cdist(grid_pts, points)  # (n_grid, n_points)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / (dists ** power)
    weights /= weights.sum(axis=1, keepdims=True)
    z_grid = (weights * values).sum(axis=1)
    return z_grid.reshape(len(grid_y), len(grid_x))


def run_idw_for_year(df: pd.DataFrame, year: int, power: float = IDW_POWER, res: float = GRID_RES) -> dict:
    """
    Run IDW for one year. Returns dict with grid arrays and axes.
    """
    sub = df[df["Year"] == year].dropna(subset=["Longitude", "Latitude", "days_above_100"])
    if sub.empty:
        return None
    lon = sub["Longitude"].values
    lat = sub["Latitude"].values
    vals = sub["days_above_100"].values.astype(float)
    points = np.column_stack([lon, lat])

    # US bounding box (continental)
    lon_min, lon_max = lon.min() - 1, lon.max() + 1
    lat_min, lat_max = lat.min() - 1, lat.max() + 1
    lon_min, lon_max = max(lon_min, -125), min(lon_max, -66)
    lat_min, lat_max = max(lat_min, 24), min(lat_max, 50)

    grid_lon = np.arange(lon_min, lon_max + res / 2, res)
    grid_lat = np.arange(lat_min, lat_max + res / 2, res)
    Z = idw_interpolate(points, vals, grid_lon, grid_lat, power=power)
    return {
        "year": year,
        "grid": Z,
        "grid_lon": grid_lon,
        "grid_lat": grid_lat,
        "n_points": len(points),
    }


def save_grids_per_year(df: pd.DataFrame):
    """Run IDW for each year and save grids as CSV (lon, lat, days_above_100) for use in viz."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for year in YEARS:
        r = run_idw_for_year(df, year)
        if r is None:
            continue
        results[year] = r
        # Save as long-format CSV: lon, lat, days_above_100
        xx, yy = np.meshgrid(r["grid_lon"], r["grid_lat"])
        out_df = pd.DataFrame({
            "lon": xx.ravel(),
            "lat": yy.ravel(),
            "days_above_100": r["grid"].ravel(),
        })
        out_path = OUT_DIR / f"idw_{year}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
    return results


def main():
    processed_path = PROCESSED_DIR / "days_above_aqi100_by_site_year.csv"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Run 01_clean_aqi_days_above_100.py first to create {processed_path}"
        )
    df = pd.read_csv(processed_path)
    save_grids_per_year(df)


if __name__ == "__main__":
    main()
