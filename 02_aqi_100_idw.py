"""
02_aqi_100_idw.py
Regional IDW interpolation (CONUS, AK, HI) from script 1 outputs.
Writes yearly GeoPackage + CSV grid and static PNG maps.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, box

DATA_DIR = Path(__file__).resolve().parent / "data"
INTERMEDIATE_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "idw"
FIG_DIR = Path(__file__).resolve().parent / "outputs" / "idw_static"
YEARS = list(range(2020, 2026))
THRESHOLD = 100

REGIONS = {
    "conus": {"stusps": None, "drop": {"AK", "HI", "AS", "GU", "MP", "PR", "VI"}, "crs": 5070},
    "ak": {"stusps": {"AK"}, "drop": None, "crs": 3338},
    "hi": {"stusps": {"HI"}, "drop": None, "crs": 3759},
}

CENSUS_STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip"
FALLBACK_REGION_BOUNDS_WGS84 = {
    "conus": (-125.0, 24.0, -66.0, 50.0),
    "ak": (-170.0, 50.0, -130.0, 72.0),
    "hi": (-161.0, 18.0, -154.0, 23.0),
}


@lru_cache(maxsize=None)
def fetch_region_boundary(region_key: str) -> gpd.GeoDataFrame:
    cfg = REGIONS[region_key]
    try:
        states = gpd.read_file(CENSUS_STATES_URL).set_crs(4326)
        if cfg["stusps"] is not None:
            states = states[states["STUSPS"].isin(cfg["stusps"])].copy()
        else:
            states = states[~states["STUSPS"].isin(cfg["drop"])].copy()
        return states.dissolve().reset_index(drop=True)
    except Exception:
        minx, miny, maxx, maxy = FALLBACK_REGION_BOUNDS_WGS84[region_key]
        return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326)


def make_point_grid_within_polygon(poly_gdf: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    poly = poly_gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + spacing_m, spacing_m)
    ys = np.arange(miny, maxy + spacing_m, spacing_m)
    pts = [Point(x, y) for x in xs for y in ys]
    gdf = gpd.GeoDataFrame(geometry=pts, crs=poly_gdf.crs)
    return gdf[gdf.within(poly)].copy()


def _idw_predict_all(
    xy_grid: np.ndarray,
    xy_data: np.ndarray,
    z_data: np.ndarray,
    idp: float,
    nmax: int,
) -> np.ndarray:
    tree = cKDTree(xy_data)
    k = min(nmax, len(z_data))
    dists, idxs = tree.query(xy_grid, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    zero_mask = dists == 0
    out = np.empty(xy_grid.shape[0], dtype=float)

    any_zero = zero_mask.any(axis=1)
    if any_zero.any():
        hit_rows = np.where(any_zero)[0]
        hit_cols = zero_mask[hit_rows].argmax(axis=1)
        out[hit_rows] = z_data[idxs[hit_rows, hit_cols]]

    regular_rows = np.where(~any_zero)[0]
    if regular_rows.size > 0:
        d = dists[regular_rows]
        i = idxs[regular_rows]
        w = 1.0 / np.power(d, idp)
        out[regular_rows] = (w * z_data[i]).sum(axis=1) / w.sum(axis=1)

    return out


def idw_region(
    pts_wgs84: gpd.GeoDataFrame,
    region_key: str,
    value_col: str,
    grid_km: float,
    idp: float,
    nmax: int,
) -> gpd.GeoDataFrame:
    crs = REGIONS[region_key]["crs"]
    boundary_wgs84 = fetch_region_boundary(region_key)
    boundary = boundary_wgs84.to_crs(crs)
    pts = pts_wgs84.to_crs(crs)

    region_poly = boundary.geometry.iloc[0]
    pts = pts[pts.within(region_poly)].copy()
    if pts.empty:
        return gpd.GeoDataFrame(columns=[value_col, "region", "geometry"], geometry="geometry", crs=crs)

    spacing_m = grid_km * 1000.0
    grid = make_point_grid_within_polygon(boundary, spacing_m=spacing_m)

    xy_data = np.column_stack([pts.geometry.x.values, pts.geometry.y.values])
    z_data = pts[value_col].values.astype(float)
    xy_grid = np.column_stack([grid.geometry.x.values, grid.geometry.y.values])
    preds = _idw_predict_all(xy_grid, xy_data, z_data, idp=idp, nmax=nmax)

    out = grid.copy()
    out[value_col] = preds
    out["region"] = region_key
    return out.set_crs(crs)


def run_us_idw_all_regions(
    year: int,
    intermediate_dir: Path,
    out_dir: Path,
    threshold: int = 100,
    grid_km_by_region: dict[str, float] | None = None,
    idp: float = 5.0,
    nmax: int = 10,
    out_crs_for_animation: int = 4326,
) -> gpd.GeoDataFrame:
    if grid_km_by_region is None:
        grid_km_by_region = {"conus": 50.0, "ak": 80.0, "hi": 20.0}

    in_csv = intermediate_dir / f"us_days_above_{threshold}_{year}.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input from script 1: {in_csv}")

    df = pd.read_csv(in_csv)
    value_col = f"days_above_{threshold}"
    needed = {"longitude", "latitude", value_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{in_csv.name} missing columns: {sorted(missing)}")

    df = df.dropna(subset=["longitude", "latitude", value_col]).copy()
    pts = gpd.GeoDataFrame(
        df[[value_col]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326,
    )

    conus = idw_region(pts, "conus", value_col, grid_km_by_region["conus"], idp, nmax)
    ak = idw_region(pts, "ak", value_col, grid_km_by_region["ak"], idp, nmax)
    hi = idw_region(pts, "hi", value_col, grid_km_by_region["hi"], idp, nmax)

    conus = conus.to_crs(out_crs_for_animation)
    ak = ak.to_crs(out_crs_for_animation)
    hi = hi.to_crs(out_crs_for_animation)

    combined = gpd.GeoDataFrame(
        pd.concat([conus, ak, hi], ignore_index=True),
        geometry="geometry",
        crs=out_crs_for_animation,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg = out_dir / f"us_idw_days_above_{threshold}_{year}_all_regions_epsg{out_crs_for_animation}.gpkg"
    combined.to_file(out_gpkg, layer="idw", driver="GPKG")

    combined_xy = combined.copy()
    combined_xy["lon"] = combined_xy.geometry.x
    combined_xy["lat"] = combined_xy.geometry.y
    out_csv = out_dir / f"idw_{year}.csv"
    combined_xy[["lon", "lat", value_col, "region"]].to_csv(out_csv, index=False)

    print(f"Saved {out_gpkg}")
    print(f"Saved {out_csv}")
    return combined


def save_static_map(gdf_wgs84: gpd.GeoDataFrame, year: int, threshold: int = 100) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    value_col = f"days_above_{threshold}"
    fig, ax = plt.subplots(figsize=(11, 7))

    boundaries = [fetch_region_boundary("conus"), fetch_region_boundary("ak"), fetch_region_boundary("hi")]
    for b in boundaries:
        b.boundary.plot(ax=ax, color="black", linewidth=0.6)

    if not gdf_wgs84.empty:
        gdf_wgs84.plot(
            ax=ax,
            column=value_col,
            cmap="YlOrRd",
            markersize=4,
            legend=True,
            legend_kwds={"label": f"Days above AQI {threshold}"},
        )

    ax.set_title(f"IDW Surface of Days Above AQI {threshold} ({year})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-179, -65)
    ax.set_ylim(17, 73)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_png = FIG_DIR / f"idw_static_{year}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        result = run_us_idw_all_regions(
            year=year,
            intermediate_dir=INTERMEDIATE_DIR,
            out_dir=OUT_DIR,
            threshold=THRESHOLD,
        )
        save_static_map(result, year=year, threshold=THRESHOLD)


if __name__ == "__main__":
    main()
