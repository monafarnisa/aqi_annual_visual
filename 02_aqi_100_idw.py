"""
02_aqi_100_idw.py
Raster-based IDW interpolation for days above AQI 100.
Consumes script 1 outputs and writes GeoTIFFs + static previews.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import box
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
INTERMEDIATE_DIR = DATA_DIR / "processed"
OUT_DIR = DATA_DIR / "idw_raster"
PREVIEW_DIR = Path(__file__).resolve().parent / "outputs" / "idw_raster_static"
INSET_DIR = Path(__file__).resolve().parent / "outputs" / "idw_raster_inset"
YEARS = list(range(2020, 2026))
THRESHOLD = 100
NODATA = -9999.0

# Region CRSs chosen for better distance behavior in IDW.
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


@lru_cache(maxsize=1)
def load_us_states() -> gpd.GeoDataFrame:
    """Load US state polygons from Census URL; fallback caller handles failures."""
    def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if gdf.crs is None:
            return gdf.set_crs(4326)
        return gdf.to_crs(4326)

    try:
        return _to_wgs84(gpd.read_file(CENSUS_STATES_URL))
    except Exception:
        # Retry via explicit download when direct read has SSL/cert issues.
        resp = requests.get(CENSUS_STATES_URL, timeout=120, verify=False)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            return _to_wgs84(gpd.read_file(tmp.name))


@lru_cache(maxsize=None)
def fetch_region_boundary(region_key: str) -> gpd.GeoDataFrame:
    cfg = REGIONS[region_key]
    try:
        states = load_us_states()
        if cfg["stusps"] is not None:
            states = states[states["STUSPS"].isin(cfg["stusps"])].copy()
        else:
            states = states[~states["STUSPS"].isin(cfg["drop"])].copy()
        return states.dissolve().reset_index(drop=True)
    except Exception:
        # Fallback keeps workflow running when remote shapefile fails (SSL/network issues).
        minx, miny, maxx, maxy = FALLBACK_REGION_BOUNDS_WGS84[region_key]
        return gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326)


@lru_cache(maxsize=None)
def fetch_region_states(region_key: str) -> gpd.GeoDataFrame:
    cfg = REGIONS[region_key]
    try:
        states = load_us_states()
        if cfg["stusps"] is not None:
            return states[states["STUSPS"].isin(cfg["stusps"])].copy()
        return states[~states["STUSPS"].isin(cfg["drop"])].copy()
    except Exception:
        # Empty fallback when state polygons are unavailable.
        return gpd.GeoDataFrame(geometry=[], crs=4326)


def idw_to_grid(
    xy_data: np.ndarray,
    values: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    idp: float = 5.0,
    nmax: int = 10,
) -> np.ndarray:
    """
    IDW over a raster grid defined by xgrid (cols) and ygrid (rows).
    Returns a 2D array shaped (len(ygrid), len(xgrid)).
    """
    k = min(max(1, nmax), len(values))
    tree = cKDTree(xy_data)

    X, Y = np.meshgrid(xgrid, ygrid)
    q = np.column_stack([X.ravel(), Y.ravel()])
    dists, idxs = tree.query(q, k=k)

    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    out = np.empty(q.shape[0], dtype=float)
    zero = (dists == 0).any(axis=1)
    if zero.any():
        zr = np.where(zero)[0]
        zc = (dists[zr] == 0).argmax(axis=1)
        out[zr] = values[idxs[zr, zc]]

    nr = np.where(~zero)[0]
    if nr.size:
        d = dists[nr]
        i = idxs[nr]
        w = 1.0 / np.power(d, idp)
        out[nr] = (w * values[i]).sum(axis=1) / w.sum(axis=1)

    return out.reshape(len(ygrid), len(xgrid))


def make_grid_from_bounds(bounds: tuple[float, float, float, float], res_m: float):
    """
    bounds: (minx, miny, maxx, maxy) in projected meters.
    res_m: pixel size in meters.
    """
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / res_m))
    height = int(np.ceil((maxy - miny) / res_m))

    xgrid = minx + (np.arange(width) + 0.5) * res_m
    ygrid = maxy - (np.arange(height) + 0.5) * res_m
    transform = from_origin(minx, maxy, res_m, res_m)
    return xgrid, ygrid, transform, width, height


def interpolate_days_above_100_to_tifs(
    df: pd.DataFrame,
    year: int,
    out_dir: Path,
    value_col: str = "days_above_100",
    idp: float = 5.0,
    nmax: int = 10,
    res_km_by_region: dict[str, float] | None = None,
    nodata: float = NODATA,
) -> dict[str, Path]:
    if res_km_by_region is None:
        res_km_by_region = {"conus": 5.0, "ak": 20.0, "hi": 5.0}

    required = {"longitude", "latitude", value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pts = gpd.GeoDataFrame(
        df[[value_col]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for region_key, cfg in REGIONS.items():
        crs = cfg["crs"]
        res_m = res_km_by_region[region_key] * 1000.0
        boundary = fetch_region_boundary(region_key).to_crs(crs)
        region_poly = boundary.geometry.iloc[0]

        pts_r = pts.to_crs(crs)
        pts_r = pts_r[pts_r.within(region_poly)].copy()
        if pts_r.empty:
            print(f"{year} {region_key}: no points in region, skipping.")
            continue

        bounds = boundary.total_bounds
        xgrid, ygrid, transform, width, height = make_grid_from_bounds(
            (bounds[0], bounds[1], bounds[2], bounds[3]),
            res_m=res_m,
        )

        xy = np.column_stack([pts_r.geometry.x.values, pts_r.geometry.y.values])
        vals = pts_r[value_col].values.astype(float)
        z = idw_to_grid(xy, vals, xgrid, ygrid, idp=idp, nmax=nmax)

        mask = geometry_mask(
            geometries=boundary.geometry,
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
        z_masked = np.where(mask, z, nodata).astype(np.float32)

        out_path = out_dir / f"days_above_100_idw_{region_key}_{year}_res{int(res_m)}m.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=f"EPSG:{crs}",
            transform=transform,
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(z_masked, 1)

        outputs[region_key] = out_path
        print(f"Saved {out_path}")

    return outputs


def save_tif_preview(tif_path: Path, preview_dir: Path) -> Path:
    preview_dir.mkdir(parents=True, exist_ok=True)
    region_key = next((k for k in REGIONS if f"_{k}_" in tif_path.name), "conus")
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        bounds = src.bounds
        tif_crs = src.crs

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        arr,
        cmap="YlOrRd",
        interpolation="nearest",
        origin="upper",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        vmin=0,
    )
    fig.colorbar(im, ax=ax, label="Days above AQI 100")

    region_boundary = fetch_region_boundary(region_key)
    if not region_boundary.empty:
        region_boundary.to_crs(tif_crs).boundary.plot(ax=ax, color="black", linewidth=1.0)

    states = fetch_region_states(region_key)
    if not states.empty:
        states.to_crs(tif_crs).boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.6)

    ax.set_title(tif_path.stem)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    out_png = preview_dir / f"{tif_path.stem}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def _read_tif_array(tif_path: Path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr, src.bounds, src.crs


def save_inset_map(year: int, tif_paths: dict[str, Path], out_dir: Path) -> Path | None:
    if not all(k in tif_paths for k in ("conus", "ak", "hi")):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax_main = fig.add_axes([0.03, 0.06, 0.78, 0.88])
    ax_ak = fig.add_axes([0.72, 0.58, 0.25, 0.30])
    ax_hi = fig.add_axes([0.76, 0.24, 0.18, 0.20])

    def plot_region(ax, region_key: str):
        arr, bounds, tif_crs = _read_tif_array(tif_paths[region_key])
        im = ax.imshow(
            arr,
            cmap="YlOrRd",
            interpolation="nearest",
            origin="upper",
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
            vmin=0,
        )
        region_boundary = fetch_region_boundary(region_key)
        if not region_boundary.empty:
            region_boundary.to_crs(tif_crs).boundary.plot(ax=ax, color="black", linewidth=1.1)

        states = fetch_region_states(region_key)
        if not states.empty:
            states.to_crs(tif_crs).boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.65)
        ax.set_axis_off()
        ax.set_aspect("equal")
        return im

    im = plot_region(ax_main, "conus")
    plot_region(ax_ak, "ak")
    plot_region(ax_hi, "hi")

    ax_main.set_title(f"AQI days > 100 ({year})", loc="left", fontsize=22, fontweight="bold")
    ax_main.text(0.0, 0.98, f"{year}; raster IDW", transform=ax_main.transAxes, fontsize=14, va="top")
    ax_ak.set_title("AK", fontsize=11)
    ax_hi.set_title("HI", fontsize=11)

    cax = fig.add_axes([0.86, 0.18, 0.02, 0.55])
    fig.colorbar(im, cax=cax, label="Days above AQI 100")

    out_png = out_dir / f"us_days_above_100_inset_{year}.png"
    fig.savefig(out_png, dpi=190)
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def run_year(year: int) -> dict[str, Path]:
    in_csv = INTERMEDIATE_DIR / f"us_days_above_{THRESHOLD}_{year}.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input from script 1: {in_csv}")

    df = pd.read_csv(in_csv).dropna(subset=["longitude", "latitude", f"days_above_{THRESHOLD}"])
    outputs = interpolate_days_above_100_to_tifs(
        df=df,
        year=year,
        out_dir=OUT_DIR,
        value_col=f"days_above_{THRESHOLD}",
        idp=5.0,
        nmax=10,
    )
    for tif in outputs.values():
        save_tif_preview(tif, PREVIEW_DIR)
    save_inset_map(year, outputs, INSET_DIR)
    return outputs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    INSET_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        run_year(year)


if __name__ == "__main__":
    main()
