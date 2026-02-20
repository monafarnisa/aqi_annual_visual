"""
03_visualize.py
Interactive Leaflet visualization with county borders and year layers.
Uses IDW interpolation rasters from script 2 as interactive layers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
IDW_RASTER_DIR = DATA_DIR / "idw_raster"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "leaflet"
YEARS = list(range(2020, 2026))
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip"
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip"
DROP_STUSPS = {"AS", "GU", "MP", "PR", "VI"}
NODATA = -9999.0


def _load_shapefile_url(url: str) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(url)
    except Exception:
        resp = requests.get(url, timeout=120, verify=False)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
            tmp.write(resp.content)
            tmp.flush()
            gdf = gpd.read_file(tmp.name)
    if gdf.crs is None:
        return gdf.set_crs(4326)
    return gdf.to_crs(4326)


def _tif_paths_for_year(year: int) -> dict[str, Path]:
    return {
        "conus": IDW_RASTER_DIR / f"days_above_100_idw_conus_{year}_res5000m.tif",
        "ak": IDW_RASTER_DIR / f"days_above_100_idw_ak_{year}_res20000m.tif",
        "hi": IDW_RASTER_DIR / f"days_above_100_idw_hi_{year}_res5000m.tif",
    }


def afmhot_colormap(vmin: float, vmax: float) -> bcm.LinearColormap:
    """Approximate matplotlib afmhot for Leaflet color rendering."""
    cmap = matplotlib.colormaps["afmhot"]
    stops = [mcolors.to_hex(cmap(i)) for i in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    return bcm.LinearColormap(colors=stops, vmin=vmin, vmax=vmax)


def _global_raster_max() -> float:
    max_val = 0.0
    for year in YEARS:
        for path in _tif_paths_for_year(year).values():
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                arr = src.read(1).astype(float)
                nodata = src.nodata if src.nodata is not None else NODATA
                arr[arr == nodata] = np.nan
                if np.isfinite(arr).any():
                    max_val = max(max_val, float(np.nanmax(arr)))
    return max_val if max_val > 0 else 1.0


def _reproject_to_wgs84(tif_path: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(tif_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata if src.nodata is not None else NODATA
        src_arr[src_arr == src_nodata] = np.nan

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    west, south, east, north = array_bounds(dst_height, dst_width, dst_transform)
    return dst_arr, (south, west, north, east)


def _rgba_for_leaflet(arr: np.ndarray, vmax: float) -> np.ndarray:
    cmap = matplotlib.colormaps["afmhot"]
    norm = mcolors.Normalize(vmin=0, vmax=vmax, clip=True)
    rgba = (cmap(norm(np.nan_to_num(arr, nan=0.0))) * 255).astype(np.uint8)
    rgba[np.isnan(arr), 3] = 0
    return rgba


def build_leaflet_map() -> Path:
    counties = _load_shapefile_url(COUNTIES_URL)
    states = _load_shapefile_url(STATES_URL)
    states = states[~states["STUSPS"].isin(DROP_STUSPS)].copy()
    counties = counties[counties["STATEFP"].isin(states["STATEFP"])].copy()

    max_val = _global_raster_max()
    cmap = afmhot_colormap(0, max_val)
    cmap.caption = "Days above AQI 100 (IDW interpolation)"

    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

    # County borders as context layer.
    folium.GeoJson(
        data=counties.to_json(),
        name="County borders",
        style_function=lambda _: {"color": "#666666", "weight": 0.35, "fillOpacity": 0.0},
    ).add_to(m)

    for year in YEARS:
        fg = folium.FeatureGroup(name=f"{year}", show=(year == YEARS[-1]))
        for region, tif_path in _tif_paths_for_year(year).items():
            if not tif_path.exists():
                continue
            arr, (south, west, north, east) = _reproject_to_wgs84(tif_path)
            rgba = _rgba_for_leaflet(arr, vmax=max_val)
            folium.raster_layers.ImageOverlay(
                image=rgba,
                bounds=[[south, west], [north, east]],
                opacity=0.75,
                interactive=False,
                name=f"{region} {year}",
            ).add_to(fg)

        fg.add_to(m)

    # State borders as a crisp overlay.
    folium.GeoJson(
        data=states.to_json(),
        name="State borders",
        style_function=lambda _: {"color": "#222222", "weight": 1.1, "fillOpacity": 0.0},
    ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_html = OUT_DIR / "aqi_days_above_100_county_leaflet.html"
    m.save(str(out_html))
    print(f"Saved {out_html}")
    return out_html


def main() -> None:
    build_leaflet_map()


if __name__ == "__main__":
    main()
