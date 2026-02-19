"""
03_visualize.py
Interactive Leaflet visualization with county borders and year layers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "leaflet"
YEARS = list(range(2020, 2026))
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip"
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip"
DROP_STUSPS = {"AS", "GU", "MP", "PR", "VI"}


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


def load_processed_county_values() -> pd.DataFrame:
    in_csv = PROCESSED_DIR / "days_above_aqi100_by_site_year.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing processed file: {in_csv}")

    df = pd.read_csv(in_csv)
    for col, width in (("state_code", 2), ("county_code", 3)):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).astype(str).str.zfill(width)
    df["GEOID"] = df["state_code"] + df["county_code"]

    # County-level summary for mapping.
    county_year = (
        df.groupby(["Year", "GEOID"], as_index=False)["days_above_100"]
        .mean()
        .rename(columns={"days_above_100": "days_above_100_mean"})
    )
    return county_year


def build_leaflet_map() -> Path:
    counties = _load_shapefile_url(COUNTIES_URL)
    states = _load_shapefile_url(STATES_URL)
    states = states[~states["STUSPS"].isin(DROP_STUSPS)].copy()
    counties = counties[counties["STATEFP"].isin(states["STATEFP"])].copy()

    county_year = load_processed_county_values()
    max_val = float(county_year["days_above_100_mean"].max()) if not county_year.empty else 1.0
    cmap = bcm.linear.YlOrRd_09.scale(0, max_val)
    cmap.caption = "Mean days above AQI 100 (county)"

    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

    for year in YEARS:
        vals = county_year[county_year["Year"] == year][["GEOID", "days_above_100_mean"]]
        joined = counties.merge(vals, how="left", left_on="GEOID", right_on="GEOID")

        fg = folium.FeatureGroup(name=f"{year}", show=(year == YEARS[-1]))

        def style_fn(feature):
            v = feature["properties"].get("days_above_100_mean")
            color = "#f2f2f2" if v is None else cmap(v)
            return {
                "fillColor": color,
                "fillOpacity": 0.75,
                "color": "#8c8c8c",
                "weight": 0.35,  # county borders
            }

        tooltip = folium.GeoJsonTooltip(
            fields=["NAME", "STATE_NAME", "days_above_100_mean"],
            aliases=["County", "State", "Days > 100 (mean)"],
            localize=True,
            sticky=False,
            labels=True,
        )

        folium.GeoJson(
            data=joined.to_json(),
            style_function=style_fn,
            tooltip=tooltip,
            name=f"Counties {year}",
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
