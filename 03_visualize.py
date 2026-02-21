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
from branca.element import Element
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "leaflet"
YEARS = list(range(2020, 2026))
COUNTIES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_5m.zip"
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_5m.zip"
DROP_STUSPS = {"AS", "GU", "MP", "PR", "VI"}
LEAFLET_CMAP_NAME = "turbo"


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


def leaflet_colormap(vmin: float, vmax: float, cmap_name: str = LEAFLET_CMAP_NAME) -> bcm.LinearColormap:
    """Build a Leaflet legend from a matplotlib colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    stops = [mcolors.to_hex(cmap(i)) for i in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    return bcm.LinearColormap(colors=stops, vmin=vmin, vmax=vmax)


def _year_csv_path(year: int) -> Path:
    return PROCESSED_DIR / f"us_days_above_100_{year}.csv"


def _load_county_year_values(year: int) -> dict[str, float]:
    path = _year_csv_path(year)
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    needed = {"state_code", "county_code", "days_above_100"}
    if not needed.issubset(df.columns):
        return {}

    county = (
        df.groupby(["state_code", "county_code"], as_index=False)["days_above_100"]
        .max()
        .rename(columns={"days_above_100": "county_days_above_100"})
    )
    county["GEOID"] = (
        county["state_code"].astype("Int64").astype(str).str.zfill(2)
        + county["county_code"].astype("Int64").astype(str).str.zfill(3)
    )
    return dict(zip(county["GEOID"], county["county_days_above_100"].astype(float)))


def _global_county_max() -> float:
    max_val = 0.0
    for year in YEARS:
        values = _load_county_year_values(year)
        if values:
            max_val = max(max_val, max(values.values()))
    return max_val if max_val > 0 else 1.0


def _style_fn_factory(
    year_cols: list[str],
    cmap: bcm.LinearColormap,
) -> callable:
    def style_fn(feature: dict) -> dict:
        props = feature["properties"]
        vals = [props.get(col) for col in year_cols]
        has_data = any(v is not None for v in vals)
        total = float(sum(v for v in vals if v is not None)) if has_data else 0.0
        if (not has_data) or total <= 0:
            return {
                "fillColor": "#d9d9d9",
                "fillOpacity": 0.20,
                "color": "#666666",
                "weight": 0.25,
            }
        return {
            "fillColor": cmap(total),
            "fillOpacity": 0.78,
            "color": "#4a4a4a",
            "weight": 0.25,
        }

    return style_fn


def _global_county_sum_max() -> float:
    per_year = {year: _load_county_year_values(year) for year in YEARS}
    all_geoids: set[str] = set()
    for d in per_year.values():
        all_geoids.update(d.keys())

    max_sum = 0.0
    for geoid in all_geoids:
        s = sum(per_year[year].get(geoid, 0.0) for year in YEARS)
        max_sum = max(max_sum, float(s))
    return max_sum if max_sum > 0 else 1.0


def build_leaflet_map() -> Path:
    counties = _load_shapefile_url(COUNTIES_URL)
    states = _load_shapefile_url(STATES_URL)
    states = states[~states["STUSPS"].isin(DROP_STUSPS)].copy()
    counties = counties[counties["STATEFP"].isin(states["STATEFP"])].copy()

    max_val = _global_county_sum_max()
    cmap = leaflet_colormap(0, max_val, cmap_name=LEAFLET_CMAP_NAME)
    cmap.caption = "County days above AQI 100 (sum across selected years)"

    m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

    year_cols: list[str] = []
    for year in YEARS:
        value_by_geoid = _load_county_year_values(year)
        col = f"y{year}"
        year_cols.append(col)
        counties[col] = counties["GEOID"].map(value_by_geoid).astype(float)
        counties[f"label_{year}"] = counties[col].map(
            lambda v: f"{int(round(v))}" if pd.notna(v) else "No monitor data"
        )

    year_layers: dict[int, folium.FeatureGroup] = {}
    for year in YEARS:
        fg = folium.FeatureGroup(name=str(year), show=(year == YEARS[-1]))
        fg.add_to(m)
        year_layers[year] = fg

    tooltip_fields = ["NAME"] + [f"label_{year}" for year in YEARS]
    tooltip_aliases = ["County"] + [f"Days above AQI 100 ({year})" for year in YEARS]
    county_geojson = folium.GeoJson(
        data=counties.to_json(),
        name="County totals",
        control=False,
        style_function=_style_fn_factory([f"y{YEARS[-1]}"], cmap),
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True,
        ),
    )
    county_geojson.add_to(m)

    # State borders as a crisp overlay.
    folium.GeoJson(
        data=states.to_json(),
        name="State borders",
        style_function=lambda _: {"color": "#222222", "weight": 1.1, "fillOpacity": 0.0},
    ).add_to(m)

    cmap.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    cmap_obj = matplotlib.colormaps[LEAFLET_CMAP_NAME]
    stops = [mcolors.to_hex(cmap_obj(i)) for i in np.linspace(0, 1, 11)]
    layer_items = ",\n".join(
        [f'      "{year}": {year_layers[year].get_name()}' for year in YEARS]
    )
    year_cols_js = ", ".join([f'"{year}": "y{year}"' for year in YEARS])
    js = f"""
<script>
(function() {{
  var map = {m.get_name()};
  var countyLayer = {county_geojson.get_name()};
  var yearLayers = {{
{layer_items}
  }};
  var yearCols = {{{year_cols_js}}};
  var vmin = 0.0;
  var vmax = {max_val:.6f};
  var colorStops = {stops};

  function colorForValue(v) {{
    if (v <= 0) return "#d9d9d9";
    if (vmax <= vmin) return colorStops[colorStops.length - 1];
    var t = (v - vmin) / (vmax - vmin);
    t = Math.max(0, Math.min(1, t));
    var idx = Math.floor(t * (colorStops.length - 1));
    return colorStops[idx];
  }}

  function activeYears() {{
    var years = [];
    Object.keys(yearLayers).forEach(function(y) {{
      if (map.hasLayer(yearLayers[y])) years.push(y);
    }});
    return years;
  }}

  function styleFromActive(feature, years) {{
    var props = feature.properties || {{}};
    var hasData = false;
    var total = 0;
    years.forEach(function(y) {{
      var col = yearCols[y];
      var val = props[col];
      if (val !== null && val !== undefined) {{
        hasData = true;
        total += Number(val);
      }}
    }});
    if (!hasData || total <= 0) {{
      return {{
        fillColor: "#d9d9d9",
        fillOpacity: 0.20,
        color: "#666666",
        weight: 0.25
      }};
    }}
    return {{
      fillColor: colorForValue(total),
      fillOpacity: 0.78,
      color: "#4a4a4a",
      weight: 0.25
    }};
  }}

  function refreshCountyStyles() {{
    var years = activeYears();
    countyLayer.setStyle(function(feature) {{
      return styleFromActive(feature, years);
    }});
  }}

  map.on("overlayadd", refreshCountyStyles);
  map.on("overlayremove", refreshCountyStyles);
  refreshCountyStyles();
}})();
</script>
"""
    m.get_root().html.add_child(Element(js))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_html = OUT_DIR / "aqi_days_above_100_county_leaflet.html"
    m.save(str(out_html))
    print(f"Saved {out_html}")
    return out_html


def main() -> None:
    build_leaflet_map()


if __name__ == "__main__":
    main()
