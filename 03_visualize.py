"""
03_visualize.py
Five data visualizations: days above AQI 100 by year (2020-2025).
1. Trend: mean days above AQI 100 per year (bar or line).
2–7. One map per year (2020–2025): IDW surface of days above AQI 100.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: basemap tiles (install contextily for web tiles)
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

DATA_DIR = Path(__file__).resolve().parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IDW_DIR = DATA_DIR / "idw"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
YEARS = list(range(2020, 2026))


def load_data():
    """Load processed site-year table and optional IDW grids."""
    path = PROCESSED_DIR / "days_above_aqi100_by_site_year.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 01_clean_aqi_days_above_100.py first. Missing {path}")
    df = pd.read_csv(path)
    return df


def viz1_trend_by_year(df: pd.DataFrame):
    """Visualization 1: Mean (or total) days above AQI 100 per year."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_year = df.groupby("Year")["days_above_100"].agg(["mean", "sum", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_year["Year"], by_year["mean"], color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean days above AQI 100 (per site)")
    ax.set_title("Days Above AQI 100 by Year (2020–2025)")
    ax.set_xticks(YEARS)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_trend_days_above_aqi100_by_year.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR / '01_trend_days_above_aqi100_by_year.png'}")


def viz2_to_5_maps(df: pd.DataFrame, years_to_plot=(2020, 2021, 2022, 2023, 2024, 2025)):
    """Visualizations 2–7: One map per year (2020–2025) using IDW grid if available, else points."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, year in enumerate(years_to_plot):
        fig, ax = plt.subplots(figsize=(10, 6))
        idw_path = IDW_DIR / f"idw_{year}.csv"
        if idw_path.exists():
            grid_df = pd.read_csv(idw_path)
            lon_vals = np.sort(grid_df["lon"].unique())
            lat_vals = np.sort(grid_df["lat"].unique())
            if len(lon_vals) > 1 and len(lat_vals) > 1:
                pivot = grid_df.pivot(index="lat", columns="lon", values="days_above_100")
                pivot = pivot.reindex(index=lat_vals, columns=lon_vals)
                Z = pivot.values
                im = ax.pcolormesh(lon_vals, lat_vals, Z, cmap="YlOrRd", shading="auto", vmin=0)
                plt.colorbar(im, ax=ax, label="Days above AQI 100")
            else:
                sc = ax.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["days_above_100"], cmap="YlOrRd", s=2)
                plt.colorbar(sc, ax=ax, label="Days above AQI 100")
        else:
            sub = df[df["Year"] == year].dropna(subset=["Longitude", "Latitude"])
            if sub.empty:
                ax.set_title(f"{year}: No data")
                fig.savefig(OUT_DIR / f"02_map_{year}.png", dpi=150)
                plt.close()
                continue
            sc = ax.scatter(sub["Longitude"], sub["Latitude"], c=sub["days_above_100"], cmap="YlOrRd", s=15, alpha=0.8)
            plt.colorbar(sc, ax=ax, label="Days above AQI 100")
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Days Above AQI 100 — {year}")
        if HAS_CTX:
            try:
                ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, zoom=4)
            except Exception:
                pass
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"02_map_{year}.png", dpi=150)
        plt.close()
        print(f"Saved {OUT_DIR / f'02_map_{year}.png'}")


def main():
    df = load_data()
    viz1_trend_by_year(df)
    # 1 trend + one map per year (2020–2025)
    viz2_to_5_maps(df, years_to_plot=(2020, 2021, 2022, 2023, 2024, 2025))
    print("Done. Check the 'outputs' folder.")


if __name__ == "__main__":
    main()
