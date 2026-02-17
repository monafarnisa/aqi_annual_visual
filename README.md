# aqi_annual_visual

Calculate **days above AQI 100** per year (2020–2025) and produce **five data visualizations**. Pipeline is in **Python** (converted from R scripts used in [wwri/domains](https://github.com/wwri/domains) air quality workflow).

## What this repo does

1. **Clean** daily AQI data and aggregate to **days above AQI 100** per monitor per year.
2. **Interpolate** with **Inverse Distance Weighted (IDW)** to create continuous surfaces.
3. **Visualize**:  
   - **Viz 1:** Trend chart — mean days above AQI 100 per year (2020–2025).  
   - **Viz 2–5:** One map per year (2020, 2021, 2022, 2023) showing the IDW surface (or monitor points if no IDW).

## Setup

```bash
cd aqi_annual_visual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

- **EPA AirData** daily files: [Download Daily Data](https://www.epa.gov/outdoor-air-quality-data/download-daily-data) or [AirData download files](https://aqs.epa.gov/aqsweb/airdata/download_files.html).
- Place CSVs in `data/raw/` with names like:
  - `daily_44201_2020.csv` (PM2.5), `daily_44201_2021.csv`, … for each year, and/or
  - `daily_42401_YYYY.csv` for ozone.
- If no files are present, the cleaning script generates **synthetic example data** so you can run the full pipeline and see the visualizations.

## Run the pipeline

```bash
# 1. Clean and compute days above AQI 100 per site per year
python 01_clean_aqi_days_above_100.py

# 2. IDW interpolation (grids per year)
python 02_aqi_100_idw.py

# 3. Create the five visualizations
python 03_visualize.py
```

Outputs:

- **Processed data:** `data/processed/days_above_aqi100_by_site_year.csv`
- **IDW grids:** `data/idw/idw_2020.csv`, … `idw_2025.csv`
- **Figures:** `outputs/01_trend_days_above_aqi100_by_year.png`, `outputs/02_map_2020.png`, … `02_map_2023.png`

## Scripts (R → Python)

| R (wwri/domains) | Python (this repo) |
|------------------|---------------------|
| `01_us_calculate_days_above_aqi.R` | `01_clean_aqi_days_above_100.py` |
| `02_aqi_100_idw.R` | `02_aqi_100_idw.py` |
| — | `03_visualize.py` (five visualizations) |

## Adding 2024 and 2025 to the maps

In `03_visualize.py`, extend the `years_to_plot` tuple in `viz2_to_5_maps()` (e.g. add 2024 and 2025) and run again to generate additional map figures.

## Create this repo on GitHub

1. On GitHub: **New repository** → name: `aqi_annual_visual` (public, no README/license if you already have them locally).
2. Locally, from the project folder:

```bash
cd ~/aqi_annual_visual
git init
git add .
git commit -m "Initial commit: AQI days above 100 pipeline and five visualizations"
git branch -M main
git remote add origin https://github.com/monafarnisa/aqi_annual_visual.git
git push -u origin main
```

## License

Use and adapt as needed; consider aligning with the wwri/domains repo if you pull logic from there.
