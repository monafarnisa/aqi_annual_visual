# aqi_annual_visual

Script to interpolate **days above AQI 100** per year (2020–2025) from sesnsor locations across the United States and produce **five data visualizations**. Pipeline is in **Python**. 

## What this repo does

1. **Clean** daily AQI data and aggregate to **days above AQI 100** per monitor per year.
2. **Interpolate** with **Inverse Distance Weighted (IDW)** to create continuous surfaces.
3. **Visualize**:  
   - ** One map per year (2020–2025) showing the IDW surface (or monitor points if no IDW).
   - ** An interactive html leaflet to zoom into county level and identify days above AQI > 100. 

## Setup

```bash
cd aqi_annual_visual
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

- **EPA AirData** daily files: [Download Daily AQI Data by County](https://aqs.epa.gov/aqsweb/airdata/download_files.html#Daily) for each year from 2020 - 2025.
- Place CSVs in `data/raw/` with names like:

### Including your 5 CSV files (~25 MB each)

You have **five CSVs at ~25 MB each** (~125 MB total). Best options:

| Approach | When to use |
|----------|-------------|
| **Keep raw data out of Git (recommended)** | Repo stays small; others run the pipeline with their own download. `data/raw/` is already in `.gitignore`. Download EPA ZIPs, unzip into `data/raw/`, and run the pipeline. Use the optional script: `python scripts/download_raw_data.py` (see below). |
| **Git LFS** | You want the exact 5 files versioned and shared via the repo. Run `git lfs install`, then `git lfs track "data/raw/*.csv"`, add and commit. Note: GitHub LFS has bandwidth/storage limits on free plans. |
| **DVC (Data Version Control)** | Best for large data and reproducibility: data lives in remote storage (S3, GCS, or a shared drive); the repo stores only small pointer files. See [dvc.org](https://dvc.org). |
| **External storage + script** | Store the 5 CSVs in Google Drive, S3, or similar; add a small script or README instructions to download them into `data/raw/` before running the pipeline. |

**Recommended:** Keep `data/raw/` ignored. Use the download script or the EPA links above to populate `data/raw/` once per machine (or use DVC if you need versioned, shareable data).

## Run the pipeline

```bash
# Optional: download EPA daily CSVs into data/raw/ (otherwise place your 5 CSVs there)
python scripts/download_raw_data.py

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
- **Figures:** `outputs/01_trend_days_above_aqi100_by_year.png`, `outputs/02_map_2020.png`, … `02_map_2025.png`

## Scripts (R → Python)

| R (wwri/domains) | Python (this repo) |
|------------------|---------------------|
| `01_us_calculate_days_above_aqi.R` | `01_clean_aqi_days_above_100.py` |
| `02_aqi_100_idw.R` | `02_aqi_100_idw.py` |
| — | `03_visualize.py` (five visualizations) |

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
