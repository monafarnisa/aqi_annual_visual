"""
Optional: Download EPA AirData daily CSVs into data/raw/.
Use this to fetch the ~25 MB files per year instead of committing them to Git.

EPA pre-generated files: https://aqs.epa.gov/aqsweb/airdata/download_files.html
ZIPs are named e.g. daily_44201_2020.zip (PM2.5). This script downloads the ZIP,
extracts the CSV, and saves as data/raw/daily_44201_YYYY.csv.
"""

import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
BASE_URL = "https://aqs.epa.gov/aqsweb/airdata"
YEARS = list(range(2020, 2026))
# Parameter: 44201 = PM2.5 (daily). Use 42401 for Ozone if needed.
PARAMETER = "44201"


def download_year(year: int) -> bool:
    """Download one year's daily file (ZIP), extract CSV to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = f"daily_{PARAMETER}_{year}.csv"
    csv_path = RAW_DIR / csv_name
    if csv_path.exists():
        print(f"Already have {csv_path}, skipping.")
        return True
    zip_name = f"daily_{PARAMETER}_{year}.zip"
    url = f"{BASE_URL}/{zip_name}"
    if not requests:
        print("Install 'requests' to use auto-download: pip install requests")
        return False
    print(f"Downloading {url} ...")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        print(f"Manually download from {BASE_URL} and extract {csv_name} into {RAW_DIR}")
        return False
    zip_path = RAW_DIR / zip_name
    zip_path.write_bytes(r.content)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # ZIP usually contains a single CSV with the same base name
            for name in z.namelist():
                if name.endswith(".csv"):
                    z.extract(name, RAW_DIR)
                    extracted = RAW_DIR / name
                    if extracted != csv_path:
                        extracted.rename(csv_path)
                    break
    finally:
        if zip_path.exists():
            zip_path.unlink()
    print(f"Saved {csv_path}")
    return True


def main():
    print(f"Raw data directory: {RAW_DIR}")
    for year in YEARS:
        download_year(year)
    print("Done. Run: python 01_clean_aqi_days_above_100.py")


if __name__ == "__main__":
    main()
