import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "noaa-nclimgrid"
RAW_DATA.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.ncei.noaa.gov/data/nclimgrid-daily/access/grids/"
YEARS = range(1951, 2026)
MONTHS = range(1, 13)
MAX_WORKERS = 8
TIMEOUT = 60
DELAY = 0.1

def download_file(year, month):
    fname = f"ncdd-{year}{month:02d}-grd-scaled.nc"
    url = f"{BASE_URL}/{year}/{fname}"
    path = RAW_DATA / fname
    if path.exists():
        print(f"Skipping {fname} (already exists)")
        return fname, True
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 200:
            with open(path, 'wb') as f:
                f.write(resp.content)
            print(f"Downloaded {fname}")
            return fname, True
        else:
            print(f"Failed {fname}: HTTP {resp.status_code}")
            return fname, False
    except Exception as e:
        print(f"Error downloading {fname}: {e}")
        return fname, False


tasks = [(y, m) for y in YEARS for m in MONTHS]

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(download_file, y, m): (y, m) for y, m in tasks}

    for future in tqdm(as_completed(futures), total=len(tasks), desc="Downloading"):
        fname, success = future.result()
        results.append((fname, success))
        time.sleep(DELAY)
