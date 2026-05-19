# project/data/download_gdelt_202301_202304.py
import os
import requests
from datetime import datetime, timedelta

RAW_DIR = "./raw/gdelt"
os.makedirs(RAW_DIR, exist_ok=True)

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 4, 30)

current_date = start_date
while current_date <= end_date:
    yyyymmdd = current_date.strftime("%Y%m%d")
    url = f"http://data.gdeltproject.org/events/{yyyymmdd}.export.CSV.zip"
    local_path = os.path.join(RAW_DIR, f"{yyyymmdd}.export.CSV.zip")

    if not os.path.exists(local_path):
        try:
            print(f"Downloading {url} ...")
            r = requests.get(url)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"{local_path} already exists, skipping.")

    current_date += timedelta(days=1)

print("GDELT zip 下载完成，存放在 data/raw/gdelt/")