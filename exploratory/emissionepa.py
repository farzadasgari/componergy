import os
import requests
import zipfile
from datetime import datetime

emissions = {
    "CO": "42101",
    "Pb": "14129",
    "NO2": "42602",
    "Ozone": "44201",
    "PM10": "81102",
    "PM2.5": "88502",
    "SO2": "42401"
}

base_url = "https://aqs.epa.gov/aqsweb/airdata/daily_{}_{}.zip"

zip_dir = "../dataset/emission/zips"
csv_dir = "../dataset/emission/csvs"
os.makedirs(zip_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

start_year = 1980
current_year = datetime.now().year

for year in range(start_year, current_year + 1):
    for emission, code in emissions.items():
        url = base_url.format(code, year)
        zip_file = os.path.join(zip_dir, f"daily_{emission}_{year}.zip")
        extracted_csv = os.path.join(csv_dir, f"{emission}_{year}.csv")
        print(f"Downloading {emission} data for {year} from {url}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(zip_file, "wb") as file: file.write(response.content)
            print(f"Saved {emission} data to {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as z:
                csv_name_in_zip = z.namelist()[0]
                z.extract(csv_name_in_zip, csv_dir)
                os.rename(
                    os.path.join(csv_dir, csv_name_in_zip),
                    extracted_csv
                )
            print(f"Extracted and saved {emission} data to {extracted_csv}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {emission} data for {year}: {e}")
        except zipfile.BadZipFile as e:
            print(f"Failed to extract {zip_file}: {e}")
