import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_lat_lon_from_indiamapia(pincode):
    """Fetch place, district, state, latitude, longitude from IndiaMapia"""
    url = f"https://indiamapia.com/{pincode}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    info = {"pincode": pincode, "place": None, "district": None, "state": None, "latitude": None, "longitude": None}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"❌ {pincode} -> Page not found ({response.status_code})")
            return info

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract info from the main table
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True).lower()
                    value = cols[1].get_text(strip=True)
                    if "place" in key:
                        info["place"] = value
                    elif "district" in key:
                        info["district"] = value
                    elif "state" in key:
                        info["state"] = value

        # Go to place page to find lat/long
        link = soup.find("a", href=True, text=True)
        if link:
            place_url = "https://indiamapia.com" + link['href']
            place_page = requests.get(place_url, headers=headers, timeout=10)
            if place_page.status_code == 200:
                text = place_page.text
                # Latitude and longitude sometimes appear as e.g. Latitude: 12.8658 Longitude: 74.8574
                if "Latitude" in text and "Longitude" in text:
                    try:
                        lat_str = text.split("Latitude:")[1].split("Longitude:")[0].strip().split()[0]
                        lon_str = text.split("Longitude:")[1].split()[0]
                        info["latitude"] = float(lat_str)
                        info["longitude"] = float(lon_str)
                    except Exception:
                        pass
        return info

    except Exception as e:
        print(f"⚠️ Error fetching {pincode}: {e}")
        return info


# ---------- MAIN SCRIPT ----------
# 1️⃣ Read input CSV file
input_file = "pincodes.csv"   # your input CSV (must have column: pincode)
output_file = "pincodes_with_latlon.csv"

df = pd.read_csv(input_file)
if 'pincode' not in df.columns:
    raise ValueError("Input CSV must have a column named 'pincode'")

results = []
for pin in df['pincode']:
    pin = str(pin).strip()
    print(f"🔍 Fetching details for {pin}...")
    data = get_lat_lon_from_indiamapia(pin)
    results.append(data)
    time.sleep(2)  # avoid rate limiting

# 2️⃣ Combine results and save
result_df = pd.DataFrame(results)
merged_df = df.merge(result_df, on='pincode', how='left')
merged_df.to_csv(output_file, index=False)
print(f"\n✅ Saved results to {output_file}")
