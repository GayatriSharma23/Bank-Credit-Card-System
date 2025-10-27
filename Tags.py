import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_lat_lon_from_indiamapia(pincode):
    """Scrape IndiaMapia for place, district, state, lat, lon"""
    base_url = f"https://indiamapia.com/{pincode}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    info = {
        "pincode": pincode,
        "place": None,
        "tehsil": None,
        "district": None,
        "state": None,
        "latitude": None,
        "longitude": None
    }

    try:
        res = requests.get(base_url, headers=headers, timeout=10)
        if res.status_code != 200:
            print(f"❌ {pincode} page not found")
            return info

        soup = BeautifulSoup(res.text, "html.parser")

        # find main data table
        table = soup.find("table")
        if table:
            for row in table.find_all("tr"):
                cols = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cols) == 2:
                    key, val = cols
                    if "Place" in key:
                        info["place"] = val
                    elif "Tehsil" in key:
                        info["tehsil"] = val
                    elif "District" in key:
                        info["district"] = val
                    elif "State" in key:
                        info["state"] = val

        # Find link to place page
        link_tag = soup.find("a", href=True, text=True)
        if link_tag:
            place_url = "https://indiamapia.com" + link_tag["href"]
            place_res = requests.get(place_url, headers=headers, timeout=10)
            if place_res.status_code == 200:
                place_soup = BeautifulSoup(place_res.text, "html.parser")
                text = place_soup.get_text(" ", strip=True)

                # Extract coordinates more robustly
                import re
                match = re.search(r"Latitude[: ]+([0-9.\-]+)\s+Longitude[: ]+([0-9.\-]+)", text)
                if match:
                    info["latitude"] = float(match.group(1))
                    info["longitude"] = float(match.group(2))

        return info

    except Exception as e:
        print(f"⚠️ Error for {pincode}: {e}")
        return info


# ------------- MAIN EXECUTION -------------
input_file = "pincodes.csv"
output_file = "pincodes_with_latlon.csv"

df = pd.read_csv(input_file)
if "pincode" not in df.columns:
    raise ValueError("CSV must contain a 'pincode' column")

results = []
for pin in df["pincode"]:
    pin = str(pin).strip()
    print(f"🔍 Fetching data for {pin}...")
    data = get_lat_lon_from_indiamapia(pin)
    results.append(data)
    time.sleep(2)  # avoid blocking

final_df = pd.DataFrame(results)
final_df.to_csv(output_file, index=False)
print(f"\n✅ Saved results to {output_file}")
