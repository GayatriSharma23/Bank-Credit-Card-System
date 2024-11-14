import requests
import pandas as pd

# Function to query Overpass API for police stations around a given pincode (lat, lon)
def get_police_stations_nearby(lat, lon, radius=10000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="police"](around:{radius},{lat},{lon});
      way["amenity"="police"](around:{radius},{lat},{lon});
      relation["amenity"="police"](around:{radius},{lat},{lon});
    );
    out body;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Parse data to get lat/lon of each police station
    police_stations = []
    for element in data['elements']:
        if 'lat' in element and 'lon' in element:
            police_stations.append({
                'name': element.get('tags', {}).get('name', 'Unnamed'),
                'latitude': element['lat'],
                'longitude': element['lon']
            })
    
    return police_stations

# Example of how to integrate with pincode data
def check_within_10km(pincode_lat, pincode_lon):
    # Get police stations within 10 km of the pincode
    police_stations = get_police_stations_nearby(pincode_lat, pincode_lon, radius=10000)
    
    # If there are police stations nearby, return True
    return bool(police_stations)

# Sample pincodes DataFrame with latitude and longitude for each pincode
pincodes_df = pd.read_csv(r'/content/pincode_lat_lon.csv')

# Check proximity for each pincode
pincodes_df['has_police_station_within_10km'] = pincodes_df.apply(
    lambda row: check_within_10km(row['latitude'], row['longitude']), axis=1
)

# Save the updated DataFrame to CSV
pincodes_df.to_csv('pincodes_with_police_station_proximity.csv', index=False)

# Display the updated DataFrame
print(pincodes_df)
