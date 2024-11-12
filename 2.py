import osmnx as ox
import pandas as pd
import requests

# Assuming `df` is your DataFrame with columns `pincode`, `latitude`, and `longitude`
session = requests.Session()
session.verify = False
ox.config(log_console=True, use_cache=True, requests_kwargs={'verify': False})

# Function to get amenities (like hospitals, police stations) for a location
def get_amenities(lat, lon, radius=1000):
    # Define the point around which to search
    point = (lat, lon)
    
    # Get amenities around the point
    try:
        gdf = ox.geometries_from_point(point, tags={'amenity': True}, dist=radius)
        hospitals = gdf[gdf['amenity'] == 'hospital']
        police_stations = gdf[gdf['amenity'] == 'police']
    except Exception as e:
        print(f"Error retrieving amenities for location ({lat}, {lon}): {e}")
        hospitals = pd.DataFrame()  # Empty DataFrame if there's an error
        police_stations = pd.DataFrame()
        
    return hospitals, police_stations

# Process each row in your DataFrame
hospital_list = []
police_station_list = []

for index, row in df.iterrows():
    hospitals, police_stations = get_amenities(row['latitude'], row['longitude'])
    hospital_list.append(hospitals)
    police_station_list.append(police_stations)

# Add the lists of amenities as new columns in your DataFrame
df['hospitals'] = hospital_list
df['police_stations'] = police_station_list

# View the result
print(df.head())
