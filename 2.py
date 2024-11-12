import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Assuming `df` is your DataFrame with columns 'pincode', 'latitude', and 'longitude'
session = requests.Session()
session.verify = False
ox.config(log_console=True, use_cache=True, requests_kwargs={'verify': False})

# Function to retrieve amenities around a pincode location
def get_amenities_by_pincode(lat, lon, radius=1000):
    # Define the point around which to search (1 km radius)
    point = (lat, lon)
    
    try:
        # Retrieve amenities around this point
        gdf = ox.geometries_from_point(point, tags={'amenity': True}, dist=radius)
        
        # Filter for specific amenities
        hospitals = gdf[gdf['amenity'] == 'hospital']
        police_stations = gdf[gdf['amenity'] == 'police']
        
        # Return True if any hospitals or police stations are found
        has_hospital = not hospitals.empty
        has_police_station = not police_stations.empty
    except Exception as e:
        print(f"Error retrieving amenities for ({lat}, {lon}): {e}")
        has_hospital = False
        has_police_station = False
    
    return has_hospital, has_police_station

# Apply function to each row in the DataFrame
df['has_hospital'] = False
df['has_police_station'] = False

for index, row in df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    has_hospital, has_police_station = get_amenities_by_pincode(lat, lon)
    
    # Update DataFrame
    df.at[index, 'has_hospital'] = has_hospital
    df.at[index, 'has_police_station'] = has_police_station

# Display the updated DataFrame with proximity results
print(df[['pincode', 'latitude', 'longitude', 'has_hospital', 'has_police_station']])
