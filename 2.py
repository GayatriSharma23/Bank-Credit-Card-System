import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Assuming df is your DataFrame with columns 'pincode', 'latitude', and 'longitude'
# Set up `osmnx` for your area to get all amenities initially
session = requests.Session()
session.verify = False
ox.config(log_console=True, use_cache=True, requests_kwargs={'verify': False})

# Retrieve amenities for a broader region
gdf = ox.geometries_from_place("Jalandhar, Punjab, India", tags={'amenity': True})
hospitals = gdf[gdf['amenity'] == 'hospital']
police_stations = gdf[gdf['amenity'] == 'police']

# Convert amenities to GeoDataFrames with appropriate CRS
hospitals = hospitals.set_geometry(hospitals.centroid).to_crs("EPSG:4326")
police_stations = police_stations.set_geometry(police_stations.centroid).to_crs("EPSG:4326")

# Convert your DataFrame `df` to a GeoDataFrame
df['geometry'] = df.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
gdf_locations = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Define a proximity threshold in meters (e.g., 1000 meters)
proximity_threshold = 1000

# Function to check if amenities are within the threshold distance
def check_proximity(location, amenities):
    # Calculate distances and check if any amenities are within the threshold
    return amenities.distance(location).min() <= (proximity_threshold / 1000 / 111)  # Approx conversion to degrees

# Check proximity for each location
df['near_hospital'] = gdf_locations['geometry'].apply(lambda loc: check_proximity(loc, hospitals))
df['near_police_station'] = gdf_locations['geometry'].apply(lambda loc: check_proximity(loc, police_stations))

# Display the DataFrame with proximity results
print(df[['pincode', 'latitude', 'longitude', 'near_hospital', 'near_police_station']])

