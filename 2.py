import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox

# Sample customer data
df_customers = pd.DataFrame({
    'customer_id': [1, 2],
    'address': ['Address1', 'Address2'],
    'pincode': ['123456', '654321'],
    'latitude': [28.7041, 19.0760],  # Replace with actual latitude if available
    'longitude': [77.1025, 72.8777]   # Replace with actual longitude if available
})

# Convert DataFrame to GeoDataFrame
df_customers['geometry'] = df_customers.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
gdf_customers = gpd.GeoDataFrame(df_customers, geometry='geometry')

# Function to get count of police stations within a specified radius
def get_nearby_police_stations(gdf, radius=1000):
    police_station_counts = []
    for idx, row in gdf.iterrows():
        location = (row['latitude'], row['longitude'])
        # Search for police stations within the specified radius
        police_stations = ox.geometries.geometries_from_point(location, tags={'amenity': 'police'}, dist=radius)
        # Count the number of police stations found
        police_station_counts.append(len(police_stations))
    gdf['police_station_count'] = police_station_counts
    return gdf

# Apply function to get police stations count within 1 km (1000 meters)
gdf_customers = get_nearby_police_stations(gdf_customers, radius=1000)

# Display results
print(gdf_customers[['customer_id', 'police_station_count']])
