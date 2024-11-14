import requests
from geopy.distance import geodesic

# Function to query Overpass API for police stations around a given city (lat, lon)
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

# Example of how to integrate with your city data
def check_within_10km(city_lat, city_lon):
    # Get police stations within 10 km of the city
    police_stations = get_police_stations_nearby(city_lat, city_lon, radius=10000)
    
    # If there are police stations nearby, return True
    if police_stations:
        return True
    return False

# Sample cities DataFrame
cities_df = pd.read_csv(r'/content/city_ps_check_4.csv')
# Check proximity for each city
cities_df['has_police_station_within_10km'] = cities_df.apply(
    lambda row: check_within_10km(row['latitude'], row['longitude']), axis=1
)

# Save the updated DataFrame to CSV
cities_df.to_csv('cities_with_police_station_proximity_part_3.csv', index=False)

# Display the updated DataFrame
print(cities_df)
#6:10 pm
