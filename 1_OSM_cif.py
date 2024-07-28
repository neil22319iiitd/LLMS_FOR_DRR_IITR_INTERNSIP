import requests
import pandas as pd
from geopy.geocoders import Nominatim
import time

# Define the area of interest (Kerala, India)
aoi = "Kerala, India"

# Initialize Nominatim API
geolocator = Nominatim(user_agent="cif_locator")

# Get the location details of the AOI
location = geolocator.geocode(aoi)
if location:
    lat, lon = location.latitude, location.longitude
    print(f"Latitude and Longitude of Kerala: {lat}, {lon}")
else:
    raise ValueError("AOI not found")

# Define CIF types to search
cif_types = ['hospital', 'fire_station']

# Function to retrieve CIFs using Overpass API
def get_cifs(cif_type, lat, lon, radius=50000):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node(around:{radius},{lat},{lon})["amenity"="{cif_type}"];
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    cifs = []
    for element in data['elements']:
        cif = {
            'name': element['tags'].get('name', 'N/A'),
            'type': cif_type,
            'lat': element['lat'],
            'lon': element['lon'],
            'address': get_address(element['lat'], element['lon'])
        }
        cifs.append(cif)
    return cifs

# Function to get address with retry logic
def get_address(lat, lon, retries=5, delay=2):
    for i in range(retries):
        try:
            return geolocator.reverse((lat, lon), timeout=10, language='en').address
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    return 'Address lookup failed'

# Retrieve CIFs for each type
all_cifs = []
for cif_type in cif_types:
    cifs = get_cifs(cif_type, lat, lon)
    all_cifs.extend(cifs)

# Convert to DataFrame
df_cifs = pd.DataFrame(all_cifs)
print(df_cifs)

# Save to CSV
df_cifs.to_csv('kerala_cifs.csv', index=False)
