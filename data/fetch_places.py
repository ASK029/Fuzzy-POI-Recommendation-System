import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

# URLs
loc_url = "https://travel-advisor.p.rapidapi.com/locations/search"
endpoint = "https://travel-advisor.p.rapidapi.com/attractions/list"
restaurant_endpoint = "https://travel-advisor.p.rapidapi.com/restaurants/list"

def get_headers():
    from utils.config import RAPIDAPI_KEY

    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
    }

def find_location_id(city: str):
    import requests

    loc_params = {
        "query": city,
        "limit": "1"
    }
    loc_response = requests.get(loc_url, headers=get_headers(), params=loc_params)
    loc_data = loc_response.json()
    
    try:
        location_id = loc_data['data'][0]['result_object']['location_id']
        return location_id
    except (KeyError, IndexError):
        print("City not found or API error.")
        return None

# just fetching attractions now

# stuttgart city id : 187291
#damascus city id: 294011
def fetch_attractions(location_id: str):
    import requests

    query_params = {
        "location_id": location_id,
        "limit": "30"
    }
    response = requests.get(endpoint, headers=get_headers(), params=query_params)
    data = response.json()
    return data.get("data", [])

def fetch_restaurants(location_id: str):
    import requests

    query_params = {
        "location_id": location_id,
        "limit": "30"
    }
    response = requests.get(restaurant_endpoint, headers=get_headers(), params=query_params)
    data = response.json()
    return data.get("data", [])

def save_data(city: str, places, place_type="attractions"):

    folder_path = os.path.join(os.path.dirname(__file__), "APIs", city.lower().replace(" ", "_"), place_type)
    os.makedirs(folder_path, exist_ok=True)

    for index, location in enumerate(places):
        filename = os.path.join(folder_path, f"{place_type}_{index+1}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(location, f, ensure_ascii=False, indent=2)

        print(f"Data for {place_type} {index+1} saved to {filename}.")

def _place_file_sort_key(filename: str):
    name, _ = os.path.splitext(filename)
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return name

def load_places_from_api_folder(folder_path: str):
    places = []

    for filename in sorted(os.listdir(folder_path), key=_place_file_sort_key):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON file {file_path}: {e}")
            continue

        if isinstance(data, list):
            places.extend(data)
        else:
            places.append(data)

    return places

def store_city_api_data_as_raw(city: str, base_api_path=None, raw_path=None):
    base_api_path = base_api_path or os.path.join(os.path.dirname(__file__), "APIs")
    raw_path = raw_path or os.path.join(os.path.dirname(__file__), "raw")
    city_key = city.lower().replace(" ", "_")
    city_api_path = os.path.join(base_api_path, city_key)

    if not os.path.isdir(city_api_path):
        exact_city_api_path = os.path.join(base_api_path, city)
        if os.path.isdir(exact_city_api_path):
            city_key = city
            city_api_path = exact_city_api_path
        else:
            raise FileNotFoundError(f"No API data found for city '{city}' at {city_api_path}")

    city_raw_path = os.path.join(raw_path, city_key)
    os.makedirs(city_raw_path, exist_ok=True)

    saved_files = {}
    for place_type in ["attractions", "restaurants"]:
        source_folder = os.path.join(city_api_path, place_type)
        if not os.path.isdir(source_folder):
            print(f"Warning: Folder not found - {source_folder}")
            continue

        places = load_places_from_api_folder(source_folder)
        output_file = os.path.join(city_raw_path, f"{place_type}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(places, f, ensure_ascii=False, indent=2)

        saved_files[place_type] = output_file
        print(f"Saved {len(places)} {place_type} for {city_key} to {output_file}.")

    return saved_files

def store_all_api_data_as_raw(base_api_path=None, raw_path=None):
    base_api_path = base_api_path or os.path.join(os.path.dirname(__file__), "APIs")
    raw_path = raw_path or os.path.join(os.path.dirname(__file__), "raw")
    saved_files = {}

    for city in sorted(os.listdir(base_api_path)):
        city_path = os.path.join(base_api_path, city)
        if not os.path.isdir(city_path):
            continue

        saved_files[city] = store_city_api_data_as_raw(
            city,
            base_api_path=base_api_path,
            raw_path=raw_path
        )

    return saved_files

def fetch_places(city: str):
    base_path = os.path.join(os.path.dirname(__file__), "APIs", city.lower().replace(" ", "_"))
    attractions_path = os.path.join(base_path, "attractions")
    restaurants_path = os.path.join(base_path, "restaurants")

    # Check if both directories already exist and contain data
    attractions_exist = os.path.exists(attractions_path) and len(os.listdir(attractions_path)) > 0
    restaurants_exist = os.path.exists(restaurants_path) and len(os.listdir(restaurants_path)) > 0

    if attractions_exist and restaurants_exist:
        print(f"Data for '{city}' already exists. Skipping fetch.")
        return

    location_id = find_location_id(city)
    if location_id:
        if not attractions_exist:
            attractions = fetch_attractions(location_id)
            save_data(city, attractions, place_type="attractions")

        if not restaurants_exist:
            restaurants = fetch_restaurants(location_id)
            save_data(city, restaurants, place_type="restaurants")


#fetch_places("stuttgart")
