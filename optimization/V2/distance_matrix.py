from haversine import haversine
import json

import requests
from requests.structures import CaseInsensitiveDict

class POI:
    place: str
    type: str
    budget: float
    rating: float
    lat: float
    lon: float

    def __init__(self, place, type, budget, rating, location):
        self.place = place
        self.type = type
        self.budget = float(budget[1:]) if budget != "unknown" else 0
        self.rating = self.parse_rating(rating)
        self.lat = float(location.get("lat"))
        self.lon = float(location.get("lon"))

    def parse_rating(self, rating):
        if rating is None:
            return 0.0

        if isinstance(rating, (int, float)):
            return float(rating)

        rating = str(rating).strip()

        if "/" in rating:
            rating = rating.split("/")[0]

        try:
            return float(rating)
        except ValueError:
            return 0.0

    def to_dict(self):
        """Convert POI to dictionary for JSON serialization"""
        return {
            "place": self.place,
            "type": self.type,
            "budget": self.budget,
            "rating": self.rating,
            "latitude": self.lat,
            "longitude": self.lon
        }

    def __repr__(self):
        return f"POI(place={self.place}, type={self.type}, budget={self.budget}, rating={self.rating}, lat={self.lat}, lon={self.lon})"

def flatten_itinerary(json_data):
    # Parse JSON if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    itinerary = data.get("itinerary", {})
    
    # Initialize result arrays
    days_array = []
    
    # Sort days to maintain order (Day1, Day2, Day3)
    sorted_days = sorted(itinerary.keys(), key=lambda x: int(x.replace("Day", "")))

    itenerary_structure = []
    i = 1
    for day_key in sorted_days:
        day_data = itinerary[day_key]
        
        # Create arrays for each part of the day (each containing POI objects)
        morning_array = []
        afternoon_array = []
        evening_array = []
        
        # Process Morning
        if "Morning" in day_data:
            for item in day_data["Morning"]:
                morning_array.append(POI(
                    place=item.get("place"),
                    type=item.get("type"),
                    budget=item.get("expected_budget"),
                    rating=item.get("rating"),
                    location=item.get("location", {})
                ))
        
        # Process Afternoon
        if "Afternoon" in day_data:
            for item in day_data["Afternoon"]:
                afternoon_array.append(POI(
                    place=item.get("place"),
                    type=item.get("type"),
                    budget=item.get("expected_budget"),
                    rating=item.get("rating"),
                    location=item.get("location", {})
                ))
        
        # Process Evening
        if "Evening" in day_data:
            for item in day_data["Evening"]:
                evening_array.append(POI(
                    place=item.get("place"),
                    type=item.get("type"),
                    budget=item.get("expected_budget"),
                    rating=item.get("rating"),
                    location=item.get("location", {})
                ))
        
        itenerary_structure.append([len(morning_array), len(afternoon_array), len(evening_array)])
        # Create day block with 3 arrays
        day_block = [morning_array, afternoon_array, evening_array]
        days_array.append(day_block)
        i += 1
    return (days_array, itenerary_structure)

def flatten_locations(itinerary):
    """
    Flatten the itinerary into a list of (latitude, longitude) tuples.

    Parameters:
    itinerary (list): A nested list of POI objects structured by day and time of day.

    Returns:
    list of tuples: A flat list of (latitude, longitude) tuples for all locations in the itinerary.
    """
    locations = []
    
    for day in itinerary:
        for time_of_day in day:  # Morning, Afternoon, Evening
            for poi in time_of_day:
                locations.append(poi)
    
    return locations

def distance_matrix(locations):
    """
    Calculate the distance matrix for a list of locations.

    Parameters:
    locations (list of POI): A list of POI objects.

    Returns:
    list of lists: A distance matrix where the entry at [i][j] is the distance between locations[i] and locations[j].
    """
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine(locations[i], locations[j])
            matrix[i][j] = distance
            matrix[j][i] = distance  # Symmetric matrix

    return matrix

def geoepify_distance_matrix(locations, api_key = "321b043285654c3e90d399601005641c"):
    """
    Calculate the distance matrix using Geoapify API.

    Parameters:
    locations (list): List of tuples/lists -> [(lat, lon), ...]
    api_key (str): Geoapify API key

    Returns:
    list: Distance matrix response from Geoapify
    """

    url = f"https://api.geoapify.com/v1/routematrix?apiKey={api_key}"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"

    location_data = [
        {"location": [poi[1], poi[0]]}  # Geoapify expects [lon, lat]
        for poi in locations
    ]

    payload = {
        "mode": "drive",
        "sources": location_data,
        "targets": location_data
    }

    print("Requesting distance matrix from Geoapify API...")
    print(payload)

    # Use json= instead of data=
    resp = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if resp.status_code != 200:
        raise Exception(
            f"Geoapify API request failed "
            f"with status code {resp.status_code}: {resp.text}"
        )

    return resp.json()["sources_to_targets"]