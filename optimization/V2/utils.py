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

    @classmethod
    def hotel(cls, lat, lon):
        poi = cls.__new__(cls)

        poi.place = "Hotel"
        poi.type = "Hotel"
        poi.budget = 0
        poi.rating = 0

        poi.lat = lat
        poi.lon = lon

        return poi

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

def flatten_day(day_data):
    pois = []

    for part_name in ["Morning", "Afternoon", "Evening"]:
        for item in day_data.get(part_name, []):
            pois.append(
                POI(
                    place=item.get("place"),
                    type=item.get("type"),
                    budget=item.get("expected_budget"),
                    rating=item.get("rating"),
                    location=item.get("location", {})
                )
            )

    return pois

def flatten_itinerary(json_data, hotel_location=None, include_hotel=True):
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
    
    hotel_poi = POI.hotel(hotel_location["lat"], hotel_location["lon"]) if hotel_location else None
    days_structure = []

    for day_key in sorted_days:
        day_data = itinerary[day_key]
        day_structure = [len(day_data.get("Morning", [])), len(day_data.get("Afternoon", [])), len(day_data.get("Evening", []))]
        days_structure.append(day_structure)
        days_array.append([hotel_poi] + flatten_day(day_data)) if include_hotel else days_array.append([flatten_day(day_data)])

    return (days_array, days_structure)
'''
# def flatten_locations(days_array=None, day_locations=None, per_day=False):
#     locations = []
#     if not per_day:
#         if day_locations is None and days_array is not None:
#             for day in days_array:
#                 locations += [day]
#         else:
#             raise ValueError("Either days_array or day_locations must be provided when per_day is False.")
#     else:
#         if day_locations:
#             return day_locations
#         else:
#             raise ValueError("Either days_array or day_locations must be provided when per_day is False.")

#     return locations


# def distance_matrix(locations):
#     """
#     Calculate the distance matrix for a list of locations.

#     Parameters:
#     locations (list of POI): A list of POI objects.

#     Returns:
#     list of lists: A distance matrix where the entry at [i][j] is the distance between locations[i] and locations[j].
#     """
#     n = len(locations)
#     matrix = [[0] * n for _ in range(n)]

#     for i in range(n):
#         for j in range(i + 1, n):
#             distance = haversine(locations[i], locations[j])
#             matrix[i][j] = distance
#             matrix[j][i] = distance  # Symmetric matrix

#     return matrix
'''

def geoepify_matrices(locations, api_key = "321b043285654c3e90d399601005641c"):
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