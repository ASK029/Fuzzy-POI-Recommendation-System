import json

try:
    from haversine import haversine
except ImportError:  # The V3 utilities do not currently use haversine directly.
    haversine = None

try:
    import requests
    from requests.structures import CaseInsensitiveDict
except ImportError:  # Only needed when geoepify_distance_matrix is called.
    requests = None

    class CaseInsensitiveDict(dict):
        pass

class POI:
    location_id: str
    place: str
    type: str
    budget: float
    rating: float
    lat: float
    lon: float

    def __init__(
        self,
        place=None,
        type=None,
        budget=None,
        rating=None,
        location=None,
        location_id=None
    ):
        self.location_id = str(location_id) if location_id else None
        self.place = place
        self.type = type
        self.budget = self.parse_budget(budget)
        self.rating = self.parse_rating(rating)
        location = location or {}
        self.lat = float(location.get("lat"))
        self.lon = float(location.get("lon"))
        self.raw_data = None
        self.visit_duration_minutes = 60

    def parse_budget(self, budget):
        if budget is None:
            return 0.0

        if isinstance(budget, (int, float)):
            return float(budget)

        budget = str(budget).strip()

        if not budget or budget.lower() == "unknown":
            return 0.0

        budget = budget.replace("$", "").replace(",", "")

        try:
            return float(budget)
        except ValueError:
            return 0.0

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
            "location_id": self.location_id,
            "place": self.place,
            "type": self.type,
            "budget": self.budget,
            "rating": self.rating,
            "latitude": self.lat,
            "longitude": self.lon
        }

    def __repr__(self):
        return f"POI(location_id={self.location_id}, place={self.place}, type={self.type}, budget={self.budget}, rating={self.rating}, lat={self.lat}, lon={self.lon})"

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
                    location_id=item.get("location_id"),
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
                    location_id=item.get("location_id"),
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
                    location_id=item.get("location_id"),
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


def load_raw_pois(*file_paths):
    raw_pois = {}

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            location_id = str(item.get("location_id"))
            raw_pois[location_id] = item

    return raw_pois

def enrich_pois(poi_list, raw_pois):
    enriched = []

    for poi in poi_list:
        raw = None

        if poi.location_id:
            raw = raw_pois.get(str(poi.location_id))

        poi.raw_data = raw
        enriched.append(poi)

    return enriched

def get_opening_windows(raw_poi, weekday_index):
    """
    Returns opening windows for a POI for a specific weekday.

    weekday_index depends on your dataset convention.
    Keep it configurable until you verify whether index 0 is Sunday or Monday.
    """

    if not raw_poi:
        return []

    hours = raw_poi.get("hours")

    if not hours:
        return []

    week_ranges = hours.get("week_ranges", [])

    if weekday_index >= len(week_ranges):
        return []

    return week_ranges[weekday_index]

def convert_opening_window_to_route_seconds(
    open_time_minutes,
    close_time_minutes,
    start_hour,
    daily_time_budget_seconds
):
    route_start_minutes = start_hour * 60

    start_seconds = max(
        0,
        (open_time_minutes - route_start_minutes) * 60
    )

    end_seconds = min(
        daily_time_budget_seconds,
        (close_time_minutes - route_start_minutes) * 60
    )

    if end_seconds <= 0:
        return None

    if start_seconds >= daily_time_budget_seconds:
        return None

    if start_seconds >= end_seconds:
        return None

    return int(start_seconds), int(end_seconds)

def get_part_of_day_from_seconds(start_hour, arrival_seconds):
    total_minutes = start_hour * 60 + arrival_seconds // 60

    if total_minutes < 12 * 60:
        return "Morning"

    if total_minutes < 17 * 60:
        return "Afternoon"

    return "Evening"

def geoepify_distance_matrix(locations, api_key = "321b043285654c3e90d399601005641c"):
    """
    Calculate the distance matrix using Geoapify API.

    Parameters:
    locations (list): List of tuples/lists -> [(lat, lon), ...]
    api_key (str): Geoapify API key

    Returns:
    list: Distance matrix response from Geoapify
    """

    if requests is None:
        raise ImportError(
            "requests is required to call geoepify_distance_matrix. "
            "Install project dependencies before running live optimization."
        )

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
