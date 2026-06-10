import json
import os
import re


def normalize_name(name):
    if not name:
        return ""

    name = name.lower()
    name = name.replace("’", "'")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def load_raw_pois(file_paths, *additional_file_paths):
    """
    Load full raw POI records from JSON files.

    Supports files that contain:
    - a list of POIs
    - or a dictionary with a list inside
    """

    if additional_file_paths:
        paths = (file_paths, *additional_file_paths)
    elif isinstance(file_paths, (str, os.PathLike)):
        paths = [file_paths]
    else:
        paths = file_paths

    raw_records = []

    for file_path in paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            raw_records.extend(data)

        elif isinstance(data, dict):
            # If your JSON is {"data": [...]} or similar
            for value in data.values():
                if isinstance(value, list):
                    raw_records.extend(value)

    return raw_records


def build_raw_lookup(raw_records):
    """
    Build lookup dictionaries.

    Best lookup:
        location_id

    Fallback lookup:
        normalized name
    """

    by_id = {}
    by_name = {}

    for record in raw_records:
        location_id = record.get("location_id")
        name = record.get("name")

        if location_id:
            by_id[str(location_id)] = record

        if name:
            by_name[normalize_name(name)] = record

    return {
        "by_id": by_id,
        "by_name": by_name
    }


def find_raw_record_for_poi(poi, raw_lookup):
    """
    Match LLM-selected POI to full raw record.

    Priority:
    1. location_id
    2. normalized place name
    """

    if poi.location_id:
        raw = raw_lookup["by_id"].get(str(poi.location_id))
        if raw:
            return raw

    normalized = normalize_name(poi.place)
    return raw_lookup["by_name"].get(normalized)


def enrich_pois_with_raw_data(poi_list, raw_lookup):
    """
    Attach raw metadata to each POI object.
    """

    for poi in poi_list:
        raw = find_raw_record_for_poi(poi, raw_lookup)
        poi.raw_data = raw

    return poi_list

def get_opening_windows(raw_poi, weekday_index=0):
    """
    Return opening windows for a POI.

    weekday_index:
        This depends on your dataset convention.
        Some datasets use 0 = Sunday.
        Python normally uses 0 = Monday.

    For first implementation, if your data has same hours every day,
    this does not matter much.
    """

    if not raw_poi:
        return []

    hours = raw_poi.get("hours")

    if not hours:
        return []

    week_ranges = hours.get("week_ranges", [])

    if not week_ranges:
        return []

    if weekday_index >= len(week_ranges):
        return []

    return week_ranges[weekday_index]

def convert_opening_window_to_solver_seconds(
    open_time_minutes,
    close_time_minutes,
    start_hour,
    daily_time_budget_seconds,
    service_time_seconds
):
    """
    Convert real opening hours to OR-Tools route-relative seconds.

    Example:
        route starts at 09:00
        POI opens 12:00
        result starts at 3 hours after route start
    """

    route_start_minutes = start_hour * 60
    service_time_minutes = service_time_seconds / 60

    # Handle places that close after midnight
    if close_time_minutes < open_time_minutes:
        close_time_minutes += 24 * 60

    # Latest arrival must allow enough time to finish before closing
    latest_arrival_minutes = close_time_minutes - service_time_minutes

    start_seconds = max(
        0,
        int((open_time_minutes - route_start_minutes) * 60)
    )

    end_seconds = min(
        daily_time_budget_seconds,
        int((latest_arrival_minutes - route_start_minutes) * 60)
    )

    if end_seconds <= 0:
        return None

    if start_seconds >= daily_time_budget_seconds:
        return None

    if start_seconds >= end_seconds:
        return None

    return start_seconds, end_seconds

def get_text_list(raw_poi, key):
    values = raw_poi.get(key, [])

    if not isinstance(values, list):
        return []

    return [
        item.get("name", "").lower()
        for item in values
        if isinstance(item, dict)
    ]

def infer_visit_duration_minutes_from_raw(poi):
    """
    Estimate visit duration using raw data.
    Does not require changing the LLM JSON.
    """

    raw_poi = poi.raw_data

    if not raw_poi:
        # Fallback from LLM type
        poi_type = poi.type.lower() if poi.type else ""

        if "restaurant" in poi_type:
            return 75

        return 60

    name = raw_poi.get("name", "").lower()
    description = raw_poi.get("description", "").lower()

    category = raw_poi.get("category", {}).get("key", "").lower()
    subtypes = get_text_list(raw_poi, "subtype")
    subcategories = get_text_list(raw_poi, "subcategory")

    subtype_text = " ".join(subtypes)
    subcategory_text = " ".join(subcategories)

    if category == "restaurant":
        return 75

    if "each revolution takes about 30 minutes" in description:
        return 45

    if "observation decks" in subtype_text or "observation" in subtype_text:
        return 60

    if "shopping" in subcategory_text or "mall" in name or "shoppes" in name:
        return 90

    if "national conservation area" in name:
        return 180

    if "hoover dam" in name:
        return 120

    if "fremont street" in name:
        return 120

    if "casino" in name:
        return 90

    if category == "attraction":
        return 75

    return 60
