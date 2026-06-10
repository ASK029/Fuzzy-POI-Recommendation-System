import json
from pathlib import Path

import folium


PARTS_OF_DAY = ("Morning", "Afternoon", "Evening")
DAY_COLORS = {
    "Day1": "red",
    "Day2": "green",
    "Day3": "blue",
    "Day4": "purple",
    "Day5": "darkred",
    "Day6": "cadetblue",
    "Day7": "darkgreen",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_vrp_result(vrp_results):
    results = vrp_results.get("results")
    if isinstance(results, list) and results:
        return results[-1]

    return vrp_results


def hotel_from_vrp(vrp_result, fallback=None):
    hotel = vrp_result.get("hotel")
    if hotel:
        return hotel

    return fallback


def original_itinerary_routes(original_itinerary):
    itinerary = original_itinerary.get("itinerary", {})
    routes = {}

    for day_name in sorted(itinerary, key=_day_sort_key):
        route = []
        for part_name in PARTS_OF_DAY:
            for poi in itinerary[day_name].get(part_name, []):
                location = poi.get("location", {})
                route.append(
                    {
                        "lat": location.get("lat"),
                        "lon": location.get("lon"),
                        "name": poi.get("place"),
                        "day": day_name,
                        "time": part_name,
                        "type": poi.get("type"),
                        "budget": poi.get("expected_budget"),
                        "rating": poi.get("rating"),
                        "description": poi.get("description"),
                    }
                )
        routes[day_name] = route

    return routes


def optimized_vrp_routes(vrp_result):
    if "routes" in vrp_result:
        routes = {}
        for day_result in vrp_result["routes"]:
            day_name = day_result["day"]
            routes[day_name] = [
                _optimized_poi_to_point(day_name, poi)
                for poi in day_result.get("route", [])
            ]
        return routes

    itinerary = vrp_result.get("optimized_itinerary", {})
    routes = {}
    for day_name in sorted(itinerary, key=_day_sort_key):
        route = []
        for part_name in PARTS_OF_DAY:
            for poi in itinerary[day_name].get(part_name, []):
                point = _optimized_poi_to_point(day_name, poi)
                point["time"] = part_name
                route.append(point)
        routes[day_name] = route

    return routes


def create_folium_map(routes_by_day, title="Itinerary Map", hotel_location=None):
    points = [
        point
        for route in routes_by_day.values()
        for point in route
        if point.get("lat") is not None and point.get("lon") is not None
    ]

    if hotel_location:
        points.append(
            {
                "lat": hotel_location["lat"],
                "lon": hotel_location["lon"],
                "name": "Hotel",
                "day": "Hotel",
                "time": "Start / End",
                "type": "Hotel",
            }
        )

    if not points:
        raise ValueError(f"No points found for {title}")

    avg_lat = sum(point["lat"] for point in points) / len(points)
    avg_lon = sum(point["lon"] for point in points) / len(points)

    itinerary_map = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    title_html = f"""
    <h3 style="position: fixed; top: 10px; left: 50px; z-index: 9999;
               background: white; padding: 8px 12px; border: 1px solid #bbb;
               font-family: sans-serif; margin: 0;">
        {title}
    </h3>
    """
    itinerary_map.get_root().html.add_child(folium.Element(title_html))

    if hotel_location:
        folium.Marker(
            location=[hotel_location["lat"], hotel_location["lon"]],
            popup=folium.Popup("<b>Hotel</b><br>Start and end point", max_width=260),
            tooltip="Hotel",
            icon=folium.Icon(color="orange", icon="home"),
        ).add_to(itinerary_map)

    for day_name in sorted(routes_by_day, key=_day_sort_key):
        route = [
            point
            for point in routes_by_day[day_name]
            if point.get("lat") is not None and point.get("lon") is not None
        ]
        if not route:
            continue

        color = DAY_COLORS.get(day_name, "gray")
        day_layer = folium.FeatureGroup(name=day_name, show=True)

        route_coords = [[point["lat"], point["lon"]] for point in route]
        if hotel_location:
            hotel_coords = [hotel_location["lat"], hotel_location["lon"]]
            route_coords = [hotel_coords] + route_coords + [hotel_coords]

        folium.PolyLine(
            route_coords,
            color=color,
            weight=3,
            opacity=0.7,
            dash_array="5, 5",
            tooltip=f"{day_name} route",
        ).add_to(day_layer)

        for idx, point in enumerate(route, start=1):
            popup_html = _popup_html(idx, point)
            folium.Marker(
                location=[point["lat"], point["lon"]],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{day_name} stop {idx}: {point['name']}",
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(day_layer)

        day_layer.add_to(itinerary_map)

    folium.LayerControl(collapsed=False).add_to(itinerary_map)
    return itinerary_map


def build_maps(
    original_path="itinerary.json",
    optimized_path="vrp_results.json",
    output_dir=".",
):
    output_dir = Path(output_dir)
    original_itinerary = load_json(original_path)
    vrp_result = latest_vrp_result(load_json(optimized_path))
    hotel_location = hotel_from_vrp(vrp_result)

    original_map = create_folium_map(
        original_itinerary_routes(original_itinerary),
        title="Original itinerary",
        hotel_location=hotel_location,
    )
    optimized_map = create_folium_map(
        optimized_vrp_routes(vrp_result),
        title="Optimized VRP itinerary",
        hotel_location=hotel_location,
    )

    original_html = output_dir / "original_itinerary_map.html"
    optimized_html = output_dir / "optimized_vrp_itinerary_map.html"

    original_map.save(original_html)
    optimized_map.save(optimized_html)

    return original_map, optimized_map


def _optimized_poi_to_point(day_name, poi):
    return {
        "lat": poi.get("latitude"),
        "lon": poi.get("longitude"),
        "name": poi.get("place"),
        "day": day_name,
        "time": _time_label(poi),
        "type": poi.get("type"),
        "budget": poi.get("budget"),
        "rating": poi.get("rating"),
        "arrival_time": poi.get("arrival_time"),
        "departure_time": poi.get("departure_time"),
        "travel_from_previous_minutes": poi.get("travel_from_previous_minutes"),
    }


def _time_label(poi):
    arrival = poi.get("arrival_time")
    departure = poi.get("departure_time")

    if arrival and departure:
        return f"{arrival} - {departure}"

    return poi.get("time", "N/A")


def _popup_html(idx, point):
    rows = [
        ("Day", point.get("day")),
        ("Time", point.get("time")),
        ("Type", point.get("type")),
        ("Rating", point.get("rating")),
        ("Budget", point.get("budget")),
        ("Travel from previous", _minutes(point.get("travel_from_previous_minutes"))),
    ]
    details = "".join(
        f"<b>{label}:</b> {value}<br>"
        for label, value in rows
        if value not in (None, "", "N/A")
    )

    description = point.get("description")
    if description:
        details += f"<p style='margin: 8px 0 0;'>{description}</p>"

    return f"""
    <div style="min-width: 180px; font-family: sans-serif;">
        <h4 style="margin: 0 0 5px;">{idx}. {point.get("name")}</h4>
        <hr style="margin: 5px 0;">
        {details}
    </div>
    """


def _minutes(value):
    if value is None:
        return None

    return f"{value} min"


def _day_sort_key(day_name):
    try:
        return int(str(day_name).replace("Day", ""))
    except ValueError:
        return 999


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    build_maps(
        original_path=current_dir / "itinerary.json",
        optimized_path=current_dir / "vrp_results.json",
        output_dir=current_dir,
    )
    print("Saved original_itinerary_map.html and optimized_vrp_itinerary_map.html")
