import json
import os
from pathlib import Path

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from raw_poi_utils import (
    build_raw_lookup,
    enrich_pois_with_raw_data,
    get_opening_windows,
    convert_opening_window_to_solver_seconds,
    infer_visit_duration_minutes_from_raw,
    load_raw_pois,
)
from utils import (
    flatten_itinerary,
    flatten_locations,
    geoepify_distance_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

def apply_opening_hour_constraints(
    routing,
    manager,
    time_dimension,
    poi_list,
    service_times,
    start_hour,
    daily_time_budget_seconds,
    weekday_index=0
):
    """
    Apply opening hours as OR-Tools time-window constraints.

    Node mapping:
        node 0 = hotel
        node 1 = poi_list[0]
        node 2 = poi_list[1]
        ...
    """

    stats = {
        "applied": 0,
        "missing_opening_hours": 0,
        "infeasible_opening_windows": 0
    }

    for poi_index, poi in enumerate(poi_list, start=1):
        raw_poi = poi.raw_data

        windows = get_opening_windows(
            raw_poi,
            weekday_index=weekday_index
        )

        if not windows:
            # No opening hours available.
            # For now, do not constrain it.
            stats["missing_opening_hours"] += 1
            continue

        valid_solver_windows = []

        for window in windows:
            converted = convert_opening_window_to_solver_seconds(
                open_time_minutes=window["open_time"],
                close_time_minutes=window["close_time"],
                start_hour=start_hour,
                daily_time_budget_seconds=daily_time_budget_seconds,
                service_time_seconds=service_times[poi_index]
            )

            if converted:
                valid_solver_windows.append(converted)

        if not valid_solver_windows:
            # No feasible time window for this POI.
            # For first implementation, skip constraint instead of crashing.
            # Later, you can mark this POI as impossible or droppable.
            stats["infeasible_opening_windows"] += 1
            continue

        # First implementation: use the widest valid window.
        start_seconds = min(window[0] for window in valid_solver_windows)
        end_seconds = max(window[1] for window in valid_solver_windows)

        routing_index = manager.NodeToIndex(poi_index)

        time_dimension.CumulVar(routing_index).SetRange(
            start_seconds,
            end_seconds
        )

        stats["applied"] += 1

    return stats

def classify_restaurant_time_role(raw_poi):
    if not raw_poi:
        return "meal"

    category = raw_poi.get("category", {}).get("key", "").lower()

    if category != "restaurant":
        return None

    hours = raw_poi.get("hours", {})
    week_ranges = hours.get("week_ranges", [])

    if not week_ranges or not week_ranges[0]:
        return "meal"

    open_time = week_ranges[0][0]["open_time"]
    close_time = week_ranges[0][0]["close_time"]

    if open_time <= 480 and close_time <= 960:
        return "breakfast_lunch"

    if open_time >= 960:
        return "dinner"

    if close_time >= 1200:
        return "lunch_dinner"

    return "meal"

def seconds_to_clock(start_hour, offset_seconds):
    """
    Convert seconds from start of day into HH:MM clock time.
    Example:
        start_hour = 9
        offset_seconds = 3600
        result = 10:00
    """

    total_minutes = (start_hour * 60) + int(offset_seconds // 60)
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60

    return f"{hour:02d}:{minute:02d}"


def get_part_of_day(clock_time):
    """
    Assign a POI to Morning, Afternoon, or Evening based on arrival time.
    """

    hour = int(clock_time.split(":")[0])

    if hour < 12:
        return "Morning"

    if hour < 17:
        return "Afternoon"

    return "Evening"


def build_time_and_distance_matrices(locations):
    """
    locations format:
        [(lat, lon), (lat, lon), ...]

    Geoapify returns:
        distance in meters
        time in seconds
    """

    geoapify_matrix = geoepify_distance_matrix(locations)

    distance_matrix = [
        [int(item["distance"]) for item in row]
        for row in geoapify_matrix
    ]

    time_matrix = [
        [int(item["time"]) for item in row]
        for row in geoapify_matrix
    ]

    return distance_matrix, time_matrix


def solve_multi_day_vrp(
    itinerary,
    hotel_lat,
    hotel_lon,
    raw_poi_files=None,
    daily_time_budget_hours=8,
    start_hour=9,
    weekday_index=0,
    balance_penalty=100,
    search_time_limit_seconds=10,
):
    """
    Multi-day VRP optimizer.

    Each day is treated as one vehicle.

    Constraints:
    1. Every POI is visited once.
    2. Every day starts at hotel.
    3. Every day ends at hotel.
    4. Each day cannot exceed daily_time_budget_hours.
    5. Day durations are softly balanced using penalties.
    """

    itinerary_nested, _original_structure = flatten_itinerary(itinerary)
    poi_list = flatten_locations(itinerary_nested)

    num_days = len(itinerary_nested)

    if not poi_list:
        return json.dumps(
            {"error": "No POIs found in itinerary"},
            indent=2
        )

    if raw_poi_files is None:
        raw_poi_files = [
            PROJECT_ROOT / "data" / "raw" / "las_vegas" / "attractions.json",
            PROJECT_ROOT / "data" / "raw" / "las_vegas" / "restaurants.json",
        ]
    elif isinstance(raw_poi_files, (str, os.PathLike)):
        raw_poi_files = [raw_poi_files]
    else:
        raw_poi_files = list(raw_poi_files)

    raw_records = load_raw_pois(raw_poi_files)
    raw_lookup = build_raw_lookup(raw_records)
    poi_list = enrich_pois_with_raw_data(poi_list, raw_lookup)
    matched_raw_pois = sum(1 for poi in poi_list if poi.raw_data)

    # Node 0 is always the hotel / residence
    locations = [(hotel_lat, hotel_lon)] + [
        (poi.lat, poi.lon)
        for poi in poi_list
    ]

    distance_matrix, time_matrix = build_time_and_distance_matrices(
        locations
    )

    # Service time means time spent at the location.
    # Hotel service time is zero.
    service_times = [0] + [
        infer_visit_duration_minutes_from_raw(poi) * 60
        for poi in poi_list
    ]

    for poi, service_time in zip(poi_list, service_times[1:]):
        poi.visit_duration_minutes = int(service_time / 60)

    daily_time_budget_seconds = int(daily_time_budget_hours * 60 * 60)

    # All days start and end at hotel node 0
    starts = [0] * num_days
    ends = [0] * num_days

    manager = pywrapcp.RoutingIndexManager(
        len(locations),
        num_days,
        starts,
        ends
    )

    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        """
        Cost from one node to another.

        Includes:
        - travel time from current node to next node
        - visit duration at current node

        This means the cumulative time at a POI represents arrival time.
        """

        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        travel_time = time_matrix[from_node][to_node]
        visit_time = service_times[from_node]

        return int(travel_time + visit_time)

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Objective: minimize total travel + service time
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

    # Add daily time budget constraint
    routing.AddDimension(
        time_callback_index,
        0,
        daily_time_budget_seconds,
        True,
        "Time"
    )

    time_dimension = routing.GetDimensionOrDie("Time")

    opening_constraint_stats = apply_opening_hour_constraints(
        routing=routing,
        manager=manager,
        time_dimension=time_dimension,
        poi_list=poi_list,
        service_times=service_times,
        start_hour=start_hour,
        daily_time_budget_seconds=daily_time_budget_seconds,
        weekday_index=weekday_index
    )

    # Balanced days constraint using soft bounds.
    # Estimate target based on total service time.
    # Travel time is unknown before solving, so this is an approximation.
    total_service_time = sum(service_times)
    estimated_target_day_time = int(total_service_time / num_days)

    # Avoid target being too tiny
    estimated_target_day_time = max(
        estimated_target_day_time,
        int(daily_time_budget_seconds * 0.45)
    )

    # Never set target above daily max
    estimated_target_day_time = min(
        estimated_target_day_time,
        daily_time_budget_seconds
    )

    lower_balance_bound = int(estimated_target_day_time * 0.80)
    upper_balance_bound = int(estimated_target_day_time * 1.20)

    upper_balance_bound = min(
        upper_balance_bound,
        daily_time_budget_seconds
    )

    for vehicle_id in range(num_days):
        end_index = routing.End(vehicle_id)

        # Penalize days that are too short
        time_dimension.SetCumulVarSoftLowerBound(
            end_index,
            lower_balance_bound,
            balance_penalty
        )

        # Penalize days that are too long
        time_dimension.SetCumulVarSoftUpperBound(
            end_index,
            upper_balance_bound,
            balance_penalty
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = search_time_limit_seconds

    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return json.dumps(
            {
                "error": "No feasible VRP solution found",
                "possible_reasons": [
                    "Daily time budget is too small",
                    "Visit durations are too high",
                    "Opening-hour constraints are too strict",
                    "Too many POIs for the number of days",
                    "Some POIs are very far from the city center"
                ],
                "suggestions": [
                    "Increase daily_time_budget_hours",
                    "Reduce inferred visit durations",
                    "Relax opening-hour constraints",
                    "Allow dropping low-priority POIs in a future version"
                ]
            },
            indent=2
        )

    optimized_itinerary = {}
    daily_summaries = []
    all_routes = []

    for vehicle_id in range(num_days):
        day_key = f"Day{vehicle_id + 1}"

        optimized_itinerary[day_key] = {
            "Morning": [],
            "Afternoon": [],
            "Evening": []
        }

        index = routing.Start(vehicle_id)

        route_items = []
        route_distance_meters = 0
        previous_node = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)

            if node != 0:
                poi = poi_list[node - 1]

                arrival_seconds = solution.Value(
                    time_dimension.CumulVar(index)
                )

                visit_seconds = service_times[node]
                departure_seconds = arrival_seconds + visit_seconds

                arrival_time = seconds_to_clock(
                    start_hour,
                    arrival_seconds
                )

                departure_time = seconds_to_clock(
                    start_hour,
                    departure_seconds
                )

                travel_from_previous_seconds = time_matrix[previous_node][node]
                travel_from_previous_minutes = round(
                    travel_from_previous_seconds / 60,
                    1
                )

                item = poi.to_dict()

                item["arrival_time"] = arrival_time
                item["departure_time"] = departure_time
                item["visit_duration_minutes"] = int(visit_seconds / 60)
                item["travel_from_previous_minutes"] = travel_from_previous_minutes
                item["raw_data_matched"] = poi.raw_data is not None
                item["opening_hours_constrained"] = bool(
                    get_opening_windows(poi.raw_data, weekday_index)
                )

                part_of_day = get_part_of_day(arrival_time)

                optimized_itinerary[day_key][part_of_day].append(item)

                route_items.append(item)

                route_distance_meters += distance_matrix[previous_node][node]
                previous_node = node

            index = solution.Value(routing.NextVar(index))

        # Add return-to-hotel distance
        route_distance_meters += distance_matrix[previous_node][0]

        day_total_seconds = solution.Value(
            time_dimension.CumulVar(index)
        )

        day_summary = {
            "day": day_key,
            "total_time_minutes": round(day_total_seconds / 60, 1),
            "total_time_hours": round(day_total_seconds / 3600, 2),
            "total_distance_km": round(route_distance_meters / 1000, 2),
            "number_of_pois": len(route_items),
            "starts_at": seconds_to_clock(start_hour, 0),
            "ends_at": seconds_to_clock(start_hour, day_total_seconds)
        }

        daily_summaries.append(day_summary)

        all_routes.append({
            "day": day_key,
            "route": route_items,
            "summary": day_summary
        })

    result = {
        "optimization_type": "multi_day_vrp_v3",
        "description": "Each day is treated as a vehicle. V3 enriches POIs with raw metadata, uses inferred visit durations, and applies opening-hour constraints.",
        "hotel": {
            "lat": hotel_lat,
            "lon": hotel_lon
        },
        "settings": {
            "daily_time_budget_hours": daily_time_budget_hours,
            "start_hour": start_hour,
            "weekday_index": weekday_index,
            "balance_penalty": balance_penalty,
            "number_of_days": num_days,
            "total_pois": len(poi_list),
            "raw_poi_records_loaded": len(raw_records),
            "matched_raw_pois": matched_raw_pois,
            "raw_poi_files": [
                str(Path(file_path))
                for file_path in raw_poi_files
            ],
            "opening_hour_constraints": opening_constraint_stats,
            "lower_balance_bound_minutes": round(lower_balance_bound / 60, 1),
            "upper_balance_bound_minutes": round(upper_balance_bound / 60, 1)
        },
        "optimized_itinerary": optimized_itinerary,
        "daily_summaries": daily_summaries,
        "routes": all_routes
    }

    return json.dumps(result, indent=2)


def save_vrp_result_to_json(result, filename="vrp_results.json", append=True):
    """
    Save VRP result to JSON file.
    """

    if append and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"results": []}

        if "results" in existing_data:
            existing_data["results"].append(result)
        else:
            existing_data = {
                "results": [existing_data, result]
            }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)

    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).resolve().parent

    json_path = CURRENT_DIR / "itinerary.json"

    with open(json_path, "r", encoding="utf-8") as f:
        itinerary = json.load(f)

    print("Original itinerary loaded from itinerary.json")

    # Example hotel / residence point in Las Vegas.
    # Later this comes from the user in your app.
    hotel_lat = 36.1147
    hotel_lon = -115.1728

    vrp_result_json = solve_multi_day_vrp(
        itinerary=itinerary,
        hotel_lat=hotel_lat,
        hotel_lon=hotel_lon,
        daily_time_budget_hours=11,
        start_hour=9,
        balance_penalty=100,
        search_time_limit_seconds=10
    )

    vrp_result = json.loads(vrp_result_json)

    save_vrp_result_to_json(
        vrp_result,
        filename=CURRENT_DIR / "vrp_results.json",
        append=True
    )

    print("Optimized VRP result:")
    print(vrp_result_json)
