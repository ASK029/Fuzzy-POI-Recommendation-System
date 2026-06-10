import json
import os
from pathlib import Path  # <-- This is the one you want

from distance_matrix import distance_matrix, flatten_itinerary, POI, flatten_locations, geoepify_distance_matrix
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def reconstruct_itinerary(optimal_order, itinerary_structure):
    """
    Rebuild itinerary using stored structure.

    itinerary_structure example:
    [
        [2, 3, 1],  # Day1 -> morning=2, afternoon=3, evening=1
        [1, 2, 2]   # Day2 -> morning=1, afternoon=2, evening=2
    ]
    """
    reconstructed = {}
    current_idx = 0

    part_names = ["Morning", "Afternoon", "Evening"]

    for day_idx, day_structure in enumerate(itinerary_structure, start=1):
        day_key = f"Day{day_idx}"
        reconstructed[day_key] = {}

        for part_idx, num_pois in enumerate(day_structure):
            part_name = part_names[part_idx]

            # Slice optimized route according to saved structure
            pois = optimal_order[current_idx: current_idx + int(num_pois)]

            reconstructed[day_key][part_name] = [
                poi.to_dict() for poi in pois
            ]

            current_idx += num_pois

    return reconstructed


def solve_tsp(itinerary):
    # Returns nested itinerary + structure
    itinerary_nested, itinerary_structure = flatten_itinerary(itinerary)

    # Flatten all POIs into a single list
    poi_list = flatten_locations(itinerary_nested)

    # Coordinates
    locations = [(poi.lat, poi.lon) for poi in poi_list]

    # Distance matrix
    # dist_matrix = distance_matrix(locations)
    geoepify_dist_matrix = geoepify_distance_matrix(locations)
    dist_matrix = [[item["distance"] for item in row] for row in geoepify_dist_matrix]

    # Routing manager
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)

    # Routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        return int(dist_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return json.dumps(
            {"error": "No solution found"},
            indent=2
        )

    # Extract optimal route indices
    index = routing.Start(0)
    route_indices = []

    while not routing.IsEnd(index):
        route_indices.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    # Remove duplicated start node
    route_indices = route_indices[:-1]

    # Map to POIs
    optimal_order = [poi_list[i] for i in route_indices]

    # Reconstruct itinerary
    optimized_itinerary = reconstruct_itinerary(
        optimal_order,
        itinerary_structure
    )

    result = {
        "optimized_itinerary": optimized_itinerary,
        "optimal_route": [
            poi.to_dict() for poi in optimal_order
        ],
        "total_pois": len(optimal_order),
        "itinerary_structure": itinerary_structure
    }

    return json.dumps(result, indent=2)
    

def save_tsp_result_to_json(result, filename="tsp_results.json", append=True):
    """
    Save TSP result to a JSON file.
    
    Parameters:
    result (dict): The TSP result dictionary
    filename (str): Name of the JSON file
    append (bool): If True, append to existing file; if False, overwrite
    """
    # Prepare the data structure
    if append and os.path.exists(filename):
        # Read existing data
        with open(filename, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # If file is empty or corrupted, start new
                existing_data = {"results": []}
        
        # Append new result
        if "results" in existing_data:
            existing_data["results"].append(result)
        else:
            # If structure is different, create results array
            existing_data = {"results": [existing_data, result]}
        
        # Write back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
    else:
        # Create new file with single result
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
    
    print(f"Results saved to {filename}")
    

if __name__ == "__main__":
    # Example usage
    # 1. Get the absolute path to the directory where this script lives
    CURRENT_DIR = Path(__file__).resolve().parent

    # 2. Join it with the filename safely
    json_path = CURRENT_DIR / "itinerary.json"
    itinerary = json.load(open(json_path))
    print("Original itinerary loaded from itinerary.json")
    # print(json.dumps(itinerary, indent=2))
    # a, b = flatten_itinerary(itinerary)
    # print(f"Flattened itinerary: {b}")
    optimal_route = solve_tsp(itinerary)
    save_tsp_result_to_json(json.loads(optimal_route), filename=CURRENT_DIR / "tsp_results3.json", append=True)
    print("Optimal route:", optimal_route)