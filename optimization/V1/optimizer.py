import json
import os
from pathlib import Path  # <-- This is the one you want

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from utils import flatten_itinerary, geoepify_matrices

def reconstruct_itinerary(optimal_order, itinerary_structure, day_idx=1, reconstructed=None):
    """
    Rebuild itinerary using stored structure.

    itinerary_structure example:
    [
        [2, 3, 1],  # Day1 -> morning=2, afternoon=3, evening=1
        [1, 2, 2]   # Day2 -> morning=1, afternoon=2, evening=2
    ]
    """

    if itinerary_structure is None or day_idx > len(itinerary_structure):
        return None
    
    if reconstructed is None:
        reconstructed = {}
        
    current_idx = 0

    part_names = ["Morning", "Afternoon", "Evening"]

    day_structure = itinerary_structure[day_idx - 1]  # Get structure for the current day

    day_key = f"Day{day_idx}"
    reconstructed[day_key] = {}

    for part_idx, num_pois in enumerate(day_structure):
        part_name = part_names[part_idx]

        # Slice optimized route according to saved structure
        pois = optimal_order[day_idx - 1][current_idx: current_idx + int(num_pois)]
        # print(f"pois for {day_key} {part_name}: {pois}")

        reconstructed[day_key][part_name] = [
            poi.to_dict() for poi in pois
        ]

        current_idx += num_pois

    if day_idx == len(itinerary_structure):
        return reconstructed
    
    return reconstruct_itinerary(optimal_order, itinerary_structure, day_idx + 1, reconstructed)


def solve_tsp(itinerary, hotel_location=None):
    # Returns nested itinerary + structure
    itinerary_nested, itinerary_structure = flatten_itinerary(itinerary, hotel_location=hotel_location, include_hotel=True)
    optimal_order = []

    for day in itinerary_nested:
        print(f"Processing day with {len(day)} POIs {day}")
        locations = [(poi.lat, poi.lon) for poi in day]

        # Distance matrix
        # dist_matrix = distance_matrix(locations)
        matrix_response = geoepify_matrices(locations)
        time_matrix = [[item["time"] for item in row] for row in matrix_response]

        # Routing manager
        manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)

        # Routing model
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)

            return int(time_matrix[from_node][to_node] * 1000)

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
        route_indices = route_indices[1:]  # Start and end at the same point

        # Map to POIs
        optimal_order.append([day[i] for i in route_indices])

    # Reconstruct itinerary
    optimized_itinerary = reconstruct_itinerary(
        optimal_order,
        itinerary_structure
    )

    result = {
        "optimized_itinerary": optimized_itinerary,
        # "optimal_route": [
        #     poi.to_dict() for poi in optimal_order
        # ],
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
    json_path = CURRENT_DIR.parent / "itinerary.json"
    itinerary = json.load(open(json_path))
    print("Original itinerary loaded from itinerary.json")
    # print(json.dumps(itinerary, indent=2))
    # a, b = flatten_itinerary(itinerary)
    # print(f"Flattened itinerary: {b}")
    optimal_route = solve_tsp(itinerary, hotel_location={
        "lat": 36.1699,
        "lon": -115.1398
    })
    save_tsp_result_to_json(json.loads(optimal_route), filename=CURRENT_DIR / "tsp_results.json", append=True)
    print("Optimal route:", optimal_route)