import random
import sys
import unittest
from pathlib import Path


V3_DIR = Path(__file__).resolve().parents[1]
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))

from ga_optimizer import (  # noqa: E402
    Chromosome,
    build_original_itinerary_chromosome,
    decode_chromosome_to_itinerary,
    evaluate_chromosome,
    mutate,
    repair_chromosome,
    route_aware_crossover,
    run_ga,
)
from utils import POI, flatten_itinerary, flatten_locations  # noqa: E402


def make_pois(count):
    return [
        POI(
            location_id=str(index),
            place=f"POI {index}",
            type="Attraction",
            budget=0,
            rating=4.0,
            location={"lat": 36.0 + index * 0.01, "lon": -115.0},
        )
        for index in range(count)
    ]


def make_matrix(size, off_diagonal_value):
    return [
        [
            0 if row == column else off_diagonal_value
            for column in range(size)
        ]
        for row in range(size)
    ]


def flatten_optimized_itinerary(optimized_itinerary):
    items = []

    for day in optimized_itinerary.values():
        for part in ("Morning", "Afternoon", "Evening"):
            items.extend(day[part])

    return items


class TestGAOptimizer(unittest.TestCase):
    def test_repair_chromosome_removes_duplicates_and_adds_missing_pois(self):
        chromosome = Chromosome(routes=[[0, 1, 1, 8], [2], [4]])

        repaired = repair_chromosome(
            chromosome=chromosome,
            num_pois=5,
            num_days=3,
        )
        flat = sorted(poi for route in repaired.routes for poi in route)

        self.assertEqual(flat, [0, 1, 2, 3, 4])
        self.assertEqual(len(repaired.routes), 3)

    def test_over_budget_route_has_worse_cost(self):
        pois = make_pois(3)
        distance_matrix = make_matrix(4, 1000)
        time_matrix = make_matrix(4, 60)
        service_times = [0, 60, 60, 60]

        over_budget = evaluate_chromosome(
            chromosome=Chromosome(routes=[[0, 1, 2], []]),
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            poi_list=pois,
            daily_time_budget_seconds=5 * 60,
            start_hour=9,
            weekday_index=0,
        )
        feasible = evaluate_chromosome(
            chromosome=Chromosome(routes=[[0, 1], [2]]),
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            poi_list=pois,
            daily_time_budget_seconds=5 * 60,
            start_hour=9,
            weekday_index=0,
        )

        self.assertGreater(over_budget.cost, feasible.cost)

    def test_mutation_preserves_all_pois_once(self):
        chromosome = Chromosome(routes=[[0, 1], [2, 3], [4]])

        mutated = mutate(
            chromosome=chromosome,
            mutation_rate=1.0,
            num_days=3,
            rng=random.Random(7),
        )
        flat = sorted(poi for route in mutated.routes for poi in route)

        self.assertEqual(flat, [0, 1, 2, 3, 4])

    def test_route_aware_crossover_child_is_valid(self):
        parent_a = Chromosome(routes=[[0, 1, 2], [3], [4, 5]])
        parent_b = Chromosome(routes=[[5, 4], [3, 2], [1, 0]])

        child = route_aware_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            num_days=3,
            rng=random.Random(11),
        )
        flat = sorted(poi for route in child.routes for poi in route)

        self.assertEqual(flat, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(child.routes), 3)

    def test_ga_decoded_result_contains_same_number_of_pois_as_input(self):
        itinerary = {
            "itinerary": {
                "Day1": {
                    "Morning": [
                        {
                            "location_id": "1",
                            "place": "A",
                            "type": "Attraction",
                            "expected_budget": "$0",
                            "rating": "4.0/5",
                            "location": {"lat": 36.1, "lon": -115.1},
                        },
                        {
                            "location_id": "2",
                            "place": "B",
                            "type": "Attraction",
                            "expected_budget": "$0",
                            "rating": "4.0/5",
                            "location": {"lat": 36.2, "lon": -115.2},
                        },
                    ],
                    "Afternoon": [],
                    "Evening": [],
                },
                "Day2": {
                    "Morning": [
                        {
                            "location_id": "3",
                            "place": "C",
                            "type": "Attraction",
                            "expected_budget": "$0",
                            "rating": "4.0/5",
                            "location": {"lat": 36.3, "lon": -115.3},
                        }
                    ],
                    "Afternoon": [
                        {
                            "location_id": "4",
                            "place": "D",
                            "type": "Attraction",
                            "expected_budget": "$0",
                            "rating": "4.0/5",
                            "location": {"lat": 36.4, "lon": -115.4},
                        }
                    ],
                    "Evening": [],
                },
            }
        }
        itinerary_nested, _structure = flatten_itinerary(itinerary)
        pois = flatten_locations(itinerary_nested)
        distance_matrix = make_matrix(5, 1000)
        time_matrix = make_matrix(5, 60)
        service_times = [0, 60, 60, 60, 60]
        original_chromosome = build_original_itinerary_chromosome(itinerary)

        ga_run = run_ga(
            poi_list=pois,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            num_days=2,
            daily_time_budget_seconds=8 * 60 * 60,
            population_size=10,
            generations=5,
            random_seed=21,
            verbose=False,
            original_chromosome=original_chromosome,
        )
        optimized_itinerary, _daily_summaries, _routes = (
            decode_chromosome_to_itinerary(
                chromosome=ga_run.best_chromosome,
                poi_list=pois,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                service_times=service_times,
                start_hour=9,
                weekday_index=0,
            )
        )

        self.assertEqual(
            len(flatten_optimized_itinerary(optimized_itinerary)),
            4,
        )


if __name__ == "__main__":
    unittest.main()
