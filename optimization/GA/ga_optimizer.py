"""Genetic Algorithm optimizer for multi-day trip itineraries.

This module keeps the existing OR-Tools optimizer untouched and provides a
separate GA implementation for the same multi-day routing problem. The code is
written to be explainable for project assessment: each stage has a clear
function, and the fitness function is explicit about every penalty term.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Allows both package imports and running from optimization/V3 directly.
    from .raw_poi_utils import (
        build_raw_lookup,
        enrich_pois_with_raw_data,
        get_opening_windows,
        infer_visit_duration_minutes_from_raw,
        load_raw_pois,
    )
    from .utils import (
        flatten_itinerary,
        flatten_locations,
        geoepify_distance_matrix,
    )
except ImportError:  # pragma: no cover - used when running this file as script.
    from raw_poi_utils import (
        build_raw_lookup,
        enrich_pois_with_raw_data,
        get_opening_windows,
        infer_visit_duration_minutes_from_raw,
        load_raw_pois,
    )
    from utils import (
        flatten_itinerary,
        flatten_locations,
        geoepify_distance_matrix,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WEIGHTS = {
    "travel_time": 1.0,
    "travel_distance": 0.001,
    "over_budget": 10_000.0,
    "opening_hours": 5_000.0,
    "balance": 2.0,
    "empty_day": 20_000.0,
}


@dataclass
class Chromosome:
    """Route-based representation of a multi-day itinerary.

    routes[d] contains the 0-based POI indices assigned to day d. The distance
    and time matrices still use node 0 for the hotel, so POI i is matrix node
    i + 1 during evaluation.
    """

    routes: list[list[int]]
    fitness: float | None = None
    cost: float | None = None
    breakdown: dict[str, float | bool] | None = None


@dataclass
class GARunResult:
    """Container returned by ``run_ga``."""

    best_chromosome: Chromosome
    history: list[dict[str, float | int]]
    metrics: dict[str, float | int | bool]


def merge_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    """Return default GA weights with any user overrides applied."""

    merged = DEFAULT_WEIGHTS.copy()

    if weights:
        for key, value in weights.items():
            if key in merged:
                merged[key] = float(value)

    return merged


def build_time_and_distance_matrices(
    locations: list[tuple[float, float]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Build Geoapify distance and time matrices once before the GA loop.

    Geoapify returns distance in meters and time in seconds. Fitness evaluation
    uses these cached matrices only; it never calls the API inside the GA.
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


def seconds_to_clock(start_hour: int, offset_seconds: float) -> str:
    """Convert route-relative seconds into a HH:MM clock string."""

    total_minutes = (start_hour * 60) + int(offset_seconds // 60)
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60

    return f"{hour:02d}:{minute:02d}"


def get_part_of_day(clock_time: str) -> str:
    """Assign a clock time to Morning, Afternoon, or Evening."""

    hour = int(clock_time.split(":")[0])

    if hour < 12:
        return "Morning"

    if hour < 17:
        return "Afternoon"

    return "Evening"


def flatten_routes(routes: list[list[int]]) -> list[int]:
    """Flatten a list of daily routes into one POI permutation."""

    return [poi_index for route in routes for poi_index in route]


def split_permutation_by_lengths(
    permutation: list[int],
    lengths: list[int],
) -> list[list[int]]:
    """Split a permutation into routes using the requested day lengths.

    The final day receives any remaining POIs. This keeps the function robust
    when crossover produces length totals that are slightly too short or long.
    """

    if not lengths:
        return [list(permutation)]

    routes: list[list[int]] = []
    cursor = 0

    for length in lengths[:-1]:
        safe_length = max(0, int(length))
        routes.append(permutation[cursor:cursor + safe_length])
        cursor += safe_length

    routes.append(permutation[cursor:])

    return routes


def clone_chromosome(chromosome: Chromosome) -> Chromosome:
    """Create a deep copy so GA operators do not mutate parents in place."""

    return Chromosome(
        routes=[list(route) for route in chromosome.routes],
        fitness=chromosome.fitness,
        cost=chromosome.cost,
        breakdown=deepcopy(chromosome.breakdown),
    )


def repair_chromosome(
    chromosome: Chromosome,
    num_pois: int,
    num_days: int,
) -> Chromosome:
    """Repair a chromosome so each POI appears exactly once.

    Repair rules:
    - The number of routes must equal ``num_days``.
    - Invalid POI indices are removed.
    - Duplicate POIs are removed after their first occurrence.
    - Missing POIs are inserted into the currently shortest day route.
    """

    normalized_routes = [list(route) for route in chromosome.routes[:num_days]]

    while len(normalized_routes) < num_days:
        normalized_routes.append([])

    if len(chromosome.routes) > num_days and num_days > 0:
        overflow = flatten_routes(chromosome.routes[num_days:])
        normalized_routes[-1].extend(overflow)

    seen: set[int] = set()
    repaired_routes: list[list[int]] = []

    for route in normalized_routes:
        repaired_route: list[int] = []

        for poi_index in route:
            if not isinstance(poi_index, int):
                continue

            if poi_index < 0 or poi_index >= num_pois:
                continue

            if poi_index in seen:
                continue

            seen.add(poi_index)
            repaired_route.append(poi_index)

        repaired_routes.append(repaired_route)

    for missing_poi in range(num_pois):
        if missing_poi in seen:
            continue

        shortest_day_index = min(
            range(num_days),
            key=lambda day_index: len(repaired_routes[day_index]),
        )
        repaired_routes[shortest_day_index].append(missing_poi)
        seen.add(missing_poi)

    return Chromosome(routes=repaired_routes)


def _normalize_lengths(
    lengths: list[int],
    total_items: int,
    num_days: int,
    rng: random.Random,
) -> list[int]:
    """Adjust route lengths so they sum to the number of POIs."""

    normalized = [max(0, int(length)) for length in lengths[:num_days]]

    while len(normalized) < num_days:
        normalized.append(0)

    while sum(normalized) > total_items and normalized:
        candidates = [
            index for index, length in enumerate(normalized)
            if length > 0
        ]
        if not candidates:
            break
        chosen = rng.choice(candidates)
        normalized[chosen] -= 1

    while sum(normalized) < total_items and normalized:
        shortest_length = min(normalized)
        candidates = [
            index for index, length in enumerate(normalized)
            if length == shortest_length
        ]
        chosen = rng.choice(candidates)
        normalized[chosen] += 1

    return normalized


def tournament_selection(
    population: list[Chromosome],
    tournament_size: int,
    rng: random.Random | None = None,
) -> Chromosome:
    """Select one parent using tournament selection.

    A random subset of the population competes, and the chromosome with the
    highest fitness wins. This balances exploration with pressure toward better
    routes.
    """

    if not population:
        raise ValueError("Cannot select from an empty population.")

    rng = rng or random
    sample_size = max(1, min(tournament_size, len(population)))
    competitors = rng.sample(population, sample_size)

    return max(
        competitors,
        key=lambda chromosome: chromosome.fitness
        if chromosome.fitness is not None
        else -math.inf,
    )


def ordered_crossover(
    parent_a: list[int],
    parent_b: list[int],
    rng: random.Random | None = None,
) -> list[int]:
    """Apply ordered crossover (OX) to two POI permutations.

    OX copies a slice from parent A and fills remaining positions using parent
    B's order. The result is a valid permutation when both parents are valid.
    """

    rng = rng or random
    size = len(parent_a)

    if size <= 1:
        return list(parent_a)

    start, end = sorted(rng.sample(range(size), 2))
    child: list[int | None] = [None] * size
    child[start:end + 1] = parent_a[start:end + 1]

    copied = set(parent_a[start:end + 1])
    fill_values = [value for value in parent_b if value not in copied]
    fill_index = 0

    for position in list(range(end + 1, size)) + list(range(0, end + 1)):
        if child[position] is None and fill_index < len(fill_values):
            child[position] = fill_values[fill_index]
            fill_index += 1

    return [value for value in child if value is not None]


def route_aware_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    num_days: int,
    rng: random.Random | None = None,
) -> Chromosome:
    """Create a child by combining permutation order and daily route lengths.

    The operator flattens both parents, applies OX to the POI order, then splits
    the child permutation using a mixture of parent day lengths.
    """

    rng = rng or random
    permutation_a = flatten_routes(parent_a.routes)
    permutation_b = flatten_routes(parent_b.routes)
    num_pois = len(permutation_a)

    if num_pois == 0:
        return Chromosome(routes=[[] for _ in range(num_days)])

    child_permutation = ordered_crossover(permutation_a, permutation_b, rng)

    parent_lengths_a = [
        len(parent_a.routes[day_index])
        if day_index < len(parent_a.routes)
        else 0
        for day_index in range(num_days)
    ]
    parent_lengths_b = [
        len(parent_b.routes[day_index])
        if day_index < len(parent_b.routes)
        else 0
        for day_index in range(num_days)
    ]

    mixed_lengths = [
        parent_lengths_a[index]
        if rng.random() < 0.5
        else parent_lengths_b[index]
        for index in range(num_days)
    ]
    mixed_lengths = _normalize_lengths(
        mixed_lengths,
        total_items=num_pois,
        num_days=num_days,
        rng=rng,
    )

    child_routes = split_permutation_by_lengths(
        child_permutation,
        mixed_lengths,
    )

    return repair_chromosome(
        Chromosome(routes=child_routes),
        num_pois=num_pois,
        num_days=num_days,
    )


def mutate(
    chromosome: Chromosome,
    mutation_rate: float,
    num_days: int,
    rng: random.Random | None = None,
) -> Chromosome:
    """Mutate a chromosome while preserving the one-visit-per-POI rule.

    Mutation types:
    - swap two POIs inside the same day
    - move one POI from one day to another
    - reverse a segment inside one day
    - swap POIs between two different days
    """

    rng = rng or random
    mutated = clone_chromosome(chromosome)
    original_pois = flatten_routes(chromosome.routes)
    num_pois = max(original_pois) + 1 if original_pois else 0

    while len(mutated.routes) < num_days:
        mutated.routes.append([])

    mutated.routes = mutated.routes[:num_days]

    if rng.random() >= mutation_rate or num_pois <= 1:
        return repair_chromosome(mutated, num_pois, num_days)

    mutation_type = rng.choice([
        "swap_within_day",
        "move_between_days",
        "reverse_segment",
        "swap_between_days",
    ])

    if mutation_type == "swap_within_day":
        candidate_days = [
            day_index for day_index, route in enumerate(mutated.routes)
            if len(route) >= 2
        ]
        if candidate_days:
            day_index = rng.choice(candidate_days)
            first, second = rng.sample(
                range(len(mutated.routes[day_index])),
                2,
            )
            route = mutated.routes[day_index]
            route[first], route[second] = route[second], route[first]

    elif mutation_type == "move_between_days" and num_days >= 2:
        source_days = [
            day_index for day_index, route in enumerate(mutated.routes)
            if route
        ]
        if source_days:
            source_day = rng.choice(source_days)
            destination_choices = [
                day_index for day_index in range(num_days)
                if day_index != source_day
            ]
            destination_day = rng.choice(destination_choices)
            poi_position = rng.randrange(len(mutated.routes[source_day]))
            poi_index = mutated.routes[source_day].pop(poi_position)
            insert_position = rng.randrange(
                len(mutated.routes[destination_day]) + 1,
            )
            mutated.routes[destination_day].insert(
                insert_position,
                poi_index,
            )

    elif mutation_type == "reverse_segment":
        candidate_days = [
            day_index for day_index, route in enumerate(mutated.routes)
            if len(route) >= 2
        ]
        if candidate_days:
            day_index = rng.choice(candidate_days)
            start, end = sorted(
                rng.sample(range(len(mutated.routes[day_index])), 2),
            )
            route = mutated.routes[day_index]
            route[start:end + 1] = reversed(route[start:end + 1])

    elif mutation_type == "swap_between_days" and num_days >= 2:
        candidate_days = [
            day_index for day_index, route in enumerate(mutated.routes)
            if route
        ]
        if len(candidate_days) >= 2:
            day_a, day_b = rng.sample(candidate_days, 2)
            position_a = rng.randrange(len(mutated.routes[day_a]))
            position_b = rng.randrange(len(mutated.routes[day_b]))
            route_a = mutated.routes[day_a]
            route_b = mutated.routes[day_b]
            route_a[position_a], route_b[position_b] = (
                route_b[position_b],
                route_a[position_a],
            )

    return repair_chromosome(mutated, num_pois, num_days)


def build_original_itinerary_chromosome(itinerary: dict[str, Any] | str) -> Chromosome:
    """Build a chromosome that preserves the original LLM itinerary order."""

    itinerary_nested, _original_structure = flatten_itinerary(itinerary)
    routes: list[list[int]] = []
    next_poi_index = 0

    for day in itinerary_nested:
        day_route: list[int] = []

        for part_of_day in day:
            for _poi in part_of_day:
                day_route.append(next_poi_index)
                next_poi_index += 1

        routes.append(day_route)

    return Chromosome(routes=routes)


def _balanced_random_chromosome(
    num_pois: int,
    num_days: int,
    rng: random.Random,
) -> Chromosome:
    """Create a random chromosome with POIs spread roughly evenly by day."""

    indices = list(range(num_pois))
    rng.shuffle(indices)
    day_order = list(range(num_days))
    rng.shuffle(day_order)

    routes = [[] for _ in range(num_days)]

    for position, poi_index in enumerate(indices):
        day_index = day_order[position % num_days]
        routes[day_index].append(poi_index)

    return Chromosome(routes=routes)


def _original_distribution_chromosome(
    num_pois: int,
    num_days: int,
    original_chromosome: Chromosome,
    rng: random.Random,
) -> Chromosome:
    """Shuffle POIs while preserving the original number of POIs per day."""

    permutation = list(range(num_pois))
    rng.shuffle(permutation)
    original_lengths = [
        len(original_chromosome.routes[day_index])
        if day_index < len(original_chromosome.routes)
        else 0
        for day_index in range(num_days)
    ]
    lengths = _normalize_lengths(original_lengths, num_pois, num_days, rng)

    return Chromosome(routes=split_permutation_by_lengths(permutation, lengths))


def _fully_random_chromosome(
    num_pois: int,
    num_days: int,
    rng: random.Random,
) -> Chromosome:
    """Create a random chromosome that may be intentionally unbalanced."""

    indices = list(range(num_pois))
    rng.shuffle(indices)
    routes = [[] for _ in range(num_days)]

    for poi_index in indices:
        routes[rng.randrange(num_days)].append(poi_index)

    return Chromosome(routes=routes)


def generate_initial_population(
    num_pois: int,
    num_days: int,
    population_size: int,
    original_chromosome: Chromosome,
    rng: random.Random,
) -> list[Chromosome]:
    """Generate a diverse initial population for the GA.

    The population includes:
    - one chromosome matching the original itinerary
    - balanced random chromosomes
    - chromosomes with the original day-size distribution
    - fully random chromosomes for diversity
    """

    if population_size < 1:
        raise ValueError("population_size must be at least 1.")

    population = [
        repair_chromosome(original_chromosome, num_pois, num_days)
    ]

    while len(population) < population_size:
        ratio = len(population) / population_size

        if ratio < 0.35:
            candidate = _balanced_random_chromosome(num_pois, num_days, rng)
        elif ratio < 0.70:
            candidate = _original_distribution_chromosome(
                num_pois,
                num_days,
                original_chromosome,
                rng,
            )
        else:
            candidate = _fully_random_chromosome(num_pois, num_days, rng)

        population.append(
            repair_chromosome(candidate, num_pois, num_days),
        )

    return population


def _opening_hours_violation(
    poi: Any,
    arrival_seconds: float,
    visit_seconds: float,
    start_hour: int,
    weekday_index: int,
) -> tuple[bool, bool]:
    """Return ``(has_opening_hours, violates_hours)`` for one arrival."""

    raw_data = getattr(poi, "raw_data", None)
    windows = get_opening_windows(raw_data, weekday_index)

    if not windows:
        return False, False

    arrival_minutes = start_hour * 60 + int(arrival_seconds // 60)
    departure_minutes = start_hour * 60 + int(
        (arrival_seconds + visit_seconds) // 60,
    )

    for window in windows:
        try:
            open_minutes = int(window["open_time"])
            close_minutes = int(window["close_time"])
        except (KeyError, TypeError, ValueError):
            continue

        if close_minutes < open_minutes:
            close_minutes += 24 * 60

        candidate_arrival = arrival_minutes
        candidate_departure = departure_minutes

        if close_minutes >= 24 * 60 and candidate_arrival < open_minutes:
            candidate_arrival += 24 * 60
            candidate_departure += 24 * 60

        if (
            open_minutes <= candidate_arrival
            and candidate_departure <= close_minutes
        ):
            return True, False

    return True, True


def evaluate_chromosome(
    chromosome: Chromosome,
    distance_matrix: list[list[int]],
    time_matrix: list[list[int]],
    service_times: list[int],
    poi_list: list[Any],
    daily_time_budget_seconds: int,
    start_hour: int,
    weekday_index: int,
    weights: dict[str, float] | None = None,
) -> Chromosome:
    """Evaluate one chromosome and attach cost, fitness, and breakdown.

    The GA minimizes ``cost`` but stores ``fitness = 1 / (1 + cost)`` so higher
    values are better during selection.
    """

    weights = merge_weights(weights)
    num_days = len(chromosome.routes)
    num_pois = len(poi_list)
    repaired = repair_chromosome(chromosome, num_pois, num_days)

    total_travel_seconds = 0.0
    total_distance_meters = 0.0
    total_service_seconds = 0.0
    over_budget_minutes = 0.0
    opening_hour_violations = 0.0
    empty_days = 0.0
    day_total_minutes: list[float] = []

    for route in repaired.routes:
        previous_node = 0
        day_elapsed_seconds = 0.0
        day_travel_seconds = 0.0
        day_distance_meters = 0.0
        day_service_seconds = 0.0

        if not route:
            empty_days += 1.0

        for poi_index in route:
            node = poi_index + 1
            travel_seconds = time_matrix[previous_node][node]
            travel_distance = distance_matrix[previous_node][node]

            arrival_seconds = day_elapsed_seconds + travel_seconds
            visit_seconds = service_times[node]
            _has_hours, violates_hours = _opening_hours_violation(
                poi=poi_list[poi_index],
                arrival_seconds=arrival_seconds,
                visit_seconds=visit_seconds,
                start_hour=start_hour,
                weekday_index=weekday_index,
            )

            if violates_hours:
                opening_hour_violations += 1.0

            day_elapsed_seconds = arrival_seconds + visit_seconds
            day_travel_seconds += travel_seconds
            day_distance_meters += travel_distance
            day_service_seconds += visit_seconds
            previous_node = node

        return_seconds = time_matrix[previous_node][0]
        return_distance = distance_matrix[previous_node][0]
        day_elapsed_seconds += return_seconds
        day_travel_seconds += return_seconds
        day_distance_meters += return_distance

        over_budget_seconds = max(
            0.0,
            day_elapsed_seconds - daily_time_budget_seconds,
        )
        over_budget_minutes += over_budget_seconds / 60.0
        day_total_minutes.append(day_elapsed_seconds / 60.0)

        total_travel_seconds += day_travel_seconds
        total_distance_meters += day_distance_meters
        total_service_seconds += day_service_seconds

    balance_penalty_minutes = (
        statistics.pstdev(day_total_minutes)
        if len(day_total_minutes) > 1
        else 0.0
    )
    travel_time_minutes = total_travel_seconds / 60.0
    total_route_time_minutes = (
        total_travel_seconds + total_service_seconds
    ) / 60.0

    cost = (
        weights["travel_time"] * total_route_time_minutes
        + weights["travel_distance"] * total_distance_meters
        + weights["over_budget"] * over_budget_minutes
        + weights["opening_hours"] * opening_hour_violations
        + weights["balance"] * balance_penalty_minutes
        + weights["empty_day"] * empty_days
    )

    fitness = 1.0 / (1.0 + max(cost, 0.0))

    repaired.cost = float(cost)
    repaired.fitness = float(fitness)
    repaired.breakdown = {
        "travel_time_minutes": round(travel_time_minutes, 3),
        "travel_distance_km": round(total_distance_meters / 1000.0, 3),
        "total_route_time_minutes": round(total_route_time_minutes, 3),
        "daily_time_budget_penalty": round(over_budget_minutes, 3),
        "opening_hours_penalty": round(opening_hour_violations, 3),
        "balance_penalty": round(balance_penalty_minutes, 3),
        "empty_day_penalty": round(empty_days, 3),
        "total_cost": round(cost, 3),
        "feasible": (
            over_budget_minutes == 0.0
            and opening_hour_violations == 0.0
            and empty_days == 0.0
        ),
    }

    return repaired


def evaluate_population(
    population: list[Chromosome],
    distance_matrix: list[list[int]],
    time_matrix: list[list[int]],
    service_times: list[int],
    poi_list: list[Any],
    daily_time_budget_seconds: int,
    start_hour: int,
    weekday_index: int,
    weights: dict[str, float],
) -> list[Chromosome]:
    """Evaluate every chromosome in a population."""

    return [
        evaluate_chromosome(
            chromosome=chromosome,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            poi_list=poi_list,
            daily_time_budget_seconds=daily_time_budget_seconds,
            start_hour=start_hour,
            weekday_index=weekday_index,
            weights=weights,
        )
        for chromosome in population
    ]


def _population_stats(
    generation: int,
    population: list[Chromosome],
) -> dict[str, float | int]:
    """Return best and average metrics for one generation."""

    costs = [
        chromosome.cost
        for chromosome in population
        if chromosome.cost is not None
    ]
    fitness_values = [
        chromosome.fitness
        for chromosome in population
        if chromosome.fitness is not None
    ]

    return {
        "generation": generation,
        "best_cost": round(min(costs), 6),
        "average_cost": round(sum(costs) / len(costs), 6),
        "best_fitness": round(max(fitness_values), 10),
        "average_fitness": round(
            sum(fitness_values) / len(fitness_values),
            10,
        ),
    }


def run_ga(
    poi_list: list[Any],
    distance_matrix: list[list[int]],
    time_matrix: list[list[int]],
    service_times: list[int],
    num_days: int,
    daily_time_budget_seconds: int,
    start_hour: int = 9,
    weekday_index: int = 0,
    population_size: int = 80,
    generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    elitism_size: int = 4,
    tournament_size: int = 5,
    random_seed: int | None = None,
    weights: dict[str, float] | None = None,
    save_history: bool = True,
    early_stopping_patience: int = 40,
    verbose: bool = True,
    original_chromosome: Chromosome | None = None,
) -> GARunResult:
    """Run the Genetic Algorithm and return the best chromosome found."""

    if num_days < 1:
        raise ValueError("num_days must be at least 1.")

    if population_size < 1:
        raise ValueError("population_size must be at least 1.")

    rng = random.Random(random_seed)
    weights = merge_weights(weights)
    num_pois = len(poi_list)
    elitism_size = max(0, min(elitism_size, population_size))
    tournament_size = max(1, tournament_size)
    generations = max(0, generations)

    if original_chromosome is None:
        original_chromosome = Chromosome(
            routes=[list(range(num_pois))] + [[] for _ in range(num_days - 1)],
        )

    population = generate_initial_population(
        num_pois=num_pois,
        num_days=num_days,
        population_size=population_size,
        original_chromosome=original_chromosome,
        rng=rng,
    )
    population = evaluate_population(
        population=population,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        service_times=service_times,
        poi_list=poi_list,
        daily_time_budget_seconds=daily_time_budget_seconds,
        start_hour=start_hour,
        weekday_index=weekday_index,
        weights=weights,
    )
    population.sort(
        key=lambda chromosome: (
            chromosome.cost
            if chromosome.cost is not None
            else math.inf
        ),
    )

    best = clone_chromosome(population[0])
    best_generation = 0
    no_improvement_count = 0
    history: list[dict[str, float | int]] = []

    initial_stats = _population_stats(0, population)
    if save_history:
        history.append(initial_stats)

    if verbose:
        print(
            "Generation 0 | "
            f"Best cost: {initial_stats['best_cost']} | "
            f"Avg cost: {initial_stats['average_cost']} | "
            f"Best fitness: {initial_stats['best_fitness']}"
        )

    completed_generation = 0

    for generation in range(1, generations + 1):
        new_population = [
            clone_chromosome(chromosome)
            for chromosome in population[:elitism_size]
        ]

        while len(new_population) < population_size:
            parent_a = tournament_selection(population, tournament_size, rng)
            parent_b = tournament_selection(population, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = route_aware_crossover(
                    parent_a,
                    parent_b,
                    num_days=num_days,
                    rng=rng,
                )
            else:
                child = clone_chromosome(
                    parent_a
                    if (
                        parent_a.cost
                        if parent_a.cost is not None
                        else math.inf
                    )
                    <= (
                        parent_b.cost
                        if parent_b.cost is not None
                        else math.inf
                    )
                    else parent_b,
                )

            child = mutate(
                chromosome=child,
                mutation_rate=mutation_rate,
                num_days=num_days,
                rng=rng,
            )
            child = repair_chromosome(child, num_pois, num_days)
            new_population.append(child)

        population = evaluate_population(
            population=new_population,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            poi_list=poi_list,
            daily_time_budget_seconds=daily_time_budget_seconds,
            start_hour=start_hour,
            weekday_index=weekday_index,
            weights=weights,
        )
        population.sort(
            key=lambda chromosome: (
                chromosome.cost
                if chromosome.cost is not None
                else math.inf
            ),
        )

        generation_best = population[0]
        generation_best_cost = (
            generation_best.cost
            if generation_best.cost is not None
            else math.inf
        )
        best_cost = best.cost if best.cost is not None else math.inf

        if generation_best_cost < best_cost - 1e-9:
            best = clone_chromosome(generation_best)
            best_generation = generation
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        stats = _population_stats(generation, population)
        if save_history:
            history.append(stats)

        completed_generation = generation

        if verbose and (
            generation % 20 == 0
            or generation == generations
            or no_improvement_count == early_stopping_patience
        ):
            print(
                f"Generation {generation} | "
                f"Best cost: {stats['best_cost']} | "
                f"Avg cost: {stats['average_cost']} | "
                f"Best fitness: {stats['best_fitness']}"
            )

        if (
            early_stopping_patience > 0
            and no_improvement_count >= early_stopping_patience
        ):
            break

    metrics = {
        "best_fitness": round(best.fitness or 0.0, 10),
        "best_cost": round(best.cost or 0.0, 3),
        "best_generation": best_generation,
        "convergence_generation": best_generation,
        "completed_generations": completed_generation,
        "feasible_solution_found": bool(
            best.breakdown and best.breakdown.get("feasible")
        ),
    }

    return GARunResult(
        best_chromosome=best,
        history=history if save_history else [],
        metrics=metrics,
    )


def decode_chromosome_to_itinerary(
    chromosome: Chromosome,
    poi_list: list[Any],
    distance_matrix: list[list[int]],
    time_matrix: list[list[int]],
    service_times: list[int],
    start_hour: int,
    weekday_index: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert a chromosome back into itinerary JSON structures."""

    optimized_itinerary: dict[str, dict[str, list[dict[str, Any]]]] = {}
    daily_summaries: list[dict[str, Any]] = []
    routes_output: list[dict[str, Any]] = []

    for day_index, route in enumerate(chromosome.routes):
        day_key = f"Day{day_index + 1}"
        optimized_itinerary[day_key] = {
            "Morning": [],
            "Afternoon": [],
            "Evening": [],
        }

        previous_node = 0
        day_elapsed_seconds = 0.0
        route_distance_meters = 0.0
        route_travel_seconds = 0.0
        route_visit_seconds = 0.0
        route_items: list[dict[str, Any]] = []

        for poi_index in route:
            node = poi_index + 1
            travel_seconds = time_matrix[previous_node][node]
            travel_distance = distance_matrix[previous_node][node]
            arrival_seconds = day_elapsed_seconds + travel_seconds
            visit_seconds = service_times[node]
            departure_seconds = arrival_seconds + visit_seconds

            arrival_time = seconds_to_clock(start_hour, arrival_seconds)
            departure_time = seconds_to_clock(start_hour, departure_seconds)
            has_hours, violates_hours = _opening_hours_violation(
                poi=poi_list[poi_index],
                arrival_seconds=arrival_seconds,
                visit_seconds=visit_seconds,
                start_hour=start_hour,
                weekday_index=weekday_index,
            )

            item = poi_list[poi_index].to_dict()
            item["arrival_time"] = arrival_time
            item["departure_time"] = departure_time
            item["visit_duration_minutes"] = int(visit_seconds / 60)
            item["travel_from_previous_minutes"] = round(
                travel_seconds / 60.0,
                1,
            )
            item["travel_from_previous_km"] = round(
                travel_distance / 1000.0,
                2,
            )
            item["raw_data_matched"] = getattr(
                poi_list[poi_index],
                "raw_data",
                None,
            ) is not None
            item["opening_hours_constrained"] = has_hours
            item["opening_hours_violation"] = violates_hours

            part_of_day = get_part_of_day(arrival_time)
            optimized_itinerary[day_key][part_of_day].append(item)
            route_items.append(item)

            day_elapsed_seconds = departure_seconds
            route_distance_meters += travel_distance
            route_travel_seconds += travel_seconds
            route_visit_seconds += visit_seconds
            previous_node = node

        return_seconds = time_matrix[previous_node][0]
        return_distance = distance_matrix[previous_node][0]
        day_elapsed_seconds += return_seconds
        route_distance_meters += return_distance
        route_travel_seconds += return_seconds

        day_summary = {
            "day": day_key,
            "total_time_minutes": round(day_elapsed_seconds / 60.0, 1),
            "total_time_hours": round(day_elapsed_seconds / 3600.0, 2),
            "travel_time_minutes": round(route_travel_seconds / 60.0, 1),
            "visit_time_minutes": round(route_visit_seconds / 60.0, 1),
            "total_distance_km": round(route_distance_meters / 1000.0, 2),
            "number_of_pois": len(route_items),
            "starts_at": seconds_to_clock(start_hour, 0),
            "ends_at": seconds_to_clock(start_hour, day_elapsed_seconds),
            "return_to_hotel_minutes": round(return_seconds / 60.0, 1),
        }

        daily_summaries.append(day_summary)
        routes_output.append({
            "day": day_key,
            "route": route_items,
            "summary": day_summary,
        })

    return optimized_itinerary, daily_summaries, routes_output


def _daily_balance_minutes(daily_summaries: list[dict[str, Any]]) -> float:
    """Return population standard deviation of daily route times."""

    route_times = [
        float(summary.get("total_time_minutes", 0.0))
        for summary in daily_summaries
    ]

    if len(route_times) <= 1:
        return 0.0

    return round(statistics.pstdev(route_times), 3)


def evaluate_original_baseline(
    itinerary: dict[str, Any] | str,
    distance_matrix: list[list[int]],
    time_matrix: list[list[int]],
    service_times: list[int],
    poi_list: list[Any],
    daily_time_budget_seconds: int,
    start_hour: int,
    weekday_index: int,
    weights: dict[str, float] | None = None,
) -> Chromosome:
    """Evaluate the original LLM itinerary as a baseline chromosome."""

    original_chromosome = build_original_itinerary_chromosome(itinerary)

    return evaluate_chromosome(
        chromosome=original_chromosome,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        service_times=service_times,
        poi_list=poi_list,
        daily_time_budget_seconds=daily_time_budget_seconds,
        start_hour=start_hour,
        weekday_index=weekday_index,
        weights=weights,
    )


def _normalize_raw_poi_files(
    raw_poi_files: str | os.PathLike[str] | list[str | os.PathLike[str]] | None,
) -> list[Path]:
    """Return the raw POI files to load, skipping missing paths safely."""

    if raw_poi_files is None:
        candidate_files = [
            PROJECT_ROOT / "data" / "raw" / "las_vegas" / "attractions.json",
            PROJECT_ROOT / "data" / "raw" / "las_vegas" / "restaurants.json",
        ]
    elif isinstance(raw_poi_files, (str, os.PathLike)):
        candidate_files = [Path(raw_poi_files)]
    else:
        candidate_files = [Path(file_path) for file_path in raw_poi_files]

    return [
        file_path
        for file_path in candidate_files
        if file_path.exists()
    ]


def _load_and_enrich_pois(
    poi_list: list[Any],
    raw_poi_files: str | os.PathLike[str] | list[str | os.PathLike[str]] | None,
) -> tuple[list[Any], list[dict[str, Any]], list[Path]]:
    """Load raw POI data and attach matched raw records to POI objects."""

    usable_raw_poi_files = _normalize_raw_poi_files(raw_poi_files)

    if not usable_raw_poi_files:
        return poi_list, [], []

    raw_records = load_raw_pois(usable_raw_poi_files)
    raw_lookup = build_raw_lookup(raw_records)

    return (
        enrich_pois_with_raw_data(poi_list, raw_lookup),
        raw_records,
        usable_raw_poi_files,
    )


def solve_multi_day_ga(
    itinerary: dict[str, Any] | str,
    hotel_lat: float,
    hotel_lon: float,
    raw_poi_files: str | os.PathLike[str] | list[str | os.PathLike[str]] | None = None,
    daily_time_budget_hours: float = 8,
    start_hour: int = 9,
    weekday_index: int = 0,
    population_size: int = 80,
    generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    elitism_size: int = 4,
    tournament_size: int = 5,
    random_seed: int | None = None,
    weights: dict[str, float] | None = None,
    save_history: bool = True,
    early_stopping_patience: int = 40,
    verbose: bool = True,
) -> str:
    """Optimize a multi-day itinerary using a Genetic Algorithm.

    Returns a JSON string with the optimized itinerary, route summaries, GA
    metrics, fitness breakdown, and baseline comparison.
    """

    start_time = time.perf_counter()
    weights = merge_weights(weights)
    itinerary_nested, _original_structure = flatten_itinerary(itinerary)
    poi_list = flatten_locations(itinerary_nested)
    num_days = len(itinerary_nested)

    if not poi_list:
        return json.dumps(
            {"error": "No POIs found in itinerary"},
            indent=2,
        )

    poi_list, raw_records, usable_raw_poi_files = _load_and_enrich_pois(
        poi_list=poi_list,
        raw_poi_files=raw_poi_files,
    )
    matched_raw_pois = sum(
        1 for poi in poi_list
        if getattr(poi, "raw_data", None)
    )

    locations = [(hotel_lat, hotel_lon)] + [
        (poi.lat, poi.lon)
        for poi in poi_list
    ]
    distance_matrix, time_matrix = build_time_and_distance_matrices(locations)

    service_times = [0] + [
        int(infer_visit_duration_minutes_from_raw(poi) * 60)
        for poi in poi_list
    ]

    for poi, service_time in zip(poi_list, service_times[1:]):
        poi.visit_duration_minutes = int(service_time / 60)

    daily_time_budget_seconds = int(daily_time_budget_hours * 60 * 60)
    original_chromosome = build_original_itinerary_chromosome(itinerary)
    original_baseline = evaluate_original_baseline(
        itinerary=itinerary,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        service_times=service_times,
        poi_list=poi_list,
        daily_time_budget_seconds=daily_time_budget_seconds,
        start_hour=start_hour,
        weekday_index=weekday_index,
        weights=weights,
    )

    ga_run = run_ga(
        poi_list=poi_list,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        service_times=service_times,
        num_days=num_days,
        daily_time_budget_seconds=daily_time_budget_seconds,
        start_hour=start_hour,
        weekday_index=weekday_index,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=elitism_size,
        tournament_size=tournament_size,
        random_seed=random_seed,
        weights=weights,
        save_history=save_history,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        original_chromosome=original_chromosome,
    )

    optimized_itinerary, daily_summaries, routes = decode_chromosome_to_itinerary(
        chromosome=ga_run.best_chromosome,
        poi_list=poi_list,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        service_times=service_times,
        start_hour=start_hour,
        weekday_index=weekday_index,
    )
    _original_itinerary, original_daily_summaries, _original_routes = (
        decode_chromosome_to_itinerary(
            chromosome=original_baseline,
            poi_list=poi_list,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            service_times=service_times,
            start_hour=start_hour,
            weekday_index=weekday_index,
        )
    )

    runtime_seconds = time.perf_counter() - start_time
    ga_metrics = dict(ga_run.metrics)
    ga_metrics["runtime_seconds"] = round(runtime_seconds, 3)

    original_cost = float(original_baseline.cost or 0.0)
    optimized_cost = float(ga_run.best_chromosome.cost or 0.0)
    improvement_percentage = (
        ((original_cost - optimized_cost) / original_cost) * 100.0
        if original_cost > 0
        else 0.0
    )

    result = {
        "optimization_type": "multi_day_ga_v1",
        "description": (
            "Genetic Algorithm optimizer for a multi-day tourist itinerary. "
            "It reorders and redistributes POIs across days while minimizing "
            "travel time, travel distance, time-budget violations, opening-hour "
            "violations, and day imbalance."
        ),
        "hotel": {
            "lat": hotel_lat,
            "lon": hotel_lon,
        },
        "settings": {
            "daily_time_budget_hours": daily_time_budget_hours,
            "start_hour": start_hour,
            "weekday_index": weekday_index,
            "number_of_days": num_days,
            "total_pois": len(poi_list),
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "elitism_size": elitism_size,
            "tournament_size": tournament_size,
            "early_stopping_patience": early_stopping_patience,
            "random_seed": random_seed,
            "weights": weights,
            "raw_poi_records_loaded": len(raw_records),
            "matched_raw_pois": matched_raw_pois,
            "raw_poi_files": [
                str(file_path)
                for file_path in usable_raw_poi_files
            ],
        },
        "ga_metrics": ga_metrics,
        "fitness_breakdown": ga_run.best_chromosome.breakdown or {},
        "baseline_comparison": {
            "original_itinerary_cost": round(original_cost, 3),
            "ga_optimized_cost": round(optimized_cost, 3),
            "improvement_percentage": round(improvement_percentage, 2),
            "original_daily_time_balance_minutes": _daily_balance_minutes(
                original_daily_summaries,
            ),
            "optimized_daily_time_balance_minutes": _daily_balance_minutes(
                daily_summaries,
            ),
            "original_fitness_breakdown": original_baseline.breakdown or {},
        },
        "optimized_itinerary": optimized_itinerary,
        "daily_summaries": daily_summaries,
        "routes": routes,
        "evolution_history": ga_run.history if save_history else [],
    }

    return json.dumps(result, indent=2)


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable route metrics from a GA or OR-Tools result."""

    daily_summaries = result.get("daily_summaries", [])
    total_distance_km = sum(
        float(summary.get("total_distance_km", 0.0))
        for summary in daily_summaries
    )
    total_time_minutes = sum(
        float(summary.get("total_time_minutes", 0.0))
        for summary in daily_summaries
    )
    total_pois = sum(
        int(summary.get("number_of_pois", 0))
        for summary in daily_summaries
    )
    route_times = [
        float(summary.get("total_time_minutes", 0.0))
        for summary in daily_summaries
    ]

    feasible = bool(
        result.get("ga_metrics", {}).get("feasible_solution_found", True)
    )

    return {
        "optimization_type": result.get("optimization_type"),
        "total_distance_km": round(total_distance_km, 2),
        "total_time_minutes": round(total_time_minutes, 1),
        "total_pois": total_pois,
        "daily_balance_minutes": (
            round(statistics.pstdev(route_times), 3)
            if len(route_times) > 1
            else 0.0
        ),
        "feasible": feasible,
        "daily_summaries": daily_summaries,
    }


def compare_ga_with_ortools(
    ga_result: dict[str, Any] | str,
    ortools_result: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Compare a GA result with OR-Tools, or with the original baseline.

    If ``ortools_result`` is omitted, the function uses the
    ``baseline_comparison`` block embedded in the GA output.
    """

    if isinstance(ga_result, str):
        ga_result = json.loads(ga_result)

    ga_summary = _result_summary(ga_result)

    if ortools_result is not None:
        if isinstance(ortools_result, str):
            ortools_result = json.loads(ortools_result)

        ortools_summary = _result_summary(ortools_result)

        return {
            "comparison_type": "ga_vs_ortools",
            "ga": ga_summary,
            "ortools": ortools_summary,
            "differences": {
                "distance_km": round(
                    ga_summary["total_distance_km"]
                    - ortools_summary["total_distance_km"],
                    2,
                ),
                "time_minutes": round(
                    ga_summary["total_time_minutes"]
                    - ortools_summary["total_time_minutes"],
                    1,
                ),
                "daily_balance_minutes": round(
                    ga_summary["daily_balance_minutes"]
                    - ortools_summary["daily_balance_minutes"],
                    3,
                ),
                "poi_count": (
                    ga_summary["total_pois"]
                    - ortools_summary["total_pois"]
                ),
            },
        }

    baseline = ga_result.get("baseline_comparison", {})

    return {
        "comparison_type": "ga_vs_original_baseline",
        "ga": ga_summary,
        "original_baseline": baseline,
        "differences": {
            "cost_improvement_percentage": baseline.get(
                "improvement_percentage",
                0.0,
            ),
            "daily_balance_delta_minutes": round(
                baseline.get("optimized_daily_time_balance_minutes", 0.0)
                - baseline.get("original_daily_time_balance_minutes", 0.0),
                3,
            ),
        },
    }


def plot_evolution_history(history: list[dict[str, Any]]) -> Any:
    """Plot best and average GA cost over generations using matplotlib."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "matplotlib is required for plotting evolution history."
        ) from exc

    generations = [entry["generation"] for entry in history]
    best_costs = [entry["best_cost"] for entry in history]
    average_costs = [entry["average_cost"] for entry in history]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(generations, best_costs, label="Best cost", linewidth=2)
    axis.plot(generations, average_costs, label="Average cost", linewidth=2)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Cost")
    axis.set_title("GA Evolution History")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    return figure


def plot_daily_route_times(daily_summaries: list[dict[str, Any]]) -> Any:
    """Plot optimized route time by day using matplotlib."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "matplotlib is required for plotting daily route times."
        ) from exc

    days = [summary["day"] for summary in daily_summaries]
    route_times = [
        summary["total_time_minutes"]
        for summary in daily_summaries
    ]

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(days, route_times)
    axis.set_xlabel("Day")
    axis.set_ylabel("Total route time (minutes)")
    axis.set_title("Optimized Daily Route Times")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    return figure
