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

