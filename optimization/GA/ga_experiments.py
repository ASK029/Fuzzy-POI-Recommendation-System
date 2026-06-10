"""Command-line and notebook-friendly runner for the GA itinerary optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .ga_optimizer import (
        compare_ga_with_ortools,
        plot_daily_route_times,
        plot_evolution_history,
        solve_multi_day_ga,
    )
except ImportError:  # pragma: no cover - used when running this file directly.
    from ga_optimizer import (
        compare_ga_with_ortools,
        plot_daily_route_times,
        plot_evolution_history,
        solve_multi_day_ga,
    )


CURRENT_DIR = Path(__file__).resolve().parent


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a Python dictionary to a formatted JSON file."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def print_daily_summaries(daily_summaries: list[dict[str, Any]]) -> None:
    """Print compact daily route summaries for quick CLI inspection."""

    print("\nOptimized daily summaries:")

    for summary in daily_summaries:
        print(
            f"- {summary['day']}: "
            f"{summary['number_of_pois']} POIs, "
            f"{summary['total_time_minutes']} min, "
            f"{summary['total_distance_km']} km, "
            f"{summary['starts_at']} -> {summary['ends_at']}"
        )


def run_experiment(
    itinerary_path: str | Path = CURRENT_DIR / "itinerary.json",
    hotel_lat: float = 36.1147,
    hotel_lon: float = -115.1728,
    raw_poi_files: list[str | Path] | None = None,
    output_path: str | Path = CURRENT_DIR / "ga_results.json",
    daily_time_budget_hours: float = 8,
    start_hour: int = 9,
    weekday_index: int = 0,
    population_size: int = 80,
    generations: int = 200,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    elitism_size: int = 4,
    tournament_size: int = 5,
    random_seed: int | None = 42,
    early_stopping_patience: int = 40,
    save_history: bool = True,
    verbose: bool = True,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Run one GA optimization experiment and save the result JSON.

    This function is intentionally notebook-friendly: import it, pass parameter
    values, and it returns the parsed result dictionary for further analysis.
    """

    itinerary = load_json(itinerary_path)
    result_json = solve_multi_day_ga(
        itinerary=itinerary,
        hotel_lat=hotel_lat,
        hotel_lon=hotel_lon,
        raw_poi_files=raw_poi_files,
        daily_time_budget_hours=daily_time_budget_hours,
        start_hour=start_hour,
        weekday_index=weekday_index,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=elitism_size,
        tournament_size=tournament_size,
        random_seed=random_seed,
        save_history=save_history,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )
    result = json.loads(result_json)
    save_json(result, output_path)

    print(f"\nGA result saved to {output_path}")
    print_daily_summaries(result.get("daily_summaries", []))

    if create_plots:
        _save_plots(result, Path(output_path).parent)

    comparison = compare_ga_with_ortools(result)
    print("\nBaseline comparison:")
    print(json.dumps(comparison.get("differences", {}), indent=2))

    return result


def _save_plots(result: dict[str, Any], output_dir: Path) -> None:
    """Create and save matplotlib plots if matplotlib is installed."""

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        history = result.get("evolution_history", [])
        if history:
            evolution_figure = plot_evolution_history(history)
            evolution_path = output_dir / "ga_evolution_history.png"
            evolution_figure.savefig(evolution_path, dpi=150)
            print(f"Evolution plot saved to {evolution_path}")

        daily_summaries = result.get("daily_summaries", [])
        if daily_summaries:
            route_times_figure = plot_daily_route_times(daily_summaries)
            route_times_path = output_dir / "ga_daily_route_times.png"
            route_times_figure.savefig(route_times_path, dpi=150)
            print(f"Daily route-time plot saved to {route_times_path}")

    except ImportError as exc:
        print(f"Skipping plots: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for GA experiments."""

    parser = argparse.ArgumentParser(
        description="Run the Genetic Algorithm multi-day itinerary optimizer.",
    )
    parser.add_argument(
        "--itinerary",
        default=str(CURRENT_DIR / "itinerary.json"),
        help="Path to the itinerary JSON file.",
    )
    parser.add_argument(
        "--hotel-lat",
        type=float,
        default=36.1147,
        help="Hotel latitude.",
    )
    parser.add_argument(
        "--hotel-lon",
        type=float,
        default=-115.1728,
        help="Hotel longitude.",
    )
    parser.add_argument(
        "--raw-poi-files",
        nargs="*",
        default=None,
        help="Optional raw POI JSON files for enrichment.",
    )
    parser.add_argument(
        "--output",
        default=str(CURRENT_DIR / "ga_results.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--daily-time-budget-hours",
        type=float,
        default=8,
        help="Daily route budget in hours.",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=9,
        help="Route start hour using 24-hour clock.",
    )
    parser.add_argument(
        "--weekday-index",
        type=int,
        default=0,
        help="Weekday index used for opening-hour windows.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=80,
        help="GA population size.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=200,
        help="Maximum number of GA generations.",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.9,
        help="Probability of applying crossover.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.2,
        help="Probability of mutating a child chromosome.",
    )
    parser.add_argument(
        "--elitism-size",
        type=int,
        default=4,
        help="Number of best chromosomes copied unchanged each generation.",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=5,
        help="Number of chromosomes sampled for tournament selection.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible GA runs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=40,
        help="Stop after this many generations without improvement.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not include generation history in the result JSON.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save matplotlib plots.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable GA progress logging.",
    )

    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()

    run_experiment(
        itinerary_path=args.itinerary,
        hotel_lat=args.hotel_lat,
        hotel_lon=args.hotel_lon,
        raw_poi_files=args.raw_poi_files,
        output_path=args.output,
        daily_time_budget_hours=args.daily_time_budget_hours,
        start_hour=args.start_hour,
        weekday_index=args.weekday_index,
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_size=args.elitism_size,
        tournament_size=args.tournament_size,
        random_seed=args.random_seed,
        early_stopping_patience=args.early_stopping_patience,
        save_history=not args.no_history,
        verbose=not args.quiet,
        create_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
