import argparse
from pathlib import Path

from rocket_landing.baseline import (
    DEFAULT_REPORT_PATH,
    print_baseline_summary,
    run_baseline,
)
from rocket_landing.check_quality import (
    print_quality_result,
    run_quality_check,
)
from rocket_landing.main import (
    play_random_episode,
    print_episode_result,
)


def handle_play(args: argparse.Namespace) -> int:


    result = play_random_episode(
        seed=args.seed,
        max_steps=args.max_steps,
    )

    print_episode_result(result)

    return 0


def handle_baseline(args: argparse.Namespace) -> int:
  

    summary, report_path = run_baseline(
        episodes=args.episodes,
        base_seed=args.seed,
        max_steps=args.max_steps,
        output_path=args.output,
    )

    print_baseline_summary(
        summary=summary,
        report_path=report_path,
    )

    return 0


def handle_check_quality(args: argparse.Namespace) -> int:


    try:
        summary, result = run_quality_check(
            report_path=args.report,
            minimum_mean_reward=args.minimum_mean_reward,
            maximum_truncated_episodes=args.maximum_truncated,
            minimum_successful_episodes=args.minimum_successful,
        )
    except FileNotFoundError as error:
        print(error)
        return 2

    print_quality_result(
        summary=summary,
        result=result,
        report_path=args.report,
    )

    if result.passed:
        return 0

    return 1


def build_parser() -> argparse.ArgumentParser:


    parser = argparse.ArgumentParser(
        prog="rocket-landing",
        description=(
            "Rocket Landing Lab - eksperymenty z LunarLander-v3"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    play_parser = subparsers.add_parser(
        "play",
        help="Uruchamia jeden losowy epizod z wizualizacją",
    )

    play_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed środowiska i losowych akcji",
    )

    play_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maksymalna liczba kroków epizodu",
    )

    play_parser.set_defaults(handler=handle_play)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Ewaluuje losowego agenta",
    )

    baseline_parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Liczba epizodów ewaluacyjnych",
    )

    baseline_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bazowy seed eksperymentu",
    )

    baseline_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maksymalna liczba kroków jednego epizodu",
    )

    baseline_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Ścieżka raportu JSON",
    )

    baseline_parser.set_defaults(handler=handle_baseline)

    quality_parser = subparsers.add_parser(
        "check-quality",
        help="Sprawdza raport przez quality gate",
    )

    quality_parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Ścieżka raportu JSON",
    )

    quality_parser.add_argument(
        "--minimum-mean-reward",
        type=float,
        default=-300.0,
        help="Minimalna dopuszczalna średnia nagroda",
    )

    quality_parser.add_argument(
        "--maximum-truncated",
        type=int,
        default=10,
        help="Maksymalna liczba przerwanych epizodów",
    )

    quality_parser.add_argument(
        "--minimum-successful",
        type=int,
        default=0,
        help="Minimalna liczba udanych epizodów",
    )

    quality_parser.set_defaults(handler=handle_check_quality)

    return parser


def main() -> int:
    """Główny punkt wejścia CLI."""

    parser = build_parser()
    args = parser.parse_args()

    return args.handler(args)