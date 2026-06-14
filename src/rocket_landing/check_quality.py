from pathlib import Path

from rocket_landing.quality_gate import evaluate_quality_gate
from rocket_landing.reporting import load_evaluation_summary


REPORT_PATH = Path("reports/random_agent_baseline.json")


def main() -> None:
    if not REPORT_PATH.exists():
        print(f"Nie znaleziono raportu: {REPORT_PATH}")
        print(
            "Najpierw uruchom: "
            "python -m rocket_landing.baseline"
        )
        raise SystemExit(2)

    summary = load_evaluation_summary(REPORT_PATH)

    result = evaluate_quality_gate(
        summary=summary,
        minimum_mean_reward=-300.0,
        maximum_truncated_episodes=10,
        minimum_successful_episodes=0,
    )

    print("\nQuality Gate")
    print(f"Report: {REPORT_PATH}")
    print(f"Mean reward: {summary.mean_reward:.2f}")
    print(
        "Truncated episodes: "
        f"{summary.truncated_episodes}/{summary.episodes}"
    )
    print(
        "Successful episodes: "
        f"{summary.successful_episodes}/{summary.episodes}"
    )

    if result.passed:
        print("\nResult: PASSED")
        return

    print("\nResult: FAILED")

    for failure in result.failures:
        print(f"- {failure}")

    raise SystemExit(1)


if __name__ == "__main__":
    main()