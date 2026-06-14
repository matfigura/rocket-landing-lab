from pathlib import Path

from rocket_landing.models import EvaluationSummary, QualityGateResult
from rocket_landing.quality_gate import evaluate_quality_gate
from rocket_landing.reporting import load_evaluation_summary


DEFAULT_REPORT_PATH = Path("reports/random_agent_baseline.json")


def run_quality_check(
    report_path: str | Path = DEFAULT_REPORT_PATH,
    minimum_mean_reward: float = -300.0,
    maximum_truncated_episodes: int = 10,
    minimum_successful_episodes: int = 0,
) -> tuple[EvaluationSummary, QualityGateResult]:


    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono raportu: {path}"
        )

    summary = load_evaluation_summary(path)

    result = evaluate_quality_gate(
        summary=summary,
        minimum_mean_reward=minimum_mean_reward,
        maximum_truncated_episodes=maximum_truncated_episodes,
        minimum_successful_episodes=minimum_successful_episodes,
    )

    return summary, result


def print_quality_result(
    summary: EvaluationSummary,
    result: QualityGateResult,
    report_path: str | Path,
) -> None:


    print("\nQuality Gate")
    print(f"Report: {report_path}")
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


def main() -> None:
    try:
        summary, result = run_quality_check()
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(2) from error

    print_quality_result(
        summary=summary,
        result=result,
        report_path=DEFAULT_REPORT_PATH,
    )

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()