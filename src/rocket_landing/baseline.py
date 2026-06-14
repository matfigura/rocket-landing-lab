from pathlib import Path

from rocket_landing.environment import create_environment
from rocket_landing.evaluator import evaluate_random_agent
from rocket_landing.models import EvaluationSummary
from rocket_landing.reporting import save_evaluation_summary


DEFAULT_REPORT_PATH = Path("reports/random_agent_baseline.json")


def run_baseline(
    episodes: int = 100,
    base_seed: int | None = 42,
    max_steps: int = 1000,
    output_path: str | Path = DEFAULT_REPORT_PATH,
) -> tuple[EvaluationSummary, Path]:
    """Uruchamia ewaluację losowego agenta i zapisuje raport."""

    env = create_environment()

    try:
        summary = evaluate_random_agent(
            env=env,
            episodes=episodes,
            base_seed=base_seed,
            max_steps=max_steps,
        )
    finally:
        env.close()

    report_path = save_evaluation_summary(
        summary=summary,
        output_path=output_path,
    )

    return summary, report_path


def print_baseline_summary(
    summary: EvaluationSummary,
    report_path: Path,
) -> None:
    """Wyświetla podsumowanie baseline'u."""

    print("\nRandom Agent Baseline")
    print(f"Episodes: {summary.episodes}")
    print(f"Mean reward: {summary.mean_reward:.2f}")
    print(f"Best reward: {summary.best_reward:.2f}")
    print(f"Worst reward: {summary.worst_reward:.2f}")
    print(f"Mean steps: {summary.mean_steps:.2f}")
    print(
        "Successful episodes: "
        f"{summary.successful_episodes}/{summary.episodes}"
    )
    print(f"Terminated episodes: {summary.terminated_episodes}")
    print(f"Truncated episodes: {summary.truncated_episodes}")
    print(f"\nReport saved to: {report_path.resolve()}")


def main() -> None:
    summary, report_path = run_baseline()

    print_baseline_summary(
        summary=summary,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()