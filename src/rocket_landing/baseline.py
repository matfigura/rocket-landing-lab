from pathlib import Path

from rocket_landing.environment import create_environment
from rocket_landing.evaluator import evaluate_random_agent
from rocket_landing.reporting import save_evaluation_summary


REPORT_PATH = Path("reports/random_agent_baseline.json")


def main() -> None:
    env = create_environment()

    try:
        summary = evaluate_random_agent(
            env=env,
            episodes=100,
            base_seed=42,
            max_steps=1000,
        )

        report_path = save_evaluation_summary(
            summary=summary,
            output_path=REPORT_PATH,
        )

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

    finally:
        env.close()


if __name__ == "__main__":
    main()