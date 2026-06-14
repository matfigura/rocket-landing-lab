from rocket_landing.environment import create_environment
from rocket_landing.models import EpisodeResult
from rocket_landing.random_agent import run_random_episode


def play_random_episode(
    seed: int | None = None,
    max_steps: int = 1000,
) -> EpisodeResult:
  

    env = create_environment(render_mode="human")

    try:
        return run_random_episode(
            env=env,
            seed=seed,
            max_steps=max_steps,
        )
    finally:
        env.close()


def print_episode_result(result: EpisodeResult) -> None:
    """Wyświetla wynik pojedynczego epizodu."""

    print("\nEpizod zakończony")
    print(f"Liczba kroków: {result.steps}")
    print(f"Łączna nagroda: {result.total_reward:.2f}")
    print(f"Terminated: {result.terminated}")
    print(f"Truncated: {result.truncated}")


def main() -> None:
    result = play_random_episode()
    print_episode_result(result)


if __name__ == "__main__":
    main()