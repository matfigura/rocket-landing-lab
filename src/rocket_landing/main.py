from rocket_landing.environment import create_environment
from rocket_landing.random_agent import run_random_episode


def main() -> None:
    env = create_environment(render_mode="human")

    try:
        result = run_random_episode(env)

        print("\nEpizod zakończony")
        print(f"Liczba kroków: {result.steps}")
        print(f"Łączna nagroda: {result.total_reward:.2f}")
        print(f"Terminated: {result.terminated}")
        print(f"Truncated: {result.truncated}")

    finally:
        env.close()


if __name__ == "__main__":
    main()