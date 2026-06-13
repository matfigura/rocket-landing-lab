from rocket_landing.environment import create_environment
from rocket_landing.random_agent import run_random_episode


def main() -> None:
    env = create_environment(render_mode="human")

    try:
        total_reward, step_count = run_random_episode(env)

        print("\nEpizod zakończony")
        print(f"Liczba kroków: {step_count}")
        print(f"Łączna nagroda: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()