from gymnasium import Env


def run_random_episode(
    env: Env,
    seed: int | None = None,
) -> tuple[float, int]:
    """Uruchamia jeden epizod z losowymi akcjami.

    Zwraca:
        tuple zawierającą łączną nagrodę i liczbę kroków.
    """

    observation, info = env.reset(seed=seed)

    if seed is not None:
        env.action_space.seed(seed)

    total_reward = 0.0
    step_count = 0
    episode_finished = False

    while not episode_finished:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        step_count += 1

        episode_finished = terminated or truncated

    return total_reward, step_count