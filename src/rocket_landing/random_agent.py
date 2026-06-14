from gymnasium import Env

from rocket_landing.models import EpisodeResult


def run_random_episode(
    env: Env,
    seed: int | None = None,
    max_steps: int = 1000,
) -> EpisodeResult:
    

    if max_steps <= 0:
        raise ValueError("max_steps musi być większe od zera")

    env.reset(seed=seed)

    if seed is not None:
        env.action_space.seed(seed)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step_count < max_steps:
        action = env.action_space.sample()

        _, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)
        step_count += 1

    if step_count >= max_steps and not terminated:
        truncated = True

    return EpisodeResult(
        total_reward=total_reward,
        steps=step_count,
        terminated=terminated,
        truncated=truncated,
    )