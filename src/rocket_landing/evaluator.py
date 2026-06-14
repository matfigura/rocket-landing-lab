from gymnasium import Env

from rocket_landing.models import EvaluationSummary
from rocket_landing.random_agent import run_random_episode


def evaluate_random_agent(
    env: Env,
    episodes: int = 100,
    base_seed: int | None = 42,
    max_steps: int = 1000,
    success_threshold: float = 200.0,
) -> EvaluationSummary:
    """Ewaluuje losowego agenta na wielu epizodach."""

    if episodes <= 0:
        raise ValueError("episodes musi być większe od zera")

    rewards: list[float] = []
    steps: list[int] = []

    successful_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0

    for episode_index in range(episodes):
        seed = (
            base_seed + episode_index
            if base_seed is not None
            else None
        )

        result = run_random_episode(
            env=env,
            seed=seed,
            max_steps=max_steps,
        )

        rewards.append(result.total_reward)
        steps.append(result.steps)

        if result.total_reward >= success_threshold:
            successful_episodes += 1

        if result.terminated:
            terminated_episodes += 1

        if result.truncated:
            truncated_episodes += 1

    return EvaluationSummary(
        episodes=episodes,
        mean_reward=sum(rewards) / episodes,
        best_reward=max(rewards),
        worst_reward=min(rewards),
        mean_steps=sum(steps) / episodes,
        successful_episodes=successful_episodes,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
    )