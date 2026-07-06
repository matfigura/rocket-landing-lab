from pathlib import Path

import numpy as np
from gymnasium import Env
from stable_baselines3 import DQN

from rocket_landing.agents.dqn_agent import load_dqn_agent
from rocket_landing.config import (
    DEFAULT_DQN_EVALUATION_EPISODES,
    DEFAULT_DQN_REPORT_PATH,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
)
from rocket_landing.environment import create_environment
from rocket_landing.models import EpisodeResult, EvaluationSummary
from rocket_landing.reporting import save_evaluation_summary


def run_dqn_episode(
    agent: DQN,
    env: Env,
    seed: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> EpisodeResult:
    if max_steps <= 0:
        raise ValueError("max_steps musi być większe od zera")

    observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not terminated and not truncated and step_count < max_steps:
        action, _ = agent.predict(
            observation,
            deterministic=True,
        )

        action_value = int(np.asarray(action).item())

        observation, reward, terminated, truncated, _ = env.step(
            action_value
        )

        total_reward += float(reward)
        step_count += 1

    if step_count >= max_steps and not terminated and not truncated:
        truncated = True

    return EpisodeResult(
        total_reward=total_reward,
        steps=step_count,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def evaluate_dqn_agent(
    model_path: str | Path,
    episodes: int = DEFAULT_DQN_EVALUATION_EPISODES,
    base_seed: int | None = DEFAULT_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
    success_threshold: float = 200.0,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes musi być większe od zera")

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu: {path}")

    agent = load_dqn_agent(path)
    env = create_environment()

    rewards: list[float] = []
    steps: list[int] = []

    successful_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0

    try:
        for episode_index in range(episodes):
            seed = (
                base_seed + episode_index
                if base_seed is not None
                else None
            )

            result = run_dqn_episode(
                agent=agent,
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

    finally:
        env.close()

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


def run_dqn_evaluation(
    model_path: str | Path,
    episodes: int = DEFAULT_DQN_EVALUATION_EPISODES,
    base_seed: int | None = DEFAULT_SEED,
    max_steps: int = DEFAULT_MAX_STEPS,
    output_path: str | Path = DEFAULT_DQN_REPORT_PATH,
) -> tuple[EvaluationSummary, Path]:
    summary = evaluate_dqn_agent(
        model_path=model_path,
        episodes=episodes,
        base_seed=base_seed,
        max_steps=max_steps,
    )

    report_path = save_evaluation_summary(
        summary=summary,
        output_path=output_path,
    )

    return summary, report_path