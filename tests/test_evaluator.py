import pytest

from rocket_landing.environment import create_environment
from rocket_landing.evaluator import evaluate_random_agent
from rocket_landing.models import EvaluationSummary


def test_evaluator_returns_summary() -> None:
    env = create_environment()

    try:
        summary = evaluate_random_agent(
            env=env,
            episodes=3,
            base_seed=42,
            max_steps=20,
        )

        assert isinstance(summary, EvaluationSummary)
        assert summary.episodes == 3
        assert isinstance(summary.mean_reward, float)
        assert isinstance(summary.best_reward, float)
        assert isinstance(summary.worst_reward, float)
        assert isinstance(summary.mean_steps, float)

        assert summary.best_reward >= summary.worst_reward
        assert 1 <= summary.mean_steps <= 20

        assert (
            summary.terminated_episodes
            + summary.truncated_episodes
            >= summary.episodes
        )

    finally:
        env.close()


def test_evaluator_rejects_invalid_episode_count() -> None:
    env = create_environment()

    try:
        with pytest.raises(ValueError, match="episodes"):
            evaluate_random_agent(
                env=env,
                episodes=0,
            )

    finally:
        env.close()