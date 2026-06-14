import pytest

from rocket_landing.environment import create_environment
from rocket_landing.models import EpisodeResult
from rocket_landing.random_agent import run_random_episode


def test_random_episode_returns_result() -> None:
    env = create_environment()

    try:
        result = run_random_episode(
            env=env,
            seed=42,
            max_steps=20,
        )

        assert isinstance(result, EpisodeResult)
        assert isinstance(result.total_reward, float)
        assert 1 <= result.steps <= 20
        assert result.terminated or result.truncated

    finally:
        env.close()

def test_random_episode_rejects_invalid_max_steps() -> None:
    env = create_environment()

    try:
        with pytest.raises(ValueError, match="max_steps"):
            run_random_episode(
                env=env,
                max_steps=0,
            )

    finally:
        env.close()