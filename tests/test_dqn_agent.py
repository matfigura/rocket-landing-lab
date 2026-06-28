from pathlib import Path

from rocket_landing.agents.dqn_agent import (
    create_dqn_agent,
    load_dqn_agent,
    save_dqn_agent,
)
from rocket_landing.environment import create_environment


def test_dqn_agent_can_predict_valid_action() -> None:
    env = create_environment()

    try:
        agent = create_dqn_agent(
            env=env,
            seed=42,
        )

        observation, _ = env.reset(seed=42)
        action, _ = agent.predict(
            observation,
            deterministic=True,
        )

        assert env.action_space.contains(int(action))

    finally:
        env.close()


def test_dqn_agent_can_be_saved_and_loaded(
    tmp_path: Path,
) -> None:
    env = create_environment()

    try:
        agent = create_dqn_agent(
            env=env,
            seed=42,
        )

        model_path = tmp_path / "models" / "dqn_smoke.zip"

        saved_path = save_dqn_agent(
            agent=agent,
            output_path=model_path,
        )

        loaded_agent = load_dqn_agent(saved_path)

        observation, _ = env.reset(seed=42)
        action, _ = loaded_agent.predict(
            observation,
            deterministic=True,
        )

        assert saved_path.exists()
        assert env.action_space.contains(int(action))

    finally:
        env.close()