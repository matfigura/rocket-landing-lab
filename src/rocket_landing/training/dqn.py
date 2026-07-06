from pathlib import Path

from rocket_landing.agents.dqn_agent import (
    create_dqn_agent,
    save_dqn_agent,
)
from rocket_landing.config import (
    DEFAULT_DQN_MODEL_PATH,
    DEFAULT_DQN_TIMESTEPS,
    DEFAULT_SEED,
)
from rocket_landing.environment import create_environment


def run_dqn_training(
    total_timesteps: int = DEFAULT_DQN_TIMESTEPS,
    seed: int | None = DEFAULT_SEED,
    output_path: str | Path = DEFAULT_DQN_MODEL_PATH,
    verbose: int = 1,
) -> Path:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps musi być większe od zera")

    env = create_environment()

    try:
        agent = create_dqn_agent(
            env=env,
            seed=seed,
            verbose=verbose,
        )

        agent.learn(
            total_timesteps=total_timesteps,
        )

        model_path = save_dqn_agent(
            agent=agent,
            output_path=output_path,
        )

        return model_path

    finally:
        env.close()