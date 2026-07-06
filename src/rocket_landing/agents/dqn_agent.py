from pathlib import Path

from gymnasium import Env
from stable_baselines3 import DQN

from rocket_landing.config import (
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_EXPLORATION_FINAL_EPS,
    DQN_EXPLORATION_FRACTION,
    DQN_GAMMA,
    DQN_GRADIENT_STEPS,
    DQN_LEARNING_RATE,
    DQN_LEARNING_STARTS,
    DQN_NETWORK_ARCHITECTURE,
    DQN_TARGET_UPDATE_INTERVAL,
    DQN_TRAIN_FREQ,
)


def create_dqn_agent(
    env: Env,
    seed: int | None = None,
    verbose: int = 0,
) -> DQN:
    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=DQN_LEARNING_RATE,
        batch_size=DQN_BATCH_SIZE,
        buffer_size=DQN_BUFFER_SIZE,
        learning_starts=DQN_LEARNING_STARTS,
        gamma=DQN_GAMMA,
        target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
        train_freq=DQN_TRAIN_FREQ,
        gradient_steps=DQN_GRADIENT_STEPS,
        exploration_fraction=DQN_EXPLORATION_FRACTION,
        exploration_final_eps=DQN_EXPLORATION_FINAL_EPS,
        policy_kwargs={
            "net_arch": DQN_NETWORK_ARCHITECTURE,
        },
        seed=seed,
        verbose=verbose,
    )


def save_dqn_agent(
    agent: DQN,
    output_path: str | Path,
) -> Path:


    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent.save(path)

    return path


def load_dqn_agent(
    model_path: str | Path,
) -> DQN:
  

    return DQN.load(model_path)