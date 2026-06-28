from pathlib import Path

from gymnasium import Env
from stable_baselines3 import DQN


def create_dqn_agent(
    env: Env,
    seed: int | None = None,
    verbose: int = 0,
) -> DQN:


    return DQN(
        policy="MlpPolicy",
        env=env,
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