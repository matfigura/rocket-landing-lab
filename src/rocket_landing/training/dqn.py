from pathlib import Path

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from rocket_landing.agents.dqn_agent import (
    create_dqn_agent,
    save_dqn_agent,
)
from rocket_landing.config import (
    DEFAULT_DQN_CHECKPOINT_FREQ,
    DEFAULT_DQN_EVAL_EPISODES_DURING_TRAINING,
    DEFAULT_DQN_EVAL_FREQ,
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
    eval_freq: int = DEFAULT_DQN_EVAL_FREQ,
    eval_episodes: int = DEFAULT_DQN_EVAL_EPISODES_DURING_TRAINING,
    checkpoint_freq: int = DEFAULT_DQN_CHECKPOINT_FREQ,
) -> Path:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps musi być większe od zera")

    if eval_freq <= 0:
        raise ValueError("eval_freq musi być większe od zera")

    if eval_episodes <= 0:
        raise ValueError("eval_episodes musi być większe od zera")

    if checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq musi być większe od zera")

    model_path = Path(output_path)

    run_artifacts_path = model_path.parent / model_path.stem
    best_model_path = run_artifacts_path / "best"
    checkpoint_path = run_artifacts_path / "checkpoints"
    evaluation_log_path = run_artifacts_path / "evaluations"

    best_model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    evaluation_log_path.mkdir(parents=True, exist_ok=True)

    training_env = create_environment()
    evaluation_env = Monitor(create_environment())

    try:
        agent = create_dqn_agent(
            env=training_env,
            seed=seed,
            verbose=verbose,
        )

        evaluation_callback = EvalCallback(
            eval_env=evaluation_env,
            best_model_save_path=str(best_model_path),
            log_path=str(evaluation_log_path),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_path),
            name_prefix=f"{model_path.stem}_checkpoint",
            verbose=2,
        )

        callbacks = CallbackList(
            [
                evaluation_callback,
                checkpoint_callback,
            ]
        )

        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

        return save_dqn_agent(
            agent=agent,
            output_path=model_path,
        )

    finally:
        training_env.close()
        evaluation_env.close()