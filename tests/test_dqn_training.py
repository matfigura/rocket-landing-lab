from pathlib import Path

import pytest

from rocket_landing.training.dqn import run_dqn_training


def test_dqn_training_saves_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "models" / "dqn_test.zip"

    saved_path = run_dqn_training(
        total_timesteps=10,
        seed=42,
        output_path=model_path,
        verbose=0,
    )

    assert saved_path.exists()
    assert saved_path == model_path


def test_dqn_training_rejects_invalid_timesteps() -> None:
    with pytest.raises(
        ValueError,
        match="total_timesteps musi być większe od zera",
    ):
        run_dqn_training(
            total_timesteps=0,
        )