import json
from pathlib import Path

from rocket_landing.models import EvaluationSummary
from rocket_landing.reporting import save_evaluation_summary


def test_save_evaluation_summary_creates_json(
    tmp_path: Path,
) -> None:
    summary = EvaluationSummary(
        episodes=3,
        mean_reward=-150.5,
        best_reward=-100.0,
        worst_reward=-200.0,
        mean_steps=42.5,
        successful_episodes=0,
        terminated_episodes=2,
        truncated_episodes=1,
    )

    output_path = tmp_path / "reports" / "summary.json"

    saved_path = save_evaluation_summary(
        summary=summary,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert saved_data["episodes"] == 3
    assert saved_data["mean_reward"] == -150.5
    assert saved_data["best_reward"] == -100.0
    assert saved_data["worst_reward"] == -200.0
    assert saved_data["successful_episodes"] == 0
    assert saved_data["terminated_episodes"] == 2
    assert saved_data["truncated_episodes"] == 1