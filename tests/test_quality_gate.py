from rocket_landing.models import EvaluationSummary
from rocket_landing.quality_gate import evaluate_quality_gate


def create_summary(
    mean_reward: float = -180.0,
    successful_episodes: int = 0,
    truncated_episodes: int = 0,
) -> EvaluationSummary:
    return EvaluationSummary(
        episodes=100,
        mean_reward=mean_reward,
        best_reward=-50.0,
        worst_reward=-400.0,
        mean_steps=90.0,
        successful_episodes=successful_episodes,
        terminated_episodes=100 - truncated_episodes,
        truncated_episodes=truncated_episodes,
    )


def test_quality_gate_passes_for_valid_summary() -> None:
    summary = create_summary(
        mean_reward=-180.0,
        truncated_episodes=0,
    )

    result = evaluate_quality_gate(
        summary=summary,
        minimum_mean_reward=-300.0,
        maximum_truncated_episodes=10,
    )

    assert result.passed is True
    assert result.failures == ()


def test_quality_gate_fails_for_low_mean_reward() -> None:
    summary = create_summary(
        mean_reward=-350.0,
    )

    result = evaluate_quality_gate(
        summary=summary,
        minimum_mean_reward=-300.0,
    )

    assert result.passed is False
    assert len(result.failures) == 1
    assert "Średnia nagroda" in result.failures[0]


def test_quality_gate_reports_multiple_failures() -> None:
    summary = create_summary(
        mean_reward=-400.0,
        successful_episodes=0,
        truncated_episodes=20,
    )

    result = evaluate_quality_gate(
        summary=summary,
        minimum_mean_reward=-300.0,
        maximum_truncated_episodes=10,
        minimum_successful_episodes=5,
    )

    assert result.passed is False
    assert len(result.failures) == 3