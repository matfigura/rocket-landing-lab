from pathlib import Path

import pytest

import rocket_landing.cli as cli
from rocket_landing.models import (
    EpisodeResult,
    EvaluationSummary,
    QualityGateResult,
)


def create_summary() -> EvaluationSummary:
    """Tworzy przykładowe podsumowanie do testów CLI."""

    return EvaluationSummary(
        episodes=10,
        mean_reward=-180.0,
        best_reward=-80.0,
        worst_reward=-350.0,
        mean_steps=90.0,
        successful_episodes=0,
        terminated_episodes=10,
        truncated_episodes=0,
    )


def test_play_command_passes_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_arguments: dict[str, int | None] = {}

    def fake_play_random_episode(
        seed: int | None,
        max_steps: int,
    ) -> EpisodeResult:
        received_arguments["seed"] = seed
        received_arguments["max_steps"] = max_steps

        return EpisodeResult(
            total_reward=-120.0,
            steps=15,
            terminated=True,
            truncated=False,
        )

    monkeypatch.setattr(
        cli,
        "play_random_episode",
        fake_play_random_episode,
    )
    monkeypatch.setattr(
        cli,
        "print_episode_result",
        lambda result: None,
    )

    exit_code = cli.main(
        [
            "play",
            "--seed",
            "42",
            "--max-steps",
            "20",
        ]
    )

    assert exit_code == 0
    assert received_arguments["seed"] == 42
    assert received_arguments["max_steps"] == 20


def test_baseline_command_passes_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_arguments: dict[str, object] = {}

    def fake_run_baseline(
        episodes: int,
        base_seed: int | None,
        max_steps: int,
        output_path: str | Path,
    ) -> tuple[EvaluationSummary, Path]:
        received_arguments["episodes"] = episodes
        received_arguments["base_seed"] = base_seed
        received_arguments["max_steps"] = max_steps
        received_arguments["output_path"] = output_path

        return create_summary(), Path("reports/test.json")

    monkeypatch.setattr(
        cli,
        "run_baseline",
        fake_run_baseline,
    )
    monkeypatch.setattr(
        cli,
        "print_baseline_summary",
        lambda summary, report_path: None,
    )

    exit_code = cli.main(
        [
            "baseline",
            "--episodes",
            "5",
            "--seed",
            "123",
            "--max-steps",
            "50",
            "--output",
            "reports/custom.json",
        ]
    )

    assert exit_code == 0
    assert received_arguments["episodes"] == 5
    assert received_arguments["base_seed"] == 123
    assert received_arguments["max_steps"] == 50
    assert received_arguments["output_path"] == Path(
        "reports/custom.json"
    )


def test_check_quality_returns_zero_when_gate_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_quality_check(
        report_path: str | Path,
        minimum_mean_reward: float,
        maximum_truncated_episodes: int,
        minimum_successful_episodes: int,
    ) -> tuple[EvaluationSummary, QualityGateResult]:
        return (
            create_summary(),
            QualityGateResult(
                passed=True,
                failures=(),
            ),
        )

    monkeypatch.setattr(
        cli,
        "run_quality_check",
        fake_run_quality_check,
    )
    monkeypatch.setattr(
        cli,
        "print_quality_result",
        lambda summary, result, report_path: None,
    )

    exit_code = cli.main(["check-quality"])

    assert exit_code == 0


def test_check_quality_returns_one_when_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_quality_check(
        report_path: str | Path,
        minimum_mean_reward: float,
        maximum_truncated_episodes: int,
        minimum_successful_episodes: int,
    ) -> tuple[EvaluationSummary, QualityGateResult]:
        return (
            create_summary(),
            QualityGateResult(
                passed=False,
                failures=(
                    "Średnia nagroda jest zbyt niska",
                ),
            ),
        )

    monkeypatch.setattr(
        cli,
        "run_quality_check",
        fake_run_quality_check,
    )
    monkeypatch.setattr(
        cli,
        "print_quality_result",
        lambda summary, result, report_path: None,
    )

    exit_code = cli.main(
        [
            "check-quality",
            "--minimum-mean-reward",
            "200",
        ]
    )

    assert exit_code == 1


def test_check_quality_returns_two_when_report_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_quality_check(
        report_path: str | Path,
        minimum_mean_reward: float,
        maximum_truncated_episodes: int,
        minimum_successful_episodes: int,
    ) -> tuple[EvaluationSummary, QualityGateResult]:
        raise FileNotFoundError(
            "Nie znaleziono raportu: reports/missing.json"
        )

    monkeypatch.setattr(
        cli,
        "run_quality_check",
        fake_run_quality_check,
    )

    exit_code = cli.main(
        [
            "check-quality",
            "--report",
            "reports/missing.json",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Nie znaleziono raportu" in captured.out


def test_unknown_command_returns_argparse_error() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown-command"])

    assert error.value.code == 2