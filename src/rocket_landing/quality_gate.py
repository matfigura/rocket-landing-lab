from rocket_landing.models import EvaluationSummary, QualityGateResult


def evaluate_quality_gate(
    summary: EvaluationSummary,
    minimum_mean_reward: float = -300.0,
    maximum_truncated_episodes: int = 10,
    minimum_successful_episodes: int = 0,
) -> QualityGateResult:
    
    if summary.episodes <= 0:
        raise ValueError("summary.episodes musi być większe od zera")

    if maximum_truncated_episodes < 0:
        raise ValueError(
            "maximum_truncated_episodes nie może być ujemne"
        )

    if minimum_successful_episodes < 0:
        raise ValueError(
            "minimum_successful_episodes nie może być ujemne"
        )

    failures: list[str] = []

    if summary.mean_reward < minimum_mean_reward:
        failures.append(
            "Średnia nagroda jest zbyt niska: "
            f"{summary.mean_reward:.2f} < {minimum_mean_reward:.2f}"
        )

    if summary.truncated_episodes > maximum_truncated_episodes:
        failures.append(
            "Zbyt wiele epizodów zostało przerwanych: "
            f"{summary.truncated_episodes} > "
            f"{maximum_truncated_episodes}"
        )

    if summary.successful_episodes < minimum_successful_episodes:
        failures.append(
            "Zbyt mało udanych epizodów: "
            f"{summary.successful_episodes} < "
            f"{minimum_successful_episodes}"
        )

    return QualityGateResult(
        passed=len(failures) == 0,
        failures=tuple(failures),
    )