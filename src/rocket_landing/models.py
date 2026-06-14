from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:

    total_reward: float
    steps: int
    terminated: bool
    truncated: bool

@dataclass(frozen=True)
class EvaluationSummary:

    episodes: int
    mean_reward: float
    best_reward: float
    worst_reward: float
    mean_steps: float
    successful_episodes: int
    terminated_episodes: int
    truncated_episodes: int

@dataclass(frozen=True)
class QualityGateResult:

    passed: bool
    failures: tuple[str, ...]