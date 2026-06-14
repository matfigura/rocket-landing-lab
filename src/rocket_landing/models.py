from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:

    total_reward: float
    steps: int
    terminated: bool
    truncated: bool