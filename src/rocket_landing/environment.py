from typing import Literal

import gymnasium as gym
from gymnasium import Env


RenderMode = Literal["human", "rgb_array"] | None


def create_environment(render_mode: RenderMode = None) -> Env:
    

    return gym.make(
        "LunarLander-v3",
        render_mode=render_mode,
    )