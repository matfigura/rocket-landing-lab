from typing import Literal

import gymnasium as gym
from gymnasium import Env

from rocket_landing.config import ENVIRONMENT_ID


RenderMode = Literal["human", "rgb_array"] | None


def create_environment(render_mode: RenderMode = None) -> Env:
    

    return gym.make(
        ENVIRONMENT_ID,
        render_mode=render_mode,
    )