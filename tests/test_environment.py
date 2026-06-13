from rocket_landing.environment import create_environment


def test_environment_can_be_created() -> None:
    env = create_environment()

    try:
        observation, info = env.reset(seed=42)

        assert env.observation_space.contains(observation)
        assert env.action_space.n == 4

    finally:
        env.close()