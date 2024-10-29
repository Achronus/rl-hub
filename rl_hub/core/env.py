import gymnasium as gym
from pydantic import BaseModel

from rl_hub.core.config import GymEnvConfig
from rl_hub.core.enums import RenderMode


class GymEnvHandler(BaseModel):
    """A Gym environment handler for working with [Gymnasium](https://gymnasium.farama.org/) environments."""

    config: GymEnvConfig

    @property
    def env(self) -> gym.Env:
        """A property for accessing a simple copy of the Gym environment."""
        return gym.make(self.config.ENV.NAME)

    def run_demo(
        self, episodes: int = 10, render_mode: RenderMode | None = RenderMode.HUMAN
    ) -> None:
        """
        Runs a demonstration of the environment with random actions. Strictly for initially exploring the environment and checking it is operating correctly.
        """
        env = gym.make(self.config.ENV.NAME, render_mode=render_mode)
        self.__training_loop(env, episodes)

    def __training_loop(
        self, env: gym.Env, episodes: int, seed: int | None = None
    ) -> None:
        """A helper method for creating the training loop."""
        try:
            for i_episode in range(1, episodes + 1):
                state, info = env.reset(seed=seed)
                episode_over = False
                score = 0

                while not episode_over:
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    score += reward

                    episode_over = terminated or truncated

                print(f"Episode {i_episode}/{episodes}: {score} score")
        finally:
            env.close()
