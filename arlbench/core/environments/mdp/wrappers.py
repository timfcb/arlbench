import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Any
from arlbench.core.wrappers import Wrapper

jax.config.update("jax_enable_x64", True)

# Implements abstract Wrapper class
class AutoResetWrapper(Wrapper):

    def __init__(self, env:Any):
        super().__init__(env)

    def __reset_after_episode(self, rng: jax.random.PRNGKey, timestep):

        key, _ = jax.random.split(rng)
        env_state_reset, observation_reset = self._env.reset(key)
        env_state, (observation, reward, done, info) = timestep

        return env_state_reset, (observation, reward, done, info)

    def step(self, env_state: Any, action: Any, rng: jax.random.PRNGKey):

        env_state, (observation, reward, done, info) = self._env.step(env_state, action, rng)
        timestep = env_state, (observation, reward, done, info)

        timestep = jax.lax.cond(
            done,
            lambda: self.__reset_after_episode(rng, timestep),
            lambda: timestep,
        )

        return timestep
