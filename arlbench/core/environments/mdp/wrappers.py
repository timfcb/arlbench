import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Any

jax.config.update("jax_enable_x64", True)

### In case more than one Wrapper is needed --> Abstract wrapper class like in https://github.com/dunnolab/xland-minigrid/blob/main/src/xminigrid/wrappers.py

### TODO Frage an Julian: wird der Autoresetwrapper dann auch bei Eval aufgerufen?
class AutoResetWrapper():

    def __init__(self, env:Any):
        self._env = env

    def __reset_after_episode(self, rng: jax.random.PRNGKey, timestep):

        key, _ = jax.random.split(rng)
        env_state_reset, observation_reset = self._env.reset(key)
        env_state, (observation, reward, done, info) = timestep

        return env_state_reset, (observation_reset, reward, done, info)

    def step(self, env_state: Any, action: Any, rng: jax.random.PRNGKey):

        env_state, (observation, reward, done, info) = self._env.step(env_state, action, rng)
        timestep = env_state, (observation, reward, done, info)

        timestep = jax.lax.cond(
            done,
            lambda: self.__reset_after_episode(rng, timestep),
            lambda: timestep,
        )

        return timestep
    
    def __getattr__(self, name):
        return getattr(self._env, name)