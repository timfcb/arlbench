"""MDP Playground environment adapter"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from flax import struct

from ..autorl_env import Environment
from .data_classes import EnvState, RewardShape
from .wrappers import AutoResetWrapper
from .grid_env import GridEnv

if TYPE_CHECKING:
    from chex import PRNGKey

# Found on Stack Overflow (only needed for Windows)
jax.config.update("jax_enable_x64", True)
 
### Start of the MDP Playground ###
class MdpPlaygroundEnv(Environment):
    """A MDP playground-based RL environment."""
    def __init__(self, env_name: str, n_envs: int, config: dict[str, Any] | None = None):
        """Creates an MDP Playground environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the MDP Playground environment.
            n_envs (int): Number of environments.
            config (dict[str, Any] | None, optional): Configuration dictionary
                for the environment. Defaults to None.
        """
        if config is None:
            config = {}
        env = GridEnv(config=config)
        env = AutoResetWrapper(env)
        super().__init__(env_name, env, n_envs)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: jax.random.PRNGKey):
        """Resets the MDP Playground environment."""
        # Split into number_envs keys
        reset_rng = jax.random.split(rng, self.n_envs)

        # Reset the environment
        env_state, curr_obs = jax.vmap(self._env.reset, in_axes=(0,))(reset_rng)

        return env_state, curr_obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: jax.random.PRNGKey):
        """Steps the environment forward by one step."""

        # Split into number_envs keys
        step_rng = jax.random.split(rng, self.n_envs)

        # vmap the step function
        env_state, (obs, reward, done, info) = jax.vmap(self._env.step, in_axes=(0, 0, 0))(env_state, action, step_rng)

        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        """Action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self):
        """Observation space of the environment."""
        return self._env.observation_space
    
