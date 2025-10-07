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
from .utils import get_obs, compute_reward, init_obs_space, get_random_positions

if TYPE_CHECKING:
    from chex import PRNGKey

# Found on Stack Overflow (only needed for Windows)
jax.config.update("jax_enable_x64", True)

# TODO #
# Term state testing
# Done flag testing
# Reward state with struct dataclass -> for readability

#Dataclass for defining Environment States
@struct.dataclass
class EnvState():
    """Environment state for MDP Playground."""
    agent_position: jnp.ndarray
    target_position: jnp.ndarray
    terminal_states: jnp.ndarray = jnp.array([], dtype=jnp.int64)
    delayed_rewards: jnp.ndarray = jnp.array([], dtype=jnp.float64)
    counter: int = 1
 
### Start of the MDP Playground ###
class GridEnv:
    """A Grid Environment for MDP Playground."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):

        '''Dimensions of Hardness in the Environment'''

        if 'grid_shape' in config:
            self.grid_shape = tuple(config['grid_shape'])
        else:
            self.grid_shape = (5, 5)

        if 'state_representation' in config:
            self.state_representation = config['state_representation']
        else:
            self.state_representation = 'vector'

        if 'max_steps_in_episode' in config:
            self.max_steps_in_episode = config['max_steps_in_episode']
        else:
            self.max_steps_in_episode = 2 * (self.grid_shape[0] + self.grid_shape[1] - 2)

        if 'transition_noise' in config:
            self.transition_noise = config['transition_noise']
        else:
            self.transition_noise = 0.0

        if 'reward_scale' in config:
            self.reward_scale = config['reward_scale']
        else:
            self.reward_scale = 1.0

        if 'reward_probability' in config:
            self.reward_probability = config['reward_probability']
        else:
            self.reward_probability = 1.0

        if 'reward_shift' in config:
            self.reward_shift = config['reward_shift']
        else:
            self.reward_shift = 0.0

        if 'dense_reward' in config:
            self.dense_reward = config['dense_reward']
        else:
            self.dense_reward = False

        if 'reward_delay' in config:
            self.reward_delay = config['reward_delay']
        else:
            self.reward_delay = 0

        if 'reward_noise' in config:
            self.reward_noise_std = config['reward_noise']
        else:
            self.reward_noise_std = 0.0

        # TODO: Implement
        if 'reward_every_n_steps' in config:
            self.reward_every_n_steps = config['reward_every_n_steps']
        else:
            self.reward_every_n_steps = 1

        if 'number_terminal_states' in config:
            self.number_terminal_states = config['number_terminal_states']
        else:
            self.number_terminal_states = 0

        if 'irrelevant_features' in config:
            self.irrelevant_features = config['irrelevant_features']
        else:
            self.irrelevant_features = False

        '''Initializing the Environment Spaces'''

        # Initializing the action space
        self._action_space = gymnax.environments.spaces.Discrete(4)
        # Action to direction mapping
        self._action_to_direction = jnp.array([
            [-1, 0],  # 0: Move left (negative x)
            [1, 0],   # 1: Move right (positive x)
            [0, -1],  # 2: Move down (negative y)
            [0, 1],   # 3: Move up (positive y)
        ])

        self._observation_space = init_obs_space(self.state_representation, self.irrelevant_features, self.grid_shape, self.number_terminal_states)

    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        """Steps the environment forward by one step."""

        # Compute new agent position based on action
        rng, rng_noise, rng_action = jax.random.split(rng, 3)
        prob_noise = jax.random.uniform(rng_noise) 
        final_action = jax.lax.cond(
            prob_noise < self.transition_noise,
            lambda _: self.action_space.sample(rng_action),
            lambda _: action,
            operand=None
        )
        new_agent_position = jnp.clip(env_state.agent_position + self._action_to_direction[final_action], 0, self.grid_shape[0] - 1)

        # Check if episode is done
        truncated = jnp.where(env_state.counter >= self.max_steps_in_episode, True, False)
        terminated = jnp.all(new_agent_position == env_state.target_position)
        reached_term = jnp.any(jnp.all(env_state.terminal_states == new_agent_position, axis=1))
        done = jnp.any(jnp.array([truncated, terminated, reached_term]))

        # Compute reward signal
        reward, delayed_rewards = compute_reward(
            rng,
            env_state,
            new_agent_position,
            terminated,
            self.dense_reward,
            self.reward_scale,
            self.reward_shift,
            self.reward_probability,
            self.reward_delay,
            self.reward_noise_std
        )

        # Compute new environment state and observation
        env_state = EnvState(
            agent_position=new_agent_position,
            target_position=env_state.target_position,
            terminal_states=env_state.terminal_states,
            delayed_rewards=delayed_rewards,
            counter=env_state.counter + 1, 
        )

        observation = get_obs(
            self.state_representation,
            self.irrelevant_features,
            self.grid_shape,
            self.observation_space,
            env_state,
        )

        return env_state, (observation, reward, done, {})

    # Reset the environment before each episode
    def reset(self, rng: jax.random.PRNGKey):
        
        required_positions = 2 + self.number_terminal_states
        agent_position, target_position, terminal_states = get_random_positions(
            rng=rng, 
            grid_shape=self.grid_shape, 
            n=required_positions,
        )

        delayed_rewards = jnp.zeros(self.reward_delay, dtype=jnp.float64)
        
        env_state = EnvState(
            agent_position=agent_position,
            target_position=target_position,
            delayed_rewards=delayed_rewards, 
            terminal_states=terminal_states, 
            counter=1,
        )

        observation = get_obs(
            self.state_representation, 
            self.irrelevant_features,
            self.grid_shape,
            self.observation_space,
            env_state,
        )

        return env_state, observation

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

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
    

 