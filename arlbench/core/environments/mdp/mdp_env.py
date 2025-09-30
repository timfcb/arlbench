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
from .spaces import BoxExtended, ImageContinuous
from .utils import get_obs, compute_reward

if TYPE_CHECKING:
    from chex import PRNGKey

# Found on Stack Overflow (only needed for Windows)
jax.config.update("jax_enable_x64", True)

### TODO
# Irrelevat features (extra dimensions in observation space that do not contain any information)
# Changing target position during an episodes?
# Terminal States

#Dataclass for defining Environment States
@struct.dataclass
class EnvState():
    """Environment state for MDP Playground."""
    agent_position: jnp.ndarray
    target_position: jnp.ndarray
    counter: int = 1
    delayed_rewards: jnp.ndarray = jnp.array([], dtype=jnp.float64)
 
### Start of the MDP Playground ###
class GridEnv:
    """A Grid Environment for MDP Playground."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ):

        '''Dimensions of Hardness in the Environment'''

        self.grid_shape = tuple(config['grid_shape'])

        # State Representation (vector, matrix, image)
        self.state_representation = config['state_representation']

        # Epsiode is truncated if goal is not reached within max_steps_in_episode
        if 'max_steps_in_episode' in config:
            # If not specified in config, compute as 2 times max distance in grid for reaching target
            self.max_steps_in_episode = config['max_steps_in_episode']
        else:
            self.max_steps_in_episode = 2 * (self.grid_shape[0] + self.grid_shape[1] - 2)

        # Transition noise (probability of taking a random action instead of the intended one)
        if 'transition_noise' in config:
            self.transition_noise = config['transition_noise']
        else:
            self.transition_noise = 0.0

        # Scaling factor of reward signal
        if 'reward_scale' in config:
            self.reward_scale = config['reward_scale']
        else:
            self.reward_scale = 1.0

        # Reward is given with a certain probability (otherwise prob is 1)
        if 'reward_probability' in config:
            self.reward_probability = config['reward_probability']
        else:
            self.reward_probability = 1.0

        # Reward is shifted by a constant value
        if 'reward_shift' in config:
            self.reward_shift = config['reward_shift']
        else:
            self.reward_shift = 0.0

        # Dense reward signal (otherwise sparse)
        if 'dense_reward' in config:
            self.dense_reward = config['dense_reward']
        else:
            self.dense_reward = False

        # Reward delay (0 = no delay, 1 = reward is given at next step, etc.)
        if 'reward_delay' in config:
            self.reward_delay = config['reward_delay']
        else:
            self.reward_delay = 0

        if 'reward_noise' in config:
            self.reward_noise_std = config['reward_noise']
        else:
            self.reward_noise_std = 0.0

        ### TODO implement irrelevant features
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

        # Initializing the observation space (RGB image or agent position as array)
        if self.state_representation == 'vector':
            if self.irrelevant_features:
                self.grid_shape = self.grid_shape * 2
            self._observation_space = BoxExtended(
                jnp.array(jnp.zeros(len(self.grid_shape)) * 2, dtype=jnp.int64),
                jnp.array([(self.grid_shape[i] - 1) for i in range(len(self.grid_shape))]*2, dtype=jnp.int64),
                (len(self.grid_shape)*2,),
                dtype=jnp.int64
            )
        elif self.state_representation == 'matrix':
            self._observation_space = BoxExtended(
                low=0,
                high=2,
                shape= self.grid_shape * 2,
                dtype=jnp.int64
            )

        # TODO Überlegen wie Irrelevante Features in Image Representation aussehen können
        elif self.state_representation == 'image':
            self._observation_space = ImageContinuous()
        else:
            raise ValueError(f"Unknown state representation: {self.state_representation}. Supported are 'image', 'vector', and 'matrix'.")

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
        done = jnp.logical_or(terminated, truncated)

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
        env_state = EnvState(agent_position=new_agent_position, target_position=env_state.target_position, counter=env_state.counter + 1, delayed_rewards=delayed_rewards)
        observation = get_obs(self.state_representation, self.irrelevant_features, self.grid_shape, self.observation_space, new_agent_position, env_state.target_position)

        return env_state, (observation, reward, done, {})

    # Reset the environment before each episode
    def reset(self, rng: jax.random.PRNGKey):

        # Draw random agent location
        rng, key_agent_x, key_agent_y = jax.random.split(rng, 3)
        agent_position = jnp.stack([
            jax.random.randint(key_agent_x, (), 0, self.grid_shape[0]),
            jax.random.randint(key_agent_y, (), 0, self.grid_shape[1]),
        ])
        agent_position = jnp.array(agent_position, dtype=jnp.int64)

        # While loop to draw random target location until it is not equal to the agent position
        def body_fn(state):
            """Body function for JAX while loop. Generates possible new target location.

            Args:
                state (tuple): Key and impossible target location.

            Returns:
                tuple: Key and possible new target location.
            """
            key, target = state
            key, key_x, key_y = jax.random.split(key,3)
            target = jnp.stack([
                jax.random.randint(key_x, (), 0, self.grid_shape[0]),
                jax.random.randint(key_y, (), 0, self.grid_shape[1]),
            ])
            target = jnp.array(target, dtype=jnp.int64)
            return key, target

        def cond_fn(state):
            """Condition function for JAX while loop. Returns true if agent location is equal to target location.

            Args:
                state (tuple): Key and possible target location.

            Returns:
                jnp.bool: True if agent location is equal to target location and new target location is required.
            """
            _, target = state
            return jnp.all(target == agent_position)
        
        state = (rng, agent_position)
        key_final, target_position = jax.lax.while_loop(cond_fn, body_fn, state)

        delayed_rewards = jnp.zeros(self.reward_delay, dtype=jnp.float64)
        env_state = EnvState(agent_position=agent_position, target_position=target_position, delayed_rewards=delayed_rewards)

        observation = get_obs(self.state_representation, self.irrelevant_features, self.grid_shape, self.observation_space, agent_position, target_position)

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
    

 