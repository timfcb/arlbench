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

if TYPE_CHECKING:
    from chex import PRNGKey

# Found on Stack Overflow (only needed for Windows)
jax.config.update("jax_enable_x64", True)

#Dataclass for defining Environment States
@struct.dataclass
class EnvState():
    """Environment state for MDP Playground."""
    agent_position: jnp.ndarray
    target_position: jnp.ndarray
    counter: int = 1
 
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
        if 'max_steps_in_episode' not in config:
            # If not specified in config, compute as 2 times max distance in grid for reaching target
            self.max_steps_in_episode = 2 * (self.grid_shape[0] + self.grid_shape[1] - 2)
        else:
            self.max_steps_in_episode = config['max_steps_in_episode']

        if 'transition_noise' not in config:
            self.transition_noise = 0.0
        else:
            self.transition_noise = config['transition_noise']

        # Scaling factor of reward signal
        if 'reward_scale' not in config:
            self.reward_scale = 1.0
        else:
            self.reward_scale = config['reward_scale']

        # Reward is given with a certain probability (otherwise prob is 1)
        if 'reward_probability' in config:
            self.reward_probability = config['reward_probability']
        else:
            self.reward_probability = 1.0

        # Reward is shifted by a constant value
        if 'reward_shift' not in config:
            self.reward_shift = 0.0
        else:
            self.reward_shift = config['reward_shift']

        # Dense reward signal (otherwise sparse)
        if 'dense_reward' in config:
            self.dense_reward = config['dense_reward']
        else:
            self.dense_reward = False

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
            self._observation_space = BoxExtended(
                jnp.array([0,0,0,0], dtype=jnp.int64),
                jnp.array([self.grid_shape[0] - 1, self.grid_shape[1] - 1, self.grid_shape[0] - 1, self.grid_shape[1] - 1], dtype=jnp.int64),
                (4,),
                dtype=jnp.int64
            )
        elif self.state_representation == 'matrix':
            self._observation_space = BoxExtended(
                low=0,
                high=2,
                shape=(self.grid_shape[0], self.grid_shape[1]),
                dtype=jnp.int64
            )
        elif self.state_representation == 'image':
            self._observation_space = ImageContinuous()
        else:
            raise ValueError(f"Unknown state representation: {self.state_representation}. Supported are 'image', 'vector', and 'matrix'.")

    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        """Steps the environment forward by one step."""
        
        # Transition noise
        rng, rng_noise = jax.random.split(rng)
        prob_noise = jax.random.uniform(rng_noise)
        rng, rng_action = jax.random.split(rng)

        final_action = jax.lax.cond(
            prob_noise < self.transition_noise,
            lambda _: self.action_space.sample(rng_action),
            lambda _: action,
            operand=None
        )

        # Update agent location based on action
        new_agent_position = jnp.clip(env_state.agent_position + self._action_to_direction[final_action], 0, self.grid_shape[0] - 1)

        #jax.debug.print("Action taken/Action agent: {}/{}, Agent position before: {}, Agent Position after: {}, Target position: {}, Counter: {}, Random Value: {}, Transition Noise: {}, Max Steps: {}", final_action, action, env_state.agent_position, new_agent_position, env_state.target_position, env_state.counter, rand_value, self.transition_noise, self.max_steps_in_episode)

        # Check if episode is done (max steps or reached target)
        # TODO debug/understand train function arlbench
        truncated = jnp.where(env_state.counter >= self.max_steps_in_episode, True, False)
        terminated = jnp.all(new_agent_position == env_state.target_position)
        done = jnp.logical_or(terminated, truncated)

        # Compute reward signal
        rng, rng_reward = jax.random.split(rng)
        rand_value = jax.random.uniform(rng_reward)
        reward = 0.0
        if self.dense_reward:
            # Dense reward: Reward is given for every step (change in manhattan distance to target)
            manhat_dist_old = jnp.sum(jnp.abs(env_state.agent_position - env_state.target_position))
            manhat_dist_new = jnp.sum(jnp.abs(new_agent_position - env_state.target_position))
            reward = manhat_dist_old - manhat_dist_new
        else:
            # Sparse reward: Reward is only given when target is reached
            reward = jnp.where(terminated, 1.0, 0.0)
        # Reward scaling, probability, and shifting  
        reward = (self.reward_scale * jnp.where(rand_value < self.reward_probability, reward, 0.0)) + self

        # Update the environment state
        env_state = EnvState(agent_position=new_agent_position, target_position=env_state.target_position, counter=env_state.counter + 1)

        # Compute observation based on state representation (image, vector, matrix)
        observation = self.get_obs(new_agent_position, env_state.target_position)

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
            """Body function for JAX while loop. Performs one parallel step in all environments.

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

        # New environment state with new randomized agent and target locations
        env_state = EnvState(agent_position=agent_position, target_position=target_position)

        # Compute observation based on state representation (image, vector, matrix)
        observation = self.get_obs(agent_position, target_position)

        return env_state, observation

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
    
    # Greyscaling used in Atari preprocessing (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    def rgb_to_greyscale(self, rgb_image: jnp.ndarray) -> jnp.ndarray:
        """Converts an RGB image to greyscale by extracting the Y channel.
        Args:
            rgb_image (jnp.ndarray): Input RGB image of shape (H, W, 3).

        Returns:
            jnp.ndarray: Greyscaled image of shape (H, W, 1).
        """
        # Standard formula to convert RGB to greyscale
        R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
        # Extracting the y component (luminance) from the YUV color space
        greyscale_image = 0.299 * R + 0.587 * G + 0.114 * B
        greyscale_image = jnp.expand_dims(greyscale_image, axis=-1)
        return greyscale_image
    
    def get_obs(self, agent_position: jax.Array, target_position: jax.Array) -> Any:
        """Computes the observation based on the state representation (image, vector, matrix).

        Args:
            agent_position: Current agent position.
            target_position: Current target position.

        Returns:
            Any: Observation based on the state representation.
        """
        # Image returns a greyscaled image
        if self.state_representation == 'image':
            rgb_image = self._observation_space.generate_image(agent_position, target_position)
            observation = self.rgb_to_greyscale(rgb_image)
        # Vector returns a 4 dimensional vector with agent and target positions
        elif self.state_representation == 'vector':
            observation = jnp.concatenate([agent_position, target_position])
        # Matrix returns a matrix with 0 for empty cells, 1 for agent position and 2 for target position
        elif self.state_representation == 'matrix':
            grid_matrix = jnp.zeros(self.grid_shape, dtype=jnp.int64)
            # Set agent position to 1 and target position to 2 (flip x/y coordinates for correct orientation)
            grid_matrix = grid_matrix.at[agent_position[1], agent_position[0]].set(1).at[target_position[1], target_position[0]].set(2)
            observation = jnp.expand_dims(grid_matrix, axis=-1)
        else:
            raise ValueError(f"Unknown state representation: {self.state_representation}. Supported are 'image', 'vector', and 'matrix'.")
        
        return observation


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
    

 