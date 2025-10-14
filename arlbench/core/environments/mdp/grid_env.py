"""Generator for single Grid Environments"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from flax import struct

from .utils.utils import get_random_positions
from .utils.observation import get_obs, init_obs_space
from .utils.reward import reward_function
from .data_classes import EnvState, RewardShape

if TYPE_CHECKING:
    from chex import PRNGKey

# Found on Stack Overflow (only needed for Windows)
jax.config.update("jax_enable_x64", True)

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

        if 'frac_term_states' in config:
            frac_term_states = config['frac_term_states']
        else:
            frac_term_states = 0.0

        self.number_terminal_states = int(self.grid_shape[0] * self.grid_shape[1] * frac_term_states)

        # Specification of reward shape
        if 'reward_scaling' in config:
            reward_scaling = config['reward_scaling']
        else:
            reward_scaling = 1.0

        if 'reward_probability' in config:
            reward_probability = config['reward_probability']
        else:
            reward_probability = 1.0

        if 'reward_shift' in config:
            reward_shift = config['reward_shift']
        else:
            reward_shift = 0.0

        if 'reward_noise' in config:
            reward_noise_std = config['reward_noise']
        else:
            reward_noise_std = 0.0

        if 'reward_delay_prob' in config:
            delay_prob = config['reward_delay_prob']
        else:
            delay_prob = 1.0

        self.reward_shape = RewardShape(
            noise=reward_noise_std,
            scaling_factor=reward_scaling,
            shift=reward_shift,
            probability=reward_probability,
            delay_prob=delay_prob,
        )

        # Only debug purposes
        if 'is_eval' in config:
            self.evaluation_env = config['is_eval']
        else:
            self.evaluation_env = False

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

        self._observation_space = init_obs_space(self.state_representation, self.grid_shape, self.number_terminal_states)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        """Step function of a single grid environment
        
            Functionality
            -------------
            - Application of transition noise to the sampled action by the agent and moving the agent based on action
            - Check whether the episode has ended through the action; Possible reasons: terminated, truncated, reached terminal state
            - Compute reward signal for this transition
            - Compute new environment state
            - Compute observation for given obs shape (vector, matrix, image)

        """
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
        reward, delayed_rewards = reward_function(
            rng,
            env_state,
            new_agent_position,
            terminated,
            self.reward_shape
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
            self.number_terminal_states,
            self.grid_shape,
            self.observation_space,
            env_state,
        )

        # Use info dict to store the reason for finishing this episode. e.g. terminated, truncated, reached_terminal
        info = {
            'Reached_Target': terminated,
            'Reached_Terminal_State': reached_term,
            'Truncated': truncated
        }

        return env_state, (observation, reward, done, info)

    # Reset the environment before each episode
    def reset(self, rng: jax.random.PRNGKey):
        """
        Reset function of a single grid environment

            Functionality
            -------------
            - Randomly generates new positions for agent, target and terminal states for new episode
            - Init array for delayed rewards (needs to be stored in env_state)
            - Init new env_state
            - Compute observation for current state based on representation (vector, matrix, image)
        
        """
        required_positions = 2 + self.number_terminal_states
        agent_position, target_position, terminal_states = get_random_positions(
            rng=rng, 
            grid_shape=self.grid_shape, 
            n=required_positions,
        )

        delayed_rewards = jnp.zeros(self.max_steps_in_episode, dtype=jnp.float64)
        
        env_state = EnvState(
            agent_position=agent_position,
            target_position=target_position, 
            terminal_states=terminal_states,
            delayed_rewards = delayed_rewards,
            counter=1,
        )

        observation = get_obs(
            self.state_representation,
            self.number_terminal_states,
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