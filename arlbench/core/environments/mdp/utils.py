import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from typing import TYPE_CHECKING, Any, Tuple

# Greyscaling used in Atari preprocessing (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
def rgb_to_greyscale(rgb_image: jnp.ndarray) -> jnp.ndarray:
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

def get_obs(state_representation: str, irrelevant_features: bool, grid_shape: Tuple[int, int], observation_space: Any, agent_position: jax.Array, target_position: jax.Array) -> Any:
    """Computes the observation based on the state representation (image, vector, matrix).

    Args:
        state_representation (str): State representation type ('image', 'vector', 'matrix').
        irrelevant_features (bool): Whether to include irrelevant features in the observation.
        grid_shape (Tuple[int, int]): Shape of the grid world (width, height).
        agent_position: Current agent position.
        target_position: Current target position.

    Returns:
        Any: Observation based on the state representation.
    """
    # Image returns a greyscaled image
    if state_representation == 'image':
        rgb_image = observation_space.generate_image(agent_position, target_position, irrelevant_features, grid_shape)
        observation = rgb_to_greyscale(rgb_image)
    # Vector returns a 4 dimensional vector with agent and target positions
    elif state_representation == 'vector':
        if irrelevant_features:
            observation = jnp.concatenate([agent_position, target_position, agent_position, target_position])
        else:
            observation = jnp.concatenate([agent_position, target_position])
    # Matrix returns a matrix with 0 for empty cells, 1 for agent position and 2 for target position
    elif state_representation == 'matrix':
        if irrelevant_features:
            grid_shape = tuple(x * 2 for x in grid_shape)

        grid_matrix = jnp.zeros(grid_shape, dtype=jnp.int64)
        # Set agent position to 1 and target position to 2 (flip x/y coordinates for correct orientation)
        grid_matrix = grid_matrix.at[agent_position[1], agent_position[0]].set(1).at[target_position[1], target_position[0]].set(2)
        observation = jnp.expand_dims(grid_matrix, axis=-1)

    else:
        raise ValueError(f"Unknown state representation: {state_representation}. Supported are 'image', 'vector', and 'matrix'.")
    
    return observation

def compute_reward(rng: PRNGKey, env_state: Any, new_agent_position: jnp.ndarray, terminated: bool, dense_reward: bool, reward_scale: float, reward_shift: float, reward_probability: float, reward_delay: int, reward_noise_std:float):
    
    reward = jnp.float64(0.0)

    # Environment property: Dense vs Sparse reward
    if dense_reward:
        # Dense reward: Reward is given for every step (change in manhattan distance to target)
        manhat_dist_old = jnp.sum(jnp.abs(env_state.agent_position - env_state.target_position))
        manhat_dist_new = jnp.sum(jnp.abs(new_agent_position - env_state.target_position))
        reward = manhat_dist_old - manhat_dist_new
    else:
        # Sparse reward: Reward is only given when target is reached
        reward = jnp.where(terminated, 1.0, 0.0)

    # Environment property: Reward scaling
    reward = reward * reward_scale

    # Environment property: Reward shift
    reward = reward + reward_shift

    # Environment property: Reward noise
    rng, rng_noise = jax.random.split(rng)
    noise = jax.random.normal(rng_noise) * reward_noise_std
    reward = reward + noise

    # Environment property: Reward probability
    rng, rng_reward = jax.random.split(rng)
    rand_value = jax.random.uniform(rng_reward)
    reward = jax.lax.cond(
        rand_value < reward_probability,
        lambda _: reward,
        lambda _: jnp.float64(0.0),
        operand=None
    )
    
    # Environment property: Reward delay
    if reward_delay > 1:
        returned_reward = env_state.delayed_rewards[0]
        delayed_rewards = jnp.append(env_state.delayed_rewards[1:], reward)
    elif reward_delay == 1:
        returned_reward = env_state.delayed_rewards[0]
        delayed_rewards = jnp.array([reward])
    else: # reward delay == 0
        delayed_rewards = jnp.array([])
        returned_reward = reward

    # TODO Discuss with Julian: What happens at beginning/end of episode with delayed rewards?
    
    return returned_reward, delayed_rewards