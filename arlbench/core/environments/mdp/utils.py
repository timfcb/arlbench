import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from typing import TYPE_CHECKING, Any, Tuple

from .spaces import BoxExtended, ImageContinuous

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

def get_obs(state_representation: str, irrelevant_features: bool, grid_shape: Tuple[int, int], observation_space: Any, env_state:Any) -> Any:
    """Computes the observation based on the state representation (image, vector, matrix).

    Args:
        state_representation (str): State representation type ('image', 'vector', 'matrix').
        irrelevant_features (bool): Whether to include irrelevant features in the observation.
        grid_shape (Tuple[int, int]): Shape of the grid world (width, height).
        env_state

    Returns:
        Any: Observation based on the state representation.
    """
    agent_position = env_state.agent_position
    target_position = env_state.target_position
    terminal_states = env_state.terminal_states

    # Image returns a greyscaled image
    if state_representation == 'image':
        rgb_image = observation_space.generate_image(env_state, irrelevant_features, grid_shape)
        observation = rgb_to_greyscale(rgb_image)
    # Vector returns a 4 dimensional vector with agent and target positions
    elif state_representation == 'vector':
        if irrelevant_features:
            observation = jnp.concatenate([agent_position, target_position, terminal_states.flatten(), agent_position, target_position, terminal_states.flatten()])
        else:
            observation = jnp.concatenate([agent_position, target_position, terminal_states.flatten()])
    # Matrix returns a matrix with 0 for empty cells, 1 for agent position and 2 for target position
    elif state_representation == 'matrix':
        if irrelevant_features:
            grid_shape = tuple(x * 2 for x in grid_shape)

        grid_matrix = jnp.zeros(grid_shape, dtype=jnp.int64)
        # Set agent position to 1 and target position to 2 (flip x/y coordinates for correct orientation)
        grid_matrix = grid_matrix.at[agent_position[1], agent_position[0]].set(1).at[target_position[1], target_position[0]].set(2)

        # number terminal states not 0
        if len(terminal_states[0]):
            # Separate rows (y) and columns (x)
            xs, ys = terminal_states[:, 0], terminal_states[:, 1]
            grid_matrix = grid_matrix.at[ys, xs].set(3)

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

# Terminal states integrated, TODO: Irrelevant features
def init_obs_space(state_representation: str, irrelevant_features: bool, grid_shape: Tuple[int, int], number_terminal_states: int) -> Any:
    # Initializing the observation space (RGB image or agent position as array)
    observation_space = None
    if irrelevant_features:
        irrelevant_factor = 2
    else:
        irrelevant_factor = 1

    if state_representation == 'vector':

        # 2 coordinates for agent position, 2 coordinates for target position, 2 coordinates for each terminal state, double dimension if irrelevant features 
        obs_space_dimension = (2 + 2 + 2*number_terminal_states) * irrelevant_factor

        observation_space = BoxExtended(
            jnp.array(jnp.zeros(obs_space_dimension, dtype=jnp.int64)),
            jnp.array([(grid_shape[0]) for _ in range(obs_space_dimension)], dtype=jnp.int64),
            (obs_space_dimension,),
            dtype=jnp.int64
        )

    elif state_representation == 'matrix':

        height = grid_shape[1]*irrelevant_factor
        width = grid_shape[0]*irrelevant_factor

        observation_space = BoxExtended(
            low=0,
            # Empty cell = 0, agent position = 1, target position = 2, terminal states = 3
            high=3,
            # shape: first dim is height (rows), second dim is width (cols), last dim is channel (1)
            shape= (height, width, 1),
            dtype=jnp.int64
        )

    elif state_representation == 'image':
        observation_space = ImageContinuous()

    else:
        raise ValueError(f"Unknown state representation: {state_representation}. Supported are 'vector', 'matrix', and 'image'.")

    return observation_space


def random_position(grid_shape, occupied_cells, rng, insert_idx):

    # Draw random target location unequal to agent location
    def body_fn(state):
        key, new_position, occupied_cells, insert_idx = state
        key, key_x, key_y = jax.random.split(key,3)
        new_position = jnp.stack([
            jax.random.randint(key_x, (), 0, grid_shape[0]),
            jax.random.randint(key_y, (), 0, grid_shape[1]),
        ])
        new_position = jnp.array([new_position], dtype=jnp.int64)

        return key, new_position, occupied_cells, insert_idx

    def cond_fn(state):

        _, target, occupied_cells, _ = state
        return jnp.any(jnp.all(occupied_cells == target, axis=1))
    
    # Only needed to start while loop
    occupied_position = occupied_cells[0]
    occupied_position = jnp.expand_dims(occupied_position, axis=0)
    state = (rng, occupied_position, occupied_cells, insert_idx)

    key_final, new_position, occupied_cells, insert_idx = jax.lax.while_loop(cond_fn, body_fn, state)

    occupied_cells = jax.lax.dynamic_update_slice(occupied_cells, new_position, (insert_idx,0))

    return new_position, occupied_cells, key_final

def get_random_positions(rng: PRNGKey, grid_shape: Tuple[int,int], n: int):

    # Draw random target location unequal to agent location
    def body_fn(state):

        rng, n, occupied_cells, generated_positions, insert_idx = state
        new_position, occupied_cells, rng = random_position(grid_shape, occupied_cells, rng, insert_idx)

        generated_positions = jax.lax.dynamic_update_slice(generated_positions, new_position, (insert_idx,0))

        insert_idx += 1

        return rng, n, occupied_cells, generated_positions, insert_idx

    def cond_fn(state):

        _, n, _, _, insert_idx = state
        return (insert_idx < n)

    entry = grid_shape[0]
    occupied_cells = jnp.full((n, 2), entry, dtype=jnp.int64)
    generated_positions = jnp.empty((n, 2), dtype=jnp.int64)
    state = (rng, n, occupied_cells, generated_positions, 0)

    key_final, n, occupied_cells, generated_positions, insert_idx = jax.lax.while_loop(cond_fn, body_fn, state)

    return generated_positions[0], generated_positions[1], generated_positions[2:]