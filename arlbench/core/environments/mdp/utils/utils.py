import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from typing import TYPE_CHECKING, Any, Tuple

from ..spaces import BoxExtended, ImageContinuous
from ..data_classes import RewardShape

# File for all helper functions, TODO more sturcture

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

def get_obs(state_representation: str, number_terminal_states: int, grid_shape: Tuple[int, int], observation_space: Any, env_state:Any) -> Any:
    """Computes the observation based on the state representation (image, vector, matrix).

    Args:
        state_representation (str): State representation type ('image', 'vector', 'matrix').
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
        rgb_image = observation_space.generate_image(env_state, grid_shape)
        observation = rgb_to_greyscale(rgb_image)
    # Vector returns a 4 dimensional vector with agent and target positions
    elif state_representation == 'vector':
        observation = jnp.concatenate([agent_position, target_position, terminal_states.flatten()])
    # Matrix returns a matrix with 0 for empty cells, 1 for agent position and 2 for target position
    elif state_representation == 'matrix':

        grid_matrix_agent = jnp.zeros(grid_shape, dtype=jnp.int64).at[agent_position[1], agent_position[0]].set(1)
        grid_matrix_target = jnp.zeros(grid_shape, dtype=jnp.int64).at[target_position[1], target_position[0]].set(1)
        grid_matrix = jnp.stack([grid_matrix_agent, grid_matrix_target], axis=2)

        if number_terminal_states:
            # Separate rows (y) and columns (x)
            grid_matrix_terminal = jnp.zeros(grid_shape, dtype=jnp.int64)
            xs, ys = terminal_states[:, 0], terminal_states[:, 1]
            grid_matrix_terminal = grid_matrix_terminal.at[ys, xs].set(1)
            grid_matrix = jnp.stack([grid_matrix_agent, grid_matrix_target, grid_matrix_terminal], axis=2)

        observation = grid_matrix

    else:
        raise ValueError(f"Unknown state representation: {state_representation}. Supported are 'image', 'vector', and 'matrix'.")
    
    return observation

def compute_delayed_rewards(rng: jax.random.PRNGKey, reward: jnp.float64, delay_prob: jnp.float64, max_steps: int, delayed_rewards: jnp.ndarray):

    updated_delayed_rewards = jnp.concatenate([jnp.array([reward]), delayed_rewards[:-1]])

    rng, key = jax.random.split(rng)
    rand_values = jax.random.uniform(key, shape=(max_steps,))
    delay_threshold = jnp.full(shape=(max_steps, ), fill_value=delay_prob, dtype=jnp.float64)

    index_mask = jnp.where(rand_values < delay_threshold, False, True)

    reward_in_step = jnp.sum(jnp.where(index_mask==True, updated_delayed_rewards, jnp.float64(0.0)))
    remaining_rewards = jnp.where(index_mask==False, updated_delayed_rewards, jnp.float64(0.0))

    return reward_in_step, remaining_rewards

def compute_reward(reward_state):

    rng, env_state, new_agent_position, reward_shape = reward_state
    noise = reward_shape.noise
    scaling_factor = reward_shape.scaling_factor
    shift = reward_shape.shift
    delay_prob = reward_shape.delay_prob
    max_steps = len(env_state.delayed_rewards)

    reward = jnp.float64(0.0)

    # Dense reward: Reward is given for every step (change in manhattan distance to target)
    manhat_dist_old = jnp.sum(jnp.abs(env_state.agent_position - env_state.target_position))
    manhat_dist_new = jnp.sum(jnp.abs(new_agent_position - env_state.target_position))
    reward = manhat_dist_old - manhat_dist_new

    # Environment property: Reward scaling
    reward *= scaling_factor

    # Environment property: Reward shift
    reward += shift

    # Environment property: Reward noise
    rng, rng_noise = jax.random.split(rng)
    computed_noise = jax.random.normal(rng_noise) * noise
    reward += computed_noise

    reward_in_step, delayed_rewards = compute_delayed_rewards(rng, reward, delay_prob, max_steps, env_state.delayed_rewards)
    
    return reward_in_step, delayed_rewards

# 
def reward_function(rng: jax.random.PRNGKey, env_state: Any, new_agent_position: jnp.ndarray, terminated: bool, reward_shape: Any):

    # Reward Probability
    rng, rng_prob = jax.random.split(rng)
    random_value = jax.random.uniform(rng_prob)
    rng, rng_compute = jax.random.split(rng)
    reward_state = (rng_compute, env_state, new_agent_position, reward_shape)
    reward_in_step, delayed_rewards = jax.lax.cond(
        random_value < reward_shape.probability,
        lambda reward_state: compute_reward(reward_state),
        lambda _: (jnp.float64(0.0), jnp.concatenate([jnp.array([0.0]), env_state.delayed_rewards[:-1]])),
        operand=reward_state
    )

    return reward_in_step, delayed_rewards

# Init Observation Space
def init_obs_space(state_representation: str, grid_shape: Tuple[int, int], number_terminal_states: int) -> Any:
    # Initializing the observation space (RGB image or agent position as array)
    observation_space = None

    if state_representation == 'vector':

        # 2 coordinates for agent position, 2 coordinates for target position, 2 coordinates for each terminal state 
        obs_space_dimension = (2 + 2 + 2*number_terminal_states)

        observation_space = BoxExtended(
            jnp.array(jnp.zeros(obs_space_dimension, dtype=jnp.int64)),
            jnp.array([(grid_shape[0]-1) for _ in range(obs_space_dimension)], dtype=jnp.int64),
            (obs_space_dimension,),
            dtype=jnp.int64
        )

    elif state_representation == 'matrix':

        height = grid_shape[1]
        width = grid_shape[0]

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

# Randomly draw n positions in the grid
def get_random_positions(rng: jax.random.PRNGKey, grid_shape: Tuple[int,int], n: int):

    number_cells = grid_shape[0] * grid_shape[1]

    indices = jax.random.choice(rng, a=number_cells, shape=(n,), replace=False)

    # Mapping of indices to cells by using divmod [quotient, remainder] ### Only works for quadratic grids
    grid_dim = jnp.full(len(indices),grid_shape[0], dtype=jnp.int64)
    fract_div = jnp.divmod(indices, grid_dim)

    x_coords, y_coords = fract_div
    x_coords = jnp.expand_dims(x_coords, axis=1)
    y_coords = jnp.expand_dims(y_coords, axis=1)

    grid_positions = jnp.concatenate([x_coords, y_coords], axis=1)

    return grid_positions[0], grid_positions[1], grid_positions[2:]


#### Old Implementation of drawing randomly positions in the grid, Works completely fine but could be done faster###
'''
def random_position(grid_shape: Tuple[int,int], occupied_cells: jnp.ndarray, rng: PRNGKey, insert_idx: int):

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
'''