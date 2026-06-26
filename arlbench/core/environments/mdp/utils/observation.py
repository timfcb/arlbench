import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from typing import TYPE_CHECKING, Any, Tuple
from ..spaces import BoxExtended, ImageContinuous

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
        rgb_image = observation_space.generate_image(env_state, grid_shape, number_terminal_states)
        grayscaled_img = rgb_to_greyscale(rgb_image)

        # Normalize observation
        #observation = grayscaled_img / 255.0
        #jax.debug.callback(plot_imgs, rgb_image, 1)
        #jax.debug.callback(plot_imgs, grayscaled_img, 2)
        observation = grayscaled_img
        #jax.debug.callback(save_obs, observation)

    # Vector returns a 4 dimensional vector with agent and target positions
    elif state_representation == 'vector':
        vector = jnp.concatenate([agent_position, target_position, terminal_states.flatten()])
        observation = vector / (grid_shape[0]-1)

    # Matrix returns a matrix with 0 for empty cells, 1 for agent position and 2 for target position
    elif state_representation == 'matrix':

        grid_matrix_agent = jnp.zeros(grid_shape, dtype=jnp.int64).at[agent_position[1], agent_position[0]].set(1)
        grid_matrix_target = jnp.zeros(grid_shape, dtype=jnp.int64).at[target_position[1], target_position[0]].set(1)

        if number_terminal_states:
            # Separate rows (y) and columns (x)
            grid_matrix_terminal = jnp.zeros(grid_shape, dtype=jnp.int64)
            xs, ys = terminal_states[:, 0], terminal_states[:, 1]
            grid_matrix_terminal = grid_matrix_terminal.at[ys, xs].set(1)
            grid_matrix = jnp.stack([grid_matrix_agent, grid_matrix_target, grid_matrix_terminal], axis=0)
        else:
            grid_matrix = jnp.stack([grid_matrix_agent, grid_matrix_target], axis=0)

        observation = grid_matrix

    else:
        raise ValueError(f"Unknown state representation: {state_representation}. Supported are 'image', 'vector', and 'matrix'.")
    
    return observation


def init_obs_space(state_representation: str, grid_shape: Tuple[int, int], number_terminal_states: int) -> Any:
    """Initialization of the observation space based on selected state representation (vector, matrix, image).

    Args:
        state_representation (str): State representation type ('image', 'vector', 'matrix').
        grid_shape (Tuple[int, int]): Shape of the grid world (width, height).
        number_terminal_states (int): Indicates the number of terminal states in the grid

    Returns:
        Box or Image Space: Based on state representation
    """
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
