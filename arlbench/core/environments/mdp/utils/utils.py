import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from typing import TYPE_CHECKING, Any, Tuple

# Randomly draw n positions in the grid
def get_random_positions(rng: jax.random.PRNGKey, grid_shape: Tuple[int,int], n: int):
    """Generates n random positions in the grid which is needed at the begin of each episode
    Args:
        rng (PRNGKey): PRNG key, consumable by random functions.
        grid_shape (Tuple[Int,Int]): Quadratic shape of the grid environment (e.g. (3,3))
        n (int): number of positions required (two for agent resp. target position, and one for each terminal state)
    Returns:
        grid_positions[0] (jnp.ndarray): Start position of the agent in new episode
        grid_positions[1] (jnp.ndarray): Target position in new episode
        grid_positions[2:] (jnp.ndarray): List of terminal_states for new episode, can be empty
    """
    number_cells = grid_shape[0] * grid_shape[1]

    indices = jax.random.choice(rng, a=number_cells, shape=(n,), replace=False)

    ### Only works for quadratic grids
    # Mapping of indices to cells by using divmod [quotient, remainder]
    grid_dim = jnp.full(len(indices),grid_shape[0], dtype=jnp.int64)
    fract_div = jnp.divmod(indices, grid_dim)

    x_coords, y_coords = fract_div
    x_coords = jnp.expand_dims(x_coords, axis=1)
    y_coords = jnp.expand_dims(y_coords, axis=1)

    grid_positions = jnp.concatenate([x_coords, y_coords], axis=1)

    # Return agent position, target position and list of terminal states
    return grid_positions[0], grid_positions[1], grid_positions[2:]