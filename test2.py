import jax.numpy as jnp
import jax
from typing import TYPE_CHECKING, Any, Tuple
if TYPE_CHECKING:
    from chex import PRNGKey

jax.config.update("jax_enable_x64", True)


grid_shape = (2,2)
n=4



def get_random_positions(rng: jax.random.PRNGKey, grid_shape: Tuple[int,int], n: int):

    number_cells = grid_shape[0] * grid_shape[1]

    indices = jax.random.choice(rng, a=number_cells, shape=(n,), replace=False)

    # Mapping of indices to cells by using divmod [quotient, remainder]
    grid_dim = jnp.full(len(indices),grid_shape[0], dtype=jnp.int64)
    fract_div = jnp.divmod(indices, grid_dim)

    x_coords, y_coords = fract_div
    x_coords = jnp.expand_dims(x_coords, axis=1)
    y_coords = jnp.expand_dims(y_coords, axis=1)

    grid_positions = jnp.concatenate([x_coords, y_coords], axis=1)

    return grid_positions[0], grid_positions[1], grid_positions[2:]



def random_position(grid_shape: Tuple[int,int], occupied_cells: jnp.ndarray, rng: jax.random.PRNGKey, insert_idx: int):

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

def get_random_positions_2(rng: jax.random.PRNGKey, grid_shape: Tuple[int,int], n: int):

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

rng = jax.random.PRNGKey(0)
rng, rng_positions = jax.random.split(rng)
agent_position, target_position, terminal_states = get_random_positions(rng_positions, grid_shape, n)

print(agent_position)
print(target_position)
print(terminal_states)

### Comparison Old Implementation ###

rng, rng_positions_2 = jax.random.split(rng)
agent_position, target_position, terminal_states = get_random_positions_2(rng_positions_2, grid_shape, n)

print(agent_position)
print(target_position)
print(terminal_states)