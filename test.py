import jax.numpy as jnp


grid_shape = (3,3)


agent_position = jnp.array([2,1])
target_position = jnp.array([0,1])


grid_shape = tuple(x * 2 for x in grid_shape)


grid_matrix = jnp.zeros(grid_shape, dtype=jnp.int64)
# Set agent position to 1 and target position to 2 (flip x/y coordinates for correct orientation)
grid_matrix = grid_matrix.at[agent_position[1], agent_position[0]].set(1).at[target_position[1], target_position[0]].set(2)
observation = jnp.expand_dims(grid_matrix, axis=-1)


print(grid_matrix)