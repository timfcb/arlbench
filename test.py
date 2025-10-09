import jax.numpy as jnp
import jax
from typing import TYPE_CHECKING, Any, Tuple
if TYPE_CHECKING:
    from chex import PRNGKey

jax.config.update("jax_enable_x64", True)

grid_shape = (4,4)

grid_width, grid_height = grid_shape

number_cells = grid_shape[0] * grid_shape[1]

key = jax.random.PRNGKey(0)

numbers = jax.random.choice(key, a=number_cells, shape=(10,), replace=False)

full_array = jnp.full(len(numbers),grid_width, dtype=jnp.int64)

### Only Modulo ###
print('Modulo Operation')
fract = jnp.mod(numbers, full_array)
#print(numbers)
#print(full_array)
#print(fract)

### Now DivMod ###
fract_div = jnp.divmod(numbers, full_array)

print('Div Mod Operation')
#print(numbers)
#print(full_array)
print(fract_div)

x_coords, y_coords = fract_div

x_coords = jnp.expand_dims(x_coords, axis=1)
y_coords = jnp.expand_dims(y_coords, axis=1)

grid_positions = jnp.concatenate([x_coords, y_coords], axis=1)

print(grid_positions)
print(len(grid_positions))

jax.debug.breakpoint()