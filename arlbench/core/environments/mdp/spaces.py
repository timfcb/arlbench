import gymnax
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from typing import Tuple
from typing import TYPE_CHECKING, Any

### Spaces for the RL Toy Environment ### 
# Used for Vector and Matrix Representation
class BoxExtended(gymnax.environments.spaces.Box):
    """A space that maps the current position of the agent and target to a box representation of the environment.

    Methods
    -------
    sample(self, key: jax.random.PRNGKey)
        Generates new position inside the Box
    """
    def __init__(self, low, high, shape=None, dtype=jnp.int64, seed=None):

        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    # Overrides sample method from Box space in gymnax: We need random int instead of random float
    def sample(self, key: jax.random.PRNGKey) -> jax.Array:
        return jax.random.randint(key=key, shape=self.shape, minval=self.low, maxval=self.high, dtype=self.dtype)

# Used for Image Representation 
class ImageContinuous(gymnax.environments.spaces.Box):
    """A space that maps the current position of the agent and target to an image representation of the environment.

    Methods
    -------
    generate_image(self, agent_position, target_position)
        Creates a rgb image of the environment with the agent and target drawn as colored squares of shape [84, 84, 3]. Afterwards computes greyscaled version [84, 84, 1].
    """

    def __init__(self):

        self.width = 84
        self.height = 84
        self.num_channels = 3
 
        super(ImageContinuous, self).__init__(
            shape=(self.width, self.height, self.num_channels), dtype=jnp.uint8, low=0, high=255
        )
    
    def generate_image(self, env_state:Any, grid_shape: Tuple[int,int], number_terminal_states: int) -> jax.Array:
        """Returns an array shaped (84,84,3) representing rgb-scheme of the current grid state.
           Grid is drawn with white background and black lines
           Agent Position is represented by filling the corresponding cell in colour green (0, 255, 0)
           Target Position is represented by filling the corresponding cell in colour yellow (255, 255, 0)
           Terminal States are represented by filling the corresponding cells in colour dark blue (0,0,128)

        Args:
            env_state (EnvState): Current state of the env containing e.g. agent/target position
            grid_shape (Tuple[Int, Int]): Shape of the grid environment (e.g. (5,5)).

        Returns:
            jnp.ndarry: rgb image of the environment with shape [84, 84, 3].
        """

        agent_position = env_state.agent_position
        target_position = env_state.target_position
        terminal_states = env_state.terminal_states

        cols, rows = grid_shape

        final_size = 84
        square_ratio = 1
        grid_thickness = 1
        # black
        line_colour = (0, 0, 0)
        terminal_state_colour = (0,0,255)
        goal_colour = (0,255,0)
        agent_colour = (0,255,0)
        target_colour = (255,0,0)
        fail_colour = (255,0,0)

        # White img with 84x84 pixels
        img = jnp.full((84, 84, 3), 255, dtype=jnp.uint8)

        #Compute cell size (integer)
        cell_size = (final_size - 1) // cols

        grid_w = cols * cell_size
        grid_h = rows * cell_size
        canvas_w = grid_w + 1
        canvas_h = grid_h + 1

        # Center the grid inside the final image
        pad_total_x = final_size - canvas_w
        pad_left = pad_total_x // 2
        pad_total_y = final_size - canvas_h
        pad_top = pad_total_y // 2

        def draw_square_centered(img, pos, color):
            col, row = pos
            
            # Compute exact top-left corner of the cell in the padded image
            cell_x = pad_left + col * cell_size
            cell_y = pad_top + row * cell_size

            # Compute square size and margins
            square_size = int(cell_size * square_ratio)
            margin = (cell_size - square_size) // 2

            # Compute top-left corner of the square
            x1 = cell_x + margin
            y1 = cell_y + margin

            # Construct the replacement block (square)
            square = jnp.full((square_size, square_size, img.shape[2]), jnp.array(color), dtype=jnp.uint8)

            # Dynamically insert the square into the image
            img = jax.lax.dynamic_update_slice(img, square, (y1, x1, 0))

            return img

        # If AutoReset Wrapper is switched off --> if target and agent are same cell (reached target): color cell in color (128,128,0)
        def target_not_reached(img):
            # Draw target and agent squares centered in their cells
            img = draw_square_centered(img, target_position, target_colour)
            img = draw_square_centered(img, agent_position, agent_colour)
            return img

        def target_reached(img):
            img = draw_square_centered(img, agent_position, goal_colour)
            return img

        img = jax.lax.cond(
            jnp.all(agent_position == target_position),
            lambda img: target_reached(img),
            lambda img: target_not_reached(img),
            operand=img
        )

        if number_terminal_states:
            for ind, elem in enumerate(terminal_states):

                img = jax.lax.cond(
                    jnp.all(agent_position == elem),
                    lambda img: draw_square_centered(img, agent_position, fail_colour),
                    lambda img: draw_square_centered(img, elem, terminal_state_colour),
                    operand=img
                )

        # Draw vertical grid lines
        for i in range(cols + 1):
            x = pad_left + i * cell_size
            x_end = min(x + grid_thickness, pad_left + canvas_w)
            img = img.at[pad_top:pad_top + canvas_h, x:x_end, :].set(jnp.array(line_colour))
 
        # Draw horizontal grid lines
        for j in range(rows + 1):
            y = pad_top + j * cell_size
            y_end = min(y + grid_thickness, pad_top + canvas_h)
            img = img.at[y:y_end, pad_left:pad_left + canvas_w, :].set(jnp.array(line_colour))

        return img