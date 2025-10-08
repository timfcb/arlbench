import jax.numpy as jnp
from flax import struct


#Dataclass for defining Environment States
@struct.dataclass
class EnvState():
    """Environment state for MDP Playground."""
    agent_position: jnp.ndarray
    target_position: jnp.ndarray
    terminal_states: jnp.ndarray = jnp.array([], dtype=jnp.int64)
    delayed_rewards: jnp.ndarray = jnp.array([], dtype=jnp.float64)
    counter: int = 1

@struct.dataclass
class RewardShape():
    """Reward Structure for MDP Playground"""
    delay: int = 0
    noise: float = 0.0
    scaling_factor: float = 0.0
    shift: float = 0.0
    every_n_steps: int = 1
    dense: bool = True
    probability: float = 1.0
