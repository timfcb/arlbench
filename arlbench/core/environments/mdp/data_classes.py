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
class RewardParameters():
    """Reward Structure for MDP Playground"""
    # Theta 1 in thesis
    success_reward: float = 1.0
    # Theta 2 in thesis
    terminal_state_penalty: float = -1.0
    # Theta 3 in thesis
    shift: float = 0.0
    # Theta 4 in thesis
    scaling_factor: float = 1.0
    noise: float = 0.0
    delay_prob: float = 1.0
    probability: float = 1.0