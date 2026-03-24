import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Any


def reward_normal_step(args):

    reward_parameters, env_state, new_agent_position = args

    # Dense reward: Reward is given for every step (change in manhattan distance to target)
    manhat_dist_old = jnp.sum(jnp.abs(env_state.agent_position - env_state.target_position))
    manhat_dist_new = jnp.sum(jnp.abs(new_agent_position - env_state.target_position))
    reward = (manhat_dist_old - manhat_dist_new)

    # Scaling with reward scaling factor (theta 4)
    reward *= reward_parameters.scaling_factor

    # Shift with reward shift (theta 3)
    reward += reward_parameters.shift

    return jnp.float64(reward)

def compute_delayed_rewards(rng: jax.random.PRNGKey, reward: jnp.float64, delay_prob: jnp.float64, delayed_rewards: jnp.ndarray, done:Any):
    """Computes the accumulated reward for this step and updates the delayed rewards array.
    Args:
        rng (PRNGKey): PRNG key, consumable by random functions.
        reward (jnp.float64): Computed reward obatained for the current step
        delay_prob (jnp.float64): Probability that element in delayed_reward list is delayed and shifted one position
        delayed_rewards (jnp.ndarray): Array that stores the rewards received at single env steps: index of array equals number of step

    Returns:
        accumulated_reward_in_step (jnp.float): Accumulated reward that is distributed at the current step
        remaining_rewards (jnp.ndarray): Updated delayed reward array -> Not distributed rewards are shifted by one index
    """
    max_steps = len(delayed_rewards)

    updated_delayed_rewards = jnp.concatenate([jnp.array([reward]), delayed_rewards[:-1]])

    rng, key = jax.random.split(rng)
    rand_values = jax.random.uniform(key, shape=(max_steps,))

    delay_threshold = jnp.full(shape=(max_steps, ), fill_value=delay_prob, dtype=jnp.float64)

    index_mask = jnp.where(rand_values < delay_threshold, False, True)

    accumulated_reward_in_step = jnp.sum(jnp.where(index_mask==True, updated_delayed_rewards, jnp.float64(0.0)))

    remaining_rewards = jnp.where(index_mask==False, updated_delayed_rewards, jnp.float64(0.0))

    ### If condition required for flushing the buffer if episode done
    accumulated_reward_in_step = jax.lax.cond(
        done,
        lambda remaining_rewards : accumulated_reward_in_step + jnp.sum(remaining_rewards),
        lambda remaining_rewards: accumulated_reward_in_step,
        operand=remaining_rewards
    )

    return accumulated_reward_in_step, remaining_rewards

def compute_reward(reward_state):
    """Computes the reward value for taking a certain action in current state.
    Args:
        reward_state: (rng_compute, env_state, new_agent_position, terminated, reached_term,reward_parameters).
    Returns:
        reward_in_step (jnp.float): Reward for current step for the agent
        remaining_rewards (jnp.ndarray): Keeps track of all delayed rewards
    """
    rng, env_state, new_agent_position, terminated, reached_term, done, reward_parameters = reward_state
    reward = jnp.float64(0.0)
    rng, rng_noise = jax.random.split(rng)
    normal_step_params = (reward_parameters, env_state, new_agent_position)

    # Reward structure: three different cases: Reaching target, reaching terminal failure state, normal step
    reward = jax.lax.cond(
        terminated,
        lambda: jnp.float64(reward_parameters.success_reward),
        lambda: jax.lax.cond(
            reached_term,
            lambda _: jnp.float64(reward_parameters.terminal_state_penalty),
            lambda normal_step_params: reward_normal_step(normal_step_params),
            operand=normal_step_params,
        )
    )

    # Adding reward noise to the reward signal
    rng, rng_noise = jax.random.split(rng)
    computed_noise = jax.random.normal(rng_noise) * reward_parameters.noise
    reward += computed_noise

    reward_in_step, delayed_rewards = compute_delayed_rewards(rng, reward, reward_parameters.delay_prob, env_state.delayed_rewards, done)
    
    return reward_in_step, delayed_rewards

def reward_function(rng: jax.random.PRNGKey, env_state: Any, new_agent_position: jnp.ndarray, terminated: Any, reached_term: Any, done: Any, reward_parameters: Any):
    """Central function for managing reward structure
    Args:
        rng (PRNGKey): PRNG key, consumable by random functions.
        env_state (EnvState): State of the environment (e.g. agent and target position)
        new_agent_position (jnp.ndarray): Position of agent after taking action
        reward_shape (RewardShape): Object that contains all required properties for computing the reward signal
    Returns:
        reward_in_step (jnp.float): Reward for current step for the agent
        delayed_reward (jnp.ndarray): Keeps track of all delayed rewards
    """
    # Reward Probability
    rng, rng_prob = jax.random.split(rng)
    random_value = jax.random.uniform(rng_prob)
    rng, rng_compute = jax.random.split(rng)
    reward_state = (rng_compute, env_state, new_agent_position, terminated, reached_term, done, reward_parameters)
    reward_in_step, delayed_rewards = jax.lax.cond(
        random_value < reward_parameters.probability,
        lambda reward_state: compute_reward(reward_state),
        lambda _: (jnp.float64(0.0), jnp.concatenate([jnp.array([0.0]), env_state.delayed_rewards[:-1]])),
        operand=reward_state
    )

    return reward_in_step, delayed_rewards