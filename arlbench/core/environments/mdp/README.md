# Dimensions of Hardness for MDP Environments

| Property       | Possible Values | Default Value |
| --------------- | --------------- | ------------- |
| Grid Size       | 3-15 (squared)  | (5,5) |
| Observation Space Representations | Vector, Matrix, Image | Vector |
| Fraction of Terminal States| 0.0-0.9 | 0.2 |
| Transition Noise    | 0.0-0.5    | 0.0 |
| Maximum Number of Steps per Episode | 15-50 | 2 * (grid_width + grid_height) |
| Reward Delay | 0-5 | 0 |
| Reward Noise: Standard Deviation of Normal Dist. | 0-5 | 0 |
| Reward Scaling Factor | 1-5 | 1 |
| Reward Shift | -5-5 | 0 |
| Reward Probability | 0.5-1.0 | 1.0 |
| Dense Rewards | True/False | True |
| Reward after every n-th step | 1-5 | 1 |


Implementation of our version of MDP Playground inspired by Toy Environment of [MDP Playground](https://github.com/automl/mdp-playground/blob/master/mdp_playground/envs/rl_toy_env.py). Step/Reset function of our MDP Playground environment fully built up in JAX. More information about JAX here: [JAX](https://github.com/jax-ml/jax). Detailed explanations of the Properties of the Environment can be found here: [MDP Playground Paper](https://arxiv.org/pdf/1909.07750) by Rajan et al.
