# Dimensions of Hardness for MDP Environments

| Property       | Possible Values |
| --------------- | --------------- |
| Grid Size       | 3-15 (squared)  |
| Observation Space Representations | Vector, Matrix, Image |
| Irrelevant Features | True/False |
| Transition Noise    | 0.0-0.5    |
| Maximum Number of Steps per Episode | 15-50 |
| Reward Delay | 0-5 |
| Reward Noise: Standard Deviation of Normal Dist. | 0-5 |
| Reward Scaling Factor | 1-5 |
| Reward Shift | 0-5 |
| Reward Probability | 0.0-0.5 |
| Reward Density | True/False |


Implementation of MDP Playground inspired by [MDP Playground](https://github.com/automl/mdp-playground/blob/master/mdp_playground/envs/rl_toy_env.py). Detailed explanations of the Properties of the Environment can be found here: [MDP Playground Paper](https://arxiv.org/pdf/1909.07750) by Rajan et al.
