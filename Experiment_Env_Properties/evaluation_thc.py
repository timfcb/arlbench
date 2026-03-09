import matplotlib.pyplot as plt
import numpy as np



thc_scores = {
    'GridSize': {'buffer_batch_size': 0.0, 'buffer_size': 0.0, 'initial_epsilon': 0.0333, 'target_epsilon': 0.0333, 'exploration_fraction': 0.0, 'gamma': 0.2054, 'gradient_steps': 0.2262, 'learning_rate': 0.35, 'learning_starts': 0.15, 'train_freq': 0.119, 'target_update_interval': 0.0, 'tau': 0.0167},
    'Scaling': {'buffer_batch_size': 0.0, 'buffer_size': 0.0, 'initial_epsilon': 0.0, 'target_epsilon': 0.0, 'exploration_fraction': 0.0, 'gamma': 0.0982, 'gradient_steps': 0.0714, 'learning_rate': 0.1, 'learning_starts': 0.0, 'train_freq': 0.0, 'target_update_interval': 0.0, 'tau': 0.0},
    'Probability': {'buffer_batch_size': 0.075, 'buffer_size': 0.225, 'initial_epsilon': 0.0667, 'target_epsilon': 0.05, 'exploration_fraction': 0.0, 'gamma': 0.3125, 'gradient_steps': 0.1429, 'learning_rate': 0.325, 'learning_starts': 0.0, 'train_freq': 0.0357, 'target_update_interval': 0.0089, 'tau': 0.0278},
    'Shift': {'buffer_batch_size': 0.2, 'buffer_size': 0.25, 'initial_epsilon': 0.0, 'target_epsilon': 0.1333, 'exploration_fraction': 0.0, 'gamma': 0.4107, 'gradient_steps': 0.1429, 'learning_rate': 0.375, 'learning_starts': 0.0, 'train_freq': 0.0357, 'target_update_interval': 0.0089, 'tau': 0.0556},
    'SuccessReward': {'buffer_batch_size': 0.075, 'buffer_size': 0.3, 'initial_epsilon': 0.0, 'target_epsilon': 0.1333, 'exploration_fraction': 0.0, 'gamma': 0.2143, 'gradient_steps': 0.3095, 'learning_rate': 0.425, 'learning_starts': 0.05, 'train_freq': 0.0714, 'target_update_interval': 0.0, 'tau': 0.0},
    'TerminalStates': {'buffer_batch_size': 0.0, 'buffer_size': 0.275, 'initial_epsilon': 0.0, 'target_epsilon': 0.05, 'exploration_fraction': 0.0, 'gamma': 0.4107, 'gradient_steps': 0.0238, 'learning_rate': 0.15, 'learning_starts': 0.05, 'train_freq': 0.0476, 'target_update_interval': 0.0089, 'tau': 0.0111},
    'TransitionNoise': {'buffer_batch_size': 0.05, 'buffer_size': 0.1, 'initial_epsilon': 0.0, 'target_epsilon': 0.05, 'exploration_fraction': 0.0, 'gamma': 0.2321, 'gradient_steps': 0.0357, 'learning_rate': 0.325, 'learning_starts': 0.0, 'train_freq': 0.0476, 'target_update_interval': 0.0089, 'tau': 0.0222},
} 


x = ['buffer_batch_size', 'buffer_size', 'initial_epsilon', 'target_epsilon', 'exploration_fraction', 'gamma', 'gradient_steps', 'learning_rate', 'learning_starts', 'train_freq', 'target_update_interval', 'tau']

for key, values in thc_scores.items():
    y = [value for value in values.values()]
    plt.plot(x, y, label=key)

plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('thc_scores.png')