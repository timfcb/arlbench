import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics

from utils import get_data, load_info, change_format

### File for providing interesting visualizations on the grid search experiments

# Creates performance line plots with CIs for each value of a hyperparameter
def plot_results(experiment_name, save_to_folder=False):
    
    info_dict = load_info(experiment_name)
    hp_list = list(info_dict['hp'].keys())

    number_envs = info_dict['different_envs']

    # Will be used for x axis
    x_float = np.arange(1,number_envs+1,1)
    x = [int(elem) for elem in x_float]

    if (len(hp_list)) > 0 and save_to_folder:
        # Create dict:
        plot_folder = Path(f'results/{experiment_name}/visualisations')
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
    else:
        plot_folder = ''

    # IQM intervals for several runs on identical settings with varying seeds
    aggregate_function = lambda x: np.array([metrics.aggregate_iqm(x)])

    # Plot for each hyperparameter seperately
    for ind_hp, hp in enumerate(hp_list):
        data = get_data(hp, experiment_name)

        hp_values = info_dict['hp'][hp]

        all_iqm = []
        all_cis = []

        for ind_env, env in enumerate(data):
            
            values_dict = dict(zip(hp_values, np.expand_dims(env, axis=2)))

             ## Get IQM Values for plotting
            aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                values_dict, aggregate_function, reps=5000
            )

            iqm_scores = np.array(list(aggregate_scores.values()))
            iqm_list = np.squeeze(iqm_scores, axis=1)

            cis = np.array(list(aggregate_score_cis.values()))
            cis_list = np.squeeze(cis, axis=2)

            all_iqm.append(iqm_list)
            all_cis.append(cis_list)

        curves = []

        plt.figure(figsize=(7, 5))

        for val in range(len(hp_values)):
            iqm_val = [elem[val] for elem in all_iqm]
            ci_val = [elem[val] for elem in all_cis]

            lower_bounds_val = [bound[0] for bound in ci_val]
            upper_bounds_val = [bound[1] for bound in ci_val]

            curve_val = tuple((f'{hp_values[val]}', iqm_val, lower_bounds_val, upper_bounds_val))

            curves.append(curve_val)

        for label, iqm, low, up in curves:
            plt.plot(x, iqm, label=label)
            plt.fill_between(x, low, up, alpha=0.25)

        plt.title(f'Hyperparameter: {change_format(hp)}')
        plt.xticks(x)
        plt.xlabel('Environment ID')
        plt.ylabel('Performances (Avg Return)')
        plt.legend(loc="upper left")
        plt.tight_layout()

        if not save_to_folder:
            plt.show()
        else:
            plt.savefig(f'{plot_folder}/{hp}')


### IDEE für Plot: Auf der Env-Achse könnte man nur ein env property verändern



if __name__ == "__main__":

    plot_results('ForIQM', True)

