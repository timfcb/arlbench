import yaml
import pandas as pd
import numpy as np
from rliable import library as rly
from rliable import metrics
import math

def compute_rankings(data, hp_values, mode='iqm'):
    # Two different modes: iqm and std

    #Data is np.array from shape (number_envs, number_values, number_seeds)
    bounds = np.empty((len(data), len(data[0]), 2))

    if mode == 'std':
        # Generate list of tuple(mean, std)
        for ind_env, env in enumerate(data):
            for ind_values, values in enumerate(env):
                standard_metrics = np.mean(values), np.std(values)
                bounds[ind_env][ind_values][0] = round(standard_metrics[0] - standard_metrics[1], 4)
                bounds[ind_env][ind_values][1] = round(standard_metrics[0] + standard_metrics[1], 4)

    elif mode == 'iqm':

        aggregate_function = lambda x: np.array([metrics.aggregate_iqm(x)])
        for ind_env, env in enumerate(data):

            values_dict = dict(zip(hp_values, np.expand_dims(env, axis=2)))
            
            aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                values_dict, aggregate_function, reps=50000
            )

            ci_bounds = [val.flatten().tolist() for val in aggregate_score_cis.values()]
            ci_bounds_rounded = [[round(x, 4) for x in bound_pair] for bound_pair in ci_bounds]

            bounds[ind_env] = ci_bounds_rounded

    else:
        raise ValueError('Invalid mode')

    ### This section gets bounds and computes rankings (same for both statistical methods)
    rankings = np.empty((len(data), len(data[0]))).astype(float)

    for ind_env, env in enumerate(bounds):
        # Get upper bounds of one env:
        lower_bounds, upper_bounds = [x[0] for x in env], [x[1] for x in env]

        r = np.argsort(upper_bounds)[::-1]

        sorted_upper = [upper_bounds[j] for j in r]
        sorted_lower = [lower_bounds[k] for k in r]

        final_rankings = []

        # Iterate over each hyperparameter value
        for j in range(len(sorted_upper)):
            u,l = 0,0

            for i in range(0,j+1):
                if sorted_upper[j] >= sorted_lower[i]:
                    u = i+1
                    break

            for k in range(len(sorted_upper)-1, j-1, -1):
                if sorted_lower[j] <= sorted_upper[k]:
                    l = k+1
                    break
            
            final_rank = (u+l) / 2

            final_rankings.append(final_rank)

        correct_ranking_order = np.zeros(len(final_rankings))

        for ind, pos in enumerate(r):
            correct_ranking_order[pos] = final_rankings[ind]
        rankings[ind_env] = correct_ranking_order   

    return rankings


def get_data(experiment_name, hp_name, hp_values, number_seeds):

    path = f'results/{experiment_name}/{hp_name}.csv'
    df_data = pd.read_csv(path, sep=',')
    number_envs = df_data.shape[0]

    number_values = len(hp_values)

    data = np.zeros((number_envs, number_values, number_seeds))
    for ind_env in range(number_envs):

        for ind_value in range(number_values):

            ind_value_data = ind_value
            access_data_value = ind_value

            for ind_seed in range(number_seeds):

                performance = df_data.iloc[ind_env, (access_data_value * number_seeds + ind_seed)+1]
                data[ind_env][ind_value_data][ind_seed] = performance

    return data


def compute_top1_consistency(experiment_name, hp_values, number_seeds):

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):

        number_hp_values = len(hp_values[hp_param])

        ## Aggregated data from shape (number_envs, number_values)
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param])

        ## Compute consistency
        first_rank_counter = [0 for x in range(len(data[0]))]
        number_envs = len(data)
        for ind_rank, rank in enumerate(rankings):
            min_in_ranking = min(rank)
            for ind_val, val in enumerate(rank):
                if val == min_in_ranking:
                    first_rank_counter[ind_val] += 1

        top1 = max([(x / number_envs) for x in first_rank_counter])
 
        if number_envs <= number_hp_values:
            normalise_min = (1 / number_envs)
        else:
            normalise_min = (math.ceil(number_envs / number_hp_values) / number_envs)
        
        top1_adjusted = round((top1 - normalise_min) / (1- normalise_min),4)

        top1_consistency_per_hp[hp_param] = top1_adjusted

    top1_path = f'results/{experiment_name}/top1_consistency_with_corrected_top1.csv'
    with open(top1_path, 'w+') as f:
        f.write('Hyperparameter,Top1-Consistency\n')
        for ind_hp,hp_param in enumerate(hp_values):
            f.write(f'{hp_param},{top1_consistency_per_hp[hp_param]}\n')


def compute_normalized_ptp(rankings):

    ptp_values = []
    ### Workaround for np.ptp
    for hp in rankings:
        ptp_values.append(np.ptp(hp, axis=0))

    ptp_values_normalized = []

    for hp in ptp_values:
        ptp_values_normalized.append([x / (len(hp) - 1) for x in hp])

    return ptp_values_normalized


def compute_thc(experiment_name, hp_values, number_seeds, save_file=True):

    all_rankings = []

    for ind_hp, hp_param in enumerate(hp_values.keys()):
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param])
        all_rankings.append(rankings)

    normalized_ptp = compute_normalized_ptp(all_rankings)

    final_thc_per_hp = {}

    for ind_hp, hp_param in enumerate(hp_values.keys()):
        final_thc_per_hp[hp_param] = round(sum(normalized_ptp[ind_hp]) / len(normalized_ptp[ind_hp]),4)

    ### Save results in thc csv file
    if save_file:
        thc_path = f'results/{experiment_name}/thc_score.csv'
        with open(thc_path, 'w+') as f:
            f.write('Hyperparameter,THC score\n')
            for hp_param in final_thc_per_hp:
                f.write(f'{hp_param},{final_thc_per_hp[hp_param]}\n')

    return final_thc_per_hp




if __name__ == '__main__':

    experiment_name = 'Grid_search_Grid_Size'

    with open('configs.yaml', 'r') as f:
        hp_data = yaml.full_load(f)

    
    compute_top1_consistency(experiment_name, hp_data, 10)