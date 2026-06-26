"""Statistical Methods to evaluate the consistency experiment"""
import numpy as np
from utils import get_data
import datetime
from scipy.stats import friedmanchisquare
import math

from rliable import library as rly
from rliable import metrics

# Helper Method for THC score and Top-1 Inconsistency
def compute_rankings(data, hp_values, mode='std'):
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

# Helper method for THC score
def compute_normalized_ptp(rankings):

    ptp_values = []
    ### Workaround for np.ptp
    for hp in rankings:
        ptp_values.append(np.ptp(hp, axis=0))

    ptp_values_normalized = []

    for hp in ptp_values:
        ptp_values_normalized.append([x / (len(hp) - 1) for x in hp])

    return ptp_values_normalized

# Method to compute thc scores per hyperparameter across a set of environments based on the implementation of (https://arxiv.org/pdf/2406.17523)
def compute_thc(experiment_name, hp_values, number_seeds, save_file=True):

    all_rankings = []

    # Get experiment data and compute rankings based on CI intervals
    for ind_hp, hp_param in enumerate(hp_values.keys()):
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param])
        all_rankings.append(rankings)

    # Nomralise ptp-values for each hyperparameter value
    normalized_ptp = compute_normalized_ptp(all_rankings)

    final_thc_per_hp = {}

    # Average normalise ptp values to get thc score for each hyperparameter
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

#Local Parameter importance for each hyperparameter averaged over a set of environments based on the definition in (https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-LION12-CAVE.pdf)
def compute_lpi_score(experiment_name, hp_values, hp_data, env_number, number_seeds):

    ## Get number of env properties
    lpi_per_exp = []
    for env_number in range(env_number):

        variances_per_env = []
        for ind_hp, hp_param in enumerate(hp_values):

            # Get data and measure variances per hp per env
            data = get_data(experiment_name, hp_param, hp_data[ind_hp], number_seeds)
            averaged_data = np.mean(data, axis=2)
            all_values_performance = np.array(averaged_data[env_number])
            variance_per_env_per_hp = np.var(all_values_performance, ddof=1)
            variances_per_env.append(variance_per_env_per_hp)

        # Compute LPI score for one env and one hp
        total_variance = sum(variances_per_env)
        lpi_score_per_env = [x / total_variance for x in variances_per_env]
        #lpi_scores_per_experiment.append(lpi_score_per_env)
        lpi_per_exp.append(lpi_score_per_env)
        lpi_path = f'results/{experiment_name}/lpi_score_Env_{env_number}.csv'
        with open(lpi_path, 'w+') as f:
            f.write('Hyperparameter,LPI score\n')
            for ind_hp,hp_param in enumerate(hp_values):
                f.write(f'{hp_param},{lpi_score_per_env[ind_hp]}\n')

    lpi_path_neu = f'results/{experiment_name}/lpi_score_avg.csv'
    ### Get maximum value for every position
    with open(lpi_path_neu, 'w+') as f:
        f.write('Hyperparameter,LPI score\n')
        for ind_hp, hp_param in enumerate(hp_values):
            max_lpi_score = np.mean([lpi_per_exp[ind_env][ind_hp] for ind_env in range(env_number)])
            f.write(f'{hp_param},{max_lpi_score}\n')

# Executing the Friedman test across all hyperparameters on a set of sepecified environments (https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf)
def rankings_friedman(experiment_name, hp_values, hp_data, number_seeds):

    friedman_per_hp = {}

    for ind_hp, hp in enumerate(hp_values):
        # Friedman test needs at least 3 different hyperparameter values per hyperparameter
        if len(hp_data[ind_hp]) > 2:
            # Get data and perform friedman test across n environments with respect to k hyperparameter values
            data = get_data(experiment_name, hp, hp_data[ind_hp], number_seeds)
            aggregated_data = np.mean(data, axis=2)
            transposed_data = [list(x) for x in list(zip(*aggregated_data))]
            friedman_statistic, p_value = friedmanchisquare(*transposed_data)

            if not np.isnan(p_value):
                p_value = math.floor(p_value*100) / 100
            else:
                p_value = math.nan

            friedman_per_hp[hp] = p_value
        else:
            friedman_per_hp[hp] = math.nan

    friedman_path = f'results/{experiment_name}/friedman_test.csv'
    with open(friedman_path, 'w+') as f:
        f.write('Hyperparameter,Friedman Test Result (p_value)\n')
        for ind_hp,hp_param in enumerate(hp_values):
            f.write(f'{hp_param},{friedman_per_hp[hp_param]}\n')

# Measures the inconsistency at the first ranking position across a set of n rankings
def compute_top1_inconsistency(experiment_name, hp_values, number_seeds):

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):

        number_hp_values = len(hp_values[hp_param])

        ## Aggregated data from shape (number_envs, number_values)
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        # Compute rankings like for THC score
        rankings = compute_rankings(data, hp_values[hp_param])

        ## Compute inconsistency of top-1 ranking position across n environments
        first_rank_counter = [0 for x in range(len(data[0]))]
        number_envs = len(data)
        for ind_rank, rank in enumerate(rankings):
            min_in_ranking = min(rank)
            for ind_val, val in enumerate(rank):
                if val == min_in_ranking:
                    first_rank_counter[ind_val] += 1

        top1 = max([(x / number_envs) for x in first_rank_counter])
 
        # Normalise the top-1 inconsistency: Computation of Normalisation factor can be looked after in thesis
        if number_envs <= number_hp_values:
            normalise_min = (1 / number_envs)
        else:
            normalise_min = (math.ceil(number_envs / number_hp_values) / number_envs)
        
        top1_adjusted = round((top1 - normalise_min) / (1- normalise_min),4)

        top1_consistency_per_hp[hp_param] = top1_adjusted

    top1_path = f'results/{experiment_name}/top_1_inconistency.csv'
    with open(top1_path, 'w+') as f:
        f.write('Hyperparameter,Top1-Consistency\n')
        for ind_hp,hp_param in enumerate(hp_values):
            f.write(f'{hp_param},{top1_consistency_per_hp[hp_param]}\n')
