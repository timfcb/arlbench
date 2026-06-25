import os
from pathlib import Path
import argparse
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import numpy as np
from collections import namedtuple
from utils import get_data
import yaml
from collections import defaultdict
import datetime

from rliable import library as rly
from rliable import metrics

# TODO Top-1-Consistency, Tuneability metric anschauen
# Compute thc only makes sense if hp value ranges are given
#TODO performance differencecs in thc miteinbeziehen

# THC-score metric and helper functions

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

def compute_normalized_ptp(rankings):

    ptp_values = []
    ### Workaround for np.ptp
    for hp in rankings:
        ptp_values.append(np.ptp(hp, axis=0))

    ptp_values_normalized = []

    for hp in ptp_values:
        ptp_values_normalized.append([x / (len(hp) - 1) for x in hp])

    return ptp_values_normalized

# Function to compute thc scores per hyperparameter
def compute_thc(experiment_name, hp_values, number_seeds, save_file=True):

    all_rankings = []

    for ind_hp, hp_param in enumerate(hp_values.keys()):
        current_time = datetime.datetime.now()
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param])
        all_rankings.append(rankings)

    normalized_ptp = compute_normalized_ptp(all_rankings)

    final_thc_per_hp = {}

    for ind_hp, hp_param in enumerate(hp_values.keys()):
        final_thc_per_hp[hp_param] = round(sum(normalized_ptp[ind_hp]) / len(normalized_ptp[ind_hp]),4)

    ### Save results in thc csv file
    if save_file:
        thc_path = f'results/{experiment_name}/thc_score_iqm.csv'
        with open(thc_path, 'w+') as f:
            f.write('Hyperparameter,THC score\n')
            for hp_param in final_thc_per_hp:
                f.write(f'{hp_param},{final_thc_per_hp[hp_param]}\n')

    return final_thc_per_hp


def rankings_kendall(experiment_name, hp_values, number_seeds):

    all_kendalls_per_env = []
    for ind_hp, hp_param in enumerate(hp_values.keys()):

        ## Aggregated data from shape (number_envs, number_values)
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param]).tolist()
        kendall_w = kw.compute_w(rankings)
        all_kendalls_per_env.append(kendall_w)

    return all_kendalls_per_env

def top1_consistency_thc_rankings(experiment_name, hp_values, number_seeds):

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):

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

        first_rank_counter = max([round((x / number_envs),4) for x in first_rank_counter])

        top1_consistency_per_hp[hp_param] = first_rank_counter

    return top1_consistency_per_hp


def compute_average_rank_variance():

    experiment_name = 'Grid_search_Terminal_States'

    number_seeds = 10

    with open('configs.yaml', 'r') as f:
        hp_values = yaml.full_load(f)
    #hp_values = list(hp_data.values())

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):

        ## Aggregated data from shape (number_envs, number_values)
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        rankings = compute_rankings(data, hp_values[hp_param])

        print('Rankings')
        print(rankings)

        number_values = len(rankings[0])
        all_variances = []
        for i in range(number_values):
            val = []
            for j in rankings:
                val.append(j[i])
            all_variances.append(np.var(val, ddof=0))

        print(f'Hyperparameter: {hp_param}')
        print(all_variances)


def top1_consistency(experiment_name, hp_values, number_seeds):

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        aggregated_data = np.mean(data, axis=2)

        ## Aggregated data from shape (number_envs, number_values)
        print(f'Aggregated data for {hp_param} in Env {experiment_name}: {aggregated_data}')
        ## For final rankings
        final_rankings = []
        for ind_env, env in enumerate(aggregated_data):
            rankings = np.empty(len(env))
            sort_ind = np.argsort(env)[::-1]
            ranking = 1
            predecessor = max(env)
            for ind, pos in enumerate(sort_ind):
                if env[pos] < predecessor:
                    predecessor = env[pos]
                    ranking += 1
                rankings[pos] = ranking
            final_rankings.append(rankings)
        print(f'Final rankings: {final_rankings}')

        ## Compute consistency
        first_rank_counter = [0 for x in range(len(aggregated_data[0]))]
        number_envs = len(aggregated_data)
        for ind_rank, rank in enumerate(final_rankings):
            for ind_val, val in enumerate(rank):
                if val == 1:
                    first_rank_counter[ind_val] += 1

        print(f'First rank counter: {first_rank_counter}')
        first_rank_counter = max([round((x / number_envs),4) for x in first_rank_counter])
        print(f'First rank counter normalized: {first_rank_counter}')
        top1_consistency_per_hp[hp_param] = first_rank_counter

    top1_path = f'results/{experiment_name}/top1_consistency_50k_iqm.csv'
    with open(top1_path, 'w+') as f:
        f.write('Hyperparameter,Top1-Consistency\n')
        for ind_hp,hp_param in enumerate(hp_values):
            f.write(f'{hp_param},{top1_consistency_per_hp[hp_param]}\n')

    return top1_consistency_per_hp

def compute_lpi_score():

    with open('configs.yaml', 'r') as f:
        hp_data = yaml.full_load(f)
    hp_data = list(hp_data.values())

    env_numbers = [5,6,6,6,9,7,8,11,11,10]

    ### For every experiment independently
    #experiment_name = ['Grid_search_Grid_Size']
    experiment_name = ['Grid_search_Grid_Size', 'Grid_search_Terminal_States', 'Grid_search_Success_Reward', 'Grid_search_Terminal_State_Penalty', 'Grid_search_Reward_Shift', 'Grid_search_Reward_Scaling', 'Grid_search_Transition_Noise', 'Grid_search_Reward_Noise', 'Grid_search_Reward_Delay', 'Grid_search_Reward_Probability']

    hp_values = ['buffer_batch_size', 'buffer_size', 'initial_epsilon', 'target_epsilon', 'exploration_fraction', 'gamma', 'gradient_steps', 'learning_rate', 'learning_starts', 'train_freq', 'target_update_interval', 'tau']

    for ind_exp, experiment in enumerate(experiment_name):

        ## Get number of env properties
        lpi_per_exp = []
        for env_number in range(env_numbers[ind_exp]):

            variances_per_env = []
            for ind_hp, hp_param in enumerate(hp_values):
                print(f'Experiment: {experiment}, Env number: {env_number}, Hyperparameter: {hp_param}')

                data = get_data(experiment, hp_param, hp_data[ind_hp], 10)
                #print(data)
                averaged_data = np.mean(data, axis=2)
                print(f'Averaged data for {hp_param}: {averaged_data}')
                all_values_performance = np.array(averaged_data[env_number])
                print(f'All values performance for {hp_param}: {all_values_performance}')
                variance_per_env_per_hp = np.var(all_values_performance, ddof=1)
                print(f'Variance for {hp_param}: {variance_per_env_per_hp}')
                variances_per_env.append(variance_per_env_per_hp)

            # Compute LPI score for one env and one hp
            total_variance = sum(variances_per_env)
            print(f'Total variance for {experiment}, Env {env_number}: {total_variance}')
            lpi_score_per_env = [x / total_variance for x in variances_per_env]
            print(f'LPI score for Experiment {experiment}, Env {env_number}: {lpi_score_per_env}')
            #lpi_scores_per_experiment.append(lpi_score_per_env)
            lpi_per_exp.append(lpi_score_per_env)
            lpi_path = f'results/{experiment}/lpi_score_Env_{env_number}.csv'
            with open(lpi_path, 'w+') as f:
                f.write('Hyperparameter,LPI score\n')
                for ind_hp,hp_param in enumerate(hp_values):
                    f.write(f'{hp_param},{lpi_score_per_env[ind_hp]}\n')

        lpi_path_neu = f'results/{experiment}/lpi_score_avg.csv'
        ### Get maximum value for every position
        with open(lpi_path_neu, 'w+') as f:
            f.write('Hyperparameter,LPI score\n')
            for ind_hp, hp_param in enumerate(hp_values):
                max_lpi_score = np.mean([lpi_per_exp[ind_env][ind_hp] for ind_env in range(env_numbers[ind_exp])])
                f.write(f'{hp_param},{max_lpi_score}\n')

def rankings_friedman(experiment_name, hp_param, hp_values, number_seeds):

    data = get_data(experiment_name, hp_param, hp_values, number_seeds)
    aggregated_data = np.mean(data, axis=2)

    ## Aggregated data from shape (number_envs, number_values)
    '''
    ## For final rankings
    final_rankings = []
    for ind_env, env in enumerate(aggregated_data):
        rankings = np.empty(len(env))
        sort_ind = np.argsort(env)[::-1]
        ranking = 1
        predecessor = max(env)
        for ind, pos in enumerate(sort_ind):
            if env[pos] < predecessor:
                predecessor = env[pos]
                ranking += 1
            rankings[pos] = ranking
        final_rankings.append(rankings)
    '''
        
    return aggregated_data

def top2_consistency(experiment_name, hp_values, number_seeds):

    top1_consistency_per_hp = {}
    for ind_hp, hp_param in enumerate(hp_values.keys()):
        data = get_data(experiment_name, hp_param, hp_values[hp_param], number_seeds)
        aggregated_data = np.mean(data, axis=2)

        ## Aggregated data from shape (number_envs, number_values)

        ## For final rankings
        final_rankings = []
        for ind_env, env in enumerate(aggregated_data):
            rankings = np.empty(len(env))
            sort_ind = np.argsort(env)[::-1]
            ranking = 1
            predecessor = max(env)
            for ind, pos in enumerate(sort_ind):
                if env[pos] < predecessor:
                    predecessor = env[pos]
                    ranking += 1
                rankings[pos] = ranking
            final_rankings.append(rankings)

        ## Compute consistency
        first_rank_counter = [0 for x in range(len(aggregated_data[0]))]
        number_envs = len(aggregated_data)
        for ind_rank, rank in enumerate(final_rankings):
            for ind_val, val in enumerate(rank):
                if val == 1 or val == 2:
                    first_rank_counter[ind_val] += 1

        first_rank_counter = [round((x / number_envs),4) for x in first_rank_counter]

        top1_consistency_per_hp[hp_param] = first_rank_counter

    return top1_consistency_per_hp


if __name__ == '__main__':

    #experiment_name = 'Grid_Search_Experiment_1'
    #hp_values = {
    #    'learning_rate': [1.0e-05, 0.0001, 0.001, 0, 1]
    #}

    #number_seeds = 5

    #experiment_names = ['Consistency_Experiment_Size', 'Consistency_Experiment_TerminalStates', 'Consistency_Experiment_Transition_Noise', 'Consistency_Experiment_SuccesReward', 'Consistency_Experiment_TerminalStatePenalty', 'Consistency_Experiment_Reward_Shift', 'Consistency_Experiment_Reward_Scaling', 'Consistency_Experiment_Reward_Delay', 'Consistency_Experiment_Reward_Noise', 'Consistency_Experiment_Reward_Probability']

    #with open('configs.yaml', 'r') as f:
    #    hp_data = yaml.full_load(f)
    #hp_data = list(hp_data.values())

    #for exp in experiment_names:
    #    thc_scores = compute_thc(exp, hp_data, 5)

    #consistency = top1_consistency(experiment_names[0], hp_data, 5)

    #print(f'THC Scores for experiment {experiment_name}:')
    #print(thc_scores)
    #experiment_names = ['Grid_search_Reward_Shift']
    #experiment_names = ['Grid_search_Terminal_States', 'Grid_search_Success_Reward', 'Grid_search_Terminal_State_Penalty', 'Grid_search_Reward_Shift', 'Grid_search_Reward_Scaling', 'Grid_search_Transition_Noise', 'Grid_search_Reward_Noise', 'Grid_search_Reward_Delay', 'Grid_search_Reward_Probability']
    
    #with open('configs.yaml', 'r') as f:
    #    hp_data = yaml.full_load(f)

    #for exp in experiment_names:
    #    thc = compute_thc(exp, hp_data,10)
    #    print(f'THC scores for experiment {exp} computed successfully.')

    compute_average_rank_variance()