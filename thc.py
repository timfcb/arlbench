import os
from pathlib import Path
import argparse
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import numpy as np
from collections import namedtuple
from utils import get_data

from rliable import library as rly
from rliable import metrics


### File to compute THC scores per hyperparameter
# TODO Top-1-Consistency, Tuneability metric anschauen

# Compute thc only makes sense if hp value ranges are given

#TODO performance differnecs in thc miteinbeziehen

def compute_rankings(data, mode='std'):
    # Two different modes: iqm and std

    ### Data is np.array from shape (number_envs, number_values, number_seeds)

    if mode == 'std':
        ### Algorithm is taken for
        ### Generate list of tuple(mean, std)
        bounds = np.empty((len(data), len(data[0]), 2))
        for ind_env, env in enumerate(data):
            for ind_values, values in enumerate(env):
                metrics = np.mean(values), np.std(values)
                bounds[ind_env][ind_values][0] = metrics[0] + metrics[1]
                bounds[ind_env][ind_values][1] = metrics[0] - metrics[1]    
        ### Sortiere für jede environment die values und speichere nur das Ranking im array
        rankings = np.empty((len(data), len(data[0]))).astype(float)

        for ind_env, env in enumerate(bounds):
            ## Get upper bounds of one env:
            upper_bounds, lower_bounds = [x[0] for x in env], [x[1] for x in env]

            r = np.argsort(upper_bounds)[::-1]

            sorted_upper = [upper_bounds[j] for j in r]
            sorted_lower = [lower_bounds[k] for k in r]

            final_rankings = []

            ### Iterate over each hyperparameter value
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

    elif mode == 'iqm':
        # TODO

        ### Rliable nutzen
        pass

    # TODO other options

    else:
        raise ValueError('Invalid mode')

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

def compute_thc(folder):

    # Experiment folder
    result_dir = Path('results/' + folder)
    exp_dir = Path('examples/configs/' + folder + '/info.yaml')
    
    cfg = OmegaConf.load(exp_dir)

    # Convert to Python dict
    info_dict = OmegaConf.to_container(cfg, resolve=True)

    hp_list = list(info_dict['hp'].keys())

    all_rankings = []
    for ind_hp, hp_param in enumerate(hp_list):
        data = get_data(hp_param, result_dir)
        rankings = compute_rankings(data)
        all_rankings.append(rankings)

    normalized_ptp = compute_normalized_ptp(all_rankings)

    final_thc_per_hp = {}

    for ind_hp, hp_param in enumerate(hp_list):
        final_thc_per_hp[hp_param] = sum(normalized_ptp[ind_hp]) / len(normalized_ptp[ind_hp])

    ### Save results in thc csv file
    thc_path = f'results/{folder}/thc_per_hp.csv'
    with open(thc_path, 'w+') as f:
        f.write('Hyperparameter,THC score\n')
        for hp_param in final_thc_per_hp:
            f.write(f'{hp_param},{final_thc_per_hp[hp_param]}\n')

    return final_thc_per_hp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str)

    args = parser.parse_args()

    # THC computation works well
    # TODO add computation of iqm with confidence intervals as metric to compute rankings (In Paper: both options, here: only mean and std)
    compute_thc(args.folder)
