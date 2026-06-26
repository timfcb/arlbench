'''Class with helper methods to organise the experimental data'''
import csv
import re
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
from itertools import product
import xlsxwriter


def results_to_csv(hp_values, different_seeds, result_folder, all_config_ids): 
    ### Save values in CSV for each hyperparameter
    for hp_index, hp in enumerate(hp_values.keys()):

        with open(f'{result_folder}/{hp}.csv', 'w') as f:

            # First Row Specifications
            first_row = 'ENV_ID,'
            hp_value_range = hp_values[hp]
            for ind_val, val in enumerate(hp_value_range):
                for seed in different_seeds:
                    first_row += f'Value_{val}_Seed_{seed},'

            first_row = first_row[:-1] + '\n'
            f.write(first_row)

            for config_id in all_config_ids:
                data = np.load(f'{result_folder}/config_{config_id}.npz')

                hp_data = data[f'arr_{hp_index}']

                row_entry = f'{config_id},'
                for val in hp_data.flatten():
                    row_entry += f'{val},'
                entry = row_entry[:-1] + '\n'
                f.write(entry)


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
