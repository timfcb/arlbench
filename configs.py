import warnings
warnings.filterwarnings("ignore")

import os
import yaml
from omegaconf import DictConfig, OmegaConf
from itertools import product
from collections import OrderedDict
import numpy as np
from utils import remove_points


def create_experiment(cfg : DictConfig):

    grid_dict = OmegaConf.to_container(cfg, resolve=True)

    experiment_name = grid_dict['experiment_name']

    path = 'examples/configs/' + experiment_name

    ### Create folder for configs
    base_folder = path
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Common Part that is identical in all configs
    base_dict = {}
    base_dict['jax_enable_x64'] = True
    base_dict['load_checkpoint'] = ""

    # Nas configuration
    nas_config = get_nas_config(grid_dict)

    # Get the number of seeds
    seeds = grid_dict['different_seeds']

    create_file(base_dict, path, grid_dict, nas_config, experiment_name)
 
    env_kwargs, env_cardinalities, env_property_names = get_env_properties(grid_dict)

    hp_defaults, hp_values = get_hp_properties(grid_dict)  

    # Create info file in folder showing env properties with values for grid search and hp value ranges 
    info_dict = {}

    info_dict['hp_defaults'] = hp_defaults
    info_dict['hp_values'] = hp_values
    info_dict['env'] = env_kwargs
    info_dict['seeds'] = grid_dict['different_seeds']

    # Save config as yaml
    with open(os.path.join(path, f'info.yaml'), 'w') as f:
        yaml.dump(info_dict, f, sort_keys=False)

    return env_cardinalities, experiment_name, env_property_names, hp_defaults, hp_values, seeds


# Build end config file and save as yaml
def create_file(config, path, grid_dict, nas_config, result_folder):

    # Create autorl config
    autorl_dict = {}
    #autorl_dict['seed'] = grid_dict['seed']
    autorl_dict['env_framework'] = 'mdp'
    autorl_dict['n_envs'] = grid_dict['n_envs']
    autorl_dict['algorithm'] = 'dqn'
    autorl_dict['cnn_policy'] = grid_dict['cnn_policy']
    autorl_dict['nas_config'] = nas_config.copy()
    autorl_dict['n_total_timesteps'] = grid_dict['n_total_timesteps']
    autorl_dict['checkpoint'] = []
    autorl_dict['checkpoint_name'] = "default_checkpoint"
    autorl_dict['checkpoint_dir'] = "/tmp"
    autorl_dict['state_features'] = []
    autorl_dict['objectives'] = ["reward_mean"]
    autorl_dict['optimize_objectives'] = "upper"
    autorl_dict['n_eval_steps'] = grid_dict['n_eval_steps']
    autorl_dict['n_eval_episodes'] = grid_dict['n_eval_episodes']
    config['autorl'] = autorl_dict

    # Hydra Part
    hydra_dict = {}
    run = {}
    sweep = {}
    job = {}
    run['dir'] = 'results/' + result_folder
    sweep['dir'] = 'results/' + result_folder
    job['chdir'] = True
    hydra_dict['run'] = run
    hydra_dict['sweep'] = sweep
    hydra_dict['job'] = job
    config['hydra'] = hydra_dict

    # Save config as yaml
    config_omega = OmegaConf.create(config)

    folder_experiment = path
    with open(os.path.join(folder_experiment, f'config.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_container(config_omega, resolve=True), f)

# Get nas configuration with default values
def get_nas_config(grid_dict):

    nas_config = {}
    nas_dict = grid_dict['nas_config']
    for nas_param, value in nas_dict.items():
        nas_config[nas_param] = value['default']

    return nas_config

# Organise env properties with value ranges for grid search and assign the rest with default values
def get_env_properties(grid_dict):

    env_kwargs = {}
    env_property_names = []
    env_cardinalities = {}
    env_config = grid_dict['env_config']

    for prop, values in env_config.items():
        env_kwargs[prop] = values['values']
        env_cardinalities[prop] = len(values['values'])
        env_property_names.append(prop)

    return env_kwargs, env_cardinalities, env_property_names

# Organise hyperparameter properties with value ranges for grid search and assign the rest with default values
def get_hp_properties(grid_dict):

    hp_defaults = {}
    hp_values = {}

    hp_dict = grid_dict['hp_config']
    hp_range = []
    hp_default = []
    for hp, values in hp_dict.items():
        if len(values['value_range']):
            hp_range.append(hp)
        hp_default.append(hp)

    for hyperparam in hp_default:
        hp_defaults[hyperparam] = hp_dict[hyperparam]['default']

    for hyperparam in hp_range:
        hp_values[hyperparam] = hp_dict[hyperparam]['value_range']

    return hp_defaults, hp_values
