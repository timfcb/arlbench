import warnings
warnings.filterwarnings("ignore")

import jax
import os
import yaml
import hydra
import datetime
from omegaconf import DictConfig, OmegaConf
from itertools import product
import argparse

@hydra.main(version_base=None, config_path="examples/configs", config_name="grid_search")
def run(cfg : DictConfig):

    grid_dict = OmegaConf.to_container(cfg, resolve=True)

    path = 'examples/configs/' + grid_dict['experiment_name']

    ### Create folder for configs
    base_folder = path
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Common Part that is identical in all configs
    base_dict = {}
    base_dict['jax_enable_x64'] = True
    base_dict['load_checkpoint'] = ""

    ### Environment properties Grid Search
    env_kwargs, properties_grid, grid_search_items = organise_env_properties(grid_dict)

    ### Hyperparameter Grid Search
    hp_dict, hp_config, hp_range, hp_values = organise_hp_properties(grid_dict)

    # Nas configuration
    nas_config = get_nas_config(grid_dict)

    ### Create all combinations of grid search
    counter = 0
    for idx, item in enumerate(grid_search_items):

        # Naming environment and result path creation
        if item == ():
            env_name = 'Env_Default'
        else:
            env_name = 'Env_' + '_'.join(f'{k}_{v}' for k, v in zip(properties_grid, item))
        config = base_dict.copy()
        folder = grid_dict['experiment_name'] + '/' + env_name

        # Assign env properties with values from grid search
        for p_idx, prop in enumerate(properties_grid):
            env_kwargs[prop] = item[p_idx]

        # Case distinction whether hp values are changed over identical environments
        if hp_range:

            # Iterate over value range of one hyperparameter and set others to default
            for ind, hp in enumerate(hp_range):
                for val in hp_values[ind]:
                    hp_config[hp] = val
                    for other_hps in [h for h in hp_range if h != hp]:
                        hp_config[other_hps] = hp_dict[other_hps]['default']

                    result_folder = folder + f'/Hp_{hp}_{val}'
                    counter = create_file(config, hp_config, path, env_name, env_kwargs, grid_dict, nas_config, result_folder, counter)

        else:
            result_folder = folder
            counter = create_file(config, hp_config, path, env_name, env_kwargs, grid_dict, nas_config, result_folder, counter)


# Build end config file and save as yaml
def create_file(config, hp_config, path, env_name, env_kwargs, grid_dict, nas_config, result_folder, counter):

    config['hp_config'] = hp_config.copy()

    # Create autorl config
    autorl_dict = {}
    autorl_dict['seed'] = grid_dict['seed']
    autorl_dict['env_framework'] = 'mdp'
    autorl_dict['env_name'] = env_name
    autorl_dict['env_kwargs'] = env_kwargs.copy()
    autorl_dict['eval_env_kwargs'] = env_kwargs.copy()
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
    run['dir'] = 'results/' + result_folder + '/' + str(grid_dict['seed'])
    sweep['dir'] = 'results/' + result_folder + '/' + str(grid_dict['seed'])
    job['chdir'] = True
    hydra_dict['run'] = run
    hydra_dict['sweep'] = sweep
    hydra_dict['job'] = job
    config['hydra'] = hydra_dict

    # Save config as yaml
    config_omega = OmegaConf.create(config)

    folder_experiment = path
    with open(os.path.join(folder_experiment, f'config_{counter}.yaml'), 'w') as f:
        yaml.dump(OmegaConf.to_container(config_omega, resolve=True), f)

    counter += 1

    return counter

# Get nas configuration with default values
def get_nas_config(grid_dict):

    nas_config = {}
    nas_dict = grid_dict['nas_config']
    for nas_param, value in nas_dict.items():
        nas_config[nas_param] = value['default']

    return nas_config

# Organise env properties with value ranges for grid search and assign the rest with default values
def organise_env_properties(grid_dict):
    env_kwargs = {}
    env_config = grid_dict['env_config']
    properties_grid = []
    properties_default = []
    prop_values = []
    for property, values in env_config.items():
        if not values['value_range']:
            properties_default.append(property)
        else:
            properties_grid.append(property)
            prop_values.append(values['value_range'])

    grid_search_items = list(product(*prop_values))
    
    ## Set the default properties to their default value
    for prop in properties_default:
        env_kwargs[prop] = env_config[prop]['default']

    return env_kwargs, properties_grid, grid_search_items

# Organise hyperparameter properties with value ranges for grid search and assign the rest with default values
def organise_hp_properties(grid_dict):
    hp_config = {}
    hp_dict = grid_dict['hp_config']
    hp_range = []
    hp_default = []
    hp_values = []
    for hp, values in hp_dict.items():
        if not values['value_range']:
            hp_default.append(hp)
        else:
            hp_range.append(hp)
            hp_values.append(values['value_range'])

    for hyperparam in hp_default:
        hp_config[hyperparam] = hp_dict[hyperparam]['default']

    return hp_dict, hp_config, hp_range, hp_values

if __name__ == "__main__":
    run()
