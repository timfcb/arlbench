"""Framework to measure hyperparameter transferability across grid-world environments, Execution inspired from https://arxiv.org/pdf/2406.17523"""
import os

import submitit
import sys
import subprocess
import hydra
import copy
from omegaconf import DictConfig, OmegaConf
from configs import create_experiment
from experiment import execute_arlbench
from hydra import initialize, compose
from metrics import compute_thc, compute_top1_inconsistency, compute_lpi_score, rankings_friedman
from utils import results_to_csv
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import time
import logging
from absl import logging as absl_logging
import progressbar
import warnings
warnings.filterwarnings("ignore")

# Entry Point to the transferability experiment
# To modify the investigated environment properties and hyperparameters modify file: examples/configs/grid_search.yaml

@hydra.main(version_base=None, config_path="examples/configs", config_name="grid_search")
def run(cfg : DictConfig):

    # Mute jax logging information
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
    absl_logging.set_verbosity('error')

    # Check whether system is in SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        cluster = True
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        cluster = False

    # Device selection based on state representation
    state_repr = cfg['state_representation']
    device_config = cfg['device']
    if state_repr == 'vector':
        device = 'cpu'
    elif state_repr == 'image':
        device = 'gpu'
    # Matrix can be done on both nn architectures
    elif state_repr == 'matrix' and device_config == 'cpu':
        device = 'cpu'
    else:
        device = 'gpu'

    # Change to root directory of the project (hydra main creates hydra folder in result: hence for experiment we have to switch back)
    current_path = os.getcwd()
    root_folder = os.path.abspath(os.path.join(current_path, '..', '..'))
    os.chdir(root_folder)
     
    # Initializes scope of experiment
    env_cardinalities, experiment_name, env_property_names, hp_defaults, hp_values, different_seeds = create_experiment(cfg)
    list_number_props = list(env_cardinalities.values())

    # All config is list of list indicating the i-th property of every env property
    all_configs = list(product(*[range(length) for length in list_number_props]))
    all_config_ids = list(range(len(all_configs)))

    ### Logging that files were created ###
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info(f'Experiment: {experiment_name} starts...')

    # Create result folder here:
    result_folder = f'results/{experiment_name}'
    os.makedirs(result_folder, exist_ok=True)

    # RWTH Claix specification (800 parallel running jobs on CLAIX)
    max_parallel_jobs = 800

    ### Pass silent logger to ARLBench
    silent_logger = logging.getLogger("silent")
    silent_logger.addHandler(logging.NullHandler())
    silent_logger.propagate = False
    silent_logger.setLevel(logging.CRITICAL + 1)

    # Reward function parameters for evaluation: Identical on all environments (Identical Reward function for all runs in the entire experiment)
    eval_kwargs = {
        'success_reward': 1.0,
        'terminal_state_penalty': 0.0,
        'reward_shift': 0.0,
        'reward_noise': 0.0,
        'reward_scaling_factor': 0.0,
        'reward_probability': 1.0,
        'reward_delay_prob': 0.0,
    }

    # Code of a single job on the cluster
    def arlbench(config, config_id, folder=result_folder, logger=logger):

        # Check if experiment already done -> If yes: Job was already executed
        possible_file_name = f'{folder}/config_{config_id}.npz'
        if os.path.exists(possible_file_name):
            return None

        # Creating env_kwargs dict for base_config
        env_kwargs = {}
        # Create eval_kwargs dict:
        for i, env_property in enumerate(env_property_names):
            prop_value = cfg['env_config'][env_property]['values'][config[i]]
            env_kwargs[env_property] = prop_value

        ### Assign eval env with every env kwarg that does not belong to reward structure:
        eval_kwargs['grid_shape'] = env_kwargs['grid_shape']
        eval_kwargs['transition_noise'] = env_kwargs['transition_noise']
        eval_kwargs['number_terminal_states'] = env_kwargs['number_terminal_states']

        ### This loads base yaml file for experiment
        experiment_path = f'examples/configs/{experiment_name}/config.yaml'
        cfg_exp = OmegaConf.load(experiment_path)

        ## Merge env properties into cfg
        cfg_exp.autorl.env_kwargs = env_kwargs

        # Evaluation environment consist only of default assignemnts in the reward properties --> Comparability of policy evaluation
        cfg_exp.autorl.eval_env_kwargs = eval_kwargs

        # Set env name for this experiment
        cfg_exp.autorl.env_name = str(config_id)

        objectives = []

        # Running the local hyperparameter sweeps in the given environment for each hyperparameter
        for hp, value_range in hp_values.items():

            hp_results = []
            for value in value_range:  

                hp_dict = copy.deepcopy(hp_defaults)

                hp_dict[hp] = value
                cfg_exp.hp_config = hp_dict

                seed_results = []

                # Performing each run for the number of specified seeds
                for seed in different_seeds:
                    # Set seed in cfg
                    cfg_exp.autorl.seed = seed

                    # Objective is the single performance value obtained for a single run
                    objective = round(execute_arlbench(cfg_exp, logger=logger),4)

                    seed_results.append(objective)

                hp_results.append(seed_results)

            hp_exp = np.array(hp_results)
            # objectives is a list of arrays (arrays do not necessarily have the same shape)
            objectives.append(hp_exp)

        # Saving the results of the local hyperparameter sweeps for each hyperparameter measured in environment of id i in file npz file config_i.npz
        with open(possible_file_name, 'wb') as f:
            np.savez(f, *objectives)

        return True
    
    # Works (On SLurm cluster evaluates to true)
    if cluster:
        # SLURM submission of ARLBench
        executor = submitit.SlurmExecutor(folder="submitit_logs")

        ### Device distinction:
        if device=='cpu':
            executor.update_parameters(
                job_name=experiment_name,
                time='24:00:00',
                cpus_per_task=24,
                account='thes1998',
                nodes=1,
                partition='c23ms',
                array_parallelism=max_parallel_jobs,
                setup=[
                    'module purge',
                    'module load GCCcore/12.2.0',
                    'module load Python/3.10.8',
                    'source ~/thesis/bin/activate',
                ]
            )
        elif device=='gpu':
            executor.update_parameters(
                job_name=experiment_name,
                time='10:00:00',
                cpus_per_task=1,
                account='thes1998',
                nodes=1,
                gres='gpu:1',
                partition="c23g",
                array_parallelism=max_parallel_jobs,
                setup=[
                    'module purge',
                    'module load GCCcore/12.2.0',
                    'module load Python/3.10.8',
                    'module load CUDA/12.3.0',
                    'module load cuDNN/8.9',
                    'source ~/thesis/bin/activate',
                ]
            )
        else:
            raise ValueError('No matching hardware found')

        number_jobs = len(all_configs)

        scheduled_jobs = 0
        running_jobs = []

        # Schedules jobs until all jobs of this experiment are executed
        while scheduled_jobs < number_jobs:

            # In total running jobs on cluster for project
            number_active_jobs = subprocess.run('squeue --array -A thes1998 -h | wc -l', capture_output=True, shell=True, text=True)

            capacity = max_parallel_jobs - int(number_active_jobs.stdout)

            # Checks if clusters capacity and in case schedules new jobs
            if capacity and scheduled_jobs < number_jobs:

                job = executor.map_array(arlbench, all_configs[scheduled_jobs:scheduled_jobs+capacity], all_config_ids[scheduled_jobs:scheduled_jobs+capacity])
                number_submitted_jobs = len(job)
                running_jobs.extend(job)
                scheduled_jobs += number_submitted_jobs

            # Tracks the number of currently running jobs connected to the account
            number_active_jobs = subprocess.run('squeue --array -A thes1998 -h | wc -l', capture_output=True, shell=True, text=True)
            end_episode = int(number_active_jobs.stdout)

            logging.info(f'Current number of running jobs on SLURM: {end_episode}')

            time.sleep(120)

    # Only for testing pipeline local
    else:

        b = progressbar.ProgressBar(
            widgets=[
                'Progress: ',
                progressbar.Percentage(),
                ' of Jobs done ',                   
                progressbar.Bar(marker='#'),
                ' ',
                progressbar.Timer(),
            ], 
            max_value=len(all_config_ids)
        )
        b.start()


        for i in range(len(all_config_ids)):
            objective = arlbench(all_configs[i], all_config_ids[i])
            b.update(i + 1)

        b.finish()

    with open(f'{result_folder}/done.txt', 'w') as f:
        f.write(f'Experiment {experiment_name} done.')

    # Results from the local hyperparameter sweeps in each environment are written into a single file independently for each hyperparameter
    results_to_csv(hp_values, different_seeds, result_folder, all_config_ids)

    ### Applying statistical methods to investigate experimental results ###

    # Computing THC metric with respect to the environments investigated in this experiment
    compute_thc(experiment_name, hp_values, len(different_seeds))

    # Compute Top_1 Inconsistency for each hyperparameter with respect to
    compute_top1_inconsistency(experiment_name, hp_values, len(different_seeds))

    # Compute Local Parameter Importance for each hyperparameter in each of the investigated environments
    compute_lpi_score(experiment_name, hp_values.keys(), list(hp_values.values()), len(all_configs),len(different_seeds))

    # Executing the Friedman Test 
    rankings_friedman(experiment_name, hp_values.keys(),list(hp_values.values()),len(different_seeds))
    
    logger.info(f'Experiment finished: {len(all_config_ids)}/{len(all_config_ids)} Jobs done.')


# Entry point to the transferability experiments
if __name__ == '__main__':
    run()
