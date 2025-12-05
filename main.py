import os

import submitit
import hydra
from omegaconf import DictConfig, OmegaConf
from configs import create_experiment
from hydra import initialize, compose
from experiment import execute_arlbench
from thc import compute_thc
from utils import collecting_results
import numpy as np
from itertools import product
from pathlib import Path
import time
import logging

# TODO s before experiment


### GPU support einbauen

# DONE: Jobs für eine Environment zusammenfassen (Nur im CLuster mode wichtig)  ## TODO Testing
# DONE: While loop for rescheduling failed jobs (Julian's code) ## TODO tesing
# DONE: Local vs cluster ist an sich implementiert ## TODO testing
# DONE: Angucken wie man module laden kann gcc, pyhton 3.10 etc #TODO testing
# DONE: Logging of jobs yes, progressbar not, Progressbar anzeigen lassen für fertige jobs.
# Optimal Hyperparameter representations
# Vrtual Reality anschauen (uv)
# Logging erstellen
# Fertige Jobs: Benachrichtigung auf Telegram o.Ä.

@hydra.main(version_base=None, config_path="examples/configs", config_name="grid_search")
def run(cfg : DictConfig):

    ### Make device selection dependent on config cfg
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
    
    # Create config files: Specified as config_i.yaml with i from 0 to number_experiments - 1
    number_experiments, experiment_name, number_seeds, number_envs, number_hp_configs = create_experiment(cfg)
    num_runs = number_experiments * number_seeds

    def arlbench(config_number, seed):
        # Load config for experiment and creation of result folder
        experiment_path = f'examples/configs/{experiment_name}/config_{config_number}.yaml'
        cfg = OmegaConf.load(experiment_path)
        cfg.autorl.seed = seed
        result_path = str(cfg.hydra.run.dir) + f'/seed_{seed}'
        os.makedirs(result_path, exist_ok=True)

        # Change working directory to result folder
        old = os.getcwd()
        os.chdir(result_path)

        # Execute ARLBench and afterwards set back working dict
        try:
            execute_arlbench(cfg)
        finally:
            os.chdir(old)

    # Arlbench submission on cluster: One job is considered to be all hyperparameter configurations for one environment + all seeds
    def arlbench_cluster(env_number):
        if number_hp_configs:
            for hp_val in range(number_hp_configs):
                for seed in range(number_seeds):
                    config_id = env_number * number_hp_configs + hp_val
                    arlbench(config_id, seed)
        
        else:
            for seed in range(number_seeds):
                config_id = env_number
                arlbench(config_id, seed)

    # Works (On SLurm cluster evaluates to true)
    if "SLURM_JOB_ID" in os.environ:
        # SLURM submission of ARLBench
        executor = submitit.SlurmExecutor(folder="submitit_logs")

        ### Device distinction:
        if device=='cpu':
            executor.update_parameters(
                job_name=experiment_name,
                time=300,
                cpus_per_task=10,
                nodes=1,
                partition="c23ml",
                array_parallelism=number_envs,
                setup=[
                    "module load GCCcore/12.2.0",
                    "module load Python/3.10.8",
                    # venv activate
                ]
            )
        elif device=='gpu':
            executor.update_parameters(
                job_name=experiment_name,
                time=300,
                cpus_per_task=10,
                nodes=1,
                gres='gpu:1',
                partition="c23g",
                array_parallelism=number_envs,
                setup=[
                    'module load GCCcore/12.2.0',
                    'module load Python/3.10.8',
                    'module load CUDA/12.0.0',
                    # venv activate
                ]
            )
        else:
            raise ValueError('No matching hardware found')

        tasks = np.arange(number_envs)
        jobs = executor.map_array(arlbench_cluster, tasks)

        # Workaround to reschedule failed jobs, Furhtermore place for progress bar
        all_done = False
        while not all_done:
            all_done = all(j.done() for j in jobs)

            # Log number of finished jobs --> could be done as progress bar
            currently_finished_job = sum([1 for job in jobs if job.done()])
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.info(f'Currently finished jobs: {currently_finished_job}/{number_envs}')

            if not all_done:
                time.sleep(20)
            else:
                failed_tasks = [tasks[i] for i, job in enumerate(jobs) if job.exception() is not None]
                if failed_tasks:
                    jobs = list(executor.map(arlbench_cluster, failed_tasks))
                    all_done = False

    else:
        # Write code for local execution of pipeline
        currently_finished_jobs = 0
        for config_id in range(number_experiments):
            for seed in range(number_seeds):
                arlbench(config_id, seed)
                logger = logging.getLogger()
                logger.setLevel(logging.INFO)
                currently_finished_jobs += 1
                logger.info(f'Currently finished jobs: {currently_finished_jobs}/{num_runs}')
    
    # Excel file mit allen performances erstellen
    collecting_results(experiment_name)

    # Computation of THC Scores --> creation of csv file with thc scores per hyperparameter
    compute_thc(experiment_name)


if __name__ == '__main__':
    run()
