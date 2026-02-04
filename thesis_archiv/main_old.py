import os

import submitit
import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from thesis_archiv.configs_old import create_experiment
from hydra import initialize, compose
from experiment import execute_arlbench
from metrics import compute_thc
from utils import collecting_results, mapping_id_to_exp
from visualize import plot_results
import numpy as np
from itertools import product
from pathlib import Path
import time
import logging
from absl import logging as absl_logging
import progressbar
import warnings
warnings.filterwarnings("ignore")

### GPU support einbauen

# DONE: Jobs für eine Environment zusammenfassen (Nur im CLuster mode wichtig)  ## TODO Testing
# DONE: While loop for rescheduling failed jobs (Julian's code) ## TODO tesing
# DONE: Local vs cluster ist an sich implementiert ## TODO testing
# DONE: Angucken wie man module laden kann gcc, pyhton 3.10 etc #TODO testing
# DONE: Logging of jobs yes, progressbar not, Progressbar anzeigen lassen für fertige jobs.
# Optimal Hyperparameter representations
# Vrtual env anschauen (uv)
# Logging erstellen
# Fertige Jobs: Benachrichtigung auf Telegram o.Ä.

@hydra.main(version_base=None, config_path="examples/configs", config_name="grid_search")
def run(cfg : DictConfig):

    max_parallel_jobs_cluster = 1000

    # Mute jax logging information
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
    absl_logging.set_verbosity('error')

    if "SLURM_JOB_ID" in os.environ:
        cluster = True
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        cluster = False

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
    ### Logging that files were created ###
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if cluster:
        logger.info(f'Creation of experiment completed: {number_envs} Jobs to be done.')
    else:
        logger.info(f'Creation of experiment completed: {num_runs} Jobs to be done.')

    def arlbench(config_number, seed, logger):

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
            execute_arlbench(cfg, logger)
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
    if cluster:
        # SLURM submission of ARLBench
        executor = submitit.SlurmExecutor(folder="submitit_logs")

        ### Device distinction:
        if device=='cpu':
            executor.update_parameters(
                job_name=experiment_name,
                time='05:00:00',
                cpus_per_task=10,
                nodes=1,
                partition='c23ml',
                array_parallelism=number_envs,
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
                cpus_per_task=10,
                nodes=1,
                gres='gpu:1',
                partition="c23g",
                array_parallelism=number_envs,
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

        tasks = np.arange(number_envs)

        # RWTH Claix specification
        max_parallel_jobs = 1000

        finished_jobs = []
        successfully_finished_jobs = 0
        scheduled_jobs = 0
        running_jobs = []

        while successfully_finished_jobs < number_experiments:

            # Jobs that are done and succesfully finished (no exception) are added to finished jobs
            finished_jobs.extend([job for job in running_jobs if job.done() and job.exception() is None])
            logging.info(f'Number of finished jobs: {len(finished_jobs)}')

            # Workaround for resubmitting failed jobs
            # Get indices of failed jobs
            failed_jobs_indices = [i for i, job in enumerate(running_jobs) if job.done() and job.exception() is not None]

            # Resubmit failed jobs
            for index in failed_jobs_indices:

                job = executor.submit(arlbench_cluster, tasks[index])
                
                # Replace failed and done job with new job submission at the correct index in running jobs
                running_jobs[index] = job

            # Update running jobs list
            running_jobs = [job for job in running_jobs if not job.done()]
            logging.info(f'Currently running and/or submitted jobs: {len(running_jobs)}')

            # In total running jobs on cluster for project
            number_active_jobs = subprocess.run('squeue --array -A thes1998 -h | wc -l', capture_output=True, shell=True, text=True)

            capacity = max_parallel_jobs - int(number_active_jobs.stdout)

            if capacity and scheduled_jobs < number_experiments:

                job = executor.map_array(arlbench_cluster, tasks[scheduled_jobs:scheduled_jobs+capacity])
                number_submitted_jobs = len(job) 
                running_jobs.extend(job)
                scheduled_jobs += number_submitted_jobs

            number_active_jobs = subprocess.run('squeue --array -A thes1998 -h | wc -l', capture_output=True, shell=True, text=True)
            end_episode = int(number_active_jobs.stdout)

            logging.info(f'Number of Jobs on SLURM at the end of loop: {end_episode}')
            logging.info(f'Running jobs after iteration: {len(running_jobs)}')
            logging.info(f'Number of scheduled jobs: {scheduled_jobs}')

            successfully_finished_jobs = len(finished_jobs)

            logging.info(f'Currently succesfully finished jobs: {successfully_finished_jobs}/{number_experiments}')
            time.sleep(5)

        ## Schedule zunächst die ersten 1000 jobs, dann immer so viele sodass 1000 gleichzeitig laufen

        '''
        Old Implementation:
        ### Max 1000 pending/submitted jobs on cluster ###
        jobs = executor.map_array(arlbench_cluster, tasks)

        # Workaround to reschedule failed jobs, Furthermore place for progress bar
        all_done = False
        while not all_done:
            all_done = all(j.done() for j in jobs)

            # Log number of finished jobs --> could be done as progress bar
            currently_finished_job = sum([1 for job in jobs if job.done()])
            logger.setLevel(logging.INFO)
            logger.info(f'Currently finished jobs: {currently_finished_job}/{number_envs}')

            if not all_done:
                time.sleep(20)
            else:
                failed_tasks = [tasks[i] for i, job in enumerate(jobs) if job.exception() is not None]
                if failed_tasks:
                    jobs = list(executor.map(arlbench_cluster, failed_tasks))
                    all_done = False
        '''


        '''
        
            all_done = False
            while not all_done:

                all_done = all(j.done() for j in current_jobs)

                # Log number of finished jobs --> could be done as progress bar
                currently_finished_job = sum([1 for job in current_jobs if job.done()])
                logger.setLevel(logging.INFO)
                logger.info(f'Currently finished jobs: {currently_finished_job}/{number_envs}')

                if not all_done:
                    time.sleep(20)
                else:
                    failed_tasks = [tasks[i] for i, job in enumerate(current_jobs) if job.exception() is not None]
                    if failed_tasks:
                        current_jobs = list(executor.map(arlbench_cluster, failed_tasks))
                        all_done = False
                    else:
                        # Schedule next chunk if available
                        if len(remaining_tasks) > 0:
                            next_chunk = remaining_tasks[:1000]
                            remaining_tasks = remaining_tasks[1000:]
                            current_jobs = executor.map_array(arlbench_cluster, next_chunk)
                            all_done = False    
        
        
        '''

    else:
        # Write code for local execution of pipeline
        logger.setLevel(logging.INFO)
        logger.info(f'Experiment: {experiment_name} starts...')

        b = progressbar.ProgressBar(
            widgets=[
                'Progress: ',
                progressbar.Percentage(),
                ' of Jobs done ',                   
                progressbar.Bar(marker='#'),
                ' ',
                progressbar.Timer(),  
            ], 
            max_value=num_runs
        )
        b.start()

        ### Pass silent logger to ARLBench
        silent_logger = logging.getLogger("silent")
        silent_logger.addHandler(logging.NullHandler())
        silent_logger.propagate = False
        silent_logger.setLevel(logging.CRITICAL + 1)

        for i in range(num_runs):
            config_number, seed = mapping_id_to_exp(i, number_seeds)
            arlbench(config_number, seed, logger=silent_logger)
            b.update(i + 1)

        b.finish()

        logger.info(f'Experiment finished: {num_runs}/{num_runs} Jobs done.')

    # Excel file mit allen performances erstellen
    collecting_results(experiment_name)

    # Computation of THC Scores --> creation of csv file with thc scores per hyperparameter
    compute_thc(experiment_name)

    # Create plots and save to result folder 
    plot_results(experiment_name, True)


if __name__ == '__main__':
    run()
