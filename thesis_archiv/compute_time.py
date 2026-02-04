import os

import time
import submitit
import hydra
from omegaconf import DictConfig
from experiment import execute_arlbench
import logging
from absl import logging as absl_logging
import warnings
import datetime
warnings.filterwarnings("ignore")


#### TODO New Structure for experiments

# Exemplare scheme:
# Env_0_HP_1200


@hydra.main(version_base=None, config_path="examples/configs", config_name="base")
def run(cfg : DictConfig):

    experiment_name = 'compute_time'

    # Mute jax logging information
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
    absl_logging.set_verbosity('error')

    ### Logging that files were created ###
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ### Pass silent logger to ARLBench
    silent_logger = logging.getLogger("silent")
    silent_logger.addHandler(logging.NullHandler())
    silent_logger.propagate = False
    silent_logger.setLevel(logging.CRITICAL + 1)

    def arlbench(parallel_envs, name, logger=silent_logger):

        cfg.environment.n_envs = parallel_envs

        start_time = datetime.datetime.now()

        objective = execute_arlbench(cfg, logger=logger)

        end_time = datetime.datetime.now()
        duration = end_time - start_time

        logger.info(f'Job: {name} required {duration}.')

        return objective

    executor = submitit.SlurmExecutor(folder="submitit")
    executor.update_parameters(
        job_name=experiment_name,
        time='01:00:00',
        cpus_per_task=1,
        account='thes1998',
        nodes=1,
        partition='c23ml',
        setup=[
            'module purge',
            'module load GCCcore/12.2.0',
            'module load Python/3.10.8',
            'source ~/thesis/bin/activate',
        ]
    )

    executor_small = submitit.SlurmExecutor(folder="submitit")
    executor_small.update_parameters(
        job_name=experiment_name,
        time='01:00:00',
        cpus_per_task=1,
        account='thes1998',
        nodes=1,
        partition='c23ms',
        setup=[
            'module purge',
            'module load GCCcore/12.2.0',
            'module load Python/3.10.8',
            'source ~/thesis/bin/activate',
        ]
    )

    ### LARGE TEST FOR EVERYTHING

    #### CPU #### 

    all_jobs = []

    # First: cpu ohne parallel envs as benchmark
    name = 'Partition Large and 1 CPU: 1 Parallel Env'
    all_jobs.append(executor.submit(arlbench, 1, name))


    name = 'Partition Large and 1 CPU: 2 Parallel Envs'
    all_jobs.append(executor.submit(arlbench, 2, name))

    name = 'Partition Large and 1 CPU: 4 Parallel Envs'
    all_jobs.append(executor.submit(arlbench, 4, name))
    
    name = 'Partition Large and 1 CPU: 8 Parallel Envs'
    all_jobs.append(executor.submit(arlbench, 8, name))

    # First: cpu ohne parallel envs as benchmark
    name = 'Partition Small and 1 CPU: 1 Parallel Env'
    all_jobs.append(executor_small.submit(arlbench, 1, name))

    name = 'Partition Small and 1 CPU: 2 Parallel Envs'
    all_jobs.append(executor_small.submit(arlbench, 2, name))

    name = 'Partition Small and 1 CPU: 4 Parallel Envs'
    all_jobs.append(executor_small.submit(arlbench, 4, name))
    
    name = 'Partition Small and 1 CPU: 8 Parallel Envs'
    all_jobs.append(executor_small.submit(arlbench, 8, name))

    all_done = False
    while not all_done:
        all_done = all(j.done() for j in all_jobs)
        if not all_done:
            time.sleep(20)

    


if __name__ == '__main__':
    run()
