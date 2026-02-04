import sys
import logging
from arlbench.arlbench import run_arlbench
from omegaconf import DictConfig
import jax
import traceback
import os
import logging

def execute_arlbench(cfg: DictConfig, logger: logging.Logger):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )

    if cfg.jax_enable_x64:
        #logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger=logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

def run(cfg: DictConfig, logger: logging.Logger):
    """Console script for arlbench."""
    objectives = run_arlbench(cfg, logger=logger)
    logger.info(f"Returned objectives: {objectives}")

    ### INFO: No creation of data files in our experiments

    #with open("./performance.csv", "w+") as f:
    #    f.write(str(objectives))
    #with open("./done.txt", "w+") as f:
    #    f.write("yes")

    return objectives