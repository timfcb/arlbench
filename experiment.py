import sys
import logging
from run_arlbench import run
from omegaconf import DictConfig
import jax
import traceback
import os

def execute_arlbench(cfg: DictConfig):
    """Helper function for nice logging and error handling."""
    logging.basicConfig(
        filename="job.log", format="%(asctime)s %(message)s", filemode="w"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if cfg.jax_enable_x64:
        logger.info("Enabling x64 support for JAX.")
        jax.config.update("jax_enable_x64", True)
    try:
        return run(cfg, logger)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise