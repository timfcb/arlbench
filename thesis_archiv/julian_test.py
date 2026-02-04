import numpy as np
import logging
import sys
import time
import os
import json
import argparse

import hydra
import submitit

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from dehb_gym_lunar.ppo_hydra import hydra_launcher


def get_logger(log_file):
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    fh = logging.FileHandler(log_file, "w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(ch)
    log.addHandler(fh)
    return log


def search(log, parameter_class, parameter_name, param_range):
    executor = submitit.SlurmExecutor(folder="submitit_logs")
    executor.update_parameters(
        job_name=parameter_name,
        time=15,
        cpus_per_task=1,
        nodes=1,
        mem_per_cpu=2000,
        partition="Kathleen",
        signal_delay_s=120,
        array_parallelism=1500,
    )

    def run_training(parameter, seed):
        overrides = [
            f"algorithm.{parameter_class}.{parameter_name}={parameter}",
            f"seed={seed}",
            "algorithm.eval=False",
        ]
        log = logging.getLogger(__name__)

        with hydra.initialize(config_path="../config_lunar"):
            cfg = hydra.compose(
                config_name="local_ppo_hpo",
                overrides=overrides,
            )
            mean_performance = -hydra_launcher(cfg, log)
            result_dict = {
                parameter_name: parameter,
                "mean_performance": float(mean_performance),
            }
            result_dir = os.path.join(f"ppo_{parameter_name}", "results", f"{parameter_name}_{parameter}")
            os.makedirs(result_dir, exist_ok=True)
            json.dump(result_dict, open(os.path.join(result_dir, f"{seed}"), "w"))

    if parameter_name in ["learning_rate", "gamma"]:
        parameter_values = np.logspace(np.log10(param_range[0]), np.log10(param_range[1]), param_range[2])
    else:
        parameter_values = np.linspace(param_range[0], param_range[1], param_range[2])

    tasks = []
    for param_value in parameter_values:
        for seed in range(10):
            result_dir = os.path.join(
                f"ppo_{parameter_name}",
                "results",
                f"{parameter_name}_{param_value}",
            )
            if os.path.exists(os.path.join(result_dir, f"{seed}")):
                try:
                    result_dict = json.load(open(os.path.join(result_dir, f"{seed}")))
                    if "mean_performance" in result_dict:
                        continue
                except json.decoder.JSONDecodeError:
                    pass
            tasks.append((param_value, seed))
    jobs = executor.map_array(run_training, *zip(*tasks))

    all_done = False
    while not all_done:
        all_done = all(j.done() for j in jobs)
        if not all_done:
            time.sleep(20)
        else:
            failed_tasks = [tasks[i] for i, job in enumerate(jobs) if job.exception() is not None]
            if failed_tasks:
                log.info(f"Resubmitting {len(failed_tasks)} failed jobs as a new array job.")
                jobs = list(executor.map(run_training, *zip(*failed_tasks)))
                all_done = False


def collect_data(log, parameter_name):
    if os.path.exists(os.path.join(f"ppo_{parameter_name}", "result.csv")):
        log.warning("The result.csv file already exists. Skipping collection. Delete file to collect it again.")
        return
    parameter_list = []
    parameter_performance = {}
    for parameter in os.listdir(os.path.join(f"ppo_{parameter_name}", "results")):
        for seed in os.listdir(os.path.join(f"ppo_{parameter_name}", "results", parameter)):
            result_dict = json.load(open(os.path.join(f"ppo_{parameter_name}", "results", parameter, seed)))
            parameter_value = result_dict[parameter_name]
            if parameter_value not in parameter_list:
                parameter_list.append(parameter_value)

            if parameter not in parameter_performance:
                parameter_performance[parameter] = []
            parameter_performance[parameter].append(-result_dict["mean_performance"])
    df_dict = {
        parameter_name: parameter_list, "final_mean_return": [],
    }
    for parameter in parameter_performance:
        df_dict["final_mean_return"].append(parameter_performance[parameter])
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(f"ppo_{parameter_name}", f"result.csv"), index=False)


def plot_data(parameter_name, default_value):
    def convert_to_float_list(string):
        float_list = string.strip("[]").split(",")
        return [float(x) for x in float_list if x]
    df = pd.read_csv(
        os.path.join(f"ppo_{parameter_name}", f"result.csv"),
        converters={
            "final_mean_return": convert_to_float_list,
            "final_std_return": convert_to_float_list,
        },
    )

    df["mean"] = df["final_mean_return"].apply(np.mean)
    df["std"] = df["final_mean_return"].apply(np.std)
    plt.errorbar(df[parameter_name], df["mean"], yerr=df["std"], fmt="o")
    if parameter_name == "gamma":
        plt.xscale("log")
        plt.gca().invert_xaxis()
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{1 - x}'))
    elif parameter_name == "learning_rate":
        plt.xscale("log")
    plt.title(f"PPO {parameter_name} Grid Search")
    plt.xlabel(parameter_name)
    plt.ylabel("Mean External Objective")
    if parameter_name == "gamma":
        plt.axvline(1 - default_value, color="red", linestyle="--")
    else:
        plt.axvline(default_value, color="red", linestyle="--")

    os.makedirs(f"ppo_{parameter_name}", exist_ok=True)
    plt.savefig(os.path.join(f"ppo_{parameter_name}", f"performance.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--param-name", type=str, required=True)
    parser.add_argument("--param-class", type=str, required=True)
    parser.add_argument("--param-range", type=str, help="low,high,step")
    parser.add_argument("--only-plotting", action="store_true")
    parser.add_argument("--default-value", type=float, required=False, default=None)

    args = parser.parse_args()

    os.makedirs(os.path.join(f"ppo_{args.param_name}"), exist_ok=True)
    log = get_logger(os.path.join(os.path.join(f"ppo_{args.param_name}", "grid_search.log")))
    log.info(f"Running grid search for {args.param_name} in {args.param_class} with range {args.param_range}")


    if not args.only_plotting:
        low, high, steps = args.param_range.split(",")
        param_range = (float(low), float(high), int(steps))
        search(log, args.param_class, args.param_name, param_range)
        log.info("done searching")
        collect_data(log, args.param_name)
        log.info("done collecting data")
    plot_data(
        args.param_name,
        args.default_value,
    )
    log.info("done plotting data")