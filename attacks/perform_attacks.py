import argparse
import concurrent.futures
import copy
import functools
import heapq
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Optional

import load_experiments
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from classifier_attacker import (
    AttackerDatasetMode,
    Mode,
    SimpleAttacker,
    run_classifier_attack,
)
from LinkabilityAttack import LinkabilityAttack
from load_experiments import (
    ALL_ATTACKS,
    POSSIBLE_DATASETS,
    POSSIBLE_LOSSES,
    POSSIBLE_MODELS,
    error_catching_wrapper,
)
from RocPlotter import RocPlotter
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from threshold_attacker import run_threshold_attack

from decentralizepy.datasets.CIFAR10 import LeNet

DEBUG = False


def run_linkability_attack(
    running_model,
    model_path,
    expected_agent,
    attack_object: LinkabilityAttack,
    shapes,
    lens,
    device,
):
    load_experiments.load_model_from_path(
        model_path=model_path,
        model=running_model,
        shapes=shapes,
        lens=lens,
        device=device,
    )

    res = attack_object.log_all_losses(running_model)
    all_losses_list = [
        (int(agent.split("_")[2]), loss_agent) for agent, loss_agent in res.items()
    ]
    all_losses_list.sort(key=lambda x: x[1])
    top_5 = [all_losses_list[i][0] for i in range(5)]

    res["linkability_top1"] = expected_agent == top_5[0]
    res["linkability_top1_guess"] = top_5[0]
    res["linkability_top5"] = expected_agent in top_5

    for index, (agent, _) in enumerate(all_losses_list):
        if expected_agent == agent:
            res["linkability_real_rank"] = index
            break
    if "linkability_real_rank" not in res:
        print(
            f"ERROR: no linkability real rank found for {model_path} when it should have."
        )
        res["linkability_real_rank"] = np.nan
    return pd.DataFrame(res, index=[0])


@error_catching_wrapper
def attack_experiment(
    experiment_df,
    experiment_name: str,
    batch_size: int,
    experiment_path: str,
    datasets_dir: str,
    device_type: str = "cpu",
    attack_todo="threshold",
    debug=False,
):
    # Load this experiment's configuration file
    config_file = os.path.join(experiment_path, "config.ini")
    assert os.path.exists(config_file), f"Cannot find config file at {config_file}"
    config = load_experiments.read_ini(config_file)

    g5k_config_file = os.path.join(experiment_path, "g5k_config.json")
    assert os.path.exists(
        g5k_config_file
    ), f"Cannot find g5k config file at {g5k_config_file}"
    with open(g5k_config_file, "r") as f:
        g5k_config = json.load(f)

    nb_agents = g5k_config["NB_AGENTS"]

    dataset = config.dataset.dataset_class
    nb_classes = POSSIBLE_DATASETS[dataset][1]

    seed = load_experiments.safe_load(config, "DATASET", "random_seed", int)
    if dataset in ["Femnist", "FemnistLabelSplit", "MovieLens"]:
        kshards = None

    else:
        kshards = load_experiments.safe_load(config, "DATASET", "shards", int)

    model_name = config.dataset.model_class
    target_model_initializer = POSSIBLE_MODELS[model_name]

    simulation_loss_name = config.train_params.loss_class
    simulation_loss_initializer = POSSIBLE_LOSSES[simulation_loss_name]
    simulation_loss_function = simulation_loss_initializer(reduction="none")

    assert attack_todo in [
        "threshold",
        "linkability",
        "biasedthreshold",
        "threshold+biasedthreshold",  # Those two attacks can be combined to save time and energy.
        "classifier",
    ]
    # Dirty way to handle multiple attacks at the same time, but reduces evaluation time.
    if attack_todo == "threshold+biasedthreshold":
        results_files = {}
        result_file_threshold = os.path.join(
            experiment_path, f"threshold_{experiment_name}.csv"
        )
        if os.path.exists(result_file_threshold):
            print(
                f"threshold-attack already computed for {experiment_name}, not computing"
            )
            attack_todo = "biasedthreshold"
        else:
            results_files["threshold"] = result_file_threshold
        result_file_biasedthreshold = os.path.join(
            experiment_path, f"biasedthreshold_{experiment_name}.csv"
        )
        if os.path.exists(result_file_threshold):
            print(
                f"threshold-attack already computed for {experiment_name}, not computing"
            )
            if attack_todo == "biasedthreshold":
                print(f"Both attacks already computed for {experiment_name}, skipping")
                return
            attack_todo = "threshold"
        else:
            results_files["biasedthreshold"] = result_file_biasedthreshold

    starting_results = pd.DataFrame({})
    if attack_todo in ["linkability", "threshold", "biasedthreshold", "classifier"]:
        results_files = {}
        results_file = os.path.join(
            experiment_path, f"{attack_todo}_{experiment_name}.csv"
        )
        if os.path.exists(results_file):
            if (
                attack_todo == "classifier"
            ):  # Only case where checkpointing is designed.
                starting_results = pd.read_csv(results_file)
            else:
                print(
                    f"{attack_todo}-attack already computed for {experiment_name}, skipping"
                )
                return
        results_files[attack_todo] = results_file

    device = torch.device(device_type)
    print(f"Launching {attack_todo}-attack on {experiment_name}.")
    t1 = time.time()

    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        dataset,
        datasets_dir=datasets_dir,
        total_agents=nb_agents,
        seed=seed,
        config=config,
        debug=DEBUG,
    )
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    current_experiment = experiment_df[
        experiment_df["experiment_name"] == experiment_name
    ]

    running_model = target_model_initializer()
    shapes, lens = load_experiments.generate_shapes(running_model)

    attack_object = None
    # This is a trick to save time: only have one attack instance for all the experiment
    # instead of creating a new one for each model in the experiment.
    # Avoids loading repeatedly the dataset.
    if attack_todo == "linkability":
        attack_object = LinkabilityAttack(
            num_clients=nb_agents,
            client_datasets=trainset_partitioner,
            # For linkability attack, we only look at aggregated loss, so we want reduction.
            # Hence, no 'reduction="none" parameter.'
            loss=simulation_loss_initializer(),
            eval_batch_size=batch_size,
            device=device,
        )

    total_result = {}
    if attack_todo == "classifier":
        # TODO: change this to a parameter?
        # We discovered there is not much of a difference between 0.25 and 0.7
        # So I'll just consider one of them for now, to divide by 3 the attack time.
        # fractions = [0.25, 0.5, 0.7]
        fractions = [0.7]  # Default value to reduce number of trainings.
        # fractions = [0.01, 0.1, 0.25, 0.5]
        # fractions = [0.005] # Edge case
        fractions = [0.7, 0.25]

        attacker_information: list[AttackerDatasetMode] = [
            "global",
            "local",
            "local_train",
        ]
        # attacker_information: list[AttackerDatasetMode] = ["local_train"]

        attack_modes: list[Mode] = ["all", "last", "first", "middle"]
        # attack_modes: list[Mode] = ["first"]
        total_result["classifier"] = run_classifier_attack(
            models_properties=current_experiment,
            experiment_dir=experiment_path,
            trainset_partitioner=trainset_partitioner,
            testset=testset,
            simulation_loss_function=simulation_loss_function,
            attacked_model=running_model,
            shapes_target=shapes,
            lens_target=lens,
            device=device,
            batch_size=batch_size,
            attacker_model_initializer=SimpleAttacker,
            debug=debug,
            fractions=fractions,
            attack_modes=attack_modes,
            attacker_dataset_mode=attacker_information,
            starting_results=starting_results,
        )
        save_results(
            results_files=results_files,
            total_result=total_result,
            experiment_name=experiment_name,
        )
        t2 = time.time()
        print(
            f"Finished {attack_todo} attack on {experiment_name} in {(t2-t1)/3600:.2f}hours"
        )
        return

    agent_list = sorted(pd.unique(current_experiment["agent"]))
    for agent in agent_list:
        roc_plotter_classes: list[RocPlotter] = []
        if agent == 0:
            roc_plotter_unbalanced = RocPlotter(
                os.path.join(experiment_path, f"unbalanced_roc_loss_distrib.pdf"),
                nb_columns=5,
            )
            roc_plotter_balanced = RocPlotter(
                os.path.join(experiment_path, f"balanced_roc_loss_distrib.pdf"),
                nb_columns=5,
            )
            for i in range(nb_classes):
                roc_plotter_classes.append(
                    RocPlotter(
                        os.path.join(
                            experiment_path, f"unbalanced_roc_loss_distrib_class{i}.pdf"
                        ),
                        nb_columns=5,
                    )
                )
        else:
            roc_plotter_unbalanced = None
            roc_plotter_balanced = None
        current_agent_experiments = current_experiment[
            current_experiment["agent"] == agent
        ]
        trainset = trainset_partitioner.use(agent)
        trainset = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False
        )

        rows = [row for _, row in current_agent_experiments.iterrows()]
        rows = sorted(rows, key=lambda row: int(row["iteration"]))

        for row in rows:
            results = {}
            iteration = row["iteration"]
            model_path = row["file"]
            target = row["target"]
            tcurrent = time.time()
            if debug:
                endline = "\n"
            else:
                endline = "\r"
            print(
                f"Launching {attack_todo} attack {agent}->{target} at iteration {iteration}, "
                + f"{(tcurrent-t1)/60:.2f} minutes taken to reach this point"
                + " " * 10,
                end=endline,
            )
            if attack_todo == "linkability":
                assert attack_object is not None
                attack_result = run_linkability_attack(
                    running_model=running_model,
                    model_path=model_path,
                    expected_agent=agent,  # We want to guess the CURRENT agent!
                    attack_object=attack_object,
                    shapes=shapes,
                    lens=lens,
                    device=device,
                )
                results[attack_todo] = attack_result

            else:
                # TODO: remove this and simply give iteration as a parameter.
                if roc_plotter_unbalanced is not None:
                    roc_plotter_unbalanced.set_next_plot(
                        iteration=iteration,
                        log_next=iteration in [1, 100, 1000, 2500, 4000, 5000],
                    )
                if roc_plotter_balanced is not None:
                    roc_plotter_balanced.set_next_plot(
                        iteration=iteration,
                        log_next=iteration in [1, 100, 1000, 2500, 4000, 5000],
                    )
                for plotter in roc_plotter_classes:
                    plotter.set_next_plot(
                        iteration=iteration,
                        log_next=iteration in [1, 100, 1000, 2500, 4000, 5000],
                    )
                threshold_results = run_threshold_attack(
                    running_model,
                    model_path,
                    trainset,
                    testset,
                    shapes,
                    lens,
                    device=device,
                    debug=debug,
                    debug_name=experiment_name,
                    loss_training=simulation_loss_function,
                    attack=attack_todo,
                    plotter_unbalanced=roc_plotter_unbalanced,
                    plotter_balanced=roc_plotter_balanced,
                    classes_plotters=roc_plotter_classes,
                )
                # Only save the necessary results.
                for attack_to_save in results_files:
                    results[attack_to_save] = threshold_results[attack_to_save]

            for attack, result in results.items():
                result["iteration"] = iteration
                result["agent"] = agent
                result["target"] = target
                results[attack] = result
                if debug:
                    print(f"it:{iteration} -  {agent}->{target}. {attack}:\n{result}\n")
                if attack not in total_result:
                    total_result[attack] = pd.DataFrame({})
                total_result[attack] = pd.concat(
                    [total_result[attack], results[attack]]
                )

    save_results(
        results_files=results_files,
        total_result=total_result,
        experiment_name=experiment_name,
    )
    t2 = time.time()
    print(
        f"Finished {attack_todo} attack on {experiment_name} in {(t2-t1)/3600:.2f}hours"
    )
    return


def save_results(results_files, total_result, experiment_name):
    # save all the attacks
    for current_attack, result_file in results_files.items():
        res = total_result[current_attack]
        # Rewrite the columns in a better order
        columns = res.columns.tolist()
        columns.sort(key=lambda x: (1, "") if "loss_trainset_" in x else (0, x))
        if current_attack == "biasedthreshold":
            columns.sort(
                key=lambda x: (1, len(x), x) if "_acc" in x else (0, len(x), x)
            )

        res = res[columns]
        if res.shape[0] < 10:
            print(
                f"Only {res.shape[0]} in the result for {current_attack} - {experiment_name}"
            )
            print(res)
        # This code fragment breaks classifier attacks, as we only attack a few number of nodes.
        #     raise ValueError(res)

        # Save the file
        print(f"Writing results to {result_file}")
        res.to_csv(result_file)


def clear_models(experiment_name: str, experiment_path: str):
    for attack in ALL_ATTACKS:
        results_file = os.path.join(experiment_path, f"{attack}_{experiment_name}.csv")
        # Warning: this must be the same thing as is done in the model formating
        if not os.path.exists(results_file):
            print(f"Missing {attack} for experiment {experiment_name}, not clearing")
            return False
    cwd = subprocess.run(["pwd"], check=True)
    print(f"Current directory: {cwd}")
    print(f"All attacks computed for {experiment_name}: cleaning results")

    for folder in os.listdir(experiment_path):
        folder_path = os.path.join(experiment_path, folder)
        if os.path.isdir(folder_path):
            attacked_models_path = os.path.join(folder_path, "attacked_model")
            if os.path.isdir(attacked_models_path):
                print(f"Removing {attacked_models_path}")
                shutil.rmtree(attacked_models_path, ignore_errors=True)

    # files_to_remove = subprocess.run(
    #     ["find", experiment_path, "-type", "f", "-name", "*.npy"],
    #     check=True,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
    # for file_byte in files_to_remove.stdout.splitlines():
    #     file = file_byte.decode("utf-8")
    #     assert os.path.isfile(file), f"Files should exist: {file}"
    #     os.remove(file)

    return True


def print_finished():
    n = 40
    print("-" * n + "\nAll attacks have been launched!\n" + "-" * n)


def main(
    experiments_dir,
    datasets_dir,
    should_clear,
    nb_processes=10,
    batch_size=256,
    attacks=ALL_ATTACKS,
):
    device_type = "cuda:0"  # TODO: change this to ensure it is available?

    # ---------
    # Running the main attacks on all the experiments
    # ---------
    print(f"CUDA status: {torch.cuda.is_available()}")
    print("---- Starting main attacks ----")

    # experiments_dir = "attacks/my_results/icml_experiments/cifar10/"
    # experiments_dir = "attacks/my_results/icml_experiments/additional_cifar10/"

    print("Loading experiments dirs")
    all_experiments_df = load_experiments.get_all_experiments_properties(
        experiments_dir, attacks
    )

    # When debugging, save the dataframe and load it to avoid cold starts.
    # all_experiments_df.to_csv("experiments_df.csv")
    # all_experiments_df = pd.read_csv("experiments_df.csv")

    # Use this we want to reduce the size of the experiments to consider for quick results
    # results_path = "attacks/my_results/icml_experiments/cifar10_attack_results_quick/"
    # all_experiments_df = all_experiments_df[
    #     all_experiments_df["iteration"].isin([500, 5000, 10000])
    # ]

    all_experiments_df.reset_index()
    print(all_experiments_df)
    print("Loading done, starting all attacks\n")
    print(f"Attacking:")
    for experiment_name in all_experiments_df["experiment_name"].drop_duplicates():
        print(experiment_name)
    print("-" * 10)
    futures = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=nb_processes, mp_context=context
    ) as executor:
        for attack in attacks:
            for experiment_name in sorted(
                pd.unique(all_experiments_df["experiment_name"])
            ):
                if DEBUG:
                    attack_experiment(
                        copy.deepcopy(all_experiments_df),
                        experiment_name=copy.deepcopy(experiment_name),
                        batch_size=copy.deepcopy(batch_size),
                        experiment_path=os.path.join(experiments_dir, experiment_name),
                        datasets_dir=copy.deepcopy(datasets_dir),
                        device_type=copy.deepcopy(device_type),
                        attack_todo=copy.deepcopy(attack),
                        debug=DEBUG,
                    )
                    res = input("Continue? y/n\n")
                    if res != "y":
                        return
                else:
                    future = executor.submit(
                        attack_experiment,
                        copy.deepcopy(all_experiments_df),
                        experiment_name=copy.deepcopy(experiment_name),
                        batch_size=copy.deepcopy(batch_size),
                        experiment_path=os.path.join(experiments_dir, experiment_name),
                        datasets_dir=copy.deepcopy(datasets_dir),
                        device_type=copy.deepcopy(device_type),
                        attack_todo=copy.deepcopy(attack),
                        debug=DEBUG,
                    )
                    futures.append(future)
        executor.submit(print_finished)  # To have a display when things are finished
        done, _ = concurrent.futures.wait(futures)
        results = []
        for future in done:
            results.append(future.result())

        print(results)
    if should_clear:
        print("Clearing results:")
        for experiment_name in sorted(pd.unique(all_experiments_df["experiment_name"])):
            clear_models(
                experiment_name=experiment_name,
                experiment_path=os.path.join(experiments_dir, experiment_name),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attacks saved models after simulation"
    )

    parser.add_argument(
        "experiment_dir",
        type=str,
        default="attacks/my_results/test/testing_femnist_convergence_rates/",
        help="Path to the experiments directory. It should be a directory with only subdirectories, with each subdirectory corresponding to an experiment.",
    )

    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="datasets/",
        help="Path to the datasets directory (usually all of them)."
        + " Useful when running inside a container.",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        default=20,
        help="Number of workers to use (default: 20)",
    )

    parser.add_argument(
        "--clear", action="store_true", help="Whether to clear something"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used for attacks. Scale this for faster attacks. Default 256.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode. Adds some printing.",
    )

    args = parser.parse_args()

    EXPERIMENT_DIR = args.experiment_dir
    DATASETS_DIR = args.datasets_dir
    NB_WORKERS = args.nb_workers
    SHOULD_CLEAR = args.clear
    BATCH_SIZE = args.batch_size
    DEBUG = args.debug
    ALL_ATTACKS = ["classifier", "linkability", "threshold+biasedthreshold"]

    main(
        experiments_dir=EXPERIMENT_DIR,
        datasets_dir=DATASETS_DIR,
        should_clear=SHOULD_CLEAR,
        nb_processes=NB_WORKERS,
        batch_size=BATCH_SIZE,
        attacks=ALL_ATTACKS,
    )
