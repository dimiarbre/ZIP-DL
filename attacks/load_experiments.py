import functools
import json
import os
import time
import traceback
from collections import defaultdict
from typing import Any, Optional, Type, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from localconfig import LocalConfig
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import decentralizepy.datasets.MovieLens
from decentralizepy.datasets.CIFAR10 import GN_ResNet18, LeNet
from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Femnist import CNN, RNET, Femnist
from decentralizepy.datasets.FemnistLabelSplit import GN_RNET
from decentralizepy.datasets.MovieLens import MatrixFactorization
from decentralizepy.datasets.Partitioner import (
    DataPartitioner,
    DirichletDataPartitioner,
    KShardDataPartitioner,
    SimpleDataPartitioner,
)

# Sort this by longest computation time first to have a better scheduling policy.
ALL_ATTACKS = [
    "linkability",
    "threshold+biasedthreshold",
    "biasedthreshold",
    "threshold",
]


def error_catching_wrapper(func):
    """Wrapper that prints the stack trace when an error is raised
    Useful when multiprocessing, as otherwise the error is just silent.
    Tweaks were needed as it would otherwise not work in multiprocess scenario because of some weird recursive shenanigans.
    """

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error occurred in function '{func.__name__}': {e}")
            traceback.print_exc()
            raise e

    return wrapped_function


def read_ini(file_path: str, verbose=False) -> LocalConfig:
    """Function to load the dict configuration file.

    Args:
        file_path (str): The path to the config file

    Returns:
        LocalConfig: The loaded configuration.
    """
    config = LocalConfig(file_path)
    if verbose:
        for section in config:
            print("Section: ", section)
            for key, value in config.items(section):
                print((key, value, type(value)))
        print(dict(config.items("DATASET")))
    return config


T = TypeVar("T")


def safe_load(
    config: LocalConfig, section: str, parameter: str, expected_type: Type[T]
) -> T:
    """Type enforcing function to load data from localconfig, and ensuring the desired type is met.

    Args:
        config (LocalConfig): The local configuration to load from.
        section (str): The section in the local configuration
        parameter (str): The parameter in the section
        expected_type (Type[T]): The desired type. Commonly str,int or float.

    Raises:
        ValueError: The loaded type did not match the expected_type.

    Returns:
        T: Loaded value converted to the expected_type.
    """
    value = config.get(section, parameter)
    if not isinstance(value, expected_type):
        # Check for int->float possible conversion.
        if expected_type == float and isinstance(value, int):
            value = float(value)
            value = cast(T, value)
        else:
            raise ValueError(
                f"Invalid value for parameter {parameter}: expected {expected_type}, got {value}"
            )
    return value


def load_CIFAR10(
    nb_nodes, sizes, datasets_dir: str, config: LocalConfig, debug: bool = False
):
    """Generates and partitions the CIFAR10 dataset in the same way as the experiment does."""
    print("Loading CIFAR10 dataset")
    num_classes = 10
    path = os.path.join(datasets_dir, "cifar10")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform
    )
    partition_niid = safe_load(config, "DATASET", "partition_niid", str)

    seed = safe_load(config, "DATASET", "random_seed", int)

    c_len = len(trainset)
    if sizes is None:
        # Equal distribution of data among processes, from CIFAR10 and not a simplified function.
        e = c_len // nb_nodes
        frac = e / c_len
        sizes = [frac] * nb_nodes
        sizes[-1] += 1.0 - frac * nb_nodes
        # print(f"Size fractions: {sizes}")

    if partition_niid == "iid":
        partitioner = DataPartitioner(trainset, sizes=sizes, seed=seed)

    elif partition_niid == "kshard":
        train_data: dict = {key: [] for key in range(num_classes)}
        for x, y in trainset:
            train_data[y].append(x)
        all_trainset = []
        for y, x in train_data.items():
            all_trainset.extend([(a, y) for a in x])
        shards = safe_load(config, "DATASET", "shards", int)
        partitioner = KShardDataPartitioner(
            all_trainset, sizes, shards=shards, seed=seed
        )
    elif partition_niid == "dirichlet":
        alpha = safe_load(config, "DATASET", "alpha", float)
        partitioner = DirichletDataPartitioner(
            data=trainset,
            sizes=sizes,
            seed=seed,
            alpha=alpha,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            f"Invalid configuration for partition_niid:{partition_niid}.\n"
            f"Full config: {config}."
        )

    return partitioner, testset


def femnist_read_file(file_path):
    with open(file_path, "r") as inf:
        client_data = json.load(inf)
    return (
        client_data["users"],
        client_data["num_samples"],
        client_data["user_data"],
    )


def femnist_read_dir(data_dir) -> tuple[list, list, dict]:
    """
    Function to read all the Femnist data files in the directory

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the data files

    Returns
    -------
    3-tuple
        A tuple containing list of clients, number of samples per client,
        and the data items per client

    """
    clients = []
    num_samples = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        u, n, d = femnist_read_file(file_path)
        clients.extend(u)
        num_samples.extend(n)
        data.update(d)
    return clients, num_samples, data


class PartitionerWrapper(DataPartitioner):
    """
    This class is a wrapper of a Datapartitioner that is there to match the CIFAR10 syntax.
    It is basically a list, with a .use() method that is the same as .__get__()
    It will be used for cases where partition is done intrinsically to the dataset.
    """

    def __init__(self, data, sizes=None, seed=1234):
        self.partitions = data

    def use(self, rank):
        return self.partitions[rank]


def load_Femnist_Testset(femnist_test_dir):
    # Load the test set
    _, _, d = femnist_read_dir(femnist_test_dir)
    test_x = []
    test_y = []
    for test_data in d.values():
        for x in test_data["x"]:
            test_x.append(x)
        for y in test_data["y"]:
            test_y.append(y)
    test_x = (
        np.array(test_x, dtype=np.dtype("float32"))
        .reshape(-1, 28, 28, 1)
        .transpose(0, 3, 1, 2)
    )
    test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
    assert test_x.shape[0] == test_y.shape[0]
    assert test_x.shape[0] > 0

    return Data(test_x, test_y)


def load_Femnist(
    nb_nodes,
    sizes,
    random_seed,
    datasets_dir: str,
    femnist_train_dir="Femnist/femnist/per_user_data/per_user_data/train",
    femnist_test_dir="Femnist/femnist/data/test/test",
    debug=False,
):
    """Generates and partitions the CIFAR10 dataset in the same way as the experiment does."""
    print("Loading Femnist dataset")
    true_femnist_train_dir = os.path.join(datasets_dir, femnist_train_dir)
    true_femnist_test_dir = os.path.join(datasets_dir, femnist_test_dir)
    files = os.listdir(true_femnist_train_dir)
    files = [f for f in files if f.endswith(".json")]
    files.sort()

    c_len = len(files)

    if sizes == None:  # Equal distribution of data among processes
        e = c_len // nb_nodes
        frac = e / c_len
        sizes = [frac] * nb_nodes
        sizes[-1] += 1.0 - frac * nb_nodes

    # Load all the trainsets
    files_partitioner = DataPartitioner(files, sizes, seed=random_seed)
    all_trainsets = []
    for uid in range(0, nb_nodes):
        if debug:
            print(f"Loading data for client {uid}.")
        my_clients = files_partitioner.use(uid)
        my_train_data = {"x": [], "y": []}
        node_clients = []
        node_num_samples = []
        if debug:
            print(f"{uid} - Client Length: {c_len}")
            print(f"{uid} - My_clients_len: {len(my_clients)}")
        for cur_file in my_clients:

            clients, _, train_data = femnist_read_file(
                os.path.join(true_femnist_train_dir, cur_file)
            )
            for cur_client in clients:
                node_clients.append(cur_client)
                current_client_data = train_data[cur_client]
                # f, axarr = plt.subplots(10, 1)
                # for i in range(10):
                #     to_plot = (
                #         np.array(current_client_data["x"][i])
                #         .reshape(-1, 28, 28, 1)
                #         .transpose()
                #     )
                #     axarr[i].imshow(to_plot[0])
                # plt.show()

                my_train_data["x"].extend(current_client_data["x"])
                my_train_data["y"].extend(current_client_data["y"])
                node_num_samples.append(len(train_data[cur_client]["y"]))
        node_train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        node_train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        assert node_train_x.shape[0] == node_train_y.shape[0]
        assert node_train_x.shape[0] > 0
        node_data = Data(node_train_x, node_train_y)
        all_trainsets.append(node_data)

    #     if debug:
    #         # subplot(r,c) provide the no. of rows and columns
    #         f, axarr = plt.subplots(10, 1)
    #         for i in range(10):
    #             to_plot = node_data.x[i][0]
    #             axarr[i].imshow(to_plot)
    # if debug:
    #     plt.show()

    testset = load_Femnist_Testset(femnist_test_dir=true_femnist_test_dir)

    return PartitionerWrapper(all_trainsets, [], 0), testset


def load_Femnist_labelsplit(
    nb_nodes,
    sizes,
    random_seed,
    datasets_dir: str,
    femnist_train_dir="Femnist_labelsplit/femnist/data/train/64nodes_10shards",  # TODO: fix this hack, and load it from the config file
    femnist_test_dir="Femnist_labelsplit/femnist/data/test/test",
    debug=False,
):
    true_femnist_train_dir = os.path.join(datasets_dir, femnist_train_dir)
    true_femnist_test_dir = os.path.join(datasets_dir, femnist_test_dir)
    all_data = []
    for node in range(nb_nodes):
        filename = f"data_{node}.pt"
        if debug:
            print(f"Loading FEMNISTLabelSplit {filename} for node {node}.")
        node_file = os.path.join(true_femnist_train_dir, filename)
        node_data = torch.load(node_file)
        temp = np.array(
            [data[0] for data in node_data],
            dtype=np.dtype("float32"),
        )
        node_data_reshaped_x = temp.reshape(-1, 28, 28, 1).transpose(0, 3, 1, 2)
        node_data_reshaped_y = np.array(
            [data[1] for data in node_data], dtype=np.dtype("int64")
        ).reshape(-1)
        node_data_reshaped = Data(node_data_reshaped_x, node_data_reshaped_y)
        all_data.append(node_data_reshaped)

        if debug:
            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(10, 1)
            for i in range(10):
                to_plot = node_data_reshaped.x[i][0]
                axarr[i].imshow(to_plot)
            f.savefig(f"assets/temp_images/{node}.png")
            plt.close()

    testset = load_Femnist_Testset(femnist_test_dir=true_femnist_test_dir)

    return PartitionerWrapper(all_data), testset


def load_movielens(
    nb_nodes, sizes, random_seed, dataset_dir, debug=False
) -> tuple[DataPartitioner, Data]:
    movielens_dir = os.path.join(dataset_dir, "MovieLens")
    if random_seed is None:
        random_seed = 1234  # To match with the code's default value
    users_count, _, df_train, df_test = decentralizepy.datasets.MovieLens.load_data(
        train_dir=movielens_dir,
        random_seed=random_seed,
    )
    local_train_datasets = []
    local_test_datasets_x = []
    local_test_datasets_y = []
    for i in range(nb_nodes):
        local_train_data, local_test_data = (
            decentralizepy.datasets.MovieLens.split_data(
                df_train,
                df_test,
                n_users=users_count,
                world_size=nb_nodes,
                dataset_id=i,
            )
        )
        local_train_x = local_train_data[["user_id", "item_id"]].to_numpy()
        local_train_y = local_train_data.rating.values.astype("float32")
        local_test_x = local_test_data[["user_id", "item_id"]].to_numpy()
        local_test_y = local_test_data.rating.values.astype(
            "float32"
        )  # NB: This line was changed compared to the original code to also output float32, so that losses have the same dtype
        # TODO: ensure this line change is correct

        local_train_datasets.append(Data(local_train_x, local_train_y))
        local_test_datasets_x.append(local_test_x)
        local_test_datasets_y.append(local_test_y)

    # Agregate the test dataset ?
    # TODO: ensure we want to do this
    global_test_dataset_x = np.concatenate(local_test_datasets_x)
    global_test_dataset_y = np.concatenate(local_test_datasets_y)

    global_test_dataset = Data(global_test_dataset_x, global_test_dataset_y)
    # global_test_dataset = ConcatDataset(local_test_datasets)

    if debug:
        print(
            f"Generated {nb_nodes} trainsets. Dataset sizes: {[len(local_trainset) for local_trainset in local_train_datasets]}"
        )
        # print(f"Shape of test data: {global_test_dataset.cumulative_sizes}") # Cannot use this if it is not a ConcatDataset
        print(f"Nb of test data: {len(global_test_dataset)}")
    return PartitionerWrapper(local_train_datasets), global_test_dataset


POSSIBLE_DATASETS = {
    "CIFAR10": (
        load_CIFAR10,
        10,
    ),
    "Femnist": (
        load_Femnist,
        62,
    ),
    "FemnistLabelSplit": (
        load_Femnist_labelsplit,
        62,
    ),
    "MovieLens": (
        load_movielens,
        10,
    ),
}

POSSIBLE_MODELS = {
    "RNET": RNET,
    "LeNet": LeNet,
    "GN_RNET": GN_RNET,
    "CNN": CNN,
    "MatrixFactorization": MatrixFactorization,
    "GN_ResNet18": GN_ResNet18,  # The one from CIFAR!!!
}

POSSIBLE_LOSSES: dict[str, type[torch.nn.Module]] = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "MSELoss": MSELoss,
}


def load_dataset_partitioner(
    dataset_name: str,
    datasets_dir: str,
    total_agents: int,
    seed: int,
    config: LocalConfig,
    sizes: Optional[list[float]] = None,
    debug=False,
) -> tuple[DataPartitioner, Data]:
    """Loads a data partitioner in an identical way as the experiments do

    Args:
        dataset_name (str): The name of the dataset: CIFAR10
        total_agents (int): The number of agents for the partition
        seed (int): The random seed used in the experiment
        shards (int): The number of shards for each agent
        config (LocalConfig): The configuration file.
        sizes (float array, optional): A list of data fraction for each agent . Defaults to None, corresponding to an equal distribution.

    Raises:
        ValueError: When the dataset name is not a currently implemented dataset to be loaded.

    Returns:
        decentralizepy.datasets.KShardDataPartitioner: A data partitioner
        testset: The test dataset
    """
    if dataset_name not in POSSIBLE_DATASETS:
        raise ValueError(
            f"{dataset_name} is not in the list of possible datasets: {list(POSSIBLE_DATASETS)}"
        )
    loader, num_classes = POSSIBLE_DATASETS[dataset_name]
    if dataset_name == "Femnist":
        # Because Femnist is handled by clients, it is a very different scenario here.
        all_trainsets, testset = load_Femnist(
            total_agents,
            sizes=sizes,
            random_seed=seed,
            debug=debug,
            datasets_dir=datasets_dir,
        )
        return all_trainsets, testset
    elif dataset_name == "FemnistLabelSplit":
        # Since the sharding is done intresequely for optimization reasons, we load manually here
        # TODO: specify a manual train dir to mach to other number of nodes.
        all_trainsets, testset = load_Femnist_labelsplit(
            nb_nodes=total_agents,
            sizes=sizes,
            random_seed=seed,
            debug=debug,
            datasets_dir=datasets_dir,
        )
        return all_trainsets, testset
    elif dataset_name == "MovieLens":
        all_trainsets, testset = load_movielens(
            nb_nodes=total_agents,
            sizes=sizes,
            random_seed=seed,
            dataset_dir=datasets_dir,
            debug=debug,
        )
        return all_trainsets, testset
    elif dataset_name == "CIFAR10":
        all_trainsets, testset = load_CIFAR10(
            nb_nodes=total_agents,
            sizes=sizes,
            datasets_dir=datasets_dir,
            config=config,
            debug=debug,
        )
        return all_trainsets, testset
    raise ValueError(f"Unexpected dataset {dataset_name}")


def get_dataset_stats(dataset, nb_classes: int, dataset_name: Optional[str]):
    """Logs the data repartition for all classes of a dataset


    Args:
        dataset (decentralizepy.datasets.Partition): A dataset partition for a local agent
        nb_classes (int): The number of classes for the dataset
        dataset_name (Optinal str): The name of the dataset. Used for MovieLens, that has a pecculiar class repartition.

    Returns:
        int array: An array of size nb_classes corresponding to the number of element for each class
    """
    classes = [0 for _ in range(nb_classes)]
    for _, target in dataset:
        if dataset_name is not None and dataset_name == "MovieLens":
            target = int(2 * target) - 1
        classes[target] += 1
    return classes


def get_dataset_stats_batch(dataset, nb_classes: int) -> list[int]:
    classes = [0 for _ in range(nb_classes)]
    for _, target_batches in dataset:
        for target in target_batches:
            classes[target] += 1
    return classes


def deserialized_model(weights, model, shapes, lens):
    """
    Convert received dict to state_dict.

    Parameters
    ----------
    m : dict
        received dict

    Returns
    -------
    state_dict
        state_dict of received

    """
    state_dict = dict()
    start_index = 0
    for i, key in enumerate(model.state_dict()):
        end_index = start_index + lens[i]
        state_dict[key] = torch.from_numpy(
            weights[start_index:end_index].reshape(shapes[i])
        )
        start_index = end_index
    return state_dict


def load_model_from_path(model_path, model, shapes, lens, device=None):
    model_weigths = np.load(model_path)
    model.load_state_dict(deserialized_model(model_weigths, model, shapes, lens))
    if device is not None:
        model.to(device)


def generate_shapes(model: torch.nn.Module) -> tuple[list[tuple[int, int]], list[int]]:
    """Generates the shapes of a model

    Args:
        model (torch.nn.Module): The model we will be loading data into

    Returns:
        shapes: a list of the shapes for each layer of the model.
        lens (list[int]): a list of all the number of parameter for each layer.
    """
    shapes = []
    lens: list[int] = []
    with torch.no_grad():
        for _, v in model.state_dict().items():
            shapes.append(v.shape)
            t = v.flatten().cpu().numpy()
            lens.append(t.shape[0])
    return shapes, lens


def generate_losses(
    model: torch.nn.Module,
    dataset,
    loss_function: torch.nn.Module = torch.nn.CrossEntropyLoss(
        reduction="none"
    ),  # We want each individual losses.
    device=torch.device("cpu"),
    debug=False,
):
    assert (
        loss_function.reduction == "none"
    ), "Reduction should be none to generate all losses"
    is_mse = isinstance(
        loss_function, MSELoss
    )  # We can't compute some metrics in this case
    losses = torch.tensor([])
    classes = torch.tensor([])
    # Fixes inconsistent batch size
    # See https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/262741
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_predicted = 0
        for x, y in dataset:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss = loss.to("cpu")
            losses = torch.cat([losses, loss])
            y_cpu = y.to("cpu")
            classes = torch.cat([y_cpu, classes])
            if not is_mse:
                _, predictions = torch.max(y_pred, 1)
                for label, prediction in zip(y, predictions):
                    if label == prediction:
                        total_correct += 1
                    total_predicted += 1
    if losses.dtype == torch.float64:
        losses.float()
    if not is_mse:
        accuracy = total_correct / total_predicted
        return (losses, classes, accuracy)
    else:
        return (losses, classes, None)


def filter_nans(losses: torch.Tensor, classes, debug_name, loss_type):
    if losses.isnan().any():
        nans_loc = losses.isnan()
        losses_nonan = losses[~nans_loc]
        percent_fail = (len(losses) - len(losses_nonan)) / len(losses) * 100
        print(
            f"{debug_name} - Found NaNs in {loss_type} loss! Removed {percent_fail:.2f}% of {loss_type} losses"
        )
        losses = losses_nonan
        if classes is not None:
            classes = classes[~nans_loc]
    return losses, classes


def get_model_attributes(name, path):
    parsed_model = name.split("_")
    iteration = int(parsed_model[1][2:])
    agent = int(parsed_model[2])
    target = int(parsed_model[3][2:-4])  # Ignore the `.npy`

    res = pd.DataFrame(
        {
            "file": os.path.join(path, name),
            "iteration": iteration,
            "agent": agent,
            "target": target,
        },
        index=[0],
    )
    return res


def list_models_path(
    experiment_path: str, machine_id: int, agent_id: int, attacks
) -> pd.DataFrame:
    """Generates dataframe for a given agent and machine id

    Args:
        experiment_path (str): The directory of the experiment
        machine_id (int): The id of the machine for the agent
        agent_id (int): The id of the agent on the machine
        attacks (str list): list of attacks to perform

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid.
    """
    models_list = pd.DataFrame({})
    agent_models_directory = os.path.join(
        experiment_path,
        f"machine{machine_id}",
        "attacked_model",
        f"machine{machine_id}",
        f"{agent_id}",
    )
    if not os.path.isdir(agent_models_directory):
        return models_list
    for file in sorted(os.listdir(agent_models_directory)):
        file_attributes = get_model_attributes(file, agent_models_directory)
        models_list = pd.concat([models_list, file_attributes])
    return models_list


def get_all_models_properties(
    experiment_dir: str, nb_agents: int, nb_machines: int, attacks
) -> pd.DataFrame:
    """Generates a dataframe with all models path and properties (agent, iteration, target agent)

    Args:
        experiment_dir (str): The path to the experiment
        nb_agents (int): The total number of agent for the experiment
        nb_machines (int): The total number of machines for the experiment
        attacks (string list): List of attacks to perform

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid.
    """
    models_df = pd.DataFrame({})
    for agent_uid in range(nb_agents):
        current_machine_id = agent_uid // (nb_agents // nb_machines)
        agent_id = agent_uid % (nb_agents // nb_machines)
        models_df = pd.concat(
            [
                models_df,
                list_models_path(experiment_dir, current_machine_id, agent_id, attacks),
            ]
        )

    return models_df


def get_all_experiments_properties(all_experiments_dir: str, attacks) -> pd.DataFrame:
    """Generates a dataframe with all models path and properties (agent, iteration, target agent)
    for all the experiments in a folder.

    Args:
        all_experiments_dir (str): The path to the experiments

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid, unified for all the experiments.
    """
    experiment_wide_df = pd.DataFrame({})
    nb_agents = None
    for experiment_name in os.listdir(all_experiments_dir):
        experiment_path = os.path.join(all_experiments_dir, experiment_name)

        # Fetch information from the config file automatically.
        g5k_config_file = os.path.join(experiment_path, "g5k_config.json")
        assert os.path.exists(
            g5k_config_file
        ), f"Cannot find g5k config file at {g5k_config_file}"
        with open(g5k_config_file, "r") as f:
            g5k_config = json.load(f)
        if nb_agents is not None:
            assert (
                nb_agents == g5k_config["NB_AGENTS"]
            ), f"Inconsistent number of agents: {nb_agents} and {g5k_config['NB_AGENTS']}"
        nb_agents = g5k_config["NB_AGENTS"]
        nb_machines = g5k_config["NB_MACHINE"]
        current_experiment_df = get_all_models_properties(
            experiment_path,
            nb_agents=nb_agents,
            nb_machines=nb_machines,
            attacks=attacks,
        )
        if not current_experiment_df.empty:
            current_experiment_df["experiment_name"] = experiment_name
            experiment_wide_df = pd.concat([experiment_wide_df, current_experiment_df])
        else:
            print(f"Skipping {experiment_name} as no models were found.")
    experiment_wide_df = experiment_wide_df.astype(
        {"iteration": "int32", "agent": "int32", "target": "int32"}
    )
    return experiment_wide_df


def main():
    dataset = "MovieLens"
    dataset_dir = "datasets/"
    nb_classes = POSSIBLE_DATASETS[dataset][1]
    nb_agents = 100
    nb_machines = 2
    dummy_config = LocalConfig()
    dummy_config.add_section("DATASET")
    dummy_config.set("DATASET", "kshards", 2)
    dummy_config.set("DATASET", "partition_niid", "kshard")
    train_partitioner, test_data = load_dataset_partitioner(
        dataset,
        datasets_dir=dataset_dir,
        total_agents=nb_agents,
        seed=1234,
        config=dummy_config,
        debug=True,
    )
    nb_data = 0
    for agent in range(nb_agents):
        train_data_current_agent = train_partitioner.use(agent)
        agent_classes_trainset = get_dataset_stats(
            train_data_current_agent, nb_classes, dataset_name=dataset
        )
        print(f"Classes for agent {agent}: {agent_classes_trainset}")
        nb_data_agent = sum(agent_classes_trainset)
        print(f"Total number of data for agent {agent}: {nb_data_agent}")
        nb_data += nb_data_agent
    print(f"Total data for the {dataset} dataset: {nb_data}")
    experiment_dir = "attacks/my_results/movielens/2456067_movielens_nonoise_100nodes_1avgsteps_static"

    all_models_df = get_all_models_properties(
        experiment_dir, nb_agents, nb_machines, ["threshold"]
    )
    print(all_models_df)

    experiments_dir = "attacks/my_results/movielens/"

    t0 = time.time()
    all_experiments_df = get_all_experiments_properties(experiments_dir, ALL_ATTACKS)
    t1 = time.time()
    print(all_experiments_df)
    print(f"All models paths retrieved in {t1-t0:.2f}s")


if __name__ == "__main__":
    main()
