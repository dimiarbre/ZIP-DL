import copy

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scienceplots
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import tqdm
from scipy.stats import ks_2samp

import decentralizepy.datasets.CIFAR10
import decentralizepy.datasets.MovieLens as MovieLens
from attacks.loaders.load_experiments import (
    deserialized_model,
    generate_shapes,
    load_movielens,
)

plt.style.use(["science"])
plt.rcParams.update({"font.size": 14})


def get_gradient(model: nn.Module):
    """Retrieve gradients of model parameters and store them in a single vector."""
    gradient = torch.cat(
        [param.grad.view(-1) for param in model.parameters() if param.grad is not None]
    ).cpu()
    return gradient


def train(
    model: nn.Module,
    nb_iter: int,
    criterion: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device = torch.device("cpu"),
):
    model.train()
    train_losses = []
    # gradients = []
    pbar = tqdm.tqdm(range(nb_iter))

    for epoch in pbar:  # loop over the dataset multiple times

        running_loss = 0.0
        total_samples = 0  # Initialize total samples counter
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # gradients.append(get_gradient(model))

            # print statistics
            running_loss += loss.item()
            total_samples += labels.size(0)
        running_loss = running_loss / total_samples
        pbar.set_description(f"Epoch {epoch} - train loss: {running_loss}")
        train_losses.append(running_loss)
        # scheduler.step()

    print("Finished Training")
    # gradients_torch = torch.stack(
    #     gradients
    # )  # Gradients collected over training iterations
    # covariance_matrix = torch.cov(gradients_torch.T)  # Covariance matrix
    # shapiro_results = [
    #     stats.shapiro(gradients_torch[:, i].cpu().numpy())
    #     for i in range(gradients_torch.shape[1])
    # ]

    # # Interpret results
    # for param_idx, (stat, p_value) in enumerate(shapiro_results):
    #     if p_value > 0.05:
    #         print(
    #             f"Parameter {param_idx} follows a Gaussian distribution (p-value={p_value:.4f})."
    #         )
    #     else:
    #         print(
    #             f"Parameter {param_idx} does NOT follow a Gaussian distribution (p-value={p_value:.4f})."
    #         )

    return train_losses


# From decentralizepy.sharing.Sharing
def serialized_model(model):
    """
    Convert model to a dictionary. Here we can choose how much to share

    Returns
    -------
    flat
        Model parameters flattened in a torch array.

    """
    to_cat = []
    with torch.no_grad():
        for name, v in model.state_dict().items():
            t = v.flatten()
            to_cat.append(t)
    flat = torch.cat(to_cat)
    return flat.cpu().numpy()


def compute_wasserstein_distance(sample1, sample2):
    M = ot.dist(sample1, sample2)

    # Assume uniform distributions over the samples
    n = sample1.shape[0]
    m = sample2.shape[0]
    a = np.ones(n) / n  # Uniform distribution for x1
    b = np.ones(m) / m  # Uniform distribution for x2

    return ot.emd2(a, b, M)


def test_1_gradient(
    model: nn.Module,
    nb_test: int,
    sigma: float,
    criterion: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    dataset_name: str,
    nb_training: int,
    device: torch.device = torch.device("cpu"),
):
    nb_parameters = 10000  # Number of parameters to do the test on

    experiment_name = (
        f"{dataset_name}_{nb_test}tests_{sigma}sigma_{nb_training}training"
    )

    lens, shapes = generate_shapes(model)
    base_parameters = serialized_model(model)
    generator = np.random.RandomState(421)
    gradients = []

    # Get first batch.
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        break

    print(model)

    print(f"Generating descent then noise")
    current_parameters = copy.deepcopy(base_parameters)

    model.load_state_dict(deserialized_model(current_parameters, model, lens, shapes))
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    true_gradient = get_gradient(model)

    idx = np.arange(0, len(true_gradient))
    nb_parameters = min(nb_parameters, len(true_gradient))
    rand_idx = np.random.choice(idx, nb_parameters, replace=False)

    for i in tqdm.tqdm(range(nb_test)):
        noise = generator.normal(0, sigma, current_parameters.shape)
        gradient = true_gradient + noise
        gradients.append(gradient[rand_idx])

    gradients_sgd_then_noise = np.array(gradients)

    gradients = []
    print(f"Generating noise then descent")
    for i in tqdm.tqdm(range(nb_test)):
        current_parameters = copy.deepcopy(base_parameters)
        noise = generator.normal(0, sigma, current_parameters.shape)
        current_parameters += noise

        model.load_state_dict(
            deserialized_model(current_parameters, model, lens, shapes)
        )
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        gradient = get_gradient(model)
        gradients.append(gradient[rand_idx])

    gradients_noise_then_sgd = np.array(gradients)
    del gradients

    # fig, ax = plt.subplots(1, 1)
    print("Checking statistics with normalization.")
    results = []
    nb_0 = 0

    for parameter in tqdm.tqdm(range(nb_parameters)):
        current_v1, current_v2 = (
            gradients_noise_then_sgd[:, parameter],
            gradients_sgd_then_noise[:, parameter],
        )
        assert current_v1.shape == current_v2.shape
        norm_v1 = np.linalg.norm(current_v1)
        norm_v2 = np.linalg.norm(current_v2)
        if np.isclose(norm_v1, 0) or np.isclose(norm_v2, 0):
            nb_0 += 1
            continue
        ks_results = ks_2samp(
            current_v1 / norm_v1,
            current_v2 / norm_v2,
        )
        results.append((parameter, ks_results))

    print(f"Discarded {nb_0} parameters because of 0 norm.")
    res_normalized = [x.statistic for _, x in results]

    ws_distance = compute_wasserstein_distance(
        gradients_noise_then_sgd, gradients_sgd_then_noise
    )

    print(f"Wasserstein distance: {ws_distance}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    res = pd.DataFrame({})
    for i, parameter in enumerate(
        [np.argmin(res_normalized), np.argmax(res_normalized)]
    ):
        parameter_distribution = gradients_noise_then_sgd[:, results[parameter][0]]
        current_res = pd.DataFrame({})
        current_res["data"] = parameter_distribution
        current_res["parameter"] = "best" if i == 0 else "worse"

        # Plot the histogram as counts (not densities)
        counts, bins, patches = axs[i].hist(
            parameter_distribution,
            bins=20,
            label="Parameter distribution",
            alpha=0.6,
            color="blue",
            edgecolor="black",
        )

        # Calculate mean and standard deviation
        mu = np.mean(parameter_distribution)
        empirical_sigma = np.std(parameter_distribution)

        current_res["mu"] = mu
        current_res["sigma"] = empirical_sigma

        # Calculate bin width for scaling the Gaussian
        bin_width = bins[1] - bins[0]
        scaling_factor = len(parameter_distribution) * bin_width

        # Generate values for the theoretical Gaussian
        x = np.linspace(bins[0], bins[-1], 1000)
        y = stats.norm.pdf(x, mu, empirical_sigma) * scaling_factor

        # Plot the scaled Gaussian
        axs[i].plot(x, y, label="Theoretical Gaussian", color="red", linewidth=2)
        axs[i].set_xlim(bins[0], bins[-1])
        axs[i].legend(fontsize=12)

        res = pd.concat([res, current_res])

    res.to_csv(f"assets/parameter_distribution/{experiment_name}.csv")

    axs[0].set_title("Best parameter")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("Parameter value")

    axs[1].set_xlabel("Parameter value")
    axs[1].set_title("Worst parameter")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"assets/parameter_distribution/{experiment_name}.pdf")
    return res_normalized


def init_cifar(batch_size=2048):
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # INIT MODEL
    model = decentralizepy.datasets.CIFAR10.GN_ResNet18()

    # LOSSES
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    return "CIFAR10", model, trainloader, testloader, criterion, optimizer


def init_movielens(batch_size=2048):
    partition_train, test_dataset = load_movielens(1, None, 421, "datasets")
    model = MovieLens.MatrixFactorization()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    train_dataset = partition_train.use(0)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset)
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False)
    return "MovieLens", model, trainloader, test_loader, criterion, optimizer


class LogisticRegression(torch.nn.Module):
    # build the constructor
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)

    # make predictions
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


def init_mnist(batch_size=2048):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1, 28 * 28).squeeze(0)),
        ]
    )
    # loading training data
    train_dataset = torchvision.datasets.MNIST(
        root="datasets/", train=True, transform=transform, download=True
    )
    # loading test data
    test_dataset = torchvision.datasets.MNIST(
        root="datasets/", train=False, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    model = LogisticRegression(28 * 28, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # defining Cross-Entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    return "MNIST", model, train_loader, test_loader, criterion, optimizer


def main():
    torch.manual_seed(421)
    np.random.seed(421)
    nb_iter = 100  # Number of training steps
    batch_size = 2048
    nb_test = 1000  # Number of noises to consider for the statistics
    sigma = 0.225

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset_name, model, trainloader, testloader, criterion, optimizer = init_cifar(batch_size)
    # dataset_name, model, trainloader, testloader, criterion, optimizer = init_movielens(
    #     batch_size
    # )
    # dataset_name, model, trainloader, testloader, criterion, optimizer = init_mnist(batch_size)

    for dataset_initializer in [init_movielens, init_mnist, init_cifar]:
        for nb_iter in [0, 1, 10, 100]:
            print("-" * 40 + f"\nStarting new configuration")
            dataset_name, model, trainloader, testloader, criterion, optimizer = (
                dataset_initializer(batch_size)
            )
            model.to(device)
            shapes, lens = generate_shapes(model)

            print(f"Training")
            train(
                model,
                nb_iter=nb_iter,
                criterion=criterion,
                trainloader=trainloader,
                optimizer=optimizer,
                device=device,
            )

            print(f"Testing")
            model_trained_params = serialized_model(model)
            for sigma in [0.225, 0.225 / 4, 2 * 0.225, 0.225 / 128]:
                model.load_state_dict(
                    deserialized_model(model_trained_params, model, shapes, lens)
                )
                test_1_gradient(
                    model,
                    nb_test=nb_test,
                    sigma=sigma,
                    criterion=criterion,
                    trainloader=trainloader,
                    dataset_name=dataset_name,
                    nb_training=nb_iter,
                    device=device,
                )


if __name__ == "__main__":
    main()
