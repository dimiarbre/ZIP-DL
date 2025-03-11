import heapq
from typing import Optional

import load_experiments
import pandas as pd
import torch
from load_experiments import ALL_ATTACKS
from RocPlotter import RocPlotter
from sklearn.metrics import roc_auc_score, roc_curve

ALL_TOPK_ATTACKS = [1, 5, 10, 100, 250, 500]


def get_biased_threshold_acc(train_losses, test_losses, top_kth: list[int]):

    max_k = max(top_kth)
    paired_train = [(loss, 1) for loss in train_losses]
    paired_test = [(loss, 0) for loss in test_losses]

    merged_list = paired_train + paired_test

    if max_k > len(paired_train):
        print(
            f"---------\nWARNING: using window of size {max_k} when the training set only has {len(paired_train)} elements!\n---------"
        )
    # Remove elements when the window is bigger than the
    fixed_top_kth = [k for k in top_kth if k <= len(merged_list)]

    max_k = max(fixed_top_kth)

    lowest_losses_max_k = heapq.nsmallest(max_k, merged_list)

    res = {}
    for k in top_kth:
        window = lowest_losses_max_k[:k]
        nb_success = sum([origin for (_, origin) in window])
        success_rate = nb_success / k
        assert success_rate <= 1
        res[f"top{k}_acc"] = success_rate

    return res


def threshold_attack(
    local_train_losses,
    test_losses,
    balanced=False,
    plotter: Optional[RocPlotter] = None,
):
    # We need the losses to be increasing the more likely we are to be in the train set,
    # so 1-loss should be given as argument.
    num_true = local_train_losses.shape[0]
    if num_true == 0:
        return {
            # "fpr": fpr,
            # "tpr": tpr,
            # "thresholds": thresholds,
            "roc_auc": torch.nan,
            "gini_auc": torch.nan,
        }
    # print("Number of training samples: ", num_true)

    # Had to remove an hard assertion for per-class data split.
    if num_true > test_losses.shape[0]:
        print(
            f"Not enough test elements: {test_losses.shape[0]} when at least {num_true} where expected"
        )
        # TODO: should we remove some train elements in the case of a biaised attack?
        if balanced:
            num_true = test_losses.shape[0]

    if balanced:
        y_true = torch.ones((num_true + num_true,), dtype=torch.int32)
        y_pred = torch.zeros((num_true + num_true,), dtype=torch.float32)
    else:
        y_true = torch.ones((num_true + test_losses.shape[0]), dtype=torch.int32)
        y_pred = torch.zeros((num_true + test_losses.shape[0]), dtype=torch.float32)

    y_true[:num_true] = 0
    y_pred[:num_true] = local_train_losses

    if balanced:
        y_pred[num_true:] = test_losses[torch.randperm(test_losses.shape[0])[:num_true]]
    else:
        y_pred[num_true:] = test_losses

    # Use the balanced y_pred for the ROC curve
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    # print("Shapes: ", y_pred.shape, y_true.shape)

    roc_auc = roc_auc_score(y_true, y_pred)

    gini_auc = 2 * roc_auc - 1

    if plotter is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

        plotter.plot_all(
            fpr,
            tpr,
            thresholds,
            roc_auc,
            losses_train=y_pred[:num_true],
            losses_test=y_pred[num_true:],
        )

    res = {
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
        "roc_auc": roc_auc,
        "gini_auc": gini_auc,
    }
    return res


def run_threshold_attack(
    running_model,
    model_path,
    trainset,
    testset,
    shapes,
    lens,
    device=torch.device("cpu"),
    debug=False,
    debug_name="",
    attack="threshold",
    loss_training: torch.nn.Module = torch.nn.CrossEntropyLoss(reduction="none"),
    plotter_unbalanced: Optional[RocPlotter] = None,
    plotter_balanced: Optional[RocPlotter] = None,
    classes_plotters: list[RocPlotter] = [],
):
    both_attacks = False
    default_res_threshold = {
        "roc_auc": torch.nan,
        "roc_auc_balanced": torch.nan,
    }
    default_res_biasedthreshold = {f"top{k}-acc": torch.nan for k in ALL_TOPK_ATTACKS}
    if "+" in attack:
        assert attack == "threshold+biasedthreshold"
        both_attacks = True
    elif attack == "threshold" or attack == "biasedthreshold":
        both_attacks = False
    else:
        raise ValueError(f"Unknown attack {attack}")

    # Load data and generate losses
    # Generate the train losses
    load_experiments.load_model_from_path(
        model_path=model_path,
        model=running_model,
        shapes=shapes,
        lens=lens,
        device=device,
    )
    losses_train, classes_train, acc_train = load_experiments.generate_losses(
        running_model, trainset, device=device, loss_function=loss_training, debug=debug
    )
    # Remove nans, usefull for RESNET + FEMNIST at least.
    losses_train, classes_train = load_experiments.filter_nans(
        losses=losses_train,
        classes=classes_train,
        debug_name=debug_name,
        loss_type="train set",
    )

    # Generate the test losses
    losses_test, classes_test, acc_test = load_experiments.generate_losses(
        running_model, testset, device=device, loss_function=loss_training, debug=debug
    )
    # Remove nans, usefull for RESNET + FEMNIST at least.
    losses_test, classes_test = load_experiments.filter_nans(
        losses=losses_test,
        classes=classes_test,
        debug_name=debug_name,
        loss_type="test set",
    )
    if losses_test.isnan().any():
        losses_test_nonan = losses_test[~losses_test.isnan()]
        percent_fail = (
            (len(losses_test) - len(losses_test_nonan)) / len(losses_test) * 100
        )
        print(
            f"{debug_name} - Found NaNs in test loss! Removed {percent_fail:.2f}% of test losses"
        )
        losses_test = losses_test_nonan
        classes_test = classes_test[~losses_test.isnan()]
    if len(losses_test) == 0 or len(losses_train) == 0:
        # print(
        #     "Found a losses tensor of size 0, found lengths -"
        #     + f" train:{len(losses_train)} - test:{len(losses_test)}"
        # )

        return {
            "threshold": pd.DataFrame(default_res_threshold, index=[0]),
            "biasedthreshold": pd.DataFrame(default_res_biasedthreshold, index=[0]),
        }
    if acc_train is not None and acc_test is not None:
        print(
            f"{debug_name} - Train accuracy {acc_train*100:.2f}%, Test accuracy {acc_test*100:.2f}%"
        )

    if debug:
        print(
            f"Train losses - avg:{losses_train.mean()}, std:{losses_train.std()}, min:{losses_train.min()}, max:{losses_train.max()}."
        )
        print(
            f"Test losses - avg:{losses_test.mean()}, std:{losses_test.std()}, min:{losses_test.min()}, max:{losses_test.max()}."
        )
    res_threshold = {}
    if both_attacks or attack == "threshold":
        threshold_result_unbalanced = threshold_attack(
            losses_train, losses_test, balanced=False, plotter=plotter_unbalanced
        )
        res_threshold["roc_auc"] = threshold_result_unbalanced["roc_auc"]
        res_threshold["gini_auc"] = threshold_result_unbalanced["gini_auc"]

        threshold_result_balanced = threshold_attack(
            losses_train, losses_test, balanced=True, plotter=plotter_balanced
        )
        res_threshold["roc_auc_balanced"] = threshold_result_balanced["roc_auc"]
        res_threshold["gini_auc_balanced"] = threshold_result_balanced["gini_auc"]

        # Do threshold attack per class
        nb_classes = int(torch.max(classes_train.max(), classes_test.max()).item()) + 1
        for class_id in range(nb_classes):
            if class_id < len(classes_plotters):
                current_class_plotter = classes_plotters[class_id]
            else:
                current_class_plotter = None
            current_class_train_losses = losses_train[
                classes_train == torch.tensor(class_id)
            ]
            current_class_test_losses = losses_test[
                classes_test == torch.tensor(class_id)
            ]

            current_class_threshold = threshold_attack(
                current_class_train_losses,
                current_class_test_losses,
                balanced=False,
                plotter=current_class_plotter,
            )

            res_threshold[f"roc_auc_class{class_id}"] = current_class_threshold[
                "roc_auc"
            ]
            res_threshold[f"gini_auc_class{class_id}"] = current_class_threshold[
                "gini_auc"
            ]

    res_biasedthreshold = {}
    if both_attacks or attack == "biasedthreshold":
        res_biasedthreshold = get_biased_threshold_acc(
            train_losses=losses_train,
            test_losses=losses_test,
            top_kth=ALL_TOPK_ATTACKS,
        )
        # TODO: make a biasedthreshold on the balanced losses?
        # TODO: investigate the small success of this attack.

    return {
        "threshold": pd.DataFrame(res_threshold, index=[0]),
        "biasedthreshold": pd.DataFrame(res_biasedthreshold, index=[0]),
    }
