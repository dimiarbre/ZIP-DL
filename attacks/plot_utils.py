import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plot_loaders
import scienceplots
import seaborn as sns

# plt.style.use(["science"])

# Attributes that will always be needed to export for the pgfplots.
EXPORT_ATTRIBUTES = [
    "noise_level_value",
    "log_noise",
    "variant",
    "noise_level",
    "avgsteps",
    "low_noise_mult",
    "seed",
]

ATTRIBUTE_DICT = {
    "topology_type": ["static", "dynamic"],
    "variant": ["nonoise", "muffliato", "zerosum", "anticor"],
    # "avgsteps": ["20avgsteps", "15avgsteps", "10avgsteps", "5avgsteps", "1avgsteps"],
    "additional_attribute": ["selfnoise", "noselfnoise", "nonoise", "muffliato"],
    "noise_level": [
        "nonoise",
        "0p25th",
        "0p5th",
        "0p75th",
        "1th",
        "2th",
        "2p5th",
        "3th",
        "3p5th",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "16th",
        "32th",
        "64th",
        "128th",
    ],
    "lr": ["lr0.05", "lr0.01", "lr0.075", "lr0.10", "lr0.5"],
    "local_rounds": [
        "1rounds",
        "2rounds",
        "3rounds",
        "5rounds",
        "10rounds",
        "20rounds",
    ],
    "batch_size": ["batchsize64", "batchsize512", "batchsize1024", "batchsize2048"],
    "seed": [f"seed{val}" for val in range(90, 106)],
    "model": ["LeNet", "CNN", "RNET"],
}


def extend_attributes_for_plot(attributes):
    for bonus_attribute in EXPORT_ATTRIBUTES:
        if bonus_attribute not in attributes:
            attributes.append(bonus_attribute)
    return attributes


def percentile(n):
    """Function to create the percentile aggregator.
    Taken from https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function

    Args:
        n (float): The percentage of the percentile. Can be 0.5,0.95, ...
    """

    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = "percentile_{:02.0f}".format(100 * n)
    return percentile_


def get_attributes(filename, attribute_dict=ATTRIBUTE_DICT):
    parsed_filename = filename.split("_")
    res_attribute = {}
    for global_attribute, possible_values in attribute_dict.items():
        for current_attribute in parsed_filename:
            if current_attribute in possible_values:
                if current_attribute not in res_attribute:
                    res_attribute[global_attribute] = current_attribute
                else:
                    raise RuntimeError(f"Duplicate of attributes in {filename}")

        if global_attribute not in res_attribute:
            # print(f"No value found for attribute {global_attribute} for {filename}")
            res_attribute[global_attribute] = None
    return res_attribute


def get_experiments_dict(names, attribute_dict=ATTRIBUTE_DICT):
    experiment_dict = {}
    for filename in names:
        experiment_dict[filename] = get_attributes(filename, attribute_dict)
    return experiment_dict


def get_full_path_dict(experiments_dir):
    files_list = os.listdir(experiments_dir)
    full_path_dict = {}
    for experiment in files_list:
        full_path_dict[experiment] = os.path.join(experiments_dir, experiment)
    return full_path_dict


def check_attribute(experiment_attributes, attributes_to_check):
    for attribute, expected_values in attributes_to_check.items():
        if experiment_attributes[attribute] not in expected_values:
            return False
    return True


def filter_attribute(experiments_attributes, attributes_to_check):
    res = []
    for experiment_name, experiment_attribute in sorted(experiments_attributes.items()):
        matches_attributes = check_attribute(experiment_attribute, attributes_to_check)
        if matches_attributes:
            res.append(experiment_name)
    return sorted(res)


def filter_attribute_list(experiment_attributes, attributes_to_check_list):
    res = []
    for attributes_to_check in attributes_to_check_list:
        res = res + filter_attribute(
            experiments_attributes=experiment_attributes,
            attributes_to_check=attributes_to_check,
        )
    return res


def get_style(current_attributes, mapping, option_name):
    res = None
    for attribute in current_attributes:
        attr_to_consider = current_attributes[attribute]
        if attr_to_consider in mapping:
            if res is None:
                res = mapping[current_attributes[attribute]]
            else:
                raise RuntimeError(
                    f"Two options found for {option_name} and mapping {mapping}:\n",
                    f"{res} and {mapping[current_attributes[attribute]]}",
                )
    return res


def plot_evolution_iterations(
    data,
    name,
    column_name="test_acc mean",
    current_attributes=None,
    attribute_mapping=None,
):
    print(data.columns)
    data_to_consider = data[column_name].dropna()
    if current_attributes is None or attribute_mapping is None:
        plt.plot(data_to_consider.index, data_to_consider, label=name)
        return
    color = get_style(current_attributes, attribute_mapping["color"], "color")
    linestyle = get_style(
        current_attributes, attribute_mapping["linestyle"], "linestyle"
    )
    linewidth = get_style(
        current_attributes, attribute_mapping["linewidth"], "linewidth"
    )
    plt.plot(
        data_to_consider.index,
        data_to_consider,
        label=name,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def get_attributes_columns(name, display_attributes, data_to_plot, orderings):
    if name in display_attributes:
        column_labels = display_attributes[name]
        if isinstance(column_labels, list):
            return data_to_plot[column_labels].apply(tuple, axis=1), None
        elif isinstance(column_labels, str):
            if orderings is not None and column_labels in orderings:
                return column_labels, orderings[column_labels]
            return column_labels, None
    return None, None


def select_attributes_to_keep(display_attributes, x_axis_name) -> list[str]:
    attributes_to_keep = [] if x_axis_name == "iteration" else [x_axis_name]
    if display_attributes is not None:
        for _, item in display_attributes.items():
            if isinstance(item, list):
                for attribute in item:
                    if attribute not in attributes_to_keep:
                        attributes_to_keep.append(attribute)
            elif item not in attributes_to_keep:
                attributes_to_keep.append(item)
    return attributes_to_keep


def all_equals(l):
    if len(l) == 0:
        return True
    element = l[0]
    for i in range(1, len(l)):
        if l[i] != element:
            return False
    return True


def gather_data(
    data: dict[str, pd.DataFrame],
    experiments: list[str],
    column_name: str,
    attributes_to_keep: list[str],
    save_directory: Optional[str],
) -> pd.DataFrame:
    """
    Gathers and processes experiment data for plotting.

    This function extracts specified columns from multiple experiments,
    drops any rows with missing values, and saves the resulting data to
    CSV files. It checks for the uniqueness of specified attributes across
    experiments and provides feedback if any experiment results are empty.

    Parameters:
        data (dict[str, pd.DataFrame]): A dictionary of DataFrames where keys are experiment names.
        experiments (list[str]): A list of experiment names to process.
        column_name (str): The name of the primary column to extract.
        attributes_to_keep (list[str]): A list of additional attributes to keep.
        save_directory (str): The directory path where CSV files will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated results
        from all processed experiments, with duplicate rows removed.
    """
    data_to_plot = pd.DataFrame({})

    for experiment in experiments:
        data_experiment_before_dropna = data[experiment][
            [column_name] + attributes_to_keep
        ]
        data_experiment = data_experiment_before_dropna.dropna()
        data_experiment = data_experiment.drop_duplicates()
        if data_experiment.empty:
            print(
                f"Found empty results for {experiment}. Experiment state: {data_experiment_before_dropna}."
            )
            _ = input("Input anything to resume.")

        experiment_attributes = {}
        # TODO: Avoid hardcoding this list.
        for attribute in ["variant", "noise_level", "avgsteps"]:
            assert (
                data_experiment[attribute].nunique() == 1
            ), f"Found {data_experiment[attribute].nunique()} values for {attribute} in experiment {experiment}, when there should be exactly 1."
            experiment_attributes[attribute] = data_experiment[attribute].iloc[0]
        # TODO: change this to a more sane filename? using, for instance, plot_loaders.EXPERIMENT_WIDE_ATTRIBUTES?
        if save_directory is not None:
            filename = f"{experiment_attributes['variant']}{experiment_attributes['noise_level']}_{experiment_attributes['avgsteps']}avgsteps"
            data_experiment.to_csv(
                f"{save_directory}plot_data/entire_experiment_data/{filename}.csv"
            )
        data_to_plot = pd.concat([data_to_plot, data_experiment])
    return data_to_plot


def plot_all_experiments(
    data: dict[str, pd.DataFrame],
    experiments,
    display_attributes,
    plot_name,
    x_axis="iteration",
    column_name="test_acc mean",
    save_directory=None,
    orderings=None,
    baseline: Optional[dict[str, pd.DataFrame]] = None,
):
    attributes_to_keep = plot_loaders.select_attributes_to_keep(
        display_attributes, x_axis
    )
    attributes_to_keep = extend_attributes_for_plot(attributes_to_keep)

    data_to_plot = gather_data(
        data, experiments, column_name, attributes_to_keep, save_directory
    )

    col, col_ordering = plot_loaders.get_attributes_columns(
        "col", display_attributes, data_to_plot, orderings
    )

    # Dupplicate the baseline along the column
    # TODO: do the same trick along rows if they start being used.
    if baseline is not None and "col" in display_attributes:
        col_attribute = display_attributes["col"]
        col_levels = data_to_plot[col_attribute].unique()

        data_baseline = gather_data(
            baseline,
            list(baseline.keys()),
            column_name,
            attributes_to_keep,
            save_directory,
        )
        baseline_replicated = pd.DataFrame({})
        for col_level in col_levels:
            current_baseline = data_baseline.copy()
            current_baseline[col_attribute] = col_level
            baseline_replicated = pd.concat([baseline_replicated, current_baseline])
        data_to_plot = pd.concat([data_to_plot, baseline_replicated])

    hue, hue_ordering = plot_loaders.get_attributes_columns(
        "hue", display_attributes, data_to_plot, orderings
    )
    style, style_ordering = plot_loaders.get_attributes_columns(
        "style", display_attributes, data_to_plot, orderings
    )
    size, size_ordering = plot_loaders.get_attributes_columns(
        "size", display_attributes, data_to_plot, orderings
    )

    plot = sns.relplot(
        data=data_to_plot,
        kind="line",
        x=x_axis,
        y=column_name,
        hue=hue,
        hue_order=hue_ordering,
        style=style,
        style_order=style_ordering,
        size=size,
        size_order=size_ordering,
        col=col,
        col_order=col_ordering,
        sizes=(1, 5),
    )
    plot.figure.suptitle(plot_name)
    plot.figure.subplots_adjust(top=0.9)
    is_unique = plot_loaders.all_equals(
        [
            attribute_value
            for attribute, attribute_value in display_attributes.items()
            if attribute != "col"
        ]
    )
    # if is_unique:
    #     axes = plot.axes
    #     handles, current_legend = axes[0, 0].get_legend_handles_labels()
    #     # plot._legend.remove()
    #     new_legend = []
    #     for element in current_legend:
    #         current_name = ""
    #         for tuple_element in element:
    #             current_name += f" {tuple_element}"
    #         new_legend.append(current_name)
    #     print(new_legend)
    #     plot.fig.legend(handles, labels=new_legend, title="Method")
    if save_directory is not None:
        filename = plot_name.replace(" ", "_")
        data_to_plot.to_csv(
            f"{save_directory}plot_data/entire_experiment_data/{filename}.csv"
        )
        savefile = f"{save_directory}{filename}.pdf"
        print(f"Saving to {savefile}")
        plt.savefig(savefile)
    return plot


def scatter_all_experiments(
    data: dict[str, pd.DataFrame],
    experiments,
    display_attributes,
    plot_name,
    x_axis_name="iteration",
    column_name="test_acc mean",
    save_directory=None,
    orderings=None,
):
    attributes_to_keep = plot_loaders.select_attributes_to_keep(
        display_attributes=display_attributes, x_axis_name=x_axis_name
    )
    data_to_plot = pd.DataFrame({})
    for experiment in experiments:
        data_experiment = data[experiment][[column_name] + attributes_to_keep]
        data_experiment = data_experiment.dropna()
        data_to_plot = pd.concat([data_to_plot, data_experiment])
    data_to_plot = gather_data(
        data=data,
        experiments=experiments,
        column_name=column_name,
        attributes_to_keep=attributes_to_keep,
        save_directory=save_directory,
    )
    # sns.set_theme()
    hue, hue_ordering = plot_loaders.get_attributes_columns(
        "hue", display_attributes, data_to_plot, orderings
    )
    # style, style_ordering = get_attributes_columns(
    #     "style", display_attributes, data_to_plot, orderings
    # )
    # size, size_ordering = get_attributes_columns(
    #     "size", display_attributes, data_to_plot, orderings
    # )
    # col, col_ordering = get_attributes_columns(
    #     "col", display_attributes, data_to_plot, orderings
    # )

    plot = sns.jointplot(
        data=data_to_plot,
        x=x_axis_name,
        y=column_name,
        hue=hue,
        hue_order=hue_ordering,
        # style=style,
        # style_order=style_ordering,
        # size=size,
        # size_order=size_ordering,
        # col=col,
        # col_order=col_ordering,
        # sizes=(5, 10),
    )
    plot.figure.suptitle(plot_name)
    plot.figure.subplots_adjust(top=0.9)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name.replace(' ','_')}.pdf"
        print(f"Saving to {savefile}")
        plt.savefig(savefile)
    return


# Function to draw an arrow from bottom right to top left
def draw_arrow(ax, xmin, xmax, ymin, ymax, arrow_ratio):
    arrow_start = np.array([xmax, ymin])  # Bottom right corner
    arrow_end = np.array([xmin, ymax])  # Top left corner

    true_arrow_end = arrow_start + (arrow_end - arrow_start) * arrow_ratio

    # Draw the arrow
    ax.annotate(
        "Better",
        true_arrow_end,
        arrow_start,
        arrowprops=dict(
            # head_width=3 * xy_ratio,
            # head_length=0.01 / xy_ratio,
            fc="black",
            ec="black",
        ),
    )


def scatter_averaged_experiments(
    data: dict[str, pd.DataFrame],
    experiments: list[str],
    display_attributes,
    plot_name,
    y_axis_name="test_acc mean",
    x_axis_name="iteration",
    save_directory=None,
    x_method="mean",
    y_method="mean",
    orderings=None,
    name_formater={},
    arrow_position="bottom right",
):
    if x_axis_name == "iteration":
        raise ValueError("Shouldn't scatter averaged data on iterations")
    true_x_axis = x_axis_name + " " + x_method
    true_y_axis = y_axis_name + " " + y_method
    data_to_plot = pd.DataFrame({})
    attributes_to_keep = plot_loaders.select_attributes_to_keep(
        display_attributes, x_axis_name
    )
    top_acc_axis_name = x_axis_name + " mean to max " + y_axis_name

    attributes_to_keep = extend_attributes_for_plot(attributes_to_keep)
    attributes_to_keep.remove(x_axis_name)
    for experiment in sorted(experiments):
        experiment_data: pd.DataFrame = data[experiment]
        if experiment_data.empty:
            print(f"Got empty data for {experiment}, probably filtered beforehand.")
            continue

        experiment_data = experiment_data[
            [x_axis_name, y_axis_name, "experience_name"] + attributes_to_keep
        ]
        experiment_data = experiment_data.drop_duplicates()
        if experiment_data.index.duplicated().any():
            raise ValueError(
                f"Got duplicate indexes after filtering {experiment}- consider filtering by attack results (classifier_attack properties)"
            )
        if experiment_data.empty:
            raise ValueError(
                f"Got empty experiment data for {experiment}. Check the desired columns."
            )
        # experiment_data = experiment_data.dropna()

        # For debugging purposes
        aggregated_loss = experiment_data[y_axis_name]
        max_aggregated_loss = experiment_data[y_axis_name].max()

        experiment_data = experiment_data.sort_values("iteration")
        matching_max = experiment_data[
            experiment_data[y_axis_name] == max_aggregated_loss
        ]

        max_y = matching_max.index[0]
        x_mean_to_max_y = experiment_data[x_axis_name].loc[0:max_y].mean()

        experiment_data[top_acc_axis_name] = x_mean_to_max_y

        data_to_plot = pd.concat([data_to_plot, experiment_data])
    data_groups = data_to_plot.groupby(
        by=["experience_name", top_acc_axis_name] + attributes_to_keep
    )

    def last(series):
        return series.dropna().iloc[-1]

    data_to_plot = data_groups.agg(["max", "mean", "min", last]).reset_index()

    data_to_plot.columns = [
        " ".join(e) if len(e[-1]) > 0 else e[0] for e in data_to_plot.columns
    ]
    data_to_plot.set_index("noise_level_value")
    data_to_plot.sort_values(
        "noise_level_value",
        inplace=True,
        # key=lambda x: pd.Series([(len(i), i) for i in x]),
    )
    # core_data = data_to_plot[["noise_level_value", "variant", true_x_axis, true_y_axis]]
    # core_data = core_data[core_data["variant"] == "zerosum"]
    # print(data_to_plot)

    # sns.set_theme()
    hue, hue_ordering = plot_loaders.get_attributes_columns(
        "hue", display_attributes, data_to_plot, orderings
    )
    style, style_ordering = plot_loaders.get_attributes_columns(
        "style", display_attributes, data_to_plot, orderings
    )
    size, size_ordering = plot_loaders.get_attributes_columns(
        "size", display_attributes, data_to_plot, orderings
    )
    # col, col_ordering = get_attributes_columns(
    #     "col", display_attributes, data_to_plot, orderings
    # )

    ax = sns.lineplot(
        data=data_to_plot,
        x=true_x_axis,
        y=true_y_axis,
        hue=hue,
        hue_order=hue_ordering,
        style=style,
        style_order=style_ordering,
        size=size,
        size_order=size_ordering,
        # col=col,
        # col_order=col_ordering,
        markers=True,
        # sizes=(3, 5),
        sort=False,
    )
    plt.title(plot_name)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    if arrow_position == "bottom right":
        draw_arrow(ax, xmin, xmax, ymin, ymax, 0.1)
    elif arrow_position == "top right":
        draw_arrow(ax, xmin, xmax, ymax, ymin, 0.1)
    else:
        raise NotImplementedError(
            f"Unimplemented 'arrow_position', should be either 'bottom right' or 'top right' for now."
        )

    # plot.fig.subplots_adjust(top=0.9)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name}.pdf".replace(" ", "_").replace(
            "\n", ""
        )
        datafile = os.path.join(
            save_directory,
            "plot_data",
            f"{plot_name}.csv".replace(" ", "_").replace("\n", ""),
        )
        data_to_plot.to_csv(datafile)
        fig = plt.gcf()
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted()
        )
        print(f"Saving to {savefile}")
        plt.savefig(savefile, bbox_inches=extent)
    return


def fix_linkability_attack_results(experiment_name, attack_results_path):
    expected_file_name = f"linkability_{experiment_name}.csv"
    directories = sorted(os.listdir(attack_results_path))
    if expected_file_name not in directories:
        print(
            f"Not loading attack results: {expected_file_name} was not listed with attack results in {attack_results_path}. Entire directory:\n{directories}"
        )
        raise FileNotFoundError(expected_file_name)

    linkability_attack_df = pd.read_csv(
        os.path.join(attack_results_path, expected_file_name)
    )

    # Fixing all the missing values/wrongly filled values
    linkability_attack_df["linkability_top1"] = (
        linkability_attack_df["linkability_top1_guess"]
        == linkability_attack_df["agent"]
    )

    linkability_attack_df.reset_index(drop=True)
    linkability_attack_df.set_index(["agent", "iteration"])
    columns = linkability_attack_df.columns.to_list()
    columns_losses = [column for column in columns if "loss_trainset_" in column]

    linkability_attack_df["linkability_real_rank"] = np.nan
    linkability_attack_df["linkability_real_rank"].astype("Int64", copy=False)
    for index, row in linkability_attack_df.iterrows():
        losses = [(int(column.split("_")[2]), row[column]) for column in columns_losses]
        losses_sorted = sorted(losses, key=lambda x: x[1])
        agents_sorted = [x[0] for x in losses_sorted]
        linkability_rank = agents_sorted.index(row["agent"])
        linkability_attack_df.at[index, "linkability_real_rank"] = linkability_rank

    linkability_attack_df = linkability_attack_df.drop(columns="Unnamed: 0")
    linkability_attack_df.to_csv(
        os.path.join(attack_results_path, f"fixed_{expected_file_name}")
    )

    return linkability_attack_df


def plot_communication(
    data: dict[str, pd.DataFrame],
    experiments,
    target_acc,
    plot_name,
    save_directory,
    accuracy_metric="test_acc mean",
    order=None,
):
    y_axis_name = "total_bytes sum"
    data_to_plot = pd.DataFrame({})
    reference = None
    for experiment_name in experiments:
        experiment_data: pd.DataFrame = data[experiment_name]
        experiment_data = experiment_data[
            [
                y_axis_name,
                "noise_level",
                "log_noise",
                "noise_level_value",
                "variant",
                "test_acc mean",
                "test_loss mean",
                "topology_type",
                "seed",
            ]
        ]
        if "acc" in accuracy_metric:
            restrained_acc = experiment_data[
                experiment_data[accuracy_metric] >= target_acc
            ]
        elif "loss" in accuracy_metric:
            restrained_acc = experiment_data[
                experiment_data[accuracy_metric] <= target_acc
            ]
        correct_accuracies = restrained_acc.dropna().drop_duplicates()
        if len(correct_accuracies.index) == 0:
            iteration_accuracy_reached = (
                experiment_data[accuracy_metric].dropna().index[-1]
            )
            print(
                f"Target accuracy of {target_acc} {accuracy_metric} not reached by {experiment_name}. Defaulting to {iteration_accuracy_reached}."
            )
            experiment_result = experiment_data.loc[[iteration_accuracy_reached]]
            experiment_result[y_axis_name] = np.nan
        else:
            iteration_accuracy_reached = correct_accuracies.index[0]
            experiment_result = experiment_data.loc[[iteration_accuracy_reached]]
        current_variant = experiment_result["variant"].values[0]
        if current_variant == "nonoise":
            print(f"Found reference {experiment_name}")
            reference = experiment_result
        experiment_result = experiment_result.reset_index()
        best_test_acc = experiment_data["test_acc mean"].max()
        best_test_loss = experiment_data["test_loss mean"].min()
        experiment_result["test_acc mean max"] = best_test_acc
        experiment_result["test_loss mean min"] = best_test_loss
        experiment_result["experiment_name"] = experiment_name

        data_to_plot = pd.concat(
            [
                data_to_plot,
                pd.DataFrame(experiment_result),
            ]
        )
    assert reference is not None, "Should have found a reference"

    data_to_plot["log_" + y_axis_name] = np.log(data_to_plot[y_axis_name])
    reference["log_" + y_axis_name] = np.log(reference[y_axis_name])

    data_to_plot.reset_index(drop=True)

    data_to_plot = data_to_plot.sort_values(
        ["variant", "topology_type", "noise_level_value"], ascending=False
    )
    data_to_plot["low_noise_mult"] = data_to_plot["noise_level_value"] * 128 / 0.225
    # Fix to avoid 1.00000000000427
    data_to_plot.loc[data_to_plot["noise_level"] == "128th", "low_noise_mult"] = 1

    data_to_plot = data_to_plot.drop_duplicates()

    ax = sns.barplot(
        data=data_to_plot.dropna(),
        x="noise_level",
        y=y_axis_name,
        hue="variant",
        order=order,
    )

    # ax.axhline(reference[y_axis_name].values[0])

    plt.title(plot_name)

    # plot.fig.subplots_adjust(top=0.9)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name.replace(' ','_')}.pdf"
        data_to_plot.to_csv(
            os.path.join(
                save_directory, "plot_data", f"{plot_name.replace(' ','_')}.csv"
            )
        )
        # fig = plt.gcf()
        # extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        #     fig.dpi_scale_trans.inverted()
        # )
        # print(f"Saving to {savefile}")
        # plt.savefig(savefile, bbox_inches=extent)
    return
