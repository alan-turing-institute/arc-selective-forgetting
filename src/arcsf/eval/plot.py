import matplotlib.pyplot as plt
import numpy as np

from arcsf.eval.metrics import ecdf, ks_test


def plot_cdf(
    base_vals: dict,
    test_vals: dict,
    save_path: str,
    exp_name: str,
    split: str,
) -> None:
    """
    Function to plot and save CDF functions for model comparison

    Args:
        base_vals : values for the base model to be compared against
        test_vals : values for the test model to be compared
        save_path : relative path where the model is stored
        exp_name : the name of the experiment for naming purposes
        split (optional): the split to be used. Defaults to "retain".
    """
    base_data, base_y_values = ecdf(base_vals[f"{split}_tr"])
    test_data, test_y_values = ecdf(test_vals[f"{split}_tr"])

    p_val = ks_test(test_vals[f"{split}_tr"], base_vals[f"{split}_tr"])

    plt.title(f"{split} data CDF")
    plt.plot(base_data, base_y_values, label="base-model")
    plt.plot(test_data, test_y_values, label=f"forget-model {exp_name}")
    plt.annotate(f"Model Utility {np.log(p_val)}", (0.9, 0.1))
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(save_path + f"/eval/analysis/tr-cdf-{split}-{exp_name}-.pdf")
    plt.close()


def plot_scatter(dict: dict[float], **plot_kwargs) -> None:
    """
    Plot of the data from a given model/experiment

    Args:
        dict : dictionary containing the aggrevated evaluation metrics of the model.
        plot_kwargs : key word arguments for the `matplotlib.pylot.scatter` function.
    """
    plt.scatter(dict["model_utility"], dict["forget_quality_1"], **plot_kwargs)
