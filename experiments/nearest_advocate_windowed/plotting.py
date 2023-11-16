"""Plotting utils for the windowed Nearest Advocate algorithm for the correction of
linear and non-linear clock-drifts in event-based time-delays."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')
sns.set_context('notebook')

# Load config for Nearest Advocate
import nearest_advocate
# pylint: disable=C0401, C0415
from .nearest_advocate_config import PATH_SYNCED, IMG_FORMAT, SEED
from .data_loader import read_r_peaks, read_j_peaks

# set the initial value for pseudo-random functions
np.random.seed(SEED)


# #######################################################
# #################### Plotting Utils ###################
# #######################################################


def plot_windowed_result(df_results: pd.DataFrame, pred_y: np.ndarray,
                         quantile_range: float = 0.9, save_fig: str = None,
                         verbose: int = 2) -> None:
    """
    Plot the windowed result of time delays.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame containing the results of the time delays.
    pred_y : np.ndarray
        Array with the predicted time delay should have the same shape as x_linspace.
    quantile_range : float, optional
        Quantile range for confidence interval, by default 0.90 (=90%).
    save_fig : str, optional
        Path to save the figure, by default None.
    verbose : int, optional
        Verbosity level, by default 2.
    """
    x_linspace = np.linspace(df_results["timestamp"].values[0], df_results["timestamp"].values[-1], len(pred_y))
    sns.scatterplot(data=df_results, x="timestamp", y="time_delta", color="steelblue",
                    label="Time delay", alpha=0.6, linewidth=0.0)
    sns.lineplot(x=x_linspace, y=pred_y, color="firebrick", label="Robust linear fit")

    try:
        # Add statistic
        ci_low, ci_high = np.quantile(df_results["time_delta"].values,
                                      ((1-quantile_range)/2, 1-(1-quantile_range)/2))
        sns.lineplot(x=x_linspace, y=ci_low, color="gray", linestyle='--',
                     label=f"{int(100*quantile_range)}% Quantile Range")
        sns.lineplot(x=x_linspace, y=ci_high, color="gray", linestyle='--')
        plt.text(max(x_linspace), ci_low - 0.1, f"{ci_low:.3f}", ha="right", va="center", color="gray")
        plt.text(max(x_linspace), ci_high + 0.08, f"{ci_high:.3f}", ha="right", va="center", color="gray")

        plt.xlabel("Timestamp of the Windowed Signal (s)")
        plt.ylabel("Time Delay (s)")
        plt.title("Time Delays by Windows")
        ylim = tuple(df_results["time_delta"].quantile([0.05, 0.95]).values + [-1.0, 1.0])
        plt.ylim(ylim)
        if save_fig:
            plt.savefig(save_fig, dpi=250)
        if verbose >= 1.5:
            plt.show()
        else:
            plt.close()
            plt.cla()
            plt.clf()

    except ValueError as value_exception:
        print("Exception while 'plot_windowed_result'", (str(value_exception)))


def double_check_sync(
        path_rpeaks: str, path_jpeaks: str, array_key: Optional[str] = None,
        scopes: List[str] = ["broad", "mid", "near"], td_min: float = -60,
        td_max: float = 60, td_mid: float = 30, td_near: float = 5,
        td_sps: float = 1.0, dist_max: float = 0.25, verbose: int = 5, save_dir: str = "",
        ) -> None:
    """
    Double-check the corrected version of the synchronization between R-peaks and J-peaks.

    Parameters
    ----------
    path_rpeaks : str
        Path to the sorted R-peak file assumed to be correct (ground truth).
    path_jpeaks : str
        Path to the sorted J-peak file from the BCG.
    array_key : str, optional
        Key to access the specific array, by default None.
    scopes : List[str], optional
        Scopes for the analysis, by default ["broad", "mid", "near"].
    td_min : float, optional
        Lower bound of the search space for the time-shift, by default -60 s.
    td_max : float, optional
        Upper bound of the search space for the time-shift, by default 60 s.
    td_mid : float, optional
        Middle bound for the time-shift, by default 30 s.
    td_near : float, optional
        Near bound for the time-shift, by default 5 s.
    td_sps : float, optional
        Time-shifts per second, by default 1.0.
    dist_max : float, optional
        Maximal accepted distances between two advocate events, by default 0.25 s.
    verbose : int, optional
        Verbosity level, by default 5.
    save_dir : str, optional
        Directory to save the figure, by default empty string.

    Returns
    -------
    None
        The function does not return any value but may save a figure if save_fig is True.
    """
    # load the r-peak file (ground truth)
    df_r = read_r_peaks(path_rpeaks=path_rpeaks)
    arr_r = df_r["time"].values

    # load the j-peak file from the BCG
    df_j, arr_j, _ = read_j_peaks(path_jpeaks=path_jpeaks, directory=PATH_SYNCED, array_key=array_key)

    # Correct the timestamp of arr_j
    if "nonlinear" in df_j.columns:
        title_suffix = " of non-linear corrected Signal"
        fig_suffix = "_nonlinear"
    elif "corrected" in df_j.columns:
        title_suffix = " of linear corrected Signal"
        fig_suffix = "_linear"
    else:
        title_suffix = ""
        fig_suffix = ""

    # Nearest Advocate
    df_nearest = pd.DataFrame(
        nearest_advocate.nearest_advocate(
            arr_r, arr_j, td_min=td_min, td_max=td_max,
            sparse_factor=1, sps=td_sps, dist_max=dist_max),
        columns=["time-delta", "distance"])
    td_hat = df_nearest.loc[np.argmin(df_nearest["distance"])]["time-delta"]
    df_away = df_nearest.loc[np.abs(df_nearest["time-delta"]-td_hat) > td_near]
    print(f"Found optimum at {td_hat:.4f} s, should be at 0.0 s.")
    print(f"Minimum is at {df_nearest['distance'].min():.6f}, others are {df_away['distance'].dropna().min():.6f}, ",
          f"( -> difference of {df_away['distance'].dropna().min()-df_nearest['distance'].min():.6f})")

    try:
        # Plot the broad (full) scope
        if "broad" in scopes:
            sns.lineplot(x=df_nearest["time-delta"], y=df_nearest["distance"], color="steelblue", label="Mean distance")
            td_hat = df_nearest.loc[np.argmin(df_nearest["distance"])]["time-delta"]
            plt.vlines(x=td_hat, ymin=0.9*df_nearest["distance"].min(), ymax=1.1*df_nearest["distance"].max(),
                       color="firebrick", label="Predicted Time Delay")
            plt.hlines(y=df_nearest["distance"].min(), xmin=td_min, xmax=td_max, color="black", label="Min. distance")
            plt.title(f"Nearest Advocate Synchronization{title_suffix}: Broad scope")
            plt.xlabel("Time delay (s)")
            plt.ylabel("Distance (s)")
            plt.legend(loc="upper right")
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"full_sync_broad{fig_suffix}.{IMG_FORMAT}"))
            if verbose >= 2:
                plt.show()
            else:
                plt.close()
                plt.cla()
                plt.clf()

        # Plot the mid scope
        if "mid" in scopes:
            sns.lineplot(x=df_nearest["time-delta"], y=df_nearest["distance"], color="steelblue", label="Mean distance")
            plt.vlines(x=td_hat, ymin=0.9*df_nearest["distance"].min(), ymax=1.1*df_nearest["distance"].max(),
                       color="firebrick", label="Predicted Time Delay")
            plt.hlines(y=df_nearest["distance"].min(), xmin=td_min, xmax=td_max, color="black", label="Min. distance")
            plt.title(f"Nearest Advocate Synchronization{title_suffix}: Mid scope")
            plt.xlim(td_hat-td_mid, td_hat+td_mid)
            plt.xlabel("Time delay (s)")
            plt.ylabel("Distance (s)")
            plt.legend(loc="upper right")
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"full_sync_mid{fig_suffix}.{IMG_FORMAT}"))
            if verbose >= 2:
                plt.show()
            else:
                plt.close()
                plt.cla()
                plt.clf()

        # Plot the near scope
        if "near" in scopes:
            sns.lineplot(x=df_nearest["time-delta"], y=df_nearest["distance"], color="steelblue", label="Mean distance")
            plt.vlines(x=td_hat, ymin=0.9*df_nearest["distance"].min(), ymax=1.1*df_nearest["distance"].max(),
                       color="firebrick", label="Predicted Time Delay")
            plt.hlines(y=df_nearest["distance"].min(), xmin=td_min, xmax=td_max, color="black", label="Min. distance")
            plt.title(f"Nearest Advocate Synchronization{title_suffix}: Near scope")
            plt.xlim(td_hat-td_near, td_hat+td_near)
            plt.xlabel("Time delay (s)")
            plt.ylabel("Distance (s)")
            plt.legend(loc="upper right")
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"full_sync_near{fig_suffix}.{IMG_FORMAT}"))
            if verbose >= 2:
                plt.show()
            else:
                plt.close()
                plt.cla()
                plt.clf()

    except ValueError as value_exception:
        print("Exception while 'double_check_sync'", (str(value_exception)))


def create_overlay_plots(df_r, df_j, n_plots=10, width=10, save_fig=None, verbose=2):
    """
    Create overlay plots with R- and J-peaks.

    Parameters:
    df_r (DataFrame):
        DataFrame containing R-peaks data with columns "time" and "rr".
    df_j (DataFrame):
        DataFrame containing J-peaks data with columns "time_corrected_nonlinear" and "jj".
    n_plots (int, optional):
        Number of plots to create. Default is 10.
    width (float, optional):
        Width of the time window for each plot in seconds. Default is 10.
    save_fig : bool, optional
        If True, save the figure, by default False.
    verbose : int, optional
        Verbosity level, by default 2.

    Returns
    -------
    None
        The function creates and displays the plots but does not return any value.
    """
    # recreate the JJ-intervals from the corected timestamps
    df_j["jj"].iloc[1:] = np.diff(df_j["time_corrected_nonlinear"])

    # pps = 10  # points per second

    # # Create the DataFrame
    # xvals = np.linspace(int(arr_j[0]),
    #                    int(arr_j[-1]),
    #                    pps * (int(arr_j[-1])-int(arr_j[0]))+1)

    # df = pd.DataFrame({"time": xvals})
    # df["IBI-ECG"] = np.interp(xvals, df_r["time"], df_r["rr"])
    # df["IBI-BCG"] = np.interp(xvals, df_j["time_corrected_nonlinear"], df_j["jj"])
    try:
        fig, axes = plt.subplots(n_plots, figsize=(8, 3*n_plots))
        fig.suptitle('Random windows with both IBIs')

        random_time = (df_j["time"].max() - df_j["time"].min()) / n_plots * np.random.random(1)[0] + df_j["time"].min()
        time_delta = (df_j["time"].max() - df_j["time"].min()) / n_plots

        for ax in axes:
            # df_ = df[np.logical_and(random_time - width/2 < df["time"], df["time"] < random_time + width/2)]
            df_r_ = df_r[np.logical_and(random_time - width/2 < df_r["time"], df_r["time"] < random_time + width/2)]
            df_j_ = df_j[np.logical_and(
                random_time - width/2 < df_j["time_corrected_nonlinear"],
                df_j["time_corrected_nonlinear"] < random_time + width/2)
                ]

            # # ECG and BCG lines
            # ax.plot(df_["time"], df_["IBI-ECG"], label="IBI-ECG")
            # ax.plot(df_["time"], df_["IBI-BCG"], label="IBI-BCG")

            # ECG and BCG IBIs and R(J)-peaks
            ax.plot(df_r_["time"], df_r_["rr"], marker="o", label="IBI-ECG")
            ax.plot(df_j_["time_corrected_nonlinear"], df_j_["jj"], marker="o", label="IBI-BCG")

            ax.set_ylabel("IBI (s)")
            random_time += time_delta

        # global plot settings
        axes[0].set_xlabel("Time (s)")
        axes[0].legend()
        if save_fig:
            plt.savefig(save_fig)
        if verbose >= 2:
            plt.show()
        else:
            plt.close()
            plt.cla()
            plt.clf()

    except ValueError as value_exception:
        print("Exception while 'create_overlay_plots'", (str(value_exception)))
