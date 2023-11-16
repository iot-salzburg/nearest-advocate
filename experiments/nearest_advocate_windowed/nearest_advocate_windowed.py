"""Code for the windowed Nearest Advocate algorithm for the correction of
linear and non-linear clock-drifts in event-based time-delays."""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TheilSenRegressor

try:
    # import only if module is used
    # pylint: disable=C0401, C0415
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    print("Error: Can't import torch, nonlinear fit won't be possible!")


# Load config and functions for Nearest Advocate
import nearest_advocate
# pylint: disable=C0402
from .data_loader import read_r_peaks, read_j_peaks, save_j_peaks_corrected
from .plotting import plot_windowed_result
from .nearest_advocate_config import TD_MIN, TD_MAX, TD_SPS
from .nearest_advocate_config import PATH_SYNCED, SEED

# Config for the MLP approximation of the non-linear signal
N_EPOCHS = 4_000
N_HIDDEN = 400

# set the initial value for pseudo-random functions
np.random.seed(SEED)
torch.manual_seed(SEED)


# #######################################################
# ############## Windowed Nearest Advocate ##############
# #######################################################


def nearest_advocate_windowed(
        arr_r, arr_j, td_min=TD_MIN, td_max=TD_MAX, td_sps=TD_SPS, sparse_factor=1, dist_max=0.25,
        frame_length=500, sliding_length=250, quantile_filter=0.25, bandwidth_filter=None, verbose=5):
    """Nearest Advocate over sliding windows.

    Apply the nearest advocate algorithm in sliding windows to capture
    deviations of time delays caused by a variable sample rates or clock-drift.

    Parameters
    ----------
    arr_r : np.ndarray
        R-peaks as 1D array assumed to be correct (ground truth).
    arr_j : np.ndarray
        J-peaks as 1D array from the BCG, assumed to be shifted by a variable time-delta
    sps_rel : float, optional
        Initial relative sample rate deviation factor, by default 1.0.
    td_min : float, optional
        Lower bound of the search space for the time-shift, by default -60 s.
    td_max : float, optional
        Upper bound of the search space for the time-shift, by default 60 s.
    td_sps : int, optional
        Number of investigated time-shifts per second. Default None: sets it at 10 divided by the median gap of
        each array.
    sparse_factor : int, optional
        Factor for the sparseness of signal array for the calculation, higher is faster at the cost of precision,
        by default 1.
    dist_max : float, optional
        Maximal accepted distances between two advocate events. Default 0.25 s
    frame_length : int, optional
        Length of the sliding window, by default 500 events.
    sliding_length : int, optional
        Sliding length for the window, by default 250 events.
    quantile_filter : int, optional
        Filter based on the Nearest Advocate algorithm for outlier removal, by default 0.25 to remove the 25%
        strongest outlier.
    bandwidth_filter : int, optional
        Bandwidth for the kernel density estimation for outlier removal, by default None for no KDE-based filtering.
    verbose : int, optional
        Verbosity level, by default 5.

    Returns
    -------
    df_results
        A pandas DataFrame with the nearest advocate algorithm's result for each window.
    """
    # define the window centers (with indices)
    window_centers = range(int(frame_length), int(len(arr_j)-frame_length/2), sliding_length)
    n_windows = len(window_centers)

    # results DataFrame
    df_results = pd.DataFrame({"index": np.zeros(n_windows, dtype=np.int32),
                               "timestamp": np.zeros(n_windows, dtype=np.float32),
                               "time_delta": np.zeros(n_windows, dtype=np.float32),
                               "td_hat_distance": np.zeros(n_windows, dtype=np.float32),
                               "background_min_distance": np.zeros(n_windows, dtype=np.float32),
                               "diff_distance": np.zeros(n_windows, dtype=np.float32)})
    min_distance = 1e9
    max_delta = 0.0
    for idx, slice_idx_center in enumerate(window_centers):
        # set the indices for the current slice
        slice_idx = np.arange(
            int(slice_idx_center - frame_length / 2),
            int(slice_idx_center + frame_length / 2))

        # slice the R-peak array such that J-peak is in the intervall for all possible time-deltas
        arr_r_slice = arr_r[np.logical_and(
            arr_r > arr_j[slice_idx[0]]+td_min-1,
            arr_r < arr_j[slice_idx[-1]]+td_max+1
            )]
        # skip if the array slice has less than two events
        if len(arr_r_slice) < 2:
            continue

        # Broad search using nearest advocate
        time_shifts = nearest_advocate.nearest_advocate(
            arr_ref=arr_r_slice, arr_sig=arr_j[slice_idx],
            td_min=td_min, td_max=td_max, sps=td_sps,
            dist_max=dist_max, sparse_factor=sparse_factor)
        df_res_broad = pd.DataFrame(time_shifts, columns=["time-delta", "distance"])
        # get curvature statistics
        td_hat, td_hat_distance = time_shifts[np.argmin(time_shifts[:, 1])]  # get the optimum
        background_min_distance = df_res_broad.loc[
            np.abs(df_res_broad["time-delta"]-td_hat) > 30, "distance"
            ].min()

        # check cumulative stats
        postfix1, postfix2 = "", ""
        if background_min_distance - td_hat_distance > max_delta:
            max_delta = background_min_distance - td_hat_distance
            postfix1 = "*"
        if td_hat_distance < min_distance:
            min_distance = td_hat_distance
            postfix2 = "+"

        # fill the results
        df_results.loc[idx, "index"] = slice_idx_center
        df_results.loc[idx, "timestamp"] = arr_j[slice_idx_center]
        df_results.loc[idx, "time_delta"] = td_hat
        df_results.loc[idx, "td_hat_distance"] = td_hat_distance
        df_results.loc[idx, "background_min_distance"] = background_min_distance
        df_results.loc[idx, "diff_distance"] = background_min_distance - td_hat_distance
        if verbose >= 3:
            print(f"Window around: {str(slice_idx_center).rjust(5)}: ",
                  f"({df_results.loc[idx, 'timestamp']:>8.2f}s), \t", end="")
            print(f"Minimum at {td_hat:>8.3f} s, score: {td_hat_distance:.6f} ", end="")
            print(f"   others: {background_min_distance:.6f}, diff: {background_min_distance-td_hat_distance:.6f}"
                  + postfix1 + postfix2)

    # filter out default value td_min
    df_results = df_results[df_results["time_delta"] > td_min + 1e-9]

    if quantile_filter and quantile_filter > 0.0:
        # filter on internal criterion measure
        df_results = df_results[
            df_results["diff_distance"] >= df_results["diff_distance"].quantile(quantile_filter)
            ]

    # filter on KDE cluster criterion
    if bandwidth_filter:
        kde = KernelDensity(
            kernel='cosine', bandwidth=bandwidth_filter
            ).fit(df_results[["time_delta"]].values)
        time_deltas = np.linspace(td_min, td_max, td_max-td_min+1).reshape(-1, 1)
        kde_opt = time_deltas[np.argmax(kde.score_samples(time_deltas))][0]
        df_results = df_results.loc[
            (kde_opt-bandwidth_filter < df_results["time_delta"])
            & (df_results["time_delta"] < kde_opt+bandwidth_filter)]

    return df_results


def nearest_advocate_windowed_linear(
        path_rpeaks, path_jpeaks, sps_rel=1.0,
        td_min=TD_MIN, td_max=TD_MAX, td_sps=TD_SPS, sparse_factor=1, dist_max=0.25,
        frame_length=500, sliding_length=250, quantile_filter=0.25, bandwidth_filter=None,
        quantile_range=0.9, verbose=5, save_fig=False):
    """Nearest Advocate over sliding windows.

    Apply the nearest advocate algorithm in sliding windows to capture
    deviations of time delays caused by a variable sample rates or clock-drift.
    A robust regression is applied to estimate the the time-delay and drift

    Parameters
    ----------
    path_rpeaks : str
        Path to the sorted r-peak file assumed to be correct (ground truth).
    path_jpeaks : str
        Path to the sorted j-peak file from the BCG, assumed to be shifted by a variable time-delta
    sps_rel : float, optional
        Initial relative sample rate deviation factor, by default 1.0.
    td_min : float, optional
        Lower bound of the search space for the time-shift, by default -60 s.
    td_max : float, optional
        Upper bound of the search space for the time-shift, by default 60 s.
    td_sps : int, optional
        Number of investigated time-shifts per second. Default None: sets it at 10 divided by the median gap of
        each array.
    sparse_factor : int, optional
        Factor for the sparseness of signal array for the calculation, higher is faster at the cost of precision,
        by default 1.
    dist_max : float, optional
        Maximal accepted distances between two advocate events. Default 0.25 s
    frame_length : int, optional
        Length of the sliding window, by default 500 events.
    sliding_length : int, optional
        Sliding length for the window, by default 250 events.
    quantile_filter : int, optional
        Filter based on the Nearest Advocate algorithm for outlier removal, by default 0.25 to remove the 25%
        strongest outlier.
    bandwidth_filter : int, optional
        Bandwidth for the kernel density estimation for outlier removal, by default None for no KDE-based filtering.
    verbose : int, optional
        Verbosity level, by default 5.
    save_fig : bool, optional
        If True, save the figure, by default False.

    Returns
    -------
    df_results
        A pandas DataFrame with the nearest advocate algorithm's result for each window.
    stats
        A dictionary with the robust linear regressions' results.
    """
    # load the r-peak file (ground truth)
    df_r = read_r_peaks(path_rpeaks=path_rpeaks)
    arr_r = df_r["time"].values

    # load the j-peak file from the BCG
    df_j, arr_j, column_key = read_j_peaks(path_jpeaks=path_jpeaks, directory=PATH_SYNCED,
                                           array_key=["time_corrected"])
    if verbose > 3:
        print(f"Length of df_r: {len(df_r)}, length of df_j: {len(df_j)}.")

    # Apply Nearest Advocated Algorithm in Windows
    df_results = nearest_advocate_windowed(
        arr_r, arr_j,
        td_min=td_min, td_max=td_max, td_sps=td_sps, sparse_factor=sparse_factor, dist_max=dist_max,
        frame_length=frame_length, sliding_length=sliding_length,
        bandwidth_filter=bandwidth_filter, quantile_filter=quantile_filter, verbose=verbose)
    if verbose > 3:
        print(f"Length of df_results: {df_results.shape[0]}.")
    assert len(df_results) > 10  # For the regression there must be multiple results in df_results

    # Estimate a robust linear regression
    regressor = TheilSenRegressor(random_state=1)
    regressor.fit(X=df_results[["timestamp"]].values, y=df_results["time_delta"].values)
    x_linspace = np.linspace(df_results["timestamp"].values[0], df_results["timestamp"].values[-1],
                             int(len(df_results)/3))
    pred_y = regressor.predict(x_linspace.reshape(-1, 1))

    # Add statistic
    ci_low, ci_high = np.quantile(df_results["time_delta"].values,
                                  ((1-quantile_range)/2, 1-(1-quantile_range)/2))

    # Plot the resulting regression and estimations
    if verbose >= 1.5 or save_fig:
        plot_windowed_result(df_results, pred_y=pred_y, quantile_range=quantile_range,
                             save_fig=save_fig, verbose=verbose)

    # correct with the estimated time-delta and sps
    stats = {
        "td_pred_head": pred_y[0],
        "td_pred_tail": pred_y[-1],
        "td_pred_median": np.median(pred_y),
        "td_preds":  list(pred_y),
        "regressor": regressor,
        "sps_pred": sps_rel - sps_rel * regressor.coef_[0],
        "td_quantile_range": ci_high-ci_low
    }

    # Correct the timestamps and save in path for synced files
    # df_j = read_j_peaks(file_j_peaks=path_jpeaks, sps_rel=stats['sps_pred'])
    df_j["time_corrected"] = df_j["time_corrected"] * (1 - regressor.coef_[0])
    df_j["time_corrected"] = df_j["time_corrected"] - stats["td_pred_head"]
    save_j_peaks_corrected(df_j=df_j, path_jpeaks=path_jpeaks, directory=PATH_SYNCED)

    if verbose >= 2:
        print(f"Estimated time-delta of {stats['td_pred_median']:.4f} s ",
              f"({stats['td_pred_head']:.4f} to {stats['td_pred_tail']:.4f}), rel. sps={stats['sps_pred']:.4f}, ",
              f"90% Interval-Range={stats['td_quantile_range']:.4f} s")

    return df_results, stats, column_key


def nearest_advocate_windowed_nonlinear(
        path_rpeaks, path_jpeaks,
        td_min=TD_MIN, td_max=TD_MAX, td_sps=TD_SPS, sparse_factor=1, dist_max=0.25,
        frame_length=500, sliding_length=250, bandwidth_filter=None, quantile_filter=0.25,
        weight_decay=1e-6, quantile_range=0.9, verbose=5, save_fig=False):
    """Nearest Advocate over sliding windows.

    Apply the nearest advocate algorithm in sliding windows to capture
    deviations of time delays caused by a variable sample rates or clock-drift.
    A robust regression is applied to estimate the the time-delay and drift

    Parameters
    ----------
    path_rpeaks : str
        Path to the sorted r-peak file assumed to be correct (ground truth).
    path_jpeaks : str
        Path to the sorted j-peak file from the BCG, assumed to be shifted by a variable time-delta
    td_min : float, optional
        Lower bound of the search space for the time-shift, by default -60 s.
    td_max : float, optional
        Upper bound of the search space for the time-shift, by default 60 s.
    td_sps : int, optional
        Number of investigated time-shifts per second. Default None: sets it at 10 divided by the median gap of
        each array.
    sparse_factor : int, optional
        Factor for the sparseness of signal array for the calculation, higher is faster at the cost of precision,
        by default 1.
    dist_max : float, optional
        Maximal accepted distances between two advocate events. Default 0.25 s
    frame_length : int, optional
        Length of the sliding window, by default 500 events.
    sliding_length : int, optional
        Sliding length for the window, by default 250 events.
    quantile_filter : int, optional
        Filter based on the Nearest Advocate algorithm for outlier removal, by default 0.25 to remove the 25%
        strongest outlier.
    bandwidth_filter : int, optional
        Bandwidth for the kernel density estimation for outlier removal, by default None for no KDE-based filtering.
    verbose : int, optional
        Verbosity level, by default 5.
    save_fig : bool, optional
        If True, save the figure, by default False.

    Returns
    -------
    df_results
        A pandas DataFrame with the nearest advocate algorithm's result for each window.
    stats
        A dictionary with the robust linear regressions' results.
    """
    # load the r-peak file (ground truth)
    df_r = read_r_peaks(path_rpeaks=path_rpeaks)
    arr_r = df_r["time"].values

    # load the j-peak file from the BCG
    df_j, arr_j, column_key = read_j_peaks(path_jpeaks=path_jpeaks, directory=PATH_SYNCED,
                                           array_key=["time_corrected_nonlinear", "time_corrected"])

    # Apply Nearest Advocated Algorithm in Windows
    df_results = nearest_advocate_windowed(
        arr_r, arr_j,
        td_min=td_min, td_max=td_max, td_sps=td_sps, sparse_factor=sparse_factor, dist_max=dist_max,
        frame_length=frame_length, sliding_length=sliding_length,
        quantile_filter=quantile_filter, bandwidth_filter=bandwidth_filter, verbose=verbose)
    df_results.dropna(inplace=True)
    df_results = df_results[np.logical_and(
        df_results["index"] != 0, df_results["background_min_distance"] != 0)]

    # set heading and trailing time-delta to zero for regularization
    OUTLIER_FACTOR = 0.2
    outer_ts_diff = OUTLIER_FACTOR * (df_results["timestamp"].max() - df_results["timestamp"].min())
    df_results.loc[df_results.index[0], "timestamp"] = df_results["timestamp"].min() - outer_ts_diff
    df_results.loc[df_results.index[0], "time_delta"] = 0
    df_results.loc[df_results.index[-1], "timestamp"] = df_results["timestamp"].max() + outer_ts_diff
    df_results.loc[df_results.index[-1], "time_delta"] = 0

    # Estimate the non-linear mapping using a MLP
    # Define the neural network
    class MLP(nn.Module):
        """Basic MLP with three layers."""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, N_HIDDEN)  # one input feature, N neurons in hidden layer
            self.fc2 = nn.Linear(N_HIDDEN, 1)  # one output feature

        def forward(self, x):
            """Map the input onto the final layer."""
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define a MLP with loss function and an optimizer
    regressor = MLP()
    criterion = nn.HuberLoss(delta=0.05)  # HuberLoss with L1 for residuals > 0.05
    optimizer = torch.optim.Adam(regressor.parameters(), weight_decay=weight_decay)

    # Scale the timestamps into the interval [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1-OUTLIER_FACTOR, 1+OUTLIER_FACTOR))
    scaler.fit(df_results["timestamp"].values.reshape(-1, 1))
    timestamps_scaled = scaler.transform(df_results["timestamp"].values.reshape(-1, 1))

    # Assuming X_train is your one-dimensional input signal and y_train are your target values
    # Convert them to PyTorch tensors
    X_train = torch.tensor(timestamps_scaled, dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(df_results["time_delta"].values, dtype=torch.float32).view(-1, 1)

    # Train the MLP
    for _ in range(N_EPOCHS):  # number of epochs
        optimizer.zero_grad()  # zero the gradients
        pred_y = regressor(X_train)  # forward pass
        loss = criterion(pred_y, y_train)  # compute loss
        loss.backward()  # backward pass
        optimizer.step()  # update weights

    with torch.no_grad():
        regressor.eval()
        pred_y = regressor(X_train)
        mae = torch.nn.functional.l1_loss(pred_y, y_train).item()
        mse = torch.nn.functional.mse_loss(pred_y, y_train).item()
        rmse = np.sqrt(mse)

    # remove artificial heading and trailing point
    df_results = df_results[1:-1]

    # Predict the residuals for the J-peak DataFrame
    arr_j_linspace = np.linspace(arr_j[0], arr_j[-1], int(len(arr_j)/3))  # use the J-peak array for prediction
    pred_y = regressor(
        torch.tensor(scaler.transform(arr_j_linspace.reshape(-1, 1))).float()
        ).detach().numpy().reshape(-1)

    # Nonlinear correction of the predictions, pred_y are just the residuals of the linear trend
    f_timestamp = interp1d(arr_j_linspace, arr_j_linspace - pred_y, kind='linear')
    df_j["time_corrected_nonlinear"] = f_timestamp(arr_j)

    # save the corrected DataFrame
    save_j_peaks_corrected(df_j=df_j, path_jpeaks=path_jpeaks, directory=PATH_SYNCED)

    # Add statistic
    ci_low, ci_high = np.quantile(df_results["time_delta"].values,
                                  ((1-quantile_range)/2, 1-(1-quantile_range)/2))

    # Plot the resulting regression and estimations
    if verbose >= 1.5 or save_fig:
        plot_windowed_result(df_results, pred_y=pred_y, quantile_range=quantile_range,
                             save_fig=save_fig, verbose=verbose)

    # correct with the estimated time-delta and sps
    stats = {
        "td_pred_head": pred_y[0],
        "td_pred_tail": pred_y[-1],
        "td_pred_median": np.median(pred_y),
        "td_preds":  list(pred_y),
        "regressor": regressor,
        "sps_pred": "NaN",
        "td_quantile_range": ci_high-ci_low
    }
    if verbose >= 2:
        print(f"Corrected the time delay with a MLP with MAE={mae:.4f} s, RMSE={rmse:.4f} s,",
              f" 90% Interval-Range={stats['td_quantile_range']:.4f} s")

    return df_results, stats, column_key
