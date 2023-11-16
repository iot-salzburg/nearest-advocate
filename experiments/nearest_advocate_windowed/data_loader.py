"""Data Loader for the windowed Nearest Advocate algorithm for the correction of
linear and non-linear clock-drifts in event-based time-delays."""

import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd


# #######################################################
# ################## Data Loader ########################
# #######################################################

def read_r_peaks(path_rpeaks: str) -> pd.DataFrame:
    """
    Read the R-peaks file.

    Parameters
    ----------
    path_rpeaks : str
        Path to the sorted r-peak file assumed to be correct (ground truth).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the R-peak timestamps and columns 'rr', 'time', and 'stage' (sleep stage).
    """
    df_r = pd.read_csv(path_rpeaks)
    return df_r


def read_j_peaks(path_jpeaks: str, directory: str = None, array_key: Union[list, str] = None,
                 sps_rel: float = 1.0, td_hat: float = 0.0, verbose: int = 2
                 ) -> Tuple[pd.DataFrame, 'np.NDarray', str]:
    """
    Read the J-peaks file from the corrected directory.

    Parameters
    ----------
    path_jpeaks : str
        Path to the sorted j-peak file from the BCG, assumed to be shifted by a variable time-delta
    directory : str, optional
        Directory to the J-peak file to load. If set, the directory of the file_j_peaks is exchanged, by default None.
    array_key : list, str, optional
        Column of the J-peak file to extract. If not set, a new time_corrected column is created using sps_rel and
        td_hat.
    sps_rel : float, optional
        Initial relative sample rate deviation factor, by default 1.0. Not compatible with array_key.
    td_hat : float, optional
        Initial time delay shift, by default 0.0 s. Not compatible with array_key.
    verbose : int, optional
        Verbosity level, by default 1.


    Parameters
    ----------
    # ... (same as original)

    Returns
    -------
    df_j : pd.DataFrame
        A pandas DataFrame with the original and corrected timestamps of the J-peaks and JJ-intervals.
        Has always the keys 'time', 'jj', and 'time_corrected', optionally 'time_corrected_nonlinear'.
    arr_j : np.array
        A numpy array of the column of interest of df_j, by default, a new corrected time is created.
    column_key : str
        The key of the column used for 'arr_j'.
    """
    if directory:
        new_path_jpeaks = os.path.join(
            directory,
            Path(path_jpeaks.split(os.sep)[-1]).with_suffix(""),
            path_jpeaks.split(os.sep)[-1]
        )
        if os.path.exists(new_path_jpeaks):
            path_jpeaks = new_path_jpeaks

    # read the path_jpeaks file, either the original or in another updated directory
    df_j = pd.read_csv(path_jpeaks)

    # rename the
    if "time" not in df_j.columns and len(df_j.columns) == 1:
        df_j.rename(columns={df_j.columns[0]: "time"}, inplace=True)

    # extract the column of interest
    arr_j = None
    column_key_str = "time"
    if array_key and isinstance(array_key, str):
        arr_j = df_j[array_key].values
        column_key_str = array_key
    elif array_key and isinstance(array_key, list):
        for column_key in array_key:
            if column_key in df_j.columns:
                arr_j = df_j[column_key].values
                column_key_str = column_key
                break
    if verbose >= 2:
        print(f"Loading J-peaks from '{path_jpeaks}' with array from column '{column_key_str}'.")

    # If no array was extracted yet
    if arr_j is None:
        # search for the most advanced corrected key
        if "time_corrected_nonlinear" in df_j.columns:
            arr_j = df_j["time_corrected_nonlinear"].values
        elif "time_corrected" in df_j.columns:
            arr_j = df_j["time_corrected"].values
        else:
            # create a new corrected column based on sps_rel and td_hat
            if verbose:
                print("No corrected time in file, correcting using ",
                      f"sps_rel = {sps_rel} and td_hat = {td_hat} s")
            first_j_time = df_j["time"].values[0]
            df_j["time"] = df_j["time"] - first_j_time
            df_j["time_corrected"] = df_j["time"] * sps_rel
            arr_j = df_j["time_corrected"].values - td_hat

    # create the JJ-intervals based on the series of interst
    df_j.loc[1:, "jj"] = np.diff(arr_j)
    return df_j, arr_j, column_key_str


def save_j_peaks_corrected(df_j: pd.DataFrame, path_jpeaks: str, directory: str = None) -> None:
    """
    Saves a J-peak DataFrame to the synced path.

    Parameters
    ----------
    df_j : pd.DataFrame
        DataFrame containing the J-peak data to be saved.
    path_jpeaks : str
        Path to the sorted j-peak file from the BCG, assumed to be shifted by a variable time-delta.
    directory : str, optional
        Directory to the J-peak file to load. If set, the directory of the file_j_peaks is exchanged, by default None.

    Returns
    -------
    None
    """
    if directory:
        new_path_jpeaks = os.path.join(
            directory,
            Path(path_jpeaks.split(os.sep)[-1]).with_suffix(""),
            path_jpeaks.split(os.sep)[-1]
        )
        path_jpeaks = new_path_jpeaks

    # create directory if it does not exist
    parent_directory = os.sep.join(path_jpeaks.split(os.sep)[:-1])
    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)

    # write to csv file
    df_j.to_csv(path_jpeaks, index=False)

