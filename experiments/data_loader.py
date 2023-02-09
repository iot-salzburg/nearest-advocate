"""Data Loader functions"""

import os
import numpy as np
import pandas as pd

def load_heart_beats(path_to_dir: str, is_synched: bool=True, verbose: int=1):
    """Loads all pairs of heart rate data.
    
    Parameters
    ----------
    path_to_dir: str
        Path to the directory of heart beat data, containing the PSG and 
        already synchronized and shifted offbody measurement.
    is_synched: bool
        Load the signal data that is already synched and clock-drift corrected.
    verbose: int
        Show more output, 0 or larger.
    Returns
    -------
    pair_dataset : list
        List of tuples of two timestamp np.ndarrays for the R-peaks
    """
    ref_files = sorted([file for file in os.listdir(path_to_dir) 
                        if file.endswith("_nn.txt")])
    if is_synched:
        signal_files = [file for file in os.listdir(path_to_dir) 
                        if file.endswith("_beats_synced.csv")]
    else:
        signal_files = [file for file in os.listdir(path_to_dir) 
                        if file.endswith("_beats.csv")]
    
    # iterate through ref_files and load the data pairwise
    pair_dataset = list()
    for ref_file in ref_files:
        # skip files with the note 'm' in it
        if "m" in ref_file:
            continue
        pid = ref_file[1:].split("_nn")[0][:-1]
        
        # search number in signal_files, or continue
        for sig_file in signal_files:
            if pid in sig_file:
                break
        else:
            continue
        if verbose >= 1:
            print(f"Loading files {ref_file} and {sig_file}.")
        
        # load reference file
        arr_ref = pd.read_table(os.path.join(path_to_dir, ref_file), sep=",")["time"].values

        # load signal file 
        arr_sig = pd.read_csv(os.path.join(path_to_dir, sig_file))["time"].values
        
        # append to heart_beat_dataset
        pair_dataset.append((arr_ref, arr_sig))
    return pair_dataset
        
def load_breath_rate(path_to_dir: str, verbose: int=1):
    """Loads all pairs of breath rate data (from Bra4Vit).
    
    Parameters
    ----------
    path_to_dir: str
        Path to the directory of heart beat data, containing the Reference and 
        SportsSRS measurement.
    verbose: int
        Show more output, 0 or larger.
    Returns
    -------
    pair_dataset : list
        List of tuples of two timestamp np.ndarrays for the FlowReversals
    """
    ref_files = sorted([file for file in os.listdir(path_to_dir) 
                        if "Reference" in file])
    signal_files = [file for file in os.listdir(path_to_dir) 
                    if "SportsSRS" in file]
    time_delay_df = pd.read_csv("TimeDelays_BreathRate.csv", index_col="Unnamed: 0")

    # iterate through ref_files and load the data pairwise
    pair_dataset = list()
    for ref_file in ref_files:
        # skip files with the note 'm' in it
        pid = ref_file[1:].split("_")[0]
        
        # search number in signal_files, or continue
        for sig_file in signal_files:
            if pid in sig_file:
                break
        else:
            continue
        
        # get and check if the sync was successful
        sync_res = time_delay_df.loc["P" + pid + "_SportsSRS_FlowReversals_RUNS_MERGED.csv"]        
        if not sync_res["is_successfull"]:
            continue
            
        if verbose >= 1:
            print(f"Loading files {ref_file} and {sig_file}.")
            
        # load reference file
        arr_ref = pd.read_csv(
            os.path.join(path_to_dir, ref_file), sep=";", usecols=["Time"], 
            converters={"Time": lambda x: float(x.replace(",", "."))}
        )["Time"].values

        # load signal file
        arr_sig = pd.read_csv(
            os.path.join(path_to_dir, sig_file), sep=";", usecols=["Time"], 
            converters={"Time": lambda x: float(x.replace(",", "."))}
        )["Time"].values
        # adapt the signal array by a linear time scale and shift by td_hat
        arr_sig = arr_sig * sync_res["sps_hat"] + sync_res["td_hat"]
        
        # append to heart_beat_dataset
        pair_dataset.append((arr_ref, arr_sig))
    return pair_dataset


def load_step_rate(path_to_dir: str, verbose: int=1):
    """Loads all pairs of breath rate data (from Bra4Vit).
    
    Parameters
    ----------
    path_to_dir: str
        Path to the directory of step beat data, containing the Reference and 
        SportsSRS measurement.
    verbose: int
        Show more output, 0 or larger.
    Returns
    -------
    pair_dataset : list
        List of tuples of two timestamp np.ndarrays for the FlowReversals
    """
    ref_files = sorted([file for file in os.listdir(path_to_dir) 
                        if "Reference" in file])
    signal_files = [file for file in os.listdir(path_to_dir) 
                    if "SportsSRS" in file]
    time_delay_df = pd.read_csv("TimeDelays_Strides.csv", index_col="Unnamed: 0")
            
    # iterate through ref_files and load the data pairwise
    pair_dataset = list()
    for ref_file in ref_files:
        # skip files with the note 'm' in it
        pid = ref_file[1:].split("_")[0]
        
        # search number in signal_files, or continue
        for sig_file in signal_files:
            if pid in sig_file:
                break
        else:
            continue
            
        # get and check if the sync was successful
        sync_res = time_delay_df.loc["P" + pid + "_SportsSRS_Strides_RUNS_MERGED.csv"]       
        if not sync_res["is_successfull"]:
            continue
            
        if verbose >= 1:
            print(f"Loading files {ref_file} and {sig_file}.")
        
        # load reference file
        arr_ref = pd.read_csv(
            os.path.join(path_to_dir, ref_file), sep=";", usecols=["x"], 
            converters={"x": lambda x: float(x.replace(",", "."))}
        )["x"].values

        # load signal file
        arr_sig = pd.read_csv(
            os.path.join(path_to_dir, sig_file), sep=";", usecols=["x"], 
            converters={"x": lambda x: float(x.replace(",", "."))}
        )["x"].values
        # adapt the signal array by a linear time scale and shift by td_hat
        arr_sig = arr_sig * sync_res["sps_hat"] + sync_res["td_hat"]

        # append to heart_beat_dataset
        pair_dataset.append((arr_ref, arr_sig))
    return pair_dataset
