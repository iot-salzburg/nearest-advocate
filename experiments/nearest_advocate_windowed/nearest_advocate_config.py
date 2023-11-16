"""Config for the windowed Nearest Advocate algorithm for the correction of
linear and non-linear clock-drifts in event-based time-delays."""

# Nearest Advocate search configuration
TD_MIN = -900  # +- 15 minutes
TD_MAX = 900
TD_MID = 30  # time delay range for all subsequent runs
TD_NEAR = 5  # near range for more focused analysis
TD_SPS = 20.0  # time-delta to investigate for shift (>10 times the event frequency)
SPS_REL = 1.0

# Path to write the synced results
PATH_SYNCED = "../data/HeartBeat_synched/"

# output image format
IMG_FORMAT = "png"

# set the initial value for pseudo-random functions
SEED = 1
