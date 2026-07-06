"""Preprocessing: build gait-cycle sequences and smooth / differentiate them.

Extracted from the project notebook. The one adaptation from the notebook is
that the inline sequence-building loop is wrapped into ``build_sequences`` so it
can be reused; the smoothing and derivative logic is unchanged.
"""

import numpy as np
from scipy.signal import butter, filtfilt

# Column order for the six joint-leg channels: (leg, joint) tuples,
# i.e. left ankle/knee/hip then right ankle/knee/hip.
CHANNEL_ORDER = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]


def build_sequences(df, value_col="angle_norm"):
    """
    Pivot a long-format gait table into an array of cycles.

    Each (subject, condition, replication) group becomes one 101 x 6 cycle.
    Returns X of shape (n_cycles, 101, 6) and y of shape (n_cycles,), with
    labels mapped condition 1/2/3 -> 0/1/2.
    """
    seqs = []
    labels = []
    for (subj, cond, rep), grp in df.groupby(["subject", "condition", "replication"]):
        # pivot so columns MultiIndex = (leg, joint)
        piv = grp.pivot(index="time", columns=["leg", "joint"], values=value_col)
        # ensure time runs 0..100
        piv = piv.reindex(range(101), fill_value=np.nan).sort_index()

        # handle any genuine gaps: interpolate then assert no NaNs remain
        if piv.isnull().values.any():
            piv.interpolate(method="linear", axis=0, inplace=True)
        assert not piv.isnull().values.any(), "Found NaNs after interpolation"

        # reorder columns to [ (1,1),(1,2),(1,3),(2,1),(2,2),(2,3) ]
        piv = piv.loc[:, CHANNEL_ORDER]

        seqs.append(piv.values)      # (101, 6)
        labels.append(cond - 1)      # map 1->0, 2->1, 3->2

    return np.array(seqs), np.array(labels)


def moving_average_smooth(X, window=5, mode="same"):
    """Per-channel moving-average smoothing over the time axis of X (n, T, C)."""
    kernel = np.ones(window) / window
    X_smooth = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            X_smooth[i, :, j] = np.convolve(X[i, :, j], kernel, mode=mode)
    return X_smooth


def butter_lowpass(cutoff, fs, order=3):
    """Design a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_smooth(X, cutoff=2.0, fs=100.0, order=3):
    """
    Zero-phase Butterworth low-pass smoothing over the time axis of X (n, T, C).

    cutoff: desired cutoff frequency (Hz)
    fs: sampling rate (Hz); cycles are normalized to 101 points over 0-100%,
        treated as fs=100 so each percent point is one unit of time.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    X_smooth = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            # filtfilt does zero-phase filtering
            X_smooth[i, :, j] = filtfilt(b, a, X[i, :, j])
    return X_smooth


def compute_derivatives(X):
    """Return (velocity, acceleration) via finite differences along the time axis."""
    vel = np.diff(X, axis=1)      # (n, T-1, C)
    acc = np.diff(vel, axis=1)    # (n, T-2, C)
    return vel, acc
