"""Core reusable components for the multivariate gait analysis project.

The notebook remains the canonical narrative that runs the full analysis and
produces the figures. This package holds the pieces worth importing and reusing:
sequence building and smoothing, the HMM / LSTM / attention models, and the
ST-GCN.

Re-exports are lazy (PEP 562) so that importing the numpy/scipy preprocessing
utilities does not require TensorFlow, hmmlearn, or sklearn to be installed.
"""

import importlib

_EXPORTS = {
    "build_sequences": "preprocessing",
    "moving_average_smooth": "preprocessing",
    "butter_lowpass": "preprocessing",
    "butter_smooth": "preprocessing",
    "compute_derivatives": "preprocessing",
    "HMMClassifier": "models",
    "make_lstm_model": "models",
    "Attention": "models",
    "build_gait_adjacency": "graph",
    "GraphConv": "stgcn",
    "STGCN_block": "stgcn",
    "make_wide_residual_stgcn": "stgcn",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
