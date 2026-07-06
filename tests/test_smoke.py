"""Minimal smoke tests for the gait package.

These check that the extracted pieces still wire together (shapes and basic
runs), not scientific accuracy, which comes from running the notebook. Tests
that need TensorFlow, hmmlearn, or pandas skip automatically when those are not
installed. Run with `pytest` from the repository root.
"""

import numpy as np
import pytest


def _synthetic_long_df(pd, n_conditions=2):
    """Build a tiny long-format gait table with one cycle per condition."""
    rows = []
    for cond in range(1, n_conditions + 1):
        for leg in (1, 2):
            for joint in (1, 2, 3):
                for t in range(101):
                    rows.append({
                        "subject": 1, "condition": cond, "replication": 1,
                        "leg": leg, "joint": joint, "time": t,
                        "angle_norm": np.sin(t / 10.0) + 0.1 * cond,
                    })
    return pd.DataFrame(rows)


def test_build_sequences_shape():
    pd = pytest.importorskip("pandas")
    from gait.preprocessing import build_sequences
    X, y = build_sequences(_synthetic_long_df(pd, n_conditions=2))
    assert X.shape == (2, 101, 6)
    assert list(y) == [0, 1]


def test_smoothing_preserves_shape():
    from gait.preprocessing import moving_average_smooth, butter_smooth
    X = np.random.randn(3, 101, 6)
    assert moving_average_smooth(X, window=5).shape == X.shape
    assert butter_smooth(X, cutoff=2.0, fs=100.0, order=3).shape == X.shape


def test_derivative_shapes():
    from gait.preprocessing import compute_derivatives
    X = np.random.randn(3, 101, 6)
    vel, acc = compute_derivatives(X)
    assert vel.shape == (3, 100, 6)
    assert acc.shape == (3, 99, 6)


def test_gait_adjacency_is_symmetric():
    from gait.graph import build_gait_adjacency
    A = build_gait_adjacency()
    assert A.shape == (6, 6)
    assert np.allclose(A, A.T)


def test_hmm_classifier_runs():
    pytest.importorskip("hmmlearn")
    from gait.models import HMMClassifier
    rng = np.random.default_rng(0)
    X = [rng.standard_normal((10, 3)) for _ in range(6)]
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = HMMClassifier(n_components=2, n_iter=5, random_state=0).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (6,)


def test_lstm_builds():
    pytest.importorskip("tensorflow")
    from gait.models import make_lstm_model
    model = make_lstm_model(input_shape=(101, 6), n_classes=3)
    assert model.output_shape[-1] == 3


def test_stgcn_builds():
    pytest.importorskip("tensorflow")
    from gait.stgcn import make_wide_residual_stgcn
    model = make_wide_residual_stgcn(T=101, N=6, C=1, n_classes=3)
    assert model.output_shape[-1] == 3
