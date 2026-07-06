"""Model definitions: a per-class Gaussian HMM classifier, an LSTM builder, and
a simple attention layer.

Extracted from the project notebook. The only adaptation is that
``make_lstm_model`` takes an explicit ``n_classes`` argument instead of reading
a notebook-level ``y``; the logic is otherwise unchanged.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from hmmlearn.hmm import GaussianHMM

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


class HMMClassifier(BaseEstimator, ClassifierMixin):
    """One Gaussian HMM per class; predict by maximum log-likelihood."""

    def __init__(self, n_components=4, covariance_type="full", n_iter=100, random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # X is array-like of shape (n_cycles, T, n_chan); y is labels 0/1/2
        self.models_ = []
        for c in np.unique(y):
            seqs = [X[i] for i in range(len(y)) if y[i] == c]
            lengths = [s.shape[0] for s in seqs]
            concat = np.vstack(seqs)
            m = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            m.fit(concat, lengths)
            self.models_.append(m)
        return self

    def predict(self, X):
        # for each cycle, pick the HMM with highest log-likelihood
        preds = []
        for cycle in X:
            ll = [m.score(cycle) for m in self.models_]
            preds.append(int(np.argmax(ll)))
        return np.array(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def make_lstm_model(input_shape, n_classes=3, lr=1e-3):
    """
    Return a fresh, compiled LSTM classifier.

    input_shape = (T, F), e.g. (101, 6).
    n_classes is the number of output classes.
    """
    model = Sequential([
        LSTM(64, input_shape=input_shape, name="lstm"),
        Dropout(0.3, name="dropout"),
        Dense(32, activation="relu", name="dense1"),
        Dense(n_classes, activation="softmax", name="class_out"),
    ], name="lstm_classifier")

    model.compile(
        optimizer=Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class Attention(Layer):
    """Additive attention over the time axis; returns (context, weights)."""

    def build(self, input_shape):
        # a trainable vector to score each timestep
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer="random_normal",
            trainable=True,
            name="attn_vec",
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        # raw score for each timestep via dot-product -> (batch, time)
        scores = tf.tensordot(inputs, self.w, axes=[2, 0])
        # normalize to get attention weights
        weights = tf.nn.softmax(scores, axis=1)                  # (batch, time)
        # context vector as weighted sum of inputs
        context = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)
        return context, weights
