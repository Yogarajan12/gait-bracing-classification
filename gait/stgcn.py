"""Spatio-temporal graph convolutional network for the gait skeleton.

Extracted from the project notebook (the wide-residual trial). The adaptations
that make it importable: ``build_gait_adjacency`` returns the normalized
adjacency instead of it being a notebook global, and ``STGCN_block`` takes that
adjacency as an argument rather than referencing a global ``A_norm``. The layer
math is unchanged.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

from .graph import build_gait_adjacency


class GraphConv(layers.Layer):
    """Graph convolution with a fixed normalized adjacency."""

    def __init__(self, A, out_channels, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.A = tf.constant(A, tf.float32)   # fixed normalized adjacency
        self.out_channels = out_channels
        self.use_bias = use_bias

    def build(self, input_shape):
        # input_shape = (batch, time, nodes, in_channels)
        in_ch = input_shape[-1]
        self.W = self.add_weight(
            name="W",
            shape=(in_ch, self.out_channels),
            initializer="glorot_uniform",
        )
        if self.use_bias:
            self.b = self.add_weight(
                name="b",
                shape=(self.out_channels,),
                initializer="zeros",
            )
        super().build(input_shape)

    def call(self, X):
        # X: (batch, T, N, in_ch)
        # spatial mixing: Y[b,t,i,f] = sum_j A_hat[i,j] * X[b,t,j,f]
        Y = tf.einsum("ij,btjf->btif", self.A, X)
        # feature mixing
        Y = tf.tensordot(Y, self.W, axes=[[3], [0]])   # (batch, T, N, out_ch)
        if self.use_bias:
            Y = Y + self.b
        return Y


def STGCN_block(x, out_channels, A_norm, Kt=7, name=None):
    """Residual ST-GCN block: graph conv, temporal conv, and a projected skip."""
    # spatial conv
    x_sp = GraphConv(A_norm, out_channels, name=f"{name}_gcn")(x)
    x_sp = layers.BatchNormalization(name=f"{name}_bn_sp")(x_sp)
    x_sp = layers.ReLU(name=f"{name}_relu_sp")(x_sp)

    # temporal conv (Conv2D over (time, node) with kernel (Kt, 1))
    x_tp = layers.Conv2D(
        filters=out_channels, kernel_size=(Kt, 1), padding="same",
        name=f"{name}_tconv",
    )(x_sp)
    x_tp = layers.BatchNormalization(name=f"{name}_bn_tp")(x_tp)
    x_tp = layers.ReLU(name=f"{name}_relu_tp")(x_tp)

    # residual skip: project input x -> out_channels then add
    res = layers.Conv2D(
        filters=out_channels, kernel_size=(1, 1), padding="same",
        name=f"{name}_res_conv",
    )(x)
    res = layers.BatchNormalization(name=f"{name}_bn_res")(res)

    x_out = layers.Add(name=f"{name}_add")([x_tp, res])
    x_out = layers.ReLU(name=f"{name}_relu_out")(x_out)
    return x_out


def make_wide_residual_stgcn(T, N, C, n_classes=3, lr=5e-3, A_norm=None):
    """Build and compile the wide residual ST-GCN. Input shape is (T, N, C)."""
    if A_norm is None:
        A_norm = build_gait_adjacency()

    inp = Input(shape=(T, N, C), name="gait_graph_in")
    x = STGCN_block(inp, out_channels=64, A_norm=A_norm, Kt=7, name="stgcn1")
    x = STGCN_block(x, out_channels=128, A_norm=A_norm, Kt=7, name="stgcn2")
    x = STGCN_block(x, out_channels=256, A_norm=A_norm, Kt=7, name="stgcn3")
    x = layers.GlobalAveragePooling2D(name="global_avg")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="class_out")(x)

    m = Model(inp, out, name="STGCN_wide_residual")
    m.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m
