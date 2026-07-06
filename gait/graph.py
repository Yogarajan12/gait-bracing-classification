"""Gait skeleton graph structure (framework-free).

Kept separate from the TensorFlow ST-GCN layers so the adjacency can be built,
imported, and tested with only NumPy.
"""

import numpy as np

# Skeleton edges. Nodes: 0=L_ankle, 1=L_knee, 2=L_hip, 3=R_ankle, 4=R_knee, 5=R_hip
GAIT_EDGES = [
    (0, 1), (1, 2),   # left ankle-knee-hip
    (3, 4), (4, 5),   # right ankle-knee-hip
    (2, 5),           # hip-hip
    (1, 4),           # knee-knee
    (0, 3),           # ankle-ankle
]
N_NODES = 6


def build_gait_adjacency(edges=GAIT_EDGES, n_nodes=N_NODES):
    """Return the symmetrically normalized adjacency A_hat = D^-1/2 A D^-1/2."""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(n_nodes, dtype=np.float32)              # self-loops
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    return D_inv_sqrt @ A @ D_inv_sqrt
