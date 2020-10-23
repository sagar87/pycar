import numpy as np


def weights_to_edgelist(W):
    return np.stack(np.where(np.triu(W) == 1), 1)


def sparse_car_eigenvals(W):
    D_sparse = W.sum(1)
    invsqrtD = np.diag(1 / np.sqrt(D_sparse))
    return np.linalg.eigvalsh(invsqrtD @ W @ invsqrtD)


def edgelist_to_weights(W_sparse, n):
    """Converts an edgelist to an adjacency matrix.

    Args:
      W_sparse: (e x 2) edgelist
      n: number of nodes in the graph
    Returns:
      An (n x n) adjacency matrix.
    """
    W_mat = np.zeros([n, n])

    for i in range(n):
        idx = np.where(W_sparse[:, 0] == i)[0]
        W_mat[i, W_sparse[idx, 1]] = 1
        W_mat[W_sparse[idx, 1], i] = 1

    return W_mat
