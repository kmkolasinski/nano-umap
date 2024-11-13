import numba as nb
import numpy as np
from nano_umap import core
from nano_umap.umap_v0 import NanoUMAPV0


class NanoUMAPV1(NanoUMAPV0):

    def _get_update_step_fn(self):
        return core.get_compiled_fn(_update_step, self._n_jobs == -1)



def _update_step(
    x: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    lr: float = 1.0,
    n_neighbors: int = 15,
    n_neg_samples: int = 15,
    repulsion_strength: float = 1.0,
    a: float = 1.6,
    b: float = 0.9,
):
    n_vectors, dim = x.shape
    b1, a1 = (b - 1.0), 0.001
    c0 = (-2.0 * a * b)
    c1 = (2.0 * repulsion_strength * b)
    lr_pos = lr / n_neighbors
    gradient = np.zeros_like(x)

    for i in nb.prange(rows.shape[0]):
        row, col, s_ij = rows[i], cols[i], scores[i]
        xi = x[row]
        xj = x[col]

        # attraction force, copied from UMAP source code
        diff = xi - xj
        dij = np.dot(diff, diff)
        if dij == 0.0:
            continue

        grad_coeff = c0 * pow(dij, b1) / (a * pow(dij, b) + 1.0)
        grad = s_ij * grad_coeff * diff
        grad = np.clip(grad, -2, 2)
        for d in range(dim):
            grad_value = grad[d] * lr_pos
            gradient[row][d] += grad_value
            gradient[col][d] -= grad_value

    # sampling negatives for i-th point
    if n_neg_samples > 0:
        lr_neg = lr / n_neg_samples
        for i in nb.prange(n_vectors):
            for idx in range(n_neg_samples):
                j = np.random.randint(n_vectors)
                if j == i:
                    continue

                # repulsion force, copied from UMAP source code
                diff = x[i] - x[j]
                dij = np.dot(diff, diff)
                if dij == 0.0:
                    continue
                grad_coeff = c1 / ((a1 + dij) * (a * pow(dij, b) + 1.0))
                grad = grad_coeff * diff
                grad = np.clip(grad, -2, 2)
                for d in range(dim):
                    gradient[i][d] += grad[d] * lr_neg

    x += gradient

