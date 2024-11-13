import numba as nb
import numpy as np
from nano_umap import core
from nano_umap.umap_v0 import NanoUMAPV0


class NanoUMAPV2(NanoUMAPV0):
    def _get_update_step_fn(self):
        float32_vars = ["dij", "a", "b", "grad_value"]
        local_vars = {var: nb.float32 for var in float32_vars}
        return core.get_compiled_fn(
            _update_step, self._n_jobs == -1, local_vars=local_vars
        )


def _update_step(
    x: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    lr: float = 1.0,
    n_neighbors: int = 15,
    n_neg_samples: int = 15,
    repulsion_strength: float = 1.0,
    a: float = np.float32(1.6),
    b: float = np.float32(0.9),
):
    n_vectors, dim = x.shape
    dtype = np.float32
    zero, one = dtype(0.0), dtype(1.0)
    b1, a1 = dtype(b - 1.0), dtype(0.001)
    c0 = dtype(-2.0 * a * b)
    c1 = dtype(2.0 * repulsion_strength * b)
    lr_pos = dtype(lr / n_neighbors)

    for i in nb.prange(rows.shape[0]):
        row, col, s_ij = rows[i], cols[i], scores[i]
        xi = x[row]
        xj = x[col]

        # attraction force, copied from UMAP source code
        diff = xi - xj
        dij = core.rdot(diff)
        if dij == zero:
            continue

        grad_coeff = c0 * (dij**b1) / (a * (dij**b) + one)
        grad = s_ij * grad_coeff * diff
        for d in range(dim):
            grad_value = core.rclip(grad[d]) * lr_pos
            xi[d] += grad_value
            xj[d] -= grad_value

    # sampling negatives for i-th point
    if n_neg_samples > 0:
        lr_neg = dtype(lr / n_neg_samples)
        for i in nb.prange(n_vectors):
            for idx in range(n_neg_samples):
                j = np.random.randint(n_vectors)
                if j == i:
                    continue

                # repulsion force, copied from UMAP source code
                diff = x[i] - x[j]
                dij = core.rdot(diff)
                if dij == zero:
                    continue
                grad_coeff = c1 / ((a1 + dij) * (a * (dij**b) + one))
                grad = grad_coeff * diff
                for d in range(dim):
                    grad_value = core.rclip(grad[d]) * lr_neg
                    x[i][d] += grad_value
