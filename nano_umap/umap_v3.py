import numba as nb
import numpy as np
from nano_umap import core


class NanoUMAPV3(core.NanoUMAPBase):
    def _get_update_step_fn(self):
        float32_vars = ["dij", "a", "b", "grad_value", "lr"]
        local_vars = {var: nb.float32 for var in float32_vars}
        return core.get_compiled_fn(
            _update_step, self._n_jobs == -1, local_vars=local_vars
        )

    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        knn_indices, knn_similarities = self._get_knn(dataset)
        pbar = self._get_progress_bar()
        n_epochs = self._get_n_epochs(dataset.shape[0])

        update_step_fn = self._get_update_step_fn()

        graph = core.to_adjacency_matrix(knn_indices, knn_similarities).tocoo()
        x = self._get_initial_embedding(dataset, graph)
        for epoch in pbar(range(n_epochs)):
            update_step_fn(
                x,
                rows=graph.row,
                cols=graph.col,
                scores=graph.data,
                lr=self._get_lr(epoch, n_epochs),
                n_neighbors=self._n_neighbors,
                n_neg_samples=self._negative_sample_rate, # this is constant
                repulsion_strength=self._repulsion_strength,
            )

        return x


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
            grad_value = core.rclip(grad[d]) * lr
            xi[d] += grad_value
            xj[d] -= grad_value

        # sampling negatives for i-th point
        for _ in range(n_neg_samples):
            j = np.random.randint(n_vectors)
            if j == i:
                continue

            # repulsion force, copied from UMAP source code
            diff = xi - x[j]
            dij = core.rdot(diff)
            if dij == zero:
                continue
            grad_coeff = c1 / ((a1 + dij) * (a * (dij**b) + one))
            grad = grad_coeff * diff
            for d in range(dim):
                grad_value = core.rclip(grad[d]) * lr
                xi[d] += grad_value
