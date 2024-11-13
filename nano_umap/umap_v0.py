import numba as nb
import numpy as np
from nano_umap import core


class NanoUMAPV0(core.NanoUMAPBase):

    def _get_update_step_fn(self):
        return core.get_compiled_fn(_update_step, self._n_jobs == -1)

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
                n_neg_samples=self._get_n_neg_samples(epoch, n_epochs),
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


    x += gradient

