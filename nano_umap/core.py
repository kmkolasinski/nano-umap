import logging
import numba as nb
import numpy as np
from scipy import sparse
from tqdm import tqdm
from pynndescent import NNDescent
from nano_umap import utils
from umap import spectral


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class NanoUMAP:
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        learning_rate: float = 1.0,
        n_epochs: int | None = None,
        repulsion_strength: float = 1.0,
        negative_sample_rate: float = 1.0,
        n_jobs: int = -1,
        precomputed_knn: tuple[np.ndarray, np.ndarray] | None = None,
        init: np.ndarray | str = "spectral",
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._learning_rate = learning_rate
        self._n_epochs = n_epochs
        self._precomputed_knn = precomputed_knn
        self._random_state = random_state
        self._verbose = verbose
        self._init = init
        self._repulsion_strength = repulsion_strength
        self._negative_sample_rate = negative_sample_rate
        self._n_jobs = n_jobs

    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        knn_indices, knn_similarities = self._get_knn(dataset)
        graph = utils.to_adjacency_matrix(knn_indices, knn_similarities).tocoo()

        pbar = self._get_progress_bar()
        update_step_fn = _get_update_step_fn(self._n_jobs == -1)

        x = self._get_initial_embedding(dataset, graph)
        n_epochs = self._get_n_epochs(graph.shape[0])

        for iteration in pbar(range(n_epochs)):
            x = update_step_fn(
                x,
                rows=graph.row,
                cols=graph.col,
                scores=graph.data,
                lr=self._get_lr(iteration, n_epochs),
                n_neighbors=self._n_neighbors,
                n_neg_samples=self._get_n_neg_samples(iteration, n_epochs),
                repulsion_strength=self._repulsion_strength,
            )

        return x

    def _get_progress_bar(self):
        if self._verbose:
            return lambda _: tqdm(_, desc="Optimizing")
        else:
            return lambda x: x

    def _get_knn(self, dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._precomputed_knn:
            self._n_neighbors = self._precomputed_knn[0].shape[1]
            return self._precomputed_knn

        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)

        LOGGER.info(f"Building Usearch KNN index for dataset of shape {dataset.shape}")
        n_trees = min(64, 5 + int(round((dataset.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(dataset.shape[0]))))
        knn_search_index = NNDescent(
            dataset,
            n_neighbors=self._n_neighbors + 1,
            metric="cosine",
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
        )
        knn_indices, knn_dists = knn_search_index.neighbor_graph
        knn_scores = 1 - knn_dists
        return knn_indices, knn_scores.astype(np.float32)

    def _get_initial_embedding(self, dataset: np.ndarray, graph: sparse.coo_matrix) -> np.ndarray:
        if self._random_state is not None:
            np.random.seed(self._random_state)

        if isinstance(self._init, np.ndarray):
            return self._init.copy()
        elif self._init == "spectral":
            affinity = 0.5 * (graph.T + graph)
            LOGGER.info("Initializing embedding with UMAP spectral initialization")
            embedding = spectral.spectral_layout(
                dataset, affinity, dim=self._n_components, random_state=self._random_state
            )
        elif self._init == "random":
            LOGGER.info("Initializing embedding with random")
            embedding = np.random.uniform(size=(graph.shape[0], self._n_components))
        else:
            raise ValueError(f"Unknown init method: {self._init}")
        return embedding.astype(np.float32)

    def _get_n_epochs(self, n_values: int) -> int:
        if self._n_epochs is not None:
            return self._n_epochs
        return 500 if n_values <= 10000 else 200

    def _get_lr(self, iteration: int, n_epochs: int) -> float:
        return np.float32(self._learning_rate * (1 - iteration / n_epochs))

    def _get_n_neg_samples(self, iteration: int, n_epochs: int) -> int:
        n_neg = self._n_neighbors * (1 - iteration / n_epochs)
        return int(max(0.0, self._negative_sample_rate * n_neg))


def _update_step(
    x: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
    lr: float = 1.0,
    n_neighbors: int = 15,
    n_neg_samples: int = 15,
    repulsion_strength: float = 1.0,
    a: float = 1.5,
    b: float = 1.1,
) -> np.ndarray:
    n_vectors, dim = x.shape
    dtype = np.float32
    b1, a1 = dtype(1.0 - b), dtype(0.001)
    c0 = dtype(-2.0 * a * b)
    c1 = dtype(2.0 * repulsion_strength * b)
    lr_pos = dtype(lr / n_neighbors)

    for i in nb.prange(rows.shape[0]):
        row, col, s_ij = rows[i], cols[i], scores[i]
        xi = x[row]
        xj = x[col]

        # attraction force, copied from UMAP source code
        diff = xi - xj
        dij = rdot(diff)
        if dij == 0.0:
            continue

        grad_coeff = c0 * pow(dij, b1) / (a * pow(dij, b) + dtype(1.0))
        grad = s_ij * grad_coeff * diff
        for d in range(dim):
            grad_value = rclip(grad[d]) * lr_pos
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
                dij = rdot(diff)
                if dij == 0.0:
                    continue
                grad_coeff = c1 / ((a1 + dij) * (a * pow(dij, b) + dtype(1.0)))
                grad = grad_coeff * diff
                for d in range(dim):
                    x[i][d] += rclip(grad[d]) * lr_neg

    return x


@nb.njit("f4(f4)", inline="always", fastmath=True, cache=True)
def rclip(val: float) -> float:
    if val > 2.0:
        return 2.0
    elif val < -2.0:
        return -2.0
    else:
        return val


@nb.njit("f4(f4[::1])", fastmath=True, cache=True, inline="always", boundscheck=False)
def rdot(x: np.ndarray) -> float:
    result = 0.0
    for i in range(x.shape[0]):
        result += x[i] ** 2
    return result


_update_step_njit_parallel = nb.njit(
    _update_step, parallel=True, fastmath=True, boundscheck=False
)

_update_step_njit = nb.njit(
    _update_step, parallel=False, fastmath=True, boundscheck=False
)


def _get_update_step_fn(parallel: bool = False):
    if parallel:
        return _update_step_njit_parallel
    return _update_step_njit
