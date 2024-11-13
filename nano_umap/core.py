import logging
from abc import ABC, abstractmethod
import numpy as np
import numba
from scipy.sparse import coo_matrix
import numba as nb
from scipy import sparse
from tqdm import tqdm
from pynndescent import NNDescent
from umap import spectral


logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

_NUMBA_COMPILED_FN = {}


class NanoUMAPBase(ABC):
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

    @abstractmethod
    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        pass

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

        if self._verbose:
            LOGGER.info(
                f"Building Usearch KNN index for dataset of shape {dataset.shape}"
            )
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
        self._precomputed_knn = knn_indices, knn_scores.astype(np.float32)
        return self._precomputed_knn

    def _get_initial_embedding(
        self, dataset: np.ndarray, graph: sparse.coo_matrix
    ) -> np.ndarray:
        if self._random_state is not None:
            np.random.seed(self._random_state)

        if isinstance(self._init, np.ndarray):
            return self._init.copy()
        elif self._init == "spectral":
            affinity = 0.5 * (graph.T + graph)
            if self._verbose:
                LOGGER.info("Initializing embedding with UMAP spectral initialization")
            embedding = spectral.spectral_layout(
                dataset,
                affinity,
                dim=self._n_components,
                random_state=self._random_state,
            )
        elif self._init == "random":
            if self._verbose:
                LOGGER.info("Initializing embedding with random")
            embedding = np.random.uniform(size=(graph.shape[0], self._n_components))
        else:
            raise ValueError(f"Unknown init method: {self._init}")

        min_vals, max_vals = np.min(embedding, 0), np.max(embedding, 0)
        embedding = 10.0 * (embedding - min_vals) / (max_vals - min_vals)
        return embedding.astype(np.float32, order="C")

    def _get_n_epochs(self, n_values: int) -> int:
        if self._n_epochs is not None:
            return self._n_epochs
        return 500 if n_values <= 10000 else 200

    def _get_lr(self, iteration: int, n_epochs: int) -> float:
        return np.float32(self._learning_rate * (1 - iteration / n_epochs))

    def _get_n_neg_samples(self, iteration: int, n_epochs: int) -> int:
        n_neg = self._n_neighbors * (1 - iteration / n_epochs)
        return int(max(0.0, self._negative_sample_rate * n_neg))


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


def make_epochs_per_sample(weights: np.ndarray, n_epochs: int) -> np.ndarray:
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    min_value = weights.max() / float(n_epochs)
    weights[weights < min_value] = min_value

    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


def get_compiled_fn(func, parallel: bool = False, local_vars: dict | None = None):
    if (func, parallel) in _NUMBA_COMPILED_FN:
        return _NUMBA_COMPILED_FN[(func, parallel)]
    if local_vars is None:
        local_vars = {}
    compiled_fn = nb.njit(
        func, parallel=parallel, fastmath=True, boundscheck=False, locals=local_vars
    )
    _NUMBA_COMPILED_FN[(func, parallel)] = compiled_fn
    return compiled_fn


@numba.njit
def convert_to_sparse(knn_indices: np.ndarray, knn_values: np.ndarray):
    n_vectors, n_neighbors = knn_indices.shape
    total_edges = n_vectors * n_neighbors * 2  # Since we add both (i, j) and (j, i)
    rows = np.empty(total_edges, dtype=np.int32)
    cols = np.empty(total_edges, dtype=np.int32)
    distances = np.empty(total_edges, dtype=np.float32)
    n_values = 0

    for i in range(n_vectors):
        for k in range(n_neighbors):
            j = knn_indices[i, k]
            d = knn_values[i, k]
            min_i_j = min(i, j)
            max_i_j = max(i, j)

            rows[n_values] = min_i_j
            cols[n_values] = max_i_j
            distances[n_values] = d
            n_values = n_values + 1

            rows[n_values] = min_i_j
            cols[n_values] = max_i_j
            distances[n_values] = d
            n_values = n_values + 1

    return rows[:n_values], cols[:n_values], distances[:n_values]


@numba.njit()
def aggregate_edges(sorted_rows, sorted_cols, sorted_distances):
    n = len(sorted_rows)
    if n == 0:
        return (
            np.empty(0, dtype=sorted_rows.dtype),
            np.empty(0, dtype=sorted_cols.dtype),
            np.empty(0, dtype=sorted_distances.dtype),
        )

    final_rows = [sorted_rows[0]]
    final_cols = [sorted_cols[0]]
    final_distances = [sorted_distances[0]]

    for i in range(1, n):
        if (
            sorted_rows[i] == sorted_rows[i - 1]
            and sorted_cols[i] == sorted_cols[i - 1]
        ):
            # Duplicate edge, take the minimum distance
            if sorted_distances[i] < final_distances[-1]:
                final_distances[-1] = sorted_distances[i]
        else:
            # New edge
            final_rows.append(sorted_rows[i])
            final_cols.append(sorted_cols[i])
            final_distances.append(sorted_distances[i])

    return np.array(final_rows), np.array(final_cols), np.array(final_distances)


def to_adjacency_matrix(knn_indices: np.ndarray, knn_values: np.ndarray):
    """Returns upper triangular adjacency matrix from kNN indices and values"""
    rows, cols, distances = convert_to_sparse(knn_indices, knn_values)
    order = np.lexsort((cols, rows))
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    sorted_distances = distances[order]
    final_rows, final_cols, final_distances = aggregate_edges(
        sorted_rows, sorted_cols, sorted_distances
    )
    graph = coo_matrix(
        (final_distances, (final_rows, final_cols)),
        shape=(knn_indices.shape[0], knn_indices.shape[0]),
    )
    return graph
