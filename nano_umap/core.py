import logging
import numba as nb
import numpy as np
from tqdm import tqdm
from usearch import index as usearch_index

LOGGER = logging.getLogger(__name__)


class NanoUMAP:
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        learning_rate: float = 1.0,
        n_epochs: int = 500,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 2,
        precomputed_knn: tuple[np.ndarray, np.ndarray] | None = None,
        x0: np.ndarray | None = None,
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
        self._x0 = x0
        self._repulsion_strength = repulsion_strength
        self._negative_sample_rate = negative_sample_rate

    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        x = self._initialize_embedding(dataset)
        knn_indices, knn_similarities = self._get_knn(dataset)
        # ignore points that are not in the knn index (-1 is the ignore indicator)
        mask = np.where((knn_indices == -1).mean(-1) > 0.5)[0]

        pbar = self._get_progress_bar()
        for iteration in pbar(range(self._n_epochs)):
            x = update_step(
                x,
                knn_indices[:, 1 : self._n_neighbors + 1],
                knn_similarities[:, 1 : self._n_neighbors + 1],
                lr=self._get_lr(iteration),
                n_neg_samples=self._get_n_neg_samples(iteration),
                repulsion_strength=self._repulsion_strength,
            )
            x -= np.mean(x, axis=0, keepdims=True)
            x[mask] *= 0

        return x

    def _get_progress_bar(self):
        if self._verbose:
            return tqdm
        else:
            return lambda x: x

    def _get_knn(self, dataset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._precomputed_knn:
            return self._precomputed_knn
        dtype = np.float32
        LOGGER.info(f"Building Usearch KNN index for dataset of shape {dataset.shape}")

        shuffle_indices = np.arange(dataset.shape[0])
        np.random.shuffle(shuffle_indices)
        dataset = dataset[shuffle_indices].astype(dtype)

        # finding the k nearest neighbors
        indices = np.arange(dataset.shape[0])
        index = usearch_index.Index(ndim=dataset.shape[-1], dtype="f32", metric="cos")
        index.add(indices, dataset, log=True)
        result = index.search(dataset, self._n_neighbors + 1)
        knn_indices = result.keys
        knn_scores = 1 - result.distances

        # calibrate scores to that the mean is 1 for the first neighbor
        knn_scores = knn_scores / knn_scores[:, 1].mean()
        knn_indices = knn_indices.astype(np.int32)
        knn_scores = knn_scores.astype(dtype)
        self._precomputed_knn = (knn_indices, knn_scores)
        return knn_indices, knn_scores

    def _initialize_embedding(self, dataset: np.ndarray) -> np.ndarray:
        if self._x0 is not None:
            return self._x0.copy()
        if self._random_state is not None:
            np.random.seed(self._random_state)
        num_points, _ = dataset.shape
        noise = np.random.normal(size=(num_points, self._n_components))
        return noise.astype(np.float32)

    def _get_lr(self, iteration: int) -> float:
        return np.float32(self._learning_rate * (1 - iteration / self._n_epochs) ** 0.2)

    def _get_n_neg_samples(self, iteration: int) -> int:
        n_neg = int(self._n_neighbors * (1 - iteration / self._n_epochs))
        n_neg = max(0, self._negative_sample_rate * n_neg)
        return n_neg


@nb.njit()
def clip(val: float) -> float:
    if val > 2.0:
        return 2.0
    elif val < -2.0:
        return -2.0
    else:
        return val


@nb.njit(
    "f4(f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": nb.types.float32,
        "dim": nb.types.intp,
        "i": nb.types.intp,
    },
)
def rdot(x: np.ndarray) -> float:
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] ** 2
    return result


@nb.njit(parallel=False, fastmath=True, boundscheck=False, error_model="numpy")
def update_step(
    x: np.ndarray,
    knn_indices: np.ndarray,
    knn_scores: np.ndarray,
    lr: float = 1.0,
    n_neg_samples: int = 15,
    repulsion_strength: float = 1.0,
    a: float = 1.5,
    b: float = 1.1,
) -> np.ndarray:
    n_vectors, dim = x.shape
    n_neighbors = knn_indices.shape[1]
    gradient = np.zeros_like(x)

    dtype = np.float32
    b1 = dtype(1.0 - b)
    a1 = dtype(0.001)
    c0 = dtype(-2.0 * a * b)
    c1 = dtype(2.0 * repulsion_strength * b)
    lr_pos = dtype(lr / n_neighbors)
    lr_neg = dtype(lr / n_neg_samples)

    for i in nb.prange(n_vectors):
        grad_i = gradient[i]

        for idx in range(n_neighbors):
            j = knn_indices[i, idx]
            if j == i:
                continue
            s_ij = knn_scores[i, idx]

            # attraction force, copied from UMAP source code
            diff = x[i] - x[j]
            dij = rdot(diff)
            grad_coeff = c0 * pow(dij, b1) / (a * pow(dij, b) + dtype(1.0))
            grad = s_ij * grad_coeff * diff

            for d in range(dim):
                grad_i[d] += clip(grad[d]) * lr_pos

        if n_neg_samples <= 0:
            continue

        # sampling negatives for i-th point
        for idx in range(n_neg_samples):
            j = np.random.randint(n_vectors)
            if j == i:
                continue

            # repulsion force, copied from UMAP source code
            diff = x[i] - x[j]
            dij = rdot(diff)
            grad_coeff = c1 / ((a1 + dij) * (a * pow(dij, b) + dtype(1.0)))
            grad = grad_coeff * diff

            for d in range(dim):
                grad_i[d] += clip(grad[d]) * lr_neg

    x += gradient
    return x
