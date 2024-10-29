import numpy as np
import numba
from scipy.sparse import coo_matrix


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
