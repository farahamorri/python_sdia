import numpy as np
cimport numpy as np
from libc.math cimport sqrt
cimport cython

@cython.boundscheck(False)  # Disable bounds checking for speed
@cython.wraparound(False)   # Disable negative index wraparound
def knn_cy_optimized(double[:, :] X_train, int[:] y_train, double[:, :] X_test, int K=5):
    cdef int n_test = X_test.shape[0]
    cdef int n_train = X_train.shape[0]
    cdef int n_features = X_train.shape[1]
    cdef int i, j, k, m
    cdef double dist
    cdef int min_idx
    cdef double min_val
    cdef int max_count, majority_class

    cdef double[:] distances = np.zeros(n_train, dtype=np.float64)
    cdef int[:] y_pred = np.zeros(n_test, dtype=np.int32)
    cdef int[:] neighbors_idx = np.zeros(K, dtype=np.int32)

    # Count array for majority vote (assuming class labels are small non-negative ints)
    cdef int max_label = np.max(y_train) + 1
    cdef int[:] counts = np.zeros(max_label, dtype=np.int32)

    for i in range(n_test):
        # 1. Compute distances
        for j in range(n_train):
            dist = 0.0
            for k in range(n_features):
                dist += (X_train[j,k] - X_test[i,k]) ** 2
            distances[j] = sqrt(dist)

        # 2. Find K nearest neighbors (simple selection)
        for m in range(K):
            min_idx = -1
            min_val = 1e20
            for j in range(n_train):
                if distances[j] < min_val:
                    min_val = distances[j]
                    min_idx = j
            neighbors_idx[m] = min_idx
            distances[min_idx] = 1e20  # mark as used

        # 3. Majority vote
        counts[:] = 0  # reset counts
        for m in range(K):
            counts[y_train[neighbors_idx[m]]] += 1

        # 4. Find class with max count
        max_count = -1
        majority_class = -1
        for j in range(max_label):
            if counts[j] > max_count:
                max_count = counts[j]
                majority_class = j

        y_pred[i] = majority_class

    return y_pred
