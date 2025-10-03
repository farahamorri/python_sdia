import numpy as np
cimport numpy as np
cimport cython

# -------------------------------
# Basic version (direct translation)
# -------------------------------
def knn_cy_basic(np.ndarray[np.float64_t, ndim=2] X_train,
                 np.ndarray[np.int32_t, ndim=1] y_train,
                 np.ndarray[np.float64_t, ndim=2] X_test,
                 int K=5):
    cdef int n_test = X_test.shape[0]
    cdef int n_train = X_train.shape[0]
    cdef int i, j
    cdef np.ndarray[np.int32_t, ndim=1] y_pred = np.zeros(n_test, dtype=np.int32)
    cdef double[:] dist = np.zeros(n_train, dtype=np.float64)
    cdef double[:] x_ref

    for i in range(n_test):
        x_ref = X_test[i]
        for j in range(n_train):
            dist[j] = np.sqrt(np.sum((X_train[j] - x_ref)**2))
        neighbors_idx = np.argsort(dist)[:K]
        labels = y_train[neighbors_idx]
        y_pred[i] = np.bincount(labels).argmax()
    return y_pred


# -------------------------------
# Optimized version using memoryviews and loops
# -------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
def knn_cy_optimized(double[:, :] X_train,
                     np.int32_t[:] y_train,
                     double[:, :] X_test,
                     int K=5):
    """
    Optimized KNN using memoryviews and C loops.
    X_train: 2D float64 C-contiguous
    y_train: 1D int32 C-contiguous
    X_test: 2D float64 C-contiguous
    K: number of neighbors
    """

    cdef int n_test = X_test.shape[0]
    cdef int n_train = X_train.shape[0]
    cdef int n_features = X_train.shape[1]

    cdef int i, j, k
    cdef double dist

    # pre-allocate distance array
    cdef double[:] dists = np.zeros(n_train, dtype=np.float64)
    # pre-allocate K labels
    cdef np.int32_t[:] labels = np.zeros(K, dtype=np.int32)
    # prediction array
    cdef np.ndarray[np.int32_t, ndim=1] y_pred = np.zeros(n_test, dtype=np.int32)

    cdef np.ndarray[np.int32_t, ndim=1] sort_idx

    for i in range(n_test):
        # compute distances
        for j in range(n_train):
            dist = 0
            for k in range(n_features):
                dist += (X_train[j, k] - X_test[i, k])**2
            dists[j] = dist**0.5

        # get K nearest indices
        sort_idx = np.argsort(dists)[:K]
        for k in range(K):
            labels[k] = y_train[sort_idx[k]]

        # majority vote
        y_pred[i] = np.bincount(labels).argmax()

    return y_pred
