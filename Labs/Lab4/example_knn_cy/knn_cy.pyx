# knn_cython_step3.pyx
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def knn_cy_basic(double[:, :] X_train, int[:] y_train, double[:, :] X_test, int K=5):
    cdef int n_test = X_test.shape[0]
    cdef int n_train = X_train.shape[0]
    cdef int n_features = X_train.shape[1]
    cdef int i, j, k

    cdef double dist   # declare once
    cdef int[:] y_pred = np.zeros(n_test, dtype=np.int32)
    cdef double[:] distances = np.zeros(n_train, dtype=np.float64)

    for i in range(n_test):
        for j in range(n_train):
            dist = 0.0   # reset each iteration
            for k in range(n_features):
                dist += (X_train[j,k] - X_test[i,k]) ** 2
            distances[j] = sqrt(dist)
        # Partial sort: get indices of K smallest distances
        sorted_idx = np.argsort(distances)[:K]
        neighbors_labels = y_train[sorted_idx]

        counts = np.bincount(neighbors_labels)
        y_pred[i] = np.argmax(counts)

    return y_pred
