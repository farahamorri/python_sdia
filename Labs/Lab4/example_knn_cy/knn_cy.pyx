# knn_cy.pyx
import numpy as np
cimport numpy as np
cimport cython

def knn_cy_basic(np.ndarray[np.float64_t, ndim=2] X_train,
                 np.ndarray[np.int_t, ndim=1] y_train,
                 np.ndarray[np.float64_t, ndim=2] X_test,
                 int K=5):
    cdef int n_test = X_test.shape[0]
    cdef int n_train = X_train.shape[0]
    cdef int i, j
    cdef np.ndarray[np.int_t, ndim=1] y_pred = np.zeros(n_test, dtype=np.int32)
    cdef double[:] dist = np.zeros(n_train, dtype=np.float64)
    cdef double[:] x_ref

    for i in range(n_test):
        x_ref = X_test[i]
        for j in range(n_train):
            dist[j] = np.sqrt(np.sum((X_train[j] - x_ref)**2))
        # get indices of sorted distances
        neighbors_idx = np.argsort(dist)[:K]
        # get labels and majority vote
        labels = y_train[neighbors_idx]
        y_pred[i] = np.bincount(labels).argmax()
    return y_pred
