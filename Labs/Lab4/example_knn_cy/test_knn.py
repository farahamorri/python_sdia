import numpy as np
import time


train = np.loadtxt('data/synth_train.txt')
class_train = np.asarray(train[:,0], dtype=np.int32)  # <-- cast to integer
x_train = np.asarray(train[:,1:], dtype=np.float64)   # <-- cast to float64

test = np.loadtxt('data/synth_test.txt')
class_test = np.asarray(test[:,0],dtype=np.int32)
x_test = np.asarray(test[:,1:], dtype=np.float64)     # <-- cast to float64

# K = 9

# # Python baseline
# start = time.time()
# y_py = knn(x_train, class_train, x_test, K)
# print("Python KNN:", time.time() - start)

# # Cython basic
# start = time.time()
# y_cy = knn_cy_basic(x_train, class_train, x_test, K)
# print("Cython basic KNN:", time.time() - start)
