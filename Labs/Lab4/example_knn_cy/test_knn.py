import numpy as np

train = np.loadtxt('data/synth_train.txt')
class_train = np.asarray(train[:,0], dtype=np.intc)  # <-- cast to C-compatible integer
x_train = np.asarray(train[:,1:], dtype=np.float64)   # <-- cast to float64

test = np.loadtxt('data/synth_test.txt')
class_test = np.asarray(test[:,0],dtype=np.intc)
x_test = np.asarray(test[:,1:], dtype=np.float64)     # <-- cast to float64


# # X and y arrays
# x_train = np.array(train[:, 1:], dtype=np.float64, order='C')
# class_train = np.array(train[:, 0], dtype=np.int32, order='C')

# x_test = np.array(test[:, 1:], dtype=np.float64, order='C')
# class_test = np.array(test[:, 0], dtype=np.int32, order='C')
