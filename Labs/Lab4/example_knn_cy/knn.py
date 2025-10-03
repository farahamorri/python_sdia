import numpy as np

def knn(X_train, y_train, X_test, K=5):
    X = np.array(X_train)
    y = np.array(y_train)
    X_t = np.array(X_test)

    # List to store the predicted class for each point
    y_pred = []
    # Loop over each point in X (leave-one-out approach)
    for i in range(len(X_t)):
        x_ref = X_t[i]
        # Compute Euclidean distance between x_ref and all other points
        distances = np.linalg.norm(X - x_ref, axis=1)
        # Sort the indices of points by increasing distance
        sorted_dist = np.argsort(distances)
        # Select indices of the N nearest neighbors (ignoring the point itself at index 0)
        neighbors_idx = sorted_dist[1:K+1]
        # Get the labels of these neighbors
        neighbors_labels = y[neighbors_idx]
        # Majority vote: count occurrences of each class and select the most frequent one
        majority_class = np.bincount(neighbors_labels.astype(int)).argmax()
        # Store the predicted class for this point
        y_pred.append(majority_class)
    return y_pred
