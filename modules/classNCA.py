class NCA:
    def __init__(self, optimization_func=None, max_iter=50, k=5):
        self.optimization_func = optimization_func
        self.max_iter = max_iter
        self.k = k

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import time

        y = LabelEncoder().fit_transform(y)            
        t_train = time.time()
        # Compute a mask that stays fixed during optimization:
        same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]
        # (n_samples, n_samples)

        # Initialize the transformation
        transformation = np.ravel(np.eye(X.shape[1],X.shape[1]))

        
        # Call the optimizer
        opt_result = self.optimization_func(X,same_class_mask,transformation,max_iter=self.max_iter)
        
           
        # Reshape the solution found by the optimizer
        self.components_ = opt_result.reshape(-1, X.shape[1])

        self.X_train_transformed = self.transform(X)
        self.y_train = y
        
        # Stop timer
        t_train = time.time() - t_train
        self.fit_time = t_train
        
        return self

    def transform(self, X):
        import numpy as np
        
        return np.dot(X, self.components_.T)

    def _classify_test_object(self,test_object):
        from scipy.spatial.distance import euclidean
        import numpy as np
        
        k = self.k
        class_probabilities = {i:0 for i in np.unique(self.y_train)}
        distances = np.array([euclidean(test_object, i) for i in self.X_train_transformed])
        top_k_closest_distances_indexs = np.argpartition(distances,k)[:k]
        sum_k_distances = np.sum(distances[top_k_closest_distances_indexs])
        for neighbor_index in top_k_closest_distances_indexs:
            class_probabilities[self.y_train[neighbor_index]] += distances[neighbor_index] / sum_k_distances
        return max(class_probabilities, key=class_probabilities.get)
    
    def predict(self, X_test):
        X_test_transformed = self.transform(X_test)
        
        return [self._classify_test_object(i) for i in X_test_transformed]