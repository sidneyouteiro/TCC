import time
import numpy as np
import pandas as pd

from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder

class NCA:
    def __init__(self, optimization_func=None, max_iter=50, k=5, cpu_count=None):
        self.optimization_func = optimization_func
        self.max_iter = max_iter
        self.k = k
        self.cpu_count = cpu_count

    def fit(self, X, y):
        if isinstance(X, pd.core.frame.DataFrame):
            self.accepted_columns = X.columns.copy()
            X = np.asarray(X)
        if isinstance(y, pd.core.frame.DataFrame):
            self.accepted_columns = self.accepted_columns + y.columns.copy() if hasattr(self,'accepted_columns') else y.columns.copy() 
            y = np.asarray(y)
            
        y = LabelEncoder().fit_transform(y)            
        t_train = time.time()
        # Compute a mask that stays fixed during optimization:
        same_class_mask = y[:, np.newaxis] == y[np.newaxis, :]
        # (n_samples, n_samples)

        # Initialize the transformation
        transformation = np.ravel(np.eye(X.shape[1],X.shape[1]))

        params = {
            'X':X,
            'same_class_mask':same_class_mask,
            'transformation': transformation,
            'max_iter': self.max_iter,
            'cpu_count': self.cpu_count
        }


        # Call the optimizer
        opt_result = self.optimization_func(**params)
        
           
        # Reshape the solution found by the optimizer
        self.components_ = opt_result.reshape(-1, X.shape[1])

        self.X_train_transformed = self.transform(X)
        self.y_train = y
        
        # Stop timer
        t_train = time.time() - t_train
        self.fit_time = t_train
        
        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def _classify_test_object(self,test_object):
        k = self.k
        class_probabilities = {i:0 for i in np.unique(self.y_train)}
        distances = np.array([euclidean(test_object, i) for i in self.X_train_transformed])
        top_k_closest_distances_indexs = np.argpartition(distances,k)[:k]
        sum_k_distances = np.sum(distances[top_k_closest_distances_indexs])
        # usar o softmax
        for neighbor_index in top_k_closest_distances_indexs:
            class_probabilities[self.y_train[neighbor_index]] += distances[neighbor_index] / sum_k_distances
        return max(class_probabilities, key=class_probabilities.get)
    
    def predict(self, X_test):
        try:
            if type(X_test) == type(np.array([])):
                X_test_transformed = self.transform(X_test)
                return np.array([self._classify_test_object(i) for i in X_test_transformed])
            elif (type(X_test) == type(pd.DataFrame([]))) and (X_test.columns == self.accepted_columns):
                print('b')
                X_test = np.asarray(X_test)
                X_test_transformed = self.transform(X_test)
                y_predict = [self._classify_test_object(i) for i in X_test_transformed]
                return pd.DataFrame(y_predict,columns=self.accepted_columns[-1:])
            else:
                raise RuntimeError('[Error] foi encontrado um problema ao lidar com {X_test}')
            
        except Exception as err:
            print(f'foi encontrado o seguinte error: {repr(err)}')