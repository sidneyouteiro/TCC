class NCA:
    from scipy.optimize import minimize

    def __init__(self, optimization_func=None, max_iter=50, tol=1e-5):
        self.optimization_func = optimization_func
        self.max_iter = max_iter
        self.tol = tol

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
        self.n_iter_ = 0
        opt_result = minimize(**optimizer_params)

        # Reshape the solution found by the optimizer
        self.components_ = opt_result.x.reshape(-1, X.shape[1])

        # Stop timer
        t_train = time.time() - t_train

        return self

    def transform(self, X):
        return np.dot(X, self.components_.T)

    def _loss_grad_lbfgs(self, transformation, X, same_class_mask, sign=1.0):
        from sklearn.utils.extmath import softmax
        from skelearn.metrics import pairwise_distances
        
        t_funcall = time.time()

        transformation = transformation.reshape(-1, X.shape[1])
        X_embedded = np.dot(X, transformation.T)  # (n_samples, n_components)

        # Compute softmax distances
        p_ij = pairwise_distances(X_embedded, squared=True)
        np.fill_diagonal(p_ij, np.inf)
        p_ij = softmax(-p_ij)  # (n_samples, n_samples)

        # Compute loss
        masked_p_ij = p_ij * same_class_mask
        p = np.sum(masked_p_ij, axis=1, keepdims=True)  # (n_samples, 1)
        loss = np.sum(p)

        # Compute gradient of loss w.r.t. `transform`
        weighted_p_ij = masked_p_ij - p_ij * p
        weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
        np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))
        gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(X)
        # time complexity of the gradient: O(n_components x n_samples x (
        # n_samples + n_features))

        return sign * loss, sign * gradient.ravel()