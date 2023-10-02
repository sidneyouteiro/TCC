def PSO(X, same_class_mask, transformation,max_iter=1000):
    import pyswarms as ps
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.utils.extmath import softmax
    
    def obj_func(x,same_class_mask,X_train):
        from sklearn.utils.extmath import softmax
        from sklearn.metrics import pairwise_distances
    
        x = x.reshape(-1, X_train.shape[1])
        X_embedded = np.dot(X_train, x.T)  # (n_samples, n_components)

        # Compute softmax distances
        p_ij = pairwise_distances(X_embedded, squared=True)
        p_ij = softmax(-p_ij)  # (n_samples, n_samples)
        np.fill_diagonal(p_ij, 0.0)
        
        masked_p_ij = p_ij * same_class_mask
        p = np.sum(masked_p_ij, axis=1, keepdims=True)  
        loss = np.sum(p)

        return -1.0 * loss
    
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9} # parametros cognitivo, social e de inercia
    optimizer = ps.single.GlobalBestPSO(n_particles= 10,
                                        dimensions = len(transformation),
                                        options=options)
    _, A = optimizer.optimize(obj_func, max_iter, same_class_mask=same_class_mask,X_train=X,verbose=False)
    return A