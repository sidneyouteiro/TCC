def PSO(X, same_class_mask, transformation,max_iter=1000, cpu_count=None, my_pso=False, swarm_size=10):
    import pickle
    import numpy as np
    import pyswarms as ps
    from .classGlobalBestPSO import GlobalBestPSO
    from sklearn.metrics import pairwise_distances
    from sklearn.utils.extmath import softmax
    
    global obj_func
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
    GBPSO_options = {
        'n_particles': swarm_size,
        'dimensions': len(transformation),
        'options':options
    }
    if my_pso:
        optimizer = GlobalBestPSO(**GBPSO_options)
    else:
        optimizer = ps.single.GlobalBestPSO(**GBPSO_options)
    _, A = optimizer.optimize(obj_func, max_iter, same_class_mask=same_class_mask,
                              X_train=X, verbose=False, n_processes=cpu_count)
    return A


def Gradient(X, same_class_mask, transformation,max_iter=1000,cpu_count=None):
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.utils.extmath import softmax
    
    def alt_obj_func(x, same_class_mask, X_train):
        from sklearn.metrics import pairwise_distances
        from sklearn.utils.extmath import softmax
        
        X_embedded = np.dot(X_train,x.T)
        p_ij = pairwise_distances(X_embedded, squared=True)
        p_ij = softmax(-p_ij)
        np.fill_diagonal(p_ij, 0.0)
        
        pi = same_class_mask * p_ij
        
    
    def obj_func_grad(x, same_class_mask, X_train):
        from sklearn.metrics import pairwise_distances
        from sklearn.utils.extmath import softmax
        
        for i in range(X_train.shape[0]):
            
            X_embedded = np.dot(X_train, x.T)
            p_ij = pairwise_distances(X_embedded, squared=True)
            p_ij = softmax(-p_ij)
            np.fill_diagonal(p_ij, 0.0)
            
            p = 0.0
            for j in range(X_train.shape[0]):
                if same_class_mask[i][j]:
                    p += p_ij[i][j]
            
            primeiro_termo = np.zeros( (x.shape[1],x.shape[1]) )
            segundo_termo = np.zeros( (x.shape[1],x.shape[1]) )
            
            for k in range(X_train.shape[0]):
                if i == k: continue
                
                xik = X_train[i] - X_train[k]
                pik = p_ij[i][k]
                term = pik * np.outer(xik,xik)
                primeiro_termo += term
                if same_class_mask[i][j]:
                    segundo_termo += term
            primeiro_termo *= p
            
            x += 0.5 * (primeiro_termo - segundo_termo)
        return x
    
    def score(x, same_class_mask, X_train):
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
    
    transformation = transformation.reshape(-1, X.shape[1])
    for it in range(max_iter):
        if it % 200 == 0: 
            print('a',end=' ')
        old_score = score(transformation, same_class_mask, X)
        new_transformation = obj_func_grad(transformation, same_class_mask, X)
        new_score = score(new_transformation, same_class_mask, X)
        if abs(old_score - new_score) < 0.001:
            break

    return np.ravel(transformation)