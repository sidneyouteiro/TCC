from scipy.spatial.distance import euclidean
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import trange
import pandas as pd
import numpy as np
import time

def nca_pso_run(X, y, K_Folds=5, model_class=None, model_options={}):
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc','partition_time']}
    for (train_index, test_index) in kf.split(X):
        t_start = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.loc[train_index,:])
        X_test = scaler.transform(X.loc[test_index,:])
        
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        
        model = model_class(**model_options)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        mcc = matthews_corrcoef(y_test,y_pred)
        t_end = time.time()
        
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
        folds_results['partition_time'].append(t_end - t_start)
    
    return { 'NCA+PSO' : folds_results }

def classify_test_object(test_object, X_train_nca,y_train,k):
    class_probabilities = {i:0 for i in np.unique(y_train)}
    distances = np.array([euclidean(test_object, i) for i in X_train_nca])
    top_k_closest_distances_indexs = np.argpartition(distances,k)[:k]
    sum_k_distances = np.sum(distances[top_k_closest_distances_indexs])
    for neighbor_index in top_k_closest_distances_indexs:
        class_probabilities[y_train[neighbor_index]] += distances[neighbor_index] / sum_k_distances
    return max(class_probabilities, key=class_probabilities.get)

def nca_grad_run(X, y, k, K_Folds=5, model_class=None, model_options={}):
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc','partition_time']}
    for (train_index, test_index) in kf.split(X):
        t_start = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index,:])
        X_test = scaler.transform(X[test_index,:])
        
        y_train, y_test = y[train_index].copy(), y[test_index].copy()
        
        model = model_class(**model_options)
        X_train_nca = model.fit_transform(X_train,y_train)
        X_test_nca = model.transform(X_test)
        y_pred = np.array([classify_test_object(test_obj,X_train_nca,y_train,k) for test_obj in X_test_nca])
        
        acc = accuracy_score(y_test,y_pred)
        mcc = matthews_corrcoef(y_test,y_pred)
        t_end = time.time()
        
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
        folds_results['partition_time'].append(t_end - t_start)
    
    return { 'NCA+Gradient': folds_results }

def knn_run(X,y, K_Folds=5, model_class=None, model_options={}):
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc','partition_time']}
    for (train_index, test_index) in kf.split(X):
        t_start = time.time()
        scaler, encoder = StandardScaler(), LabelEncoder()
        X_train = scaler.fit_transform(X[train_index,:])
        X_test = scaler.transform(X[test_index,:])
        
        y_train = encoder.fit_transform(y[train_index].copy())
        y_test = encoder.transform(y[test_index].copy())
        
        model = model_class(**model_options)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test,y_pred)
        mcc = matthews_corrcoef(y_test,y_pred)
        t_end = time.time()
        
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
        folds_results['partition_time'].append(t_end - t_start)
    
    return { 'KNN' : folds_results }

def process_results(results, K_Folds, results_name):
    columns = ['Model','Run Index','ACC','MCC','Partition Time']
    processed_results = {i:[] for i in columns}
    for evaluated_run in results:
        for partition_index in range(K_Folds):
            get_data = lambda x: evaluated_run[results_name][x][partition_index]
            
            acc = get_data('acc')
            mcc = get_data('mcc')
            partition_time = get_data('partition_time')
        
    return processed_results

def run_experiments(models_options, n_runs = 30):
    #nca_pso_results = [nca_pso_run(**models_options['nca_pso_options']) for _ in trange(n_runs,desc='NCA+PSO')]
    #nca_gradient_results = [nca_grad_run(**models_options['nca_gradient_options']) for _ in trange(n_runs,desc='NCA+Gradient')]
    knn_results = [knn_run(**models_options['knn_options']) for _ in trange(n_runs,desc='KNN')]
    
    print(knn_results)
    
    #K_Folds = models_options['K_Folds']
    #nca_pso_processed_results = process_results(nca_pso_results, K_Folds, 'NCA+PSO')
    #nca_gradient_processed_results = process_results(nca_gradient_results, K_Folds, 'NCA+Gradient')
    #knn_processed_results = process_results(knn_results, K_Folds, 'KNN')
    
    #df = pd.concat(
    #    [
    #        pd.DataFrame(nca_pso_processed_results),
    #        pd.DataFrame(nca_gradient_processed_results),
    #        pd.DataFrame(knn_processed_results),
    #    ]
    #).set_index('Model')
    
    #return df