from scipy.spatial.distance import euclidean
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm
import numpy as np
import time

def nca_pso_folds_run(X, y, K_Folds=5, model_class=None, model_options={}):
    t_start = time.time()
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc']}
    for (train_index, test_index) in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.loc[train_index,:])
        X_test = scaler.transform(X.loc[test_index,:])
        
        y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
        
        model = model_class(**model_options)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        mcc = matthews_corrcoef(y_test,y_pred)
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
    t_end = time.time()
    return folds_results, t_end-t_start

def classify_test_object(test_object, X_train_nca,y_train,k):
    class_probabilities = {i:0 for i in np.unique(y_train)}
    distances = np.array([euclidean(test_object, i) for i in X_train_nca])
    top_k_closest_distances_indexs = np.argpartition(distances,k)[:k]
    sum_k_distances = np.sum(distances[top_k_closest_distances_indexs])
    for neighbor_index in top_k_closest_distances_indexs:
        class_probabilities[y_train[neighbor_index]] += distances[neighbor_index] / sum_k_distances
    return max(class_probabilities, key=class_probabilities.get)

def nca_gradient_folds_run(X, y, k, K_Folds=5, model_class=None, model_options={}):
    t_start = time.time()
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc']}
    for (train_index, test_index) in tqdm(kf.split(X), position=1,leave=True):
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
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
    t_end = time.time()
    return folds_results, t_end - t_start

def knn_folds_run(X,y, K_Folds=5, model_class=None, model_options={}):
    t_start = time.time()
    kf = KFold(n_splits=K_Folds)
    folds_results = {i:[] for i in ['acc','mcc']}
    for (train_index, test_index) in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index,:])
        X_test = scaler.transform(X[test_index,:])
        
        y_train, y_test = y[train_index].copy(), y[test_index].copy()
        
        model = model_class(**model_options)
        model.fit(X_train,y_train)
        y_pred = model.predict(y_test)
        
        acc = accuracy_score(y_test,y_pred)
        mcc = matthews_corrcoef(y_test,y_pred)
        folds_results['acc'].append(acc)
        folds_results['mcc'].append(mcc)
    t_end = time.time()
    return folds_results, t_end - t_start

def process_results(results, K_Folds, results_name):
    mean_acc, std_acc = [], []
    mean_mcc, std_mcc = [], []
    times = []
    for evaluated_partition in results:
        evaluated_dict = evaluated_partition[0]
        mean_acc.append(np.mean(evaluated_dict['acc']))
        mean_mcc.append(np.mean(evaluated_dict['mcc']))
        std_acc.append(np.std(evaluated_dict['acc']))
        std_mcc.append(np.std(evaluated_dict['mcc']))
        times.append(evaluated_partition[1])
    mean_time, std_time = np.mean(times), np.std(times)
    mean_acc, std_acc = np.mean(mean_acc), np.mean(std_acc)
    mean_mcc, std_mcc = np.mean(mean_mcc), np.mean(std_mcc)
    
    processed_results = {
        'Model': results_name, 'Mean Time': [mean_time], 
        'Std Time': [std_time], 'Mean Acc': [mean_acc], 
        'Std Acc': [std_acc], 'Mean Mcc': [mean_mcc], 
        'Std Mcc': [std_mcc]
    }
    return processed_results