from scipy.spatial.distance import euclidean
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from os.path import isfile, join
from os import listdir
from tqdm import trange
import smtplib
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
    for run_index, evaluated_run in enumerate(results):
        for partition_index in range(K_Folds):
            get_data = lambda x: evaluated_run[results_name][x][partition_index]
            
            processed_results['Model'].append( results_name )
            processed_results['ACC'].append( get_data('acc') )
            processed_results['MCC'].append( get_data('mcc') )
            processed_results['Partition Time'].append( get_data('partition_time') )
            
            processed_results['Run Index'].append( run_index + 1 )
        
    return processed_results

def run_experiments(exp_config, n_runs = 30):
    K_Folds = exp_config['K_Folds']
    kwargs = { 'iterable': n_runs, 'ascii': True}
    results_list = []
    
    if 'nca_pso' in exp_config:
        nca_pso = [nca_pso_run(**exp_config['nca_pso']) for _ in trange(n_runs, ascii=True, desc='NCA+PSO')]
        nca_pso = process_results(nca_pso, K_Folds, 'NCA+PSO')
        results_list.append(pd.DataFrame(nca_pso))
    if 'nca_gradient' in exp_config:
        nca_gradient = [nca_grad_run(**exp_config['nca_gradient']) for _ in trange(n_runs, ascii=True, desc='NCA+Gradient')]
        nca_gradient = process_results(nca_gradient, K_Folds, 'NCA+Gradient')
        results_list.append(pd.DataFrame(nca_gradient))
    if 'knn' in exp_config:
        knn = [knn_run(**exp_config['knn']) for _ in trange(n_runs, ascii=True, desc='KNN')]
        knn = process_results(knn, K_Folds, 'KNN')
        results_list.append(pd.DataFrame(knn))
    
    df = pd.concat(results_list)
    
    return df

def under_sampling(dataset, sample_size, objective_column):
    proportion = dataset[objective_column].value_counts(normalize=True)
    proportion = (proportion * sample_size).astype(int)
    df_group = dataset.groupby(by=objective_column)
    df_sampled = df_group.apply(
        lambda x: x.sample(n=proportion[x.iloc[0][objective_column]]))
    df_sampled = df_sampled.reset_index(drop=True)
    csv_name = f'dataset/Data_for_UCI_named_{sample_size}.csv'
    df_sampled.to_csv(csv_name, index=False)

def send_email(creds, desc):
    remetente, senha = creds['login'], creds['senha']
    assunto = 'Experimento Finalizado'
    mensagem = f'''Esse é um e-mail automático enviado para alertar que a execução do experimento foi finalizada no LC3 e enviar os resultados encontrados.
    O experimento realizado foi {desc}'''
    
    msg = MIMEMultipart()
    msg['From'], msg['To'], msg['Subject'] = remetente, destinatario, assunto
    msg.attach(MIMEText(mensagem, 'plain'))
    
    csv_files = [f for f in listdir('results') if isfile(join('./results',f))]
    
    for file in csv_files:
        with open(file,'rb') as f:
            part = MIMEApplication(f.read())
            part.add_header('Content-Disposition',f'attachment; filename="{arquivo}"')
            msg.attach(part)
            
    smtp_server, smtp_port = 'smtp.gmail.com', 587
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(remetente, senha)
        server.sendmail(rementente, destinatario, msg.as_string())
        print('Email com anexos enviados com sucesso!')
        
    except Exception as err:
        print(f'Ocorreu um erro ao enviar o email: {str(err)}')
        
    finally:
        server.quit()
