#!/usr/bin/env python
# coding: utf-8

# # Experimento TCC

# ## Libs

# In[1]:


from sklearn.neighbors import NeighborhoodComponentsAnalysis
from psutil import cpu_count
import pandas as pd
import numpy as np


# In[2]:


from modules.classNCA import NCA
from modules.optimization_funcs import PSO, Gradient
from modules.utils import folds_run, sklearn_folds_run

#%load_ext autoreload
#%autoreload 2


# In[3]:


dtypes = {
    'tau1': float, 'tau2': float, 'tau3': float, 'tau4': float,
    'p1': float, 'p2': float,'p3': float, 'p4': float, 
    'g1': float, 'g2': float, 'g3': float, 'g4': float}

EletricalGrid = pd.read_csv('dataset/Data_for_UCI_named.csv',
                            dtype=dtypes, nrows=300).drop('stab', axis=1)


# In[4]:


X, y = EletricalGrid.drop('stabf',axis=1), EletricalGrid.stabf 


# In[5]:


my_pso_options = {
    'optimization_func':PSO, 
    'max_iter':100, 'k':5, 
    'cpu_count':cpu_count() - 1,
    'my_pso':True,
    'swarm_size':50
}

not_my_pso_options = {
    'optimization_func':PSO, 
    'max_iter':100, 'k':5, 
    'cpu_count':None,
    'my_pso':False,
    'swarm_size':50
}


# In[6]:


results = folds_run(X, y, model_class=NCA, model_options=my_pso_options)
for i in results:
    print(f'{i} mean = {np.array(results[i]).mean()}')


# In[ ]:


results = folds_run(X, y, model_class=NCA, model_options=not_my_pso_options)
for i in results:
    print(f'{i} mean = {np.array(results[i]).mean()}')


# In[ ]:


#nca_sklearn = NeighborhoodComponentsAnalysis()
#sklearn_options = {'n_components':None, 'max_iter':1000,'random_state':42}


# In[ ]:


#npX, npy = np.asarray(X), np.asarray(y)
#results = sklearn_folds_run(npX, npy, 100, model_class=NeighborhoodComponentsAnalysis, model_options=sklearn_options)
#for i in results:
#    print(f'{i} mean = {np.array(results[i]).mean()}')

