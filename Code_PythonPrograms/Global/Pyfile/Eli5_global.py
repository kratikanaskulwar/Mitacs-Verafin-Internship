#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install eli5


# In[ ]:


import eli5
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


# In[2]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[4]:


#%%time
eli5.explain_weights_dfs(RFModel,feature_names = Xtest.columns.tolist())


# In[3]:


#%%time
eli5.show_weights(RFModel)


# In[4]:


#%%time
eli5.show_weights(RFModel, feature_names = Xtest.columns.tolist())


# In[5]:


#%%time
eli5.explain_weights(RFModel)


# In[6]:


#%%time
eli5.explain_weights_sklearn(RFModel,feature_names = Xtest.columns.tolist())


# In[7]:


#%%time
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(RFModel, random_state=123).fit(Xtest, ytest)
eli5.show_weights(perm, feature_names = Xtest.columns.tolist(),top=24)


# ### For 300 observations

# In[8]:


#%%time
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))
loaded_class1y = pickle.load(open('./Data/PickledData300/class1y.pickle', 'rb'))
loaded_class0x = pickle.load(open('./Data/PickledData300/class0X.pickle', 'rb'))
loaded_class0y = pickle.load(open('./Data/PickledData300/class0y.pickle', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))

eli5150fx = loaded_class1x[0:150]
eli5150fy = loaded_class1y[0:150]

eli5150nfx = loaded_class0x[0:150]
eli5150nfy = loaded_class0y[0:150]

from sklearn.utils import shuffle
x = pd.concat([eli5150fx,eli5150nfx], axis = 0)
y = pd.concat([eli5150fy,eli5150nfy], axis = 0)
xy = y = pd.concat([x,y], axis = 1)
xy = shuffle(xy)
dataX = xy.copy().drop(['Class'],axis=1)
dataY = xy['Class'].copy()

permc = PermutationImportance(RFModel, random_state=123).fit(dataX, dataY)
eli5.explain_weights_df(permc, feature_names = dataX.columns.tolist(),top=5)

