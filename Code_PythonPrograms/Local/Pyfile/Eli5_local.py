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


# In[3]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[4]:


row_no_to_interpret = 954
data_for_prediction = Xtest.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# In[9]:


#%%time
eli5.explain_prediction_dfs(RFModel, doc = data_for_prediction)


# In[5]:


#%%time
eli5.show_prediction(RFModel, doc = data_for_prediction)


# In[6]:


#%%time
eli5.explain_prediction(RFModel, doc = data_for_prediction)

