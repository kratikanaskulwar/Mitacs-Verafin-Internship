#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install git+https://github.com/marcotcr/lime
#!pip install dill


# In[1]:


import lime
import lime.lime_tabular
import pickle
import dill
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[4]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[5]:


row_no_to_interpret = 954
data_for_prediction = Xtest.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# In[6]:


explainer = lime.lime_tabular.LimeTabularExplainer(Xtrain.values,feature_names=Xtrain.columns, verbose=True, feature_selection = 'none')
#filenamel = 'limeexplner.dill'
#dill.dump(explainer, open(filenamel, 'wb'))


# In[16]:


exp = explainer.explain_instance(data_for_prediction, RFModel.predict_proba, num_features=29, distance_metric='euclidean',num_samples=10000)


# In[17]:


#print(exp.as_list())
#print(exp.as_map())
#exp.as_pyplot_figure(label=1)
#exp.save_to_file('./LIME.png',predict_proba=True, show_predicted_value=True)


# In[18]:


#%%time
exp.show_in_notebook(show_table=True, show_all=False)


# In[18]:


#exp.available_labels()
#%matplotlib inline
fig = exp.as_pyplot_figure();


# In[135]:


#middle plot shows that for example when v4 > 0.74 it contributes for prediction being 1 and the weight assigned to v4 > 0.74 is 0.06
#3rd plot shows the actual values of the features of the instance
#firsr plot shows the prediction

