#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from piBreakDown.Attributions import Attributions
from piBreakDown.PlotUtils import PlotUtils


# In[11]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
Xtrain = pickle.load(open('./Data/pickledtraintestdata/X_train.pkl', 'rb'))
ytrain = pickle.load(open('./Data/pickledtraintestdata/y_train.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[12]:


row_no_to_interpret = 954
data_for_prediction = Xtest.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# In[13]:


#%%time
# plot for class 1 - fraud
attr = Attributions(model = RFModel, data = Xtrain, target_label = 'Class')
results = attr.local_attributions(data_for_prediction,  classes_names = [0,1])
print(results)
PlotUtils.plot_contribution(results, plot_class = 1) # class to plot


# In[5]:


#%%time
# plot for class 0 - non fraud
attr = Attributions(RFModel, Xtrain, 'Class')
results = attr.local_attributions(data_for_prediction,  classes_names = [0,1])
PlotUtils.plot_contribution(results, plot_class = 0) # class to plot


# In[13]:


#model: scikit-learn model
    #a model to be explained, with `fit` and `predict` functions
#data: pandas.DataFrame
    #data that was used to train model
#target_label: str
    #label of target variable

