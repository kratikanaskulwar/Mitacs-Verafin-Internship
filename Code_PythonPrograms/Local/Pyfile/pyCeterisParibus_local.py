#!/usr/bin/env python
# coding: utf-8

# In[16]:


# This approach calculates individual_variable_profile for a given observation. For Example - if grid_point parameter is 
# given the value of 100, it changes the value of 1 variable 100 times, keeping all other features value constant and gets the model prediction.
#In plot it shows those prediction value for number of grid_points times.
# S0 if total number of variables are 29 and grid_points was passed with a value of 100, 
#it will calculate 2900 profiles (100 for each variable). Plots will show how prediction changes as a value of particular variable changes.


# In[ ]:





# In[ ]:


import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.profiles import CeterisParibus
from ceteris_paribus.plots.plots import plot, plot_notebook

Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))
Xtrain = pickle.load(open('./Data/pickledtraintestdata/X_train.pkl', 'rb'))
ytrain = pickle.load(open('./Data/pickledtraintestdata/y_train.pkl', 'rb'))


# In[ ]:


row_no_to_interpret = 954 #1132
data_for_prediction = Xtest.iloc[row_no_to_interpret] 
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)
#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# In[18]:


#%%time
data = np.array(Xtrain)
yt = np.array(ytrain)
labels = yt.ravel()
variable_names = Xtest.columns

predict_function = lambda X: RFModel.predict_proba(X)[::, 1]
explainer_rf = explain(RFModel, variable_names, data, labels, predict_function=predict_function)

#to calulate profiles for selected variables add variables parameter in below line (variables = = ['V4', 'V14'], if no value is provided, profiles for all features will be calculated)
#add grid_points parameter if required, by default grid_points = 101 (101 times value of a variable will be changed and prediction is noted keeping all other feature values same.)
cp_profile = individual_variable_profile(explainer_rf, Xtest.iloc[954], y=ytest.iloc[954],grid_points = 100)

#plot profile for all variables
plot(cp_profile, show_profiles=True, show_rugs=True, show_observations = True, show_residuals = True)

#plot profile for one variable
plot(cp_profile,selected_variables = ["V14"])

#cp_profile.profile gets the dataframe of calculated profiles
#print(cp_profile.profile)
#print the profile using print_profile() function


# In[ ]:


# USe these settings to get the untruncated jupyter cell output dataframe
'''pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)'''

