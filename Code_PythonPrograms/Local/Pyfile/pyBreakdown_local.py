#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyBreakDown.explainer import Explainer
from pyBreakDown.explanation import Explanation


# In[2]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[3]:


row_no_to_interpret = 954
data_for_prediction = Xtest.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# In[5]:


#%%time
#make explainer object
exp = Explainer(clf=RFModel, data=Xtrain, colnames=Xtrain.columns)


# In[7]:


filename2 = 'pybreakdownExplainer.pkl'
pickle.dump(exp, open(filename2, 'wb'))


# ### Step-up approach

# In[24]:


#%%time
#make explanation object that contains all information
pybrpkl = pickle.load(open('./pybreakdownExplainer.pkl', 'rb'))
explanation = pybrpkl.explain(observation=data_for_prediction,direction="up")

#explanation = exp.explain(observation=data_for_prediction,direction="up")


# In[4]:


#to get explanation in dataframe
#a = explanation._attributes


# In[26]:


#%%time
explanation.text(fwidth=40, contwidth=40, cumulwidth = 40, digits=4)


# In[27]:


#%%time
#customize height, width and dpi of plot
explanation.visualize(figsize=(20,15),dpi=100)


# In[28]:


#%%time
#for different baselines than zero
explanation2 = pybrpkl.explain(observation=data_for_prediction,direction="up",useIntercept=True)  # baseline==intercept


# In[29]:


#%%time
explanation2.text(fwidth=40, contwidth=40, cumulwidth = 40, digits=4)


# ### Step-down approach

# In[31]:


#%%time
explanationd = exp.explain(observation=data_for_prediction,direction="down")


# In[32]:


#%%time
explanationd.text(fwidth=40, contwidth=40, cumulwidth = 40, digits=4)


# In[33]:


#%%time
#make explanation object that contains all information
explanationd.visualize(figsize=(20,15),dpi=100)

