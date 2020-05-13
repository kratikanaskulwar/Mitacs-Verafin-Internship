#!/usr/bin/env python



import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel


# In[5]:


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


# In[8]:


#%%time
interpreter = Interpretation(Xtest, feature_names=Xtest.columns)


# In[9]:


#%%time
model = InMemoryModel(RFModel.predict_proba, examples=Xtest)


# In[6]:


#%%time
interpreter = Interpretation(Xtest, feature_names=Xtest.columns)

model = InMemoryModel(RFModel.predict_proba, examples=Xtest)

plots = interpreter.feature_importance.plot_feature_importance(model, ascending = False)


# In[7]:


#%%time
pyint_model = InMemoryModel(RFModel.predict_proba, examples=Xtest, target_names=[0,1])


# In[8]:


#%%time
axes_list = interpreter.partial_dependence.plot_partial_dependence(['V14'],
                                                                       pyint_model, 
                                                                       grid_resolution=30, 
                                                                       with_variance=True,
                                                                       figsize = (9, 7))


# In[9]:


axes_list = interpreter.partial_dependence.plot_partial_dependence(['V4'],
                                                                       pyint_model, 
                                                                       grid_resolution=30, 
                                                                       with_variance=True,
                                                                       figsize = (9, 7))



# In[10]:


axes_list = interpreter.partial_dependence.plot_partial_dependence(['V12'],
                                                                       pyint_model, 
                                                                       grid_resolution=30, 
                                                                       with_variance=True,
                                                                       figsize = (9, 7))


# In[11]:


axes_list = interpreter.partial_dependence.plot_partial_dependence(['V10'],
                                                                       pyint_model, 
                                                                       grid_resolution=30, 
                                                                       with_variance=True,
                                                                       figsize = (9, 7))


# In[12]:


from sklearn.metrics import f1_score
print("RF -> F1 Score: {1}". format('RF', f1_score(ytest, RFModel.predict(Xtest))))


# In[13]:


#%%time
model = InMemoryModel(RFModel.predict_proba, examples=Xtest, target_names=[0, 1])
                                                                                  
interpreter.partial_dependence.plot_partial_dependence([('V14', 'V4')], model, grid_resolution=10)


# In[14]:


#%%time
interpreter.partial_dependence.plot_partial_dependence([('V14', 'V12')], model, grid_resolution=10)


# In[15]:


#%%time
model_no_proba = InMemoryModel(RFModel.predict, 
                      examples=Xtest, 
                      unique_values=[0,1])
plots = interpreter.feature_importance.plot_feature_importance(model_no_proba, ascending = False)


# In[ ]:




