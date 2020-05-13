#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[1]:





import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import shap
# load JS visualization code to notebook
shap.initjs()


# In[5]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# ### Get RF prediction for instance of interest

# In[6]:


row_no_to_interpret = 954
data_for_prediction = Xtest.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(Xtest)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", ytest.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))


# ### Create SHAP explainer using TreeExplainer

# In[9]:


#%%time
explainer = shap.TreeExplainer(RFModel)


# In[10]:


#%%time
shap_values = explainer.shap_values(data_for_prediction)


# In[11]:


shap_values[0]


# In[12]:


#%%time
import shap
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction,show=False)


# In[37]:


#%%time
#relationship between the value of a feature and the impact on the prediction
shap_values1 = explainer.shap_values(Xtest, check_additivity=False)


# In[38]:


#%%time
shap.summary_plot(shap_values1[1], features= data_for_prediction, color_bar=True,show=False)


# In[39]:


#%%time
#shap.summary_plot(shap_values1[0], features= data_for_prediction, color_bar=True)


# ### Waterfall plot

# In[66]:


#%%time
#954 for class 1
#https://github.com/slundberg/shap/blob/master/shap/plots/waterfall.py

shap.waterfall_plot(explainer.expected_value[1], shap_values[1], feature_names=Xtest.columns, max_display=10, show=True)


# In[40]:


#%%time
shap.decision_plot(base_value= explainer.expected_value[1], shap_values= shap_values[1], features= data_for_prediction, feature_names=Xtest.columns.tolist())


# In[17]:


np.sum(shap_values[1])
(explainer.expected_value[1])


# ### Misclassified instance

# In[13]:


import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
# load JS visualization code to notebook
shap.initjs()
dataX = pd.read_csv("./Data/dataX.txt",sep='\t')
dataY = pd.read_csv("./Data/dataY.txt",sep='\t')

row_no_to_interpret = 6
data_for_pred= dataX.iloc[row_no_to_interpret]
data_for_pred_array = data_for_pred.values.reshape(1, -1)

explainer = shap.TreeExplainer(RFModel)
shap_values = explainer.shap_values(data_for_pred_array,check_additivity=False)


# In[14]:


print("Real target", dataY.iloc[row_no_to_interpret])
print("Ranom Forest precited", int(RFModel.predict(data_for_pred.values.reshape(1, -1))))
print("Predicted probability", ((np.round(RFModel.predict_proba(data_for_pred.values.reshape(1, -1)), 
                                        decimals=2)).flatten())[1])


# In[15]:


#with original feature values
explainer = shap.TreeExplainer(RFModel)
shap_values6 = explainer.shap_values(data_for_pred_array,check_additivity=False)
shap.force_plot(explainer.expected_value[1], shap_values6[1], data_for_pred,show=False)


# In[24]:



shap.decision_plot(base_value= explainer.expected_value[1], shap_values= shap_values6[1], features= dataX.iloc[6], feature_names=dataX.columns.tolist())


# In[16]:


#change V14 value
data_for_pred[13] =  -3.4
explainer = shap.TreeExplainer(RFModel)
shap_values6 = explainer.shap_values(data_for_pred_array,check_additivity=False)
shap.force_plot(explainer.expected_value[1], shap_values6[1], data_for_pred,show=False)


# In[26]:


RFprediction = RFModel.predict(data_for_pred.values.reshape(1, -1))
RFprediction


# In[27]:


#data_for_pred[13] =  -1.818
shap.decision_plot(base_value= explainer.expected_value[1], shap_values= shap_values6[1], features= dataX.iloc[6], feature_names=dataX.columns.tolist())

