#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
shap.initjs()


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
explainer = shap.TreeExplainer(RFModel)


# In[ ]:


#%%time
shap_values = explainer.shap_values(Xtest)


# In[6]:


#%%time
#SHAP summary plot for class 1
shap.summary_plot(shap_values = shap_values[1], features= Xtest, feature_names=Xtest.columns.tolist())


# In[7]:


#SHAP summary plot for class 0
shap.summary_plot(shap_values = shap_values[0], features= Xtest, feature_names=Xtest.columns.tolist())


# In[8]:


#%%time
shap.summary_plot(shap_values = shap_values, features= Xtest, feature_names=Xtest.columns.tolist())


# In[9]:


#%%time
shap.summary_plot(shap_values = shap_values[1], features= Xtest, feature_names=Xtest.columns.tolist(), plot_type = 'bar')


# In[10]:


#%%time
shap.dependence_plot('V14', shap_values[1], Xtest , dot_size=25)


# In[11]:


#each dot reperesnts a single prediction(instance) from dataset.
#X axis is the value of the feature and y is the shap value.
#color of dot shows the feature value of the feature(v4) that may have an interaction with the feature(v14)
#left most dot in plot shows that when v14 has a higher negative value with the high value of v4, the chances of prediction being fraud (1) increases.
#The dot in the middle and topmost on y axis shows that prediction to being fraud has the highest probablity when v14 is around -5 and v4 has a high value.


# ### For 300 observations

# In[1]:


#%%time
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
shap.initjs()

loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))
loaded_class1y = pickle.load(open('./Data/PickledData300/class1y.pickle', 'rb'))
loaded_class0x = pickle.load(open('./Data/PickledData300/class0X.pickle', 'rb'))
loaded_class0y = pickle.load(open('./Data/PickledData300/class0y.pickle', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))

fx = loaded_class1x[0:150]
fy = loaded_class1y[0:150]

nfx = loaded_class0x[0:150]
nfy = loaded_class0y[0:150]

from sklearn.utils import shuffle
x = pd.concat([fx,nfx], axis = 0)
y = pd.concat([fy,nfy], axis = 0)
xy = y = pd.concat([x,y], axis = 1)
xy = shuffle(xy)
dataX = xy.copy().drop(['Class'],axis=1)
dataY = xy['Class'].copy()

explainer = shap.TreeExplainer(RFModel)

shap_value = explainer.shap_values(dataX, check_additivity=False)
shap.summary_plot(shap_values = shap_value, features= dataX, feature_names=dataX.columns.tolist())


# In[ ]:




