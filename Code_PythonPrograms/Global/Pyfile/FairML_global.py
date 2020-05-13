#!/usr/bin/env python
# coding: utf-8

# In[17]:


import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Display in jupyter notebook
from IPython.display import Image
# Convert to png using system command (requires Graphviz)
from subprocess import call

from fairml import audit_model
from fairml import plot_dependencies


# In[18]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# In[19]:


#%%time
total1, _ = audit_model(RFModel.predict, Xtest)

# print feature importance
print(total1)


# In[20]:


df = pd.DataFrame(total1)


# In[21]:


#%%time
# generate feature dependence plot
fig = plot_dependencies(
    total1.median(),
    reverse_values=False,
    title="FairML feature dependence",
    fig_size=(8, 8)
)


# In[22]:


plt.savefig("fairml_ldp.eps", transparent=True, bbox_inches='tight')


# In[23]:


#FairlML - features in Red bar shows the high contribution according to fairML. (FairML uses orthogonal projection (when features are correlated) to perturb the input space to measure the dependence of model on each feature)
#it uses orthogonal projection in order to remove the linear dependence between variables. For model F and features X and Y, input for X is perturbed and Y is made orthognal to X. Distnace between the output of the original 
#data and transformed/ perturbed data shows how important the variable is.


# ### For 300 observations

# In[24]:


#%%time
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fairml import audit_model
from fairml import plot_dependencies

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

tota1, _ = audit_model(RFModel.predict, dataX)

fig = plot_dependencies(
    tota1.median(),
    reverse_values=False,
    title="FairML feature dependence",
    fig_size=(8, 8)
)


# In[ ]:




