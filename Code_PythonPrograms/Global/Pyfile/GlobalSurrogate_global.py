#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[6]:


Xtest = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
ytest = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


# ### Predict with random forest, visulaize a single decision tree from random forest

# In[9]:


#https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
    
new_xtest_target = RFModel.predict(Xtest)

estimator = RFModel.estimators_[599]

# Export as dot file
export_graphviz(estimator, out_file='treerf.dot', feature_names = Xtest.columns,  max_depth=4, class_names = ['0', '1'])

call(['dot', '-Tpng', 'treerf.dot', '-o', 'treerf.png', '-Gdpi=600'])

Image(filename = 'treerf.png')


# ### Create a surrogate decision tree model, fit it with new target to approximate prediction of random forest

# In[10]:


#%%time
# defining the interpretable decision tree model
surrogate_model = DecisionTreeClassifier(max_depth=4, random_state=10)
# fitting the surrogate decision tree model using the training set and new target
surrogate_model.fit(Xtest,new_xtest_target)


# ### Visulaize the surrogate decision tree

# In[9]:


#%%time
#https://www.analyticsvidhya.com/blog/2019/08/decoding-black-box-step-by-step-guide-interpretable-machine-learning-models-python/

d_tree = tree.export_graphviz(surrogate_model, out_file='treedt.dot', feature_names=Xtest.columns, class_names = ['0', '1'])

# converting the dot image to png format
get_ipython().system('dot -Tpng treedt.dot -o treedt.png')

#plotting the decision tree
image = plt.imread('treedt.png')
plt.figure(figsize=(25,25))
plt.imshow(image)


# In[10]:


ytest['Class'].unique()


# ### For 300 observations

# In[9]:


#%%time
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

#https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c
    
new_xtest_target = RFModel.predict(dataX)
estimator = RFModel.estimators_[599]
export_graphviz(estimator, out_file='treerf.dot', feature_names = dataX.columns,  max_depth=4, class_names = ['0', '1'])
#call(['dot', '-Tpng', 'treerf.dot', '-o', 'treerf.png', '-Gdpi=600'])
Image(filename = 'treerf.png')


# In[10]:


#%%time
# defining the interpretable decision tree model
surrogate_model = DecisionTreeClassifier(max_depth=4, random_state=10)
# fitting the surrogate decision tree model using the training set and new target
surrogate_model.fit(dataX,new_xtest_target)

d_tree = tree.export_graphviz(surrogate_model, out_file='treedt.dot', feature_names=dataX.columns, class_names = ['0', '1'])

get_ipython().system('dot -Tpng treedt.dot -o treedt.png')

image = plt.imread('treedt.png')
plt.figure(figsize=(25,25))
plt.imshow(image)

