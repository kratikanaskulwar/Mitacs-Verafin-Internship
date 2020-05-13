#!/usr/bin/env python
# coding: utf-8

# In[16]:


'''Main'''
import sys
import pickle
import numpy as np
import pandas as pd
import pylab as pl
import os

'''Data Viz'''
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

'''Data Prep'''
from sklearn import preprocessing as pp 
from scipy.stats import pearsonr 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.metrics import precision_recall_curve, average_precision_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report 
from scipy import interp
from sklearn import metrics
from IPython.display import display
from sklearn.model_selection import cross_val_predict, GridSearchCV, StratifiedKFold, cross_val_score, KFold, train_test_split
from sklearn.metrics import average_precision_score, r2_score, roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, auc, plot_precision_recall_curve, make_scorer

'''Algos'''
from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv("./Data/cleandataH.txt",sep='\t')
dataX1 = data.copy().drop(['Class'],axis=1)
dataX = dataX1.copy().drop(['Time'],axis=1)
dataY = data['Class'].copy()

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=2018, stratify=dataY)


def findbestRFparameter():
    rfclf=RandomForestClassifier(random_state=0,n_jobs = -1)
    param_grid = { 
    'n_estimators': [600,800,1000],
    'max_features': ['sqrt','auto'],
    'oob_score' : ['True'],
    'class_weight' : ['balanced'],
    'max_depth': range(1,10,2)
    }
    print(param_grid)
    grid1 = GridSearchCV(rfclf, param_grid, iid= False, cv=5, scoring='accuracy')
    grid1.fit(X_train, y_train)
    print(grid1.best_score_)
    print(grid1.best_params_)
    print(grid1.best_estimator_)
    return grid1.best_estimator_

RFC = findbestRFparameter()



k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)

predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])
model = RFC


# In[20]:


#Train(Cross Validate)

for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                          y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:],         X_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index],         y_train.iloc[cv_index]
    
    model.fit(X_train_fold, y_train_fold)
    predictionsBasedOnKFolds.loc[X_cv_fold.index,:] =         model.predict_proba(X_cv_fold)  

print(predictionsBasedOnKFolds.shape)
print(predictionsBasedOnKFolds.loc[:,1])


# ### Save/dump model

# In[21]:


filename = 'Python_RF.pickle'
pickle.dump(model, open(filename, 'wb'))


# In[22]:


#PR curve for cross validation

preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsRandomForests = preds.copy()

precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],
                                                       preds['prediction'])

average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])
plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve (cross validation) : Average Precision = {0:0.2f}'.format(
          average_precision))
plt.savefig('Precision-Recallcurve_for_CV_Final_RF_python.png')
print(preds)

print("precision",np.mean(precision))
print("recall",np.mean(recall))

auc = auc(recall, precision)

print(auc)

