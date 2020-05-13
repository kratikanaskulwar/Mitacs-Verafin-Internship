#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Using Alibi anchor package

from anchor import utils
from anchor import anchor_tabular
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from alibi.explainers import AnchorTabular
from sklearn.metrics import accuracy_score

X_test = pickle.load(open('./Data/pickledtraintestdata/X_test.pkl', 'rb'))
y_test = pickle.load(open('./Data/pickledtraintestdata/y_test.pkl', 'rb'))
RFModel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))


row_no_to_interpret = 954
data_for_prediction = X_test.iloc[row_no_to_interpret]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
predicted_proba = RFModel.predict_proba(X_test)
RFprediction = RFModel.predict(data_for_prediction_array)

#print(data_for_prediction)
print("Real target", y_test.iloc[row_no_to_interpret])
print("Random forest Predicted", RFprediction)
print("Predict_proba", np.round_(predicted_proba[row_no_to_interpret], decimals=2))

predict_fn = lambda x: RFModel.predict(x)

explainer = AnchorTabular(predict_fn, X_train.columns)
explainer.fit(X_train.values[:,:], disc_perc=[25, 50, 75])
id = X_test.iloc[954].values.reshape(1, -1)
explainer.predict_fn(id)[0]

explanation = explainer.explain(id, threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation['names'])))
print('Precision: %.2f' % explanation['precision'])
print('Coverage: %.2f' % explanation['coverage'])



