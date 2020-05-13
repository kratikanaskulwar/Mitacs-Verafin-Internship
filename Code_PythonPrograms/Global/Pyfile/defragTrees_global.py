#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#%%time
import matplotlib.pyplot as plt
#%matplotlib inline
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))
loaded_class1y = pickle.load(open('./Data/PickledData300/class1y.pickle', 'rb'))
loaded_class0x = pickle.load(open('.Data/PickledData300/class0X.pickle', 'rb'))
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

from defragTrees import DefragModel
splitter = DefragModel.parseSLtrees(RFModel)

mdl = DefragModel(modeltype='classification', maxitr=100, verbose=0)

Kmax = 10

mdl.fit(dataX.values, dataY.values, splitter, Kmax)

score, cover, coll = mdl.evaluate(dataX.values, dataY.values)

print('<< defragTrees >>')
print('----- Evaluated Results -----')
print('Test Error = %f' % (score,))
print('Test Coverage = %f' % (cover,))
print('Overlap = %f' % (coll,))

print('----- Found Rules -----')
print(mdl)





