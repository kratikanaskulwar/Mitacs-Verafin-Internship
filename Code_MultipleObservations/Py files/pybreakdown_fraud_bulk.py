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


###  for fraud observations  with step - up approach

# In[12]:


#%%time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyBreakDown.explainer import Explainer
from pyBreakDown.explanation import Explanation

loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))
loaded_RFmodel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))
loaded_exp = pickle.load(open('./SavedExplainers/pybreakdownExplainer.pkl', 'rb'))

columnnames = np.array(loaded_class1x.columns)
datatopyBreak = loaded_class1x[0:150]

Result = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()

for i in range(len(datatopyBreak)):
    print(i)
    explanationu = loaded_exp.explain(observation=loaded_class1x.iloc[i],direction="down")
    pyBru = explanationu._attributes
    pyBru = pd.DataFrame(pyBru, columns = ['Feature', 'Value', 'Contribution', 'Cumulative'])
    pyBru = pyBru.drop(['Value', 'Cumulative'], axis=1)
    pyBru = pyBru.tail(-1)
    pyBru_array = pyBru.to_numpy()
    pyBru_sortArray = pyBru_array[pyBru_array[:,1].argsort()]
    pyBru_sortedArray = pyBru_sortArray[::-1]
    df = pd.DataFrame(data=pyBru_sortedArray, columns=["Feature", "Contribution"])
    Result = pd.concat([Result, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['Contribution'].abs()
    df['Abs_contribution'] = contrAbs
    absSum = df['Abs_contribution'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_contribution']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    R_norm = pd.concat([R_norm, df_norm], axis=1)
    R_norm_TOP5 = R_norm.head(5)
    
#Result.to_csv('pyBDown_fraud_originalweights.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('pyBDown_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
#R_norm_TOP5.to_csv('pyBDown_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")
print(R_norm_TOP5)

####  for fraud observations  with step - down approach

# In[14]:


#%%time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyBreakDown.explainer import Explainer
from pyBreakDown.explanation import Explanation

loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))
loaded_RFmodel = pickle.load(open('./SavedRFModel/Python_RF.pickle', 'rb'))
loaded_exp = pickle.load(open('./SavedExplainers/pybreakdownExplainer.pkl', 'rb'))

columnnames = np.array(loaded_class1x.columns)
datatopyBreak = loaded_class1x[0:150]

Result = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()

for i in range(len(datatopyBreak)):
    print(i)
    explanationu = loaded_exp.explain(observation=loaded_class1x.iloc[i],direction="up")
    pyBru = explanationu._attributes
    pyBru = pd.DataFrame(pyBru, columns = ['Feature', 'Value', 'Contribution', 'Cumulative'])
    pyBru = pyBru.drop(['Value', 'Cumulative'], axis=1)
    pyBru = pyBru.tail(-1)
    pyBru_array = pyBru.to_numpy()
    pyBru_sortArray = pyBru_array[pyBru_array[:,1].argsort()]
    pyBru_sortedArray = pyBru_sortArray[::-1]
    df = pd.DataFrame(data=pyBru_sortedArray, columns=["Feature", "Contribution"])
    Result = pd.concat([Result, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['Contribution'].abs()
    df['Abs_contribution'] = contrAbs
    absSum = df['Abs_contribution'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_contribution']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    R_norm = pd.concat([R_norm, df_norm], axis=1)
    R_norm_TOP5 = R_norm.head(5)
    
#Result.to_csv('pyBup_fraud_originalweights.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('pyBup_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
#R_norm_TOP5.to_csv('pyBup_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")
print(R_norm_TOP5)


# In[ ]:




