#!/usr/bin/env python

# ### for fraud observations



#%%time
loaded_class1x = pickle.load(open('./Data/PickledData300/class1X.pickle', 'rb'))

columnnames = np.array(loaded_class1x.columns)
datatopiBreakDown = loaded_class1x[0:150]

order = None
yhatpred = RFModel.predict_proba(Xtrain.loc[:,Xtrain.columns != 'Class'])
baseline_yhat = yhatpred.mean(axis = 0)

x = loaded_class1x.columns.tolist()
R = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()

for i in range(len(datatopiBreakDown)):
    print(i)
    target_yhat = RFModel.predict_proba(loaded_class1x.iloc[i].loc[Xtrain.columns != 'Class'].values.reshape(1,-1))[0]
    classes_names = list(range(0,len(target_yhat)))
    average_yhats = attr._calculated_1d_changes(Xtrain.loc[:, Xtrain.columns != 'Class'],loaded_class1x.iloc[i].loc[Xtrain.columns != 'Class'], classes_names)
    diffs_1d = (average_yhats.subtract(baseline_yhat)**2).mean(axis = 1)
    feature_path = attr._create_ordered_path(diffs_1d, order)

    result = attr._calculate_contributions_along_path(data = Xtrain.loc[:,Xtrain.columns != 'Class'],new_observation = loaded_class1x.iloc[i], feature_path = feature_path, keep_distributions = False, label = 'Class',baseline_yhat = baseline_yhat, target_yhat = target_yhat, classes_names = classes_names)

    dfcontr = result.contribution
    
    newdf = pd.DataFrame(data=dfcontr)
    newdf.columns = newdf.columns.astype(str)
    
    a = newdf.reset_index()
    #drop first and last row
    a1 = a.tail(-1)
    a2= a1[:-1]

    a2.rename(columns={'index':'Features','0':'Class0Contribution','1':'Class1Contribution'}, inplace=True)
    a3 = a2.drop(['Class0Contribution'], axis=1)
     
    piBr_array = a3.to_numpy()
    piBr_sortArray = piBr_array[piBr_array[:,1].argsort()]
    piBr_sortedArray = piBr_sortArray[::-1]
    df = pd.DataFrame(data=piBr_sortedArray, columns=["Feature", "Contribution"])
    R = pd.concat([R, df], axis=1)
    
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
    
#R.to_csv('piBreakdownop_fraud.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('piBreakdownop_NORM_ALL_row_fraud.csv', index = False, header=True, sep = "\t")
#R_norm_TOP5.to_csv('piBreakdownop_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")


# ### for non fraud observations

# In[ ]:


#%%time
loaded_class0x = pickle.load(open('./Data/PickledData300/class0X.pickle', 'rb'))

columnnames = np.array(loaded_class0x.columns)
datatopiBreakDown = loaded_class0x[0:150]

order = None
yhatpred = RFModel.predict_proba(Xtrain.loc[:,Xtrain.columns != 'Class'])
baseline_yhat = yhatpred.mean(axis = 0)

x = loaded_class0x.columns.tolist()
R = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()

for i in range(len(datatopiBreakDown)):
    print(i)
    target_yhat = RFModel.predict_proba(loaded_class0x.iloc[i].loc[Xtrain.columns != 'Class'].values.reshape(1,-1))[0]
    classes_names = list(range(0,len(target_yhat)))
    average_yhats = attr._calculated_1d_changes(Xtrain.loc[:, Xtrain.columns != 'Class'],loaded_class0x.iloc[i].loc[Xtrain.columns != 'Class'], classes_names)
    diffs_1d = (average_yhats.subtract(baseline_yhat)**2).mean(axis = 1)
    feature_path = attr._create_ordered_path(diffs_1d, order)

    result = attr._calculate_contributions_along_path(data = Xtrain.loc[:,Xtrain.columns != 'Class'],new_observation = loaded_class0x.iloc[i], feature_path = feature_path, keep_distributions = False, label = 'Class',baseline_yhat = baseline_yhat, target_yhat = target_yhat, classes_names = classes_names)

    dfcontr = result.contribution
    
    newdf = pd.DataFrame(data=dfcontr)
    newdf.columns = newdf.columns.astype(str)
    
    a = newdf.reset_index()
    #drop first and last row
    a1 = a.tail(-1)
    a2= a1[:-1]

    a2.rename(columns={'index':'Features','0':'Class0Contribution','1':'Class1Contribution'}, inplace=True)
    #drop class1 contributions
    a3 = a2.drop(['Class1Contribution'], axis=1)
     
    piBr_array = a3.to_numpy()
    piBr_sortArray = piBr_array[piBr_array[:,1].argsort()]
    piBr_sortedArray = piBr_sortArray[::-1]
    df = pd.DataFrame(data=piBr_sortedArray, columns=["Feature", "Contribution"])
    R = pd.concat([R, df], axis=1)
    
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

#R.to_csv('piBreakdownop_nonfraud.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('piBreakdownop_NORM_ALL_row_nonfraud.csv', index = False, header=True, sep = "\t")
#R_norm_TOP5.to_csv('piBreakdownop_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")


# In[ ]:




