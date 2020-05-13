#!/usr/bin/env nextflow

//-------------------------channels----------------------------//

channel1 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1X.pickle')
channel11 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0X.pickle')

ch2 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/Python_RF.pickle')
ch2.into{channel2; channel22}

ch3 = Channel.fromPath('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/X_train.pkl')
ch3.into{channel3; channel33}

ch4 = Channel.fromPath('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/y_train.pkl')
ch4.into{channel4; channel44}

channel5 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1Y.pickle')
channel55 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0Y.pickle')


//-------------------------channels----------------------------//

process getpyCeterisParibusInterpretation_fraud{
  input:
    file p1f1 from channel1
    file p1f2 from channel2
    file p1f3 from channel3
    file p1f4 from channel4
    file p1f5 from channel5

  output:
    file "pyCP_ALL_NORM_fraud.csv" into p1result 

	script:
 	"""
#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.profiles import CeterisParibus
from ceteris_paribus.plots.plots import plot

loaded_class1x = pickle.load(open('$p1f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p1f2', 'rb'))
loaded_xtrain = pickle.load(open('$p1f3', 'rb'))
loaded_ytrain = pickle.load(open('$p1f4', 'rb'))
loaded_class1y = pickle.load(open('$p1f5', 'rb'))

dataxt = np.array(loaded_xtrain)
yt = np.array(loaded_ytrain)
datayt = yt.ravel()
variable_names = loaded_class1x.columns

predict_fn = lambda X: loaded_RFmodel.predict_proba(X)[::, 1]
explainer_rf = explain(loaded_RFmodel, variable_names, dataxt, datayt, predict_function=predict_fn)
    
datatoCP = loaded_class1x[0:150]
column_list = loaded_class1x.columns
R = pd.DataFrame()	
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatoCP)):

    cp1 = pd.DataFrame(columns=['Feature', 'yhatRange'])

    for j in range(len(column_list)):

        cpr = individual_variable_profile(explainer_rf, loaded_class1x.iloc[i], y=loaded_class1y.iloc[i], grid_points = 200, variables = [column_list[j]])
        cpdf = cpr.profile
        cpdf1 = cpdf[['_yhat_']]
        cpdf1 = cpdf1.sort_values(by=['_yhat_'], ascending =False)
        yhatmin = cpdf1['_yhat_'].min()
        yhatmax = cpdf1['_yhat_'].max()
        yhatrange = yhatmax - yhatmin
        cp1 = cp1.append({'Feature': str(column_list[j]), 'yhatRange': yhatrange}, ignore_index=True)

    cpdf1_array = cp1.to_numpy()
    cpdf1_sortArray = cpdf1_array[cpdf1_array[:,1].argsort()]
    cpdf1_sortedArray = cpdf1_sortArray[::-1]
    df = pd.DataFrame(cpdf1_sortedArray, columns=['Feature', 'yhatRange'])
    R = pd.concat([R, df], axis=1)

    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['yhatRange'].abs()
    df['Abs_contribution'] = contrAbs
    absSum = df['Abs_contribution'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_contribution']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)

R.to_csv('pyCP_fraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('pyCP_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('pyCP_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")

"""
}

p1result.collectFile(name: file("pyCP_ALL_NORM_fraud.csv")).set{setResult1}



process getpyCeterisParibusInterpretation_nonfraud{
  input:
    file p2f1 from channel11
    file p2f2 from channel22
    file p2f3 from channel33
    file p2f4 from channel44
    file p2f5 from channel55

  output:
    file "pyCP_ALL_NORM_nonfraud.csv" into p2result 

    script:
    """
#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ceteris_paribus.explainer import explain
from ceteris_paribus.profiles import individual_variable_profile
from ceteris_paribus.profiles import CeterisParibus
from ceteris_paribus.plots.plots import plot

loaded_class0x = pickle.load(open('$p2f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p2f2', 'rb'))
loaded_xtrain = pickle.load(open('$p2f3', 'rb'))
loaded_ytrain = pickle.load(open('$p2f4', 'rb'))
loaded_class0y = pickle.load(open('$p2f5', 'rb'))

dataxt = np.array(loaded_xtrain)
yt = np.array(loaded_ytrain)
datayt = yt.ravel()
variable_names = loaded_class0x.columns

predict_fn = lambda X: loaded_RFmodel.predict_proba(X)[::, 0]
explainer_rf = explain(loaded_RFmodel, variable_names, dataxt, datayt, predict_function=predict_fn)
    
datatoCP = loaded_class0x[0:150]
column_list = loaded_class0x.columns
R = pd.DataFrame()  
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatoCP)):

    cp1 = pd.DataFrame(columns=['Feature', 'yhatRange'])

    for j in range(len(column_list)):

        cpr = individual_variable_profile(explainer_rf, loaded_class0x.iloc[i], y=loaded_class0y.iloc[i], grid_points = 200, variables = [column_list[j]])
        cpdf = cpr.profile
        cpdf1 = cpdf[['_yhat_']]
        cpdf1 = cpdf1.sort_values(by=['_yhat_'], ascending =False)
        yhatmin = cpdf1['_yhat_'].min()
        yhatmax = cpdf1['_yhat_'].max()
        yhatrange = yhatmax - yhatmin
        cp1 = cp1.append({'Feature': str(column_list[j]), 'yhatRange': yhatrange}, ignore_index=True)

    cpdf1_array = cp1.to_numpy()
    cpdf1_sortArray = cpdf1_array[cpdf1_array[:,1].argsort()]
    cpdf1_sortedArray = cpdf1_sortArray[::-1]
    df = pd.DataFrame(cpdf1_sortedArray, columns=['Feature', 'yhatRange'])
    R = pd.concat([R, df], axis=1)

    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['yhatRange'].abs()
    df['Abs_contribution'] = contrAbs
    absSum = df['Abs_contribution'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_contribution']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)


R.to_csv('pyCP_nonfraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('pyCP_ALL_NORM_nonfraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('pyCP_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")

""" 
}

p2result.collectFile(name: file("pyCP_ALL_NORM_nonfraud.csv")).set{setResult2}

workflow.onComplete {   ss
println(
"""
Pipeline execution summary
---------------------------
Run as : ${workflow.commandLine}
Completed at: ${workflow.complete}
Duration : ${workflow.duration}
Success : ${workflow.success}
workDir : ${workflow.workDir} 
exit status : ${workflow.exitStatus}
""")
}


