#!/usr/bin/env nextflow

//-------------------------channels----------------------------//



channel1 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1X.pickle')
channel11 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0X.pickle')
ch2 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/X_train.pkl')
ch2.into{channel2; channel22}
ch3 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/Python_RF.pickle')
ch3.into{channel3; channel33}

process getLimeInterpretation_fraud{
  input:
    file p1f1 from channel1
    file p1f2 from channel2
    file p1f3 from channel3

  output:
    file "LIME_ALL_NORM_fraud.csv" into p1result 

	script:
 	"""
#!/usr/bin/env python3
import lime
import lime.lime_tabular
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

loaded_class1x = pickle.load(open('$p1f1', 'rb'))
loaded_xtrain = pickle.load(open('$p1f2', 'rb'))
loaded_RFmodel = pickle.load(open('$p1f3', 'rb'))

datatolime = loaded_class1x[0:150]
explainer = lime.lime_tabular.LimeTabularExplainer(loaded_xtrain.values,feature_names=loaded_xtrain.columns, verbose=True, feature_selection = 'none')

result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatolime)):
    exp1 = explainer.explain_instance(loaded_class1x.iloc[i], loaded_RFmodel.predict_proba, num_features=29, distance_metric='euclidean',num_samples=10000)
    exp1.as_list(label=1)
    dflime= pd.DataFrame(data=exp1.as_list(), columns = ['Rule','Contribution'])
    dflime['Rule'] = dflime['Rule'].str.replace('[><=]', '')
    dflime['Feature']  = dflime['Rule'].str.extract('([VA][0-9][0-9].)')
    dflime['Feature']  = dflime['Rule'].str.extract('([VA][0-9].)')

    dflime1 = dflime.drop(['Rule'], axis=1)
    cols = dflime1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = dflime[cols]
    df = df.replace(np.nan, 'Amount', regex=True)
    result = pd.concat([result, df], axis=1)

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
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)
    
result.to_csv('LIME_fraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('LIME_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('LIME_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")

"""
}

p1result.collectFile(name: file("LIME_ALL_NORM_fraud.csv")).set{setResult1}


process getLimeInterpretation_nonfraud{
  input:
    file p2f1 from channel11
    file p2f2 from channel22
    file p2f3 from channel33

  output:
    file "LIME_ALL_NORM_nonfraud.csv" into p2result 

    script:
    """
#!/usr/bin/env python3
import lime
import lime.lime_tabular
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

loaded_class0x = pickle.load(open('$p2f1', 'rb'))
loaded_xtrain = pickle.load(open('$p2f2', 'rb'))
loaded_RFmodel = pickle.load(open('$p2f3', 'rb'))

datatolime = loaded_class0x[0:150]
explainer = lime.lime_tabular.LimeTabularExplainer(loaded_xtrain.values,feature_names=loaded_xtrain.columns, verbose=True, feature_selection = 'none')

result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatolime)):
    exp1 = explainer.explain_instance(loaded_class0x.iloc[i], loaded_RFmodel.predict_proba, num_features=29, distance_metric='euclidean',num_samples=10000)
    exp1.as_list()
    dflime= pd.DataFrame(data=exp1.as_list(), columns = ['Rule','Contribution'])
    dflime['Rule'] = dflime['Rule'].str.replace('[><=]', '')
    dflime['Feature']  = dflime['Rule'].str.extract('([VA][0-9][0-9].)')
    dflime['Feature']  = dflime['Rule'].str.extract('([VA][0-9].)')

    dflime1 = dflime.drop(['Rule'], axis=1)
    cols = dflime1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = dflime[cols]
    df = df.replace(np.nan, 'Amount', regex=True)
    result = pd.concat([result, df], axis=1)

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
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)
    
result.to_csv('LIME_nonfraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('LIME_ALL_NORM_nonfraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('LIME_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")

"""
}

p2result.collectFile(name: file("LIME_ALL_NORM_nonfraud.csv")).set{setResult2}



workflow.onComplete {
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



