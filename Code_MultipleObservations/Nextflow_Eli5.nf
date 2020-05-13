#!/usr/bin/env nextflow

//-------------------------channels----------------------------//



channel1 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1X.pickle')
channel11 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0X.pickle')

ch2 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/Python_RF.pickle')
ch2.into{channel2;channel22}

process getEli5ExplainPredictionInterpretation_fraud{
  input:
    file p1f1 from channel1
    file p1f2 from channel2

  output:
    file "Eli5_ALL_NORM_fraud.csv" into p1result 

	script:
 	"""
#!/usr/bin/env python3

import eli5
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


loaded_class1x = pickle.load(open('$p1f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p1f2', 'rb'))

columnnames = np.array(loaded_class1x.columns)
datatoeli5 = loaded_class1x[0:150]

x = loaded_class1x.columns.tolist()
R = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()

for i in range(len(datatoeli5)):
    df1 = eli5.explain_prediction_df(estimator=loaded_RFmodel, doc = loaded_class1x.iloc[i])
    df2 = df1.copy().drop(['target'],axis=1)
    df3 = df2.copy().drop(['value'],axis=1)
    eli5_df = df3.drop(df3.index[0])
    eli5_df = eli5_df.rename(columns = {'feature':'Feature'})
    df = eli5_df.sort_values(['weight'], ascending=False)           
    R = pd.concat([R, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['weight'].abs()
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


R.to_csv('Eli5_fraud_originalweights.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('Eli5_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
R_norm_TOP5.to_csv('Eli5_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")



"""
}

p1result.collectFile(name: file("Eli5_NORM_ALL_NORM_fraud.csv")).set{setResult1}



process getEli5ExplainPredictionInterpretation_nonfraud{
  input:
    file p2f1 from channel11
    file p2f2 from channel22

  output:
    file "Eli5_NORM_ALL_NORM_nonfraud.csv" into p2result 

    script:
    """
#!/usr/bin/env python3

import eli5
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

loaded_class0x = pickle.load(open('$p2f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p2f2', 'rb'))

columnnames = np.array(loaded_class0x.columns)
datatoeli5 = loaded_class0x[0:150]

x = loaded_class0x.columns.tolist()
R = pd.DataFrame()
R_norm = pd.DataFrame()
R_norm_TOP5 = pd.DataFrame()


for i in range(len(datatoeli5)):
    df1 = eli5.explain_prediction_df(estimator=loaded_RFmodel, doc = loaded_class0x.iloc[i])
    df2 = df1.copy().drop(['target'],axis=1)
    df3 = df2.copy().drop(['value'],axis=1)
    eli5_df = df3.drop(df3.index[0])
    eli5_df = eli5_df.rename(columns = {'feature':'Feature'})
    df = eli5_df.sort_values(['weight'], ascending=False)      
    R = pd.concat([R, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['weight'].abs()
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


R.to_csv('Eli5_nonfraud_originalweights.csv', index = False, header=True, sep = "\t")
R_norm.to_csv('Eli5_NORM_ALL_NORM_nonfraud.csv', index = False, header=True, sep = "\t")
R_norm_TOP5.to_csv('Eli5_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")

"""
}

p2result.collectFile(name: file("Eli5_NORM_ALL_NORM_nonfraud.csv")).set{setResult2}


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

