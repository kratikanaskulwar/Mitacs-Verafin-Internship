#!/usr/bin/env nextflow

//-------------------------channels----------------------------//
channel1 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1X.pickle')
channel11 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0X.pickle')

ch2 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/Python_RF.pickle')
ch2.into{channel2; channel22}

process getTreeInterpreterInterpretation_fraud{
  input:
    file p1f1 from channel1
    file p1f2 from channel2

  output:
    file "TREEINTER_ALL_NORM_fraud.csv" into p1result 

	script:
 	"""
#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti

loaded_class1x = pickle.load(open('$p1f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p1f2', 'rb'))

columnnames = np.array(loaded_class1x.columns)
datatotreeinter = loaded_class1x[0:150]

x = loaded_class1x.columns.tolist()
R = pd.DataFrame()

tree_df = pd.DataFrame()
result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatotreeinter)):
    print(i)
    prediction, bias, contributions = ti.predict(loaded_RFmodel, loaded_class1x.iloc[i].values.reshape(1, -1))
    y = contributions[:, :, 1].flatten()
    tree_df = pd.DataFrame({'Contribution':y,'Feature':x})
    print(tree_df)
    tree_df = tree_df.to_numpy()
    tree_df = tree_df[tree_df[:,0].argsort()]
    tree_df = tree_df[::-1]
   
    tree_df = pd.DataFrame(data=tree_df, columns=["Contribution", "Feature"])
    cols = tree_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = tree_df[cols]
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

result.to_csv('TREEINTER_fraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('TREEINTER_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('TREEINTER_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")

"""
}

p1result.collectFile(name: file("TREEINTER_ALL_NORM_fraud.csv")).set{setResult1}




process getTreeInterpreterInterpretation_nonfraud{
  input:
    file p2f1 from channel11
    file p2f2 from channel22

  output:
    file "TREEINTER_ALL_NORM_nonfraud.csv" into p2result 

    script:
    """
#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from treeinterpreter import treeinterpreter as ti

loaded_class0x = pickle.load(open('$p2f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p2f2', 'rb'))

columnnames = np.array(loaded_class0x.columns)
datatotreeinter = loaded_class0x[0:150]

x = loaded_class0x.columns.tolist()
R = pd.DataFrame()

tree_df = pd.DataFrame()
result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatotreeinter)):
    prediction, bias, contributions = ti.predict(loaded_RFmodel, loaded_class0x.iloc[i].values.reshape(1, -1))
    y = contributions[:, :, 0].flatten()
    tree_df = pd.DataFrame({'Contribution':y,'Feature':x})
    tree_df = tree_df.to_numpy()
    tree_df = tree_df[tree_df[:,0].argsort()]
    tree_df = tree_df[::-1]
   
    tree_df = pd.DataFrame(data=tree_df, columns=["Contribution", "Feature"])
    cols = tree_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = tree_df[cols]
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

result.to_csv('TREEINTER_nonfraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('TREEINTER_ALL_NORM_nonfraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('TREEINTER_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")

"""
}

p2result.collectFile(name: file("TREEINTER_ALL_NORM_nonfraud.csv")).set{setResult2}

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


