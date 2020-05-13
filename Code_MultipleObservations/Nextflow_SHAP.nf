#!/usr/bin/env nextflow

//-------------------------channels----------------------------//

channel1 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class1X.pickle')

channel11 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0X.pickle')

ch2 = Channel.fromPath( '/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/Python_RF.pickle')
ch2.into{channel2;channel22}

process getSHAPInterpretation_fraud{
  input:
    file p1f1 from channel1
    file p1f2 from channel2

  output:
    file "SHAP_ALL_NORM_fraud.csv" into p1result 

	script:
 	"""
#!/usr/bin/env python3

import shap
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

loaded_class1x = pickle.load(open('$p1f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p1f2', 'rb'))

explainer = shap.TreeExplainer(loaded_RFmodel)
columnnames = np.array(loaded_class1x.columns)
datatoshap = loaded_class1x[0:150]

result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatoshap)):
    shap_val = explainer.shap_values(loaded_class1x.iloc[i],check_additivity=False)
    shap_df = pd.DataFrame({'Shap_values': shap_val[1],'Feature': columnnames})
    shap_array = shap_df.to_numpy()
    shap_sortArray = shap_array[shap_array[:,0].argsort()]
    shap_sortedArray = shap_sortArray[::-1]
    shap_sortedArray_todf = pd.DataFrame(data=shap_sortedArray, columns=["Shap_value", "Feature"])
    
    cols = shap_sortedArray_todf.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = shap_sortedArray_todf[cols]
    result = pd.concat([result, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['Shap_value'].abs()
    df['Abs_Shap_value'] = contrAbs
    absSum = df['Abs_Shap_value'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_Shap_value']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)
  

result.to_csv('SHAP_fraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('SHAP_ALL_NORM_fraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('SHAP_NORM_TOP5_fraud.csv', index = False, header=True, sep = "\t")

"""
}

p1result.collectFile(name: file("SHAP_ALL_NORM_fraud.csv")).set{setResult1}



process getSHAPInterpretation_nonfraud{
  input:
    file p2f1 from channel11
    file p2f2 from channel22

  output:
    file "SHAP_ALL_NORM_nonfraud.csv" into p2result 

    script:
    """
#!/usr/bin/env python3

import shap
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

loaded_class0x = pickle.load(open('$p2f1', 'rb'))
loaded_RFmodel = pickle.load(open('$p2f2', 'rb'))

explainer = shap.TreeExplainer(loaded_RFmodel)
columnnames = np.array(loaded_class0x.columns)
datatoshap = loaded_class0x[0:150]

result = pd.DataFrame()
result_norm = pd.DataFrame()
result_norm_TOP5 = pd.DataFrame()

for i in range(len(datatoshap)):
    shap_val = explainer.shap_values(loaded_class0x.iloc[i],check_additivity=False)
    shap_df = pd.DataFrame({'Shap_values': shap_val[0],'Feature': columnnames})
    shap_array = shap_df.to_numpy()
    shap_sortArray = shap_array[shap_array[:,0].argsort()]
    shap_sortedArray = shap_sortArray[::-1]
    shap_sortedArray_todf = pd.DataFrame(data=shap_sortedArray, columns=["Shap_value", "Feature"])
    
    cols = shap_sortedArray_todf.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = shap_sortedArray_todf[cols]
    result = pd.concat([result, df], axis=1)
    
    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    contrAbs = df['Shap_value'].abs()
    df['Abs_Shap_value'] = contrAbs
    absSum = df['Abs_Shap_value'].sum()
    #New_Norm_Contribution column sum up to 1
    df['New_Norm_Contribution'] = df['Abs_Shap_value']/absSum
    df_norm = df[['Feature','New_Norm_Contribution']]
    
    df_norm = df_norm.to_numpy()
    df_norm = df_norm[df_norm[:,1].argsort()]
    df_norm = df_norm[::-1]
    df_norm = pd.DataFrame(data=df_norm, columns=["Feature", "Norm_Contribution(Abs)"])
    
    result_norm = pd.concat([result_norm, df_norm], axis=1)
    result_norm_TOP5 = result_norm.head(5)
  
result.to_csv('SHAP_nonfraud_originalweights.csv', index = False, header=True, sep = "\t")
result_norm.to_csv('SHAP_ALL_NORM_nonfraud.csv', index = False, header=True, sep = "\t")
result_norm_TOP5.to_csv('SHAP_NORM_TOP5_nonfraud.csv', index = False, header=True, sep = "\t")

"""
}

p2result.collectFile(name: file("SHAP_ALL_NORM_nonfraud.csv")).set{setResult2}


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

