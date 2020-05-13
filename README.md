## 1. Introduction
This repository contains all the codes, data, results and supplementary files related to "Exploration and agreement assessment among interpretability methods for fraud detection machine learning model" project.

## 2. Clone repository
Follow these steps to clone the repository:

1) Install git lfs if it is not installed already. This is needed here because the repository contains data files upload through git lfs which can be viewed in the original form only if git lfs is installed.

    Installation instructions here: https://git-lfs.github.com/
  
2) Clone the repository using git clone command.

## 3. Running program
To run any program, keep the Data folder in the directory where the program is to run.

## 4. Code_PythonPrograms and Code_RPrograms
This folder contains code for programs using python/R packages/library. It has been divided in Global and Local subfolders. For example, SHAP method works for local and global interpretation. So code for local has been kept in Local subfolder and similarly for SHAP global interpretation, it has been kept in Global subfolder.

## 5. ProgramResults
This folder contains results of the programs and has been kept in respective local and global program subfolders.

## 6. ResultsforMultipleObservations
This folder has program results of 150 fraud and non-fraud observations. Result files consist of feature names and the corresponding normalized contribution.

In addition to the Heatmaps folder, there is also a heatmap in the Interpretability_Programs_Summary.xlsx file in the Summary folder. This is for agreement among local methods with 1 fraud observation and global methods.
