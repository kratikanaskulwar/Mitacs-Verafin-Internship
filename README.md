## 1. Introduction
This repository contains all the codes, data, results and supplementary files related to "Exploration and agreement assessment among interpretability methods for fraud detection machine learning model" project.

## 2. Clone repository
Follows these steps to clone the repository:

1) Install git lfs if it is not installed already. This is needed here because the repository contains data files upload through git lfs which can be viewed in the original form only if git lfs is installed.

  installation instructions here: https://git-lfs.github.com/
  
2) Clone the repository using git clone command.

## 3. Running program
To run any program, keep the Data folder in the directory where the program is being run from.

## 4. Heatmaps
This folder contains heatmps for local and global programs.

### Heatmap_fortestset_globalmethods.png
This heatmap has been created for the agreement among global methods.

### Heatmap_for1fraudinstance_localmethods.png
This heatmap has been created for the agreement among local methods for 1 fraud observation.

### Heatmap_for150fraudinstances_localmethods.png
This heatmap has been created for the agreement among local methods across 150 fraud observations.

### Heatmap_for150nonfraudinstance_localmethods.png
This heatmap has been created for the agreement among local methods across 150 non-fraud observations.

### Heatmap_for300obs(global)and150fraudobs(local).png
This heatmap has been created for the agreement among local methods for 150 fraud observation and global methods across total 300 fraud and non-fraud observations from test set.
