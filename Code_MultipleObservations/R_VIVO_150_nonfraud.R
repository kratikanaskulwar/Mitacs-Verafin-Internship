#!/usr/bin/env Rscript
library(DALEX)
library(dplyr)
library(randomForest)
library(ingredients)
library(vivo)


X_train <- read.table('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/X_train.txt', header = TRUE, sep = ",")
y_train <- read.table('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/MainRF/y_train.txt', header = TRUE, sep = ",")
loaded_class0x <- read.table('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/jupyter/VerafinCodefiles/InterpretabilityMethods/Local/class0XforR.csv', header = TRUE, sep = ",")
loaded_RFmodel <- readRDS('/Users/kratikanaskulwar/GoogleDrive/VerafinProject/R_CODES/R_RF.rds')

df_xtrain <- data.frame(X_train)
df_ytrain <- data.frame(y_train)
loaded_class0x <- data.frame(loaded_class0x)

ptm <- proc.time()
#Create DALEX explainer
explain_rf <- explain(loaded_RFmodel,
                      data = df_xtrain,
                      y = df_ytrain,
                      label = "Random Forest")


datalist = list()
R_normlist = list()
datatovivo <- loaded_class0x[1:150,]

for (i in 1:nrow(datatovivo))
{
    profiles <- ceteris_paribus(explain_rf, loaded_class0x[i,])
    meas <- local_variable_importance(profiles, df_xtrain,absolute_deviation = TRUE, point = TRUE, density = TRUE)
    
    df <- data.frame(meas)
    df_sort <- df[with(df, order(-measure)),]
    names(df_sort)[names(df_sort) == "variable_name"] <- "Feature"
    datalist[[i]] <- df_sort

    #take absolute of contribution column and sum, then divide each row(abs) by the sum.
    df_sort$abs_Contribution <- abs(df_sort$measure)
    total <- sum(df_sort[, 'abs_Contribution'])
    df_sort$Norm_ContributionAbs <- df_sort$abs_Contribution/total
    cols <- c("Feature", "Norm_ContributionAbs")
    df_sort_norm <- df_sort[cols]
    df_sort_norm <- df_sort_norm[with(df_sort_norm, order(-Norm_ContributionAbs)),]

    R_normlist[[i]] <- df_sort_norm
}
Result = do.call(cbind, datalist)
Result_norm = do.call(cbind, R_normlist) 
proc.time() - ptm
top5_norm <- head(Result_norm,5)


write.csv(Result, file = "VIVO_nonfraud_originalweights.csv", row.names = FALSE)
write.csv(Result_norm, file = "VIVO_ALL_NORM_nonfraud.csv", row.names = FALSE)
write.csv(top5_norm, file = "VIVO_NORM_TOP5_nonfraud.csv", row.names = FALSE)
