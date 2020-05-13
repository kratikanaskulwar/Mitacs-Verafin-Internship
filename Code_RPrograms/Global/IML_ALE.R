library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)

#Accumulated Local Effect (ALE)

#955
#Load RF model
RFModel <- readRDS("./SavedRFModel/R_RF.rds")
print(RFModel)

#Load train and test sets (that were used in python Model)
X_train <- read.table('./Data/X_train.txt', header = TRUE, sep = ",")
y_train <- read.table('./Data/y_train.txt', header = TRUE, sep = ",")

X_test <- read.table('./Data/X_test.txt', header = TRUE, sep = ",")
y_test <- read.table('./Data/y_test.txt', header = TRUE, sep = ",")

df_xtrain <- data.frame(X_train)
df_ytrain <- data.frame(y_train)
df_xtest <- data.frame(X_test)
df_ytest <- data.frame(y_test)

#Create DALEX explainer
ptm <- proc.time()
predictor = Predictor$new(RFModel, data = df_xtest, y = df_ytest)
proc.time() - ptm
str(predictor)

#feature_importance function from ingredients
ptm1 <- proc.time()
ale14 = FeatureEffect$new(predictor, feature = "V14")
pdf(file = "IML_V14_Accumulated_Local_Effect.pdf")
plot(ale14)

ale14 = FeatureEffect$new(predictor, feature = "V4")
pdf(file = "IML_V4_Accumulated_Local_Effect.pdf")
plot(ale14)

ale14 = FeatureEffect$new(predictor, feature = "V12")
pdf(file = "IML_V12_Accumulated_Local_Effect.pdf")
plot(ale14)

ale14 = FeatureEffect$new(predictor, feature = "V10")
pdf(file = "IML_V0_Accumulated_Local_Effect.pdf")
plot(ale14)
dev.off()
proc.time() - ptm1

ptm2 <- proc.time()
aleatwo = FeatureEffect$new(predictor, feature = c("V14","V4"))
pdf(file = "IML_V14V4_Accumulated_Local_Effect.pdf")
proc.time() - ptm2
plot(aleatwo)
dev.off()


