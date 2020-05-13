library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)
library(ingredients)

#955
#Load RF model
#RFModel <- readRDS("/Users/kratikanaskulwar/Desktop/R_CODES/R_RF.rds")
#print(RFModel)

#Load train and test sets (that were used in python Model)
#X_train <- read.table('./Data/X_train.txt', header = TRUE, sep = ",")
#y_train <- read.table('./Data/y_train.txt', header = TRUE, sep = ",")

#X_test <- read.table('./Data/X_test.txt', header = TRUE, sep = ",")
#y_test <- read.table('./Data/y_test.txt', header = TRUE, sep = ",")

#df_xtrain <- data.frame(X_train)
#df_ytrain <- data.frame(y_train)
#df_dataX <- data.frame(dataX)
#df_dataY <- data.frame(dataY)

# load the dalex explainer that has already been created with the training data, to create a new comment load line and uncomment explain_rf line wand above lines
load("Dalesexplainer.RData")
#Create DALEX explainer
#explain_rf <- explain(RFModel,data = df_xtrain,y = df_ytrain,label = "Random Forest")

#str(explain_rf)

#feature_importance function from ingredients
ptm1 <- proc.time()
fi_rf <- feature_importance(explain_rf)
head(fi_rf)

pdf(file = "Ingredients_Feature_importance.pdf")
plot(fi_rf)
dev.off()
proc.time() - ptm1
save.image(file='Ingredients_FI.RData')
