#LINKhttp://uc-r.github.io/iml-pkg#procedures

#Decision tree surrogate approach

library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)

# for test set
#RFModel <- readRDS("/Users/kratikanaskulwar/Desktop/R_CODES/R_RF.rds")
#print(RFModel)

##Load train and test sets (that were used in python Model)
#X_train <- read.table('./Data/X_train.txt', header = TRUE, sep = "\t")
#y_train <- read.table('./Data/y_train.txt', header = TRUE, sep = "\t")

#X_test <- read.table('./Data/X_test.txt', header = TRUE, sep = "\t")
#y_test <- read.table('./Data/y_test.txt', header = TRUE, sep = "\t")

#df_xtrain <- data.frame(X_train)
#df_ytrain <- data.frame(y_train)
#df_xtest <- data.frame(X_test)
#df_ytest <- data.frame(y_test)

ptm1 <- proc.time()
#Create DALEX explainer
#explain_rf <- explain(RFModel,data = df_xtrain,y = df_ytrain,label = "Random Forest")

load("Dalesexplainer.RData")
str(explain_rf)

predictor = Predictor$new(explain_rf, data = df_xtest, y = df_ytest)

tree = TreeSurrogate$new(predictor,  maxdepth = 5)


pdf(file = "IML_Treesurrogate_boxplot_testset.pdf")

proc.time() - ptm1

plot(tree)
dev.off()


predictor1 = Predictor$new(RFModel, data = df_xtest, y = df_ytest)

tree1 = TreeSurrogate$new(predictor1,  maxdepth = 5)


pdf(file = "IML_Treesurrogate_barplot_testset.pdf")
plot(tree1)
dev.off()

# For 300 observations from test set
load("Dalesexplainer.RData")

dataX <- read.table('./Data/dataX.txt', header = TRUE, sep = "\t")
dataY <- read.table('./Data/dataY.txt', header = TRUE, sep = "\t")

df_dataX <- data.frame(dataX)
df_dataY <- data.frame(dataY)

ptm1 <- proc.time()

predictor3 = Predictor$new(explain_rf, data = df_dataX, y = df_dataY)

dt = TreeSurrogate$new(predictor3, maxdepth = 9)
pdf(file = "IML_Treesurrogate_boxplot_300obs.pdf")
plot(dt)

proc.time() - ptm1
a = predict(dt, dataX, type = "class")
# Extract the dataset
dat <- dt$results
print(head(dat))

predictor4 = Predictor$new(RFModel, data = df_dataX, y = df_dataY)

dt1 = TreeSurrogate$new(predictor4, maxdepth = 9)
pdf(file = "IML_Treesurrogate_barplot_300obs.pdf")
plot(dt1)
dev.off()

