
library(randomForest)


X_train <- read.table('/Data/X_train.txt', header = TRUE, sep = "\t")
y_train <- read.table('/Data/y_train.txt', header = TRUE, sep = "\t")


df_xtrain <- data.frame(X_train)
df_ytrain <- data.frame(y_train)


df_ytrain$Class = as.factor(df_ytrain$Class)

mtry <- tuneRF(df_xtrain, df_ytrain$Class, ntreeTry=500, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)

best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]

print(mtry)
print(best.m)
set.seed(71)

rfvf <-randomForest(x = df_xtrain, y = df_ytrain$Class, mtry=best.m, ntree=500)


# save the model to disk
saveRDS(rfvf, "./R_RF.rds")

# load the model
#super_model <- readRDS("./final_model_rfvf.rds")
#print(super_model)


