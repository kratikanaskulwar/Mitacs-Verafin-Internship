library(inTrees)
library(randomForest)

RFModel <- readRDS("./SavedRFModel/R_RF.rds")
print(RFModel)
      
X_test <- read.table('./Data/X_test.txt', header = TRUE, sep = ",")
y_test <- read.table('./Data/y_test.txt', header = TRUE, sep = ",")
    
df_xtest <- data.frame(X_test)
df_ytest <- data.frame(y_test)

ptm <- proc.time()
# extractRules(treeList, X, ntree = 100, maxdepth = 6, random = FALSE, digits = NULL)
# ntree is default 100, change the value to give number of trees
treeList <- RF2List(RFModel)  # transform rf object to an inTrees' format
exec <- extractRules(treeList, df_xtest) # R-executable conditions
exec[1:6,]
      
#Measure rules
ruleMetric <- getRuleMetric(exec,df_xtest,df_ytest$Class)
ruleMetric[1:6,]
      
#Prune each rule:
pruneRule <- pruneRule(ruleMetric,df_xtest,df_ytest$Class)
pruneRule[1:6,]

#Select a compact rule set:
selectRuleRRF <- selectRuleRRF(pruneRule,df_xtest,df_ytest$Class)
selectRuleRRF


#Build an ordered rule list as a classifier:
learner <- buildLearner(selectRuleRRF,df_xtest,df_ytest$Class)
learner

#Make rules more readable:
readableRules <- presentRules(selectRuleRRF,colnames(df_xtest))  
readableRules[1:6,]

save.image(file='inTreesreadableRules.RData')
proc.time() - ptm


ptm1 <- proc.time()
#Extract frequent variable interactions (note the rules are not pruned or selected):
#freqPattern <- getFreqPattern(ruleMetric)
#freqPattern[which(as.numeric(freqPattern[,"len"])>=2),][1:4,] # interactions of at least two predictor variables

save.image(file='inTrees.RData')

proc.time() - ptm1

##------------------------For 300 observations-------------------------------------##

dataX <- read.table('./Data/dataX.txt', header = TRUE, sep = "\t")
dataY <- read.table('./Data/dataY.txt', header = TRUE, sep = "\t")

ptm <- proc.time()
treeList <- RF2List(RFModel)  # transform rf object to an inTrees' format
exec <- extractRules(treeList, ntree = 500, dataX) # R-executable conditions
exec[1:6,]
      
#Measure rules
ruleMetric <- getRuleMetric(exec,dataX,dataY$Class)
ruleMetric[1:6,]
      
#Prune each rule:
pruneRule <- pruneRule(ruleMetric,dataX,dataY$Class)
pruneRule[1:6,]

#Select a compact rule set:
selectRuleRRF <- selectRuleRRF(pruneRule,dataX,dataY$Class)
selectRuleRRF


#Build an ordered rule list as a classifier:
learner <- buildLearner(selectRuleRRF,dataX,dataY$Class)
learner

#Make rules more readable:
readableRules <- presentRules(selectRuleRRF,colnames(dataX))
readableRules[1:10,]

proc.time() - ptm
