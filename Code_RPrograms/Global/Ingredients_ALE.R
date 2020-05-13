library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)
library(ingredients)


load("Dalesexplainer.RData")

# Conditional dependence Plot for four features ("V14", "V12", "V10", "V4")
ptm1 <- proc.time()
ap_V14V12V10V4  <- accumulated_dependence(explain_rf, variables =  c("V14", "V12", "V10", "V4"))

pdf(file = "Ingredients_Accumulated_Local_Effect.pdf")
proc.time() - ptm1
plot(ap_V14V12V10V4)
dev.off()

# Conditional dependence Plot for all features

ptm2 <- proc.time()
ap_all  <- accumulated_dependence(explain_rf)

pdf(file = "Ingredients_Accumulated_Local_Effect_allfeatures.pdf")
proc.time() - ptm2
plot(ap_all)
dev.off()
