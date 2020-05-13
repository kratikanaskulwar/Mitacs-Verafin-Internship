library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)
library(ingredients)


load("Dalesexplainer.RData")

# Partial dependence Plot for four features ("V14", "V12", "V10", "V4")
ptm1 <- proc.time()

pp_V14V12V10V4  <- partial_dependence(explain_rf, variables =  c("V14", "V12", "V10", "V4"))
head(pp_V14V12V10V4)

pdf(file = "Ingredients_Partial_Dependence_Plot.pdf")
proc.time() - ptm1

plot(pp_V14V12V10V4)
dev.off()

# Partial dependence Plot for all features

#1. Partial Dependence Profiles
ptm2 <- proc.time()
pp_all  <- partial_dependence(explain_rf)
head(pp_all)
pdf(file = "Ingredients_Partial_Dependence_Plot_allfeatures.pdf")
proc.time() - ptm2
plot(pp_all)
dev.off()
