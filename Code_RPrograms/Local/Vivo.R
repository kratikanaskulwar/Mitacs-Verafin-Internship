#LINKhttp://uc-r.github.io/iml-pkg#procedures

#Decision tree surrogate approach

library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)
library(ingredients)
library(vivo)


load("Dalesexplainer.RData")

new_trans <- df_xtest[955,]

profiles <- ceteris_paribus(explain_rf, new_trans)

measure <- local_variable_importance(profiles, df_xtrain,
                                     absolute_deviation = TRUE, point = TRUE, density = TRUE)
pdf(file = "Vivo.pdf")
plot(measure)
dev.off()
