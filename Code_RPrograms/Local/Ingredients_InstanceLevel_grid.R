#LINKhttp://uc-r.github.io/iml-pkg#procedures, Decision tree surrogate approach

install.packages("plotly", repos='https://mirror.its.sfu.ca/mirror/CRAN/')
library(plotly)
library(dplyr)
library(randomForest)
library(DALEX)
library(DALEXtra)
library(iml)
library(ggplot2)
library(ingredients)

load("Dalesexplainer.RData")

new_trans <- df_xtest[955,]

ptm1 <- proc.time()
cp_rf <- ceteris_paribus(explain_rf, new_trans)
dim(cp_rf)
print(cp_rf)
first <- plotD3(cp_rf, variables = c("V14","V12"), facet_ncol = 2, scale_plot = TRUE)


sec <- plotD3(cp_rf, variables = c("V10","V4"), facet_ncol = 2, scale_plot = TRUE)

htmlwidgets::saveWidget(as_widget(first), "first_grid.html")
htmlwidgets::saveWidget(as_widget(sec), "sec_grid.html")

proc.time() - ptm1

ptm2 <- proc.time()
sp_rf <- ceteris_paribus(explain_rf, new_trans, grid_points = 200)

pdf(file = "InstanceLevel_Ingredients_grid.pdf")
proc.time() - ptm2
plot(sp_rf) + show_observations(sp_rf)
dev.off()

print(sp_rf)
dim(sp_rf)
des <- describe(sp_rf, nonsignificance_treshold = 0.15,variables = 'V14')

print(des)
dim(des)
