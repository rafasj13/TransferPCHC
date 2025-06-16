# install.packages(c("ggplot2", "graph", "reshape2", "Rgraphviz"))
# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install(c("graph", "Rgraphviz"))
# install.packages("./scmamp_0.2.1.tar.gz", repos=NULL, type="source")
library(scmamp)



data <- read.csv("./critical_diff/BOTH_51020p_shd_525/logl.csv",  header=TRUE)
data

test.res <- postHocTest(data = data, test = 'friedman', correct = 'bergmann')
test.res

average.ranking <- colMeans(rankMatrix(data, decreasing=TRUE))
average.ranking
drawAlgorithmGraph(pvalue.matrix = test.res$corrected.pval, mean.value = average.ranking)




test.res.df <- as.data.frame(test.res$corrected.pval)
avg.ranking.df <- as.data.frame(average.ranking)
avg.ranking.df
write.csv(test.res.df, file = "critical_diff/BOTH_51020p_shd_525/res/logl_bergmann_hommel.csv", row.names = TRUE)
write.csv(avg.ranking.df, file = "critical_diff/BOTH_51020p_shd_525/res/logl_avg_ranking.csv", row.names = TRUE)


