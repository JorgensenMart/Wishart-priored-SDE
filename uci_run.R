library(tensorflow)
source("optimize.R"); source("get_data.R")

args <- commandArgs(trailingOnly = TRUE)
chosen_model <- args[1]
dataset <- args[2]

print(chosen_model) # TEST

data <- get_data(dataset)
x <- data$x; y <- data$y

splits <- 20
testsetsize <- 0.1

set.seed(666)
seeds <- sample(1:10000, 20)
results <- data.frame()

j = 0
for( i in seeds ){
  set.seed(i)
  j = j + 1
  N <- length(x[,1])
  test_set_size <- as.integer(N*testsetsize)
  I <- sample(1:N, test_set_size)
  
  x_test <- x[I,]; y_test <- matrix(y[I,])
  x_train <- x[-I,]; y_train <- matrix(y[-I,])
  
  tf$reset_default_graph()
  results <- rbind(results,optimize(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, 
                      model_name = chosen_model))
  cat("SPLIT", j, "OUT OF", splits)
}
## K = 10
#filename = paste("diffWGPK10",dataset,sep= "_")
filename = paste(chosen_model,dataset, sep = "_")

filename = paste("results/",filename, sep = "")
filename = paste(filename, "Rda", sep = ".")
save(results, file = filename)