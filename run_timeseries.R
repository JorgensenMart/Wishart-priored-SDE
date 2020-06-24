# Timeseries run

library(tensorflow)
source("optimize_ts.R");

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]
rho <- args[2]

if( dataset == "lotka"){
  load("data/timeseries/train.Rda")
  time <- seq(0:1999)/20 # check if work
  load("data/timeseries/test.Rda")
}
if( dataset == "mocap"){
  source("load_mocap.R")
  data <- load_mocap()
  train <- (data$y_trains[[1]][[1]] - colMeans(data$y_trains[[1]][[1]] ) ) / sqrt(max(var(data$y_trains[[1]][[1]])))
  time <- seq(0:225)/120 # check this
}
if( dataset == "china"){
  load("data/timeseries/china_train.RDa")
  load("data/timeseries/china_test.RDa")
  time <- seq(1:(17520+8784))
}

if( dataset == "uncor_sim"){
  source("data/timeseries/simulated/uncor.R")
  time <- seq(1:1000)
}
if( dataset == "cor_sim"){
  source("data/timeseries/simulated/cor.R")
  time <- seq(1:1000)
}

y_train <- as.matrix(train)
y_test <- as.matrix(test)
y_mean <- colMeans(y_train); y_sd <- sqrt((length(y_train[,1])-1)^(-1) * colSums(sweep(y_train,2,y_mean)^2))
y_train <- t(apply(y_train,1,function(x){(x-y_mean)/y_sd}))
y_test <- t(apply(y_test,1,function(x){(x-y_mean)/y_sd}))
#time <- create dependent on dataset
  
tf$reset_default_graph()
results <- optimize_ts(y_train, y_test, time, rho) # Spit out some datafram (change in function)


filename = paste(dataset,rho,"WGPtimeseries", sep = "_")
filename = paste("timeseries/results/",filename, sep = "")
filename = paste(filename, "Rda", sep = ".")
save(results, file = filename)