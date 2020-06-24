# Utility functions
library(tensorflow)

float_32 <- function(t){
  return(tf$cast(t, tf$float32))
}

int_32 <- function(t){
  return(tf$cast(t, tf$int32))
}

squared_dist <- function(x,y, equals = FALSE){
  if(equals == TRUE){
    r_norms_x <- tf$reduce_sum(tf$square(x), as.integer(1), keepdims = TRUE)
    d <- r_norms_x - 2*tf$matmul(x,x,transpose_b = TRUE) + tf$transpose(r_norms_x)
    return(d)
  }
  else{
    r_norms_x <- tf$reduce_sum(tf$square(x), as.integer(1), keepdims = TRUE)
    r_norms_y <- tf$reduce_sum(tf$square(y), as.integer(1), keepdims =TRUE)
    r_norms_y <- tf$transpose(r_norms_y)
    return(r_norms_x - 2*tf$matmul(x,y,transpose_b = TRUE) + r_norms_y)
  }
}

nearest_neighbor <- function(data,query){
  #' data is axc, query is bxc
  distance <- squared_dist(data,query)
  pred <- tf$squeeze(tf$arg_min(distance,as.integer(0)))
  return(pred)
  # Returns the index of the nearest neighbor
}

get_chol <- function(vec){
  # make a vector into lower-triangular matrix
  L <- tf$contrib$distributions$fill_triangular(vec)
  return(L)
}

constrain_pos <- function(tensor){
  out <- tf$clip_by_value(tensor,1e-5,Inf)
  return(out)
}

gp_to_sqrtwishart <- function(L,samp){
  # samp is NxDxNU or NxKxNU
  # Should return NxDxD
  # Return only chol-type
  # L is DxD or DxK
  L_tile <- tf$tile(L[NULL,,], as.integer(c(samp$get_shape()$as_list()[1],1,1))) # stupid way of broadcasting matmul ?
  sq_Sig <- tf$matmul(L_tile,samp) # NxDxNU
  return(sq_Sig)
}

row_sum_L2 <- function(A){
  row_sums <- tf$reduce_sum(tf$square(A), axis = as.integer(1))
  out <- A / tf$sqrt(tf$reshape(row_sums, as.integer(c(-1,1))))
  return(out)
}

softplus <- function(x){
  return(tf$log(1 + tf$exp(x)) + 1e-5)
}

initialize_L <- function(A, session, iter = 500, B){
  D <- A$get_shape()$as_list()[1]
  if(missing(B)){
    B <- tf$eye(as.integer(D))
  }
  loss <- tf$norm(B - tf$matmul(A,A, transpose_b = TRUE), ord = Inf)
  optimizer <- tf$train$AdamOptimizer(0.01)$minimize(loss)
  session$run(tf$global_variables_initializer())
  for(i in 1:500){
    session$run(optimizer)
  }
  return(A)
}

init_inducing <- function(train_data, num_inducing){
  dummy <- NULL
  while(!is.numeric(dummy)){
    I <- sample(length(train_data[,1]), num_inducing)
    init_centers <- train_data[I,] + rnorm(length(I)*length(train_data[1,]), 0 , 1e-8)
    attempt <- try(expr = {kmeans(train_data, centers = init_centers, iter.max = 99)}, silent = TRUE)
    dummy <- try(expr = {dummy <- attempt$centers}, silent = TRUE)
  }
  return(dummy)
}
  