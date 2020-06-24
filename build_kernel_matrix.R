source("util.R")
build_kernel_matrix <- function(model,z1,z2,equals = FALSE, only_diag = FALSE){
  if(only_diag){
    if(model$kern$is.RBF){
      kern <- model$kern;
      sig <- kern$RBF$var
      temp <- tf$reduce_sum(z1, shape(1,1), keepdims=TRUE) # how dumb is this :D
      K <- sig*tf$ones_like(temp)
    }
    if(model$kern$is.ARD){
      kern <- model$kern;
      sig <- kern$ARD$var
      temp <- tf$reduce_sum(z1, shape(1,1), keepdims=TRUE) # how dumb is this :D
      K <- sig*tf$ones_like(temp)
    }
    if(model$kern$is.lin){
      a <- tf$nn$relu(model$kern$lin$a)
      b <- tf$nn$relu(model$kern$lin$b)
      temp <- tf$multiply(z1,z1); out <- tf$reduce_sum(temp, shape(1,1), keepdims =TRUE)
      K <- a*out + b
    }
    if(model$kern$is.polynomial){
      d <- model$kern$polynomial$d
      a <- tf$nn$relu(model$kern$polynomial$a)
      b <- tf$nn$relu(model$kern$polynomial$b)
      temp <- tf$multiply(z1,z1); out <- tf$reduce_sum(temp, shape(1,1), keepdims =TRUE)
      K <- (a*out + b)^d
    }
  } else{
    if(model$kern$is.RBF){
      kern <- model$kern;
      sig <- kern$RBF$var; 
      nu <- softplus(kern$RBF$ls);
      
      K <- sig*exp(-1/(2*nu^2) * squared_dist(z1,z2, equals = equals))
    }
    if(model$kern$is.ARD){
      kern <- model$kern;
      sig <- kern$ARD$var;
      nu <- softplus(kern$ARD$ls)
      
      K <- sig*exp(-0.5 * squared_dist(z1/nu,z2/nu, equals = equals))
    }
    if(model$kern$is.lin){
      a <- tf$nn$relu(model$kern$lin$a)
      b <- tf$nn$relu(model$kern$lin$b)
      K <- a*tf$matmul(z1,z2,transpose_b = TRUE) + b 
    }
    if(model$kern$is.polynomial){
      d <- model$kern$polynomial$d
      a <- tf$nn$relu(model$kern$polynomial$a)
      b <- tf$nn$relu(model$kern$polynomial$b)
      K <- (a*tf$matmul(z1,z2,transpose_b = TRUE) + b)^d
    }
  }
  return(K)
}