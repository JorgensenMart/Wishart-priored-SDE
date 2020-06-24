get_mu_and_var <- function(x_batch,model, K_q = NULL, K_MN = NULL){
  if(is.null(K_q)){
    jitter = 1e-5
    z <- model$v_par$v_x;
    K_MM <- build_kernel_matrix(model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing))
    C_MM <- tf$cholesky(K_MM)
  } else{
    K_MM <- K_q$Kmm
    C_MM <- K_q$Kmmchol
  }
  if(is.null(K_MN)){
    z <- model$v_par$v_x; x <- x_batch
    K_MN <- build_kernel_matrix(model,z,x)
  } else{
    K_MN  <- K_MN
  }
  
  A <- tf$matrix_triangular_solve(C_MM, K_MN, lower = TRUE) #MxN
  MU <- model$v_par$mu # MxD or MxDxNU
  
  if(MU$get_shape()$ndims == 3){     # IF WISHART MODEL
    MU_rs <- tf$reshape(MU, as.integer(c(MU$get_shape()$as_list()[1],-1)))
    out_mean <- tf$matmul(A,MU_rs,transpose_a = TRUE)
    out_mean <- tf$reshape(out_mean, as.integer(c(out_mean$get_shape()$as_list()[1],
                                                  MU$get_shape()$as_list()[2],
                                                  model$deg_free)))
  } else{
    out_mean <- tf$matmul(A,MU,transpose_a = TRUE) #NxD
  }
  output_dim <- MU$shape$as_list()[2] # ?? 
  
  S <- get_chol(model$v_par$chol); #DxMxM or DxNUxMxM
  if(S$get_shape()$ndims == 4){ #WISHART (non-constrained)
    tile_A <- tf$tile(A[NULL,NULL,,], as.integer(c(output_dim,model$deg_free,1,1)))
    S <- tf$matmul(S,tf$transpose(S,as.integer(c(0,1,3,2))))
    smink <- S - tf$eye(int_32(model$v_par$num_inducing), dtype = tf$float32)#[output_dim,model$deg_free,,]
    B <- tf$matmul(smink,tile_A) 
    
    d_cov <- tf$reduce_sum(tile_A * B, as.integer(2)) #DxNUxN --- summing over M
    K_NN <- build_kernel_matrix(model,x_batch,x_batch,equals = TRUE, only_diag = TRUE) #Nx1
    out_var <- tf$transpose(tf$transpose(K_NN) + d_cov, as.integer(c(2,0,1))) #NxDxNU
    out <- list(mean = out_mean, var = out_var)
  } else{
    tile_A <- tf$tile(A[NULL,,], as.integer(c(output_dim,1,1)))
    S <- tf$matmul(S,tf$transpose(S,as.integer(c(0,2,1)))) #DxMxM
    I <- tf$eye(as.integer(model$v_par$num_inducing), dtype = tf$float32)[NULL,,] #1xMxM
    smink <- S - I
    B <- tf$matmul(smink,tile_A) #DxMxN
    
    d_cov <- tf$reduce_sum(tile_A * B, as.integer(1)) #DxN --- summing over M
    K_NN <- build_kernel_matrix(model,x_batch,x_batch,equals = TRUE, only_diag = TRUE) #Nx1
    
    out_var <- tf$transpose(tf$transpose(K_NN) + d_cov) #NxD
    out <- list(mean = out_mean, var = out_var)
  }
  return(out)
}