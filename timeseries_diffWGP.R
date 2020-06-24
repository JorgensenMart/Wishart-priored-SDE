source("get_mu_and_var.R"); source("compute_kl.R"); source("build_kernel_matrix.R"); source("util.R"); source("sample_gp_marginal.R")

forwardtime_diffWGP <- function(model,y_batch, time,
                                y_mean = 0, y_std = 1, c, mc_samples = 1, train_size, batch_size, test_time = FALSE){
  # model is wis-gp and mean-gp 
  current_x = y_batch[1,]# D 
  current_x = tf$transpose(current_x[,NULL])# 1xD
  wis_model <- model[[1]]; mean_model <- model[[2]]
  #time <- #vector-ish of timestamps
  
  z <- wis_model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(wis_model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(wis_model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  
  N <- batch_size; 
  D <- y_batch$get_shape()$as_list()[[2]] 
  NU <- wis_model$deg_free
  K <- wis_model$wis_factor
  
  L <- wis_model$L_scale_matrix # DxD or DxK
  
  sqc = tf$sqrt(c)
  #time <- tf$constant(time)
  t <- time[2:N] - time[1:(N-as.integer(1))] # (N-1)
  # compute likelihood at x_0
  #t_and_y has to be N-1x1xD
  # x_and_llh is (1xDx1)
  K_MN <- build_kernel_matrix(wis_model,z,current_x) #Mx1
  samp <- sample_gp_marginal(wis_model, current_x, K_q = K_q, K_MN = K_MN) #SxNxDxNU or SxNxKxNU, S = 1 (?)
  # if 1xNxD then squeeze
  samp <- tf$reshape(samp, shape = as.integer(c(1,K,NU))) # 1xKxNU
  sqSig <- gp_to_sqrtwishart(L,samp) # 1xDxNU or 1xKxNU
  
  llh <- tf$reshape(compute_loglikelihood(current_x,current_x,sqSig,c,NU,D,train_size), as.integer(c(NULL,1,1))) # y = x here
  x_and_llh <- tf$concat(c(current_x,llh), axis = as.integer(1))
  #x_and_llh <- tf$tuple(c(current_x,llh))
  t_and_y <- tf$tuple(c(t,y_batch[2:N,]))
  step_function <- function(x_and_llh,t_and_y){
    dt <- t_and_y[[1]]
    K_MN <- build_kernel_matrix(wis_model,z,x_and_llh[,1:D]) #Mx1
    samp <- sample_gp_marginal(wis_model, x_and_llh[,1:D], K_q = K_q, K_MN = K_MN) #SxNxDxNU or SxNxKxNU, S = 1 (?)
    # if 1xNxD then squeeze
    samp <- tf$reshape(samp, shape = as.integer(c(1,K,NU))) # 1xKxNU
    sqSig <- gp_to_sqrtwishart(L,samp) # 1xDxNU or 1xKxNU
    
    mu <- cond_mean(mean_model, x_and_llh[,1:D], K_q = K_q, K_MN = K_MN) # Assume we share kernels (mean and Wishart)
    #llh <- tf$squeeze(compute_loglikelihood(x_and_llh[,1:D],t_and_y[[2]],sqSig,c,NU,D))
    #llh <- tf$reshape(compute_loglikelihood(x_and_llh[,1:D],t_and_y[[2]],sqSig,c,NU,D,train_size), as.integer(c(NULL,1,1)))
    # Push forward
    sqdt <- tf$sqrt(dt)
    W <- tf$random_normal(as.integer(c(1,NU,1))) # N = 1
    W_err <- tf$random_normal(as.integer(c(1,D))) # White noise
    next_x <- x_and_llh[,1:D] + mu*dt + sqdt*(tf$squeeze(tf$matmul(sqSig,W))) + sqdt*sqc*W_err
    llh <- tf$reshape(compute_loglikelihood(next_x,t_and_y[[2]],sqSig,c,NU,D,train_size), as.integer(c(NULL,1,1)))
    if(test_time == FALSE){
      next_llh <- x_and_llh[,D+1] + llh
    } else{
      next_llh <- llh
    }
    out <- tf$concat(c(next_x,next_llh),axis = as.integer(1))
    return(out)
  }
  if(test_time == FALSE){
  callable_func <- function(x_and_llh){
    out <- tf$foldl(step_function,t_and_y,x_and_llh)
  }
  } else{
    callable_func <- function(x_and_llh){
      out <- tf$scan(step_function,t_and_y,x_and_llh)
    }
  }
  X <- tf$tile(x_and_llh[NULL,,], as.integer(c(mc_samples,1,1)))
  out <- tf$map_fn(callable_func,X)
  if(test_time == FALSE){
    x_out <- out[,,1:D]
    LLH <- tf$reduce_mean(out[,1,D+1],axis = as.integer(0))
  } else{
    x_out <- out[,,,1:D]
    LLH <- out[,,,D+1]
  }
  #LLH <- tf$reduce_mean(out[,1,D+1],axis = as.integer(0))
  KL_wis <- compute_kl(wis_model)
  KL_mean <- fixed_var_kl(mean_model)
  KL <- KL_mean + KL_wis
  return(list(llh = LLH, final_state = x_out, kl = KL)) #last state for forecast and llh
}

compute_loglikelihood <- function(x,y,sqsigma,c,NU,D,N){ # remove N ? 
  #y is Dx1
  dif = y - x # 1xD
  dif = tf$transpose(dif) #Dx1
  D <- length(dif) # work ?
  diag_inv <- tf$diag(1/c)
  sqsigma <- tf$squeeze(sqsigma)
  B <- tf$matrix_inverse( tf$diag(rep(1,NU)) + tf$matmul(tf$matmul(sqsigma,diag_inv,transpose_a = TRUE),sqsigma) ) #NUxNU
  inv_sigma <- diag_inv - tf$matmul(tf$matmul(diag_inv,tf$matmul(tf$matmul(sqsigma,B),sqsigma,transpose_b = TRUE)),diag_inv)
  # Check likelihoos
  llh <- D/2*log(2*pi) + 0.5 * tf$log(tf$matrix_determinant(B)) - 0.5*tf$reduce_sum(tf$log(c)) - 0.5*tf$matmul(tf$matmul(dif,inv_sigma,transpose_a = TRUE),dif)
  return(llh) # []
}

forecast <- function(state, horizon, model, samples = 1){
  current_x <- state
  current_x <- tf$transpose(current_x[,NULL])
  wis_model <- model[[1]]; mean_model <- model[[2]]
  #time <- #vector-ish of timestamps
  
  z <- wis_model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(wis_model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(wis_model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  
  N <- length(horizon); #length of horizon
  D <- current_x$get_shape()$as_list()[[2]] 
  NU <- wis_model$deg_free
  K <- wis_model$wis_factor
  L <- wis_model$L_scale_matrix # DxD or DxK
  sqc = tf$sqrt(c)
  time <- horizon
  t <- tf$constant(time[2:N] - time[1:(N-as.integer(1))]) # (N-1)
  step_function <- function(x,dt){
    K_MN <- build_kernel_matrix(wis_model,z,x) #Mx1
    samp <- sample_gp_marginal(wis_model, x, K_q = K_q, K_MN = K_MN) #SxNxDxNU or SxNxKxNU, S = 1 (?)
    # if 1xNxD then squeeze
    samp <- tf$reshape(samp, shape = as.integer(c(1,K,NU))) # 1xKxNU
    sqSig <- gp_to_sqrtwishart(L,samp) # 1xDxNU or 1xKxNU
    mu <- cond_mean(mean_model, x, K_q = K_q, K_MN = K_MN) # Assume we share kernels (mean and Wishart)
    #llh <- tf$squeeze(compute_loglikelihood(x_and_llh[,1:D],t_and_y[[2]],sqSig,c,NU,D))
    # Push forward
    sqdt <- tf$sqrt(dt)
    W <- tf$random_normal(as.integer(c(1,NU,1))) # N = 1
    W_err <- tf$random_normal(as.integer(c(1,D))) # White noise
    next_x <- x + mu*dt + sqdt*(tf$squeeze(tf$matmul(sqSig,W))) + sqdt*sqc*W_err
    return(next_x)
  }
  callable_func <- function(current_x){
    out <- tf$scan(step_function,t,current_x)
    return(out)
  }
  out <- tf$map_fn(callable_func,tf$tile(current_x[NULL,,], as.integer(c(samples,1,1))))
  return(out)
}



cond_mean <- function(model, x_batch, K_q = NULL, K_MN = NULL){
  if(is.null(K_q)){
    z <- model$v_par$v_x;
    jitter = 1e-5
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
  A <- tf$matrix_triangular_solve(C_MM, K_MN, lower = TRUE)
  MU <- model$v_par$mu # MxD or MxDxNU
  out_mean <- tf$matmul(A,MU,transpose_a = TRUE) #NxD
}
