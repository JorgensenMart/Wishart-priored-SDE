source("get_mu_and_var.R"); source("compute_kl.R"); source("build_kernel_matrix.R"); source("util.R"); source("sample_gp_marginal.R")

forward_diffWGP <- function(model,x_batch,y_batch, samples = 4, 
                            step_size = 1/20, flowtime = 1,
                            y_mean = 0, y_std = 1,
                            test_time = FALSE,
                            time_independent = FALSE, warm_start = FALSE, scalar = NULL){
  
  tile_x <- tf$tile(x_batch[NULL,,], as.integer(c(samples,1,1)))
  time_grid <- seq(step_size,flowtime-step_size, by = step_size); time_grid <- tf$convert_to_tensor(time_grid)
  if(time_independent == FALSE){
    wrapper_diffWGPforward <- function(x_batch){
      return( diffWGP_forward(x0 = x_batch, step_size = step_size,
                              time = time_grid, model = model[[1]]) )
      
    }
  } else{
    wrapper_diffWGPforward <- function(x_batch){
      return(diffWGP_forward_timeind(x0 = x_batch, step_size = step_size,
                                      time = time_grid, model = model[[1]]))
    }
  }
  
  f_list <- tf$map_fn(wrapper_diffWGPforward, tile_x)
  
  
  KL_wis <- compute_kl(model[[1]][[1]])
  KL_mean <- fixed_var_kl(model[[1]][[2]])
  
  z <- model[[2]]$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(model[[2]],z,z,equals = TRUE) + jitter*tf$eye(as.integer(model[[2]]$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  wrapper_pred <- function(x_batch){
    Fpars <- get_mu_and_var(x_batch, model[[2]], K_q = K_q) # list of Fmu's and Fvar's
    Fmu <- Fpars$mean; Fvar <- Fpars$var
    L <- tf$stack(c(Fmu,Fvar),axis = as.integer(0))
    return(L)
  }
  
  S <- tf$map_fn(wrapper_pred, f_list) # Sx2xNxD
  ## RMSE
  Fmu_mean <- tf$reduce_mean(S[,1,,], axis = as.integer(0)) # NxD
  RMSE <- tf$sqrt(tf$reduce_mean(tf$square(Fmu_mean*y_std-(y_batch*y_std))))
  
  ## Log-likelihood
  likelihood <- function(S){
    if(model[[2]]$kern$is.RBF == TRUE){e <- softplus(model[[2]]$kern$RBF$eps)}
    if(model[[2]]$kern$is.ARD == TRUE){e <- softplus(model[[2]]$kern$ARD$eps)}
    if(model[[2]]$kern$is.lin == TRUE){e <- softplus(model[[2]]$kern$lin$eps)}
    Fmu <- S[1,,]; Fvar <- S[2,,] #NxD (here Nx1)
    if(test_time == FALSE){
      sum_log_lik <- - 0.5 * log(2 * pi) - 0.5*tf$log(y_std^2*e) -
        0.5*(tf$square(y_batch - Fmu) + Fvar)/e
      sum_log_lik <- tf$reduce_sum(sum_log_lik)
    } else if( test_time == TRUE){
      pred_dist <- tf$distributions$Normal(loc = Fmu*y_std, scale = y_std*tf$sqrt(Fvar + e))
      sum_log_lik <- pred_dist$log_prob(y_batch*y_std)
      sum_log_lik <- tf$reduce_sum(sum_log_lik) # ()
    }
    return(sum_log_lik)
  }
  LogLik <- tf$map_fn(likelihood,S) # Sx
  LogLik <- tf$reduce_mean(LogLik) 
  
  KL_pred <- compute_kl(model[[2]], K_q = K_q)
  if(warm_start == TRUE){
    KL <- KL_pred + scalar*KL_wis + scalar^2*KL_mean
  } else{
    KL <- KL_pred + KL_wis + KL_mean
  }
  E <- list(log_lh = LogLik, kl = KL, RMSE = RMSE)
  return(E)
}

diffWGP_forward <- function(x0, step_size, time, model){
  # model is a nested model
  wis_model <- model[[1]]; mean_model <- model[[2]]
  # is this smart ?
  dt <- step_size; dt <- tf$cast(dt, dtype = tf$float32)
  #err <- tf$constant(5e-3, dtype = tf$float32)
  z <- wis_model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(wis_model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(wis_model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  
  if(wis_model$kern$is.RBF == TRUE){sc <- wis_model$kern$RBF$var}
  if(wis_model$kern$is.ARD == TRUE){sc <- wis_model$kern$ARD$var}
  sc <- tf$sqrt(sc)
  #normal_dist <- tf$distributions$Normal(loc = 0, scale = 1)
  
  N <- x0$get_shape()$as_list()[1]; D <- x0$get_shape()$as_list()[2]
  NU <- wis_model$deg_free
  
  L <- wis_model$L_scale_matrix # DxD or DxK
  sqdt <- tf$sqrt(dt)
  step_function<- function(x_t,t){
    # x_t is NxD, t is () probably
    tc <- tf$tile(t[NULL,NULL], as.integer(c(N,1)))
    conc_xt <- tf$concat(c(x_t,tc),as.integer(1)) # Should be # Nx(D+1)
    
    K_MN <- build_kernel_matrix(wis_model,z,conc_xt)
    samp <- sample_gp_marginal(wis_model, conc_xt, K_q = K_q, K_MN = K_MN) #SxNxDxNU or SxNxKxNU, S = 1
    # if 1xNxD then squeeze
    samp <- tf$squeeze(samp)
    sqSig <- gp_to_sqrtwishart(L,samp) # NxDxNU or NxKxNU
    
    mu <- cond_mean(mean_model, conc_xt, K_q = K_q, K_MN = K_MN) #NxD
    
    #W <- tf$sqrt(dt) * normal_dist$sample(as.integer(c(N,NU,1))) # NxNUx1
    W <- tf$random_normal(as.integer(c(N,NU,1)))
    #sq_err <- sqrt(0.001)
    #W_err <- sq_err * tf$sqrt(dt) * normal_dist$sample(as.integer(c(N,D))) # NxD
    #W_err <- sq_err*tf$random_normal(as.integer(c(N,D)))
    #sc <- tf$sqrt(tf$exp(wis_model$kern$RBF$var) + 1e-8)
    
    #dx <- mu * dt + sc*sqdt*(tf$squeeze(tf$matmul(sqSig,W)) + sq_err*W_err) #NxD
    dx <- mu * dt + sqdt*(tf$squeeze(tf$matmul(sqSig,W)) )# + W_err)
    return(x_t + dx)
  }
  x_T <- tf$foldl(step_function,time,x0)
  return(x_T)
}

diffWGP_forward_timeind <- function(x0, step_size, time, model){
  # model is a nested model
  wis_model <- model[[1]]; mean_model <- model[[2]]
  # is this smart ?
  dt <- step_size; dt <- tf$cast(dt, dtype = tf$float32)
  #err <- tf$constant(5e-3, dtype = tf$float32)
  z <- wis_model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(wis_model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(wis_model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  
  if(wis_model$kern$is.RBF == TRUE){sc <- wis_model$kern$RBF$var}
  if(wis_model$kern$is.ARD == TRUE){sc <- wis_model$kern$ARD$var}
  sc <- tf$sqrt(sc)
  #normal_dist <- tf$distributions$Normal(loc = 0, scale = 1)
  
  N <- x0$get_shape()$as_list()[1]; D <- x0$get_shape()$as_list()[2]
  NU <- wis_model$deg_free
  
  L <- wis_model$L_scale_matrix # DxD or DxK
  sqdt <- tf$sqrt(dt)
  step_function<- function(x_t,t){
    # x_t is NxD, t is () probably
    
    K_MN <- build_kernel_matrix(wis_model,z,x_t)
    samp <- sample_gp_marginal(wis_model, x_t, K_q = K_q, K_MN = K_MN) #SxNxDxNU or SxNxKxNU, S = 1
    # if 1xNxD then squeeze
    samp <- tf$squeeze(samp)
    sqSig <- gp_to_sqrtwishart(L,samp) # NxDxNU or NxKxNU
    
    mu <- cond_mean(mean_model, x_t, K_q = K_q, K_MN = K_MN) #NxD
    
    #W <- tf$sqrt(dt) * normal_dist$sample(as.integer(c(N,NU,1))) # NxNUx1
    W <- tf$random_normal(as.integer(c(N,NU,1)))
    #sq_err <- sqrt(0.001)
    #W_err <- sq_err * tf$sqrt(dt) * normal_dist$sample(as.integer(c(N,D))) # NxD
    #W_err <- sq_err*tf$random_normal(as.integer(c(N,D)))
    #sc <- tf$sqrt(tf$exp(wis_model$kern$RBF$var) + 1e-8)
    
    #dx <- mu * dt + sc*sqdt*(tf$squeeze(tf$matmul(sqSig,W)) + sq_err*W_err) #NxD
    dx <- mu * dt + sqdt*(tf$squeeze(tf$matmul(sqSig,W)) )# + W_err)
    return(x_t + dx)
  }
  x_T <- tf$foldl(step_function,time,x0)
  return(x_T)
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