forward_diffgp <- function(model,x_batch,y_batch, samples = 4, step_size = 1/20, flowtime = 1, 
                           y_mean = 0, y_std = 1, test_time = FALSE){
  
  tile_x <- tf$tile(x_batch[NULL,,], as.integer(c(samples,1,1)))
  time_grid <- seq(0,flowtime-step_size, by = step_size); time_grid <- tf$convert_to_tensor(time_grid)
  
  wrapper_diff_forward <- function(x_batch){
    return( diff_forward(x0 = x_batch, step_size = step_size,
                         time = time_grid, model = model[[1]]) )
  }
  
  f_list <- tf$map_fn(wrapper_diff_forward, tile_x) #SxNxD
  KL_1 <- compute_kl(model[[1]])
  
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
  LogLik <- tf$reduce_mean(LogLik) # 
  
  KL_2 <- compute_kl(model[[2]])
  KL <- KL_2 + KL_1

  E <- list(log_lh = LogLik, kl = KL, RMSE = RMSE)
  return(E)
}

diff_forward <- function(x0,step_size,time,model){
  dt <- step_size; dt <- tf$cast(dt, dtype = tf$float32);
  z <- model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  step_function <- function(x_t,t){ #,t has to be there
    L <- get_mu_and_var(x_t,model, K_q = K_q)
    mu <- L$mean # NxD
    var <- L$var # NxD
    w <- tf$random_normal(shape(x_t$get_shape()$as_list()[1],
                                x_t$get_shape()$as_list()[2]))
    dx <- mu * dt + tf$sqrt(dt)*tf$sqrt(var)*w
    return(x_t + dx)
  }
  x_T <- tf$foldl(step_function, time, initializer = x0)
  return(x_T)
}