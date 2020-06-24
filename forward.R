forward <- function(model, x_batch, y_batch, samples = 4, y_mean = 0, y_std = 1, test_time = FALSE){
  tile_x <- tf$tile(x_batch[NULL,,], as.integer(c(samples,1,1))) # SxNxD
  
  z <- model$v_par$v_x; jitter <- 1e-5
  K_q <- list(Kmm = build_kernel_matrix(model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing)))
  K_q$Kmmchol <- tf$cholesky(K_q$Kmm)
  
  wrapper_pred <- function(x_batch){
    standin_model <- model
    Fpars <- get_mu_and_var(x_batch, standin_model, K_q = K_q) # list of Fmu's and Fvar's
    Fmu <- Fpars$mean; Fvar <- Fpars$var
    L <- tf$stack(c(Fmu,Fvar),axis = as.integer(0))
    return(L)
  }
  
  S <- tf$map_fn(wrapper_pred, tile_x) # Sx2xNxD
  
  ## RMSE
  Fmu_mean <- tf$reduce_mean(S[,1,,], axis = as.integer(0)) # NxD
  #RMSE <- tf$sqrt(tf$reduce_mean(tf$square(Fmu_mean*y_std -(y_batch*y_std))))
  RMSE <- tf$sqrt(tf$reduce_mean(y_std^2*tf$square(Fmu_mean-y_batch)))
  
  ## Log-likelihood
  likelihood <- function(S){
    if(model$kern$is.RBF == TRUE){e <- softplus(model$kern$RBF$eps)}
    if(model$kern$is.ARD == TRUE){e <- softplus(model$kern$ARD$eps)}
    if(model$kern$is.lin == TRUE){e <- softplus(model$kern$lin$eps)}
    Fmu <- S[1,,]; Fvar <- S[2,,] #NxD (here Nx1)
    if(test_time == FALSE){
      sum_log_lik <- - 0.5 * log(2 * pi) - 0.5*tf$log(y_std^2*e) -
        0.5*(tf$square(y_batch - Fmu) + Fvar)/e
      sum_log_lik <- tf$reduce_sum(sum_log_lik)
    }
    if(test_time == TRUE){
      pred_dist <- tf$distributions$Normal(loc = y_std*Fmu, scale = y_std*tf$sqrt(Fvar + e))
      sum_log_lik <- pred_dist$log_prob(y_std*y_batch)
      sum_log_lik <- tf$reduce_sum(sum_log_lik)
    }
    return(sum_log_lik)
  }
  LogLik <- tf$map_fn(likelihood,S) # Sx
  LogLik <- tf$reduce_mean(LogLik) # 
  
  KL <- compute_kl(model, K_q = K_q)
  E <- list(log_lh =LogLik, kl = KL, RMSE = RMSE, Fmu_and_Fvar = S)
  return(E)
}