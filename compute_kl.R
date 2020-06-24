compute_kl <- function(model, K_q = NULL){
  if(is.null(K_q)){
    jitter = 1e-5
    z <- model$v_par$v_x;
    K_MM <- build_kernel_matrix(model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing))
    C_MM <- tf$cholesky(K_MM)
  } else{
    K_MM <- K_q$Kmm
    C_MM <- K_q$Kmmchol
  }
  
  A_sq <- get_chol(model$v_par$chol); # DxMxM or DxNUxMxM
  
  q_mu <- model$v_par$mu; #MxD or MxDxNU
  #p_mu <- float_32(model$mf(z)) ## ONLY IMPLEMENTED WITH ZERO MEAN FOR NOW
  
  ##
  log_det_p <- tf$reduce_sum(tf$log(tf$square(tf$matrix_diag_part(C_MM)))) # ()
  log_det_q <- tf$log(tf$square(tf$matrix_diag_part(A_sq))) # DxM or DxNUxM
  
  if(length(q_mu$get_shape()$as_list()) == 3){
    log_det_q <- tf$transpose(log_det_q, perm = as.integer(c(2,0,1)))#MxDxNU
    log_det_q <- tf$reduce_sum(log_det_q, axis = as.integer(0))[,,NULL] #DxNUx1
  } else{
    log_det_q <- tf$transpose(log_det_q, perm = as.integer(c(1,0))) #MxD
    log_det_q <- tf$reduce_sum(log_det_q, axis = as.integer(0))[,NULL] # Dx1
  }
  log_det_term <- log_det_p - log_det_q
  ##
  if(length(q_mu$get_shape()$as_list()) == 3){
    K_chol_tile <- tf$tile(C_MM[NULL,NULL,,], as.integer(c(A_sq$get_shape()$as_list()[1:2],1,1)))
    sol <- tf$matrix_triangular_solve(K_chol_tile,A_sq, lower = TRUE)
    trace_term <- tf$reduce_sum(tf$square(sol), axis = as.integer(c(2,3)))[,,NULL] #DxNUx1
  } else{
    K_chol_tile <- tf$tile(C_MM[NULL,,], as.integer(c(A_sq$get_shape()$as_list()[1],1,1)))
    sol <- tf$matrix_triangular_solve(K_chol_tile,A_sq, lower = TRUE)
    trace_term <- tf$reduce_sum(tf$square(sol), axis = as.integer(c(1,2)))[,NULL] #Dx1
  }
  
  ##
  if(length(q_mu$get_shape()$as_list()) == 3){
    q_mu_rs <- tf$reshape(q_mu, shape = as.integer(c(q_mu$get_shape()$as_list()[1],q_mu$get_shape()$as_list()[2]*q_mu$get_shape()$as_list()[3])))
    alph <- tf$matrix_triangular_solve(C_MM,q_mu_rs, lower = TRUE)
    alph <- tf$reshape(alph, shape = as.integer(c(q_mu$get_shape()$as_list()[1],q_mu$get_shape()$as_list()[2],q_mu$get_shape()$as_list()[3])))
    mahal_term <- tf$reduce_sum(tf$square(alph), axis = as.integer(0))[,,NULL] #DxNUx1
  } else{
    alph <- tf$matrix_triangular_solve(C_MM,q_mu, lower = TRUE) #MxD
    mahal_term <- tf$reduce_sum(tf$square(alph), axis = as.integer(0))[,NULL] #Dx1
  }
  
  kl <- 0.5*(log_det_term - model$v_par$num_inducing + trace_term + mahal_term)
  kl <- tf$reduce_sum(kl)
  return(kl)
}

fixed_var_kl <- function(model, K_q = NULL){
  if(is.null(K_q)){
    jitter = 1e-5
    z <- model$v_par$v_x;
    K_MM <- build_kernel_matrix(model,z,z,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing))
    C_MM <- tf$cholesky(K_MM)
  } else{
    K_MM <- K_q$Kmm
    C_MM <- K_q$Kmmchol
  }
  
  q_mu <- model$v_par$mu; #MxD or MxDxNU
  
  if(length(q_mu$get_shape()$as_list()) == 3){
    q_mu_rs <- tf$reshape(q_mu, shape = as.integer(c(q_mu$get_shape()$as_list()[1],q_mu$get_shape()$as_list()[2]*q_mu$get_shape()$as_list()[3])))
    alph <- tf$matrix_triangular_solve(C_MM,q_mu_rs, lower = TRUE)
    alph <- tf$reshape(alph, shape = as.integer(c(q_mu$get_shape()$as_list()[1],q_mu$get_shape()$as_list()[2],q_mu$get_shape()$as_list()[3])))
    mahal_term <- tf$reduce_sum(tf$square(alph), axis = as.integer(0))[,,NULL] #DxNUx1
  } else{
    alph <- tf$matrix_triangular_solve(C_MM,q_mu, lower = TRUE) #MxD
    mahal_term <- tf$reduce_sum(tf$square(alph), axis = as.integer(0))[,NULL] #Dx1
  }
  
  kl <- 0.5*(mahal_term) # Dx1
  kl <- tf$reduce_sum(kl)
  return(kl)
}