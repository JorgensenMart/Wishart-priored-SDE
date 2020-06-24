sample_gp_marginal <- function(model, x_batch, samples  = 1, K_q = NULL, K_MN = NULL){
  L <- get_mu_and_var(x_batch, model, K_q = K_q, K_MN = K_MN)
  mu <- L$mean; var <- L$var #NxDxNU
  if(length(mu$get_shape()$as_list()) == 3){
    w <- tf$random_normal(shape(samples,
                                mu$get_shape()$as_list()[1],
                                mu$get_shape()$as_list()[2],
                                mu$get_shape()$as_list()[3]),
                          mean = 0,
                          stddev = 1) #SxNxDxNU
    out <- mu + tf$sqrt(var)*w 
  }
  else if(model$is.WP == TRUE && model$constrain_deg_free == TRUE){
    var <- tf$expand_dims(var,as.integer(2))
    w <- tf$random_normal(shape(samples,
                                mu$get_shape()$as_list()[1],
                                mu$get_shape()$as_list()[2],
                                model$deg_free),
                          mean = 0,
                          stddev = 1) #SxNxDxNU
    mu <- tf$tile(mu[,,NULL], as.integer(c(1,1,model$deg_free)))
    out <- mu + tf$sqrt(var)*w 
  }
  else{
    w <- tf$random_normal(shape(samples,
                                mu$get_shape()$as_list()[1],
                                mu$get_shape()$as_list()[2]),
                          mean = 0,
                          stddev = 1) #SxNxD
    out <- mu + tf$sqrt(var)*w 
  } 
  return(out)
}