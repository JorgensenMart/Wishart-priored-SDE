make_gp_model <- function(kern.type = "RBF", input, output, mf = fun, order = NULL, v_par = v_par, num_inducing, 
                          likelihood = "Gaussian", in_dim, out_dim = 1, 
                          is.WP = FALSE, wis_factor = NULL, deg_free = NULL, constrain_deg_free = FALSE, restrict_L = TRUE){
  model <- list()
  model$likelihood = likelihood
  model$kern$is.RBF = FALSE; model$kern$is.lin = FALSE; model$kern$is.polynomial = FALSE; model$kern$is.ARD = FALSE
  if(missing(mf)){
    model$mf = function(s){
      a <- tf$zeros_like(s, dtype = tf$float32)
      if(a$shape == c(a$shape[0])){
        a <- tf$reshape(a,shape(-1,1))
      }
      return(a)
    }
  }
  else{
    model$mf = mf
  }
  if(is.WP == TRUE){
    model$is.WP <- TRUE
    if(missing(wis_factor)){wis_factor = out_dim}
    if(restrict_L == TRUE){
    model$L_scale_matrix <- tf$Variable(matrix(rnorm(out_dim*wis_factor), 
                                               nrow = out_dim),
                                        dtype = tf$float32,
                                        constraint = row_sum_L2)
    } else{
      model$L_scale_matrix <- tf$Variable(matrix(rnorm(out_dim*wis_factor), 
                                                 nrow = out_dim),
                                          dtype = tf$float32)
    }
    # A = L %*% L^T
    model$wis_factor <- wis_factor
    if(missing(deg_free)){ deg_free = wis_factor }
    model$deg_free <- as.integer(deg_free)
    model$constrain_deg_free = constrain_deg_free
  } else{
    model$is.WP <- FALSE
  }
  
  if(missing(kern.type)){kern.type == "RBF"}
  if(kern.type == "RBF"){
    model$kern$is.RBF = TRUE
    model$kern$RBF$ls = tf$Variable(log(exp(2)-1))
    model$kern$RBF$var = tf$Variable(2, constraint = constrain_pos)
    model$kern$RBF$eps = tf$Variable(log(exp(0.01)-1))
  }
  if(kern.type == "ARD"){
    model$kern$is.ARD = TRUE
    model$kern$ARD$ls = tf$Variable(rep(log(exp(2)-1),in_dim))
    model$kern$ARD$var = tf$Variable(2, constraint = constrain_pos)
    model$kern$ARD$eps = tf$Variable(log(exp(0.01)-1))
  }
  if(kern.type == "lin"){
    model$kern$is.lin = TRUE
    model$kern$lin$a = tf$Variable(tf$ones(shape()))
    model$kern$lin$b = tf$Variable(tf$ones(shape()))
    model$kern$lin$eps = tf$Variable(log(exp(0.01)-1))
  }
  if(kern.type == "polynomial"){
    model$kern$is.polynomial = TRUE
    model$kern$polynomial$d = float_32(tf$constant(order))
    model$kern$polynomial$a = tf$Variable(tf$ones(shape()))
    model$kern$polynomial$b = tf$Variable(tf$ones(shape()))
    model$kern$polynomial$eps = tf$Variable(log(exp(0.01)-1))
  }
  
  # v_par
  if(missing(num_inducing)){
    model$v_par = NULL
  }
  else{
    model$v_par$num_inducing = num_inducing
    N = length(input[,1])
    I <- sample(1:N, num_inducing, replace = FALSE)
    
    # Initializes with random sample
    model$v_par$v_x = tf$Variable(input[I,], shape(as.integer(c(num_inducing,in_dim))), dtype = tf$float32)
    if(is.WP == TRUE){
      if(model$constrain_deg_free == FALSE){
        model$v_par$mu = tf$Variable(array(rep(0,num_inducing*deg_free*wis_factor),c(num_inducing,wis_factor,model$deg_free)), dtype = tf$float32)
        model$v_par$chol <- tf$Variable(float_32(tf$contrib$distributions$fill_triangular_inverse(tf$eye(as.integer(model$v_par$num_inducing), 
                                                                                                         batch_shape = as.integer(c(wis_factor,model$deg_free))))))
      }
      else{
        model$v_par$mu = tf$Variable(matrix(rep(0,num_inducing*wis_factor),nrow = num_inducing), shape(c(num_inducing,wis_factor)), dtype = tf$float32)
        model$v_par$chol <- tf$Variable(float_32(tf$contrib$distributions$fill_triangular_inverse(tf$eye(as.integer(model$v_par$num_inducing), 
                                                                                                         batch_shape = c(int_32(wis_factor))))))
      }
    }
    else{
      model$v_par$mu = tf$Variable(matrix(rep(0,num_inducing*out_dim),nrow = num_inducing), shape(c(num_inducing,out_dim)), dtype = tf$float32)
      model$v_par$chol <- tf$Variable(float_32(tf$contrib$distributions$fill_triangular_inverse(tf$eye(as.integer(model$v_par$num_inducing), 
                                                                                                       batch_shape = c(int_32(out_dim))))))
    } 
  }
  return(model)
}