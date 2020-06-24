# Optimize time-series

source("util.R"); source("make_gp_model.R"); source("build_kernel_matrix.R"); 
source("get_mu_and_var.R"); source("sample_gp_marginal.R"); source("compute_kl.R")
source("timeseries_diffWGP.R")

optimize_ts <- function(y_train, y_test, time,rho){
  N <- length(y_train[,1]); D <- length(y_train[1,])
  
  #' Model and training settings
  num_inducing <- 100
  K <- min(as.integer(rho),D); nu <- K # Factorised model and degrees of freedom
  warm_start_iter <- 100 # How many initializing warm up steps
  mcsamples = 4 # Monte Carlo Samples
  train_iter <- 40000
  train_size <- N
  running_batch_size <- as.integer(10) # init at 24
  batch_size <- tf$placeholder(tf$int32,shape())
  
  #' Wishart model
  wis_model <- make_gp_model(kern.type = "ARD",
                             input = y_train,
                             num_inducing = num_inducing,
                             in_dim = D, out_dim = D,
                             is.WP = TRUE, wis_factor = K,
                             deg_free = nu, constrain_deg_free = TRUE, restrict_L = FALSE)
  #' Mean model
  mean_model <- make_gp_model(kern.type = "ARD",
                              input = y_train,
                              num_inducing = num_inducing,
                              in_dim = D, out_dim = D)
  
  #' Shared parameters
  wis_model$kern$ARD = mean_model$kern$ARD <- list(ls = tf$Variable(rep(log(exp(2)-1),D), constraint = constrain_pos), 
                                                   var = tf$Variable(1, constraint = constrain_pos), 
                                                   eps = tf$Variable(log(exp(0.01)-1)))
  
  init_ind <- init_inducing(y_train, num_inducing) # Initialize inducing locations with kmeans
  wis_model$v_par$v_x = mean_model$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32) # Initialize inducing points
  
  model <- list(wis_model,mean_model) # nested model
  
  y_batch <- tf$placeholder(tf$float32, shape(NULL,D)); time_batch <- tf$placeholder(tf$float32, shape(NULL));
  
  trainer <- tf$train$AdamOptimizer(0.01, beta1 = 0.4, beta2 = 0.8)
  #callable_func1 <- function(y_batch){
  #  return(E <- forwardtime_diffWGP(model, y_batch, time, y_mean = 0, y_std = 1, c = 0.01))
  #}
  #E <- tf$map_fn(callable_func1,tf$tile(y_batch[NULL,,],as.integer(c(5,1,1))))
  sample_covariance <- cov(y_train); c <- tf$Variable(rep(0.1*max(diag(sample_covariance)),D), constraint = constrain_pos);
  E <- forwardtime_diffWGP(model, y_batch, time_batch, y_mean = 0, y_std = 1, c = c, 
                           mc_samples = mcsamples, train_size = train_size,batch_size = batch_size) #Think about scaling
  loss <- E$llh / tf$cast(batch_size, tf$float32) - E$kl / as.double(train_size) # ELBO
  optimizer <- trainer$minimize(-loss)
  TS_test <- forwardtime_diffWGP(model, y_batch, time_batch, y_mean = 0, y_std = 1, c = c, 
                           mc_samples = 50, train_size = train_size,batch_size = batch_size, test_time = TRUE)
  # Initialize session
  init <- tf$global_variables_initializer()
  session <- tf$Session()
  session$run(init)
  
  wis_model$L_scale_matrrix <- initialize_L(wis_model$L_scale_matrix, session = session, sample_covariance) #Warmstart L
  
  #' Training
  print_every = 50
  for( i in 1:train_iter ){
    if(i %% 4000 == 0){
      running_batch_size <- as.integer(min(running_batch_size + 5,train_size))
    }
    t0 <- sample(1:(as.integer(train_size)-running_batch_size+as.integer(1)),1)
    batch_dict <- dict(y_batch = y_train[t0:(t0+running_batch_size-1),], time_batch = time[t0:(t0+running_batch_size-1)], batch_size = running_batch_size)
    session$run(optimizer, feed_dict = batch_dict)
    if(i%%print_every == 0){
      Es <- session$run(E, feed_dict = batch_dict)
      ELBO <-  Es$llh / as.double(running_batch_size) - Es$kl / as.double(train_size)
      cat("Iteraion", i, "out of", train_iter , "\n")
      cat("ELBO:", ELBO, "Training log-likelihood:", Es$llh / as.double(running_batch_size), "KL:", Es$kl / as.double(train_size), "\n")
      #cat("Final state:", Es$final_state, "\n")
    }
  }
  
  # Predict horizon
  #test_set_size <- length(y_test[1:(24*31),1])
  horizon <- as.integer(48)
  batch_dict <- dict(y_batch = y_test[1:horizon,], time_batch = seq(17521:(17521+horizon-1)), batch_size = horizon)
  outtest <- session$run(TS_test, feed_dict = batch_dict)
  results <- outtest
  return(results)
  }