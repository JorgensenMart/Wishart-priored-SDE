library(tensorflow)
source("util.R"); source("make_gp_model.R"); source("build_kernel_matrix.R"); 
source("get_mu_and_var.R"); source("sample_gp_marginal.R"); source("compute_kl.R")

optimize <- function(x_train, y_train, x_test, y_test, model_name = "SGP"){
  source("forward.R")
  if(model_name == "diffGP"){source("forward_diffgp.R")}
  if(model_name == "no_noise"){source("no_noise.R")}
  else if(model_name == "diffWGP" || model_name == "diffWGP_constrain" || model_name == "diffWGP_constrain_TI"){source("forward_diffWGP.R")}
  
  ## Experiment settings
  N <- length(x_train[,1]) + length(x_test[,1]); D <- length(x_train[1,])
  
  #' Model and training settings
  num_inducing <- 100
  K <- min(5,D); nu <- K # Factorised model and degrees of freedom
  step_size <- 1/20; flowtime <- 1 # Integrator step in Euler-Murayama and maximal flowtime.
  warm_start_iter <- 10000 # How many initializing warm up steps
  mcsamples = 6 # Monte Carlo Samples
  train_iter <- 50000
  test_set_size <- length(x_test[,1])
  train_size <- length(x_train[,1])
  batch_size <- min(2000,train_size)
  
  
  ## Scaling the data
  x_train_means <- colMeans(x_train); x_test_means <- colMeans(x_test)
  x_test_sds <- sqrt(colMeans(x_test^2)-x_test_means^2) + 1e-8
  y_train_mean <- mean(y_train); 
  y_test_sd <- sd(y_test)
  
  x_train <- t(apply(x_train,1,function(x) (x - x_train_means)/x_test_sds));
  y_train <- (y_train - y_train_mean)/y_test_sd # Standardizing data
  x_test <- t(apply(x_test,1,function(x) (x - x_train_means)/x_test_sds));
  y_test <- (y_test - y_train_mean)/y_test_sd;
  
  if(model_name == "SGP"){
    model <- make_gp_model(kern.type = "ARD", input = x_train, output = y_train,
                           num_inducing = num_inducing, in_dim = D)
    
    init_ind <- init_inducing(x_train, num_inducing) # Initialize inducing locations with kmeans
    model$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32)
    
    x_batch <- tf$placeholder(tf$float32, shape(batch_size,D)); y_batch <- tf$placeholder(tf$float32, shape(batch_size,1))
    x_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,D)); y_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,1))
    trainer <- tf$train$AdamOptimizer(0.01)
    E <- forward(model, x_batch, y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss <- E$log_lh / as.double(batch_size) - E$kl/ as.double(train_size)# ELBO
    test_run <- forward(model, x_test_batch, y_test_batch, samples = 100,
                        y_mean = 0, y_std = y_test_sd, test_time = TRUE)
    optimizer <- trainer$minimize(-loss)
    
  } else if(model_name == "diffGP" || model_name == "no_noise"){
    diff_layer <- make_gp_model(kern.type = "ARD",
                                input = x_train,
                                num_inducing = num_inducing,
                                in_dim = D, out_dim = D)
    
    init_ind <- init_inducing(x_train, num_inducing) # Initialize inducing locations with kmeans
    diff_layer$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32)
    
    diff_layer$kern$ARD <- list(ls = tf$Variable(rep(log(exp(2)-1),D), constraint = constrain_pos),
                                var = tf$Variable(0.01, constraint = constrain_pos), 
                                eps = tf$Variable(log(exp(0.01)-1)))
    
    diff_layer$v_par$chol <- sqrt(0.01)*diff_layer$v_par$chol
    #' The predictor
    pred_layer <- make_gp_model(kern.type = "ARD",
                                num_inducing = num_inducing,
                                input = x_train,
                                output = y_train,
                                likelihood = "Gaussian",
                                in_dim = D,
                                out_dim = 1)
    
    pred_layer$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32)
    
    #' MODEL
    model <- list(diff_layer,pred_layer)
    
    #' TRAINING
    x_batch <- tf$placeholder(tf$float32, shape(batch_size,D)); y_batch <- tf$placeholder(tf$float32, shape(batch_size,1))
    x_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,D)); y_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,1))
    
    trainer_ws <- tf$train$AdamOptimizer(0.01)
    trainer <- tf$train$AdamOptimizer(0.001)
    E <- forward_diffgp(model,x_batch,y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss <- E$log_lh / as.double(batch_size) - E$kl / as.double(train_size) # ELBO
    test_run <- forward_diffgp(model, x_test_batch, y_test_batch, samples = 100, y_std = y_test_sd, test_time = TRUE) # For test runs
    Ews <- forward(model[[2]], x_batch, y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss_ws <- Ews$log_lh / as.double(batch_size) - Ews$kl / as.double(train_size) # ELBO
    optimizer <- trainer$minimize(-loss)
    ws_optimizer <- trainer_ws$minimize(-loss_ws)
  } else if(model_name == "diffWGP" ){
    #' Wishart model
    wis_model <- make_gp_model(kern.type = "ARD",
                               input = x_train,
                               num_inducing = num_inducing,
                               in_dim = D+1, out_dim = D,
                               is.WP = TRUE, wis_factor = K,
                               deg_free = nu)
    #' Mean model
    mean_model <- make_gp_model(kern.type = "ARD",
                                input = x_train,
                                num_inducing = num_inducing,
                                in_dim = D+1, out_dim = D)
    
    #' Shared parameters
    wis_model$kern$ARD = mean_model$kern$ARD <- list(ls = tf$Variable(rep(log(exp(2)-1),D+1), constraint = constrain_pos), 
                                                     var = tf$Variable(0.01, constraint = constrain_pos), 
                                                     eps = tf$Variable(log(exp(0.01)-1)))
    #mean_model$kern$ARD$var <- tf$constant(1, dtype = tf$float32)
    wis_model$v_par$chol <- sqrt(0.01)*wis_model$v_par$chol
    mean_model$v_par$chol <- sqrt(0.01)*mean_model$v_par$chol
    err <- 0.1
    v_t <- matrix(rep(seq(0 + err,flowtime - err, length.out = 2),num_inducing/2),ncol = 1)
    
    cent1 <- x_train[sample(1:length(x_train[,1]),num_inducing/2),]
    cent1 <- cent1 + matrix(rnorm(length(cent1), sd = 1e-8), ncol = dim(cent1)[2])
    cent2 <- x_train[sample(1:length(x_train[,1]),num_inducing/2),]
    cent2 <- cent2 + matrix(rnorm(length(cent2), sd = 1e-8), ncol = dim(cent2)[2])
    kmx50_v1 <- kmeans(x_train, centers = cent1,
                  iter.max = 1000, nstart = 10)
    kmx50_v2 <- kmx <- kmeans(x_train, centers = cent2,
                              iter.max = 1000, nstart = 10)
    kmx50 <- kmeans(x_train, num_inducing/2, iter.max = 1000, nstart = 10)
    C1 <- kmx50_v1$centers; C2 <- kmx50_v2$centers;
    C <- rbind(C1,C2); 
    C <- cbind(C,v_t)
    
    wis_model$v_par$v_x = mean_model$v_par$v_x <- tf$Variable(C, dtype = tf$float32) # Initialize inducing points
    
    diff_layer <- list(wis_model,mean_model) # nested model
    
    #' The predictor
    pred_layer <- make_gp_model(kern.type = "ARD",
                                num_inducing = num_inducing,
                                input = x_train,
                                output = y_train,
                                likelihood = "Gaussian",
                                in_dim = D,
                                out_dim = 1)
    cent <- x_train[sample(1:length(x_train[,1]),num_inducing),]
    cent <- cent + matrix(rnorm(length(cent), sd = 1e-5), ncol = dim(cent)[2])
    kmx <- kmeans(x_train, centers = cent,
                  iter.max = 1000, nstart = 10) # Initialize inducing locations with kmeans
    pred_layer$v_par$v_x <- tf$Variable(kmx$centers, dtype = tf$float32)
    
    #' MODEL
    model <- list(diff_layer,pred_layer)
    
    #' TRAINING
    x_batch <- tf$placeholder(tf$float32, shape(batch_size,D)); y_batch <- tf$placeholder(tf$float32, shape(batch_size,1))
    x_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,D)); y_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,1))
    
    trainer_ws <- tf$train$AdamOptimizer(0.01)
    trainer <- tf$train$AdamOptimizer(0.001)
    E <- forward_diffWGP(model,x_batch,y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss <- E$log_lh / as.double(batch_size) - E$kl / as.double(train_size) # ELBO
    test_run <- forward_diffWGP(model, x_test_batch, y_test_batch, samples = 100, y_std = y_test_sd, test_time = TRUE) # For test runs
    Ews <- forward(model[[2]], x_batch, y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss_ws <- Ews$log_lh / as.double(batch_size) - Ews$kl / as.double(train_size) # ELBO
    ws_optimizer <- trainer_ws$minimize(-loss_ws)
    #clipped_trainer <- tf$contrib$estimator$clip_gradients_by_norm(trainer, clip_norm = 2.0) # GRADIENT CLIPPING
    #optimizer <- clipped_trainer$minimize(-loss)
    optimizer <- trainer$minimize(-loss)
  } else if(model_name == "diffWGP_constrain" ){
    #' Wishart model
    wis_model <- make_gp_model(kern.type = "ARD",
                               input = x_train,
                               num_inducing = num_inducing,
                               in_dim = D+1, out_dim = D,
                               is.WP = TRUE, wis_factor = K,
                               deg_free = nu, constrain_deg_free = TRUE)
    #' Mean model
    mean_model <- make_gp_model(kern.type = "ARD",
                                input = x_train,
                                num_inducing = num_inducing,
                                in_dim = D+1, out_dim = D)
    
    #' Shared parameters
    wis_model$kern$ARD = mean_model$kern$ARD <- list(ls = tf$Variable(rep(log(exp(2)-1),D+1), constraint = constrain_pos), 
                                                     var = tf$Variable(0.01, constraint = constrain_pos), 
                                                     eps = tf$Variable(log(exp(0.01)-1)))
    #mean_model$kern$ARD$var <- tf$constant(1, dtype = tf$float32)
    wis_model$v_par$chol <- sqrt(0.01)*wis_model$v_par$chol
    mean_model$v_par$chol <- sqrt(0.01)*mean_model$v_par$chol
    err <- 0.1
    v_t <- matrix(rep(seq(0 + err,flowtime - err, length.out = 2),num_inducing/2),ncol = 1)
    
    cent1 <- x_train[sample(1:length(x_train[,1]),num_inducing/2),]
    cent1 <- cent1 + matrix(rnorm(length(cent1), sd = 1e-8), ncol = dim(cent1)[2])
    cent2 <- x_train[sample(1:length(x_train[,1]),num_inducing/2),]
    cent2 <- cent2 + matrix(rnorm(length(cent2), sd = 1e-8), ncol = dim(cent2)[2])
    kmx50_v1 <- kmeans(x_train, centers = cent1,
                       iter.max = 1000, nstart = 10)
    kmx50_v2 <- kmx <- kmeans(x_train, centers = cent2,
                              iter.max = 1000, nstart = 10)
    kmx50 <- kmeans(x_train, num_inducing/2, iter.max = 1000, nstart = 10)
    C1 <- kmx50_v1$centers; C2 <- kmx50_v2$centers;
    C <- rbind(C1,C2); 
    C <- cbind(C,v_t)
    
    wis_model$v_par$v_x = mean_model$v_par$v_x <- tf$Variable(C, dtype = tf$float32) # Initialize inducing points
    
    diff_layer <- list(wis_model,mean_model) # nested model
    
    #' The predictor
    pred_layer <- make_gp_model(kern.type = "ARD",
                                num_inducing = num_inducing,
                                input = x_train,
                                output = y_train,
                                likelihood = "Gaussian",
                                in_dim = D,
                                out_dim = 1)
    cent <- x_train[sample(1:length(x_train[,1]),num_inducing),]
    cent <- cent + matrix(rnorm(length(cent), sd = 1e-5), ncol = dim(cent)[2])
    kmx <- kmeans(x_train, centers = cent,
                  iter.max = 1000, nstart = 10) # Initialize inducing locations with kmeans
    pred_layer$v_par$v_x <- tf$Variable(kmx$centers, dtype = tf$float32)
    
    #' MODEL
    model <- list(diff_layer,pred_layer)
    
    #' TRAINING
    x_batch <- tf$placeholder(tf$float32, shape(batch_size,D)); y_batch <- tf$placeholder(tf$float32, shape(batch_size,1))
    x_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,D)); y_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,1))
    
    trainer_ws <- tf$train$AdamOptimizer(0.01)
    trainer <- tf$train$AdamOptimizer(0.001)
    E <- forward_diffWGP(model,x_batch,y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss <- E$log_lh / as.double(batch_size) - E$kl / as.double(train_size) # ELBO
    test_run <- forward_diffWGP(model, x_test_batch, y_test_batch, samples = 100, y_std = y_test_sd, test_time = TRUE) # For test runs
    Ews <- forward(model[[2]], x_batch, y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss_ws <- Ews$log_lh / as.double(batch_size) - Ews$kl / as.double(train_size) # ELBO
    ws_optimizer <- trainer_ws$minimize(-loss_ws)
    #clipped_trainer <- tf$contrib$estimator$clip_gradients_by_norm(trainer, clip_norm = 2.0) # GRADIENT CLIPPING
    #optimizer <- clipped_trainer$minimize(-loss)
    optimizer <- trainer$minimize(-loss)
  } else if(model_name == "diffWGP_constrain_TI" ){
    #' Wishart model
    wis_model <- make_gp_model(kern.type = "ARD",
                               input = x_train,
                               num_inducing = num_inducing,
                               in_dim = D, out_dim = D,
                               is.WP = TRUE, wis_factor = K,
                               deg_free = nu, constrain_deg_free = TRUE)
    #' Mean model
    mean_model <- make_gp_model(kern.type = "ARD",
                                input = x_train,
                                num_inducing = num_inducing,
                                in_dim = D, out_dim = D)
    
    #' Shared parameters
    wis_model$kern$ARD = mean_model$kern$ARD <- list(ls = tf$Variable(rep(log(exp(2)-1),D), constraint = constrain_pos), 
                                                     var = tf$Variable(0.01, constraint = constrain_pos), 
                                                     eps = tf$Variable(log(exp(0.01)-1)))
    #mean_model$kern$ARD$var <- tf$constant(1, dtype = tf$float32)
    wis_model$v_par$chol <- sqrt(0.01)*wis_model$v_par$chol
    mean_model$v_par$chol <- sqrt(0.01)*mean_model$v_par$chol
    
    init_ind <- init_inducing(x_train, num_inducing) # Initialize inducing locations with kmeans
    wis_model$v_par$v_x = mean_model$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32) # Initialize inducing points
    
    diff_layer <- list(wis_model,mean_model) # nested model
    
    #' The predictor
    pred_layer <- make_gp_model(kern.type = "ARD",
                                num_inducing = num_inducing,
                                input = x_train,
                                output = y_train,
                                likelihood = "Gaussian",
                                in_dim = D,
                                out_dim = 1)

    pred_layer$v_par$v_x <- tf$Variable(init_ind, dtype = tf$float32)
    
    #' MODEL
    model <- list(diff_layer,pred_layer)
    
    #' TRAINING
    x_batch <- tf$placeholder(tf$float32, shape(batch_size,D)); y_batch <- tf$placeholder(tf$float32, shape(batch_size,1))
    x_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,D)); y_test_batch <- tf$placeholder(tf$float32, shape(test_set_size,1))
    at_iter <- tf$placeholder(tf$float32, shape())
    
    trainer_ws <- tf$train$AdamOptimizer(0.01)
    trainer <- tf$train$AdamOptimizer(0.001)
    E <- forward_diffWGP(model,x_batch,y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE, time_independent = TRUE, warm_start = TRUE, scalar = at_iter)
    loss <- E$log_lh / as.double(batch_size) - E$kl / as.double(train_size) # ELBO
    test_run <- forward_diffWGP(model, x_test_batch, y_test_batch, samples = 100, y_std = y_test_sd, test_time = TRUE, time_independent = TRUE) # For test runs
    Ews <- forward(model[[2]], x_batch, y_batch, samples = mcsamples, y_std = y_test_sd, test_time = FALSE)
    loss_ws <- Ews$log_lh / as.double(batch_size) - Ews$kl / as.double(train_size) # ELBO
    ws_optimizer <- trainer_ws$minimize(-loss_ws)
    #clipped_trainer <- tf$contrib$estimator$clip_gradients_by_norm(trainer, clip_norm = 2.0) # GRADIENT CLIPPING
    #optimizer <- clipped_trainer$minimize(-loss)
    optimizer <- trainer$minimize(-loss)
  }
  
  # Initialize session
  init <- tf$global_variables_initializer()
  session <- tf$Session()
  session$run(init)
  
  if(model_name == "diffWGP" || model_name == "diffWGP_constrain" || model_name == "diffWGP_constrain_TI"){
    sample_covariance <- cov(x_train)
    wis_model$L_scale_matrrix <- initialize_L(wis_model$L_scale_matrix, session = session, sample_covariance)
    wis_model$L_scale_matrix <- row_sum_L2(wis_model$L_scale_matrix)
  }
  
  print_every = 500
  ## TRAINING
  if(model_name == "SGP"){
    for( i in 1:(train_iter)){
      J <- sample(1:train_size, batch_size)
      x_bat <- x_train[J,]; y_bat <- matrix(y_train[J,])
      batch_dict <- dict(x_batch = x_bat, y_batch = y_bat)
      session$run(optimizer,feed_dict = batch_dict)
      if(i%%print_every == 0){
        Es <- session$run(E, feed_dict = batch_dict)
        ELBO <- Es$log_lh / as.double(batch_size) - Es$kl / as.double(train_size)
        cat("Iteraion", i, "out of", train_iter, "\n")
        cat("ELBO:", ELBO)
        batch_dict <- dict(x_test_batch = x_test, y_test_batch = y_test)
        TEST_SCORES <- session$run(test_run, feed_dict = batch_dict)
        cat("Test log-likelihood:", TEST_SCORES$log_lh / as.double(test_set_size), "Test RMSE:", TEST_SCORES$RMSE, "\n")
      }
    }
  } else{
    for( i in 1:warm_start_iter ){
      J <- sample(1:train_size, batch_size)
      x_bat <- x_train[J,]; y_bat <- matrix(y_train[J,])
      batch_dict <- dict(x_batch = x_bat, y_batch = y_bat)
      session$run(ws_optimizer, feed_dict = batch_dict)
      if(i%%print_every == 0){
        Es <- session$run(Ews, feed_dict = batch_dict)
        ELBO <-  Es$log_lh / as.double(batch_size) - Es$kl / as.double(train_size)
        cat("Iteraion", i, "out of", train_iter, "\n")
        cat("ELBO:", ELBO, "\n")
      }
    }
    for( i in (warm_start_iter+1):train_iter ){
      J <- sample(1:train_size, batch_size)
      x_bat <- x_train[J,]; y_bat <- matrix(y_train[J,])
      alpha <- min(1,(i-warm_start_iter) / ((train_iter-warm_start_iter)*0.1)) # 10perc of warmup KL
      if(model_name == "diffGP" || model_name == "no_noise"){
        batch_dict <- dict(x_batch = x_bat, y_batch = y_bat)
      } else{
        batch_dict <- dict(x_batch = x_bat, y_batch = y_bat, at_iter = alpha)
      }
      session$run(optimizer, feed_dict = batch_dict)
      if(i%%print_every == 0){
        Es <- session$run(E, feed_dict = batch_dict)
        ELBO <-  Es$log_lh / as.double(batch_size) - Es$kl / as.double(train_size)
        cat("Iteraion", i, "out of", train_iter , "\n")
        cat("ELBO:", ELBO, "Training log-likelihood:", Es$log_lh / as.double(batch_size), "KL:", Es$kl / as.double(train_size), "Training RMSE:",Es$RMSE, "\n")
        if(model_name == "diffGP" || model_name == "no_noise"){
        cat("Field variance:", session$run(diff_layer$kern$ARD$var), "\n")
        cat("Noise:", session$run(pred_layer$kern$ARD$eps), "\n")
        } else{
          cat("Field variance:", session$run(wis_model$kern$ARD$var), "\n")
          cat("Noise:", session$run(pred_layer$kern$ARD$eps), "\n")
        }
        batch_dict <- dict(x_test_batch = x_test, y_test_batch = y_test)
        TEST_SCORES <- session$run(test_run, feed_dict = batch_dict)
        cat("Test log-likelihood:", TEST_SCORES$log_lh / as.double(test_set_size), "Test RMSE:", TEST_SCORES$RMSE, "\n")
      }
    }
  }
  
  ## Evaluating
  batch_dict <- dict(x_test_batch = x_test, y_test_batch = y_test)
  df <- data.frame()
  for(i in 1:1){
    TEST_SCORES <- session$run(test_run, feed_dict = batch_dict)
    test_loglik <- TEST_SCORES$log_lh / as.double(test_set_size)
    test_rmse <- TEST_SCORES$RMSE
    this.run <- data.frame(test_loglik = test_loglik, test_rmse = test_rmse)
    df <- rbind(df,this.run)
  }
  return(df)
}