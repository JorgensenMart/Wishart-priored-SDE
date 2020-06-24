get_data <- function(dataset){
  
  if(dataset == "boston"){
    boston.uci <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"),
                           header=FALSE, sep = "")
    x <- data.matrix(boston.uci[1:13])
    y <- data.matrix(boston.uci[14])
  }
  
  if(dataset == "kin8nm"){
    kin8nm.uci <- read.csv(url("https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff"))
    x <- data.matrix(kin8nm.uci[1:8])
    y <- data.matrix(kin8nm.uci[9])
  }
  
  if(dataset == "power"){
    temp <- tempfile()
    download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",temp)
    path <- "CCPP/Folds5x2_pp.xlsx"
    unzip(zipfile = temp, files = path)
    library(readxl)
    power.uci <- read_excel(path)
    x <- data.matrix(power.uci[1:4])
    y <- data.matrix(power.uci[5])
  }
  
  if(dataset == "concrete"){
    if(!require(MAVE)){install.packages("MAVE")}
    library(MAVE)
    data(Concrete)
    x <- data.matrix(Concrete[1:8])
    y <- data.matrix(Concrete[9])
  }
  
  if(dataset == "energy"){
    temp <- tempfile()
    download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",temp, mode = "wb")
    library(readxl)
    energy.uci <- read_excel(temp)
    x <- data.matrix(energy.uci[1:8])
    y <- data.matrix(energy.uci[9])
  }
  
  if(dataset == "wine_red"){
    red_wine.uci <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),
                           header=TRUE, sep = ";")
    x <- data.matrix(red_wine.uci[1:11])
    y <- data.matrix(red_wine.uci[12])
  }
  
  if(dataset == "wine_white"){
    white_wine.uci <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"),
                             header=TRUE, sep = ";")
    x <- data.matrix(white_wine.uci[1:11])
    y <- data.matrix(white_wine.uci[12])
  }
  
  if(dataset == "naval"){
    temp <- tempfile()
    download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",temp)
    path <- "UCI CBM Dataset/data.txt"
    naval.uci <- read.table(unz(temp, path))
    x <- data.matrix(naval.uci[1:16])
    y <- data.matrix(naval.uci[17])
  }
  
  if(dataset == "protein"){
    protein.uci <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"),
                             header=TRUE, sep = ",")
    x <- data.matrix(protein.uci[2:10])
    y <- data.matrix(protein.uci[1])
  }
  
  if(dataset == "bike"){
    temp <- tempfile()
    download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",temp)
    path <- "hour.csv"
    bike.uci <- read.csv(unz(temp,path))
    x <- data.matrix(bike.uci[3:16])
    y <- data.matrix(bike.uci[17])
  }
  
  data <- list(x = x, y = y)
  return(data)
}