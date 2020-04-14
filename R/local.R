# R Studio API Code 

library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path))

# Libraries

library(haven)
library(tidyverse)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)

# Data Import and Cleaning

full_data <- read_sav("../data/GSS2006.sav")
clean_data <- full_data %>%
    select_at(vars(c(starts_with("BIG5"), "HEALTH"))) %>%
    mutate_all("as.numeric") %>%
    drop_na("HEALTH") %>%
    filter(rowSums(is.na(.[,1:10])) != 10)


# Analysis

index <- createFolds(clean_data$HEALTH, k = 10, returnTrain = T)

tic()
egb_mod <- train(
    HEALTH ~ . ^3, 
    clean_data,
    method = "xgbLinear",
    preProcess = c("knnImpute","zv", "center", "scale"),
    na.action = na.pass,
    trControl = trainControl(method = "cv", number = 10, index = index, verboseIter = T),
    tuneLength = 2
)
time <- toc()
exec_time_np <- unname(time$toc - time$tic)



local_cluster <- makeCluster(detectCores()-1)
registerDoParallel(local_cluster)

tic()
egb_mod_p <- train(
    HEALTH ~ . ^3, 
    clean_data,
    method = "xgbLinear",
    preProcess = c("knnImpute","zv", "center", "scale"),
    na.action = na.pass,
    trControl = trainControl(method = "cv", number = 10, index = index, verboseIter = T),
    tuneLength = 2
)
time_p <- toc()
exec_time_p <- unname(time_p$toc - time_p$tic)

stopCluster(local_cluster)
registerDoSEQ()

difftime <- exec_time_np - exec_time_p

# The non-parallelized execution time (using 1 processor) was 164.53 seconds. 
# The parallelized execution time (using 7 processors) was 52.2 seconds. 
# The difference was 112.33 seconds. 





