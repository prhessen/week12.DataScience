# Libraries

# Removed any libraries not directly necessary, inlcuding rstudioapi
# Used dplyr and tidyr individually rather than as a part of tidyverse (saved
# about 0.5 seconds)

library(haven)
library(dplyr)
library(tidyr)
library(caret)
library(tictoc)
library(parallel)
library(doParallel)
library(xgboost)
library(RANN)

# Data Import and Cleaning

full_data <- read_sav("week12/GSS2006.sav")

# Elected to make no changes - this portion already runs in 140 milliseconds
clean_data <- full_data %>%
    select_at(vars(c(starts_with("BIG5"), "HEALTH"))) %>%
    drop_na("HEALTH") %>%
    filter(rowSums(is.na(.[,1:10])) != 10) %>%
    mutate_all("as.numeric") 


# Analysis

tic()
egb_mod <- train(
    HEALTH ~ . ^3, 
    clean_data,
    method = "xgbLinear",
    preProcess = c("knnImpute","zv", "center", "scale"),
    na.action = na.pass,
    trControl = trainControl(method = "cv", number = 10),
    tuneLength = 3
)
time <- toc()
exec_time_np <- unname(time$toc - time$tic)



local_cluster <- makeCluster(2)
registerDoParallel(local_cluster)

tic()
egb_mod_p <- train(
    HEALTH ~ . ^3, 
    clean_data,
    method = "xgbLinear",
    preProcess = c("knnImpute","zv", "center", "scale"),
    na.action = na.pass,
    trControl = trainControl(method = "cv", number = 10),
    tuneLength = 3
)
time_p <- toc()
exec_time_p <- unname(time_p$toc - time_p$tic)

stopCluster(local_cluster)
registerDoSEQ()

difftime <- exec_time_np - exec_time_p

time_df <- data.frame(exec_time_np = exec_time_np, exec_time_p = exec_time_p)
write.csv(time_df,"interactive.csv", row.names = F)

# The non-parallelized execution time (using 1 processor) was 442.709 seconds. 
# The parallelized execution time (using 7 processors) was 237.622 seconds. 
# The difference was 205.619 seconds. 
