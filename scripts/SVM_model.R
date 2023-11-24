library(tidyverse)
library(tidymodels)
library(e1071)
library(caret)
library(rsample)
library(kernlab)
library(keras)

load("data/claims-clean-example.RData")
head(claims_clean)

# For Binary Class
source('scripts/preprocessing.R')
claims_tfidf <- nlp_fn(claims_clean)

## partition data
set.seed(102722)
partitions <- claims_tfidf %>% initial_split(prop = 0.8)

## separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass) %>%
  as.matrix()
test_labels <- testing(partitions)$bclass

## same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass) %>%
  as.matrix()
train_labels <- training(partitions)$bclass

## Build Models (SVM-radial, SVM-linear, SVM-CV)
svm_model <- svm(train_dtm, train_labels, kernel = "radial")
svm_model2 <- svm(train_dtm, train_labels, kernel = "linear")

train_ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

svm_model_cv <- train(
  train_dtm, 
  train_labels,
  method = "svmRadial",  # Use radial kernel for SVM
  trControl = train_ctrl
)

## predicted value for testing part
p <- predict(svm_model, test_dtm)
mean(p == test_labels) # 0.5357143 accuracy

p2 <- predict(svm_model2, test_dtm)
mean(p2 == test_labels) # 0.8071429 accuracy


print(svm_model_cv)

# Support Vector Machines with Radial Basis Function Kernel 
# 
# 1679 samples
# 36541 predictors
# 2 classes: 'N/A: No relevant content.', 'Relevant claim content' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 1343, 1344, 1343, 1343, 1343 
# Resampling results across tuning parameters:
#   
#   C     Accuracy   Kappa    
# 0.25  0.6711958  0.3299008
# 0.50  0.7724520  0.5437967
# 1.00  0.8046393  0.6098036
# 
# Tuning parameter 'sigma' was held constant at a value of 1.810415
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 1.810415 and C = 1.

p_cv <- predict(svm_model_cv, test_dtm)
mean(p_cv == test_labels) # 0.7833333 accuracy


saveRDS(svm_model, "results/svm-model.rda")
saveRDS(svm_model2, "results/svm-model_2.rda")
saveRDS(svm_model_cv, "results/svm-model-cv.rda")

# Tried three models for binary class:
# SVM with Linear Kernal has highest accuracy of 0.807
# SVM with 5 folds cross validation has accuracy of 0.783
# SVM with Radial Kernal has accuracy of 0.536







##########################################


# For Mclass
claims_tfidf_m <- nlp_fn_2(claims_clean)

## partition data
set.seed(102722)
partitions_m <- claims_tfidf_m %>% initial_split(prop = 0.8)


## separate DTM from labels
test_dtm_m <- testing(partitions_m) %>%
  select(-.id, -mclass) %>%
  as.matrix()
test_labels_m <- testing(partitions_m)$mclass

## same, training set
train_dtm_m <- training(partitions_m) %>%
  select(-.id, -mclass) %>%
  as.matrix()
train_labels_m <- training(partitions_m)$mclass

## Build Models (SVM-radial, SVM-linear, SVM-CV)
svm_m_radial <- svm(train_dtm_m, train_labels_m, kernel = "radial")
svm_m_linear <- svm(train_dtm_m, train_labels_m, kernel = "linear")

svm_m_cv <- train(
  train_dtm_m, 
  train_labels_m,
  method = "svmRadial",  # Use radial kernel for SVM
  trControl = train_ctrl
) #1.11 begin

## predicted value for testing part
p_m_radial <- predict(svm_m_radial, test_dtm_m)
mean(p_m_radial == test_labels_m)  # 0.4833 accuracy

p_m_linear <- predict(svm_m_linear, test_dtm_m)
mean(p_m_linear == test_labels_m)  # 0.7667 accuracy


print(svm_m_cv)
# Support Vector Machines with Radial Basis Function Kernel 
# 
# 1679 samples
# 36541 predictors
# 5 classes: 'N/A: No relevant content.', 'Physical Activity', 'Possible Fatality', 'Potentially unlawful activity', 'Other claim content' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 1344, 1343, 1342, 1343, 1344 
# Resampling results across tuning parameters:
#   
#   C     Accuracy   Kappa    
# 0.25  0.6069264  0.3348758
# 0.50  0.6819836  0.4860393
# 1.00  0.7338032  0.5825746
# 
# Tuning parameter 'sigma' was held constant at a value of 1.958185
# Accuracy was used to select the optimal model using the largest value.
# The final values used for the model were sigma = 1.958185 and C = 1.

p_m_cv <- predict(svm_m_cv, test_dtm_m)
mean(p_m_cv == test_labels_m) # 0.7595 accuracy


saveRDS(svm_m_radial, "results/svm_m_radial.rda")
saveRDS(svm_m_linear, "results/svm_m_linear.rda")
saveRDS(svm_m_cv, "results/svm_m_cv.rda")

# Tried three models for binary class:
# SVM with Linear Kernal has highest accuracy of 0.767
# SVM with 5 folds cross validation has accuracy of 0.760
# SVM with Radial Kernal has accuracy of 0.483
