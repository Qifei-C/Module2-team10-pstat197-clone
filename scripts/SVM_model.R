library(tidyverse)
library(tidymodels)
library(e1071)
library(caret)
library(rsample)
library(kernlab)
library(keras)

load("data/claims-clean-example.RData")
head(claims_clean)
source('scripts/preprocessing.R')
claims_tfidf <- nlp_fn(claims_clean)

# partition data
set.seed(102722)
partitions <- claims_tfidf %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass) %>%
  as.matrix()
test_labels <- testing(partitions)$bclass

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass) %>%
  as.matrix()
train_labels <- training(partitions)$bclass


svm_model <- svm(train_dtm, train_labels, kernel = "radial")
svm_model2 <- svm(train_dtm, train_labels, kernel = "linear")

# predicted value for testing part
p <- predict(svm_model, test_dtm)
mean(p == test_labels) # 0.5357143 accuracy

p2 <- predict(svm_model2, test_dtm)
mean(p2 == test_labels) # 0.8071429 accuracy

train_ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

svm_model_cv <- train(
  train_dtm, 
  train_labels,
  method = "svmRadial",  # Use radial kernel for SVM
  trControl = train_ctrl
)

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

