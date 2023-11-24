# Prelim Question 3

# Build and train a neural network model. Experiment with architectures and training configurations until you find a model that performs at least as well as principal component logistic regression from task 1. Quantify the predictive accuracy.

library(keras)
install_keras()


# partition
set.seed(102722)
partitions <- claims_clean %>%
  mutate(text_clean = str_trim(text_clean)) %>%
  filter(str_length(text_clean) > 5) %>%
  initial_split(prop = 0.7)

test_dtm<-testing(partitions)%>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()

train_dtm <- training(partitions) %>%
  unnest_tokens(output = 'token', 
                input = text_clean) %>%
  group_by(.id, bclass) %>%
  count(token) %>%
  bind_tf_idf(term = token, 
              document = .id, 
              n = n) %>%
  pivot_wider(id_cols = c(.id, bclass), 
              names_from = token, 
              values_from = tf_idf,
              values_fill = 0) %>%
  ungroup()


# store full DTM as a matrix
x_train <- train_dtm1 %>%
  select(-bclass, -.id) %>%
  as.matrix()

# extract labels and coerce to binary
y_train <- train_dtm1 %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1

# store full DTM as a matrix, testing
x_test <- test_dtm %>%
  select(-bclass, -.id) %>%
  as.matrix()

# extract labels for testing data
y_test <- test_dtm %>% 
  pull(bclass) %>%
  factor() %>%
  as.numeric() - 1


model <- keras_model_sequential(input_shape = ncol(x_train)) %>%
  layer_dense(10) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
model %>%
  compile(
    loss = 'binary_crossentropy',
    optimizer = "adam",
    metrics = 'binary_accuracy'
  )

history <- model %>%
  fit(x = x_train,
      y = y_train,
      epochs = 50,
      validation_split = 0.2)

plot(history)

# evaluate on specified data
evaluate(model, x_train, y_train)

# The validation set can be used to provide a soft estimate of accuracy during training.
# The code chunk below trains for longer and uses 20% of the training data for validation. You should see that the training accuracy gets quite high, but the validation accuracy plateaus around 80%.
