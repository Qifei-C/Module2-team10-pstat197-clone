# Prelim Task 2

library(tidyverse)
library(tidytext)
library(tokenizers)
library(textstem)
library(stopwords)

# Bigrams


claims_bigrams<-claims_clean %>%
  unnest_tokens(ngram, text_tmp, token = "ngrams", n = 2) %>% # optional stopword removal
  mutate(token = lemmatize_words(ngram))

claims_bigrams %>%
  count(neo_search_subject_id, ngram, sort = TRUE)


bigram_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = bigram, 
                  input = text_clean, 
                  token = 'ngrams',
                  n=2,
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(bigram.lem = lemmatize_words(bigram)) %>%
    filter(str_length(bigram.lem) > 2) %>%
    count(.id, bclass, bigram.lem, name = 'n') %>%
    bind_tf_idf(term = bigram.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'bigram.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

claims_tfidf1 <- bigram_fn(claims_clean)


# partition data
set.seed(110122)
partitions1 <- claims_tfidf1 %>% initial_split(prop = 0.7)
# train/test split w/o headers
train <- training(partitions1) %>%
  select(-.id, -bclass)
train_labels <- training(partitions1) %>%
  select(.id, bclass)
test <- testing(partitions1) %>%
  select(-.id, -bclass)
test_labels <- testing(partitions1) %>%
  select(.id, bclass)

proj_out <- projection_fn(.dtm = train, .prop = 0.7)
train_projected <- proj_out$data

### w/o headers
train_claim <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_projected)
fit_claim <- glm(bclass ~ ., data = train_claim, family = "binomial") # warning: evidence of overfitting

# project test data onto PCs
test_projected <- reproject_fn(.dtm = test, proj_out)
# coerce to matrix
x_test <- as.matrix(test_projected)
# compute predicted probabilities
preds <- predict(fit_claim, 
                 test_projected,
                 type = 'response')

pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))
# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)
# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')