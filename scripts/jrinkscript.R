
# Preliminary Task 1

parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>% # adjustment to include header tags_Qifei
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}


# preprocess (will take a minute or two)
claims_clean1 <- claims_raw %>%
  parse_data()

library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# partition
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.7)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

