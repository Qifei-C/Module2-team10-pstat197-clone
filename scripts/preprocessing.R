## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
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

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

nlp_fn_2 <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# bigram function
bigram_fn1 <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = bigram, 
                  input = text_clean, 
                  token = 'ngrams',
                  n = 2,
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