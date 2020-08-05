# -*- coding: utf-8 -*-

from functions import to_lower , to_clean_str, to_remove_stopw_and_punct

# preprocessing_titles function: it preprocess the titles before the word2vec training
# As input the list of titles, the lenght of the list are needed
def preprocessing_titles(titles):
    # Step to lowercase all titles
    titles_lowercase = []
    for line in titles:
        titles_lowercase.append(to_lower(line))
    # Step to clean all titles
    titles_cleaned = []
    for line in titles_lowercase:
        titles_cleaned.append(to_clean_str(line))
    # Step to remove stop words and punctuation symbols
    titles_without_sw_and_punct = []
    for line in titles_cleaned:
        titles_without_sw_and_punct.append(to_remove_stopw_and_punct(line))
        return(titles_cleaned)
