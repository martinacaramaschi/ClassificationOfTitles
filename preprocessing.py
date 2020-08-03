# -*- coding: utf-8 -*-

from functions import to_lower , clean_str

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
        titles_cleaned.append(clean_str(line))
    return(titles_cleaned)
