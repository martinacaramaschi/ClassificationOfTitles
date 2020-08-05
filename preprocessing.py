# -*- coding: utf-8 -*-

from functions import to_lower , to_clean_str, to_remove_sw_and_punct_from_sent, \
                          to_lemmatize_sent

# preprocessing_titles function:
# As input, a list of sentences (in my case, titles) is needed.
# It lowercased all sentences; remove stop words and punctuation symbols from sentences;
       # it lemmatized all nouns and verbs.
# As output, the list of senteces after all preprocessing steps.
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
        titles_without_sw_and_punct.append(to_remove_sw_and_punct_from_sent(line))
    # Step to lemmatize verbs and nouns
    titles_lemmatized = []
    for line in titles_without_sw_and_punct:
        titles_lemmatized.append(to_lemmatize_sent(line))    
    return(titles_lemmatized)