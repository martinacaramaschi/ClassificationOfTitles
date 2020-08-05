# -*- coding: utf-8 -*-

'''
print_ciao                           # 1. 
to_lower                             # 2.
to_read_dataset_and_create_df        # 3.
to_clean_str                         # 4.
to_tokenize_str                      # 5.
to_remove_sw_and_punct_from_list     # 6.
to_join_list                         # 7.
to_remove_sw_and_punct_from_sent     # 8.
to_lemmatize_word                    # 9.
to_lemmatize_sent                    # 10.

'''
from functions import print_ciao, to_lower, to_read_dataset_and_create_df, \
                      to_clean_str, to_tokenize_str, to_remove_sw_and_punct_from_list, \
                      to_join_list, to_remove_sw_and_punct_from_sent, to_lemmatize_word, \
                      to_lemmatize_sent, list_words
from preprocessing import preprocessing_titles

# Testing print_ciao() function
print_ciao()

# Testing to_lower(string) function
to_lower("hELLO World!")

# Testing to_read_dataset_and_create_df() function
name_dataset = 'database_TitlesAndClusters.xlsx'
dataframe = to_read_dataset_and_create_df(name_dataset)

dataframe_columns = dataframe.columns
print(dataframe_columns)
print(dataframe_columns[0])

titles = dataframe[dataframe_columns[0]]
  
dataframe_n_rows = dataframe.shape[0]
dataframe_n_columns = dataframe.shape[1]
print(dataframe_n_rows, dataframe_n_columns)

# Testing clean_str(string) function
to_clean_str("i don't know. maybe i'll do?")

# Testing to_tokenize_str(string) function
print(to_tokenize_str("i love u. and for ever!"))

# Testing to_remove_sw_and_punct_from_list(list) function
tokens_without_sw = to_remove_sw_and_punct_from_list(to_tokenize_str("i love u and for ever!"))

# Testing to_join_list(list) function
print(to_join_list(tokens_without_sw))

''' function works, but is not used
# Testing to_remove_stopw_and_punct(titles) function
print(to_remove_stopw_and_punct(titles))
'''

# New test
# Testing to_remove_stopw_and_punct_from_sent(sentence) function
print(to_remove_sw_and_punct_from_sent("I'll be a great physicist! Always, thank you!"))

# Testing to_lemmatize_word function
print(to_lemmatize_word("friends"))
print(to_lemmatize_word("playing"))

sent_test = ["hello worlds", "today i'll playing football with my best friends"]

''' function works, but is not used
# Testing to_lemmatize_list(titles) function
print(to_remove_stopw_and_punct(sent_test))
print(to_lemmatize(sent_test))
'''

# Testing to_lemmatize_sent(sentence) function
sent_test2 = "today i'll playing football with my best friends"
print(to_lemmatize_sent(sent_test2))

''' OLD TEST
# Testing preprocessing_titles(titles) function
lower_and_cleaned_titles = []
lower_and_cleaned_titles = preprocessing_titles(titles)
len(lower_and_cleaned_titles)
print(lower_and_cleaned_titles)

# Testing preprocessing_titles(titles) function
preprocessed_titles = []
preprocessed_titles = preprocessing_titles(titles)
#len(lower_and_cleaned_and_without_sw_punct_titles)
print(preprocessed_titles)
'''
# Testing preprocessing_titles(titles) function
preprocessed_titles = []
print(preprocessing_titles(sent_test))
print(preprocessing_titles(titles))

# Testing list_words(sentences) function
print(list_words(sent_test))

#test project
input_titles = []
input_titles = preprocessing_titles(titles)
print(list_words(input_titles))