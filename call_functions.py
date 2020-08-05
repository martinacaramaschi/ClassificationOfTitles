# -*- coding: utf-8 -*-

'''
print_ciao
to_lower
to_read_dataset_and_create_df
to_clean_str
to_tokenize_str
to_remove_sw_and_punct_from_list
to_join_list
to_remove_stopw_and_punct
to_lemmatize
to_lemmatize_word

'''
from functions import print_ciao, to_lower, to_read_dataset_and_create_df, \
                         to_clean_str, to_tokenize_str, to_remove_sw_and_punct_from_list, \
                         to_join_list, to_remove_stopw_and_punct, to_lemmatize, \
                         to_lemmatize_word
from preprocessing import preprocessing_titles

# Testing print_ciao() function
print_ciao()

# Testing to_lower(string) function
to_lower("hELLO World!")

string = ["hello Wolrd!", "CARAINBIII"] 
for element in string:
    to_lower(element)

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

# Testing to_remove_stopw_and_punct(titles) function
print(to_remove_stopw_and_punct(titles))

# Testing preprocessing_titles(titles) function
lower_and_cleaned_titles = []
lower_and_cleaned_titles = preprocessing_titles(titles)
len(lower_and_cleaned_titles)
print(lower_and_cleaned_titles)

# Testing preprocessing_titles(titles) function
lower_and_cleaned_and_without_sw_punct_titles = []
lower_and_cleaned_and_without_sw_punct_titles = preprocessing_titles(titles)
#len(lower_and_cleaned_and_without_sw_punct_titles)
print(lower_and_cleaned_and_without_sw_punct_titles)

# Testing to_lemmatize_word function
print(to_lemmatize_word("friends"))
print(to_lemmatize_word("playing"))

# Testing to_lemmatize_list(titles) function
sent_test = ["hello worlds", "today i'm playing footbal with my best friends"]
print(to_lemmatize(sent_test))
# Second test
print(to_lemmatize(lower_and_cleaned_and_without_sw_punct_titles))
