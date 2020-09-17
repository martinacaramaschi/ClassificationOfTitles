# -*- coding: utf-8 -*-

''' 
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
# from functions import list_words
from importing_dataset import to_import_dataset
from preprocessing import preprocessing_titles
from functions import to_list_all_words_with_repetition, to_list_all_words_no_repetition, \
                      from_word_to_vec

# importing titles and corresponding classes
my_titles, my_classes = to_import_dataset("database_TitlesAndClusters.xlsx", 0, 1)
print(my_titles)
print(my_classes)

# Testing preprocessing_titles(titles) function: print the results
preprocessed_titles = []
preprocessed_titles = preprocessing_titles(my_titles) 

# from word to vec
model, vocabulary = from_word_to_vec(preprocessed_titles)
vocabulary
model['lens']

# Testing all_words_with_repetition(sentences) function
words_with_repetition = to_list_all_words_with_repetition(preprocessed_titles)
words_without_repetition = to_list_all_words_no_repetition(words_with_repetition)
print(words_without_repetition)
len(words_without_repetition)



"""
preprocessed titles è l'input per word2vec
"""

#test project
# input_titles = []
# input_titles = preprocessing_titles(titles)
# print(list_words(input_titles))




#----------------------------------------------------------------------------#
"""
# Testing print_ciao() function
print_ciao()

# Testing to_lower(string) function
to_lower("hELLO World!")

# Testing clean_str(string) function
to_clean_str("i don't know. maybe i'll do?")

# Testing to_tokenize_str(string) function
print(to_tokenize_str("i love u. and for ever!"))

# Testing to_remove_sw_and_punct_from_list(list) function
tokens_without_sw = to_remove_sw_and_punct_from_list(to_tokenize_str("i love u and for ever!"))

# Testing to_join_list(list) function
print(to_join_list(tokens_without_sw))

# New test
# Testing to_remove_stopw_and_punct_from_sent(sentence) function
print(to_remove_sw_and_punct_from_sent("I'll be a great physicist! Always, thank you!"))

# Testing to_lemmatize_word function
print(to_lemmatize_word("friends"))
print(to_lemmatize_word("playing"))

function works, but is not used
# Testing to_remove_stopw_and_punct(titles) function
print(to_remove_stopw_and_punct(titles))

# Testing to_lemmatize_sent(sentence) function
sent_test2 = "today i'll playing football with my best friends"
print(to_lemmatize_sent(sent_test2))

#OLD TEST
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
"""