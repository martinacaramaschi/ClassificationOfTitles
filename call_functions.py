# -*- coding: utf-8 -*-

from functions import print_ciao, to_lower, read_dataset_and_create_df, \
                         clean_str
from preprocessing import preprocessing_titles

# Testing print_ciao() function
print_ciao()

# Testing to_lower(string) function
to_lower("hELLO World!")

string = ["hello Wolrd!", "CARAINBIII"] 
for element in string:
    to_lower(element)

# Testing read_dataset_and_create_df() function
name_dataset = 'database_TitlesAndClusters.xlsx'
dataframe = read_dataset_and_create_df(name_dataset)

dataframe_columns = dataframe.columns
print(dataframe_columns)
print(dataframe_columns[0])

titles = dataframe[dataframe_columns[0]]
  
dataframe_n_rows = dataframe.shape[0]
dataframe_n_columns = dataframe.shape[1]
print(dataframe_n_rows, dataframe_n_columns)

# Testing clean_str(string) function
clean_str("i don't know. maybe i'll do.")

# Testing preprocessing_titles(titles, number_of_rows) function
lower_and_cleaned_titles = []
lower_and_cleaned_titles = preprocessing_titles(titles)
len(lower_and_cleaned_titles)
print(lower_and_cleaned_titles)