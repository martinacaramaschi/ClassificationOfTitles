# -*- coding: utf-8 -*-

from functions import print_ciao, to_lower, read_dataset_and_create_df, \
                         preprocessing_titles

# Testing print_ciao() function
print_ciao()

# Testing to_lower(string) function
to_lower("hELLO World!")

# Testing read_dataset_and_create_df() function
name_dataset = 'database_TitlesAndClusters.xlsx'
dataframe = read_dataset_and_create_df(name_dataset)

dataframe_columns = dataframe.columns
print(dataframe_columns)
print(dataframe_columns[0])

titles = dataframe[dataframe_columns[0]]
print(titles)
dataframe_n_rows = dataframe.shape[0]
dataframe_n_columns = dataframe.shape[1]
print(dataframe_n_rows, dataframe_n_columns)

# Testing preprocessing_titles(titles, number_of_rows) function
lower_titles = preprocessing_titles(titles, dataframe_n_rows)
len(lower_titles)