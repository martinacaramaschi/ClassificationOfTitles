# -*- coding: utf-8 -*-

from functions import print_ciao, to_lower, read_dataset_and_create_df

#test print_ciao() function
print_ciao()

#test to_lower() function
to_lower("hELLO World!")

#test read_dataset_and_create_df() function
name_dataset = 'database_TitlesAndClusters.xlsx'
dataframe = read_dataset_and_create_df(name_dataset)

dataframe_columns = dataframe.columns
print(dataframe_columns)

dataframe_shape = dataframe.shape[0]
dataframe_n_columns = dataframe.shape[1]
print(dataframe_shape, dataframe_n_columns)