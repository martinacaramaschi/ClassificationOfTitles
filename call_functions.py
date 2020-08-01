# -*- coding: utf-8 -*-

from functions import print_ciao, to_lower, read_dataset_and_create_df

print_ciao()

to_lower("hELLO World!")

name_dataset = 'database_TitlesAndClusters.xlsx'
dataframe = read_dataset_and_create_df(name_dataset)

df_columns= dataframe.columns
print(df_columns)