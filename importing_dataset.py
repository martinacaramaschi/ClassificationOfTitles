# -*- coding: utf-8 -*-
import pandas as pd

""" This function import a two-columns dataset from an excel file called name_excel_dataset
    column_titles is the number of titles' column (must be 0 or 1)
    column_classes is the number of classes' column (must be 0 or 1)
    It returns the titles and classes to use for classifier training and testing """
    
def to_import_dataset(name_excel_dataset, 
                      number_of_titles_column, 
                      number_of_classes_column
                      ):
    try:
        dataset = pd.read_excel(name_excel_dataset, )
        dataset = pd.read_excel(name_excel_dataset)
    except: 
        raise ValueError('excel dataset {} not found or uncorrect! please check configuration'
                         .format(name_excel_dataset))
    dataframe = dataset.copy()
    if number_of_titles_column < 0 or number_of_titles_column > 2:
        raise ValueError('number of titles column must be 0 if it is the first \
                         column, 1 if it is the second')
    if number_of_classes_column < 0 or number_of_classes_column > 2:
        raise ValueError('number of classes column must be 0 if it is the first \
                         column, 1 if it is the second')
    titles = dataframe.columns[number_of_titles_column]
    classes = dataframe.columns[number_of_classes_column]
    return(titles, classes)
