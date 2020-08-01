# -*- coding: utf-8 -*-

def print_ciao():
    print("ciao!")


#lowercase function: to_lower is the function that give me all the phrases written in uppercase
def to_lower(string):
    try:
        lower_case = ""
        for character in string:
            if 'A' <= character <= 'Z':
                location = ord(character) - ord('A')
                new_ascii = location + ord('a')
                character = chr(new_ascii)
            lower_case = lower_case + character
        print(lower_case)
    except:
        print("Not valid input! must be a string!")


def read_dataset_and_create_df(name_dataset):
    import pandas as pd

    #read the dataset. ET means ESERAtable. This table that contains titles of articles presented during ESERA confrence 2019 as oral presentations
    dataset = pd.read_excel(name_dataset, )
    dataset = pd.read_excel(name_dataset)

    # dfET means dataframeESERAtable and it's the dataframe that i will manage 
    dataframe = dataset.copy()

    #show the index of the dataframe's columns
    #print(dataframe.columns)
    print("okay!")
    return(dataframe)