# -*- coding: utf-8 -*-

# I define a simple function in order to understand how pyhton files works
def print_ciao():
    print("ciao!")


# to_lower function: it transforms all input string to lower case
# As input, a string is needed 
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

# preprocessing_titles function: it preprocess the titles before the word2vec training
# As input the list of titles, the lenght of the list are needed
def preprocessing_titles(titles, number_of_rows):
    titles_lowercase = []
    for i in range(0,number_of_rows):
        titles_lowercase.append(to_lower(titles[i]))
    titles = titles_lowercase
    return(titles)


# read_dataset_and_create_df function: it read an excel dataset and returns a dataframe containing the dataset's informations
# As input the dataset's name (e.g. name_dataset = "name_dataset.xlsx") is needed
def read_dataset_and_create_df(name_dataset):
    import pandas as pd
    # It reads and stores the dataset's information in dataset
    dataset = pd.read_excel(name_dataset, )
    dataset = pd.read_excel(name_dataset)
    # It creates the dataframe containing the dataset's infos
    dataframe = dataset.copy()
    # It prints "okay!" to check all has gone well
    print("okay!")
    
    return(dataframe)