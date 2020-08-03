# -*- coding: utf-8 -*-

# I define a simple function in order to understand how pyhton files works
def print_ciao():
    print("ciao!")

# read_dataset_and_create_df function: it read an excel dataset and returns a dataframe containing the dataset's informations
# As input the dataset's name (e.g. name_dataset = "name_dataset.xlsx") is needed
def read_dataset_and_create_df(name_dataset):
    import pandas as pd
    # It reads and stores the dataset's information in dataset
    dataset = pd.read_excel(name_dataset, )
    dataset = pd.read_excel(name_dataset)
    # It creates the dataframe containing the dataset's infos
    dataframe = dataset.copy() 
    print("okay!")    # It prints "okay!" to check all has gone well
    return(dataframe)

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
        #print(lower_case)
    except:
        print("Not valid input! must be a string!")
    return(lower_case)
    
    
# clean_str function: it removes or substitutes some string (e.g. 'll is removed and n't become not)
# As input a string is needed
def clean_str(string):
    import re
    string = re.sub(r"'s", " ", string)
    string = re.sub(r"'ve", " have", string)
    string = re.sub(r"n't", " not", string)
    string = re.sub(r"'re", " are", string)
    string = re.sub(r"'d", " ", string)
    string = re.sub(r"'ll", " ", string)
    return(string.strip().lower())

