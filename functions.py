# -*- coding: utf-8 -*-

def print_ciao():
    print("ciao!")


#to_lower function: it transforms all input string to lower case
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

#read_dataset_and_create_df function: it read an excel dataset and 
    #returns a dataframe containing the dataset's informations

#the input needed is the dataset's name (e.g. name_dataset = "name_dataset.xlsx")
def read_dataset_and_create_df(name_dataset):
    import pandas as pd

    #we read and store the dataset's information in dataset
    dataset = pd.read_excel(name_dataset, )
    dataset = pd.read_excel(name_dataset)

    #than we create the dataframe containing the dataset's infos
    dataframe = dataset.copy()

    #print "okay!" to check all has gone well
    print("okay!")
    
    #the functions return the dataframe
    return(dataframe)