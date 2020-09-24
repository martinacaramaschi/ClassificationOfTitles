# -*- coding: utf-8 -*-
"""
This file called simulation is the prototipe of what a user should do, step
by step, to train and test his/her classifier, having a starting dataset, containing 
titles and corresponding class which the title belong to.
"""
# per eseguire digito:
# !python simulation.py configuration.txt

# FIRST we import all necessary fuctions
import functions as f       
import configparser    
import sys
from sys import argv

config = configparser.ConfigParser()
config.read(sys.argv[1])       

name_dataset = config.get('settings', 'excel_database')
col_titles = config.get('settings', 'column_of_titles')
col_classes = config.get('settings', 'column_of_classes')
save_as = config.get('settings', 'save_in')

col_titles = int(col_titles)
col_classes = int(col_classes)
# SECOND we import the titles and corresponding classes
# we saved them into my_titles and my_classes, two ordered vectors
my_titles, my_classes = f.to_import_dataset(name_dataset, col_titles, col_classes)

# THIRD we preprocess the titles
# the output is a vector of tokens, cleaned and uniformed, per each titles
preprocessed_titles = []
preprocessed_titles = f.preprocessing_titles(my_titles) 
len_titles = len(preprocessed_titles)

# FOURTH we create the vectors of real numbers, that represent the words
# vocabulary contains all words
# model contains all vector of real numbers per each word
model, vocabulary = f.from_word_to_vec(preprocessed_titles)

#FIFTH we represents all title as the average of its words
average_sentence = f.to_average_all_titles(preprocessed_titles, len_titles, model)

# SIXTH we assign to each class an integer, that represents that class during the classification
vector_of_classes, possible_classes = f.to_create_classes_vector(my_classes)

# SEVENTH we organize the data for the training and testing of classfier
train_x, train_y, test_x, test_y = f.divide_train_test(average_sentence, vector_of_classes)
# print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# EIGTH classificatore
score_train, score_test = f.classifier(train_x, train_y, test_x, test_y)

# NINTH see performance
file_results = open(save_as, "w")
file_results.write("the classifier has these possible classes:\n")
for element in possible_classes:
    file_results.write(element)
    file_results.write(", ")
file_results.write("\n")
file_results.write("the score of training is: ")
file_results.write(str(score_train))
file_results.write("\nthe score of testing is: ")
file_results.write(str(score_test))
file_results.close() 

