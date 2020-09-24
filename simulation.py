# -*- coding: utf-8 -*-
"""
This file, called simulation, is the prototipe of what the user should do,
step by step, to train and test his/her classifier, and classify a new title,
not yet classified.
All starting informations should be specified into configuration.txt:
    - name of excel database, containing titles and classes already known
    - number of column of titles
    - number of column of classes
    - name of text empty document in which save the results
    - title not already classified

The command to run the simulation is: !python simulation.py configuration.txt
"""

# FIRST importing necessary fuctions and variables
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
new_title_input= config.get('settings', 'to_classify')

col_titles = int(col_titles)
col_classes = int(col_classes)

new_title = []
new_title.append(new_title_input)

# SECOND reading the dataset containg titles and correspondig classes
my_titles, my_classes = f.to_import_dataset(name_dataset, col_titles, col_classes)

# THIRD preprocessing titles
# the output is a vector of tokens, cleaned and uniformed, per each titles
preprocessed_titles = []
preprocessed_titles = f.preprocessing_titles(my_titles) 
len_titles = len(preprocessed_titles)

# FOURTH creating the vectors of real numbers, that represent the words
# vocabulary contains all words
# model contains all vector of real numbers per each word
model, vocabulary = f.from_word_to_vec(preprocessed_titles)

#FIFTH calculating the average of each title
average_sentence = f.to_average_all_titles(preprocessed_titles, len_titles, model)

# SIXTH assigning to each class an integer, that represents that class during the classification
vector_of_classes, possible_classes = f.to_create_classes_vector(my_classes)

# SEVENTH organising data into training and testing data
train_x, train_y, test_x, test_y = f.divide_train_test(average_sentence, vector_of_classes)

# EIGHT transforming the unclassified title for the process of classification
new_input = f.preprocessing_titles(new_title)
new_vector = f.prepare_new_title(model, new_input)

# NINTH training and testing the classifier
# as output we obtain the score of training and testing 
# and the classification of the "unknown" title
score_train, score_test, class_new_sent = f.classifier(train_x, train_y, test_x, test_y, new_vector)
class_new_sent = int(class_new_sent[0])
class_new_sent = possible_classes[class_new_sent]

# TENTH saving results
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
file_results.write("\n")
file_results.write("class of new sentence: ")
file_results.write(str(class_new_sent))
file_results.close() 

