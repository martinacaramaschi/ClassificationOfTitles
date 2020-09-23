# -*- coding: utf-8 -*-
"""
This file called call_functions is the prototipe of what a user should do, step
by step, to train his/her classifier, having a starting dataset, containing 
titles and corresponding class which the title belong to.
"""
# FIRST we import all necessary fuctions
#from importing_dataset import to_import_dataset
#from preprocessing import preprocessing_titles
from functions import to_import_dataset, preprocessing_titles, from_word_to_vec, \
                      to_average_all_titles, to_create_classes_vector, \
                      divide_train_test, classifier
                      

# SECOND we import the titles and corresponding classes
# we saved them into my_titles and my_classes, two ordered vectors
my_titles, my_classes = to_import_dataset("database_TitlesAndClusters.xlsx", 0, 1)
#print(my_titles)
#print(my_classes)

# THIRD we preprocess the titles
#the output is a vector of tokens, cleaned and uniformed, per each titles
preprocessed_titles = []
preprocessed_titles = preprocessing_titles(my_titles) 
len_titles = len(preprocessed_titles)

# FOURTH we create the vectors of real numbers, that represent the words
# vocabulary contains all words
# model contains all vector of real numbers per each word
model, vocabulary = from_word_to_vec(preprocessed_titles)
#vocabulary
#model['lens']

#FIFTH we represents all title as the average of its words
average_sentence = to_average_all_titles(preprocessed_titles, len_titles, model)

""" forse non mi servono più
# definisco le variabili delle dimensioni
# N = average_sentence.shape[0]
# d = average_sentence.shape[1]
"""
# SIXTH we assign to each class an integer, that represents that class during the classification
vector_of_classes, possible_classes = to_create_classes_vector(my_classes)

# SEVENTH we organize the data for the training and testing of classfier
train_x, train_y, test_x, test_y = divide_train_test(average_sentence, vector_of_classes)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

#classificatore
score_train, score_test = classifier(train_x, train_y, test_x, test_y)
score_train
score_test

"""
from sklearn.neural_network import MLPClassifier
model_MLP = MLPClassifier(hidden_layer_sizes=(200, 100))
reg = model_MLP.fit(train_x, train_y)

print(reg.score(train_x, train_y))
print(reg.score(test_x, test_y))
print(print(reg.predict_proba(test_x[:10])))

print(reg.predict(test_x[:8]))
"""
