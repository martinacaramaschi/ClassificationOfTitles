# ClassificationOfTitles
This repository contains the project i have created during the Pattern Recognition course of Master in Physics at University of Bologna.

# The project

## The goal
The goal of this project is to write a program code able to classify scientific article's titles. Having a list of titles and a group of possible classes, which the statements belong to, the program have to read each title and assign a class to that title.

## The project's steps
To create a title's classifier, there are different step to follow. The last is the real classification of a unknown title: giving an unknown title as input to the program code, it gives as output the best corresponding class.
![01fig.statement_code_class](01fig.statement_code_class.png)
To reach this final step, some important steps precede that:
### 0. Importing a dataset containg titles and their corresponding class... already known.
This is the first step, necessary to start. In machine learning when a new classifier has to be created, it is necessary to have a set of right example, useful to train the machine to do a good classification. 
### 1. Preprocessing of titles
![05fig.preprocessing](05fig.preprocessing)
### 2. Transformation of titles to readable titles by the machine
(word2vec, vectors of real numbers)
### 3. Training of the classifier


# The project files
1. [function.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/functions.py) contains all the definition of functions.
2. [call_functions.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/call_functions.py) tests and calls the functions.
3. [preprocessing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/preprocessing.py) is the function the execute the preprocessing on titles, before using them to train Word2Vec models.
