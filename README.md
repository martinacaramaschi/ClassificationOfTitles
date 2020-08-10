# ClassificationOfTitles
This repository contains the project i have created during the Pattern Recognition course of Master in Physics at University of Bologna.

## The goal of the project
The goal of this project is to write a program code able to classify scientific article's titles. Having a list of titles and a group of possible classes, which the statements belong to, the program have to read each title and assign a class to that title.

## The project's steps
In this project we aim to create a statement's classifier, where statements are article's titles, that works in this way: giving an unknown title as input, the program code gives  the corresponding class, as output.

![fig. how classifier works](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/01fig.statement_code_class.png)

To create a classifier like this, several steps have to be followed:
### 0. Importing a dataset containg titles and their corresponding class... already known.
This is the first step, necessary to start. In machine learning when a new classifier want to be created, it is necessary to have a set of right example, used to train the machine to do a good classification. 
### 1. Preprocessing of titles
![fig. preprocessing steps](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/05fig.preprocessing.png)
### 2. Transformation of titles to readable titles by the machine
(word2vec, vectors of real numbers)
### 3. Training of the classifier


# The project files
1. [function.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/functions.py) contains all the definition of functions.
2. [call_functions.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/call_functions.py) tests and calls the functions.
3. [preprocessing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/preprocessing.py) is the function the execute the preprocessing on titles, before using them to train Word2Vec models.
