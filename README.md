# ClassificationOfTitles
This repository contains the project i have created during the Pattern Recognition course of Master in Physics at University of Bologna.

## The goal of the project
The goal of this project is to write a program code able to classify scientific article's titles. Having a list of titles and a group of possible classes, which the statements belong to, the program have to read each title and assign a class to that title.

## The project's steps
In this project we aim to create a statement's classifier, where statements are article's titles, that works in this way: giving an unknown title as input, the program code gives  the corresponding class, as output.

![fig. how classifier works](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/01fig.statement_code_class.png)

To create a classifier like this, several steps have to be followed:
### 0. Importing a dataset containg titles and their corresponding class... already known.
This is the first step, necessary to start. In machine learning when a new classifier want to be created, it is necessary to have a set of right classification's example, used to train the classifier. 
### 1. Pre-processing of titles
Our text classifier will classify a title on the basis of the words contained into. For this reason we need to remove from the titles all the unuseful (for semantic meaning) words and punctuation symbols. Moreover the same verb declined in two different verbal forms need to be uniformed to infinitive form; or plural and singular nouns need to be uniformed to singular form. The title's pre-processing step is necessary to do all this operations on the titles.

![fig. preprocessing steps](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/05fig.preprocessing.png)

The pre-processing step “clean” the sentences, removing all the stop words, as “the” or “than”, and removing the punctuations symbols; then turn all the sentences into lowercase. That is important because two words, written with the same characters, but with different cases, are considered different words (e.g. “science”, “SCIENCE”, “Science”). To avoid this case, all words are transformed into lowercase; then lemmatize all the words (turning plural nouns to singular or turning all verbs to simple present form). In figure it is reported an example of pre-processing made on a title.

### 2. Transformation of titles to readable titles by the machine
Any program code (as our classifier) can manage with numbers, not words. For this reason, before training the classifier, we need to transform all titles (that will be used tfor the training) in "readable" titles by the program. Readable means made of numbers. To transform the words in “numeric objects”, we will use a word embedding technique. In the Natural Language Processing (NLP) field, we speak about word embedding when we refer to the set of language modeling and features learning techniques, able to map words and phrases to vectors of real numbers.  In 2013 a team of researchers led by Tomas Mikolov created Word2Vec, a group of models used to produce word embedding. What I have called so far “numerical object”, it is better called vectors of real numbers. In fact, in NLP a word is well represented not by a single number, but by a vector of numbers, of which dimension is decided as applicable. In this step we will tranform all title's words into vectors of real numbers. Word2Vec models are able to define the numberb that best represent a word, based on which other words are most often close to the given word.

When all words have been transormed into vectors of real numbers, we represent each title as the avarege of its words. Now our titles are no more texts but vectors of numbers, and we can use them to train our classifier.

### 3. Training of the classifier


# The project files
1. [function.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/functions.py) contains all the definition of functions.
2. [call_functions.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/call_functions.py) tests and calls the functions.
3. [preprocessing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/preprocessing.py) is the function the execute the preprocessing on titles, before using them to train Word2Vec models.
