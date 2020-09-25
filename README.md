# A method to classify titles of scientific articles
This repository illustrates the project i have created during the Pattern Recognition course of Master in Physics at University of Bologna.

## Goal of the project and why of this repository
The goal of this project is to write a program code able to classify scientific article's titles. The idea of this project was born when the organizer of a scientific conference asked me if there was a way to automatically classify the titles of papers, which were sent to him by those who proposed their paper for the conference.
I decided to create this repository to illustrate to evrybody how it is possible to create an authomatic classifier. This approach can be applied potentially to all types of text's classifications, so i hope many of you who are reading can find it useful.

## What a classifier is and what we need to know from the beginning
In this project I have created a statements' classifier, where statements are article's titles. The classifier works in this way: having an unknown title as input, the program code returns the corresponding class, as output.

![fig. how classifier works](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/01fig.statement_code_class.png)

The most important thing to know to create an authomatic classifier is that is needed to teach to the classifier how to do the classifications. That means we must provide a large set of examples of right classifications to the classifier and train it. This is the main thing to do. In my case, for the training, I have used a list of titles, of which i alreay knew the classification. Moreover, these examples are useful to understand which all possible classes are. 

## Step to create a classifier
### 0. Importing a dataset containg titles and their corresponding class... already known.
the first step is to import the set of right classification's examples, useful to train and test the classifier. 
### 1. Pre-processing of titles
Our classifier will classify a title according to the words contained within it. For this reason we need to remove from titles all the unuseful (to the semantic meaning) words and punctuation symbols. Moreover the same verb declined in two different verbal forms need to be uniformed to infinitive form; or plural and singular nouns need to be uniformed to singular form. The title's pre-processing step is necessary to do all this operations on titles.

![fig. preprocessing steps](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/05fig.preprocessing.png)

An example of pre-processing operations on a title is shown. First pre-processing step is to “clean” the sentences, removing all the stop words, as “the” or “than”, and removing the punctuations symbols; the second turns all the sentences into lowercase. This step is important because two words, written with the same characters, but with different cases, are considered different words (e.g. “science”, “SCIENCE”, “Science”). To avoid this case, all words are transformed into lowercase; last step is to lemmatize all the words (turning plural nouns to singular or turning all verbs to simple present form). 
### 2. Transformation of titles to readable titles by the machine
the classifier is not able to classify sentences directly. It can manage with numbers, not words. For this reason, before training the classifier, we need to transform all titles (that will be the input training) into "readable" input by the program. Readable means made of numbers. first, we will transform all title's words into "numeric objects", and then all titles will be represented by the set of his words.
To transform the words in “numeric objects”, we will use a word embedding technique. In the Natural Language Processing (NLP) field, we speak about word embedding when we refer to the set of language modeling and features learning techniques, able to map words and phrases to vectors of real numbers.  In 2013 a team of researchers led by Tomas Mikolov created [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf%C3%AC%E2%80%94%20%C3%AC%E2%80%9E%C5%93), a group of models used to produce word embedding. What I have called so far “numerical object”, it is better called vectors of real numbers. In fact, in NLP a word is well represented not by a single number, but by a vector of numbers, of which dimension is decided as applicable. In this step we will tranform all title's words into vectors of real numbers. Word2Vec models are able to define the number that best represent a word, based on which other words are most often close to the given word. The Word2Vec models I will use is called Continuous Bag of Word (CBOW). 

![fig. Word2Vec model](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/02fig.Word2Vec_CBOW.JPG)

In figure it is schematised how this model works. CBOW is a neural network that consists of input, projection, and output layers. It predicts the target word w(t) from the context {…, w(t-2), w(t-1), w(t+1), w(t+2), …}. Given a corpus (for simplicity assume the corpus is only one sentence: “The beautiful cat is near the table”), the vocabulary is defined as the list of words contained in the corpus, without repetitions. The size of the vocabulary is V. Initially, each word of the vocabulary is coded as one hot vector (a vector with one bit “1”, and all rest “0”, length of vector = V). Then, a window size is selected for iterating over the sentences (for example, size window is 3). To each step, CBOW tries to predict the central word in the window (in the first step of our example is “beautiful”) given the context (“the” and “cat”), using a neural network. What happen is that the input word vectors are projected on the projection layer, that is shared for all the words. So, all vectors get projected in the same position. To each step, the neural network weights are updated, calculating the error between the target vector and the predicted vector.

When all words have been transormed into vectors of real numbers, we represent each title as the avarege of his words. At the end, the titles are no more texts but vectors of numbers, and we can use them to train our classifier.

### 3. Training of the classifier
Our classifier is a neural network called Multilayer Perceptron (MLP). It consists of an input, a hidden layer and an output. It utilises the supervised technique called backpropagation for training. Using as input the titles, represented as vector of real numbers, and their corresponding class, already known, we will train the classifier.
To chech if the classifier performs good classifications we also test it. For this reason a part of right classification's examples are used to train the classifier and the rest to test it.

### 4. Classification
The classifier is ready to classify a new title.

## Structure of the project
These are the steps in order to start the program and to save the results:

1. The user has to provide the inital necessary informations, filling the [configuration.txt](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/configuration.txt) file: to write the name of the excel file, that contains the examples of right classifications, next to "excel_database = " (in my case, excel_database = database_TitlesAndClusters.xlsx). The dataset must contain only two columns, one with the titles and the others with the corresponding classes, and must be saved in the same folder of the project; to write the number of titles' column of the dataset next "column_of_titles = " (in my case, column_of_titles = 0); to write the number of classes' column of the dataset next "column_of_classes = " (in my case, column_of_classes = 1). Be careful: the program works only if the two column are the first and the second of the excel file and if the firts column is indicated with the number 0, and the second with the number 1; to write the name of the text file, in which the final results will be saved, next to "save_in = " (in my case, save_in = risultati1.txt); to write a sentence not already classified, that will be classified, next to "to_classify = " (in my case, to_classify = teaching scientific methods into schools).

2. The user can runs the entire project, from the step zero tot the end, using the command: "!python simulation.py configuration.txt".

3. At the end, the user can read the results of the classification in the file we indicated for the saving.

The results' file should appear like the following picture: there are the list of the possible classes availble, the training and testing scores, and the class of the new title is indicated.

![results.JPG](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/results.JPG)

This is how I divided my project into blocks:

* In the file [function.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/functions.py) I have developed all the functions I needed in the project (from those I used to import the dataset, to those to perform the preprocessing's phase, to those to transform the words into vectors of real numbers, to those used to train and test the classifier).

* In the file [testing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/testing.py) I have tested several functions to ensure that they work properly, using positive testing and sometimes hypothesis testing.

* In the file [configuration.txt](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/configuration.txt) there are indicated all the initial settings, used in the simulation file.

* In the file [simulation.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/simulation.py) there is the code the used need to import the dataset, containing all classification's examples; to preprocess the titles; to tranform all titles into element readble by the classifier that are vectos of real numbers; to train and test the classifier; to perform the classification of the unknown title; to save all results into a text file. 

* In file [database_TitlesAndClusters.xlsx](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/database_TitlesAndClusters.xlsx) ther is the dataset i have used to perfom my own simulation.


