# A method to classify titles of scientific articles
This repository illustrates the project i have created during the Pattern Recognition course of Master in Physics at University of Bologna.

## Goal of the project and why of this repository
The goal of this project is to write a program code able to classify scientific article's titles. The idea of this project was born when the organizer of a scientific conference asked me if there was a way to automatically classify the titles of papers, which were sent to him by those applying for the conference. 
I decided to create this repository to illustrate to evrybody how it is possible to create an authomatic classifier. This approach can be applied potentially to all types of text's classifications, so i hope many of you who are reading can find it useful.

## What a classifier is and what we need to know from the beginning
In this project I have created a statements' classifier, where statements are article's titles. The classifier works in this way: having an unknown title as input, the program code returns the corresponding class, as output.

![fig. how classifier works](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/01fig.statement_code_class.png)

The most important thing to know to create an authomatic classifier is that is needed to teach to the classifier how to do the classifications. That means we must provide a large set of examples of right classifications to the classifier and train it. This is the main thing to do. In my case, for the training, I have used a list of titles, of which i alreay knew the classification. Moreover, these examples are useful to understand which all possible classes are. 

## Step to create a classifier
### 0. Importing a dataset containg titles and their corresponding class... already known.
This is the first step, necessary to start. In machine learning when a new classifier want to be created, it is necessary to have a set of right classification's example, used to train and test the classifier. 
### 1. Pre-processing of titles
Our classifier will classify a title according to the words contained within it. For this reason we need to remove from titles all the unuseful (to the semantic meaning) words and punctuation symbols. Moreover the same verb declined in two different verbal forms need to be uniformed to infinitive form; or plural and singular nouns need to be uniformed to singular form. The title's pre-processing step is necessary to do all this operations on titles.

![fig. preprocessing steps](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/05fig.preprocessing.png)

An example of pre-processing operations on a title is shown. First pre-processing step is to “clean” the sentences, removing all the stop words, as “the” or “than”, and removing the punctuations symbols; the second turns all the sentences into lowercase. This step is important because two words, written with the same characters, but with different cases, are considered different words (e.g. “science”, “SCIENCE”, “Science”). To avoid this case, all words are transformed into lowercase; last step is to lemmatize all the words (turning plural nouns to singular or turning all verbs to simple present form). 
### 2. Transformation of titles to readable titles by the machine
A classifier cannot actually classify sentences directly. It can manage with numbers, not words. For this reason, before training the classifier, we need to transform all titles (that will be the input training) into "readable" input by the program. Readable means made of numbers. first, we will transform all title's words into "numeric objects", and then all titles will be represented by the set of his words.
To transform the words in “numeric objects”, we will use a word embedding technique. In the Natural Language Processing (NLP) field, we speak about word embedding when we refer to the set of language modeling and features learning techniques, able to map words and phrases to vectors of real numbers.  In 2013 a team of researchers led by Tomas Mikolov created [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf%C3%AC%E2%80%94%20%C3%AC%E2%80%9E%C5%93), a group of models used to produce word embedding. What I have called so far “numerical object”, it is better called vectors of real numbers. In fact, in NLP a word is well represented not by a single number, but by a vector of numbers, of which dimension is decided as applicable. In this step we will tranform all title's words into vectors of real numbers. Word2Vec models are able to define the number that best represent a word, based on which other words are most often close to the given word. The Word2Vec models I will use is called Continuous Bag of Word (CBOW). 

![fig. Word2Vec model](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/readme_images/02fig.Word2Vec_CBOW.JPG)

In figure it is schematised how this model works. CBOW is a neural network that consists of input, projection, and output layers. It predicts the target word w(t) from the context {…, w(t-2), w(t-1), w(t+1), w(t+2), …}. Given a corpus (for simplicity assume the corpus is only one sentence: “The beautiful cat is near the table”), the vocabulary is defined as the list of words contained in the corpus, without repetitions. The size of the vocabulary is V. Initially, each word of the vocabulary is coded as one hot vector (a vector with one bit “1”, and all rest “0”, length of vector = V). Then, a window size is selected for iterating over the sentences (for example, size window is 3). To each step, CBOW tries to predict the central word in the window (in the first step of our example is “beautiful”) given the context (“the” and “cat”), using a neural network. What happen is that the input word vectors are projected on the projection layer, that is shared for all the words. So, all vectors get projected in the same position. To each step, the neural network weights are updated, calculating the error between the target vector and the predicted vector.

When all words have been transormed into vectors of real numbers, we represent each title as the avarege of his words. Now our titles are no more texts but vectors of numbers, and we can use them to train our classifier.

### 3. Training of the classifier
Our classifier is a neural network called Multilayer Perceptron (MLP). It consists of an input, a hidden layer and an output. It utilises the supervised technique called backpropagation for training. Using asi input the titles, represented as vector od real numbers, and their corresponding class, already known, we will train the classifier.

## Structure of the project
These are the steps in order to start the program and to save the results:

1. The user has to provide the inital informations necessary: to write the name of the dataset next "to excel_database = " ( in my case, excel_database = database_TitlesAndClusters.xlsx). The dataset must contain only two columns, one with the titles and the others with the corresponding classes, and must be saved in the same folder of the project; to write the number of titles' column of the dataset next "column_of_titles = " (in my case, column_of_titles = 0); to write the number of classes' column of the dataset next "column_of_classes = " (in my case, column_of_classes = 1). Be careful: the program works only if the two column are the first and the second of the excel file and if the firts column is indicated with the number 0, and the second with the number 1; to write the name of the text file, in which the final results will be saved, next to "save_in = " (in my case, save_in = risultati1.txt); to write a sentence not already classified, that will be classified, next to "to_classify = " (in my case, to_classify = teaching scientific methods into schools).


configurations of the lattice (in our case there is only one, configuration.txt) and eventually write a new one, using the syntax of configuration; if the user wants to do so, he has to specify the lattice parameters (N, M numberTemp and eqSteps) and also the local paths to the folders where data and graphs must be saved.
Then, to start the Ising model the user has to launch the file simulation which imports its parameters from configuration using ConfigParser library; there could be different types of configurations for the model, depending on the size of the lattice, the MonteCarlo steps or the folder where data are saved, so the user has to specify the configuration he wants when launching the simulation file from the command line with the syntax "python simulation.py name_of_the_configuration" (in our case, configuration.txt). The collected data (time evolution, energy and magnetization) are saved automatically in the data folder using their local paths.
At the end, the user has to launch the plots file with the configuration he wants; from command line, the syntax is "python plots.py name_of_the_configuration" (in our case, configuration.txt). The data are loaded from the configuration file through their local paths and then they are saved in the images folder automatically


### Support files
- [function.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/functions.py) contains all functions i have created but not already tested.
- [call_functions.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/call_functions.py) is used to call functions and see how they work.
- [testing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/testing.py) contains the rountine of testing i have built to test my functions.
### Project's files
0. [importing_dataset.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/importing_dataset.py) executes step 0 of the project.
1. [preprocessing.py](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/preprocessing.py) executes step 1 of the project.
### For user
- [configuration.txt](https://github.com/martinacaramaschi/ClassificationOfTitles/blob/master/configuration.txt) is the only one files the user have to edit to use the project by himself.
