# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd

def to_lower(string):
    """This method transforms an input string into a lower cased string.
       
    Parameters
        string : text of the string.
    
    Returns:
        a new string, in which all previous uppercased letters
        are transformed into lowercased.
        
    Raise:
        InputError if the input is not a string."""        
    try:
        lower_case = ""
        for character in string:
            # to change uppurcased letter to lowercased
            if 'A' <= character <= 'Z':
                location = ord(character) - ord('A')
                new_ascii = location + ord('a')
                character = chr(new_ascii)
            # to change accented uppercased letter to lowercased
            elif chr(192) <= character <= chr(214) or \
                chr(216) <= character <= chr(221):
                new_ascii = ord(character) + 32
                character = chr(new_ascii)
            lower_case = lower_case + character   
    except:
        print("Error! Not valid input! Must be a string!")
    return(lower_case)

def to_clean_str(string):
    """This method removes or substitutes an unwanted part of a string.
       It uses "re" library.
       
    Parameters
        string : text of the string.
    
    Returns:
        the old string, without or substituting certain parts."""
    import re
    # re.sub functuion substitutes the fist input with the second input in string
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r"n't", " not", string)
    string = re.sub(r"'ve", " have", string)
    string = re.sub(r"'re", " are", string)
    string = re.sub(r"'m", " am", string)
    string = re.sub(r"s'", "s", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"´", "", string) 
    string = re.sub(r"'", " ' ", string)
    string = re.sub(r"\:", "", string)
    string = re.sub(r"\.", "", string)
    # return the string without space at the beginning and the end
    return(string.strip().lower())

def to_remove_sw_and_punct_from_list(list):
    """This method removes english stopwords and punctuations from a list of strings.
       It uses the "stopwords" function of "nltk.corpus" library.
       
    Parameters
        list : list of tokens.
    
    Returns:
        a new list of tokens, without having stopwords and punctuations."""
    from nltk.corpus import stopwords
    new_list = []
    sw = set(stopwords.words('english')) 
    punct = {'.', ':', ',', '!', '?', '--', '``', '-','(', ')', "'", '\n', "''", '&'}
    for w in list:
        if w not in sw and w not in punct:
            new_list.append(w)
    return(new_list)

def to_join_list(list):
    """This method joins the tokens, belonging to a list of strings.
       
    Parameters
        list : list of tokens.
    
    Returns:
        a string, made of each tokens, joined, alternating with spaces."""      
    sep = ' '
    sentence = sep.join(list)
    return(sentence)

def to_remove_sw_and_punct_from_sent(sentence):
    """This method removes english stopwords and punctuation's symbols from sentence.
    It tokenize the sentence using "word_tokenise" of "nltk.tokenize" library; 
    then uses to_remove_sw_and_punct_from_list on the list of tokens; 
    then re-joins the sentence using to_join_list.
       
    Parameters
        sentence : a sentence as string.
    
    Returns:
        the same sentence, but without stopwords and punctuation's symbols."""
    from nltk.tokenize import word_tokenize
    new_sentence = [ ]
    new_sentence = word_tokenize(sentence)
    new_sentence = to_remove_sw_and_punct_from_list(new_sentence)
    sentence = to_join_list(new_sentence)
    return(sentence)

def to_lemmatize_word(w):
    """This method lemmatizes a word, that is changes plural nouns to singular;
    or verb to infinitve form.
    It uses "WordNetLemmatizer" function of "nltk.stem" library to do that.
       
    Parameters
        w : a word as string.
    
    Returns:
        the same word, but tokenised."""  
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize((w))
    new_w = lemmatizer.lemmatize((lemma), "v")
    return(new_w)

def to_lemmatize_sent(sentence):
    """This method lemmatizes a sentence, using to_lemmatize_word on all the single words.
    It tokenizes the sentence, then lemmatizes each words, and re-joins the sentence.
       
    Parameters
        sentence : a sentence as string.
    
    Returns:
        the lemmatized sentence."""
    from nltk.tokenize import word_tokenize
    new_sentence = []
    new_sentence = word_tokenize(sentence)
    lemmatized_sentence = []
    for w in new_sentence:
        lemmatized_sentence.append(to_lemmatize_word(w))
    new_sentence = to_join_list(lemmatized_sentence)
    return(new_sentence)

def to_list_all_words_with_repetition(titles):
    """This method makes the list of all words written into a list of sentences
    (with repetitions). It tokenizes each sentence, using "word_tokenise of 
    "nltk.tokenize" library; then appends each tokens into a list.
       
    Parameters
        titles : a list of sentences as strings.
    
    Returns:
        a list of words.
        
    Raise:
        Only if all words are added to the list, this function returns the list."""
    from nltk.tokenize import word_tokenize
    all_words_in_titles = [] 
    count = 0
    for line in titles:
        x  = []
        x = word_tokenize(line)
        count += len(x)
        for element in x:
            all_words_in_titles.append(element)
    # check point
    if count == len(all_words_in_titles):
         return(all_words_in_titles)
     
def to_list_all_words_no_repetition(list_all_words):
    """This method takes a list of words and exclude all repeated words. 
       
    Parameters
        list_all_words : a list of words as strings.
    
    Returns:
        a list of words as string, without repeated elements."""
    all_words_in_titles_no_repetition = list(set(list_all_words))
    return(all_words_in_titles_no_repetition)

#
def to_import_dataset(name_excel_dataset, 
                      number_of_titles_column, 
                      number_of_classes_column
                      ):
    """This method imports a two-columns dataset from an excel file called "name_excel_dataset";
    number_of_titles_column is the number of titles' column (must be 0 or 1);
    number_of_classes_column is the number of classes' column (must be 0 or 1).
    
    Parameters
    name_excel_dataset : it is a string.  
    number_of_titles_column : is an integer (0 or 1).
    number_of_classes_column : is an integer (0 or 1).
    
    Raises:
    ValueError if the dataset name is uncorrect or the dataset is not found.
    ValueError if the numbers of two columns are the same, or lessa than zero or
    much than 1

    Returns:
    The list of titles (strings) and the list of classes."""
    try:
        dataset = pd.read_excel(name_excel_dataset, )
        dataset = pd.read_excel(name_excel_dataset)
    except: 
        raise ValueError('excel dataset {} not found or uncorrect! please check configuration'
                         .format(name_excel_dataset))
    dataframe = dataset.copy()
    if number_of_titles_column < 0 or number_of_titles_column > 1:
        raise ValueError('number of titles column must be 0 if it is the first \
                         column, 1 if it is the second')
    if number_of_classes_column < 0 or number_of_classes_column > 1:
        raise ValueError('number of classes column must be 0 if it is the first \
                         column, 1 if it is the second')
    titles = dataframe.columns[number_of_titles_column]
    classes = dataframe.columns[number_of_classes_column]
    My_titles = dataframe[titles]
    My_classes = dataframe[classes]
    return(My_titles, My_classes)

def preprocessing_titles(titles):
    """This method performs the preprocessing of a list of titles, before using them
    for the Word2Vec training. It transforms all titles into lowercased titles; then
    removes useless words and homogenizes the endings of nouns and the tenses; then
    tokenizes all titles.

    Parameters
    titles : list of strings.

    Returns
    a list of cleaned and tokenized titles."""
    from nltk.tokenize import word_tokenize
    # Step to lowercase all titles
    titles_lowercase = []
    for line in titles:
        titles_lowercase.append(to_lower(line))
    # Step to clean all titles
    titles_cleaned = []
    for line in titles_lowercase:
        titles_cleaned.append(to_clean_str(line))
    # Step to remove stop words and punctuation symbols
    titles_without_sw_and_punct = []
    for line in titles_cleaned:
        titles_without_sw_and_punct.append(to_remove_sw_and_punct_from_sent(line))
    # Step to lemmatize verbs and nouns
    titles_lemmatized = []
    for line in titles_without_sw_and_punct:
        titles_lemmatized.append(to_lemmatize_sent(line))    
    # Step to tokenize each title
    input_titles = []
    for line in titles_lemmatized:
        block = word_tokenize(line)
        input_titles.append(block)
    return(input_titles)

def from_word_to_vec(input_titles):
    """This method trains a Word2Vec model, having as input a list of titles, 
    previously tokenised. It uses "Word2Vec" of "gensim.models" library, for the training.
    
    Word2vec model is a neural network that assign a vector of real numbers to each words,
    based on the position of that word into a text; the word vectors will be of "size" 
    dimension; are transformed into vectors, all words that appear almost "min_count" 
    times into the text; the neural net is trained considerig "windows" words at a time.
    

    Parameters
    input_titles : a list of lists of tokens (each token is a word, as string),
                   that would be the list of all titles.

    Returns:
        the Word2vec trained model and the vocabulary, containg all words."""
    from gensim.models import Word2Vec
    model_vectors = Word2Vec(input_titles, size=50, workers=10, min_count=1, iter=300, window=3)
    vocabulary = model_vectors.wv.vocab
    return(model_vectors, vocabulary)

def to_average_a_title(tokens_title, model):
    """This method takes a list of tokens (that are the element of a title)
    and changes each tokens to its corresponding vectors of real numbers,
    using the previously calculated Word2Vec model; then it calculates the 
    average of all vectors.
    
    Parameters
    tokens_title : a list of tokens (strings).
    model : word2vec model trained
        
    Returns:
        the average of all words of the title."""
    #each word is represented by a vector of real numbers of dimensions (1, 50) 
    average_sentence = np.zeros((1, 50)) 
    # all words are summed
    count = 0
    for word in tokens_title:
        average_sentence += model[word]  # model[word] is the vector of "word" word
        count += 1
    average_sentence = average_sentence/count
    count = 0
    return(average_sentence)

def to_average_all_titles(all_tokens_title, len_titles, model):
    """This method calculates, per each title in a list of tokenised titles,
    the average of the words of the title, using to_average_a_title function. 

    Parameters
    all_tokens_title : list of lists of tokens as strings (each list contains the tokens of a title).
    len_titles : int. the number of titles.
    model : Word2Vec model.
    
    Returns : 
        a list of vectors of real numbers.
    """
    # average_sentence will contain all averaged titles
    average_sentence = np.zeros((len_titles, 50))
    for line in range(0, len_titles):
        average_sentence[line] = to_average_a_title(all_tokens_title[line], model)
    return(average_sentence)

def to_create_classes_vector(list_of_classes):
    """This method brings the list of classes, assigned to each title, and creates
    an new list, replacing the class with an integer number. It also cretes the list
    of possible classes (for example, if we have ['me', 'you', 'me', 'him'], it became
    [[0.], [1.], [0.], [2.]] and the possible classes are ['me', 'you', 'him']).
    

    Parameters
    list_of_classes : list of strings.
       
    Returns:
    the list of classes, represented as integers and the list of possible classes."""
    possible_classes = to_list_all_words_no_repetition(list_of_classes)
    number_of_classes = len(possible_classes)
    dim = len(list_of_classes)
    # assign a progressive integer number for each possible class
    # list of classes became a list of integer called list_vector_of_classes
    list_vector_of_classes = np.zeros((dim, 1))
    for y in range(0, number_of_classes):
        for i in range(0, dim):
            if list_of_classes[i] == possible_classes[y]:
                list_vector_of_classes[i, 0] = y
    return(list_vector_of_classes, possible_classes)

def divide_train_test(average_sentence, vector_of_classes):
    """This method divides the already classified titles (and corresponding classes)
    into two groups: training and testing. That's because we need to train the 
    classifier (using a part of titles and classes) and then tests the calssifier's
    ability of classify new titles (using the testing titles and classes).
    

    Parameters
    average_sentence : vector of vectors of real numbers (represents the titles).
    vector_of_classes : vector of vectors of intergers (represents the classes)

    Returns:
    the traing and testing vectors of titles and classes.
    
    Raise:
    It works only if the number of available titles is high enough."""
    sep = 1100
    if len(average_sentence) == len(vector_of_classes) and 0 < sep < len(average_sentence):
        train_x = average_sentence[:sep]
        train_y = vector_of_classes[:sep] 
        test_x = average_sentence[sep:]
        test_y = vector_of_classes[sep:]
    else:
        print("Error! corpus too short, uses a list of titles longer than 1100")
    return(train_x, train_y, test_x, test_y)

 
def prepare_new_title(model, vocabulary, title):
    """This method takes the list of tokens of a title and represents them as vectors
    of real numbers; then makes the average of all words and return the average, ready
    to be classified.

    Parameters
    model : word2vec model, already trained.
    vocabulary : Word2Vec vocabulary, containg all word vectors words.
    title : list of tokens of an unclassified title. 

    Returns
    the title, represented as a vector of real numbers.
    
    Raise:
    It works only if all tokens are contained in the vocabulary."""
    average_new_title = np.zeros((1, 50))
    count = 0
    for element in title:
        for word in element:
            if word not in vocabulary:
                raise ValueError('Words not exist in the vocabulary. I can not classify this sentence')
            else:
                average_new_title += model[word]    
                count += 1
    average_new_title = average_new_title/count 
    return(average_new_title)

def classifier(train_x, train_y, test_x, test_y, new_sent):
    """This method trains and tests the classifier, a multilayer perceptron neural net,
    using the training and testing titles and classes. It uses the "MLPClassifier"
    neural net from "sklearn.neural_network" library.
    It also classifies an unknown title, written as vector of real numbers.
    If we would know the probability assigned to each classes after the classification,
    we could use the command "reg.predict_proba(title)".
    
    Parameters
    train_x : list of vector of real numbers.
    train_y : list of vector of integer numbers.
    test_x : list of vector of real numbers.
    test_y : list of vector of integer numbers. 
    new_sent : vector of real numbers.

    Returns:
        the score of the training and testing (number of right classifications
    divided by the number of total classifications, and the class of the unknown title."""
    from sklearn.neural_network import MLPClassifier
    model_MLP = MLPClassifier(hidden_layer_sizes=(200, 100))
    reg = model_MLP.fit(train_x, train_y)
    score_train = reg.score(train_x, train_y)
    score_test = reg.score(test_x, test_y)
    class_new_sent = reg.predict(new_sent)
    return(score_train, score_test, class_new_sent)

