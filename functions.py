# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd

def to_lower(string):
    """This method transforms an input string into a lower cased string.
       
    Parameters
        string : text of the string.
    
    Returns:
        a new string, in which all previous uppercased letter.
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
    """This method removes or substitutes a unwanted part of a string.
       It uses re library.
       
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
    """This method removes english stopwords and punctuation from a list of strings.
       It uses the function "stopwords" of nltk.corpus library.
       
    Parameters
        list : list of tokens.
    
    Returns:
        a new list of tokens, without having stopwords and punctuation."""
    from nltk.corpus import stopwords
    new_list = []
    sw = set(stopwords.words('english')) 
    punct = {'.', ':', ',', '!', '?', '--', '``', '-','(', ')', "'", '\n', "''", '&'}
    for w in list:
        if w not in sw and w not in punct:
            new_list.append(w)
    return(new_list)

def to_join_list(list):
    """This method joins the tokens, belonging to a list of string.
       
    Parameters
        list : list of tokens.
    
    Returns:
        a string, made of each tokens joint, alterning to spaces."""      
    sep = ' '
    sentence = sep.join(list)
    return(sentence)

def to_remove_sw_and_punct_from_sent(sentence):
    """This method removes english stopwords and punctuation from sentence.
    It tokenize the sentence using word_tokenise of nltk.tokenize library; 
    then uses to_remove_sw_and_punct_from_list on the list of tokens; 
    then re-join the sentence using to_join_list.
       
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
    """This method lemmatize a word, that is change plural nouns to singular;
    or verb to infinitve form.
    It uses the WordNetLemmatizer function of nltk.stem library.
       
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
    """This method lemmatize a sentence, using to_lemmatize_word on all the single words.
    It tokenize the sentence, then lemmatize each words, and re-join the sentence.
       
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
    """This method makes the list of all words into list of sentences (with repetitions).
    It tokenize each sentence using word_tokenise of nltk.tokenize library; 
    then appends each tokens into a list.
       
    Parameters
        titles : a list of sentences.
    
    Returns:
        a list of words.
        
    Raise:
        Only if all words are added to the list, it return the list."""
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

def from_word_to_vec(input_titles):
    """This method trains a Word2Vec model, having as input a list of titles, 
    previously tokenised. It uses Word2Vec of gensim.models library, for the training.
    

    Parameters
    input_titles : a list of lists of tokens (each token is a word, as string),
                   that would be the list of all titles

    Returns:
        the Word2vec trained model and the vocabulary, containg all words."""
    from gensim.models import Word2Vec
    model_vectors = Word2Vec(input_titles, size=50, workers=10, min_count=1, iter=300, window=3)
    vocabulary = model_vectors.wv.vocab
    return(model_vectors, vocabulary)

def to_average_a_title(tokens_title, model):
    """This method takes a list of tokens (that are the element of a title)
    and changes each tokens to its corresponding vectors of real numbers,
    calculated using Word2Vec; then it calculates the average of all vectors.
    model is the word2vec model trained, i need to do the change.

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
    sep = 1100
    if len(average_sentence) == len(vector_of_classes) and 0 < sep < len(average_sentence):
        train_x = average_sentence[:sep]
        train_y = vector_of_classes[:sep] 
        test_x = average_sentence[sep:]
        test_y = vector_of_classes[sep:]
    return(train_x, train_y, test_x, test_y)

def classifier(train_x, train_y, test_x, test_y, new_sent):
    from sklearn.neural_network import MLPClassifier
    model_MLP = MLPClassifier(hidden_layer_sizes=(200, 100))
    reg = model_MLP.fit(train_x, train_y)
    score_train = reg.score(train_x, train_y)
    score_test = reg.score(test_x, test_y)
    class_new_sent = reg.predict(new_sent)
    return(score_train, score_test, class_new_sent)
"""
from sklearn.neural_network import MLPClassifier
model_MLP = MLPClassifier(hidden_layer_sizes=(200, 100))
reg = model_MLP.fit(train_x, train_y)

print(reg.score(train_x, train_y))
print(reg.score(test_x, test_y))
print(print(reg.predict_proba(test_x[:10])))

print(reg.predict(test_x[:8]))
"""
   
# preprocessing_titles function:
# As input, a list of sentences (in my case, titles) is needed.
# It lowercased all sentences; remove stop words and punctuation symbols from sentences;
       # it lemmatized all nouns and verbs.
# As output, the list of senteces after all preprocessing steps.
def preprocessing_titles(titles):
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
    #creo i vettori di lemmi
    input_titles = []
    for line in titles_lemmatized:
        block = word_tokenize(line)
        input_titles.append(block)
    return(input_titles)


""" This function import a two-columns dataset from an excel file called name_excel_dataset
    column_titles is the number of titles' column (must be 0 or 1)
    column_classes is the number of classes' column (must be 0 or 1)
    It returns the titles and classes to use for classifier training and testing """
    
def to_import_dataset(name_excel_dataset, 
                      number_of_titles_column, 
                      number_of_classes_column
                      ):
    try:
        dataset = pd.read_excel(name_excel_dataset, )
        dataset = pd.read_excel(name_excel_dataset)
    except: 
        raise ValueError('excel dataset {} not found or uncorrect! please check configuration'
                         .format(name_excel_dataset))
    dataframe = dataset.copy()
    if number_of_titles_column < 0 or number_of_titles_column > 2:
        raise ValueError('number of titles column must be 0 if it is the first \
                         column, 1 if it is the second')
    if number_of_classes_column < 0 or number_of_classes_column > 2:
        raise ValueError('number of classes column must be 0 if it is the first \
                         column, 1 if it is the second')
    titles = dataframe.columns[number_of_titles_column]
    classes = dataframe.columns[number_of_classes_column]
    My_titles = dataframe[titles]
    My_classes = dataframe[classes]
    return(My_titles, My_classes)

def prepare_new_title(model, title):
    average_new_title = np.zeros((1, 50))
    count = 0
    for element in title:
        for word in element:
            average_new_title += model[word]    
            count += 1
    average_new_title = average_new_title/count 
    return(average_new_title)
