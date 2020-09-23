# -*- coding: utf-8 -*-

""" VOGLIO ARRIVARE QUI
titles_vectors = to_average_words(preprocessed_titles)

"""
import numpy as np 
import pandas as pd

def to_average_a_title(tokens_title, model):
    import numpy as np
    #definisco il vettore che conterrà la media
    average_sentence = np.zeros((1, 50)) 
    #creo un ciclo che somma le parole della frase e poi divide per il numero di parole
    count = 0
    for word in tokens_title:
        average_sentence += model[word]     #sommo tutti i vettori che compongono sentence insieme
        count += 1
    average_sentence = average_sentence/count   #divido per il numero di parole per calcolare la media
    count = 0
    return(average_sentence)

def to_average_all_titles(all_tokens_title, len_titles, model):
    average_sentence = np.zeros((len_titles, 50))
    for line in range(0, len_titles):
        average_sentence[line] = to_average_a_title(all_tokens_title[line], model)
    return(average_sentence)

def from_word_to_vec(input_titles):
    from gensim.models import Word2Vec
    model_vectors = Word2Vec(input_titles, size=50, workers=10, min_count=1, iter=300, window=3)
    vocabulary = model_vectors.wv.vocab
    return(model_vectors, vocabulary)

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
    sep = 1000
    if len(average_sentence) == len(vector_of_classes) and 0 < sep < len(average_sentence):
        train_x = average_sentence[:sep]
        train_y = vector_of_classes[:sep] 
        test_x = average_sentence[sep:]
        test_y = vector_of_classes[sep:]
    return(train_x, train_y, test_x, test_y)

def classifier(train_x, train_y, test_x, test_y):
    from sklearn.neural_network import MLPClassifier
    model_MLP = MLPClassifier(hidden_layer_sizes=(200, 100))
    reg = model_MLP.fit(train_x, train_y)
    score_train = reg.score(train_x, train_y)
    score_test = reg.score(test_x, test_y)
    return(score_train, score_test)

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

#-----da qui in poi metto le funzioni testate che funzionano------------------#

# to_lower function: it transforms all input string to lower case
# As input, a string is needed 
def to_lower(string):
    """ This function tranform an input text string into the same, 
    but written in lowercase """
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

# to_clean_str function: it removes or substitutes some string 
# (e.g. 'll is removed and n't become not)
# As input a string is needed
def to_clean_str(string):
    import re
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
    string = re.sub(r"'", " ' ", string) #here i changed
    string = re.sub(r"\:", "", string)
    string = re.sub(r"\.", "", string)
    return(string.strip().lower())

# to_remove_sw_and_punct_from_list function: it removes stopwords and punctuation from a list; 
# as output, the list of remained tokens is given
# As input, a list is needed 
def to_remove_sw_and_punct_from_list(list):
    from nltk.corpus import stopwords
    new_list = []
    sw = set(stopwords.words('english')) 
    punct = {'.', ':', ',', '!', '?', '--', '``', '-','(', ')', "'", '\n', "''", '&'}
    for w in list:
        if w not in sw and w not in punct:
            new_list.append(w)
    return(new_list)

# to_join_list function: it join a list of elements; as output, the sentence is given
# As input, a list is needed 
def to_join_list(list):
    sep = ' '
    sentence = sep.join(list)
    return(sentence)

def to_lemmatize_word(w):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize((w))
    new_w = lemmatizer.lemmatize((lemma), "v")
    return(new_w)

# to_remove_stopw_and_punct function: it removes stopw and punctuations
# as output, a list of sentences is given
# As input, a list of sentence is needed 
def to_remove_sw_and_punct_from_sent(sentence):
    from nltk.tokenize import word_tokenize
    new_sentence = [ ]
    new_sentence = word_tokenize(sentence) #non usa to_tokenize_str
    new_sentence = to_remove_sw_and_punct_from_list(new_sentence)
    sentence = to_join_list(new_sentence)
    return(sentence)

def to_lemmatize_sent(sentence):
    from nltk.tokenize import word_tokenize
    new_sentence = []
    new_sentence = word_tokenize(sentence)
    lemmatized_sentence = []
    for w in new_sentence:
        lemmatized_sentence.append(to_lemmatize_word(w))
    new_sentence = to_join_list(lemmatized_sentence)
    return(new_sentence)

def to_list_all_words_no_repetition(list_all_words):
    all_words_in_titles_no_repetition = list(set(list_all_words))
    return(all_words_in_titles_no_repetition)

#the list of all words with repetitions 
def to_list_all_words_with_repetition(titles):
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