# -*- coding: utf-8 -*-

def from_word_to_vec(input_titles):
    from gensim.models import Word2Vec
    model_vectors = Word2Vec(input_titles, size=50, workers=10, min_count=1, iter=300, window=3)
    vocabulary = model_vectors.wv.vocab
    return(model_vectors, vocabulary)

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