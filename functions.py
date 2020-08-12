# -*- coding: utf-8 -*-
   
# to_tokenize_str function: it tokenize a sentence; 
# as output, a list of tokens is given
# As input, a string is needed 
def to_tokenize_str(string):
    from nltk.tokenize import word_tokenize
    word_tokens = []
    word_tokens = word_tokenize(string)
    return(word_tokens)

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

''' not used
# to_remove_stopw_and_punct function: it removes stopw and punctuations
# as output, a list of sentences is given
# As input, a list of sentence is needed 
def to_remove_stopw_and_punct(titles): #Funziona
    filtered_titles = [[] for i in range(len(titles))]
    for i in range(len(titles)):
        filtered_titles[i] = to_tokenize_str(titles[i])
        filtered_titles[i] = to_remove_sw_and_punct_from_list(filtered_titles[i])
        filtered_titles[i] = to_join_list(filtered_titles[i])
    return(filtered_titles)
'''

# to_remove_stopw_and_punct function: it removes stopw and punctuations
# as output, a list of sentences is given
# As input, a list of sentence is needed 
def to_remove_sw_and_punct_from_sent(sentence):
    new_sentence = [ ]
    new_sentence = to_tokenize_str(sentence)
    new_sentence = to_remove_sw_and_punct_from_list(new_sentence)
    sentence = to_join_list(new_sentence)
    return(sentence)

''' not used
def to_lemmatize(titles):
    lemmatized_tokens = [[] for i in range(len(titles))]
    lemmatized_titles = [[] for i in range(len(titles))]
    for i in range(len(titles)):
        list = []
        list = to_tokenize_str(titles[i])
        for token in list:
            tokenized = to_lemmatize_word(token)
            lemmatized_tokens[i].append(tokenized)
        lemmatized_titles[i] = to_join_list(lemmatized_tokens[i])
    return(lemmatized_titles)
'''
    
def to_lemmatize_word(w):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize((w))
    new_w = lemmatizer.lemmatize((lemma), "v")
    return(new_w)
    
def to_lemmatize_sent(sentence):
    new_sentence = []
    new_sentence = to_tokenize_str(sentence)
    lemmatized_sentence = []
    for w in new_sentence:
        lemmatized_sentence.append(to_lemmatize_word(w))
    new_sentence = to_join_list(lemmatized_sentence)
    return(new_sentence)
    
print(list(set(["cat", "me", "love", "me"])))
print(list(["today is today", "today future"]))

#the list of all words with repetitions 
def list_words(titles):
    all_words_in_titles = [] 
    for line in titles:
        x  = []
        x = to_tokenize_str(line)
        for element in x:
            all_words_in_titles.append(element)
    return(all_words_in_titles)