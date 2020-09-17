# -*- coding: utf-8 -*-
from functions import to_lower, \
                          to_clean_str, \
                          to_remove_sw_and_punct_from_list,\
                          to_join_list, \
                          to_lemmatize_word, \
                          to_remove_sw_and_punct_from_sent,\
                          to_lemmatize_sent, \
                          to_list_all_words_with_repetition, \
                          to_list_all_words_no_repetition
'''
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
'''
#------------------ first block of testing functions -------------------------#    
""" Here i have created four simple positive tests"""

# An uppercased word is espected to became lowercased
def test_to_lower_1():
    assert to_lower("HELLO") == "hello"

# An empty word is espected to remain empty
def test_to_lower_2():
    string2 = ""
    assert to_lower(string2) == "" 

# All the uppercased charachters in a generic string has to became lowercased
def test_to_lower_3():
    my_string = "%TL gioco Riskio! OK"
    # .islower() gives True if at least one lowercased letter is present in the
    # string and if all letters in the string are lowercased
    assert to_lower(my_string).islower() == True 

# A wrong input have to return an empty word
def test_to_lower_4():
    assert to_lower(689) == ''

#------------------ second block of testing functions ------------------------#
""" Here i have two more generic testing functions. 
    Using 'given' from hypotesis and 'strategies', we can generate generic text
    strings. The following web site really helped me in understand how hypothesis
    functions work: https://hypothesis.works/articles/generating-the-right-data/
    """

from hypothesis import given
import hypothesis.strategies as st

# Giving as input any integer, an empty string is espected as output
@given(st.integers())            # @given(x) gives x as input to the following function
def test_to_lower_5(string):
    assert to_lower(string) == ''

""" I espect that giving a generic string text, containing letters, numbers,
    punctuations and so on, all the uppercased letters turn into lowercased. 
    
    st.text is used to generate string text, having certain features. 
    For example, st.text(min_size=2, max_size=20) generates a text using from 2
    to 20 charachters.
    st.characheters is used to generate characters, having certain features.
    For example, st.characters(min_codepoint=32) generates an ascii character
    using only code number from 32.
    Combining st.text and st.characters it is possible to specify which characters
    the text string can contain.
    
    I will create text string containg ascii charachters, excluding 'Cc' and ' Cs'
    Unicode Character Categories and fixing the min and max charachters code number.
"""
# names contains all the possible texts with the features i need
names = st.text(st.characters(min_codepoint=32, max_codepoint=221, 
                              blacklist_categories=('Cc', 'Cs')),
                min_size=2, 
                max_size=20
                )
# names is given as input to the following testing function
@given(names)
def test_to_lower_6(name):
    my_name = 'A'
    x = len(name)
    empty = ' '
    for i in range(0, x-1):
        # empty is a text string only made of speces
        empty = empty + ' '
    if name == empty:
        # all the empty string has to remain empty
        assert to_lower(name) == name
    else:
        # 'name' colud be made only of punctuation symbols.
        # in that case, name.islower() is always False
        # to avoid that case, 'name2' is the generic text sting 'name',
        # with the adding of letter.
        name2 = my_name + name
        assert to_lower(name2).islower() == True
        
# ------------- useful commands to remember ----------------------------------# 
#   
#1.    st.text().example() show you an example of st.text()
#
#      st.text().example()
#      >> '\x12'
#
#2.    category(char) show you the unicode category of char
#
#      from unicodedata import category
#      category('A')
#      >> 'Lu'
#
#3.    ord(char) show you the ascii code of char
#
#      ord('A')
#      >> 65
#
#4.    char(number) show you the char of number ascii code
#
#      chr(216)
#      >> 'Ø'
# ----------------------------------------------------------------------------#
'''
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
        string = re.sub("'", " ", string)
        string = re.sub(r"-", "", string)
        string = re.sub(r"\:", "", string)
        string = re.sub(r"\.", "", string)
        return(string.strip().lower())
'''

@given(st.text(min_size=2, max_size=3))
def test_to_clean_str_1(string):
    if string == "\'s" or string == "\'d" or string == "\'ll":
        assert to_clean_str(string) == " "
    if string == "'ve":
        assert to_clean_str(string) == " have"
    if string == "n't":
        assert to_clean_str(string) == " not"
    if string == "'re":
        assert to_clean_str(string) == " are"
    if string == "\'m":
        assert to_clean_str(string) == " am"

def test_to_clean_str_2():
    assert to_clean_str('teacher´') == 'teacher'
    assert to_clean_str('inquiry-based') == 'inquirybased'
    assert to_clean_str("t't") == "t ' t"
    
# ----------------------------------------------------------------------------#
""" non testo word_tokenize() perchè l'ho importata"""

# ----------------------------------------------------------------------------#
'''
def to_remove_sw_and_punct_from_list(list):
    from nltk.corpus import stopwords
    new_list = []
    sw = set(stopwords.words('english')) 
    punct = {'.', ':', ',', '!', '?', '--', '``', '-','(', ')', "'", '\n', "''", '&'}
    for w in list:
        if w not in sw and w not in punct:
            new_list.append(w)
    return(new_list)
'''
def test_remove_sw_punct_1():
    lista_test_1 = ['amore', '!', 'love']
    new_lista_test_1 = ['amore', 'love']
    assert to_remove_sw_and_punct_from_list(lista_test_1) == new_lista_test_1

# test sulle stop words    
def test_remove_sw_punct_2():
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    assert to_remove_sw_and_punct_from_list(stopwords) == []

# test sulle puntuations
def test_remove_sw_punct_3():
    punctuations = ['.', ':', ',', '!', '?', '--', '``', '-', '(', ')', "'",\
                    '\n', "''", '&']
    assert to_remove_sw_and_punct_from_list(punctuations) == []

# ----------------------------------------------------------------------------#
'''
def to_join_list(list):
    sep = ' '
    sentence = sep.join(list)
    return(sentence)
'''
def test_to_join_list_1():
       lista = ['m', 'i', ' ', 'l', 'a', 'u', 'r', 'e', 'e', 'r', 'ò']
       assert to_join_list(lista) == 'm i   l a u r e e r ò'
       
# mi aspetto che unendo due parole lunghe 2 caratteri, la frase finale sia lunga 5
# perchè lo spazio aggiunto è lunga uno 
@given(a = st.text(st.characters(min_codepoint=32, max_codepoint=221, 
                              blacklist_categories=('Cc', 'Cs')),
                min_size=2, 
                max_size=2
                ),
       b = st.text(st.characters(min_codepoint=32, max_codepoint=221, 
                              blacklist_categories=('Cc', 'Cs')),
                min_size=2, 
                max_size=2))
def test_to_join_list_2(a,b):
       lista = [a,b]
       assert len(to_join_list(lista)) == 5

# ----------------------------------------------------------------------------#
'''       
def to_lemmatize_word(w):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize((w))
    new_w = lemmatizer.lemmatize((lemma), "v")
    return(new_w)
'''

'''i will test if some plural nouns became singular; if verbs became all in
 infinitive form; if the adjectives and casual words remain the same'''
def test_to_lemmatize_word_1():
    plural = 'plurals'
    singular = 'plural'
    assert to_lemmatize_word(plural) == singular
    
def test_to_lemmatize_word_2():
     gerund = 'dying'
     participle = 'died'
     infinitive = 'die'
     assert to_lemmatize_word(gerund) == infinitive
     assert to_lemmatize_word(participle) == infinitive

def test_to_lemmatize_word_3():
    casual_w = 'kfsijheiwgjwl'
    adjective = 'lovely'
    assert to_lemmatize_word(casual_w) == casual_w
    assert to_lemmatize_word(adjective) == adjective

#-----------------------------------------------------------------------------#
"""
def to_remove_sw_and_punct_from_sent(sentence):
    from nltk.tokenize import word_tokenize
    new_sentence = [ ]
    new_sentence = word_tokenize(sentence) #non usa to_tokenize_str
    new_sentence = to_remove_sw_and_punct_from_list(new_sentence)
    sentence = to_join_list(new_sentence)
    return(sentence)
"""
def test_to_remove_sw_from_sent():
    with_sw_and_punct = 'this phrase is good: but soon best!'
    without_sw_and_punct = 'phrase good soon best'
    assert to_remove_sw_and_punct_from_sent(with_sw_and_punct) == without_sw_and_punct
    
#----------------------------------------------------------------------------#
"""
def to_lemmatize_sent(sentence):
    from nltk.tokenize import word_tokenize
    new_sentence = []
    new_sentence = word_tokenize(sentence)
    lemmatized_sentence = []
    for w in new_sentence:
        lemmatized_sentence.append(to_lemmatize_word(w))
    new_sentence = to_join_list(lemmatized_sentence)
    return(new_sentence)
"""
def test_to_lemmatize_sent():
    with_plurals_and_verb = 'students playing sports'
    without_plurals_and_verb = 'student play sport'
    assert to_lemmatize_sent(with_plurals_and_verb) == without_plurals_and_verb
    
#----------------------------------------------------------------------------#
"""
def to_list_all_words_no_repetition(list_all_words):
    all_words_in_titles_no_repetition = list(set(list_all_words))
    return( all_words_in_titles_no_repetition)
"""
def test_to_list_all_words_no_repetition():
    list_1 = ['me', 'em', 'me', 'me']
    assert len(to_list_all_words_no_repetition(list_1)) == 2
    list_2 = []
    assert to_list_all_words_no_repetition(list_2) == list_2
    list_3 = ['tutte', 'parole', 'diverse']
    assert len(to_list_all_words_no_repetition(list_3)) == len(list_3)
    
#----------------------------------------------------------------------------#
"""
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
"""
def test_to_list_all_words_with_repetition():
    titoli_test = ['tu come stai', 'io bene', '']
    assert len(to_list_all_words_with_repetition(titoli_test)) == 5
    