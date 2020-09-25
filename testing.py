# -*- coding: utf-8 -*-
import functions as f
from hypothesis import given
import hypothesis.strategies as st

#-------------------------- test to_lower function ---------------------------#    

def test_to_lower_1():
    """Positive test:
    An uppercased word is espected to became lowercased."""
    assert f.to_lower("HELLO") == "hello"

def test_to_lower_2():
    """Positive test:
    An empty word is espected to remain empty."""
    string2 = ""
    assert f.to_lower(string2) == "" 

def test_to_lower_3():
    """Positive test:
    All the uppercased charachters in a generic string has to became lowercased.
    
    It uses .islower(): gives True if at least one lowercased letter is present
    in the string and if all letters in the string are lowercased."""
    my_string = "%TL gioco Riskio! OK"
    assert f.to_lower(my_string).islower() == True 

def test_to_lower_4():
    """Positive test:
    A wrong input have to return an empty word."""
    assert f.to_lower(689) == ''

@given(st.integers())    # @given(x) gives x as input to the following function
def test_to_lower_5(string):
    """Positive test:
    Giving as input any integer, an empty string is espected as output."""
    assert f.to_lower(string) == ''

# names contains all the possible texts with the features i have set
names = st.text(st.characters(min_codepoint=32, max_codepoint=221, 
                              blacklist_categories=('Cc', 'Cs')),
                min_size=2, 
                max_size=20
                )
# names is given as input to the following testing function
@given(names)
def test_to_lower_6(name):
    """Positive test:
    I espect that giving a generic string text, containing letters, numbers,
    punctuations and so on, all the uppercased letters turn into lowercased."""
    my_name = 'A'
    x = len(name)
    empty = ' '
    for i in range(0, x-1):
        # empty is a text string only made of speces
        empty = empty + ' '
    if name == empty:
        # all the empty string has to remain empty
        assert f.to_lower(name) == name
    else:
        # 'name' colud be made only of punctuation symbols.
        # in that case, name.islower() is always False
        # to avoid that case, 'name2' is the generic text sting 'name',
        # with the adding of letter.
        name2 = my_name + name
        assert f.to_lower(name2).islower() == True

#------------------------- test to_clean_str function -------------------------#

@given(st.text(min_size=2, max_size=3))
def test_to_clean_str_1(string):
    """Positive test"""
    if string == "\'s" or string == "\'d" or string == "\'ll":
        assert f.to_clean_str(string) == " "
    if string == "'ve":
        assert f.to_clean_str(string) == " have"
    if string == "n't":
        assert f.to_clean_str(string) == " not"
    if string == "'re":
        assert f.to_clean_str(string) == " are"
    if string == "\'m":
        assert f.to_clean_str(string) == " am"

def test_to_clean_str_2():
    """Positive test:"""
    assert f.to_clean_str('teacher´') == 'teacher'
    assert f.to_clean_str('inquiry-based') == 'inquirybased'
    assert f.to_clean_str("t't") == "t ' t"

#------------- test to_remove_sw_and_punct_from_list function ----------------#

def test_remove_sw_punct_1():
    """Positive test:
    punctuations should be removed."""
    lista_test_1 = ['amore', '!', 'love']
    new_lista_test_1 = ['amore', 'love']
    assert f.to_remove_sw_and_punct_from_list(lista_test_1) == new_lista_test_1
   
def test_remove_sw_punct_2():
    """Positive test:
    all words belonging to stopwords should be removed."""
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    assert f.to_remove_sw_and_punct_from_list(stopwords) == []

def test_remove_sw_punct_3():
    """Positive test:
    all symbols belonging to punctuations should be removed."""
    punctuations = ['.', ':', ',', '!', '?', '--', '``', '-', '(', ')', "'",\
                    '\n', "''", '&']
    assert f.to_remove_sw_and_punct_from_list(punctuations) == []

# --------------------test to_join_list function------------------------------#

def test_to_join_list_1():
    """Positive test:
    Between each string should appear a space."""    
    lista = ['m', 'i', ' ', 'l', 'a', 'u', 'r', 'e', 'e', 'r', 'ò']
    assert f.to_join_list(lista) == 'm i   l a u r e e r ò'

# fixed sting lenght to 2        
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
    """Positive test:
    joining two element of lenght 2, the final lenght should be 5,
    because there is also a space in the middle."""
    lista = [a,b]
    assert len(f.to_join_list(lista)) == 5

# -------------------- test to_lemmatize_word function -----------------------#

def test_to_lemmatize_word_1():
    """Positive test:
    plural nouns became singular."""
    plural = 'plurals'
    singular = 'plural'
    assert f.to_lemmatize_word(plural) == singular
    
def test_to_lemmatize_word_2():
    """Positive test:
    verbs became all to infinitive form."""
    gerund = 'dying'
    participle = 'died'
    infinitive = 'die'
    assert f.to_lemmatize_word(gerund) == infinitive
    assert f.to_lemmatize_word(participle) == infinitive

def test_to_lemmatize_word_3():
    """Positive test:
    adjectives and casual words remain the same."""
    casual_w = 'kfsijheiwgjwl'
    adjective = 'lovely'
    assert f.to_lemmatize_word(casual_w) == casual_w
    assert f.to_lemmatize_word(adjective) == adjective

#---------------- test to_remove_sw_from_sent function -----------------------#

def test_to_remove_sw_from_sent():
    """Positive test:
    all punctuations and stopwords disappear from the sentence."""
    with_sw_and_punct = 'this phrase is good: but soon best!'
    without_sw_and_punct = 'phrase good soon best'
    assert f.to_remove_sw_and_punct_from_sent(with_sw_and_punct) == without_sw_and_punct
    
#------------------ test to_lemmatize_sent function --------------------------#

def test_to_lemmatize_sent():
    """Positive test:
    pural names into the sentence became sigular;
    verbs, not written in infinitive form, became infinitive."""
    with_plurals_and_verb = 'students playing sports'
    without_plurals_and_verb = 'student play sport'
    assert f.to_lemmatize_sent(with_plurals_and_verb) == without_plurals_and_verb
    
#--------------- test to_list_all_words_no_repetition function ---------------#

def test_to_list_all_words_no_repetition():
    """Positive test:
    the lenght of final list is equal to the number of objects in the list (without repetitions)."""    
    list_1 = ['me', 'em', 'me', 'me']
    assert len(f.to_list_all_words_no_repetition(list_1)) == 2
    list_2 = []
    assert f.to_list_all_words_no_repetition(list_2) == list_2
    list_3 = ['tutte', 'parole', 'diverse']
    assert len(f.to_list_all_words_no_repetition(list_3)) == len(list_3)
    
#------------- test to_list_all_words_with_repetition function ---------------#

def test_to_list_all_words_with_repetition():
    """Positive test:
    the number of element in the final list is equal to the sum of number of element 
    in the starting lists, without spaces."""
    titoli_test = ['tu come stai', 'io bene', '']
    assert len(f.to_list_all_words_with_repetition(titoli_test)) == 5


# ------------- for those who will use the project after me ------------------#
#------------------ here some useful commands to remember --------------------# 
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