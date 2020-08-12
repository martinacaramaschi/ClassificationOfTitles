# -*- coding: utf-8 -*-

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
    string = re.sub(r"'", " '", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"\:", "", string)
    string = re.sub(r"\.", "", string)
    return(string.strip().lower())

@given(st.text(min_size=1, max_size=3))
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
    if string == "'":
        assert to_clean_str(string) == " '"

