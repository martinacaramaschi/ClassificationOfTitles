# -*- coding: utf-8 -*-

from functions import to_remove_sw_and_punct_from_sent, to_lemmatize_sent

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
    string = re.sub(r"'", " '", string)
    string = re.sub(r"-", "", string)
    string = re.sub(r"\:", "", string)
    string = re.sub(r"\.", "", string)
    return(string.strip().lower())


# preprocessing_titles function:
# As input, a list of sentences (in my case, titles) is needed.
# It lowercased all sentences; remove stop words and punctuation symbols from sentences;
       # it lemmatized all nouns and verbs.
# As output, the list of senteces after all preprocessing steps.
def preprocessing_titles(titles):
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
    return(titles_lemmatized)