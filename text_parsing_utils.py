import nltk
import re

'''
Splits a string into unique words.
'''
def splitIntoWords(s):
    tokens = nltk.word_tokenize(s)
    
    # lowercase and trim
    tokens = map(lambda token: token.strip().lower(), tokens)

    # remove non-letter characters
    tokens = map(lambda token: re.sub('[^a-z]+', '', token), tokens)

    # remove empty string
    tokens = filter(lambda token: token != '', tokens)

    return list(set(tokens))