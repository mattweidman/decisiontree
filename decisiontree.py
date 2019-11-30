# This program uses decision trees to guess sentiments of Amazon reviews.
# Each input data point is an Amazon review (string).
# Each review is converted into a vector, containing a boolean value for each word.
# In vector v, v[i] is True if the word represented at index i is in the review.
# The decision tree is trained to map the vector to a star rating.

import json
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

'''
A text corpus with raw string data.
'''
class TextCorpus:
    '''
    Creates a text corpus from a file.
    fileName: name of file to read.
    '''
    def __init__(self, fileName):
        self.fileName = fileName

        print("Creating word index...")
        self.wordList = self._createWordList()
        self.wordIndices = self._createWordIndices()
        print("Word index:")
        print(str(len(self.wordIndices)) + " words indexed.")

        print("Creating vectors:")
        self.reviews = self._createVectors()
        print(str(len(self.reviews)) + " vectors created.")

    '''
    Returns a list of words used in alphabetical order.
    '''
    def _createWordList(self):
        with open(self.fileName, 'r') as f:
            wordSet = set()
            for line in f:
                jsonObj = json.loads(line)
                words = splitIntoWords(jsonObj['reviewText'])
                wordSet.update(words)
            
            wordList = list(wordSet)
            wordList.sort()

            return wordList

    '''
    Returns a dictionary mapping words to vector indices.
    '''
    def _createWordIndices(self):
        wordDict = {}
        for i in range(len(self.wordList)):
            wordDict[self.wordList[i]] = i
        
        return wordDict

    '''
    Returns a list of ReviewVectors generated from file.
    '''
    def _createVectors(self):
        with open(self.fileName, 'r') as f:
            vectors = []
            for line in f:
                jsonObj = json.loads(line)
                text = jsonObj['reviewText']
                rating = jsonObj['overall']
                vectors.append(ReviewVector(text, rating, self.wordIndices))

            return vectors

'''
A boolean vector representing a review.
'''
class ReviewVector:
    '''
    Creates a vector from a review.
    text: review text
    rating: rating from review
    wordIndicies: dictionary mapping words to indexes in all vectors.
    '''
    def __init__(self, text, rating, wordIndicies):
        self.rating = rating

        self.vector = [False] * len(wordIndicies)
        words = splitIntoWords(text)
        for word in words:
            if word in wordIndicies:
                self.vector[wordIndicies[word]] = True

    '''
    Returns the list of words in this review vector.
    totalWordList: ordered list of all words in the corpus.
    '''
    def getWordList(self, totalWordList):
        words = []
        for i in range(len(self.vector)):
            if self.vector[i]:
                words.append(totalWordList[i])
        return words

if __name__ == "__main__":
    corpus = TextCorpus('Musical_Instruments_5.json')

    print("The first review has the following words:")
    print(corpus.reviews[0].getWordList(corpus.wordList))