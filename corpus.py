import json

from review_vector import ReviewVector
import text_parsing_utils as tpu

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
                words = tpu.splitIntoWords(jsonObj['reviewText'])
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