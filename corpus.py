import json
import numpy as np

from text_vector import TextVector
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

        print("Creating data:")
        self.inputData, self.outputData = self._createData()
        print("input data size: " + str(self.inputData.shape))
        print("output data size: " + str(self.outputData.shape))

    '''
    Returns a list of words used in alphabetical order.
    '''
    def _createWordList(self):
        with open(self.fileName, 'r') as f:
            wordFrequencies = {}
            for line in f:
                jsonObj = json.loads(line)
                words = tpu.splitIntoWords(jsonObj['reviewText'])
                for word in words:
                    if word in wordFrequencies:
                        wordFrequencies[word] += 1
                    else:
                        wordFrequencies[word] = 1
            
            wordList = filter(lambda kv: kv[1] > 1, wordFrequencies.items())
            wordList = list(map(lambda kv: kv[0], wordList))
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
    Returns two numpy matrices: a matrix with a row for each data point and a column for each feature,
    and a vector with a category for each data point.
    '''
    def _createData(self):
        with open(self.fileName, 'r') as f:
            # find number of lines in file
            numLines = sum(1 for line in f)
            f.seek(0)

            # initialize data with all 0s
            inputData = np.zeros((numLines, len(self.wordIndices)))
            outputData = np.zeros((numLines, 1))
            
            i = 0
            for line in f:
                jsonObj = json.loads(line)
                text = jsonObj['reviewText']
                rating = jsonObj['overall']

                # for each word in the review, mark a 1 for that feature in inputData
                words = tpu.splitIntoWords(text)
                for word in words:
                    if word in self.wordIndices:
                        inputData[i, self.wordIndices[word]] = 1

                # mark the category in outputData
                outputData[i, 0] = rating
                
                i += 1

            return inputData, outputData

    '''
    Returns the list of words in a vector.
    index: index of vector to read
    '''
    def getWordList(self, index):
        words = []
        for i in range(len(self.inputData[index])):
            if self.inputData[index, i] == 1:
                words.append(self.wordList[i])
        return words