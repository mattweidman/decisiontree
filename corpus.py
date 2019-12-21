import json
import numpy as np

import decision_tree as dt
import text_parsing_utils as tpu

'''
A text corpus with raw string data.
'''
class TextCorpus:
    '''
    Creates a text corpus from a file.
    fileName: name of file to read.
    startIndex: first line in the corpus to read from
    endIndex: one more than the last line in the corpus to read from
    '''
    def __init__(self, fileName, startIndex=None, endIndex=None):
        self.fileName = fileName
        self.startIndex = startIndex
        self.endIndex = endIndex

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
            if self.endIndex is not None:
                if self.startIndex is not None:
                    numLines = self.endIndex - self.startIndex
                else:
                    numLines = self.endIndex
            else:
                totalLines = sum(1 for line in f)
                if self.startIndex is not None:
                    numLines = totalLines - self.startIndex
                else:
                    numLines = totalLines

            f.seek(0)

            # initialize data with all 0s
            inputData = np.zeros((numLines, len(self.wordIndices)))
            outputData = np.zeros((numLines, 1))
            
            i = 0
            for line in f:
                if self.startIndex is not None and i < self.startIndex:
                    i += 1
                    continue

                if self.endIndex is not None and i >= self.endIndex:
                    break

                dataIndex = i if self.startIndex is None else i - self.startIndex

                jsonObj = json.loads(line)

                # for each word in the review, mark a 1 for that feature in inputData
                text = jsonObj['reviewText']
                words = tpu.splitIntoWords(text)
                for word in words:
                    if word in self.wordIndices:
                        inputData[dataIndex, self.wordIndices[word]] = 1

                # mark the category in outputData
                rating = jsonObj['overall']
                outputData[dataIndex, 0] = rating
                
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

    '''
    Returns a new decision tree from this data.
    maxDepth: maximum depth of decision tree. If None, stops when data runs out.
    '''
    def getDecisionTree(self, maxDepth=None):
        indices = np.array(range(len(self.inputData)))
        tree = dt.DecisionTreeNode(self.inputData, self.outputData, indices)
        tree.split(maxDepth)
        return tree