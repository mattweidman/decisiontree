import text_parsing_utils as tpu

'''
A boolean vector representing a single text data point.
'''
class TextVector:
    '''
    Creates a vector from some text.
    text: text to generate vector from
    category: category, or expected output
    wordIndicies: dictionary mapping words to indexes in all vectors.
    '''
    def __init__(self, text, category, wordIndicies):
        self.category = category

        self.vector = [False] * len(wordIndicies)
        words = tpu.splitIntoWords(text)
        for word in words:
            if word in wordIndicies:
                self.vector[wordIndicies[word]] = True

    '''
    Returns the list of words in this vector.
    totalWordList: ordered list of all words in the corpus.
    '''
    def getWordList(self, totalWordList):
        words = []
        for i in range(len(self.vector)):
            if self.vector[i]:
                words.append(totalWordList[i])
        return words