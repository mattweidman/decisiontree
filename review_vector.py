import text_parsing_utils as tpu

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
        words = tpu.splitIntoWords(text)
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