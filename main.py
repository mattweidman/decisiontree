# This program uses decision trees to guess sentiments of Amazon reviews.
# Each input data point is an Amazon review (string).
# Each review is converted into a vector, containing a boolean value for each word.
# In vector v, v[i] is True if the word represented at index i is in the review.
# The decision tree is trained to map the vector to a star rating.

from corpus import TextCorpus

if __name__ == "__main__":
    corpus = TextCorpus('Musical_Instruments_5.json', 10, 20, True)
    tree = corpus.getDecisionTree()
    output = tree.toFlattenedIfString(corpus.wordList)
    with open("output.txt", "w") as f:
        f.write(output)