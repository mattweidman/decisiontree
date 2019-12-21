import math
import numpy as np

'''
A node of a decision tree that decides between categories based on binary data.
'''
class DecisionTreeNode:
    '''
    inputData: numpy matrix of input data
    outputData: numpy vector of output (categorical) data
    indices: indices of vectors in data that are given to this node.
    '''
    def __init__(self, inputData, outputData, indices):
        self.inputData = inputData
        self.outputData = outputData
        self.indices = indices
        self.allCategories = np.unique(outputData[:,0])

        self.left = None
        self.right = None

    '''
    Find the most frequent output in the data set.
    '''
    def _mostFrequentOutput(self):
        uniques, counts = np.unique(self.outputData[self.indices], return_counts=True)
        return uniques[np.argmax(counts)]

    '''
    Returns the feature index that should be used to determine which vectors go in which child nodes.
    Returns None if there is no way to further divide this dataset.
    '''
    def chooseSplitIndex(self):
        inputData = self.inputData[self.indices]
        outputData = self.outputData[self.indices]
        
        # featureTrueCounts[c, f] = number of data points with category c and feature f = 1
        # featureFalseCounts[c, f] = number of data points with category c and feature f = 0
        featureCountsShape = (len(self.allCategories), inputData.shape[1])
        featureTrueCounts = np.zeros(featureCountsShape)
        featureFalseCounts = np.zeros(featureCountsShape)
        for categoryIndex in range(len(self.allCategories)):
            vectorsWithCategory = (outputData == self.allCategories[categoryIndex]).astype(int)
            featureTrueCounts[categoryIndex] = (inputData * vectorsWithCategory).sum(0)
            featureFalseCounts[categoryIndex] = ((1 - inputData) * vectorsWithCategory).sum(0)

        # find total number with feature True and False
        totalWithFeatureTrue = inputData.sum(0)[np.newaxis, :]
        totalWithFeatureFalse = (1 - inputData).sum(0)[np.newaxis, :]

        # Calculate proportions that are needed to calculate entropy.
        # Some of these values will be nan, and that's okay because they will be replaced with 2 later.
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        proportionCategoryTrueInTrueNode = featureTrueCounts / totalWithFeatureTrue
        proportionCategoryFalseInTrueNode = 1 - proportionCategoryTrueInTrueNode
        proportionCategoryTrueInFalseNode = featureFalseCounts / totalWithFeatureFalse
        proportionCategoryFalseInFalseNode = 1 - proportionCategoryTrueInFalseNode
        np.seterr(**old_settings)

        # calculate entropy of each node
        nodeEntropy = lambda p: -p * np.log2(p)
        entropyOfTrueNode = nodeEntropy(proportionCategoryTrueInTrueNode) + nodeEntropy(proportionCategoryFalseInTrueNode)
        entropyOfFalseNode = nodeEntropy(proportionCategoryTrueInFalseNode) + nodeEntropy(proportionCategoryFalseInFalseNode)

        # find weights for each node
        proportionInTrueNode = totalWithFeatureTrue / inputData.shape[0]
        proportionInFalseNode = 1 - proportionInTrueNode

        # final entropy for each category and feature
        avgEntropy = entropyOfTrueNode * proportionInTrueNode + entropyOfFalseNode * proportionInFalseNode

        # Ensure any features that have data entirely in one node are not selected.
        # Entropies are constrained to be between 0 and 1, so 2 is an unused value.
        avgEntropy[(totalWithFeatureTrue == 0).repeat(avgEntropy.shape[0], axis=0)] = 2
        avgEntropy[(totalWithFeatureFalse == 0).repeat(avgEntropy.shape[0], axis=0)] = 2
        
        # return None if everything no splitting is possible
        if (avgEntropy == 2).all():
            return None

        return np.unravel_index(avgEntropy.argmin(), avgEntropy.shape)[1]

    '''
    Populates the two child nodes by splitting the data in this node into two at the right feature index.
    maxDepth: maximum number of times the tree can be split
    '''
    def split(self, maxDepth=1):
        if maxDepth <= 0:
            self.splitIndex = None
            return

        self.splitIndex = self.chooseSplitIndex()

        if self.splitIndex == None:
            return

        splitVector = self.inputData[:, self.splitIndex]
        leftIndices = np.intersect1d(np.argwhere(splitVector), self.indices)
        rightIndices = np.intersect1d(np.argwhere(1 - splitVector), self.indices)

        self.left = DecisionTreeNode(self.inputData, self.outputData, leftIndices)
        self.right = DecisionTreeNode(self.inputData, self.outputData, rightIndices)

        self.left.split(maxDepth - 1)
        self.right.split(maxDepth - 1)

    '''
    Convert decision tree into a long, human-readable string for debugging.
    split() should be called before this is called.
    wordList: list of words, in order; used to map index in vector to word
    tabs: number of tabs to prepend to each line created in the function
    '''
    def toString(self, wordList, tabs=0):
        if self.splitIndex == None:
            return ("\t" * tabs) + "return " + str(self._mostFrequentOutput()) + "\n"

        leftString = self.left.toString(wordList, tabs + 1)
        rightString = self.right.toString(wordList, tabs + 1)
        wordChecked = wordList[self.splitIndex]

        return ("\t" * tabs) + "if contains \"" + wordChecked + "\":\n" \
            + leftString \
            + ("\t" * tabs) + "else:\n" \
            + rightString

'''
Computes the entropy in one child node.
'''
def _entropyOfOneNode(proportion):
    if proportion == 0:
        return 0
    return - proportion * math.log2(proportion)

if __name__ == "__main__":
    inputData = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0]
    ])
    outputData = np.array([
        [0],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [1]
    ])
    indices = np.array(list(range(1, 15)))
    allCategories = [0, 1]
    treeNode = DecisionTreeNode(inputData, outputData, indices)
    treeNode.split()