import math

from text_vector import TextVector

'''
A node of a decision tree that decides between categories based on TextVectors.
'''
class DecisionTreeNode:
    '''
    allData: all data vectors
    indices: indices of vectors in allData that are given to this node
    allCategories: list of all categories
    '''
    def __init__(self, allData, indices, allCategories):
        self.allData = allData
        self.indices = indices
        self.allCategories = allCategories
        self.left = None
        self.right = None

    '''
    Returns the feature index that should be used to determine which vectors go in which child nodes.
    Returns -1 if there is no way to further divide this dataset.
    '''
    def chooseSplitIndex(self):
        bestFeature = -1
        bestScore = 2.0

        for feature in range(len(self.allData[0].vector)):

            # create dictionary mapping category -> # data with feature = T
            # create dictionary mapping category -> # data with feature = F
            numWithFeatureTrue = {}
            numWithFeatureFalse = {}
            for category in self.allCategories:
                numWithFeatureTrue[category] = 0
                numWithFeatureFalse[category] = 0
            
            for dataIndex in self.indices:
                dataPoint = self.allData[dataIndex]
                if dataPoint.vector[feature]:
                    numWithFeatureTrue[dataPoint.category] += 1
                else:
                    numWithFeatureFalse[dataPoint.category] += 1

            # find total number with feature True and False
            totalFeatureTrue = 0
            totalFeatureFalse = 0
            for category in self.allCategories:
                totalFeatureTrue += numWithFeatureTrue[category]
                totalFeatureFalse += numWithFeatureFalse[category]

            # if the feature has all of the categories in one side, don't use it
            if totalFeatureTrue == 0 or totalFeatureFalse == 0:
                continue

            # calculate entropy using each category as a binary classifier
            for category in self.allCategories:
                proportionCategoryTrueInTrueNode = numWithFeatureTrue[category] / totalFeatureTrue
                proportionCategoryFalseInTrueNode = 1 - proportionCategoryTrueInTrueNode
                proportionCategoryTrueInFalseNode = numWithFeatureFalse[category] / totalFeatureFalse
                proportionCategoryFalseInFalseNode = 1 - proportionCategoryTrueInFalseNode

                entropyOfTrueNode = _entropyOfOneNode(proportionCategoryTrueInTrueNode) + _entropyOfOneNode(proportionCategoryFalseInTrueNode)
                entropyOfFalseNode = _entropyOfOneNode(proportionCategoryTrueInFalseNode) + _entropyOfOneNode(proportionCategoryFalseInFalseNode)

                proportionInTrueNode = totalFeatureTrue / len(self.indices)
                proportionInFalseNode = 1 - proportionInTrueNode

                weightedAverageEntropy = proportionInTrueNode * entropyOfTrueNode + proportionInFalseNode * entropyOfFalseNode

                if weightedAverageEntropy < bestScore:
                    bestScore = weightedAverageEntropy
                    bestFeature = feature

        return bestFeature

'''
Computes the entropy in one child node.
'''
def _entropyOfOneNode(proportion):
    if proportion == 0:
        return 0
    return - proportion * math.log2(proportion)

if __name__ == "__main__":
    wordIndices = { "sunny": 0, "hot": 1, "humid": 2, "windy": 3 }
    allData = [ 
        TextVector("sunny hot humid windy", True, wordIndices),
        TextVector("sunny hot humid windy", True, wordIndices),
        TextVector("sunny hot humid windy", True, wordIndices),
        TextVector("sunny hot humid windy", False, wordIndices),
        TextVector("sunny hot humid windy", False, wordIndices),
        TextVector("sunny hot humid windy", False, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", True, wordIndices),
        TextVector("sunny hot humid", False, wordIndices),
        TextVector("sunny hot humid", False, wordIndices)
    ]
    indices = list(range(len(allData)))
    allCategories = [True, False]
    treeNode = DecisionTreeNode(allData, indices, allCategories)
    print(treeNode.chooseSplitIndex())