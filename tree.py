import math
import numpy as np

class Node:
    def __init__(self, featureNum):    
        self.left = None
        self.right = None
        self.featureNum = featureNum 

    def insert(self, newRoot, edgeValue):
        if self.featureNum:
            if edgeValue == 1:
                if self.left is None:
                    self.left = Node(newRoot)
                else:
                    self.left.insert(newRoot, edgeValue)
            elif edgeValue == 0:
                if self.right is None:
                    self.right = Node(newRoot)
                else:
                    self.right.insert(newRoot, edgeValue)
        else:
            self.featureNum = newRoot
    
    def PrintTree(self):
      if self.left:
         self.left.PrintTree()
      print( self.featureNum),
      if self.right:
         self.right.PrintTree()

class DecisionTree:
    def __init__(self):
        pass
    
    # returns best feature as a list
    def chooseFeature(self, X, Y):
        best = []
        entropies = []
        entropy = math.inf
        for feature in X.transpose():
            currentFeatureEntropy = self._calculateWeightedEntropy(feature, Y)
            entropies.append(currentFeatureEntropy)
            if currentFeatureEntropy < entropy:
                entropy = currentFeatureEntropy
                best = feature
        return best, np.array(entropies).argmin()
            
    # takes in a single feature
    def _calculateWeightedEntropy(self, x, Y):
        totalSamples = len(x)
        uniqueClasses = set(Y)
        weightedEntropy = 0
        edgeCase = 1e-20

        for targetClass in uniqueClasses:
            classSamples = [observation for observation, target in zip(x, Y) if target == targetClass]
            classProbability = len(classSamples) / totalSamples
            positive = sum(classSamples)
            negative = len(classSamples) - positive
            positiveProbability = positive / len(classSamples)
            negativeProbability = negative / len(classSamples)
            classEntropy = -(positiveProbability*math.log2(positiveProbability+edgeCase) + negativeProbability*math.log2(negativeProbability+edgeCase))
            weightedEntropy += classProbability * classEntropy
        return weightedEntropy
    
    def grabRowElements(self, data, feature, value):
        indices = np.where(feature==value)[0]
        return data[indices]
    