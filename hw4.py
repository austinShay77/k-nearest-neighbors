import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import os
from scipy.stats import mode
from PIL import Image
from tree import DecisionTree, Node

CTG_DATA_PATH = "CTG.csv"
YALEFACES_PATH = "./yalefaces"
np.random.seed(0)

def loadData(path):
    if os.path.isdir(path):
        data = np.empty(shape=[0, 1601])
        for file in os.listdir(path):
            if not file.endswith(".txt"):
                name = file.split(".")[0]
                classifier = name[-1]
                img_path = os.path.join(path, file)
                img = Image.open(img_path)
                subsampled = img.resize((40, 40))
                img_array = np.array(subsampled)
                flat = img_array.flatten()
                flat = np.append(flat, classifier)
                data = np.append(data, [flat], axis=0) 
        data = data.astype(int)
        np.random.shuffle(data)
        seperate = np.array_split(data, 3)
        train, validation = np.concatenate((seperate[0], seperate[1])), seperate[2]
    elif path.endswith(".csv"):
        # remove empty row and titles
        data = np.genfromtxt(path, delimiter=',', skip_header=2)
        # remove CLASS feature
        data = np.delete(data, data.shape[1]-2, axis=1)
        np.random.shuffle(data)
        seperate = np.array_split(data, 3)
        train, validation = np.concatenate((seperate[0], seperate[1])), seperate[2]
    #      xtrain                                ytrain                               xvalid                                               yvalid
    return train[:len(train), :len(train[0])-1], train[:len(train), len(train[0])-1], validation[:len(validation), :len(validation[0])-1], validation[:len(validation), len(validation[0])-1] 
    
def standardize(d):
    return (d-np.mean(d, axis=0)) / np.std(d, axis=0, ddof=1)

def euclideanDistance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def createConfusionMatrix(actual, predicted):
    classes = np.unique(actual)
    confusionMatrix = np.zeros((len(classes), len(classes)))

    for i in range(len(classes)):
        for j in range(len(classes)):
            confusionMatrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confusionMatrix

def categorize(data):
    categorized = []
    mean = np.mean(data, axis=0)
    for i in range(len(data[0])):
        currentFeature = data[:, i]
        category = np.where(currentFeature > mean[i], 1, 0)
        categorized.append(category)
    return np.array(categorized).transpose()

def myKNN(xtrain, ytrain, xvalid, k):
    predictions = []
    for validRow in xvalid:
        distances = []

        for i in range(len(xtrain)):
            d = euclideanDistance(np.array(xtrain[i, :]), validRow)
            distances.append(d)
        distances = np.array(distances)

        neighborsId = np.argsort(distances)[:k]
        neighbors = ytrain[neighborsId]

        majority = mode(neighbors, keepdims=True)
        majority = majority.mode[0]
        predictions.append(majority)
    return np.array(predictions)
          
def KNN():
    xtrain, ytrain, xvalid, yvalid = loadData(YALEFACES_PATH)
    yhat = myKNN(standardize(xtrain), ytrain, standardize(xvalid), 30)
    validationAccuracy = np.mean(yhat == yvalid)
    print("Validation Accuracy:", validationAccuracy)
    confustionMatrix = createConfusionMatrix(yvalid, yhat)
    classLabels = []
    ytrain = ytrain.astype(int)
    for i in set(ytrain):
        name = "Class " + str(i)
        classLabels.append(name)
    sn.heatmap(confustionMatrix, annot=True, cmap="Blues", fmt=".5g", xticklabels=classLabels, yticklabels=classLabels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # examples == [X,Y], attributes == list of possible col values indices, default is Y
def myDT(examples, attributes, default):
    treeMath = DecisionTree()
    if len(examples[0]) == 0:
        return default[0]
    elif len(set(examples[:, -1])) == 1:
        return examples[0, -1]
    elif len(attributes) == 0:
        return mode(examples, keepdims=True)[0][0]
    else:
        best, attributeIndex = treeMath.chooseFeature(examples[:, :-1], examples[:, -1])
        tree = Node(attributeIndex)
        for value in set(best):
            subtree = myDT(np.delete(treeMath.grabRowElements(examples[:, :-1], best, value), attributeIndex, axis=1), np.delete(attributes, attributeIndex), mode(examples[:, -1], keepdims=True)[0])
            tree.insert(subtree, edgeValue=value)
        return tree

def DT():
    xtrain, ytrain, xvalid, yvalid = loadData(CTG_DATA_PATH)
    trainCategoricalX, validCategoricalX = categorize(xtrain), categorize(xvalid)
    examples = np.concatenate((trainCategoricalX, ytrain.reshape((-1, 1))), axis=1)
    attributes = np.array([i for i in range(len(xtrain[0]))])
    yhat = myDT(examples, attributes, ytrain)
    yhat.PrintTree()

if __name__ == "__main__":
    if not os.path.exists('./temp'):
        os.makedirs("./temp")
    KNN()
    # DT()
    os.rmdir("./temp")

    # root = Node(22)
    # root.insert(2, 1)
    # root.insert(3, 0)
    # root.insert(1, 1)
    # root.insert(1, 0)
    # root.PrintTree()