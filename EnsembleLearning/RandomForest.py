'''
 @author - Angel Dhungana
 Random Forest Implementation
'''
import DecisionTree
import random
from random import randrange
import matplotlib.pyplot as plt
import numpy as np


def RandomForest(dataSetTrain, Attributes, target, num_iterations,
                 featureSubset):
    '''
        Generate Random Forest upto num_iterations, using featureSubset,
            dataSetTrain - Dataset to sample from
            Attributes - att for Decision Tree
            target - target Label
            num_iterations - Max number of forest to make
            featureSubset - size of feature subset we want to split on
    '''
    hypotheses = []
    for i in range(num_iterations):
        # Get sample
        mSample = drawMSample(dataSetTrain)
        # Make tree
        tree = DecisionTree.ID3(mSample, Attributes, target, 4, 20, None,
                                featureSubset)
        hypotheses.append(tree)
    return hypotheses


def calculateFinalPrediction(hypotheses, dataSetTrain, dataSetTest, Attributes,
                             target):
    '''
        Get the average of predictions from all 1000 forests, for train and test dataset
    '''
    predictionsTrain = []
    predictionsTest = []
    # Get predictions error for test and train using 1000 Random Forests
    for sampleTree in hypotheses:
        pred1 = getPredictError(sampleTree, dataSetTrain, Attributes, target)
        pred2 = getPredictError(sampleTree, dataSetTest, Attributes, target)
        predictionsTrain.append(pred1)
        predictionsTest.append(pred2)

    # Get the Mean of using 1 forest, 2 forest, ... till 1000 forests
    finalPredictionTrain = []
    finalPredictionTest = []
    # Loop and take mean for train
    for i in range(len(predictionsTrain)):
        meanN = mean(predictionsTrain[:(i + 1)])
        finalPredictionTrain.append(meanN)
    # Loop and take mean for test
    for i in range(len(predictionsTest)):
        meanN = mean(predictionsTest[:(i + 1)])
        finalPredictionTest.append(meanN)
    # Return final Prediction
    return finalPredictionTrain, finalPredictionTest


def mean(numbers):
    '''
        Returns the mean of the array
    '''
    return sum(numbers) / float(len(numbers))


def bias(predictions, x):
    '''
        Calculate bias from predictions
    '''
    x = np.array(x)
    meanN = mean(predictions)
    x = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2)
    x = sum(x) / len(x)
    return (meanN - x)**2


def variance(predictions):
    '''
        Calculate Variance
    '''
    return np.var(predictions)


def getY(dataSetTest, indx):
    '''
        Returns the Y dataset
    '''
    y = []
    for x in dataSetTest:
        y.append(x[indx])
    return y


def drawMSample(dataset, ratio=0.5):
    '''
        Draw random sample from dataset without replacement
    '''
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def printBiasVariance(prediction, dataSetTest, tarIndx):
    '''
        Calculate Bias and Variance from the predictino
    '''
    firstBias = bias(prediction, getY(dataSetTest, tarIndx))
    firstVariance = variance(prediction)
    firstGeneralErr = firstBias + firstVariance
    print("Random Forest -> Bias = " + str(firstBias) + ", Variance = " +
          str(firstVariance) + ", General Err = " + str(firstGeneralErr))


def plot_error_rate(er1, er2, err3, err4, err5, err6, n):
    '''
        Plot error rate for six forests
    '''
    arr = list(range(10, 1010, 10))
    plt.plot(arr, er1, linewidth=3, color='red', label="Train, 2")
    plt.plot(arr, er2, linewidth=3, color='blue', label="Train, 4")
    plt.plot(arr, err3, linewidth=3, color='green', label="Train, 6")
    plt.plot(arr, err4, linewidth=3, color='pink', label="Test, 2")
    plt.plot(arr, err5, linewidth=3, color='orange', label="Test, 4")
    plt.plot(arr, err6, linewidth=3, color='gray', label="Test, 6")
    plt.xlabel('Number of Forests', fontsize=12)
    plt.ylabel('Error rate', fontsize=12)
    plt.title('Error rate vs number of Forests', fontsize=16)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('errorForestTest.pdf', bbox_inches='tight')


def getPredictError(tree, dataset, Attributes, target):
    '''
        Traverses the tree and get the correct number of predictions for dataset
    '''
    correctPrediction = 0
    targetIndx = Attributes.index(target)
    # For every row
    for row in dataset:
        curr = tree
        # If tree has branches, meaning its not a leaf
        x = curr.has_child()
        # loop until we hit leaf
        while x == True:
            indx = Attributes.index(curr.data)
            # get value from row
            value = row[indx]
            # get branch that has the value
            branch = curr.get_child(value)
            # if not such branch exist we break
            if branch == None:
                break
            # get the Node from branch
            curr = branch.get_BranchNode()
            x = curr.has_child()
        # if leaf value matches the actual target
        if curr.data == row[targetIndx]:
            correctPrediction += 1
    return 1 - (correctPrediction / len(dataset))