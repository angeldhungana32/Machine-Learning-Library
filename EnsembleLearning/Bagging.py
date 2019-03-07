'''
    @author - Angel Dhungana
    Bagging Implementation
'''
import DecisionTree
import random
from random import randrange
import matplotlib.pyplot as plt
import numpy as np


def Bagging(dataSetTrain, Attributes, target, num_iterations):
    '''
        Make num_iterations of  bagged trees
            dataSetTrain - Dataset to sample from
            Attributes - att for Decision Tree
            target - target Label
            num_iterations - Max number of bagged trees to make
    '''
    hypotheses = []
    for i in range(num_iterations):
        # draw sample with replacement
        mSample = drawMSample(dataSetTrain)
        tree = DecisionTree.ID3(mSample, Attributes, target, 0, 20, None, None)
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

    # Get the Mean of using 1 bagged, 2 bagged, ... till 1000 bagged
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


def getTestError(TestPred, ActualY):
    count = 0
    for i in range(TestPred):
        if TestPred[i] != ActualY[i]:
            count += 1
    return count / len(TestPred)


def Bagging100(dataSetTrain, dataSetTest, Attributes, target, num_iterations,
               num_predictors):
    '''
        Make Bagged Prediction with 100 predictors, each with 1000 trees
    '''
    # Make 100 bagged Predictions
    bagged100Predictors = []
    for i in range(num_predictors):
        hypotheses = Bagging(dataSetTrain, Attributes, target, num_iterations)
        bagged100Predictors.append(hypotheses)

    # Get predictions using a single classifier
    singlePredictions = singleLearner(bagged100Predictors, dataSetTest,
                                      Attributes, target)
    y = getY(dataSetTest, Attributes.index(target))

    # Calculate Bias, Variance and General Squared Error for Single Learner and Print
    firstBias = bias(singlePredictions, y)
    firstVariance = variance(singlePredictions)
    firstGeneralErr = firstBias + firstVariance
    print("Single Learner, Single Tree -> Bias = " + str(firstBias) +
          ", Variance = " + str(firstVariance) + ", Err = " +
          str(firstGeneralErr))
    print()
    # Get predictions using 100 bagged classifier
    allPredictions = multiLearner(bagged100Predictors, dataSetTest, Attributes,
                                  target)
    # Calculate Bias, Variance and General Squared Error for 100 classifiers
    bia = bias(allPredictions, y)
    varia = variance(allPredictions)
    geneErr = firstBias + firstVariance
    print("Multi Learner 100 Bagged -> Bias = " + str(bia) + ", Variance = " +
          str(varia) + ", Err = " + str(geneErr))
    print()


def getY(dataSetTest, indx):
    '''
        Gets the Y column from dataset
    '''
    y = []
    for x in dataSetTest:
        y.append(x[indx])
    return y


def singleLearner(baggerd100Pred, dataSetTest, Attributes, target):
    '''
        Gets Prediction error of single classifier
    '''
    predictions = []
    for x in baggerd100Pred:
        tree = x[0]
        predictions.append(
            getPredictError(tree, dataSetTest, Attributes, target))
    return predictions


def multiLearner(bagged100Predictors, dataSetTest, Attributes, target):
    '''
        Gets averaged out prediction error from many predictors
    '''
    predictions = []
    for x in bagged100Predictors:
        pred = []
        for y in x:
            pred.append(getPredictError(y, dataSetTest, Attributes, target))
        meanN = mean(pred)
        predictions.append(meanN)
    return predictions


def bias(predictions, x):
    '''
        Calculates the bias of the predictions
    '''
    x = np.array(x)
    meanN = mean(predictions)
    x = np.exp(-x**2) + 1.5 * np.exp(-(x - 2)**2)
    x = sum(x) / len(x)
    return (meanN - x)**2


def variance(predictions):
    '''
        Calculates Variance of the predictors
    '''
    return np.var(predictions)


def drawMSample(dataset, ratio=0.5):
    '''
        Gets sample -> dataset * ratio, with replacement
    '''
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def drawMSampleRepl(dataset, ratio=0.2):
    '''
        Gets sample -> dataset * ratio, without replacement
    '''
    sample = list()
    randomSample = random.sample(range(5000), 1000)
    for index in randomSample:
        sample.append(dataset[index])
    return sample


def mean(numbers):
    '''
        Calculates mean of numbers
    '''
    return sum(numbers) / float(len(numbers))


def plot_error_rate(er_train, er_test, n):
    '''
        Plots the error for train and test
    '''
    #arr = [x for x in range(n)]
    arr = list(range(10, 1010, 10))
    plt.plot(
        arr, er_train, linewidth=3, color='lightblue', label="Train Error")
    plt.plot(arr, er_test, linewidth=3, color='darkblue', label="Test Error")
    plt.xlabel('Number of Trees', fontsize=12)
    plt.ylabel('Error rate', fontsize=12)
    plt.title('Error rate vs number of Trees', fontsize=16)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('errorBagging.pdf', bbox_inches='tight')


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
