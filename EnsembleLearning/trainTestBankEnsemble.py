'''
    Runs the adaboost for bank dataset
    @author - Angel Dhungana
'''
import ReadFiles
import AdaBoost
import copy
import statistics
import numpy as np
import Bagging
import RandomForest


def main():
    attributesValues = [
        "age", "job", "martial", "education", "default", "balance", "housing",
        "loan", "contact", "day", "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y"
    ]
    isNumeric = [
        True, False, False, False, False, True, False, False, False, True,
        False, True, True, True, True, False, False
    ]
    dataSetTrain = ReadFiles.readFromFile("bank/train.csv")
    dataSetTest = ReadFiles.readFromFile("bank/test.csv")

    # Boolean value that determines which prediction to run, for train data or test data?
    target = "y"
    # Boolean value determines if we want to fill unknown or not

    # Gets the threshold from training data set
    # uses the threshold to convert numeric attributes to binary
    threshold = getThreshold(dataSetTrain, attributesValues, isNumeric)

    dataSetTrain = getBinaryForm(dataSetTrain, attributesValues, isNumeric,
                                 threshold)
    dataSetTest = getBinaryForm(dataSetTest, attributesValues, isNumeric,
                                threshold)

    #bagging(dataSetTrain, dataSetTest, attributesValues, target)
    print()
    print("WARNING: The following tests will take considerable time to run.")
    print()
    print(
        "--------------------------------100 Bagged Predictor VS Single Learner---------------------------------"
    )
    print()
    bagging100(dataSetTrain, dataSetTest, attributesValues, target)
    print()
    print(
        "--------------------Bias, Variance and General Square Error: Single Tree VS Whole Forest-------------------"
    )
    randomForest(dataSetTrain, dataSetTest, attributesValues, target)
    print()
    #boosting(dataSetTrain, dataSetTest, attributesValues, target)


def boosting(dataSetTrain, dataSetTest, attributesValues, target):
    '''
        Run the Ada Boosting algorithm and plot the errors
    '''
    dataSetTrain = fillTargetWithNumericBinary(dataSetTrain,
                                               attributesValues.index(target))
    dataSetTest = fillTargetWithNumericBinary(dataSetTest,
                                              attributesValues.index(target))
    # Run Adaboost
    num_iterations = 100
    e1, e2, e3, e4 = AdaBoost.AdaBoost(
        dataSetTrain, dataSetTest, attributesValues, target, num_iterations)
    # Plot err
    AdaBoost.plot_error_rate(e1, e2, num_iterations)
    AdaBoost.plot_error_rate2(e3, e4, num_iterations)


def randomForest(dataSetTrain, dataSetTest, Attributes, target):
    '''
        Run the random forest algorithm and print the bias, varaince for various subsets
    '''
    dataSetTrain = fillTargetWithNumericBinary(dataSetTrain,
                                               Attributes.index(target))
    dataSetTest = fillTargetWithNumericBinary(dataSetTest,
                                              Attributes.index(target))
    num_iterations = 100
    num_trees = 1000
    subset = [2, 4, 6]
    ps = []
    print()
    # Make forest with each subset size and prints its variance, bias infor
    for su in subset:
        hypothesis = RandomForest.RandomForest(dataSetTrain, Attributes,
                                               target, num_iterations, su)

        bag = RandomForest.calculateFinalPrediction(
            hypothesis, dataSetTrain, dataSetTest, Attributes, target)
        trainPred = bag[0]
        testPred = bag[1]
        print("TRAIN, Feature Subset Size = " + str(su))
        print("Single Bias = " + str(RandomForest.bias(trainPred[:2], [1])))
        print("Single Variance = " + str(RandomForest.variance(trainPred[:2])))
        RandomForest.printBiasVariance(trainPred, dataSetTrain,
                                       Attributes.index(target))
        print()
        print("TEST, Feature Subset Size = " + str(su))
        print("Single Bias = " + str(RandomForest.bias(testPred[:2], [1])))
        print("Single Variance = " + str(RandomForest.variance(testPred[:2])))
        RandomForest.printBiasVariance(testPred, dataSetTest,
                                       Attributes.index(target))
        print()
        ps.append(trainPred)
        ps.append(testPred)
    # Plot the error rate for all 6 forests
    #RandomForest.plot_error_rate(ps[0], ps[2], ps[4], ps[1], ps[3], ps[5],
    #num_iterations)


def bagging100(dataSetTrain, dataSetTest, Attributes, target):
    '''
        Run the bagging100 to get 100 classifiers with 1000 trees each
    '''
    dataSetTrain = fillTargetWithNumericBinary(dataSetTrain,
                                               Attributes.index(target))
    dataSetTest = fillTargetWithNumericBinary(dataSetTest,
                                              Attributes.index(target))
    num_iterations = 100
    num_predictors = 100
    # Get and Print the bias,variance and general error for single versus multiple error
    Bagging.Bagging100(dataSetTrain, dataSetTest, Attributes, target,
                       num_iterations, num_predictors)


def bagging(dataSetTrain, dataSetTest, attributesValues, target):
    '''
        Run the bagging algorithm with train dataset
    '''
    dataSetTrain = fillTargetWithNumericBinary(dataSetTrain,
                                               attributesValues.index(target))
    dataSetTest = fillTargetWithNumericBinary(dataSetTest,
                                              attributesValues.index(target))
    num_iterations = 1000
    hypth = Bagging.Bagging(dataSetTrain, attributesValues, target,
                            num_iterations)
    # Get final prediction for train and test and plot
    bag = Bagging.calculateFinalPrediction(hypth, dataSetTrain, dataSetTest,
                                           attributesValues, target)
    #Bagging.plot_error_rate(bag[0], bag[1], num_iterations)


def getThreshold(dataSet, Attributes, isNumeric):
    '''
        Calculates median threshold from train dataset
    '''
    thresholds = []
    for x in Attributes:
        indx = Attributes.index(x)
        numeric = isNumeric[indx]
        if numeric == True:
            listAtt = []
            for row in dataSet:
                listAtt.append(float(row[indx]))
            # calculate median a numeric attribute column
            median = statistics.median(listAtt)
            thresholds.append(median)
    return thresholds


def getBinaryForm(dataSet, Attributes, isNumeric, thresholds):
    '''
        For all the numeric attributes, gets the median of it
        And if numeric value is less than median we consider that "No"
        Else, we consider "Yes"
    '''
    newSet = copy.copy(dataSet)
    thresholdCounter = 0
    for row in newSet:
        for x in Attributes:
            indx = Attributes.index(x)
            numeric = isNumeric[indx]
            if numeric == True:
                # If value is greater than or equal to threshold, its a "Yes"
                if thresholds[thresholdCounter] <= float(row[indx]):
                    row[indx] = "Yes"
                else:
                    row[indx] = "No"
                thresholdCounter += 1
        thresholdCounter = 0
    return newSet


def fillTargetWithNumericBinary(dataSet, targetIndx):
    '''
        Change Yes/No to 1, -1 for targetIndx
    '''
    for row in dataSet:
        if row[targetIndx] == "yes":
            row[targetIndx] = 1
        else:
            row[targetIndx] = -1
    return dataSet


if __name__ == "__main__":
    main()