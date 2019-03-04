'''
    Testing Batch Gradient and Stochastical Gradient Descent
    @author - Angel Dhungana
'''
import ReadFiles
import copy
import statistics
import numpy as np
import BatchGradientDescent
import pandas as pd
import StochasticGradient
import AnalyticalLMS


def runBatch():
    '''
        Runs the Batch Gradient Descent linear regressor
    '''
    #attributesValues = [
    #   "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr"
    #]
    # Read concrete data set
    dataSetTrain = pd.read_csv("concrete/train.csv")
    dataSetTest = pd.read_csv("concrete/test.csv")

    # Add ones column to the front of dataSet both test and Train
    dataSetTrain.insert(0, "ones", 1)
    dataSetTest.insert(0, "ones", 1)
    cols1 = dataSetTrain.shape[1]
    x1 = dataSetTrain.iloc[:, 0:cols1 - 1]
    y1 = dataSetTrain.iloc[:, cols1 - 1:cols1]
    x1 = np.matrix(x1.values)
    y1 = np.matrix(y1.values)
    cols2 = dataSetTest.shape[1]
    x2 = dataSetTest.iloc[:, 0:cols2 - 1]
    y2 = dataSetTest.iloc[:, cols2 - 1:cols2]
    x2 = np.matrix(x2.values)
    y2 = np.matrix(y2.values)

    # Initialize theta with bias
    theta2 = np.matrix(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    # learning rate
    rate = 1
    num_iterations = 1000
    tolerance = 0.000001
    counter = 0
    while rate > 0:
        # Runs the Gradient Descent
        theta, cost, rate2, bb = BatchGradientDescent.batchGradientDescent(
            rate, x1, y1, theta2, num_iterations, tolerance)
        #if counter > 8:
        #BatchGradientDescent.plotSingle(cost[::-1], rate2)
        counter += 1
        rate = rate / 2
        if bb == True:
            break
    print()
    print(
        "First I am printing results for LMS implementations because this computes faster. "
    )
    print()
    print(
        "-------------------------------- Various LMS Implemenentaions ---------------------------------"
    )
    print()
    print(
        "---------------------------------Batch Gradient Descent(Concrete DataSet)-------------------------------------"
    )
    print("Learning Rate = " + str(rate2))
    theta = np.array(theta)
    print("Weights = " + str(theta))
    costTest = BatchGradientDescent.computeCost(x2, y2, theta)
    print("Test Cost = " + str(round(costTest, 3)))
    print()
    print()


def runStochastic():
    '''

    '''
    #attributesValues = [
    #    "Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr"
    #]
    # Read dataset from the train/test dataset
    dataSetTrain = np.loadtxt("concrete/train.csv", delimiter=',')
    dataSetTest = np.loadtxt("concrete/test.csv", delimiter=',')
    # Separate X and Y Train data
    x1 = dataSetTrain[:, 0:-1]
    y1 = dataSetTrain[:, -1]
    # Separate X and Y Test data
    x2 = dataSetTest[:, 0:-1]
    y2 = dataSetTest[:, -1]
    # learning rate start from 1
    rate = 1
    num_iterations = 1000
    tolerance = 0.00001
    # Stochastic Gradient Descent classifier
    theta, cost, bb = StochasticGradient.StochasticGradientDescent(
        rate, x1, y1, num_iterations, tolerance)
    # plot cost
    #StochasticGradient.plotCostFunction(cost)
    print()
    print(
        "---------------------------------Stochastic Gradient(Concrete DataSet)-------------------------------------"
    )
    print("Learning Rate = " + str(bb))
    theta2 = [round(x, 2) for x in theta]
    print("Weights = " + str(theta2))
    # Add ones to the test x data and compute cost function
    x2 = np.insert(x2, 0, 1, axis=1)
    costTest = StochasticGradient.computeCost(x2, y2, theta)
    # Print test Cost
    print("Test Cost = " + str(round(costTest, 3)))
    print()


def runAnalytical():
    dataSetTrain = np.loadtxt("concrete/train.csv", delimiter=',')
    x1 = dataSetTrain[:, 0:-1]
    x1 = np.insert(x1, 0, 1, axis=1)
    y1 = dataSetTrain[:, -1]
    x1 = np.matrix(x1.tolist())
    y1 = np.matrix(y1.tolist())
    weights = AnalyticalLMS.analytical(x1, y1)
    print()
    print(
        "---------------------------------Analytical LMS(Concrete DataSet)-------------------------------------"
    )
    theta = [round(x[0], 2) for x in weights.tolist()]
    print("Weights = " + str(theta))
    print()


if __name__ == "__main__":
    runBatch()
    runStochastic()
    runAnalytical()
