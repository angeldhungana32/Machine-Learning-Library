'''
    @author - Angel Dhungana

    This is where you can run my LMS implementation
        - Run Batch Gradient 
        - Run Stochastic Gradient

    All you need to do is modify necessary parameters inside main() function and run the method:
        - It will print calculated weight, cost of test data and Error Rate on Test Data
'''
import numpy as np
import pandas as pd
import BatchGradientDescent
import StochasticGradient


def main():
    '''
        Change necessary params here
    '''
    # Enter filename or path to file here, make sure its a csv file with both x and y column
    dataSetTrain = ""
    dataSetTest = ""

    # Set necessary params
    learning_rate = 0.001  #default
    num_iterations = 1000
    tolerance = 0.000006  #default

    # Runs batch
    XB1, YB1, XB2, YB2 = getBatchData(dataSetTrain, dataSetTest)
    runBatch(XB1, YB1, XB2, YB2, learning_rate, num_iterations, tolerance)

    # Runs Stochastic
    XS1, YS1, XS2, YS2 = getStochasticData(dataSetTrain, dataSetTest)
    runStochastic(XS1, YS1, XS2, YS2, learning_rate, num_iterations, tolerance)


def runBatch(X1, Y1, X2, Y2, rate, num_iterations, tolerance):
    # Initialize theta with bias
    shape = X1.shape
    theta2 = np.matrix(np.zeros(shape[1]))
    theta, _, _, _ = BatchGradientDescent.batchGradientDescent(
        rate, X1, Y1, theta2, num_iterations, 0)
    theta = np.array(theta)
    print(
        "---------------------------------Batch Gradient-------------------------------------"
    )
    print()
    print("Weights = " + str(theta))
    print()
    costTest = BatchGradientDescent.computeCost(X2, Y2, theta)
    print("Cost Function for Test Data = " + str(round(costTest, 3)))
    print()


def runStochastic(X1, Y1, X2, Y2, rate, num_iterations, tolerance):
    theta, _, _ = StochasticGradient.StochasticGradientDescent(
        rate, X1, Y1, num_iterations, tolerance)
    print()
    print(
        "---------------------------------Stochastic Gradient-------------------------------------"
    )
    print("Weights = " + str(theta))
    X2 = np.insert(X2, 0, 1, axis=1)
    costTest = StochasticGradient.computeCost(X2, Y2, theta)
    # Print test Cost
    print()
    print("Cost Function for Test Data = " + str(round(costTest, 3)))
    print()


def getBatchData(fileNameTrain, fileNameTest):
    '''
        Returns x and y for train and test data as matrix
        This is for Batch Data
    '''
    dataSetTrain = pd.read_csv(fileNameTrain)
    dataSetTest = pd.read_csv(fileNameTest)

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
    return x1, y1, x2, y2


def getStochasticData(fileNameTrain, fileNameTest):
    '''
        Returns x and y for train and test data
        This is for stochastic Data
    '''
    dataSetTrain = np.loadtxt(fileNameTrain, delimiter=',')
    dataSetTest = np.loadtxt(fileNameTest, delimiter=',')
    # Separate X and Y Train data
    x1 = dataSetTrain[:, 0:-1]
    y1 = dataSetTrain[:, -1]
    # Separate X and Y Test data
    x2 = dataSetTest[:, 0:-1]
    y2 = dataSetTest[:, -1]
    return x1, y1, x2, y2


if __name__ == "__main__":
    main()
