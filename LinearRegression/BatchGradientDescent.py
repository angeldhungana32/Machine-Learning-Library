'''
    @author - Angel Dhungana
    Batch Gradient Descent
'''
import numpy as np
import matplotlib.pyplot as plt
import copy


def batchGradientDescent(rate, x, y, theta, num_iterations, tolerance):
    '''
        Gradient Descent classifier, returns the weights 
            rate - learning rate
            x - X dataset
            y - y Label actual prediction
            theta - weights, intialized to be 0
            num_iterations - max number of iterations
            toleracen - tolerance rate to find the convergence
    '''
    # temp matrix of weights
    temp = np.matrix(np.zeros(theta.shape))
    # Each Param of weights
    wInd = int(theta.ravel().shape[1])
    # Initialize cost arr with zero
    costArr = []
    # Loop until the max_iterations
    for i in range(0, num_iterations):
        # calculate loss
        xTw = x * theta.T
        xTw = y - xTw
        # compute cost
        # Keep copy of previous weights
        prev = copy.deepcopy(theta)
        # Calculate the weights
        for j in range(wInd):
            temp[0, j] = theta[0, j] - (
                (rate / len(x)) * np.sum(np.multiply(xTw, x[:, j])))
        theta = temp
        cost = computeCost(x, y, theta)
        costArr.append(cost)
        # Normalize theta - prevTheta
        norm1 = np.linalg.norm((theta - prev))
        # If it is less than tolerancem we know it converges so break out
        if norm1 < tolerance:
            return theta, costArr, rate, True
    # Return theta, costArray and convergance
    return theta, costArr, rate, False


def computeCost(X, y, w):
    '''
        Compute cost using weights
    '''
    inner = np.power((y - (X @ w.T)), 2)
    return np.sum(inner) / (2 * len(X))


def getError(X, Y, w):
    '''
        Returns the error for Test using weights
    '''
    sign = np.sign(X @ w.T)
    count = 0
    for i in range(len(Y)):
        if sign[i] != Y[i]:
            count += 1
    return count / len(Y)


def plotCostFunction(num_iterations, costArr):
    '''
        Save the cost function plot
    '''
    #arr = [x + 1 for x in range(len(costArr))]
    #plt.plot(arr, costArr, linewidth=1)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Function of Train', fontsize=16)
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('costBatch.pdf', bbox_inches='tight')


def plotSingle(costArr, rate):
    '''
        Plot each costFunction result
    '''
    arr = [x + 1 for x in range(len(costArr))]
    plt.plot(arr, costArr, linewidth=1, label="rate: " + str(rate))