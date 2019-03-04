'''
    @author - Angel Dhungana
    Stochastic Gradient Descent Implementation
'''
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


def StochasticGradientDescent(rate, x, y, num_iterations, tolerance):
    '''
        Takes X and y with rate and returns the weight when the algorithm converges
        rate - learning rate
        x - X dataset
        y - y Label
        num_interations - maximum number of iterations
        tolerance - precision rate till when to stop 
    '''
    shape = x.shape
    # Inserts the ones colum, for the bias
    x = np.insert(x, 0, 1, axis=1)
    # Inserts weights for each feature and add bias weight
    w = np.ones((shape[1] + 1))
    costArr = []
    prev = 0
    # Loop until max iteration
    for jj in range(num_iterations):
        # for each feature, calculate the weight and add it to weight arr
        for ix, i in enumerate(x):
            gradient = np.dot(w.T, i)
            # take dot product
            if gradient > 0: gradient = 1
            elif gradient < 0: gradient = -1
            # Calculate loss
            #if gradient != y[ix]:
            w = w - rate * ((y[ix] - gradient) * i * -1)
        # Calculate cost
        cost = computeCost(x, y, w)
        costArr.append(cost)
        if abs(cost - prev) < tolerance:
            break
        prev = copy.deepcopy(cost)
        # decrease the rate each 10 time
        if jj % 20 == 0:
            rate = rate / 2
        # If cost is 0, break
        if cost == 0:
            break
    return w, costArr, rate


def computeCost(x, y, w):
    '''
        Compute the cost using the weights 
    '''
    loss = (np.dot(x, w) - y)**2
    loss = np.sum(loss) / (len(x) * 2)
    return loss


def plotCostFunction(costArr):
    '''
        Plot the cost fucntion of the stochastic gradient 
    '''
    arr = [x + 1 for x in range(len(costArr))]
    plt.plot(arr, costArr, linewidth=1)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Cost Function of Train', fontsize=16)
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('costStochastic.pdf', bbox_inches='tight')