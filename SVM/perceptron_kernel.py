import numpy as np
import random


def shuffleData(X, Y):
    '''
        Randomly Shuffles the data
    '''
    randomSample = random.sample(range(len(X)), len(X))
    newX = []
    newY = []
    for i in range(len(X)):
        newX.append(X[randomSample[i]])
        newY.append(Y[randomSample[i]])
    return np.array(newX), np.array(newY)


def perceptron(X, Y, r, t):
    '''
        Perceptron with kernel
    '''
    N = len(X)
    c = np.zeros(N)
    for _ in range(t):
        # shuffle data
        X, Y = shuffleData(X, Y)
        for j in range(N):
            dot = np.sign(np.sum((c[j] * Y[j]) * kernel(X[j], X, r)))
            if Y[j] != dot:
                c[j] += 1
    return c


def pred(c, X, Y, r):
    '''
        Prediction 
    '''
    dots = []
    for j in range(len(X)):
        dot = np.sign(np.sum((c[j] * Y[j]) * kernel(X[j], X, r)))
        dots.append(dot)
    counter = 0
    for i in range(len(Y)):
        if dots[i] != Y[i]:
            counter += 1
    return counter / len(Y)


def kernel(x_j, X, r):
    '''
        Kernel Function
    '''
    return np.exp(-1.0 * (np.linalg.norm(X - x_j)**2 / r))
