'''
    @author - Angel Dhungana
    Logistic Regression Algorithm Implementation
'''
import numpy as np
import random
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def stochastic_logistic_map(X, Y, r, t, v, d):
    '''
        Standard Perception Algorithm
    '''
    shape = X.shape
    w = np.ones((shape[1]))
    N = len(X)
    for i in range(1, t + 1):
        # shuffle data
        X, Y = shuffleData(X, Y)
        rate = r / ((1 + r / d) * i)
        for j in range(N):
            gradient = _gradient(X[j], Y[j], w, v)
            w = w - rate * gradient
    return w


def stochastic_logistic_max_est(X, Y, r, t, v, d):
    '''
        Standard Perception Algorithm
    '''
    shape = X.shape
    w = np.ones((shape[1]))
    N = len(X)
    for i in range(1, t + 1):
        # shuffle data
        X, Y = shuffleData(X, Y)
        rate = r / ((1 + r / d) * i)
        for j in range(N):
            gradient = _gradient2(X[j], Y[j], w, v)
            w = w - rate * gradient
    return w


def prediction(X, Y, w):
    Y_pred = []
    for j in range(len(X)):
        pred = 1 / (1 + np.exp(-(w.T @ X[j])))
        if pred > 0.5:
            Y_pred.append(1)
        else:
            Y_pred.append(-1)
    s = np.sum(Y != np.array(Y_pred))
    return s / len(Y)


def _gradient(x, y, w, v):
    top = -x * y
    bottom = 1 + np.exp(y * w.T @ x)
    right = (2 * w) / (2 * v)
    return top / bottom + right


def _gradient2(x, y, w, v):
    top = -x * y * v
    bottom = 1 + np.exp(y * w.T @ x)
    right = (2 * w) / (2 * v)
    return top / bottom


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