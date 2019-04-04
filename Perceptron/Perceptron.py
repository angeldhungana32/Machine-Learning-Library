'''
    @author - Angel Dhungana
    Perception Algorithm Implementation
'''
import numpy as np
import random


def StandardPerception(X, Y, r, t):
    '''
        Standard Perception Algorithm
            X - X train data
            Y - Y train data
            r - learning rate
            t - epochs or max iters

        Returns the final weights
    '''
    shape = X.shape
    w = np.ones((shape[1]))
    N = len(X)
    for i in range(t):
        # shuffle data
        X, Y = shuffleData(X, Y)
        for j in range(N):
            dot = (w.T @ X[j]) * Y[j]
            if dot <= 0:
                w = w + (r * (X[j] * Y[j]))
    return w


def VotedPerception(X, Y, r, t):
    '''
        Voted Perception Algorithm
            X - X train data
            Y - Y train data
            r - learning rate
            t - epochs or max iters

        Returns the final weights and count of incorrect predictions
    '''
    shape = X.shape
    w = [np.ones((shape[1]))]
    c = [0]
    m = 0
    N = len(X)
    for i in range(t):
        for j in range(N):
            dot = np.dot(w[m].T, X[j]) * Y[j]
            if dot <= 0:
                w.append(w[m] + (r * np.dot(X[j], Y[j])))
                m = m + 1
                c.append(1)
            else:
                c[m] = c[m] + 1
    return w, c


def AveragedPerception(X, Y, r, t):
    '''
        Averaged Perception Algorithm
            X - X train data
            Y - Y train data
            r - learning rate
            t - epochs or max iters

        Returns the final averaged weights
    '''
    shape = X.shape
    w = np.ones((shape[1]))
    a = np.zeros((shape[1]))
    N = len(X)
    for i in range(t):
        for j in range(N):
            dot = (w.T @ X[j]) * Y[j]
            if dot <= 0:
                w = w + (r * (X[j] * Y[j]))
            a = a + w
    return a / (t * N)


def predictionErrorForStandard(X, Y, w):
    '''
        Get the prediction, count incorrect predictions and report error number
    '''
    count = 0
    sign = np.sign(X @ w.T)
    for i in range(len(Y)):
        if sign[i] != Y[i]:
            count += 1
    return count / len(X)


def predictionErrorForVoted(X, Y, w, c):
    '''
        Get the prediction, count incorrect predictions and report error number
    '''
    count = 0
    sum = 0
    for j in range(len(w)):
        sign = np.sign(X @ w[j].T)
        sign = c[j] * sign
        sum += sign
    sum = np.sign(sum)
    for i in range(len(Y)):
        if sum[i] != Y[i]:
            count += 1
    return count / len(X)


def predictionErrorForAveraged(X, Y, a):
    '''
        Get the prediction, count incorrect predictions and report error number
    '''
    count = 0
    sign = np.sign(X @ a.T)
    for i in range(len(Y)):
        if sign[i] != Y[i]:
            count += 1
    return count / len(X)


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
