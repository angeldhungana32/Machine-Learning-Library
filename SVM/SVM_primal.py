import numpy as np
import random
import matplotlib.pyplot as plt


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


def svm_primal(X, Y, rate, epochs, C, choice):
    shape = X.shape
    weights = np.zeros((shape[1]))
    N = len(X)
    counter = 0
    hinges = 0
    for _ in range(epochs):
        X, Y = shuffleData(X, Y)
        for i in range(N):
            gamma_t = select_gamma_primal(choice, rate, C, counter)
            if Y[i] * X[i] @ weights.T <= 1:
                weights = (1 - gamma_t) * weights + (
                    gamma_t * N * C * Y[i] * X[i])
            else:
                weights = (1 - gamma_t) * weights
            hinges += hinge_primal(X[i], Y[i], weights)
            #plt.plot(counter, hinges, marker="o")
            counter += 1
        hinges = 0
    plt.show()
    return weights


def hinge_primal(X, Y, w):
    hinger = max(0, 1 - (Y * (X @ w.T)))
    return hinger


def select_gamma_primal(choice, rate, d, t):
    if choice == 1:
        gamma_t = rate / (1 + ((rate / d) * t))
    else:
        gamma_t = rate / (1 + t)
    return gamma_t


def get_error(weights, X, Y):
    error = np.sign(X @ weights.T)
    counter = 0
    for i in range(len(X)):
        if error[i] != Y[i]:
            counter += 1
    return counter / len(Y)
