'''
    @author Angel Dhungana
    Run the Perception Algorithm
'''
import numpy as np
import Perceptron


def runPerception():
    dataSetTrain = np.loadtxt("bank-note/train.csv", delimiter=',')
    dataSetTest = np.loadtxt("bank-note/test.csv", delimiter=',')
    # Separate X and Y Train data
    x1 = dataSetTrain[:, 0:-1]
    y1 = dataSetTrain[:, -1]
    y1 = setYtoNegativeOne(y1)
    # Separate X and Y Test data
    x2 = dataSetTest[:, 0:-1]
    y2 = dataSetTest[:, -1]
    y2 = setYtoNegativeOne(y2)
    # learning rate start from 1
    rate = 1
    t = 10
    print()
    print(
        "----------------------------------------Perception Algorithms--------------------------------------------"
    )
    print()
    print(
        "----------------------------------------Standard Algorithm--------------------------------------------"
    )
    print()
    runStandard(x1, y1, x2, y2, rate, t)
    print()
    print(
        "----------------------------------------Voted Algorithm--------------------------------------------"
    )
    print()
    runVoted(x1, y1, x2, y2, rate, t)
    print()
    print(
        "----------------------------------------Averaged Algorithm--------------------------------------------"
    )
    print()
    runAveraged(x1, y1, x2, y2, rate, t)
    print()


def runStandard(x1, y1, x2, y2, rate, t):
    wStandard = Perceptron.StandardPerception(x1, y1, rate, t)
    print("Standard Weights = " + str(wStandard))
    errStandard = Perceptron.predictionErrorForStandard(x2, y2, wStandard)
    print("Standard Prediction Error" + " = " + str(errStandard))


def runVoted(x1, y1, x2, y2, rate, t):
    wVoted, c = Perceptron.VotedPerception(x1, y1, rate, t)
    errVoted = Perceptron.predictionErrorForVoted(x2, y2, wVoted, c)
    #print("Voted Weights = " + str(wVoted[i]) + " " + "Count = " + str(c[i]))
    for i in range(len(wVoted)):
        string = "w = [\t"
        for val in wVoted[i]:
            string += ('{:4}'.format(round(val, 2))) + "\t"
        string += "] \t c = " + str(c[i])
        print(string)
    print()
    print("Voted Prediction Error" + " = " + str(errVoted))


def runAveraged(x1, y1, x2, y2, rate, t):
    wAveraged = Perceptron.AveragedPerception(x1, y1, rate, t)
    print("Averaged Weights = " + str(wAveraged))
    errAveraged = Perceptron.predictionErrorForAveraged(x2, y2, wAveraged)
    print("Averaged Prediction Error" + " = " + str(errAveraged))


def setYtoNegativeOne(Y):
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    return Y


if __name__ == "__main__":
    runPerception()
