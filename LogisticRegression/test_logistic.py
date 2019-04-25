'''
    @author Angel Dhungana
    Run the Logistic Regression Algorithm
'''
import numpy as np
import logisticregression


def run():
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
    r = 1
    t = 100
    v = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    d = 0.5
    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print(
        "--------------------------Logistic Regression MAP Estimation--------------------------"
    )
    print()
    run_logistic_map(x1, y1, x2, y2, r, t, v, d)
    r = 0.01
    d = 0.5
    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print(
        "--------------------------Logistic Regression ML Estimation--------------------------"
    )
    run_logistic_ml(x1, y1, x2, y2, r, t, v, d)
    print()


def run_logistic_map(X1, Y1, X2, Y2, r, t, v, d):
    print("Rate \t Train Error \t Test Error")
    for vv in v:
        w = logisticregression.stochastic_logistic_map(X1, Y1, r, t, vv, d)
        train_pred = logisticregression.prediction(X1, Y1, w)
        test_pred = logisticregression.prediction(X2, Y2, w)
        print(
            str(vv) + "\t\t" + str(round(train_pred, 3)) + "\t\t" +
            str(round(test_pred, 3)))


def run_logistic_ml(X1, Y1, X2, Y2, r, t, v, d):
    print("Rate \t Train Error \t Test Error")
    for vv in v:
        w = logisticregression.stochastic_logistic_max_est(X1, Y1, r, t, vv, d)
        train_pred = logisticregression.prediction(X1, Y1, w)
        test_pred = logisticregression.prediction(X2, Y2, w)
        print(
            str(vv) + "\t\t" + str(round(train_pred, 3)) + "\t\t" +
            str(round(test_pred, 3)))


def setYtoNegativeOne(Y):
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    return Y


if __name__ == "__main__":
    run()