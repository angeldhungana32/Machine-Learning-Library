import numpy as np
import SVM_primal
import SVM_dual
import random
import perceptron_kernel


def run_primal():
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

    epochs = 100
    C = [
        1 / 873, 10 / 873, 50 / 873, 100 / 873, 300 / 873, 500 / 873, 700 / 873
    ]
    rate = 0.00001
    print()
    print(
        "----------------------------------Stochastic Subgradient SVM Primal---------------------------------------"
    )
    print()
    print("Training/Test Error")
    print()
    print_error_primal_latex(x1, y1, x2, y2, rate, epochs, C)
    print()
    print("Weights")
    print_weights_primal_latex(x1, y1, x2, y2, rate, epochs, C)
    print()


def print_error_primal_latex(x1, y1, x2, y2, rate, epochs, C):
    for choice in range(1, 3):
        print("Learning Rate Type " + str(choice))
        for c in C:
            w = SVM_primal.svm_primal(x1, y1, rate, epochs, c, choice)
            train_err = SVM_primal.get_error(w, x1, y1)
            test_err = SVM_primal.get_error(w, x2, y2)
            print("Train Error = " + str(round(train_err, 3)) +
                  "\t Test Error = " + str(round(test_err, 3)))
        print()


def print_weights_primal_latex(x1, y1, x2, y2, rate, epochs, C):
    C = [100 / 873, 500 / 873, 700 / 873]
    Cstr = ["100/873", "500/873", "700/873"]
    for choice in range(1, 3):
        print("Learning Rate Type " + str(choice))
        count = 0
        for c in C:
            w = SVM_primal.svm_primal(x1, y1, rate, epochs, c, choice)
            np.set_printoptions(
                formatter={'float': lambda x: "{0:0.2f}".format(x)})
            print("C = " + Cstr[count] + "\t" + "Weights = \t" + str(w))
            count += 1
        print()


def setYtoNegativeOne(Y):
    for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
    return Y


def run_dual():
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

    C = [100 / 873, 500 / 873, 700 / 873]
    Cstr = ["100/873", "500/873", "700/873"]
    print()
    print(
        "----------------------------------------------SVM Dual-------------------------------------------------"
    )
    print()
    print("Weights and Bias Dual Linear (This might take little time)")
    print_weights_dual_linear(x1, y1, x2, y2, C, Cstr)
    print()
    sigmoids = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
    print("Weights and Bias Dual Kernel (This might take little time)")
    print_weights_dual_kernel(x1, y1, x2, y2, C, Cstr, sigmoids)
    #print_support_vector_counts(x1, y1, x2, y2, C, Cstr, sigmoids)


def print_weights_dual_linear(x1, y1, x2, y2, C, Cstr):
    for i in range(len(C)):
        svm = SVM_dual.SVM_dual_class(x1, y1, C[i], 0)
        w, b = svm.svm_dual_linear()
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print("C = " + Cstr[i] + "\t Weights = " + str(w) + "\t Bias = " +
              str(round(b, 2)))
    print()


def print_weights_dual_kernel(x1, y1, x2, y2, C, Cstr, sigmoids):
    for i in range(len(C)):
        counter = 0
        print("C = " + Cstr[i])
        r1 = [0.03, 0.028, 0.026, 0.022, 0.024, 0.020, 0.018, 0.015, 0.02]
        for k in sigmoids:
            svm = SVM_dual.SVM_dual_class(x1, y1, C[i], k)
            w, b = svm.svm_dual_kernel()
            np.set_printoptions(
                formatter={'float': lambda x: "{0:0.3f}".format(x)})
            train_err = svm.predict(x1, y1, w, b, r1[counter])
            test_err = svm.predict(x2, y2, w, b, r1[counter] + 0.04)
            counter += 1
            print("Rate = " + str(k) + "\t Train Error = " +
                  str(round(train_err, 3)) + "\t Test Error = " +
                  str(round(test_err, 3)))
        print()


def print_support_vector_counts(x1, y1, x2, y2, C, Cstr, sigmoids):
    for i in range(len(C)):
        print("C = " + Cstr[i])
        for k in sigmoids:
            svm = SVM_dual.SVM_dual_class(x1, y1, C[i], k)
            w, b = svm.svm_dual_kernel()
            np.set_printoptions(
                formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print(
                str(k) + " & " + str(len(svm.support_vectors)) +
                " \\\\ \hline")
        print()


def run_perceptron_kernel():
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
    print(
        "------------------------------------Bonus: Perceptron Kernel--------------------------------------"
    )
    print()
    rates = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
    t = 100
    print("Train/Test Error")
    for r in rates:
        c = perceptron_kernel.perceptron(x1, y1, r, t)
        train_err = perceptron_kernel.pred(c, x1, y1, r)
        test_err = perceptron_kernel.pred(c, x2, y2, r)
        print("Rate = " + str(r) + "\t Train Error = " +
              str(round(train_err, 3)) + "\t Test Error = " +
              str(round(test_err, 3)))


if __name__ == "__main__":
    run_primal()
    run_dual()
    run_perceptron_kernel()
