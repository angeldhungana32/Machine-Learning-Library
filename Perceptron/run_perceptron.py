import numpy as np
import Perceptron


def main():
    # Give X and Y data, follow example on testPerceptron file if confused
    # Make sure Y data is either 1 or -1
    X_train, Y_train = np.array([])

    #Give test data X and Y
    X_test, Y_test = np.array([])

    # set learning rate
    rate = 1

    # set epochs
    t = 10
    '''
        Select the choice of Perceptron you want to run, comment out other as your need
    '''

    # Standard Perceptron
    standard_percp_weights = Perceptron.StandardPerception(
        X_train, Y_train, rate, t)

    # Standard Perceptron error rate on test data
    standard_percp_pred_error = Perceptron.predictionErrorForStandard(
        X_test, Y_test, standard_percp_weights)

    # Voted Perceptron
    voted_percp_weights, c = Perceptron.VotedPerception(
        X_train, Y_train, rate, t)

    # Voted Perceptron error rate on test data
    voted_percp_pred_error = Perceptron.predictionErrorForVoted(
        X_test, Y_test, voted_percp_weights, c)

    # Averaged Perceptron
    averaged_percp_weights = Perceptron.AveragedPerception(
        X_train, Y_train, rate, t)

    # Averaged Perceptron error rate on test data
    averaged_percp_pred_error = Perceptron.predictionErrorForAveraged(
        X_test, Y_test, averaged_percp_weights)

    # Print the error or the weights as you like below:


if __name__ == "__main__":
    main()