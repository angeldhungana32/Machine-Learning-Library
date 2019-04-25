'''
    @author Angel Dhungana
    Run the Neural Network
'''
from random import seed
from random import randrange
from random import random
from csv import reader
import numpy as np
import neural_networks
import random


# Load a CSV file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if row:
                dataset.append(row)
    return dataset


def to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


def get_error(data, y):
    y1 = data[:, -1]
    return np.sum(y1 != np.array(y)) / len(y)


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation_stochastic_random(train, test, l_rate, d, n_epoch,
                                       n_hidden):
    '''
        Run Stochastic with initial random values
    '''
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    # Initialize Network
    network = neural_networks.initialize_network_random(
        [n_inputs, n_hidden, n_outputs])
    # Train Network
    neural_networks.train_network(network, train, l_rate, d, n_epoch,
                                  n_outputs)
    prediction_train = []
    predictions_test = []
    # Get Prediction
    for row in train:
        prediction = predict(network, row)
        prediction_train.append(prediction)
    for row in test:
        prediction = predict(network, row)
        predictions_test.append(prediction)
    return (prediction_train), (predictions_test)


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation_stochastic_zeros(train, test, l_rate, d, n_epoch,
                                      n_hidden):
    '''
        Run Stochastic with initial zero values
    '''
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    # Initialize Network
    network = neural_networks.initialize_network_zeros(
        [n_inputs, n_hidden, n_outputs])
    # Train Network
    neural_networks.train_network(network, train, l_rate, d, n_epoch,
                                  n_outputs)
    prediction_train = []
    predictions_test = []
    # Get Prediction
    for row in train:
        prediction = predict(network, row)
        prediction_train.append(prediction)
    for row in test:
        prediction = predict(network, row)
        predictions_test.append(prediction)
    return (prediction_train), (predictions_test)


def to_int(dataset, column):
    '''
    '''
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]


# Make a prediction with a network
def predict(network, row):
    out = neural_networks.forward_pass(network, row)
    return out.index(max(out))


def main():
    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    train = 'bank-note/train.csv'
    test = 'bank-note/test.csv'
    dataset = load_csv(train)
    dataset2 = load_csv(test)
    for i in range(len(dataset[0]) - 1):
        to_float(dataset, i)
    # convert class column to integers
    to_int(dataset, len(dataset[0]) - 1)
    for i in range(len(dataset2[0]) - 1):
        to_float(dataset2, i)
    # convert class column to integers
    to_int(dataset2, len(dataset2[0]) - 1)
    # evaluate algorithm
    l_rate = 0.01
    n_epoch = 100
    n_hidden = [5, 10, 25, 50, 100]
    d = 5
    print()
    print(
        "--------------------------Stochastic Neural with Random Initial Weights--------------------------"
    )
    print("Width \t\t Train Error \t\t Test Error")
    for width in n_hidden:
        p, p2 = back_propagation_stochastic_random(dataset, dataset2, l_rate,
                                                   d, n_epoch, width)
        error_train = get_error(np.array(dataset), p)
        error_test = get_error(np.array(dataset2), p2)
        #print(str(width) + " & " + str(error_train) + " & " + str(error_test))
        print(str(width) + "\t\t" + str(error_train) + "\t" + str(error_test))
    print(
        "-------------------------------------------------------------------------------------------------"
    )
    print(
        "--------------------------Stochastic Neural with Zeros Initial Weights--------------------------"
    )
    print("Width \t\t Train Error \t\t Test Error")
    for width in n_hidden:
        p, p2 = back_propagation_stochastic_zeros(dataset, dataset2, l_rate, d,
                                                  n_epoch, width)
        error_train = get_error(np.array(dataset), p)
        error_test = get_error(np.array(dataset2), p2)
        #print(str(width) + " & " + str(error_train) + " & " + str(error_test))
        print(str(width) + "\t\t" + str(error_train) + "\t" + str(error_test))
    print()


if __name__ == "__main__":
    main()