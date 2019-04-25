from random import random
from math import exp
import numpy as np
import random as rand
'''
    Neural Networks
'''


def forward_pass(network, row):
    '''
        Forward Pass in the Network
    '''
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            act = neuron['w'][-1]
            for i in range(len(neuron['w']) - 1):
                act += neuron['w'][i] * inputs[i]
            neuron['o'] = 1.0 / (1.0 + exp(-act))
            new_inputs.append(neuron['o'])
        inputs = new_inputs
    return inputs


def update_weights(network, row, l_rate):
    '''
        Update Weights based on back propagation error
    '''
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['o'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['w'][j] += l_rate * neuron['d'] * inputs[j]
            neuron['w'][-1] += l_rate * neuron['d']


def train_network(network, train, rate, d, epoch, num_outputs):
    '''
        Train the network
    '''
    for t in range(epoch):
        train = shuffle_data(train)
        rate = rate / (1 + (rate / d) * t)
        for row in train:
            _ = forward_pass(network, row)
            expected = encode(num_outputs, row)
            back_propagate(network, expected)
            update_weights(network, row, rate)


def back_propagate(network, target):
    '''
       Back Propagation Error
    '''
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['w'][j] * neuron['d'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(target[j] - neuron['o'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['d'] = errors[j] * (neuron['o'] * (1.0 - neuron['o']))


def encode(outputs, row):
    target = [0 for i in range(outputs)]
    target[row[-1]] = 1
    return target


def shuffle_data(X):
    randomSample = rand.sample(range(len(X)), len(X))
    newX = []
    for i in range(len(X)):
        newX.append(X[randomSample[i]])
    return newX


def initialize_network_zeros(tab):
    '''
        Add zeros as weights, make graph
    '''
    network = []
    for layer_i in range(1, len(tab)):
        layer = []
        for _ in range(tab[layer_i]):
            weight = []
            for _ in range(tab[layer_i - 1] + 1):
                weight.append(0)
            temp = {'w': weight}
            layer.append(temp)
        network.append(layer)
    return network


def initialize_network_random(inouthidden):
    '''
        Add random as weights, Make graph
    '''
    network = []
    for layer_i in range(1, len(inouthidden)):
        layer = []
        for _ in range(inouthidden[layer_i]):
            weight = []
            for _ in range(inouthidden[layer_i - 1] + 1):
                weight.append(random())
            temp = {'w': weight}
            layer.append(temp)
        network.append(layer)
    return network
