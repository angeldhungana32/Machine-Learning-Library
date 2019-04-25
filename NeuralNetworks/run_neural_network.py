from random import seed
from random import randrange
from random import random
from csv import reader
import numpy as np
import neural_networks
import random
import test_neural_net


def main():
    # Test Backprop on Seeds dataset
    seed(1)
    # load and prepare data
    train = ''
    test = ''
    dataset = test_neural_net.load_csv(train)
    dataset2 = test_neural_net.load_csv(test)
    for i in range(len(dataset[0]) - 1):
        test_neural_net.to_float(dataset, i)
    # convert class column to integers
    test_neural_net.to_int(dataset, len(dataset[0]) - 1)
    for i in range(len(dataset2[0]) - 1):
        test_neural_net.to_float(dataset2, i)
    # convert class column to integers
    test_neural_net.to_int(dataset2, len(dataset2[0]) - 1)
    # Provide Necessary Variables
    l_rate = 0.01
    n_epoch = 100
    n_hidden = 2
    d = 5
    print()
    p, p2 = test_neural_net.back_propagation_stochastic_random(
        dataset, dataset2, l_rate, d, n_epoch, n_hidden)
    #p, p2 = test_neural_net.back_propagation_stochastic_zeros(
    #    dataset, dataset2, l_rate, d, n_epoch, n_hidden)
    error_train = test_neural_net.get_error(np.array(dataset), p)
    error_test = test_neural_net.get_error(np.array(dataset2), p2)
    #print(str(width) + " & " + str(error_train) + " & " + str(error_test))
    print(str(n_hidden) + "\t\t" + str(error_train) + "\t" + str(error_test))


if __name__ == "__main__":
    main()