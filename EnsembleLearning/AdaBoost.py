'''
    @author - Angel Dhungana

    AdaBoost implementation with decision stumps of maxDepth = 2, that allows weights
'''
import ReadFiles
import numpy as np
import DecisionTree
import matplotlib.pyplot as plt


def AdaBoost(dataSetTrain, dataSetTest, Attributes, target, num_iterations):
    '''
        AdaBoost Implementation
            dataSetTrain - Training Data
            dataSetTest - Testing Data
            Attributes - Attributes List
            target - Target Attribute
            num_iterations - Max Iterations
    '''
    # contains each hypothesis weak classifier
    hypotheses = []
    # contains weights
    weights = []
    # Get y column from train and test dataset
    y_train = getY(dataSetTrain, Attributes.index(target))
    y_test = getY(dataSetTest, Attributes.index(target))
    # Make np array
    dataSetTrain = np.array(dataSetTrain)
    dataSetTest = np.array(dataSetTest)
    # Get Size of training data
    N, _ = dataSetTrain.shape
    # Initialize weight to be 1/m
    d = np.ones(N) / N

    # Error Holders
    error_Train = []
    error_Test = []
    error_Train_Stump = []
    error_Test_Stump = []

    for t in range(num_iterations):
        # Make a stump with weights
        h = DecisionTree.ID3(dataSetTrain.tolist(), Attributes, target, 3, 2,
                             d.tolist(), None)
        # Fill prediction
        pred = fillPrediction(h, dataSetTrain, Attributes)
        # Get error
        eps = d.dot(pred != y_train)
        # Calculate alpha
        alpha = (np.log(1 - eps) - np.log(eps)) / 2
        # Update weight
        d = d * np.exp(-alpha * y_train * pred)
        d = d / d.sum()
        # Append classifier so we will run error test
        hypotheses.append(h)
        weights.append(alpha)

        # Get errors on training, testing, decision stumps
        errTrain = pred_label(dataSetTrain, hypotheses, weights,
                              len(dataSetTrain), Attributes, y_train)
        errTest = pred_label(dataSetTest, hypotheses, weights,
                             len(dataSetTest), Attributes, y_test)
        errTrainStmp = stump_error(dataSetTrain, hypotheses, Attributes,
                                   target)
        errTestStmp = stump_error(dataSetTest, hypotheses, Attributes, target)

        # append errors to holders
        error_Train.append(errTrain)
        print(errTrain)
        error_Test.append(errTest)
        error_Train_Stump.append(errTrainStmp)
        error_Test_Stump.append(errTestStmp)

    # Return errors of ada and decision stumps
    return error_Train, error_Test, error_Train_Stump, error_Test_Stump


def pred_label(X, hypotheses, weights, N, Attributes, actY):
    '''
        Using classifier hypotheses and weights, predict label and return error rate
    '''
    y = np.zeros(N)
    for (h, alpha) in zip(hypotheses, weights):
        y = y + alpha * fillPrediction(h, X, Attributes)
    y = np.sign(y)
    err = get_error_rate(y, actY)
    return err / len(hypotheses)


def stump_error(X, hypotheses, Attributes, target):
    '''
        Using decision stumps predict label and return error rate
    '''
    error = 0
    for h in hypotheses:
        error += getPredictionError(h, X, Attributes, target)
    return error / float(len(hypotheses))


def get_error_rate(pred, Y):
    '''
        "Sum and Divide Error Rate
    '''
    return sum(pred != Y) / float(len(Y))


def plot_error_rate(er_train, er_test, n):
    '''
        Plotting the error rate for Adaboost
    '''
    arr = [x for x in range(10, 1010, 10)]
    plt.plot(arr, er_train, linewidth=1.5, color='red', label="Train Error")
    plt.plot(arr, er_test, linewidth=1, color='darkblue', label="Test Error")
    plt.xlabel('Number of iterations', fontsize=12)
    plt.ylabel('Error rate', fontsize=12)
    plt.title('Error rate vs number of iterations', fontsize=16)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('errorAda4.pdf', bbox_inches='tight')


def plot_error_rate2(er_train, er_test, n):
    '''
        Plotting the error rate for decision stumps
    '''
    plt.cla()
    plt.clf()
    arr = [x for x in range(10, 1010, 10)]
    plt.plot(
        arr, er_train, linewidth=2, color='lightblue', label="Train Error")
    plt.plot(arr, er_test, linewidth=2, color='darkblue', label="Test Error")
    plt.xlabel('Number of iterations', fontsize=12)
    plt.ylabel('Error rate Stumps', fontsize=12)
    plt.title('Error rate Stumps vs number of iterations', fontsize=16)
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.savefig('errorAda2.pdf', bbox_inches='tight')


def getPredictionError(tree, dataset, Attributes, target):
    '''
        Traverses the tree and get the correct number of predictions for dataset
    '''
    correctPrediction = 0
    targetIndx = Attributes.index(target)
    # For every row
    for row in dataset:
        curr = tree
        # If tree has branches, meaning its not a leaf
        x = curr.has_child()
        # loop until we hit leaf
        while x == True:
            indx = Attributes.index(curr.data)
            # get value from row
            value = row[indx]
            # get branch that has the value
            branch = curr.get_child(value)
            # if not such branch exist we break
            if branch == None:
                break
            # get the Node from branch
            curr = branch.get_BranchNode()
            x = curr.has_child()
        # if leaf value matches the actual target
        if curr.data == row[targetIndx]:
            correctPrediction += 1
    return 1 - (correctPrediction / float(len(dataset)))


def getY(dataSetTest, indx):
    '''
        Return y column from dataSet
    '''
    y = []
    for x in dataSetTest:
        y.append(x[indx])
    arr = np.array(y)
    return arr.astype(np.float)


def fillPrediction(tree, dataset, Attributes):
    '''
        Fill Predictions by traversing the decision tree, and return it
    '''
    prediction = []
    # For every row
    for row in dataset:
        curr = tree
        # If tree has branches, meaning its not a leaf
        x = curr.has_child()
        # loop until we hit leaf
        while x == True:
            indx = Attributes.index(curr.data)
            # get value from row
            value = row[indx]
            # get branch that has the value
            branch = curr.get_child(value)
            # if not such branch exist we break
            if branch == None:
                break
            # get the Node from branch
            curr = branch.get_BranchNode()
            x = curr.has_child()
        # add the prediction
        if (branch == None):
            prediction.append(-1)
        else:
            prediction.append(curr.data)
    arr = np.array(prediction)
    return arr.astype(np.float)
