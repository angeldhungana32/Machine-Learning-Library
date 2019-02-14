import DecisionTree
import csv
import ReadFiles
import numpy as np
'''
@author  Angel Dhungana

 Varies the maximum  tree depth from 1 to 6 --- for each setting, 
 and runs the algorithm to learn a decision tree, and uses the tree to  
 predict both the training  and test examples.
 Prints a table the average prediction errors on each dataset using
 information gain, majority error and gini index heuristics, respectively.
'''


def main():
    attributesValues = ReadFiles.readDataDesc("car/data-desc.txt")
    dataSetTrain = ReadFiles.readFromFile("car/train.csv")
    dataSetTest = ReadFiles.readFromFile("car/test.csv")
    target = "label"
    errorDataTrain = runTestOnTrain(dataSetTrain, dataSetTest,
                                    attributesValues, target)
    errorDataTest = runTestOnTest(dataSetTrain, dataSetTest, attributesValues,
                                  target)
    print()
    print("This is the solution to 2b. Car DataSet")
    print()
    printErrorReport(errorDataTrain)
    print("------Prediction Errors on Car Training Set------")
    print()
    printErrorReport(errorDataTest)
    print("--------Prediction Errors on Car Test Set--------")
    print()
    # 0 = Entropy, 1 = Gini and 2 = Majority Error


def printErrorReport(errorData):
    s = [[str(e) for e in row] for row in errorData]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def runTestOnTrain(dataSetTrain, dataSetTest, attributesValues, target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6]
    errorData = []
    errorData.append(split)
    # For each depth
    for depth in treeDepth:
        s = []
        s.append(depth)
        # For each variants
        for splitter in EntropyGiniMajority:
            # Generate tree from train set
            tree = DecisionTree.ID3(dataSetTrain, attributesValues, target,
                                    splitter, depth)
            #print(str(tree))
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, dataSetTrain, attributesValues,
                                   target) / len(dataSetTrain))
            # Print it
            s.append(str(round(error, 2)))
            #print(tree)
        errorData.append(s)
    return errorData


def runTestOnTest(dataSetTrain, dataSetTest, attributesValues, target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6]
    errorData = []
    errorData.append(split)
    # For each depth
    for depth in treeDepth:
        s = []
        s.append(depth)
        # For each variants
        for splitter in EntropyGiniMajority:
            # Generate tree from train set
            tree = DecisionTree.ID3(dataSetTrain, attributesValues, target,
                                    splitter, depth)
            #print(str(tree))
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, dataSetTest, attributesValues,
                                   target) / len(dataSetTest))
            # Print it
            s.append(str(round(error, 2)))
            #print(tree)
        errorData.append(s)
    return errorData


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
    return correctPrediction


if __name__ == "__main__":
    main()