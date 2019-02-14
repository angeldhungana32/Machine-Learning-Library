import DecisionTree
import csv
import statistics
import ReadFiles
import copy
'''
@author  Angel Dhungana

 Converts the numeric attribute of a dataSet to Binary 
 Chooses the median of the attribute values (in the training set) 
 as the threshold, and examine if the feature is bigger (or less) than the threshold.
 If Bigger, the binary value is Yes and else its No.

 Varies the maximum  tree depth from 1 to 16 --- for each setting, 
 and runs the algorithm to learn a decision tree, and uses the tree to  
 predict both the training  and test examples.
 Prints a table the average prediction errors on each dataset using
 information gain, majority error and gini index heuristics, respectively.
'''


def main():
    # all attributes of Bank dataset
    attributesValues = [
        "age", "job", "martial", "education", "default", "balance", "housing",
        "loan", "contact", "day", "month", "duration", "campaign", "pdays",
        "previous", "poutcome", "y"
    ]
    # Which of the attributes are numeric
    isNumeric = [
        True, False, False, False, False, True, False, False, False, True,
        False, True, True, True, True, False, False
    ]
    # Read data from file
    dataSetTrainUnFilled = ReadFiles.readFromFile("bank/train.csv")
    dataSetTestUnFilled = ReadFiles.readFromFile("bank/test.csv")

    # Boolean value that determines which prediction to run, for train data or test data?
    target = "y"
    # Boolean value determines if we want to fill unknown or not

    # Gets the threshold from training data set
    # uses the threshold to convert numeric attributes to binary
    threshold = getThreshold(dataSetTrainUnFilled, attributesValues, isNumeric)

    dataSetTrainUnFilled = getBinaryForm(
        dataSetTrainUnFilled, attributesValues, isNumeric, threshold)
    dataSetTestUnFilled = getBinaryForm(dataSetTestUnFilled, attributesValues,
                                        isNumeric, threshold)
    #33 & 131
    unfilled1 = copy.copy(dataSetTrainUnFilled)
    unfilled2 = copy.copy(dataSetTestUnFilled)
    # If True, we fill the unknown with majority
    majority = getMajorityOfAllAttributes(unfilled1, attributesValues)
    print()
    print(
        "...Printing Errors for Problem 3. It might take little time. Please Be Patient"
        "...4 tables are being printed")
    print()
    print("This is the solution to 3a. Bank DataSet")
    print()
    errorDataTrainUnFilled = runTestOnTrainUnFilled(dataSetTrainUnFilled[:],
                                                    dataSetTestUnFilled[:],
                                                    attributesValues, target)
    printErrorReport(errorDataTrainUnFilled)
    print(
        "------Prediction Errors on Bank Training Set(Unknown considered as Attribute)------"
    )
    print()
    errorDataTestUnFilled = runTestOnTestUnFilled(dataSetTrainUnFilled[:],
                                                  dataSetTestUnFilled[:],
                                                  attributesValues, target)
    printErrorReport(errorDataTestUnFilled)
    print(
        "--------Prediction Errors on Bank Test Set(Unknown considered as Attribute)--------"
    )
    dataSetTrainFilled = fillUnknownDataset(unfilled1, attributesValues,
                                            majority)
    dataSetTestFilled = fillUnknownDataset(unfilled2, attributesValues,
                                           majority)
    print()
    print("This is the solution to 3b. Bank DataSet")
    print()
    errorDataTrainFilled = runTestOnTrainFilled(
        dataSetTrainFilled[:], dataSetTestFilled[:], attributesValues, target)
    printErrorReport(errorDataTrainFilled)
    print(
        "------Prediction Errors on Bank Training Set(Unknown is Filled with majority)------"
    )
    print()
    errorDataTestFilled = runTestOnTestFilled(
        dataSetTrainFilled[:], dataSetTestFilled[:], attributesValues, target)
    printErrorReport(errorDataTestFilled)
    print(
        "--------Prediction Errors on Bank Test Set(Unknown is Filled with majority)--------"
    )
    print()


def printErrorReport(errorData):
    '''
        Print the errors in formatted way
    '''
    s = [[str(e) for e in row] for row in errorData]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def runTestOnTrainUnFilled(dataSetTrain, dataSetTest, attributesValues,
                           target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, dataSetTrain, attributesValues,
                                   target) / len(dataSetTrain))
            # Print it
            s.append(str(round(error, 2)))
            #print(tree)
        errorData.append(s)
    return errorData


def runTestOnTestUnFilled(dataSetTrain, dataSetTest, attributesValues, target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, dataSetTest, attributesValues,
                                   target) / len(dataSetTest))
            # Print it
            s.append(str(round(error, 2)))
            #print(tree)
        errorData.append(s)
    return errorData


def runTestOnTrainFilled(dataSetTrain, dataSetTest, attributesValues, target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, dataSetTrain, attributesValues,
                                   target) / len(dataSetTrain))
            # Print it
            s.append(str(round(error, 2)))
            #print(tree)
        errorData.append(s)
    return errorData


def runTestOnTestFilled(dataSetTrain, dataSetTest, attributesValues, target):
    EntropyGiniMajority = [0, 1, 2]
    split = ["Tree Depth", "Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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


def getThreshold(dataSet, Attributes, isNumeric):
    '''
        Calculates median threshold from train dataset
    '''
    thresholds = []
    for x in Attributes:
        indx = Attributes.index(x)
        numeric = isNumeric[indx]
        if numeric == True:
            listAtt = []
            for row in dataSet:
                listAtt.append(float(row[indx]))
            # calculate median a numeric attribute column
            median = statistics.median(listAtt)
            thresholds.append(median)
    return thresholds


def getBinaryForm(dataSet, Attributes, isNumeric, thresholds):
    '''
        For all the numeric attributes, gets the median of it
        And if numeric value is less than median we consider that "No"
        Else, we consider "Yes"
    '''
    newSet = copy.copy(dataSet)
    thresholdCounter = 0
    for row in newSet:
        for x in Attributes:
            indx = Attributes.index(x)
            numeric = isNumeric[indx]
            if numeric == True:
                # If value is greater than or equal to threshold, its a "Yes"
                if thresholds[thresholdCounter] <= float(row[indx]):
                    row[indx] = "Yes"
                else:
                    row[indx] = "No"
                thresholdCounter += 1
        thresholdCounter = 0
    return newSet


def getMajorityOfAllAttributes(dataSet, Attributes):
    '''
        Return the majority of all attributes
    '''
    majority = []
    for target in Attributes:
        m = DecisionTree.mostCommonLabel(dataSet, Attributes, target)
        majority.append(m)
    return majority


def fillUnknownDataset(dataSet, Attributes, majority):
    '''
        if unknown is found, fill it with majority of at that index
    '''
    newSet = copy.copy(dataSet)
    for row in newSet:
        rowLength = len(row)
        for i in range(rowLength):
            if row[i] == 'unknown':
                row[i] = majority[i]
    return newSet


if __name__ == "__main__":
    main()