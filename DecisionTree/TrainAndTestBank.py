import DecisionTree
import csv
import statistics
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
    dataSetTrain = readFromFile("bank/train.csv")
    dataSetTest = readFromFile("bank/test.csv")

    # Boolean value that determines which prediction to run, for train data or test data?
    trainOrTest = False
    target = "y"
    # Boolean value determines if we want to fill unknown or not
    fillUnknown = True

    # Gets the threshold from training data set
    # uses the threshold to convert numeric attributes to binary
    threshold = getThreshold(dataSetTrain, attributesValues, isNumeric)
    dataSetTrain = getBinaryForm(dataSetTrain, attributesValues, isNumeric,
                                 threshold)
    dataSetTest = getBinaryForm(dataSetTest, attributesValues, isNumeric,
                                threshold)
    # If True, we fill the unknown with majority
    if fillUnknown == True:
        majority = getMajorityOfAllAttributes(dataSetTrain, attributesValues)
        dataSetTrain = fillUnknownDataset(dataSetTrain, attributesValues,
                                          majority)
        dataSetTest = fillUnknownDataset(dataSetTest, attributesValues,
                                         majority)
    # set main data set to test on
    if trainOrTest == True:
        mainDataSet = dataSetTrain
    else:
        mainDataSet = dataSetTest

    # 0 = Entropy, 1 = Gini and 2 = Majority Error
    EntropyGiniMajority = [0, 1, 2]
    # tree depth
    treeDepth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # For each depth
    for depth in treeDepth:
        # For each variants
        s = []
        for splitter in EntropyGiniMajority:
            # Generate tree from train set
            tree = DecisionTree.ID3(dataSetTrain, attributesValues, target,
                                    splitter, depth)
            # Get Error for train or test set
            error = 1 - (
                getPredictionError(tree, mainDataSet, attributesValues,
                                   target) / len(mainDataSet))
            # Print it
            s.append(str(round(error, 2)))
        # printing in a way so that, it will be easier for me to put in latex
        print(
            str(depth) + " & " + s[0] + " & " + s[1] + " & " + s[2] +
            " \\\ \hline")


def readFromFile(fileName):
    '''
        Read CSV file and make dataset
    '''
    columns = []
    with open(fileName, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            columns.append(row)
    return columns


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
    thresholdCounter = 0
    for row in dataSet:
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
    return dataSet


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
    for row in dataSet:
        rowLength = len(row)
        for i in range(rowLength):
            if row[i] == 'unknown':
                row[i] = majority[i]
    return dataSet


if __name__ == "__main__":
    main()