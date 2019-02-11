import DecisionTree
import csv
'''
@author  Angel Dhungana

 Varies the maximum  tree depth from 1 to 6 --- for each setting, 
 and runs the algorithm to learn a decision tree, and uses the tree to  
 predict both the training  and test examples.
 Prints a table the average prediction errors on each dataset using
 information gain, majority error and gini index heuristics, respectively.
'''


def main():
    attributesValues = readDataDesc("car/data-desc.txt")
    dataSetTrain = readFromFile("car/train.csv")
    dataSetTest = readFromFile("car/test.csv")
    target = "label"
    # 0 = Entropy, 1 = Gini and 2 = Majority Error
    EntropyGiniMajority = [0, 1, 2]
    split = ["Entropy", "Gini Index", "Majority Error"]
    treeDepth = [1, 2, 3, 4, 5, 6]

    # For each depth
    for depth in treeDepth:
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
            printer = ("Depth = " + str(depth) + "\t" + "Splitter = " +
                       split[splitter] + "\t" + "\t" + "Error = " + str(
                           round(error, 2)))
            print(printer)
        print()


def readDataDesc(fileName):
    '''
        Read Attributes name from the Data Description
    '''
    columns = False
    with open(fileName, "r") as ins:
        for line in ins:
            if line.strip() == '':
                continue
            elif "| columns" == line.strip():
                columns = True
            elif columns == True:
                columnSet = [x.strip() for x in line.strip().split(',')]
        return columnSet


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


if __name__ == "__main__":
    main()