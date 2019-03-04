'''
    @author - Angel Dhungana

    This is where you will run my tree
'''
import DecisionTree
import ReadFiles


def main():
    trainFileName = "bank/train.csv"
    testFileName = "bank/test.csv"

    dataSetTrain = ReadFiles.readFromFile(trainFileName)
    dataSetTest = ReadFiles.readFromFile(testFileName)

    # List out the attributes name, For Example: ["Outlook","Humidity","Wind", "Play"]
    attributes = []

    # Write the name of your target label here: If attributes are ["x1","x2","y"], then your target label could be y, write y
    target = ''

    # Enter maxDepth of your tree
    maxDepth = 10

    #Choose Splitting Method, 0 - Uses Entropy, 1 - Gini Index, 2 - Majority Error
    splitter = 0

    # This will make the tree
    tree = DecisionTree.ID3(dataSetTrain, attributes, target, splitter,
                            maxDepth)

    # if You want to print the string version of tree, uncomment the following method
    #print(str(tree))

    # if You want to run your test data and see prediction error, uncomment the following
    #PrintPredictionError(tree, dataSetTest, attributes, target)


def PrintPredictionError(tree, dataset, Attributes, target):
    error = 1 - (
        getPredictionError(tree, dataset, Attributes, target) / len(dataset))

    print("The prediction error is : " + str(error))


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