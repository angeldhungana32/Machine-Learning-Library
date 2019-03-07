'''
    @author Angel Dhungana

    Run my implementation of AdaBoost, Bagging, and Random Forests
'''
import AdaBoost
import Bagging
import RandomForest
import ReadFiles


def main():
    # Provide list of attributes
    # Example: ["Temperature","Humidity","Play?"]
    attributesValues = []
    fileTrain = ""
    fileTest = ""
    target = ""  # Target attribute, for the example above it would be "Play?""
    dataSetTrain = ReadFiles.readFromFile(fileTrain)
    dataSetTest = ReadFiles.readFromFile(fileTest)
    num_iterations = 100

    #Uncomment following function calls to run corresponding algorithm

    # boosting(dataSetTrain, dataSetTest, attributesValues, target,
    #num_iterations)

    #bagging(dataSetTrain, dataSetTest, attributesValues, target,
    #num_iterations)

    # Subset of features for Random Forest, Write index of Attributes
    subset = []
    num_trees = 1000

    #random_forest(dataSetTrain, dataSetTest, attributesValues, target,
    #              num_iterations, subset, num_trees)


def boosting(dataSetTrain, dataSetTest, attributesValues, target,
             num_iterations):
    hyp, wei = AdaBoost.AdaBoost(dataSetTrain, dataSetTest, attributesValues,
                                 target, num_iterations)
    print()
    print(
        "--------------------------- Ada Boost -----------------------------")
    YTest = AdaBoost.getY(dataSetTest, attributesValues.index(target))
    err = AdaBoost.pred_label(dataSetTest, hyp, wei, len(dataSetTest),
                              attributesValues, YTest)
    print("Prediction Error For Test Data: " + str(err))
    print()


def bagging(dataSetTrain, dataSetTest, attributesValues, target,
            num_iterations):
    '''
        Run the bagging algorithm with train dataset
    '''
    hypth = Bagging.Bagging(dataSetTrain, attributesValues, target,
                            num_iterations)
    # Get final prediction for train and test and plot
    bag = Bagging.calculateFinalPrediction(hypth, dataSetTrain, dataSetTest,
                                           attributesValues, target)
    getTestError = Bagging.getTestError(
        bag[1], Bagging.getY(dataSetTest, attributesValues.index(target)))
    print()
    print("--------------------------- Bagging -----------------------------")
    print()
    print("Prediction Error For Test Data: " + str(getTestError))
    print()


def random_forest(dataSetTrain, dataSetTest, attributesValues, target,
                  num_iterations, subset, num_trees):

    hypothesis = RandomForest.RandomForest(dataSetTrain, attributesValues,
                                           target, num_iterations, subset)
    bag = RandomForest.calculateFinalPrediction(
        hypothesis, dataSetTrain, dataSetTest, attributesValues, target)
    getTestError = Bagging.getTestError(
        bag[1], Bagging.getY(dataSetTest, attributesValues.index(target)))
    print()
    print(
        "--------------------------- Random Forest -----------------------------"
    )
    print()
    print("Prediction Error For Test Data: " + str(getTestError))
    print()


if __name__ == "__main__":
    main()