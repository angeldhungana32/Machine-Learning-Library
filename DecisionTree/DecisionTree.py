'''
@author  Angel Dhungana
'''
import re
import math
import myTree
'''
    File that implements the ID3 with varints to calculate best attributes including
    entropy, majority index and Gini Index Information Gain calculations. You can also
    set the maximum depth of the tree.  
'''


def checkIfAllLabelSame(S, Attributes, target):
    '''
        Given a dataset checks if all the target label are same for every 
        classification
    '''
    # get the index of the target label
    indx = getIndexOfAttributes(Attributes, target)
    count = 0
    current = ''
    # check if there is one different and break
    for x in S:
        if count == 0:
            current = x[indx]
            count += 1
        elif current != x[indx]:
            return None
    return current


def mostCommonLabel(S, Attributes, target):
    '''
        Given a data set retrieves the most common target level
        Generic Function, works for any target label
    '''
    # get index of target label
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = {}
    # for each unique label add to dictionary and increment if already exists
    for x in S:
        currLbl = x[indx]
        if not currLbl in dictionary:
            dictionary[currLbl] = 1
        else:
            dictionary[currLbl] += 1
    # Return the label with max occurence
    return max(dictionary, key=dictionary.get)


def getIndexOfAttributes(Attributes, target):
    '''
        Returns the index of the target label in Attributes list
    '''
    return Attributes.index(target)


def chooseBestAttribute(S, Attributes, target, EntropyGiniMajority):
    '''
        Chooses best attribute to split given the dataset S, Attributes A, and target 
        label.
        If EntropyGiniMajority == 0, uses Entropy. 
        EntropyGiniMajority == 1, uses Gini Index.
        EntropyGiniMajority == 2, uses Majority Error. 
    '''
    dictionary = {}
    # For each attribute
    for attr in Attributes:
        # If the attribute is not a target
        if attr != target:
            # Based on which variant, calculate gain and add to dictionary
            if EntropyGiniMajority == 0:
                dictionary[attr] = getInformationGainEntropy(
                    S, Attributes, target, attr)
            elif EntropyGiniMajority == 1:
                dictionary[attr] = getInformationGainGiniIndex(
                    S, Attributes, target, attr)
            else:
                dictionary[attr] = getInformationGainMajorityError(
                    S, Attributes, target, attr)
    # Returns the attribute with maximum information gain
    return max(dictionary, key=dictionary.get)


def getInformationGainEntropy(S, Attributes, target, attr):
    '''
        Helper Function that Calculates Gain using Entropy
    '''
    # Entropy of whole dataset
    mainEntropy = calculateEntropy(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    # Add frequency of attributes values to dictionary
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrEntropy = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    # for each attribute value calculate entropy
    for num in dictionary.keys():
        subset = [col for col in S if col[indx] == num]
        attrEntropy += (dictionary[num] / total) * calculateEntropy(
            subset, Attributes, target)
    #Return gain
    return mainEntropy - attrEntropy


def getInformationGainGiniIndex(S, Attributes, target, attr):
    '''
        Helper Function that Calculates Gain using Gini Index
    '''
    # Gini Index of whole dataset
    mainGiniInd = calculateGiniIndex(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    # Add frequeuncy of attributes values to dictionary
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrGiniInd = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    # For each value get subset and calculate gini index
    for num in dictionary.keys():
        subset = [col for col in S if col[indx] == num]
        # p * giniIndx
        attrGiniInd += (dictionary[num] / total) * calculateGiniIndex(
            subset, Attributes, target)
    # Return gain
    return mainGiniInd - attrGiniInd


def getInformationGainMajorityError(S, Attributes, target, attr):
    '''
        Helper Function that Calculates Gain using Majority Error
    '''
    # Get the Majority Error of whole dataset
    mainMajorityError = calculateMajorityError(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    # Add the values of Attributes
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrMajoritErr = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    # For each values, get subset and calulate majority error
    for num in dictionary.keys():
        subset = [col for col in S if col[indx] == num]
        attrMajoritErr += (dictionary[num] / total) * calculateMajorityError(
            subset, Attributes, target)
    # Return gain
    return mainMajorityError - attrMajoritErr


def helperFrequencyFinder(S, indx):
    '''
        Helper Function that finds the frequency of attributes[indx] in S 
    '''
    dictionary = {}
    total = 0
    # If existing increment, else add to dictionary
    for col in S:
        if not col[indx] in dictionary:
            dictionary[col[indx]] = 1
        else:
            dictionary[col[indx]] += 1
        total += 1
    return dictionary, total


def calculateEntropy(S, Attributes, target):
    '''
        Calculates Entropy (-P * log_2(P)) + ..... +
    '''
    # Get indx of target label
    indx = getIndexOfAttributes(Attributes, target)
    # Get attribute values labels as dict
    dictionary = helperFrequencyFinder(S, indx)[0]
    values = dictionary.values()
    total = len(S)
    entropy = 0.0
    # For each label value calculate entropy
    for num in values:
        entropy += (-num / total) * (math.log2(num / total))
    return entropy


def calculateGiniIndex(S, Attributes, target):
    '''
        Calculates Gini Index 1 - (p_1^2 + p_2^2+ ....)
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = helperFrequencyFinder(S, indx)[0]
    total = len(S)
    values = dictionary.values()
    giniIndex = 0.0
    # For each label value calculate gini index
    for num in values:
        giniIndex += (num / total) * (num / total)
    return 1 - giniIndex


def calculateMajorityError(S, Attributes, target):
    '''
        Calculates Majority Error 1 - (ME(target)) 
        Example 3/5 and 2/5 then, majority error is 1 - 3/5
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = helperFrequencyFinder(S, indx)[0]
    # Get the max number of occurences
    majorityError = max(dictionary.values())
    # Divide that by total and return error by subtracting to 1
    return 1 - (majorityError / len(S))


def getValuesOfA(S, Attributes, A):
    '''
        Returns the possible values that A
    '''
    ind = getIndexOfAttributes(Attributes, A)
    vals = [col[ind] for col in S]
    # Return the set of unique attribute features
    return set(vals)


def getSubsetAFromS(S, Attributes, A, v):
    '''
        Returns subset of examples in S with A=v
    '''
    subset = []
    ind = getIndexOfAttributes(Attributes, A)
    for col in S:
        if (col[ind] == v):
            # Make subset by removing the column A from S
            subset.append([col[i] for i in range(0, len(col)) if i != ind])
    return subset


def ID3(S, Attributes, Label, EntropyGiniMajority, treeDepth):
    '''
        Main ID3 that calls the helper recursive ID3
    '''
    # Calling with current depth 0
    currentDepth = 0
    return recursiveID3(S, Attributes, Label, EntropyGiniMajority,
                        currentDepth, treeDepth)


def recursiveID3(S, Attributes, Label, EntropyGiniMajority, currentDepth,
                 treeDepth):
    '''
        ID3 algorithm that recursively builds a decision tree
    '''
    sameLabel = checkIfAllLabelSame(S, Attributes, Label)
    majority = mostCommonLabel(S, Attributes, Label)
    # If same Label return a node with that label
    if sameLabel != None:
        return myTree.Node(sameLabel)
    # If len of Attributes is 0, return node with majority label
    if not S or len(Attributes) == 0:
        return myTree.Node(majority)
    else:
        # Else, split on best attribute
        A = chooseBestAttribute(S, Attributes, Label, EntropyGiniMajority)
        # Create a root node with that attribute
        root = myTree.Node(A)
        # Get unique values of A
        valuesOfAList = getValuesOfA(S, Attributes, A)
        # For each value
        for v in valuesOfAList:
            # Get that subset of S, A=v
            subset = getSubsetAFromS(S, Attributes, A, v)

            # If empty subset, add the most common label as Node
            if len(subset) == 0:
                root.add_branch(myTree.Branch(v))
                grandChild = myTree.Node(mostCommonLabel(S, Attributes, Label))
                root.get_child(v).add_BranchNode(grandChild)
            # If next depth will be max depth, choose most common from subset
            elif (currentDepth + 1 == treeDepth):
                # Remove A column from Atributes List, making a copy so we don't remove
                # from original one
                removedBestAttribute = Attributes[:]
                removedBestAttribute.remove(A)
                root.add_branch(myTree.Branch(v))
                grandChild = myTree.Node(
                    mostCommonLabel(subset, removedBestAttribute, Label))
                root.get_child(v).add_BranchNode(grandChild)
            else:
                removedBestAttribute = Attributes[:]
                removedBestAttribute.remove(A)
                # recursively call the ID3 for the subset
                childOfChild = recursiveID3(subset, removedBestAttribute,
                                            Label, EntropyGiniMajority,
                                            currentDepth + 1, treeDepth)
                # root adds child and child's child
                root.add_branch(myTree.Branch(v))
                root.get_child(v).add_BranchNode(childOfChild)

    return root