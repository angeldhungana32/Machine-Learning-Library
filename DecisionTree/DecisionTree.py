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
    indx = getIndexOfAttributes(Attributes, target)
    count = 0
    current = ''
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
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = {}
    for x in S:
        currLbl = x[indx]
        if not currLbl in dictionary:
            dictionary[currLbl] = 1
        else:
            dictionary[currLbl] += 1
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
    for attr in Attributes:
        if attr != target:
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
    mainEntropy = calculateEntropy(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrEntropy = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    for num in dictionary.keys():
        p = dictionary[num] / total
        subset = []
        for col in S:
            if col[indx] == num:
                subset.append(col)
        attrEntropy += p * calculateEntropy(subset, Attributes, target)
    return mainEntropy - attrEntropy


def getInformationGainGiniIndex(S, Attributes, target, attr):
    '''
        Helper Function that Calculates Gain using Gini Index
    '''
    mainGiniInd = calculateGiniIndex(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrGiniInd = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    for num in dictionary.keys():
        p = dictionary[num] / total  #probability
        subset = []
        for col in S:
            if col[indx] == num:
                subset.append(col)
        attrGiniInd += p * calculateGiniIndex(subset, Attributes, target)
    return mainGiniInd - attrGiniInd


def getInformationGainMajorityError(S, Attributes, target, attr):
    '''
        Helper Function that Calculates Gain using Majority Error
    '''
    mainMajorityError = calculateMajorityError(S, Attributes, target)
    indx = getIndexOfAttributes(Attributes, attr)
    dictionary = helperFrequencyFinder(S, indx)[0]
    attrMajoritErr = 0.0
    total = helperFrequencyFinder(S, indx)[1]
    for num in dictionary.keys():
        p = dictionary[num] / total  #probability S/|S|
        subset = []
        for col in S:
            if col[indx] == num:
                subset.append(col)
        attrMajoritErr += p * calculateMajorityError(subset, Attributes,
                                                     target)
    return mainMajorityError - attrMajoritErr


def helperFrequencyFinder(S, indx):
    '''
        Helper Function that finds the frequency of attributes[indx] in S 
    '''
    dictionary = {}
    total = 0
    for col in S:
        if not col[indx] in dictionary:
            dictionary[col[indx]] = 1
            total += 1
        else:
            dictionary[col[indx]] += 1
            total += 1
    return dictionary, total


def calculateEntropy(S, Attributes, target):
    '''
        Calculates Entropy (-P * log_2(P)) + ..... +
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = helperFrequencyFinder(S, indx)[0]
    total = len(S)
    entropy = 0.0
    for num in dictionary.values():
        entropy += (-num / total) * (math.log2(num / total))
    return entropy


def calculateGiniIndex(S, Attributes, target):
    '''
        Calculates Gini Index 1 - (p_1^2 + p_2^2+ ....)
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = helperFrequencyFinder(S, indx)[0]
    total = len(S)
    giniIndex = 0.0
    for num in dictionary.values():
        giniIndex += (num / total) * (num / total)
    return 1 - giniIndex


def calculateMajorityError(S, Attributes, target):
    '''
        Calculates Majority Error 1 - (ME(target)) 
        Example 3/5 and 2/5 then, majority error is 1 - 3/5
    '''
    indx = getIndexOfAttributes(Attributes, target)
    dictionary = helperFrequencyFinder(S, indx)[0]
    total = len(S)
    majorityError = max(dictionary.values())
    return 1 - (majorityError / total)


def getValuesOfA(S, Attributes, A):
    '''
        Returns the possible values that A has
    '''
    ind = getIndexOfAttributes(Attributes, A)
    vals = []
    for col in S:
        if col[ind] not in vals:
            vals.append(col[ind])
    return vals


def getSubset(S, Attributes, A, v):
    '''
        Returns subset of examples in S with A=v
    '''
    subset = []
    ind = getIndexOfAttributes(Attributes, A)
    for col in S:
        if (col[ind] == v):
            # Removing the column A from S
            subset.append([col[i] for i in range(0, len(col)) if i != ind])
    return subset


def ID3(S, Attributes, Label, EntropyGiniMajority, treeDepth):
    '''
        Main ID3 that calls the helper recursive ID3
    '''
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
            subset = getSubset(S, Attributes, A, v)

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