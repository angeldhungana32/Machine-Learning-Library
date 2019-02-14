'''
    @author  Angel Dhungana

    myTree a file that has Node class and Branch Class
    Each node has list of Branch Objects, and
    Each Branch object has a childNode that they point to 
'''


class Node(object):
    '''
        Node class that holds the Node object for tree
        All the branches of the node are in children
    '''

    def __init__(self, data):
        '''
            Creates a Node given data
        '''
        self.data = data
        self.branches = []

    def add_branch(self, obj):
        '''
            Adds a branch to the node
        '''
        self.branches.append(obj)

    def get_child(self, obj):
        '''
            Retrieves the children with data obj
        '''
        for x in self.branches:
            if x.data == obj:
                return x

    def has_child(self):
        if len(self.branches) == 0:
            return False
        else:
            return True

    def __str__(self, level=0):
        '''
            Function that returns the string representation of the tree
        '''
        ret = "\t" * level + repr(self.data) + "\n"
        for child in self.branches:
            ret += child.get_BranchNode().__str__(level + 1)
        return ret


class Branch(object):
    '''
        Each edge of the Node
    '''

    def __init__(self, data):
        '''
            Creates a Edge given data
        '''
        self.data = data
        self.child = ''

    def add_BranchNode(self, obj):
        '''
            Every branch has a child Node
        '''
        self.child = obj

    def get_BranchNode(self):
        '''
            Return child node
        '''
        return self.child