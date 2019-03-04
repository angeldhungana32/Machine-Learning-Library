'''
    @author - Angel Dhungana
    Batch Gradient Descent
'''
import numpy as np
import matplotlib.pyplot as plt
import copy


def analytical(X, Y):
    '''
        Analytical Calculation for Last Question
    '''
    final = (np.linalg.inv(X @ X.T) @ X).T @ Y.T
    return final