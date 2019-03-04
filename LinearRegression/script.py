'''
    @author - Angel Dhungana
    Script file that runs the LMS and bagging100/RandomForest
'''
import sys
sys.path.insert(0, 'LinearRegression/testConcreteLMS.py')
#sys.path.insert(1, 'EnsembleLearning/trainTestBankEnsemble.py')
import testConcreteLMS
import trainTestBankEnsemble


def main():
    testConcreteLMS.main()
    trainTestBankEnsemble.main()


if __name__ == "__main__":
    main()