'''
    @author - Angel Dhungana
    Given a filename, this file just reads the files and returns the dataset
'''
import csv


def readDataDesc(fileName):
    '''
        Read Attributes name from the Data Description, only for CarDataSet
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