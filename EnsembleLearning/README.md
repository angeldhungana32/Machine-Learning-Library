# README
## How to run my Ensemble Learning Implementation?
 - You will find a file name runEnsembleLearning.py in the folder.
 
 - Open the file and input necessary items in the main Function:
    - Input Filename of your dataSet both training and testing; with path if its in different folder.
      - Make Sure the file is a csv file and includes both X and Y column with Y column as a last column
    - Provide Attribute names
    - Provide target label
    - Enter learning rate
    - Enter num_iterations, number of times you want to iterate, default = 100
    - If you are running Random Forest,
        - Provide subset of features you want to split on
        - Provides indexes of features as subset
        - Provide num of trees you want
  
 - If you run the program by entering all above it print out error rate by each classifier on Test data
 - Uncomment the Bagging or AdaBoost or RandomForest functions to run as needed