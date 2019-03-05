# README
## How to run my LMS Implementation?
 - You will find a file name runLMS.py in the folder.
 
 - Open the file and input necessary items in the main Function:
    - Input Filename of your dataSet both training and testing; with path if its in different folder.
      - Make Sure the file is a csv file and includes both X and Y column with Y column as a last column
    - Enter learning rate
    - Enter num_iterations, number of times you want to iterate
    - Give tolerance, default is 0.00006, this is the number before the algorithm converges
  
 - If you run the program by entering all above it print out weights and cost function for Test Data
    - For both Batch and Stochastic Gradient Descent