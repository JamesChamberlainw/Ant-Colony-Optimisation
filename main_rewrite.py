# A rewritten version of main.py to clean up the code and remove unnecessary functions

import csv
import numpy as np 

def load_data():
    """
    Load in BankProblem.txt, sort text file into useable dataframe
    

    returns: weights, values, capacity:
    capacity: float             # capacity of the van
    weights: list of floats     # weights of the bags 
    values: list of floats      # values of the bags 

    Note: Bag index for weights/values is bag number - 1 (i.e., bag 1 is index 0)
    """

    weights = []
    values = []

    with open('BankProblem.txt','r') as f:
        reader = csv.reader(f, delimiter = '|')
        rows = list(reader)

        capacity = int(rows[0][0].split(":")[1].strip())
        rows.pop(0) # drop the first row (capacity)

        # bag index (for those too lazy to count or add one during operations aka me)
        i = 1 
        
        while rows != []:
            rows.pop(0) # drop bag number as its not needed
            weight = float(rows[0][0].split(":")[1].strip())
            weights.append(weight)
            rows.pop(0) # drop weight
            value = float(rows[0][0].split(":")[1].strip())
            values.append(value)
            rows.pop(0) # drop value
            
            i += 1
             
    return weights, values, capacity

def init_huristic_matrix(size, weight, value):
    """
    Initial huristic matrix 

    produces a matrix using the dataset to represent the value per weight ratio (vpw) 
    Note: a distance matrix is not produced as all vertical columns are the same as there is no distance relationship between points

    size:       int number of points
    weight:     corresponding weight
    value:      corresponding value 
    
    returns: np.array of shape (size, size)
    """
    
    # default values (in-case something goes wrong) 
    matrix = np.zeros((size, size))

    vpw = [value[i] / weight[i] for i in range(size)]

    for i in range(size):
        vpw.append(value[i] / weight[i])

    for i in range(size):
        vpw[i] = (vpw[i] - min(vpw)) / (max(vpw) - min(vpw))

    for i in range(size):
        for j in range(size):
            matrix[i][j] = vpw[j]

    # do not revisit the same bag    
    np.fill_diagonal(matrix, 0) 


    return matrix