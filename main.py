# file created by James William Chamberlain on 2024-10-24 06:30:00 GMT 

import csv
import numpy as np 
import pandas as pd


def load_data():
    """
    Load in BankProblem.txt, sort text file into useable dataframe
    
    returns: pd.DataFrame with columns 'weight' and 'value'
    """
    df = pd.DataFrame(columns=['weight', 'value'])

    data = []

    with open('BankProblem.txt','r') as f:
        reader = csv.reader(f, delimiter = '|')
        rows = list(reader)

        capacity = rows[0][0].split(":")[1].strip()
        rows.pop(0) # drop the first row (capacity)

        # bag index (for those too lazy to count or add one during operations aka me)
        i = 1 
        
        while rows != []:
            rows.pop(0) # drop bag number as its not needed
            weight = float(rows[0][0].split(":")[1].strip())
            rows.pop(0) # drop weight
            value = float(rows[0][0].split(":")[1].strip())
            rows.pop(0) # drop value

            data.append([i, weight, value])
            new_df = pd.DataFrame([[weight, value]], columns=['weight', 'value'])
            df = pd.concat([df, new_df], ignore_index=True) # Make sure its True as it matches bag index 
            
            i += 1
             
    return df, capacity, data

def init_huristic_matrix(size, df = None):
    """
    Initial huristic matrix 

    produces a matrix using the dataset to represent the value per weight ratio (vpw) 
    Note: a distance matrix is not produced as all vertical columns are the same as there is no distance relationship between points

    size: int number of points
    df: data with columns 'weight' and 'value'
    
    returns: np.array of shape (size, size)
    """
    
    # default values (in-case something goes wrong) 
    matrix = np.ones((size, size))

    if df is not None:
        df['vpw'] = df['value']/df['weight']
        # normalise vpw
        df['vpw'] = (df['vpw'] - df['vpw'].min()) / (df['vpw'].max() - df['vpw'].min())

        # add values into matrix
        # this feels wrong but there is no better metric to exploit and no distance relation between the points 
        for i in range(size):
            for j in range(size):
                matrix[i][j] = df['vpw'][j]

    # cannot revisit itself so diagonal is 0
    np.fill_diagonal(matrix, 0) 

    return matrix

def init_pheromone_matrix(size):
    """
    Initial pheromone matrix
    
    size: int number of points
    
    returns: np.array of shape (size, size)
    """

    matrix = np.ones((size, size))

    return matrix

def 



def main():
    # initialise the dataset 
    df, capacity, data = load_data() 

    # generate vpw (value per weight ratio)
    # vpw = value / weight 
    # chosen metric for default bag value as it combines the two values making it easier to compare 


    # initialise pheromone and distance matrices 
    distance_matrix = init_distance_matrix(len(df), df)
    pheromone_matrix = init_pheromone_matrix(len(df))

    print(distance_matrix)
    print(pheromone_matrix)
    # print(capacity)

    # print(sum(df['weight'])) 
    # print(df.sort_values('weight', ascending=False).head())

    print(df.head())
    print(df.describe())

main()
