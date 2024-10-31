# file created by James William Chamberlain on 2024-10-24 06:30:00 GMT 
# Ant Colony Optimisation (ACO) for the 0/1 knapsack problem

import csv
import numpy as np 
import pandas as pd

# Ant Colonyy Optimisation (ACO) Parameters (default values used during initial testing and development)
num_ants = 50
num_iterations = 100
alpha = 1.0       # Importance of pheromone
beta = 2.0        # Importance of heuristic
evaporation_rate = 0.5
initial_pheromone = 1.0


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

def stop_condition(itter):
    """
    Stop condition / Termination Crierion for the algorithm 
    
    The termination criterion for this assignment is a total of 10,000 fitness evaluations

    input: int itteration

    returns: bool
    """
    # TODO: add local optima check as otherwise this has no reason to be a function 

    if itter < 10000:
        return True
    
    return False

def check_cdf(cdf_row, rand):
    """
    Check the cdf to see if the random number is within the range
    
    cdf_row: np.array   containing the cdf values
    rand: float         random number to check against the cdf_row 

    returns: int index of the cdf_row (Note bag id will be index + 1)
    """

    # find index of the cdf_row that the random number is within the range of
    for i in range(len(cdf_row)):

        if rand < cdf_row[i]:
            return i

    # due to floaing point errors this can lead to .9999999999999999 instead of 1.0 so this is a catch all for that
    # so if return maximum index (last bag) if the random number is 1.0
    return len(cdf_row) - 1


def cdf_generate(pheromone_row, huristic_row, alpha, beta):
    """
    Combines a row of the pheromone and huristic matrix (all or one row*) so the ant can make movement decisions
    
    input: 
    pheromone_row   np.array    row or full matrix of pheromone values (ideally a row for reduced computation) 
    huristic_row    np.array    row or full matrix of huristic values (ideally a row for reduced computation)
    alpha           float       represents the importance of pheromone
    beta            float       represents the importance of huristic

    returns: np.array of shape (size, size)*
    """

    # combine huristic and pheromone matrix with weights on each
    combined_matrix = (pheromone_row ** alpha) * (huristic_row ** beta)

    print(combined_matrix.shape)

    # convert to probability to a cdf (cumulative distribution function) for each row of the matrix
    probablilty = combined_matrix / combined_matrix.sum()
    cdf = np.cumsum(probablilty)

    # probablilty_matrix = combined_matrix / combined_matrix.sum(axis=1)[:, None]
    # cdf = np.cumsum(probablilty_matrix, axis=1)

    return cdf


# def ant_solution(cdf_matrix, weights, capacity):
#     """
#     A Single Ant Solution is generated by the ant using the pheromone and huristic matrix
#     """
#     deposit = [] # all positions visited by the ant (to deposit pheromone)
#     solution = [] # the solution the ant has found

#     # random start point 
#     bag_id = np.random.randint(0, 100)  # generates a random number between between 0 <= x < 100 (bag 1-100)
#     solution.append(bag_id)             

#     weight = 0 + weights[bag_id]        # starting weight in van from first bag 

#     # set column to 0 as the ant cannot revisit the same bag
#     cdf_matrix[:, bag_id] = 0 # do not revisit the same bag (initial)

#     # while the ant has not visited all points
#     while weight < capacity:
#         neu_bag_id = check_cdf(cdf_matrix[bag_id, :], np.random.rand())
#         if neu_bag_id == -1:
#             # error should never happen and if it does occur this is a major issue so raise an error
#             raise ValueError("Error: check_cdf failed to find a value in the cdf_matrix")
        
#         # will the bag fit the van for another iteration? 
#         if weight + weights[neu_bag_id] > capacity:
#             # weight too high so break 
#             # a better solution outside of ACO would skip this until it finds a bag that would fit in the van
#             # else if None brea and return the solution
#             break
#         else:
#             deposit.append([neu_bag_id, bag_id])    # deposit pheromone between the two bags
#             solution.append(neu_bag_id)             # add the bag to the solution
#             weight += weights[neu_bag_id]           # add the weight of the bag to the van
#             bag_id = neu_bag_id                     # set the bag_id to the new bag for next iteration

#             # Clear column so the ant cannot revisit the same bag
#             cdf_matrix[:, bag_id] = 0
        
#         return solution, deposit


def ant(pheromones, huristics, weights, capacity, alpha = alpha, beta = beta):
    """ 
        TODO: Add docstring
    """

    # Error checking
    if pheromones.shape != huristics.shape:
        raise ValueError("Error: pheromones and huristics are not the same shape")
    
    # Ant variables
    deposit = [] # all positions visited by the ant (to deposit pheromone)
    solution = [] # the solution ids the ant has found

    # random start point
    bag_id = np.random.randint(0, 100)  # generates a random number between between 0 <= x < 100 (bag 1-100)
    solution.append(bag_id)

    weight = 0 + weights[bag_id]        # starting weight in van from first bag

    # set column to 0 as the ant cannot revisit the same bag
    pheromones[:, bag_id] = 0   # do not revisit the same bag (initial)
    huristics[:, bag_id] = 0    # do not revisit the same bag (initial)

    # ant main loop 
    while weight < capacity:
        cdf_matrix = cdf_generate(pheromones, huristics, alpha, beta)
        neu_bag_id = check_cdf(cdf_matrix[bag_id, :], np.random.rand())
        if neu_bag_id == -1:
            # error should never happen and if it does occur this is a major issue so raise an error
            raise ValueError("Error: check_cdf failed to find a value in the cdf_matrix")
        
        weight += weights[neu_bag_id] # add the weight of the bag to the van 

        # if still under capacity add the bag to the solution and deposit pheromone else its been found 
        if weight < capacity:
            # weight too high so will not fit in the van 
            deposit.append([neu_bag_id, bag_id])    # deposit pheromone between the two bags
            solution.append(neu_bag_id)             # add the bag to the solution
            bag_id = neu_bag_id                     # set the bag_id to the new bag for next iteration
            
            # Clear column so the ant cannot revisit the same bag
            pheromones[:, bag_id] = 0
            huristics[:, bag_id] = 0





df, capacity, data = load_data() 

huristic_matrix = init_huristic_matrix(len(df), df)
pheromone_matrix = init_pheromone_matrix(len(df))

weights = df['weight'].values
values = df['value'].values

# Combine the pheromone and huristic matrix so the ant can make movement decisions
combined_matrix = (pheromone_matrix ** alpha) * (huristic_matrix ** beta)

# # convert to probability to a cdf (cumulative distribution function) for each row of the matrix
# probablilty_matrix = combined_matrix / combined_matrix.sum(axis=1)[:, None]
# cdf = np.cumsum(probablilty_matrix, axis=1)

# print(cdf[0, :])

# total = probablilty_matrix[3, :].sum() # should total 1 or 0.9999999999999999 (in the case of a floating point error - so rounding will fix this) 
# print(total)

# row = combined_matrix[0, :]
# print(row)

cdf = cdf_generate(pheromone_matrix[0, :], huristic_matrix[0, :], alpha, beta)
print(cdf)
print(cdf[99])

# solution, deposit = ant_solution(cdf_matrix, weights, capacity)

# print(solution)
# print(deposit)


"""
    Testing 
"""

# testing cdf_generate
cdf = cdf_generate(pheromone_matrix[0, :], huristic_matrix[0, :], alpha, beta)
print(cdf)      # array (100, ) 
print(cdf[99])  # 1.0 

# testing check_cdf based on cdf_generate
print(check_cdf(cdf, 0.9999999999999999)) # 99 (bag 100)
print(check_cdf(cdf, 0.0)) # this is row 0 so (0, 0) is 0.0 so it will select (0, 1) (bag 2)

# gen row 1 
cdf = cdf_generate(pheromone_matrix[1, :], huristic_matrix[1, :], alpha, beta)
print(cdf)      # array (100, )
print(cdf[99])  # 1.0

# testing check_cdf based on cdf_generate
print(check_cdf(cdf, 0.9999999999999998)) # 99 (bag 100)
print(cdf[99])
print(check_cdf(cdf, 0.0)) # 0 (bag 1)



