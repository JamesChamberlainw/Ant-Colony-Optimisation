# file created by James William Chamberlain on 2024-10-24 06:30:00 GMT 
# Ant Colony Optimisation (ACO) for the 0/1 knapsack problem

import csv
import numpy as np 
import pandas as pd

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# Ant Colonyy Optimisation (ACO) Parameters (default values used during initial testing and development)
population = 25                     # population size `p` (number of ants per itteration) 
evalutations_max = 10000            # maximum number of evaluations / itterations 
alpha = 1.0                         # Importance of pheromone 
beta = 0.2                         # Importance of heuristic 
evaporation_rate = 0.6              # Evaportation Rate                 - should be between 0.5 and 0.95 
pheromone_deposit_rate = 2.0        # Pheromone Deposit Rate            - should be between [TODO: find out]
initial_pheromone = 1               # Initial Pheromone on Edges (max)  - should be between [TODO: find out] 

# multipliers for fitness values
bias_to_new_best_solution = 2.0      # bias towards the best solution (default 2.0)


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

        capacity = float(rows[0][0].split(":")[1].strip())
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

def init_pheromone_matrix(size, initial_pheromone = initial_pheromone):
    """
    Initial pheromone matrix
    
    size:               int     number of points
    initial_pheromone:  float   initial pheromone multiplier (default 1.0)

    returns: np.array of shape (size, size)
    """

    # matrix = np.ones((size, size))

    # random values between 0 and 1
    matrix = np.random.rand(size, size) * initial_pheromone

    return matrix

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


def cdf_generate(pheromone_row, huristic_row, alpha = alpha, beta = beta):
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

    # convert to probability to a cdf (cumulative distribution function) for each row of the matrix
    probablilty = combined_matrix / combined_matrix.sum()
    cdf = np.cumsum(probablilty)
    return cdf


def ant(pheromones, huristics, weights, capacity):
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

    weight = weights[bag_id]        # starting weight in van from first bag
    

    # set column to 0 as the ant cannot revisit the same bag
    pheromones[:, bag_id] = 0   # do not revisit the same bag (initial)
    huristics[:, bag_id] = 0    # do not revisit the same bag (initial)

    # ant main loop 
    while weight < capacity:
        cdf_matrix = cdf_generate(pheromones[bag_id, :], huristics[bag_id, :])
        neu_bag_id = check_cdf(cdf_matrix, np.random.rand())
        if neu_bag_id == -1:
            # error should never happen and if it does occur this is a major issue so raise an error
            raise ValueError("Error: check_cdf failed to find a value in the cdf_matrix")
        
        weight += weights[neu_bag_id]   # add the weight of the bag to the van 
        weight = round(weight, 1)       # round to 1 decimal place to prevent floating point errors from accumulating (low chance of occurring) 

        # if still under capacity add the bag to the solution and deposit pheromone else its been found 
        if weight < capacity:
            # weight too high so will not fit in the van 
            deposit.append([neu_bag_id, bag_id])    # deposit pheromone between the two bags
            solution.append(neu_bag_id)             # add the bag to the solution
            bag_id = neu_bag_id                     # set the bag_id to the new bag for next iteration
            
            # Clear column so the ant cannot revisit the same bag
            pheromones[:, bag_id] = 0
            huristics[:, bag_id] = 0

    return solution, deposit

def update_pheromone_matrix(pheromone_matrix, all_deposits, fitness, deposit_rate = pheromone_deposit_rate):
    """
        Update the pheromone matrix based on where the ants have been

        fitness: % of total fitness for each solution takes up (pre-computed before this function)
        deposit_rate: the total amount of pheromone to deposit on each node visited (default 1.0)
    """

    for deposit in all_deposits:
        for i in range(len(deposit)):
            pheromone_matrix[deposit[0], deposit[1]] += deposit_rate*fitness

    return pheromone_matrix

def draw_heatmap(matrix):
    """
        Draw a heatmap of the matrix
    """

    sns.heatmap(matrix)
    plt.show()

def sum_val(solution, values):
    """
        Sum the fitness values of the solution
    """

    return sum([values[i] for i in solution])

"""
    Main Testing and Execution Area
"""

df, capacity, data = load_data() 

huristic_matrix = init_huristic_matrix(len(df), df)
# huristic_matrix = np.ones((len(df), len(df))) # TODO remove this line when huristic matrix is working
pheromone_matrix = init_pheromone_matrix(len(df))

weights = df['weight'].values
values = df['value'].values

evaluation_totals = 0

# logging variables 
best_fitness = 0
best_solution = []
best_deposit = []


# test_var = True # test variable for testing purposes only (if false testing will hide any left over testing code)

while evaluation_totals < evalutations_max:
    fitness_totals = []
    solutions = []
    deposits = []

    # matrix = (pheromone_matrix ** alpha) * (huristic_matrix ** beta)
    # prob_mat = matrix / matrix.sum()
    # draw_heatmap(prob_mat)


    for i in range(population):
        solution, deposit = ant(pheromone_matrix.copy(), huristic_matrix.copy(), weights, capacity)
        evaluation_totals += 1 # increment evaluation counter

        # save evluation metrics and logging data 
        fitness_totals.append(sum_val(solution, values))
        solutions.append(solution)
        deposits.append(deposit)

    # TODO if some adjustments are needed to be made to fitness it should be done here 
    # fitness(fitness_totals) 
    print("highest fitness in set = ", max(fitness_totals))

    # if the solution is better than all previously found solutions save it 
    #       plus give greater bias to the best solution in the pheromone matrix
    if max(fitness_totals) > best_fitness:
        best_fitness = max(fitness_totals)
        best_solution = solutions[fitness_totals.index(best_fitness)]
        best_deposit = deposits[fitness_totals.index(best_fitness)]

        print("\__> new best solution found = ", best_fitness)

        # adding bias to the best solution in the pheromone matrix
        fitness_totals[fitness_totals.index(best_fitness)] = fitness_totals[fitness_totals.index(best_fitness)] * bias_to_new_best_solution # adds bias to the best solution


    # adjust so that deposit ammount is proportional to fitness (to give higher priority to better solutions)
    fitness_totals = fitness_totals / sum(fitness_totals)

    # print("fitness weight = ", fitness_totals)

    # update pheromone matrix
    for i in range(population):
        pheromone_matrix = update_pheromone_matrix(pheromone_matrix.copy(), deposits[i], fitness_totals[i], pheromone_deposit_rate)

    # evaporation
    pheromone_matrix = pheromone_matrix * evaporation_rate

    # pheromone_matrix = update_pheromone_matrix(pheromone_matrix, all_deposits) # TODO adjust for multiple deposits
    all_deposits = [] # clear deposits


print("best solution = ", best_solution)
print("best fitness = ", best_fitness)
print("total fitness (re-eval) = ", sum_val(best_solution, values))
print("total weight of best solution = ", sum([weights[i] for i in best_solution]))
print("best deposit = ", best_deposit)
print("total evaluations = ", evaluation_totals)

print("total weight of all bags = ", sum(weights))
print("total value of all bags = ", sum(values))

# using pd.DataFrame 
weights = df['weight'].values
values = df['value'].values

# weight against value scatter plot (for visualisation of data)
plt.scatter(weights, values)
plt.xlabel('Weight')
plt.ylabel('Value')
plt.title('Weight vs Value')
plt.show()
