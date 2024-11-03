# file created by James William Chamberlain on 2024-10-24 06:30:00 GMT 
# Ant Colony Optimisation (ACO) for the 0/1 knapsack problem

import csv
import numpy as np 
import pandas as pd

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# Ant Colonyy Optimisation (ACO) Parameters (default values used during initial testing and development)
population = 10                         # population size `p` (number of ants per generation) 
evalutations_max = 10000                # maximum number of evaluations / itterations 
alpha = 1.0                             # Importance of pheromone           - if this is 1 then the algorithm will be heavily biased towards the pheromone
beta = 1.0                              # Importance of heuristic           - if this is 1 then the algorithm will be heavily biased towards the heuristic
evaporation_rate = 0.8                  # Evaportation Rate                 - should be between 0.5 and 0.95 
pheromone_deposit_rate = 1.0            # Pheromone Deposit Rate - how much pheromone is deposited on the edge based on the fitness of the solution 
initial_pheromone = 1.0                 # Initial Pheromone on Edge/s (max)  - should be between [TODO: find out] 

# multipliers for fitness values
bias_to_new_best_solution = 1.0        # bias towards the best solution (default 1.0 - no bias) so nothing is added to the best solution fitness value

# diversification & recall mechanics 
diversification = True                 # diversification (default False) if True then the pheromone matrix is diversified to prevent local optima    
diversification_multiplier = 0.2        # multiplier for diversification (0.0 - means no diversification mechanics used)
diversification_no_change = 10          # number of generations after a optima has been found without any improvements made before diversification is used 
# NOTE: recall is only used if diversification is enabled
recall = True                           # recall the best solution found so far to the colony (default False) if diversification fails to find a better solution 
recall_no_change = 3                    # number of diversification cycles before the best solution is recalled to the colony  # 3 = 30 generations after last optima found before recall is used
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

    matrix = np.ones((size, size))

    # random values between 0 and 1
    # matrix = np.random.rand(size, size) * initial_pheromone

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

def sum_val(solution, values):
    """
        Sum the fitness values of the solution
    """

    return sum([values[i] for i in solution])

"""
    Main Testing and Execution Area
"""

def main():
    """
        Main function 
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

    best_pheromone_matrix = pheromone_matrix.copy()
    best_huristic_matrix = huristic_matrix.copy()


    # diversification variables
    diversification_counter = 0
    recall_counter = 0

    while evaluation_totals < evalutations_max:
        fitness_totals = []
        solutions = []
        deposits = []

        for i in range(population):
            solution, deposit = ant(pheromone_matrix.copy(), huristic_matrix.copy(), weights, capacity)
            evaluation_totals += 1 # increment evaluation counter

            # save evluation metrics and logging data 
            fitness_totals.append(sum_val(solution, values))
            solutions.append(solution)
            deposits.append(deposit)

        print("highest fitness in set = ", max(fitness_totals))

        if max(fitness_totals) > best_fitness:
            best_fitness = max(fitness_totals)
            best_solution = solutions[fitness_totals.index(best_fitness)]
            best_deposit = deposits[fitness_totals.index(best_fitness)]
            best_pheromone_matrix = pheromone_matrix.copy()
            best_huristic_matrix = huristic_matrix.copy()

            print("\__> new best solution found = ", best_fitness)

            # bias towards the best solution
            fitness_totals[fitness_totals.index(best_fitness)] = fitness_totals[fitness_totals.index(best_fitness)] * bias_to_new_best_solution # (default 1.0 - no bias)
            
            # reset counters
            diversification_counter = 0
            recall_counter = 0

        elif diversification and diversification_counter > diversification_no_change:
            # diversification
            print("\__> diversification")
            diversification_counter = 0 # rest diversification counter 

            salt = np.random.rand(100, 100) * diversification_multiplier # random salting between 0 an 1 * multiplier
            pheromone_matrix = pheromone_matrix + salt # add together to diversify the pheromone matrix

            if recall and recall_counter >= recall_no_change:
                print("\__> recall")

                # Here the best solution is recalled and scaled back 
                pheromone_matrix = best_pheromone_matrix.copy() / best_pheromone_matrix.copy().sum()     # adjusted so pheromone matrix is between 0 and 1 
                huristic_matrix = best_huristic_matrix.copy()                                            # this will increases the huristic value of the best solution
                recall_counter = 0 # reset recall counter
            else:
                recall_counter += 1   
        else:
            diversification_counter += 1

        # adjust values so that it favours the better solutions (higher fitness)
        # ignore the lower fitness values and drop from the pool 
        # fitness_mean = sum(fitness_totals) / len(fitness_totals)
        # fitness_totals = [i if i > fitness_mean else 0 for i in fitness_totals] # if worse than fitness mean then set to 0 

        # adjust so that deposit ammount is proportional to fitness (to give higher priority to better solutions)
        fitness_totals = fitness_totals / sum(fitness_totals)


        # update pheromone matrix
        for i in range(population):
            pheromone_matrix = update_pheromone_matrix(pheromone_matrix.copy(), deposits[i], fitness_totals[i], pheromone_deposit_rate)

        # evaporation
        pheromone_matrix = pheromone_matrix * evaporation_rate

    return best_solution, best_fitness, best_deposit, values, weights, evaluation_totals

"""
    Plotting functions 
        a list of functions to plot the data in various ways
"""

def draw_val_weight_scatter(weights, values, best_solution=[]):
    """
        Draws a scatter plot highlighting the best solution as green and unselected as red

        weights:        list of weights
        values:         list of values
        best_solution:  list of selected values (default empty list for just a plot of all values)
    """

    selected_values = []
    selected_weights = []

    # Select the points that will be highlighted
    if best_solution != []:
        # split into two lists (selected and not selected)    
        selected_weights = [weights[i] for i in best_solution]
        selected_values = [values[i] for i in best_solution]

        # drop the selected values from the original list drop selected weights and values
        weights = [i for j, i in enumerate(weights) if j not in best_solution]
        values = [i for j, i in enumerate(values) if j not in best_solution]

    # plot 
    plt.scatter(selected_weights, selected_values, color='green')   # selected 
    plt.scatter(weights, values, color='red')                       # not selected (default colour)
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.title('Weight vs Value')
    plt.show()

def draw_heatmap(matrix):
    """
        Draw a heatmap of the matrix 
            useful for debugging and visualising the data during development 

        matrix: np.array of shape (size, size)
    """

    sns.heatmap(matrix)
    plt.show()

"""
    Testing Area
"""

def select_raondom(display = False):
    """
        Select a random solution to test the ant function
    """

    df, capacity, data = load_data() 

    huristic_matrix = np.ones((100, 100)) 
    pheromone_matrix =  np.ones((100, 100))

    weights = df['weight'].values
    values = df['value'].values

    solution, _ = ant(pheromone_matrix.copy(), huristic_matrix.copy(), weights, capacity)

    if display:
        print("solution = ", solution)

        draw_val_weight_scatter(weights, values, solution)
        print("total value of solution = ", sum_val(solution, values))
        print("total weight of solution = ", sum([weights[i] for i in solution]))

    return solution

"""
    Initialisation
"""


best_solution, best_fitness, best_deposit, values, weights, evaluation_totals = main()

print("best solution = ", best_solution)
print("best fitness = ", best_fitness)
print("total fitness (re-eval) = ", sum_val(best_solution, values))
print("total weight of best solution = ", sum([weights[i] for i in best_solution]))
print("best deposit = ", best_deposit)
print("total evaluations = ", evaluation_totals)

print("total weight of all bags = ", sum(weights))
print("total value of all bags = ", sum(values))

draw_val_weight_scatter(weights, values, best_solution)

# select_raondom() # testing function
# select_raondom() # testing function
# select_raondom() # testing function
# select_raondom() # testing function