import pandas as pd
import numpy as np
import ga

# split a timeseries dataset into train/test partitions
def split_dataset(data, data_split=0.2):
    data = np.array(data)
    #if number of observations specified manually
    if data_split > 1:
        # split using observation number
        train, test = data[0:-data_split], data[-data_split:]        
    #else use data percentage
    else:
        data_index = round(len(data)*(1-data_split))
        train = data[0:data_index, :]
        test = data[data_index:, :]
    return train, test


# define input sequence
data  = pd.read_csv('../Data/Scenario1.csv', header=0, index_col=0)
train, test = split_dataset(data, 1100)


# Inputs of the equation.
input_value = [4, 16, 32, 8, 2]
# Number of the weights we are looking to optimize.
num_inputs = 1
sol_per_pop = 8


# Defining the population size.

pop_size = (sol_per_pop,num_inputs) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

#Creating the initial population.

new_population = np.int64(np.random.uniform(low=0, high=32, size=pop_size))

num_generations = 5

num_parents_mating = 4

