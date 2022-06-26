import numpy as np

def get_starting_position(starting_position_option):

    if starting_position_option=='origin':

        return np.array([0.,0.])

    if starting_position_option=='random':

        return np.random.rand(2)*4 - 2


    if starting_position_option=='upper-right':
        
        return np.array([1.5, 1.5])

    print("Starting position option non valid!")
    exit()