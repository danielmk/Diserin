from logging import warning
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


def check_configuration(conf):

    bounds_x = conf['GEOMETRY']['bounds_x']
    bounds_y = conf['GEOMETRY']['bounds_y']
    rew_radius = conf['GEOMETRY']['rew_radius']
    rew_center= conf['GEOMETRY']['rew_center']

    maze_area = (bounds_x[1]-bounds_x[0])*(bounds_y[1]-bounds_y[0])
    rew_area = np.pi*rew_radius**2

    if 110<maze_area/rew_area<130:
        
        print("Warning: the ratio between the maze and the reward areas is not standard, \
               according to Vorheels and Williams, 2006.")

    if rew_center[0]<bounds_x[0] or rew_center[0]>bounds_x[1]\
       or rew_center[1]<bounds_y[0] or rew_center[1]>bounds_x[1]:

       print("Error: reward position is out of the maze!")
       exit()
    
