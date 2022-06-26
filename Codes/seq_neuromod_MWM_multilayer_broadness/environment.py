import numpy as np

class MWM:

    def __init__(self, bounds_x, bounds_y, reward_center, reward_radius, dx,
                       obstacle, obstacle_bounds_x, obstacle_bounds_y,
                       obstacle_2, obs_x_2, obs_y_2):
        
        self.bounds_x, self.bounds_y = bounds_x, bounds_y

        self.reward_center, self.reward_radius = reward_center, reward_radius

        self.dx = dx

        self.obstacle = obstacle

        if self.obstacle:

            self.obs_x = obstacle_bounds_x
            self.obs_y = obstacle_bounds_y

        self.obstacle_2 = obstacle_2

        if self.obstacle_2:

            self.obs_x_2 = obs_x_2
            self.obs_y_2 = obs_y_2

    def update_position(self, position, step):
        
        new_position = position + step

        if np.sum((new_position-self.reward_center)**2)<=self.reward_radius**2:

            reward_found = True
            wall_hit = False
            
            return new_position, wall_hit, reward_found,


        if  not (self.bounds_x[0]<=new_position[0]<=self.bounds_x[1]\
             and self.bounds_y[0]<=new_position[1]<=self.bounds_y[1])\
            or (self.obstacle and self.obs_x[0]<=new_position[0]<=self.obs_x[1]\
             and self.obs_y[0]<=new_position[1]<=self.obs_y[1])\
            or (self.obstacle_2 and self.obs_x_2[0]<=new_position[0]<=self.obs_x_2[1]\
             and self.obs_y_2[0]<=new_position[1]<=self.obs_y_2[1]):
        
            reward_found = False
            wall_hit = True

            return position, wall_hit, reward_found

        reward_found = False
        wall_hit = False
        return new_position, wall_hit, reward_found

        