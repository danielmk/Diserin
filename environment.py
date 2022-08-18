import numpy as np

class MWM:

    def __init__(self, conf):
        
        self.bounds_x, self.bounds_y = conf['bounds_x'], conf['bounds_y']

        self.rew_center, self.rew_radius = conf['rew_center'], conf['rew_radius']

        self.obstacle = conf['obstacle_1']

        if self.obstacle:

            self.obs_x = conf['obstacle_bounds_x']
            self.obs_y = conf['obstacle_bounds_y']

        self.obstacle_2 = conf['obstacle_2']

        if self.obstacle_2:

            self.obs_x_2 = conf['obstacle_bounds_x_2']
            self.obs_y_2 = conf['obstacle_bounds_y_2']

    def update_position(self, position, step):

        new_position = position + step

        if np.sum((new_position-self.rew_center)**2)<=self.rew_radius**2:

            reward_found = True
            wall_hit = False

            return new_position, wall_hit, reward_found


        if (
            not self.bounds_x[0] <= new_position[0] <= self.bounds_x[1]
            or not self.bounds_y[0] <= new_position[1] <= self.bounds_y[1]
            or (
                self.obstacle
                and self.obs_x[0] <= new_position[0] <= self.obs_x[1]
                and self.obs_y[0] <= new_position[1] <= self.obs_y[1]
            )
            or (
                self.obstacle_2
                and self.obs_x_2[0] <= new_position[0] <= self.obs_x_2[1]
                and self.obs_y_2[0] <= new_position[1] <= self.obs_y_2[1]
            )
        ):

            reward_found = False
            wall_hit = True

            return position, wall_hit, reward_found

        reward_found = False
        wall_hit = False
        return new_position, wall_hit, reward_found
