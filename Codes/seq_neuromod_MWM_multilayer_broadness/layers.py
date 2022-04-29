import numpy as np

class CA3_layer:

    def __init__(self, bounds_x, bounds_y, space_pc, offset,
                       rho, sigma):

        self.rho, self.sigma = rho, sigma

        if offset:
            x_pc = np.round(np.arange(bounds_x[0], bounds_x[1], space_pc)+space_pc/2,2)
            y_pc = np.round(np.arange(bounds_y[0], bounds_y[1], space_pc)+space_pc/2,2)
        else:
            x_pc = np.round(np.arange(bounds_x[0], bounds_x[1]+space_pc, space_pc),2)
            y_pc = np.round(np.arange(bounds_y[0], bounds_y[1]+space_pc, space_pc),2)
        
        xx, yy = np.meshgrid(x_pc, y_pc)

        self.pc = np.stack([xx,yy], axis=2).reshape(-1,2)
        self.N = self.pc.shape[0] #number of place cells

    def firing_rate(self, pos):

        return self.rho * np.exp(- ( (pos-self.pc)**2).sum(axis=1) / self.sigma**2 )
