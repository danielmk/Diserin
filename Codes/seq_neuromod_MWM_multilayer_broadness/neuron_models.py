import numpy as np

class Place_Cells:

    def __init__(self, bounds_x, bounds_y, space_pc, offset,
                       rho, sigma):

        
        self.rho, self.sigma_sq = rho, sigma**2

        if offset:
            x_pc = np.round(np.arange(bounds_x[0], bounds_x[1], space_pc)+space_pc/2,2)
            y_pc = np.round(np.arange(bounds_y[0], bounds_y[1], space_pc)+space_pc/2,2)
        else:
            x_pc = np.round(np.arange(bounds_x[0], bounds_x[1]+space_pc, space_pc),2)
            y_pc = np.round(np.arange(bounds_y[0], bounds_y[1]+space_pc, space_pc),2)
        
        xx, yy = np.meshgrid(x_pc, y_pc)
        self.pc = np.stack([xx,yy], axis=2).reshape(-1,2)

        self.N = self.pc.shape[0]

    
    def get_firing_rates(self, pos):

        return self.rho * np.exp(- ( (pos-self.pc)**2).sum(axis=1) *(1./self.sigma_sq) )



class SRM0:

    def __init__(self, ):

        pass