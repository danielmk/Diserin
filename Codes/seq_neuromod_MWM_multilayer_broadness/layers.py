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


class CA1_layer:

    def __init__(self, bounds_x, bounds_y, space_pc, offset,
                       rho, sigma,
                       tau_m, tau_s, eps0,
                       chi,
                       theta, delta_u,
                       alpha,
                       N_ca3):

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
        self.N_in = N_ca3

        self.tau_m, self.tau_s, self.eps0 = tau_m, tau_s, eps0
        self.chi = chi
        self.theta, self.delta_u = theta, delta_u
        self.alpha = alpha

        self.epsp_decay = np.zeros((self.N, self.N_in))
        self.epsp_rise = np.zeros((self.N, self.N_in))

        self.last_spike_time = np.zeros(self.N)-1000
        

    def firing_rate(self, pos, spikes, weights, time):

        positional_firing_rate = self.rho * np.exp(- ((pos-self.pc)**2).sum(axis=1) / self.sigma**2 )
        
        if self.alpha != 0:

            self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + spikes*weights
            self.epsp_rise = self.epsp_rise - self.epsp_rise/self.tau_s + spikes*weights

            EPSP = self.eps0*(self.epsp_decay - self.epsp_rise)/(self.tau_m - self.tau_s)
            EPSP = EPSP.sum(axis=1)

            u = EPSP + self.chi*np.exp(-(time - self.last_spike_time)/self.tau_m)
            lambda_u = self.rho*np.exp((u-self.theta)/self.delta_u)

            return self.alpha*lambda_u + (1-self.alpha)*positional_firing_rate, u

        else:
            return positional_firing_rate, None


    def get_spikes(self, firing_rate, time):

        spikes = np.random.rand(self.N)<=firing_rate
        
        self.last_spike_time[spikes] = time
        
        self.epsp_rise[spikes] = 0
        self.epsp_decay[spikes] = 0

        return spikes