from turtle import position
import numpy as np


class BCM:

    def __init__(self, N,  memory_factor, weight_decay,):

        self.num_neurons = N
        self.weight_decay = weight_decay
        self.memory_factor = memory_factor

        self.thetas = np.ones(N)

    def update(self, x, y, weights):
        """
        x is the pre-synaptic activity
        y is the post-synaptic acitivity
        """

        self.thetas = self.memory_factor*self.thetas + (1-self.memory_factor)*(y**2)

        x = x.reshape(1,-1)
        y = y.reshape(-1, 1)
        dW = y(y-self.theta)*x - self.weight_decay * weights

        return dW


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

        # Internal state
        self.spikes = np.zeros(self.N)
        self.firing_rates = np.zeros(self.N)

    def update_activity(self, pos):
        
        self.firing_rates = self.rho * np.exp(- ( (pos-self.pc)**2).sum(axis=1) / self.sigma**2 )
        
        self.spikes = np.random.rand(self.N) <= self.firing_rates



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

        # activity state
        self.spikes = np.zeros(self.N)
        self.firing_rates = np.zeros(self.N)

    def update_activity(self, pos, spikes, weights, time):

        positional_firing_rate, lambda_u = 0, 0

        if self.alpha != 1:  
            positional_firing_rate = self.rho * np.exp(- ((pos-self.pc)**2).sum(axis=1) / self.sigma**2 )
        
        if self.alpha != 0:

            self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + spikes*weights
            self.epsp_rise = self.epsp_rise - self.epsp_rise/self.tau_s + spikes*weights

            EPSP = self.eps0*(self.epsp_decay - self.epsp_rise)/(self.tau_m - self.tau_s)
            EPSP = EPSP.sum(axis=1)

            u = EPSP + self.chi*np.exp(-(time - self.last_spike_time)/self.tau_m)
            lambda_u = self.rho*np.exp((u-self.theta)/self.delta_u)
        
        self.firing_rates = self.alpha*lambda_u + (1-self.alpha)*positional_firing_rate
        self.spikes = np.random.rand(self.N) <= self.firing_rates
    
        self.last_spike_time[self.spikes] = time

        self.epsp_rise[self.spikes] = 0
        self.epsp_decay[self.spikes] = 0


class Action_layer:

    def __init__(self, N,
                       tau_m, tau_s, eps0,
                       chi,
                       rho, theta, delta_u,
                       tau_gamma, v_gamma,
                       N_ca1):

        self.N = N
        self.tau_m, self.tau_s, self.eps0 = tau_m, tau_s, eps0
        self.chi = chi
        self.rho, self.theta, self.delta_u = rho, theta, delta_u
        self.N_in = N_ca1
        
        self.epsp_decay = np.zeros((N, N_ca1 + N))
        self.epsp_rise = np.zeros((N, N_ca1 + N))
        self.epsp_resetter = np.zeros((N, N_ca1 + N ))
        self.last_spike_time = np.zeros(N) - 1000

        self.tau_gamma, self.v_gamma = tau_gamma, v_gamma
        self.rho_decay = np.zeros(N)
        self.rho_rise = np.zeros(N)

        self.Canc = np.ones([N_ca1+N, N])
        self.last_spike_post=np.zeros(N)-1000

        self.firing_rates = np.zeros(self.N)
        self.spikes = np.zeros(self.N)
        self.instantaneous_firing_rates = np.zeros(self.N)

    def update_activity(self, spikes_ca1, weights, time):
        
        cat_spikes = np.concatenate([spikes_ca1, self.spikes])

        self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + np.multiply(cat_spikes,weights)
        self.epsp_rise =  self.epsp_rise  - self.epsp_rise/self.tau_s +  np.multiply(cat_spikes,weights)

        EPSP = self.eps0*(self.epsp_decay-self.epsp_rise)/(self.tau_m-self.tau_s)
        
        u = EPSP.sum(axis=1, keepdims=True).T + self.chi*np.exp((-time + self.last_spike_post)/self.tau_m)

        self.firing_rates = self.rho*np.exp((u-self.theta)/self.delta_u)

        self.spikes = np.random.rand(self.N) <= np.squeeze(self.firing_rates) #realization spike train
        
        self.last_spike_post[self.spikes]= time #update time postsyn spike
        self.Canc = 1-np.matlib.repmat(self.spikes, self.N_in+self.N, 1)

        self.epsp_decay[self.spikes] = 0
        self.epsp_rise[self.spikes] = 0

        self.update_instantaneous_firing_rate()


    def update_instantaneous_firing_rate(self):
    
        self.rho_decay = self.rho_decay - self.rho_decay/self.tau_gamma + self.spikes
        self.rho_rise =  self.rho_rise -  self.rho_rise/self.v_gamma + self.spikes

        self.instantaneous_firing_rates = (self.rho_decay - self.rho_rise)/(self.tau_gamma - self.v_gamma)
