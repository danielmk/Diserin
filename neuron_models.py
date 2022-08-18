import numpy as np
from weights import *

class Place_Cells:

    def __init__(self, conf):

        
        self.rho, self.sigma_sq = conf['rho'], conf['sigma']**2

        xs = conf['bounds_x']
        ys = conf['bounds_y']
        space = conf['space_pc']
        
        self.offset = conf['offset']

        if conf['offset']:
            x_pc = np.round(np.arange(xs[0], xs[1], space)+space/2,2)
            y_pc = np.round(np.arange(ys[0], ys[1], space)+space/2,2)
        else:
            x_pc = np.round(np.arange(xs[0], xs[1]+space, space),2)
            y_pc = np.round(np.arange(ys[0], ys[1]+space, space),2)
        
        xx, yy = np.meshgrid(x_pc, y_pc)
        self.pc = np.stack([xx,yy], axis=2).reshape(-1,2)

        self.N = self.pc.shape[0]

    
    def get_activity(self, pos, get_spikes=False):

        rates = self.rho * np.exp(- ( (pos-self.pc)**2).sum(axis=1) *(1./self.sigma_sq) )
        
        if get_spikes:
            spikes = np.random.rand(self.N) <= rates
            return rates, spikes

        return rates


class SRM0:

    def __init__(self, N_in, N_out, conf):


        self.N_in = N_in
        self.N = N_out

        self.tau_m, self.tau_s, self.eps0 = conf['tau_m'], conf['tau_s'], conf['eps0']
        self.chi = conf['chi']
        self.rho, self.theta, self.delta_u = conf['rho0'], conf['theta'], conf['delta_u']

        self.smooth_firing = conf['smooth_firing']

        if self.smooth_firing:
            self.tau_gamma, self.v_gamma = conf['tau_gamma'], conf['v_gamma']
            self.rho_decay = np.zeros(N_out)
            self.rho_rise =  np.zeros(N_out)

        self.spikes = np.zeros(N_out)

        # Weights
        self.W = feedforward_weights_initiliazation(self.N, self.N_in, conf)
        self.W_lateral = lateral_weights_initialization(self.N, conf)

        if self.W_lateral is not None:
            self.epsp_decay = np.zeros((N_out, N_in + N_out))
            self.epsp_rise = np.zeros((N_out, N_in + N_out))
        else:
            self.epsp_decay = np.zeros((N_out, N_in))
            self.epsp_rise = np.zeros((N_out, N_in))

        self.last_spike_time = np.zeros(N_out) - 1000

        self.max=0

    def get_activity(self, spikes_pre, time, get_spikes=False):

        if self.W_lateral is not None:
            cat_spikes = np.concatenate([spikes_pre, self.spikes])
            cat_weights = np.concatenate([self.W, self.W_lateral], axis=1)
        else:
            cat_spikes = spikes_pre
            cat_weights = self.W

        self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + np.multiply(cat_spikes,cat_weights)
        self.epsp_rise =  self.epsp_rise  - self.epsp_rise/self.tau_s +  np.multiply(cat_spikes,cat_weights)

        EPSP = self.eps0*(self.epsp_decay-self.epsp_rise)/(self.tau_m-self.tau_s)

        u = EPSP.sum(axis=1) + self.chi*np.exp((-time + self.last_spike_time)/self.tau_m)

        firing_rates = self.rho*np.exp((u-self.theta)/self.delta_u)
        self.spikes = np.random.rand(self.N) <= firing_rates #realization spike train

        self.last_spike_time[self.spikes]= time #update time postsyn spike
        self.epsp_decay[self.spikes] = 0
        self.epsp_rise[self.spikes] = 0

        if self.smooth_firing:
            self.rho_decay = self.rho_decay - self.rho_decay/self.tau_gamma + self.spikes
            self.rho_rise =  self.rho_rise -  self.rho_rise/self.v_gamma + self.spikes
            firing_rates = (self.rho_decay - self.rho_rise)*(1./(self.tau_gamma - self.v_gamma))

        return (firing_rates, self.spikes) if get_spikes else firing_rates
