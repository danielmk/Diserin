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

    def __init__(self, N_in, N_out,
                       tau_m, tau_s, eps0,
                       chi,
                       rho, theta, delta_u,
                       tau_gamma, v_gamma,
                       lateral_connections, psi, w_minus, w_plus):

       
        self.N_in = N_in
        self.N = N_out

        self.tau_m, self.tau_s, self.eps0 = tau_m, tau_s, eps0
        self.chi = chi
        self.rho, self.theta, self.delta_u = rho, theta, delta_u
        
        self.epsp_decay = np.zeros((N_out, N_in + N_out))
        self.epsp_rise = np.zeros((N_out, N_in + N_out))
        self.last_spike_time = np.zeros(N_out) - 1000

        self.tau_gamma, self.v_gamma = tau_gamma, v_gamma
        self.rho_decay = np.zeros(N_out)
        self.rho_rise =  np.zeros(N_out)

        self.spikes = np.zeros(N_out)

        # Weights
        self.W = 2*np.ones((N_out, N_in)) #np.random.rand(N, N_ca1)*2 + 1

        if lateral_connections:
            thetas = 2*np.pi*np.arange(1,N_out+1)*(1./N_out)
            diff_theta = thetas - thetas.reshape(-1,1)
            f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
            np.fill_diagonal(f,0)
            self.W_lateral = (w_minus*(1./N_out) + w_plus*f/f.sum(axis=0)) #lateral connectivity action neurons
        else:
            self.W_lateral = None


    def get_activity(self, spikes_pre, time):

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

        self.rho_decay = self.rho_decay - self.rho_decay/self.tau_gamma + self.spikes
        self.rho_rise =  self.rho_rise -  self.rho_rise/self.v_gamma + self.spikes

        instantaneous_firing_rates = (self.rho_decay - self.rho_rise)*(1./(self.tau_gamma - self.v_gamma))

        return instantaneous_firing_rates, self.spikes