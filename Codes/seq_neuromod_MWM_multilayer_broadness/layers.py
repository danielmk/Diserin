from pickletools import optimize
import numpy as np


class BCM:

    def __init__(self, N,  memory_factor, weight_decay, base_weight):

        self.num_neurons = N
        self.weight_decay = weight_decay
        self.base_weight = base_weight
        self.memory_factor = memory_factor

        self.thetas = None

    def get_update(self, x, y, weights, use_sum=False):
        """
        x is the pre-synaptic activity
        y is the post-synaptic acitivity
        """
        
        current_thetas = self.compute_thetas(y)
        
        if self.thetas is None:

            self.thetas = current_thetas
        
        else:

            self.thetas = self.memory_factor*self.thetas + (1-self.memory_factor)*current_thetas


        if use_sum:
            
            y = np.dot(weights, x)

        x = x.reshape(1,-1)
        y = y.reshape(-1, 1)
        thetas = self.thetas.reshape(-1,1)

        dW = y*(y-thetas)*x 

        if self.weight_decay!=0:

            decay = weights - self.base_weight
            decay = np.where(decay>0, decay, 0)
            dW -= self.weight_decay * decay

        return dW

    def compute_thetas(self, y):
            
        return y**2 # with this value inhibition and exhitation are more or less even
        #return y.mean()
        #return y**2




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
                       rho_pc, sigma,
                       tau_m, tau_s, eps0,
                       chi, rho,
                       theta, delta_u,
                       alpha,
                       N_ca3,
                       w_min, w_max,
                       w_ca1_init, max_init, sigma_init):

        self.rho_pc, self.sigma = rho_pc, sigma

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
        self.rho = rho
        self.theta, self.delta_u = theta, delta_u
        self.alpha = alpha

        self.epsp_decay = np.zeros((self.N, self.N_in))
        self.epsp_rise = np.zeros((self.N, self.N_in))

        self.last_spike_time = np.zeros(self.N)-1000

        # activity state
        self.spikes = np.zeros(self.N)
        self.firing_rates = np.zeros(self.N)

        # weights
        self.w_min = w_min
        self.w_max = w_max
        self.w_ca3 = self._initilialize_weights(w_ca1_init, max_init, sigma_init)

        

    def update_activity(self, pos, spikes, time):

        positional_firing_rate, lambda_u = 0, 0

        if self.alpha != 1:  
            positional_firing_rate = self.rho_pc * np.exp(- ((pos-self.pc)**2).sum(axis=1) / self.sigma**2 )
        
        if self.alpha != 0:

            self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + spikes*self.w_ca3
            self.epsp_rise = self.epsp_rise - self.epsp_rise/self.tau_s + spikes*self.w_ca3

            EPSP = self.eps0*(self.epsp_decay - self.epsp_rise)/(self.tau_m - self.tau_s)
            EPSP = EPSP.sum(axis=1)

            u = EPSP + self.chi*np.exp(-(time - self.last_spike_time)/self.tau_m)
            lambda_u = self.rho*np.exp((u-self.theta)/(self.delta_u))
        

        self.firing_rates = self.alpha*lambda_u + (1-self.alpha)*positional_firing_rate

        if self.firing_rates.max()>0.4:
            self.firing_rates = self.firing_rates/self.firing_rates.max()*0.4

        self.spikes = np.random.rand(self.N) <= self.firing_rates
    
        self.last_spike_time[self.spikes] = time

        self.epsp_rise[self.spikes] = 0
        self.epsp_decay[self.spikes] = 0

    def update_weights(self, update):

        self.w_ca3 += update

        self.w_ca3 = np.clip(self.w_ca3, self.w_min, self.w_max)

    
    def _convolutional_initialization(self, maximum, sigma):

        # weights shape (N_out, N_in)
        N_size = int(np.sqrt(self.N))
        K_size = int(np.sqrt(self.N_in))

        if N_size != K_size:
            print("Convolutional initialization is implemented only for N_ca1 = N_ca3.")
            exit()

        weights = np.empty((N_size, N_size, K_size, K_size))

        step = 4./K_size

        for i in range(N_size):
            
            for j in range(N_size):
                
                x, y = np.meshgrid(np.linspace(0,4,K_size), np.linspace(0,4,K_size))
                dst = np.sqrt((x- step/2 -j*step)**2+(y- step/2 - i*step)**2)

                weights[i,j] =  maximum*np.exp(-( (dst)**2 / ( 2.0 * sigma**2 ) ) )

        return weights.reshape(self.N, self.N_in)

    def _initilialize_weights(self, option, max_init, sigma_init):

        if option == 'convolutional':
            return self._convolutional_initialization(max_init, sigma_init)

        if option == 'all_ones':
            return np.ones((self.N, self.N_in))
        
        if option == 'random':
            return np.random.rand(self.N, self.N_in) 

        if option =='identity':
            return np.eye(self.N)
        

    def update_activity_simple_mode(self, rates_ca3):

        self.firing_rates = np.einsum('ij, j -> i', self.w_ca3, rates_ca3)
        self.firing_rates = self.firing_rates/self.firing_rates.max()*0.4
        self.spikes = np.random.rand(self.N) <= self.firing_rates




class Action_layer:

    def __init__(self, N,
                       tau_m, tau_s, eps0,
                       chi,
                       rho, theta, delta_u,
                       tau_gamma, v_gamma,
                       N_ca1,
                       a0, psi, w_minus, w_plus,
                       weight_decay, base_weight, w_min, w_max):

        self.N = N
        self.tau_m, self.tau_s, self.eps0 = tau_m, tau_s, eps0
        self.chi = chi
        self.rho, self.theta, self.delta_u = rho, theta, delta_u
        self.N_in = N_ca1
        
        self.epsp_decay = np.zeros((N, N_ca1 + N))
        self.epsp_rise = np.zeros((N, N_ca1 + N))
        self.last_spike_time = np.zeros(N) - 1000

        self.tau_gamma, self.v_gamma = tau_gamma, v_gamma
        self.rho_decay = np.zeros(N)
        self.rho_rise = np.zeros(N)

        self.last_spike_post=np.zeros(N)-1000

        self.firing_rates = np.zeros(self.N)
        self.spikes = np.zeros(self.N)
        self.instantaneous_firing_rates = np.zeros(self.N)

        # Actions
        self.thetas = 2*np.pi*np.arange(1,N+1)/N
        self.actions = a0*np.array([np.cos(self.thetas), np.sin(self.thetas)]) #possible actions (x,y)
        
        # Lateral Connections
        diff_theta = self.thetas - self.thetas.reshape(-1,1)
        f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
        np.fill_diagonal(f,0)
        self.w_lateral = (w_minus/N+w_plus*f/f.sum(axis=0)) #lateral connectivity action neurons

        #CA1 connections
        self.w_ca1 = np.ones((N, N_ca1))*2#np.random.rand(N, N_ca1)*2 + 1

        self.weight_decay = weight_decay
        self.base_weight = base_weight
        self.w_min = w_min
        self.w_max = w_max
        

    def update_activity(self, spikes_ca1, time):
        
        cat_spikes = np.concatenate([spikes_ca1, self.spikes])
        cat_weights = np.concatenate([self.w_ca1, self.w_lateral], axis=1)

        self.epsp_decay = self.epsp_decay - self.epsp_decay/self.tau_m + np.multiply(cat_spikes,cat_weights)
        self.epsp_rise =  self.epsp_rise  - self.epsp_rise/self.tau_s +  np.multiply(cat_spikes,cat_weights)

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

    def update_weights(self, update):

        self.w_ca1 += update
        decay = self.w_ca1 - self.base_weight
        decay = np.where(decay>0, decay, 0)
        self.w_ca1 -= self.weight_decay*decay

        self.w_ca1 = np.clip(self.w_ca1, self.w_min, self.w_max)

        

    def get_action(self,):
        
        return (self.instantaneous_firing_rates*self.actions).sum(axis=1)/self.N
