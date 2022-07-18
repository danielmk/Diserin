import numpy as np

class BCM:

    def __init__(self, N, conf):

        self.num_neurons = N

        self.weight_decay = conf['weight_decay']
        self.base_weight = conf['base_weight']
        self.memory_factor = conf['memory_factor']
        self.learning_rate = conf['learning_rate']

        self.thetas = None

    def get_update(self, x, y, use_sum=False, weights=None):
        """
        x is the pre-synaptic activity
        y is the post-synaptic acitivity
        """
    
        if use_sum:
            
            assert weights is not None
    
            y = np.einsum('ij,j->i',weights, x)
        
        current_thetas = self.compute_thetas(y)
        
        if self.thetas is None:

            self.thetas = current_thetas
        
        else:

            self.thetas = self.memory_factor*self.thetas + (1-self.memory_factor)*current_thetas

        x = x.reshape(1,-1)
        y = y.reshape(-1, 1)
        thetas = self.thetas.reshape(-1,1)

        dW = y*(y-thetas)*x 
        
        return self.learning_rate*dW

    def compute_thetas(self, y):
            
        return y**2


class Plasticity_AC:

    def __init__(self, N_out, N_in, conf):

        self.N_out, self.N_in = N_out, N_in

        self.A_DA, self.tau_DA, self.tau_e_DA = conf['A_DA'], conf['tau_DA'], conf['tau_e_DA']
        self.A_5HT, self.tau_5HT, self.tau_e_5HT = conf['A_5HT'], conf['tau_5HT'], conf['tau_e_5HT']
        self.A_ACh, self.tau_ACh = conf['A_ACh'], conf['tau_ACh']

        self.trace_DA = np.zeros([N_out, N_in])
        self.trace_5HT= np.zeros([N_out, N_in])

    def get_update(self, rates_pre, rates_post, A, tau):

        C_x = rates_post*A/(tau**-1 + rates_post)

        return C_x*rates_pre

    def update_traces(self, rates_pre, rates_post):

        rates_pre = rates_pre.reshape(1, -1)
        rates_post = rates_post.reshape(-1, 1)
        
        update_DA = self.get_update(rates_pre, rates_post, self.A_DA, self.tau_DA)
        self.trace_DA = self.trace_DA - self.trace_DA/self.tau_e_DA + update_DA
        
        update_5HT = - self.get_update(rates_pre, rates_post, self.A_5HT, self.tau_5HT)
        self.trace_5HT = self.trace_5HT - self.trace_5HT/self.tau_e_5HT + update_5HT

    def release_ACh(self, rates_pre, rates_post):

        rates_pre = rates_pre.reshape(1, -1)
        rates_post = rates_post.reshape(-1, 1)

        return - self.get_update(rates_pre, rates_post, self.A_ACh, self.tau_ACh)




