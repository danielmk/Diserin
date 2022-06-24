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
    
        if use_sum:
        
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

        if self.weight_decay!=0:

            decay = weights - self.base_weight
            decay = np.where(decay>0, decay, 0)
            dW -= self.weight_decay * decay

        return dW

    def compute_thetas(self, y):
            
        return y**2


class Plasticity_AC:

    def __init__(self, N_out, N_in, 
                 A_DA, tau_DA, tau_e_DA,
                 A_5HT, tau_5HT, tau_e_5HT,
                 A_ACh, tau_ACh):

        self.N_out, self.N_in = N_out, N_in

        self.A_DA, self.tau_DA, self.tau_e_DA = A_DA, tau_DA, tau_e_DA
        self.A_5HT, self.tau_5HT, self.tau_e_5HT = A_5HT, tau_5HT, tau_e_5HT
        self.A_ACh, self.tau_ACh = A_ACh, tau_ACh

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
        
        update_5HT = self.get_update(rates_pre, rates_post, self.A_5HT, self.tau_5HT)
        self.trace_5HT = self.trace_5HT - self.trace_DA/self.tau_e_5HT + update_5HT

    def release_ACh(self, rates_pre, rates_post):

        rates_pre = rates_pre.reshape(1, -1)
        rates_post = rates_post.reshape(-1, 1)

        return self.get_update(rates_pre, rates_post, self.A_ACh, self.tau_ACh)




