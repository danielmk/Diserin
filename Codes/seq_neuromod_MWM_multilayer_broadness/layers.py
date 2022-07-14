from pickletools import optimize
import numpy as np

from neuron_models import *

class CA3_layer:

    def __init__(self, conf):

        self.neuron_model = Place_Cells(conf)

        self.N = self.neuron_model.N
        self.firing_rates = np.zeros(self.N)
        self.spikes = np.zeros(self.N)
        

    def update_activity(self, pos):
        
        self.firing_rates, self.spikes = self.neuron_model.get_activity(pos, get_spikes=True)


class CA1_layer:

    def __init__(self, N_ca3, conf):

        self.place_model = Place_Cells(conf)

        self.N = self.place_model.N
        self.N_in = N_ca3

        self.SRM0_model = SRM0(N_ca3, self.N, conf) 

        self.alpha = conf['alpha']

        # activity state
        self.spikes = np.zeros(self.N)
        self.firing_rates = np.zeros(self.N)

        # weights
        self.w_min = conf['w_min']
        self.w_max = conf['w_max']


    def update_activity(self, pos, spikes, time):

        if self.alpha == 0:  

            self.firing_rates = self.place_model.get_activity(pos)
        
        elif self.alpha == 1:

            self.firing_rates = self.SRM0_model.get_activity(spikes,time)

        else:
            
            lambda_pos = self.place_model.get_activity(pos)
            lambda_u   = self.SRM0_model.get_activity(spikes, time)

            self.firing_rates = self.alpha*lambda_u + (1-self.alpha)*lambda_pos

        self.spikes = np.random.rand(self.N) <= self.firing_rates


    def update_weights(self, update):

        self.SRM0_model.W += update

        self.SRM0_model.W = np.clip(self.SRM0_model.W, self.w_min, self.w_max)


class Action_layer:

    def __init__(self, N_in, conf):

        self.N = conf['N']
        self.N_in = N_in

        self.neuron_model = SRM0(N_in, self.N, conf) 
        
        self.firing_rates = np.zeros(self.N)
        self.spikes = np.zeros(self.N)

        # to produce action
        self.thetas = 2*np.pi*np.arange(1,self.N+1)*(1./self.N)
        self.actions = conf['a0']*np.array([np.cos(self.thetas), np.sin(self.thetas)])
        self.fixed_step = conf['fixed_step'] if conf['use_fixed_step'] else None

        # To handle weights updates
        self.weight_decay = conf['weight_decay']
        self.base_weight = conf['base_weight']
        self.w_min = conf['w_min']
        self.w_max = conf['w_max']     
         

    def update_activity(self, spikes_ca1, time):
        
        self.firing_rates, self.spikes = self.neuron_model.get_activity(spikes_ca1, time, get_spikes=True)
        

    def update_weights(self, update):

        if self.weight_decay != 0:
            
            decay = self.w_ca1 - self.base_weight
            decay = np.where(decay>0, decay, 0)
            self.neuron_model.W -= self.weight_decay*decay

        self.neuron_model.W += update

        self.neuron_model.W = np.clip(self.neuron_model.W, self.w_min, self.w_max)        

    def get_action(self,):
        
        a = np.einsum('ij, j -> i', self.actions, self.firing_rates)
        
        if self.fixed_step is not None:
            
            return self.fixed_step * a / (np.linalg.norm(a)+ 1e-15)

        else:

            return a/self.N
