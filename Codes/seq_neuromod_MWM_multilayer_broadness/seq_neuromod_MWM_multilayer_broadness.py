"""Received from Carlos as working baseline.
"""

import sys
sys.path.extend(['../', './Codes/'])

from numba import jit, cuda
import os
import numpy as np
import numpy.matlib
import time
import pandas as pd
from NeuFun_Cuda import convolution, convolution_type2, neuron, neuron_ca1, weights_update_rate, bcm
import csv
from bisect import bisect
import multiprocessing
import pickle
import psutil
from tqdm import tqdm

from parameters import *
from layers import *
from plot_functions import *

def get_starting_position(starting_position_option):

    if starting_position_option=='origin':

        return np.array([0.,0.])

    if starting_position_option=='random':

        return np.random.rand(2)*4 - 2

    print("Starting position option non valid!")
    exit()

def main():

    results=[]

    if episodes==1:
        
        results.append(episode_run(0))

    else:

        pool = multiprocessing.Pool(os.cpu_count() - 1)

        for episode in range(0,episodes):
            print('Episode',episode)

            results.append(pool.apply_async(episode_run,(episode,)))
            
            current_process = psutil.Process()
            children = current_process.children(recursive=True)

            while len(children) > os.cpu_count() - 1:
                time.sleep(0.1)
                current_process = psutil.Process()
                children = current_process.children(recursive=True)

        results = [result.get() for result in results]
        pool.close()
        pool.join()

    with open(jobID+'.pickle', 'wb') as myfile:

        pickle.dump((descriptor,results), myfile)

    
def episode_run(episode):

    # different random seed for each pool
    np.random.seed(random_seed + episode)

    # flag to print first rewarding trial
    ever_rewarded_flag = False

    #Results to be exported for each episode
    rewarding_trials = np.zeros(trials)
    rewarding_times  = np.zeros(trials) - 1
    
    print('Initiated episode:',episode)

    ## Place cells positions

    CA3 = CA3_layer(bounds_x, bounds_y, space_pc, offset_ca3, rho_pc, sigma_pc_ca3)

    CA1 = CA1_layer(bounds_x, bounds_y, space_pc, offset_ca1, rho_pc, sigma_pc_ca1,
                    tau_m_ca1, tau_s_ca1, eps0_ca1, chi_ca1, rho0_ca1, theta_ca1, delta_u_ca1, ca3_scale, CA3.N,
                    w_min_ca1, w_max_ca1, w_ca1_init, max_init, sigma_init)

    AC  = Action_layer(N_action, tau_m, tau_s, eps0, chi,
                       rho0, theta, delta_u, tau_gamma, 
                       v_gamma, CA1.N,
                       a0, psi, w_minus, w_plus,
                       weight_decay_ac, base_weight_ac, w_min, w_max)

    bcm = BCM(CA1.N, memory_factor, weight_decay, base_weight)

    ## initialise variables
    store_pos = np.zeros([trials, T_max,2]) # stores trajectories (for plotting)
    firing_rate_store_AC = np.zeros([AC.N, T_max, trials]) #stores firing rates action neurons (for plotting)
    firing_rate_store_CA1 = np.zeros([CA1.N, T_max, trials])
    firing_rate_store_CA3 = np.zeros([CA3.N, T_max, trials])
    initial_weights = {'CA1': CA1.w_ca3.copy(),
                        'AC': AC.w_ca1.copy()}

    ## initialize plot open field
    if plot_flag:

        fig= initialize_plots( r_goal, bounds_x, bounds_y,
                               CA1, offset_ca1, offset_ca3, CA3, c )

        #calculate policy
        if CA1.alpha == 0:
            ac = np.dot(AC.actions, AC.w_ca1)/a0 #vector of preferred actions according to the weights
        elif CA1.alpha == 1:
            ac = np.dot(AC.actions, np.dot(AC.w_ca1, CA1.w_ca3))/a0
        else:
            ac = (1-CA1.alpha)*np.dot(AC.actions, AC.w_ca1)/a0 + CA1.alpha*np.dot(AC.actions, np.dot(AC.w_ca1, CA1.w_ca3))/a0

        update_plots(fig, 0, store_pos, None,
                     firing_rate_store_AC, firing_rate_store_CA1,
                     firing_rate_store_CA3, CA3, CA1, AC, ac)
        fig.show()
          
    
    ######################## START SIMULATION ######################################
    
    t_episode = 0 # counter ms

    thetas_history = np.empty((trials*T_max, 5))

    for trial in range(trials):

        starting_position = get_starting_position(starting_position_option)

        position = starting_position.copy()
        rew_found = 0
        t_trial = 0

        trace_tot = np.zeros([AC.N,CA1.N]) #sum of the traces
        eligibility_trace = np.zeros([AC.N, CA1.N]) #total convolution
        
        trace_tot_sero = np.zeros([AC.N,CA1.N]) #sum of the traces
        eligibility_trace_sero = np.zeros([AC.N, CA1.N]) #total convolution

        print('Episode:', episode, 'Trial:', trial, flush=True)

        for t_trial in tqdm(range(T_max)):

            # store variables for plotting/saving
            store_pos[trial, t_trial, :] = position 
            firing_rate_store_CA3[:,t_trial,trial] = CA3.firing_rates
            firing_rate_store_CA1[:,t_trial,trial] = CA1.firing_rates
            firing_rate_store_AC[:,t_trial,trial] = AC.instantaneous_firing_rates 

            ## CA3 Layer
            CA3.update_activity(position)
            
            ## CA1 Layer
            CA1.update_activity(position, CA3.spikes, t_episode)

            ## Action neurons
            AC.update_activity(CA1.spikes, t_episode)

            # select action
            a = AC.get_action()

            ## synaptic plasticity
            # BCM
            if CA1.alpha!=0. and BCM_ON:
                
                update = bcm.get_update(CA3.firing_rates, CA1.firing_rates, CA1.w_ca3)
                CA1.update_weights(eta_bcm * update)

                thetas_history[t_episode] = bcm.thetas[np.random.randint(0,100)]

            W, eligibility_trace, trace_tot = weights_update_rate((A_pre_post+A_post_pre)/2, tau_pre_post, np.matlib.repmat(CA1.firing_rates.T,AC.N,1), np.matlib.repmat(np.squeeze(AC.instantaneous_firing_rates),CA1.N,1).T, trace_tot, tau_e)

            #STDP with unsymmetric window and depression due to serotonin
            if Serotonine:

                W_sero, eligibility_trace_sero, trace_tot_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(CA1.firing_rates.T,AC.N,1), np.matlib.repmat(np.squeeze(AC.instantaneous_firing_rates),CA1.N,1).T, trace_tot_sero, tau_e_sero)

            ## position update
            position += a

            #check if agent is out of boundaries. If it is, bounce back in the opposite direction
            wall_hit = True
            if position[0]<=bounds_x[0]:
                position = position+dx*np.array([1,0])
            elif position[0]>= bounds_x[1]:
                position = position+dx*np.array([-1,0])
            elif position[1]<=bounds_y[0]:
                position = position+dx*np.array([0,1])
            elif position[1]>=bounds_y[1]:
                position = position+dx*np.array([0,-1])
            else:
                wall_hit = False
            
            if Acetylcholine and wall_hit:
                
                AC.update_weights(-eta_ACh*W)
            
            ## reward

            if  rew_found==0 and np.sum((position-c)**2)<=r_goal**2:

                rew_found=1 #flag reward found (so that trial is ended soon)
                rewarding_trials[trial] = 1
                rewarding_times[trial] = t_trial
                
                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)

                break

            t_episode  += 1

        ## update weights - end of trial

        # change due to serotonin or dopamine
        if Dopamine and rew_found:

            AC.update_weights(eta_DA*eligibility_trace)

        if Serotonine and not rew_found:

            AC.update_weights(-eta_Sero*eligibility_trace_sero)

        #calculate policy
        if CA1.alpha == 0:
            ac = np.dot(AC.actions, AC.w_ca1)/a0 #vector of preferred actions according to the weights
        elif CA1.alpha == 1:
            ac = np.dot(AC.actions, np.dot(AC.w_ca1, CA1.w_ca3))/a0
        else:
            ac = (1-CA1.alpha)*np.dot(AC.actions, AC.w_ca1)/a0 + CA1.alpha*np.dot(AC.actions, np.dot(AC.w_ca1, CA1.w_ca3))/a0

        ## plot
        if plot_flag:
            
            update_plots(fig,trial, store_pos, starting_position,
                         firing_rate_store_AC, firing_rate_store_CA1,
                         firing_rate_store_CA3, CA3, CA1, AC, ac)

            
    returns = { 'episode':episode,  
                'rewarding_trials':rewarding_trials, 
                'rewarding_times': rewarding_times,
                'trajectories': store_pos,
                'initial_weights': initial_weights,
                'final_weights': {'CA1': CA1.w_ca3,
                                  'AC' : AC.w_ca1}}
    
    if save_activity:

        returns['activities'] = {'CA3': firing_rate_store_CA3,
                                 'CA1': firing_rate_store_CA1,
                                 'AC': firing_rate_store_AC}
    
    if save_thetas:

        returns['thetas'] = thetas_history

    return returns
     
if __name__ == '__main__':

    main()
