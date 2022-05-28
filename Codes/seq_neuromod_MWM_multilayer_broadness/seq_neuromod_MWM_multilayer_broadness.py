"""Received from Carlos as working baseline.
"""

import sys
sys.path.extend(['../', './Codes/'])

from numba import jit, cuda
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import itertools
import time
import pandas as pd
from IPython import display
from NeuFun_Cuda import convolution, convolution_type2, neuron, neuron_ca1, weights_update_rate, bcm
import csv
from bisect import bisect
import multiprocessing
import pickle
import psutil

from parameters import *
from layers import *
from plot_functions import *

def main():

    results=[]

    if episodes==1:
        
        results.append(episode_run(0))

    else:

        pool = multiprocessing.Pool(os.cpu_count() - 1)

        for episode in range(0,episodes):
            print('Episode',episode)

            results.append(pool.apply_async(episode_run,(episode,),error_callback=log_e))
            
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


def log_e(e):
  print(e)

    
def episode_run(episode):

    # different random seed for each pool
    np.random.seed(random_seed + episode)

    # flag to print first rewarding trial
    ever_rewarded_flag = False

    #Results to be exported for each episode
    rewarding_trials = np.zeros(trials)
    
    print('Initiated episode:',episode)

    ## Place cells positions

    CA3 = CA3_layer(bounds_x, bounds_y, space_pc, offset_ca3, rho_pc, sigma_pc_ca3)

    CA1 = CA1_layer(bounds_x, bounds_y, space_pc, offset_ca1, rho_pc, sigma_pc_ca1,
                    tau_m, tau_s, eps0, chi, theta, delta_u, ca3_scale, CA3.N)

    AC  = Action_layer(N_action, tau_m, tau_s, eps0, chi,
                       rho0, theta, delta_u, tau_gamma, 
                       v_gamma, CA1.N,
                       a0, psi, w_minus, w_plus)
    
    # feedforward weights
    trace_tot_sero = np.zeros([AC.N,CA1.N]) #sum of the traces
    eligibility_trace_sero = np.zeros([AC.N, CA1.N]) #total convolution

    ## initialise variables
    w_ca1 = np.random.rand(CA1.N, CA3.N) +1 

    store_pos = np.zeros([trials, T_max,2]) # stores trajectories (for plotting)
    firing_rate_store_AC = np.zeros([AC.N, T_max, trials]) #stores firing rates action neurons (for plotting)
    firing_rate_store_CA1 = np.zeros([CA1.N, T_max, trials])
    firing_rate_store_CA3 = np.zeros([CA3.N, T_max, trials])

    ## initialize plot open field
    if plot_flag:

        fig, ax = initialize_plots( r_goal, bounds_x, bounds_y,
                                    CA1, offset_ca1, offset_ca3, CA3, c )
        update_plots(fig, ax,
                    0, store_pos, starting_position,
                    firing_rate_store_AC, firing_rate_store_CA1,
                    firing_rate_store_CA3, AC.w_lateral, AC.w_ca1, w_ca1,
                    CA1, None, AC.thetas)

        fig.show()
          
    
    ######################## START SIMULATION ######################################
    
    t_episode = 0 # counter ms

    for trial in range(trials):

        position = starting_position.copy() #initialize position at origin (centre open field)
        rew_found = 0 #flag that signals when the reward is found
        t_rew = T_max #time of reward - initialized at T_max at the beginning of the trial

        t_trial = 0

        trace_tot = np.zeros([AC.N,CA1.N]) #sum of the traces
        eligibility_trace = np.zeros([AC.N, CA1.N]) #total convolution

        dw_ca1 = np.zeros(w_ca1.shape)

        print('Episode:', episode, 'Trial:', trial, 'Mean Weight:', w_ca1.mean())

        while t_trial<T_max:
           
            t_episode  += 1
            t_trial += 1

            # store variables for plotting/saving
            store_pos[trial, t_trial-1, :] = position 
            firing_rate_store_CA3[:,t_trial-1,trial] = CA3.firing_rates
            firing_rate_store_CA1[:,t_trial-1,trial] = CA1.firing_rates
            firing_rate_store_AC[:,t_trial-1,trial] = AC.instantaneous_firing_rates 

            ## CA3 Layer
            CA3.update_activity(position)
            
            ## CA1 Layer
            CA1.update_activity(position, CA3.spikes, w_ca1, t_episode)

            ## Action neurons
            AC.update_activity(CA1.spikes, t_episode)

            # select action
            a = np.dot(AC.instantaneous_firing_rates, AC.actions.T)/AC.N

            ## synaptic plasticity

            # BCM
            if CA1.alpha!=0:

                dw_ca1 += bcm(w_ca1, CA1.firing_rates.mean(), 
                              CA3.firing_rates, CA1.firing_rates, epsilon=epsilon_bcm)

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
                
                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)

                break

        ## update weights - end of trial

        if CA1.alpha != 0:
            w_ca1 += eta_bcm * dw_ca1

        # change due to serotonin or dopamine
        if Dopamine and rew_found:

            AC.update_weights(eta_DA*eligibility_trace)

        if Serotonine and not rew_found:

            AC.update_weights(-eta_Sero*eligibility_trace_sero)

        #calculate policy
        ac =np.dot(AC.actions, AC.w_ca1)/a0 #vector of preferred actions according to the weights
        
        ## plot
        if plot_flag:
            
            update_plots(fig, ax,
                         trial, store_pos, starting_position,
                         firing_rate_store_AC, firing_rate_store_CA1,
                         firing_rate_store_CA3, AC.w_lateral, AC.w_ca1, w_ca1,
                         CA1,ac, AC.thetas)

    returns = (episode, rewarding_trials)

    return returns
     
if __name__ == '__main__':

    main()
