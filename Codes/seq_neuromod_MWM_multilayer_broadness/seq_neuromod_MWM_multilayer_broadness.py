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
        
        results.append(episode_run(jobID, 0,plot_flag,trials, Sero,
                                    eta_DA,eta_Sero, A_DA,A_Sero,Activ, Inhib, 
                                    tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3))
    else:

        pool = multiprocessing.Pool(os.cpu_count() - 1)

        for episode in range(0,episodes):
            print('Episode',episode)

            results.append(pool.apply_async(episode_run,(jobID,episode,plot_flag,trials,Sero,
                                                        eta_DA,eta_Sero, A_DA,A_Sero,Activ, Inhib, 
                                                        tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3),error_callback=log_e))
            
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

    
def episode_run(jobID,episode,plot_flag,Trials,Sero, eta_DA,eta_Sero,A_DA,A_Sero,
                Activ,Inhib,tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3):

    # different random seed for each pool
    np.random.seed(random_seed + episode)

    # flag to print first rewarding trial
    ever_rewarded_flag = False

    #Results to be exported for each episode
    rewarding_trials = np.zeros(Trials)
    
    print('Initiated episode:',episode)

    t_rew = T_max
    t_extreme = t_rew + delay_post_reward #time of reward - initialized to maximu

    ## Place cells positions

    CA3 = CA3_layer(bounds_x, bounds_y, space_pc, offset_ca3, rho_pc, sigma_pc_ca3)

    CA1 = CA1_layer(bounds_x, bounds_y, space_pc, offset_ca1, rho_pc, sigma_pc_ca1,
                    tau_m, tau_s, eps0, chi, theta, delta_u, ca3_scale, CA3.N)

    AC  = Action_layer(N_action, tau_m, tau_s, eps0, chi,
                       rho0, theta, delta_u, tau_gamma, 
                       v_gamma, CA1.N)


    #winner-take-all weights
    theta_actor = 2*np.pi*np.arange(1,AC.N+1)/AC.N #angles actions
    diff_theta = theta_actor - theta_actor.reshape(-1,1)
    f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
    np.fill_diagonal(f,0)
    w_lateral = (w_minus/AC.N+w_plus*f/f.sum(axis=0)) #lateral connectivity action neurons

    #actions
    actions = np.squeeze(a0*np.array([np.cos(theta_actor), np.sin(theta_actor)])) #possible actions (x,y)
    
    # feedforward weights
    trace_tot_sero = np.zeros([AC.N,CA1.N]) #sum of the traces
    eligibility_trace_sero = np.zeros([AC.N, CA1.N]) #total convolution

    ## initialise variables
    w_in = np.ones([AC.N, CA1.N]) * 2 # initialization feed-forward weights
    w_tot = np.concatenate([w_in,w_lateral],axis=1)#total weigthsù
    w_ca1 = np.random.rand(CA1.N, CA3.N) +1 

    store_pos = np.zeros([Trials, T_max,2]) # stores trajectories (for plotting)
    firing_rate_store_AC = np.zeros([AC.N, T_max, Trials]) #stores firing rates action neurons (for plotting)
    firing_rate_store_CA1 = np.zeros([CA1.N, T_max, Trials])
    firing_rate_store_CA3 = np.zeros([CA3.N, T_max, Trials])

    ## initialize plot open field
    if plot_flag:

        fig, ax = initialize_plots( r_goal, bounds_x, bounds_y,
                                    CA1, offset_ca1, offset_ca3, CA3, c )
        update_plots(fig, ax,
                    0, store_pos, starting_position,
                    firing_rate_store_AC, firing_rate_store_CA1,
                    firing_rate_store_CA3, w_tot, w_ca1,
                    CA1, None, theta_actor)

    ## delete actions that lead out of the maze

    #find index place cells that lie on the walls (CA1)
    sides = np.empty((4,int(np.max([np.sqrt(CA1.N), np.sqrt(CA1.N)]))))

    sides[0,:] = np.where(CA1.pc[:,1] == CA1.pc.min())[0].T #bottom wall, y=-2
    sides[1,:] = np.where(CA1.pc[:,1] == CA1.pc.max())[0].T #top wall, y=+2
    sides[2,:] = np.where(CA1.pc[:,0] == CA1.pc.max())[0].T #left wall, x=-2
    sides[3,:] = np.where(CA1.pc[:,0] == CA1.pc.min())[0].T #right wall, x=+2

    #store index of actions forbidden from each side
    forbidden_actions = np.empty((4,19))
    forbidden_actions[0,:] = np.arange(21,40) #actions that point south - theta in (180, 360) degrees approx
    forbidden_actions[1,:] = np.arange(1,20) #actions that point north - theta in (0,180) degrees approx
    forbidden_actions[2,:] = np.concatenate([np.arange(1,10), np.arange(31,41)]) #actions that point east - theta in (-90, 90) degrees approx
    forbidden_actions[3,:] = np.arange(11,30)#actions that point west - theta in (90, 270) degrees approx
    
    #kill connections between place cells on the walls and forbidden actions
    w_walls = np.ones([AC.N, CA1.N+AC.N])
    for g in range(4):
        idx = list(itertools.product(forbidden_actions[g,:].astype(int).tolist(),sides[g,:].astype(int).tolist()))
        w_walls[np.array(idx)[:,0]-1,np.array(idx)[:,1]] = 0

    
    
    ######################## START SIMULATION ######################################
    w_tot_old = w_tot[0:AC.N,0:CA1.N] #store weights before start
    
    t_episode = 0 # counter ms

    for trial in range(Trials):

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

            # BCM update
            if CA1.alpha!=0:

                dw_ca1 += bcm(w_ca1, CA1.firing_rates.mean(), 
                              CA3.firing_rates, CA1.firing_rates, epsilon=epsilon_bcm)

            ## action neurons
        
            AC.update_activity(CA1.spikes, w_tot*w_walls, t_episode)

            # select action
            a = np.dot(AC.instantaneous_firing_rates, actions.T)/AC.N

            ## synaptic plasticity

            W, eligibility_trace, trace_tot = weights_update_rate((A_pre_post+A_post_pre)/2, tau_pre_post, np.matlib.repmat(CA1.firing_rates.T,AC.N,1), np.matlib.repmat(np.squeeze(AC.instantaneous_firing_rates),CA1.N,1).T, trace_tot, tau_e)

            #STDP with unsymmetric window and depression due to serotonin
            if Sero:
                W_sero, eligibility_trace_sero, trace_tot_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(CA1.firing_rates.T,AC.N,1), np.matlib.repmat(np.squeeze(AC.instantaneous_firing_rates),CA1.N,1).T, trace_tot_sero, tau_e_sero)

            # online weights update (effective only with acetylcholine - ACh_flag=1)
            if ACh_flag:
                w_tot[0:AC.N,0:CA1.N]= w_tot[0:AC.N,0:CA1.N]-eta_ACh*W

                #weights limited between lower and upper bounds
                w_tot[np.where(w_tot[:,0:CA1.N]>w_max)] = w_max
                w_tot[np.where(w_tot[:,0:CA1.N]<w_min)] = w_min

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
            
            if walls_punish and wall_hit:
                w_tot[0:AC.N,0:CA1.N]= w_tot[0:AC.N,0:CA1.N]-eta_ACh*W
            
            ## reward

            if  rew_found==0 and np.sum((position-c)**2)<=r_goal**2:

                rew_found=1 #flag reward found (so that trial is ended soon)
                t_rew=t_trial #time of reward
                rewarding_trials[trial] = 1
                
                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)

            #time when trial end is 300ms after reward is found
            t_extreme = t_rew+delay_post_reward

            if t_trial>t_extreme and t_trial<T_max:
                t_episode = int((np.ceil(t_episode/T_max))*T_max)-1 
                break

        
        if CA1.alpha != 0:
            w_ca1 += eta_bcm * dw_ca1
        ## update weights - end of trial

        # if the reward is not found, no change (and not Inhib is true)
        # change due to serotonin or dopamine
        if Sero:
            w_tot[0:AC.N,0:CA1.N]= (w_tot_old+eta_DA*eligibility_trace)*rew_found + (w_tot_old-eta_Sero*eligibility_trace_sero)*(1-rew_found)*(not Inhib)
        else:
            #change due to dopamine or sustained weights at the end of the trial
            w_tot[0:AC.N,0:CA1.N]=w_tot[0:AC.N,0:CA1.N]*(1-rew_found)+(w_tot_old+eta_DA*eligibility_trace)*rew_found

        #weights limited between lower and upper bounds
        w_tot[np.where(w_tot[:,0:CA1.N]>w_max)]=w_max
        w_tot[np.where(w_tot[:,0:CA1.N]<w_min)]=w_min

        #store weights before the beginning of next trial (for updates in case reward is found)
        w_tot_old = np.copy(w_tot[0:AC.N,0:CA1.N])

        #calculate policy
        ac =np.dot(np.squeeze(actions),(np.multiply(w_tot_old,w_walls[:,0:CA1.N]))/a0) #vector of preferred actions according to the weights
        ac[:,np.unique(np.sort(np.reshape(sides, (np.max(sides.shape)*4, 1),order='F'))).astype(int).tolist()]=0 #do not count actions AT the boundaries (just for plotting)
        
        ## plot
        if plot_flag:
            
            update_plots(fig, ax,
                         trial, store_pos, starting_position,
                         firing_rate_store_AC, firing_rate_store_CA1,
                         firing_rate_store_CA3, w_tot, w_ca1,
                         CA1,ac, theta_actor)

    returns = (episode, rewarding_trials)

    return returns
     
if __name__ == '__main__':

    main()
