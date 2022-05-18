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
import pdb

from parameters import *
from layers import *
from plot_functions import *

"""
DESCRIPTION OF THE OUTPUT OBJECT results (saved as results.pickle)

results is a tuple of two elements

    results[0] is a dict with the name and values of parameters/options used for the simulation

    results[1] is a list of <NUMBER OF EPISODES> elements

        each results[1][i] is a tuple of ? elements, namely:

            [0] integer number, index of the episode (0,1,2,...)
            [1] array of shape (<NUMBER OF TRIALS>)
                It is 1 if the trial was rewarding, 0 otherwise 
            [2] array of shape (<NUMBER OF TRIALS>, 4)
                Each row refers to one of the trials. Elements 0,1,2,3 of a row refer respectively
                to the I,II,III,IV quadrant of the space. They are integers and correspond to the
                number of times that the agent was in that quadrant during the trial (after each move)
            [3] array of shape (<NUMBER OF TRIALS>)
                Each element is the median distance of the agent from the center, during the trial
            [4] array of shape (<NUMBER OF TRIALS>)
                Each element is the time when the reward was reached (0 if never reached)
            [5] array of shape (<NUMBER OF TRIALS>)
                Each element is the time when the reward 2 was first reached 
                (0 if never reached)
            [6] array of shape (<NUMBER OF TRIALS>)
                Each element is the time when the position of reward 1 was reached, but 
                the reward was moved to position 2
            [7] array of shape (<NUMBER OF TRIALS>, <T_max>, 2)
                The position of the agent at each time instant of each trial

            # if save_activities=True
            [8] MISSING IF , the history of layers' activities 

            # if save_w_ca1=True
            [9] w_ca1_initial
            [10] w_ca1
            [11] w_ca1.mean() history              
"""


def main():

    results=[]

    if episodes==1:
        
        results.append(episode_run(jobID, 0,plot_flag,trials,changepos,Sero,
                                    eta_DA,eta_Sero, A_DA,A_Sero,Activ, Inhib, 
                                    tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3))
    else:

        pool = multiprocessing.Pool(os.cpu_count() - 1)

        for episode in range(0,episodes):
            print('Episode',episode)

            results.append(pool.apply_async(episode_run,(jobID,episode,plot_flag,trials,changepos,Sero,
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

    
def episode_run(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,
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
    theta_actor = np.reshape(2*np.pi*np.arange(1,AC.N+1)/AC.N,(40,1)) #angles actions
    diff_theta = np.matlib.repmat(theta_actor.T,AC.N,1) - np.matlib.repmat(theta_actor,1, AC.N)
    f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
    f = f - np.multiply(f,np.eye(AC.N))
    normalised = np.sum(f,axis=0)
    w_lateral = (w_minus/AC.N+w_plus*f/normalised) #lateral connectivity action neurons

    #actions
    actions = np.squeeze(a0*np.array([np.sin(theta_actor),np.cos(theta_actor)])) #possible actions (x,y)

    # feedforward weights
    w_in = np.ones([CA1.N, AC.N]).T / 5 # initialization feed-forward weights
    w_in_ca1 = np.random.rand(CA3.N, CA1.N).T

    W1 = np.zeros([AC.N, CA1.N]) #initialize unscale trace

    W1_sero = np.zeros([AC.N, CA1.N]) #initialize unscale trace
    trace_tot_sero = np.zeros([AC.N,CA1.N]) #sum of the traces
    eligibility_trace_sero = np.zeros([AC.N, CA1.N]) #total convolution

    ## initialise variables
    w_tot = np.concatenate((np.ones([CA1.N,AC.N]).T*w_in,w_lateral),axis=1)#total weigths
    w_ca1 = np.ones([CA1.N, CA3.N]) * w_in_ca1#total weigths

    store_pos = np.zeros([Trials, T_max,2]) # stores trajectories (for plotting)
    firing_rate_store = np.zeros([AC.N, T_max, Trials]) #stores firing rates action neurons (for plotting)
    firing_rate_store_CA1 = np.zeros([CA1.N, T_max, Trials])
    firing_rate_store_CA3 = np.zeros([CA3.N, T_max, Trials])

    ## initialize plot open field
    if plot_flag:

        fig, ax1, ax2, ax3, ax4, ax5, ax6 = initialize_plots(starting_position, r_goal, bounds_x, bounds_y,
                                                             CA1, offset_ca1, offset_ca3, CA3, c)
        fig.show()


    ## delete actions that lead out of the maze

    #find index place cells that lie on the walls (CA1)
    sides = np.empty((4,int(np.max([np.sqrt(CA1.N), np.sqrt(CA1.N)]))))

    sides[0,:] = np.where(CA1.pc[:,1] == CA1.pc.min())[0].T #bottom wall, y=-2
    sides[1,:] = np.where(CA1.pc[:,1] == CA1.pc.max())[0].T #top wall, y=+2
    sides[2,:] = np.where(CA1.pc[:,0] == CA1.pc.max())[0].T #left wall, x=-2
    sides[3,:] = np.where(CA1.pc[:,0] == CA1.pc.min())[0].T #right wall, x=+2

    #store index of actions forbidden from each side
    forbidden_actions = np.empty((4,19))
    forbidden_actions[0,:] = np.arange(11,30) #actions that point south - theta in (180, 360) degrees approx
    forbidden_actions[1,:] = np.concatenate([np.arange(1,10), np.arange(31,41)]) #actions that point north - theta in (0,180) degrees approx
    forbidden_actions[2,:] = np.arange(1,20) #actions that point east - theta in (-90, 90) degrees approx
    forbidden_actions[3,:] = np.arange(21,40) #actions that point west - theta in (90, 270) degrees approx
    
    #kill connections between place cells on the walls and forbidden actions
    w_walls = np.ones([AC.N, CA1.N+AC.N])
    w_walls_ca1 = np.ones([CA1.N, CA3.N])
    #for g in range(4):
    #    idx = list(itertools.product(forbidden_actions[g,:].astype(int).tolist(),sides[g,:].astype(int).tolist()))
    #    w_walls[np.array(idx)[:,0]-1,np.array(idx)[:,1]] = 0

    # optogenetic ranges
    ranges = [(T_max*Trials/6, 2*T_max*Trials/6), 
              (3*T_max*Trials/6, 4*T_max*Trials/6), 
              (5*T_max*Trials/6, T_max*Trials)]
    
    
    ######################## START SIMULATION ######################################
    w_tot_old = w_tot[0:AC.N,0:CA1.N] #store weights before start
    
    t_episode = 0 # counter ms

    for trial in range(Trials):

        pos = starting_position #initialize position at origin (centre open field)
        rew_found = 0 #flag that signals when the reward is found
        t_rew = T_max #time of reward - initialized at T_max at the beginning of the trial

        t_trial = 0

        X_ac= np.zeros(AC.N)  #matrix of spikes action neurons
        trace_tot = np.zeros([AC.N,CA1.N]) #sum of the traces
        eligibility_trace = np.zeros([AC.N, CA1.N]) #total convolution

        dw_ca1 = np.zeros(w_ca1.shape)

        print('Episode:', episode, 'Trial:', trial, 'Mean Weight:', w_ca1.mean())

        while t_trial<T_max:
           
            t_episode  += 1
            t_trial += 1

            store_pos[trial, t_trial-1, :] = pos 

            ## CA3 Layer

            if rew_found:
                rates_ca3[:] = 0 
            else:
                rates_ca3 = CA3.firing_rate(pos) #rate inhomogeneous poisson process

            X_ca3 = np.random.rand(CA3.N) <= rates_ca3 #spike train CA3 pcs
            firing_rate_store_CA3[:,t_trial-1,trial] = rates_ca3
            
            ## CA1 Layer
            rates_ca1 = CA1.firing_rate(pos, X_ca3, w_ca1*w_walls_ca1, t_episode)
            X_ca1 = CA1.get_spikes(rates_ca1, t_episode).reshape(-1,1)

            firing_rate_store_CA1[:,t_trial-1,trial] = np.squeeze(rates_ca1)

            # BCM update
            if CA1.alpha!=0:

                dw_ca1 += bcm(w_ca1, rates_ca1.mean(), rates_ca3, rates_ca1, epsilon=epsilon_bcm)

            ## action neurons
            # reset after last post-synaptic spike
            X_cut = np.matlib.repmat(np.concatenate((np.squeeze(X_ca1),X_ac)),AC.N,1).T
            X_cut = X_cut*AC.Canc

            rates_ac = AC.firing_rate(X_cut, w_tot*w_walls, t_episode)
            X_ac = AC.get_spikes(rates_ac, t_episode)

            rho = AC.compute_instantaneous_firing_rate(X_ac)

            firing_rate_store[:,t_trial-1,trial] = rho #store action neurons' firing rates

            # select action
            a = np.dot(rho, actions.T)/AC.N
            a[np.isnan(a)]=0


            ## synaptic plasticity
            #Rate-based update
            # Maybe here it goes u_ca1 in place of prob? (N_pc_ca3, AC.N) ()
            W1, eligibility_trace, trace_tot, W = weights_update_rate((A_pre_post+A_post_pre)/2, tau_pre_post, np.matlib.repmat(rates_ca1.T,AC.N,1), np.matlib.repmat(np.squeeze(rho),CA1.N,1).T, W1, trace_tot, tau_e)

            #STDP with unsymmetric window and depression due to serotonin
            if not(Inhib) and not(Activ):
                W1_sero, eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(rates_ca1.T,AC.N,1), np.matlib.repmat(np.squeeze(rho),CA1.N,1).T, W1_sero, trace_tot_sero, tau_e_sero)
            elif Activ and any(lower <= t_episode<= upper for (lower, upper) in ranges):
                #If there is overpotentiation of serotonin, assumed as doubled
                W1_sero, eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(rates_ca1.T,AC.N,1), np.matlib.repmat(np.squeeze(rho),CA1.N,1).T, W1_sero, trace_tot_sero, tau_e_sero)
            elif Inhib and any(lower <= t_episode<= upper for (lower, upper) in ranges):
                #If there is inhibition of serotonin, no eligibility trace is produced
                pass

            # online weights update (effective only with acetylcholine - ACh_flag=1)
            if ACh_flag:
                w_tot[0:AC.N,0:CA1.N]= w_tot[0:AC.N,0:CA1.N]-eta_ACh*W

            #weights limited between lower and upper bounds
            w_tot[np.where(w_tot[:,0:CA1.N]>w_max)] = w_max
            w_tot[np.where(w_tot[:,0:CA1.N]<w_min)] = w_min

            ## position update
            pos = np.squeeze(pos+a)
            #check if agent is out of boundaries. If it is, bounce back in the opposite direction
            if pos[0]<=bounds_x[0]:
                pos = pos+dx*np.array([1,0])
            else:
                if pos[0]>= bounds_x[1]:
                    pos = pos+dx*np.array([-1,0])
                else:
                    if pos[1]<=bounds_y[0]:
                        pos = pos+dx*np.array([0,1])
                    else:
                        if pos[1]>=bounds_y[1]:
                            pos = pos+dx*np.array([0,-1])
            
            ## reward

            if  rew_found==0 and np.sum((pos-c)**2)<=r_goal**2:

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
            
            update_plots(fig, ax1, ax2, ax3, ax4, ax5, ax6,
                         trial, store_pos, starting_position,
                         firing_rate_store, firing_rate_store_CA1,
                         firing_rate_store_CA3, w_tot,
                         CA1,ac)

    returns = (episode, rewarding_trials)

    return returns
     
if __name__ == '__main__':

    main()
