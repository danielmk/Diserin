"""Received from Carlos as working baseline.
"""

import sys
sys.path.extend(['../', './Codes/'])

from numba import jit, cuda

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

def main():

    results=[]

    for episode in range(0,episodes):

        print('Episode',episode)
        # results.append(pool.apply_async(episode_run,(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,Activ,Inhib,tau_DA,tau_Sero,),error_callback=log_e))
        results.append(episode_run(jobID,episode,plot_flag,trials,changepos,Sero,
                                    eta_DA,eta_Sero, A_DA,A_Sero,Activ, Inhib, 
                                    tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3))
        # ca1_spikes = episode_run(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,Activ,Inhib,tau_DA,tau_Sero,)
        # return ca1_spikes

    # results_episode = [result.get() for result in results]

    with open(jobID+'.pickle', 'wb') as myfile:
        pickle.dump((descriptor,results), myfile)

    return results

def log_e(e):
  print(e)

def get_quadrant(x,y):
    """ 
    Receives the x and y coordinates of the agent.
    Returns 
        0,1,2,3 if the agent is in the I,II,III,IV quadrant respectively
        False if the agent is in the origin
    """

    if x>=0 and y>0 : return 0
    if x<0 and y>=0 : return 1
    if x<=0 and y<0 : return 2
    if x>0  and y<=0: return 3

    return False
    
def make_pc_layer(bounds_x, bounds_y, space_pc, offset):
    
    if offset:
        x_pc = np.round(np.arange(bounds_x[0], bounds_x[1], space_pc)+space_pc/2,2)
        y_pc = np.round(np.arange(bounds_y[0], bounds_y[1], space_pc)+space_pc/2,2)
    else:
        x_pc = np.round(np.arange(bounds_x[0], bounds_x[1]+space_pc, space_pc),2)
        y_pc = np.round(np.arange(bounds_y[0], bounds_y[1]+space_pc, space_pc),2)

    n_x = np.size(x_pc) #nr of place cells on axis x
    n_y = np.size(y_pc) #nr of place cells on axis y

    xx, yy = np.meshgrid(x_pc, y_pc)
    pc = np.stack([xx,yy], axis=2).reshape(-1,2)
    N_pc = pc.shape[0] #number of place cells

    return pc, N_pc, n_x, n_y


def episode_run(jobID,episode,plot_flag,Trials,changepos,Sero,eta_DA,eta_Sero,A_DA,A_Sero,
                Activ,Inhib,tau_DA,tau_Sero,ca3_scale, offset_ca1, offset_ca3):

    # different random seed for each pool
    np.random.seed(random_seed + episode)

    # flag to print first rewarding trial
    ever_rewarded_flag = False

    #Results to be exported for each episode
    rewarding_trials = np.zeros(Trials)
    quadrant_map = np.zeros([Trials,4])
    median_distance = np.zeros(Trials)
    activities = {'ca3': [], 'ca1': [], 'ac': []}
    
    ca1_spikes = []
    
    print('Initiated episode:',episode)
    rew1_flag=1  #rewards are in the initial positions
    rew2_flag=0  #reward are switched
    ACh_flag=0 #acetylcholine flag if required for comparisons
    
    t_rew = T_max
    t_extreme = t_rew + delay_post_reward #time of reward - initialized to maximu

    ## Place cells positions

    pc_ca1, N_pc_ca1, n_x_ca1, n_y_ca1 = make_pc_layer(bounds_x, bounds_y, space_pc, offset_ca1)
    pc_ca3, N_pc_ca3, n_x_ca3, n_y_ca3 = make_pc_layer(bounds_x, bounds_y, space_pc, offset_ca3)

    #winner-take-all weights
    theta_actor = np.reshape(2*np.pi*np.arange(1,N_action+1)/N_action,(40,1)) #angles actions
    diff_theta = np.matlib.repmat(theta_actor.T,N_action,1) - np.matlib.repmat(theta_actor,1, N_action)
    f = np.exp(psi*np.cos(diff_theta)) #lateral connectivity function
    f = f - np.multiply(f,np.eye(N_action))
    normalised = np.sum(f,axis=0)
    w_lateral = (w_minus/N_action+w_plus*f/normalised) #lateral connectivity action neurons

    #actions
    actions = np.squeeze(a0*np.array([np.sin(theta_actor),np.cos(theta_actor)])) #possible actions (x,y)

    # feedforward weights
    w_in = np.ones([N_pc_ca1, N_action]).T / 5 # initialization feed-forward weights
    w_in_ca1 = np.random.rand(N_pc_ca3, N_pc_ca1).T

    W1 = np.zeros([N_action, N_pc_ca1]) #initialize unscale trace

    W1_sero = np.zeros([N_action, N_pc_ca1]) #initialize unscale trace
    trace_tot_sero = np.zeros([N_action,N_pc_ca1]) #sum of the traces
    eligibility_trace_sero = np.zeros([N_action, N_pc_ca1]) #total convolution

    ## initialise variables
    w_tot = np.concatenate((np.ones([N_pc_ca1,N_action]).T*w_in,w_lateral),axis=1)#total weigths
    w_ca1 = np.ones([N_pc_ca3,N_pc_ca1]).T*w_in_ca1#total weigths
    new_weight_buffer = w_ca1

    time_reward     = np.zeros(Trials) #stores time of reward 1
    time_reward2    = np.zeros(Trials) #stores time of reward 2 (moved)
    time_reward_old = np.zeros(Trials) #stores time when agent enters the previously rewarded location

    store_pos = np.zeros([Trials, T_max,2]) # stores trajectories (for plotting)
    firing_rate_store = np.zeros([N_action, T_max, Trials]) #stores firing rates action neurons (for plotting)

    ## initialize plot open field
    if plot_flag:

        plt.close()
        fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
        fig.subplots_adjust(hspace = 0.5)

        plt.ion()
        #Plot of reward places and initial position
        reward_plot = ax1.plot(c[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot reward 1
        ax1.plot(starting_position[0],starting_position[1], 'r',marker='o',markersize=5) #plot initial starting point

        #plot walls
        ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[1],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[0],bounds_x[1]], [bounds_y[0],bounds_y[0]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[0],bounds_x[0]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)
        ax1.plot([bounds_x[1],bounds_x[1]], [bounds_y[0],bounds_y[1]],c='k', ls='--',lw=0.5)


        ax1.scatter(pc_ca1[:,0],pc_ca1[:,1], s=1, label="CA1")

        if offset_ca1!=offset_ca3:
            ax1.scatter(pc_ca3[:,0],pc_ca3[:,1], s=1, label ="CA3")
            ax1.legend()

        ax1.set_title('Trial 0')
        ax2.set_title('Action neurons firing rates')
        ax3.set_title('Mean weights')
        ax4.set_title('Agent''s policy')

        ax4.set_xlim([pc_ca1.min(),pc_ca1.max()])
        ax4.set_ylim([pc_ca1.min(),pc_ca1.max()])

        plt.pause(0.00001)

        fig.show()


    ## delete actions that lead out of the maze

    #find index place cells that lie on the walls (CA1)
    sides = np.empty((4,np.max([n_x_ca1,n_y_ca1])))

    sides[0,:] = np.where(pc_ca1[:,1] == pc_ca1.min())[0].T #bottom wall, y=-2
    sides[1,:] = np.where(pc_ca1[:,1] == pc_ca1.max())[0].T #top wall, y=+2
    sides[2,:] = np.where(pc_ca1[:,0] == pc_ca1.max())[0].T #left wall, x=-2
    sides[3,:] = np.where(pc_ca1[:,0] == pc_ca1.min())[0].T #right wall, x=+2

    #store index of actions forbidden from each side
    forbidden_actions = np.empty((4,19))
    forbidden_actions[0,:] = np.arange(11,30) #actions that point south - theta in (180, 360) degrees approx
    forbidden_actions[1,:] = np.concatenate([np.arange(1,10), np.arange(31,41)]) #actions that point north - theta in (0,180) degrees approx
    forbidden_actions[2,:] = np.arange(1,20) #actions that point east - theta in (-90, 90) degrees approx
    forbidden_actions[3,:] = np.arange(21,40) #actions that point west - theta in (90, 270) degrees approx
    
    #kill connections between place cells on the walls and forbidden actions
    w_walls = np.ones([N_action, N_pc_ca1+N_action])
    w_walls_ca1 = np.ones([N_pc_ca1, N_pc_ca3])
    for g in range(4):
        idx = list(itertools.product(forbidden_actions[g,:].astype(int).tolist(),sides[g,:].astype(int).tolist()))
        w_walls[np.array(idx)[:,0]-1,np.array(idx)[:,1]] = 0

    # optogenetic ranges
    ranges = [(T_max*Trials/6, 2*T_max*Trials/6), 
              (3*T_max*Trials/6, 4*T_max*Trials/6), 
              (5*T_max*Trials/6, T_max*Trials)]
    
    
    ######################## START SIMULATION ######################################
    w_tot_old = w_tot[0:N_action,0:N_pc_ca1] #store weights before start
    ca3_activities = []
    ca1_activities = []
    ac_activities = []
    w_ca1_initial = w_ca1
    
    t_episode = 0 # counter ms

    for trial in range(Trials):

        median_tr = []
        pos = starting_position #initialize position at origin (centre open field)
        rew_found = 0 #flag that signals when the reward is found
        t_rew = T_max #time of reward - initialized at T_max at the beginning of the trial

        t_trial = 0

        #initialisation variables - reset between trials
        Y_action_neurons= np.zeros([N_action, 1])  #matrix of spikes action neurons
        X_cut = np.zeros([N_pc_ca1+N_action, N_action])  #matrix of spikes place cells
        
        epsp_rise  = np.zeros([N_action+N_pc_ca1, N_action]) #epsp rise compontent convolution
        epsp_decay = np.zeros([N_action+N_pc_ca1, N_action]) #epsp decay compontent convolution
        epsp_tot   = np.zeros([N_action+N_pc_ca1, N_action]) #epsp
        Canc = np.ones([N_pc_ca1+N_action, N_action]).T

        epsp_rise_ca1  = np.zeros([N_pc_ca3, N_pc_ca1])
        epsp_decay_ca1 = np.zeros([N_pc_ca3, N_pc_ca1])
        Canc_ca1 = np.ones([N_pc_ca3, N_pc_ca1]).T
        
        rho_action_neurons= np.zeros([N_action,1]) #firing rate action neurons
        rho_rise= np.copy(rho_action_neurons)  #firing rate action neurons, rise compontent convolution
        rho_decay = np.copy(rho_action_neurons) #firing rate action neurons, decay compontent convolution
        
        last_spike_post=np.zeros([N_action,1])-1000 #vector time last spike postsynaptic neuron
        last_spike_ca1 = np.zeros([N_pc_ca1,1])-1000
        
        trace_tot = np.zeros([N_action,N_pc_ca1]) #sum of the traces
        eligibility_trace = np.zeros([N_action, N_pc_ca1]) #total convolution
        w_ca1 = new_weight_buffer

        print('Episode:', episode, 'Trial:', trial, 'Mean Weight:', w_ca1.mean())

        #change reward location in the second half of the experiment
        if ( trial==Trials//2) and changepos:
            rew1_flag=0
            rew2_flag=1
            np.linspace(-np.pi, np.pi, 100)
            if plot_flag:
                reward_plot.pop(0).remove()
                #punish_plot.pop(0).remove() NOT DEFINED!
                reward_plot = ax1.plot(c2[0]+r_goal*np.cos(np.linspace(-np.pi,np.pi,100)), c2[1]+r_goal*np.sin(np.linspace(-np.pi,np.pi,100)),'b') #plot negative reward 2

            print("Switched the position of the reward to location 2!")

        while t_trial<T_max:
            
            t_episode  += 1
            t_trial+= 1

            ## place cells (CA3 layer)
            rhos = rho_pc * np.exp(-np.sum((pos-pc_ca3)**2,axis=1)/(sigma_pc_ca3**2)) #rate inhomogeneous poisson process
            prob = rhos
            ca3_activities.append(rhos)

            #turn place cells off after reward is reached
            if t_trial>t_rew:
                prob = np.zeros_like(rhos)

            X = (np.random.rand(1,N_pc_ca3)<=prob.T).T #spike train pcs
            
            epsp_rise_ca1= epsp_rise_ca1 * Canc_ca1.T
            epsp_decay_ca1= epsp_decay_ca1 * Canc_ca1.T
            
            # CA1 cells
            epsp_ca1, epsp_decay_ca1, epsp_rise_ca1 = convolution(epsp_decay_ca1, epsp_rise_ca1, tau_m, tau_s, eps0, X, np.multiply(w_ca1,w_walls_ca1)) #EPSP in the model * weights
            
            X_ca1, last_spike_ca1, Canc_ca1, u_ca1 = neuron_ca1(epsp_ca1, chi, last_spike_ca1, tau_m, rho_pc, theta, delta_u,t_episode, pos, n_x_ca1, n_y_ca1, pc_ca1, sigma_pc_ca1, ca3_scale) #sums EPSP, calculates potential and spikes
            ca1_activities.append(u_ca1)

            ca1_spikes.append(X_ca1)

            #dw_ca1 = bcm(w_ca1, 0.009133, rhos, u_ca1, epsilon=0.001)
            dw_ca1 = bcm(w_ca1, u_ca1.mean(), rhos, u_ca1, epsilon=0.0001)

            new_weight_buffer = new_weight_buffer + dw_ca1 / 100
            
            
            #sys.exit()
            store_pos[trial, t_trial-1, :] = pos #store position (for plotting)

            #save quadrant
            quadrant_map[trial, get_quadrant(pos[0], pos[1])] += 1

            # save median distance to centre
            median_tr.append(np.linalg.norm(pos))

            ## reward
            # agent enters reward 1 in the first half of the trial
            if  rew_found==0 and rew1_flag==1 and np.sum((pos-c)**2)<=r_goal**2:

                rew_found=1 #flag reward found (so that trial is ended soon)
                t_rew=t_trial #time of reward
                time_reward[trial] = t_trial #store time of reward
                rewarding_trials[trial]+=1
                
                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)


            #cases for location switching

            # agent enters reward 2 in the second half of the trial
            elif rew_found==0 and rew2_flag==1 and np.sum((pos-c2)**2)<=r_goal2**2:

                rew_found=1  #flag reward 2 found (so that trial is ended soon)
                t_rew=t_trial #time of reward 2
                time_reward2[trial] = t_trial #store time of reward 2
                rewarding_trials[trial]+=1

                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)

            elif rew1_flag==0 and rew2_flag==1 and np.sum((pos-c)**2)<=r_goal**2:
                #this location is no longer rewarded, so the trial is not ended
                time_reward_old[trial]=t_trial #store time of entrance old reward location


            ## action neurons

            # reset after last post-synaptic spike
            X_cut = np.matlib.repmat(np.concatenate((X_ca1,Y_action_neurons)),1,N_action)
            # X_cut = np.matlib.repmat(np.concatenate((X,Y_action_neurons)),1,N_action)
            X_cut = np.multiply(X_cut,Canc.T)

            epsp_rise=np.multiply(epsp_rise,Canc.T)
            epsp_decay=np.multiply(epsp_decay,Canc.T)
            # neuron model
            epsp_tot, epsp_decay, epsp_rise = convolution(epsp_decay, epsp_rise, tau_m, tau_s, eps0, X_cut, np.multiply(w_tot,w_walls)) #EPSP in the model * weights
            Y_action_neurons,last_spike_post, Canc, u_ac = neuron(epsp_tot, chi, last_spike_post, tau_m, rho0, theta, delta_u, t_episode) #sums EPSP, calculates potential and spikes
            ac_activities.append(u_ac)
            
            # smooth firing rate of the action neurons
            rho_action_neurons, rho_decay, rho_rise = convolution (rho_decay, rho_rise, tau_gamma, v_gamma, 1, Y_action_neurons)
            firing_rate_store[:,t_trial-1,trial] = np.squeeze(rho_action_neurons) #store action neurons' firing rates
            # select action
            a = (np.dot(rho_action_neurons.T,np.squeeze(actions).T)/N_action)
            a[np.isnan(a)]=0
            ## synaptic plasticity

            #Rate-based update
            W1, eligibility_trace, trace_tot, W = weights_update_rate((A_pre_post+A_post_pre)/2, tau_pre_post, np.matlib.repmat(prob,N_action,1), np.matlib.repmat(np.squeeze(rho_action_neurons),N_pc_ca1,1).T, W1, trace_tot, tau_e)

            #STDP with unsymmetric window and depression due to serotonin
            if not(Inhib) and not(Activ):
                W1_sero, eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(prob,N_action,1), np.matlib.repmat(np.squeeze(rho_action_neurons),N_pc_ca1,1).T, W1_sero, trace_tot_sero, tau_e_sero)
            elif Activ and any(lower <= t_episode<= upper for (lower, upper) in ranges):
                #If there is overpotentiation of serotonin, assumed as doubled
                W1_sero, eligibility_trace_sero, trace_tot_sero, W_sero = weights_update_rate((A_pre_post_sero+A_post_pre_sero)/2, tau_pre_post_sero, np.matlib.repmat(prob,N_action,1), np.matlib.repmat(np.squeeze(rho_action_neurons),N_pc_ca1,1).T, W1_sero, trace_tot_sero, tau_e_sero)
            elif Inhib and any(lower <= t_episode<= upper for (lower, upper) in ranges):
                #If there is inhibition of serotonin, no eligibility trace is produced
                pass

            # online weights update (effective only with acetylcholine - ACh_flag=1)
            w_tot[0:N_action,0:N_pc_ca1]= w_tot[0:N_action,0:N_pc_ca1]-eta_ACh*W*(ACh_flag)

            #weights limited between lower and upper bounds
            w_tot[np.where(w_tot[:,0:N_pc_ca1]>w_max)] = w_max
            w_tot[np.where(w_tot[:,0:N_pc_ca1]<w_min)] = w_min

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

            #time when trial end is 300ms after reward is found
            t_extreme = t_rew+delay_post_reward

            if t_trial>t_extreme and t_trial<T_max:
                t_episode = int((np.ceil(t_episode/T_max))*T_max)-1 
                t_end = t_extreme #for plotting
                break


        ## update weights - end of trial

        # if the reward is not found, no change (and not Inhib is true)
        # change due to serotonin or dopamine
        if Sero:
            w_tot[0:N_action,0:N_pc_ca1]= (w_tot_old+eta_DA*eligibility_trace)*rew_found + (w_tot_old-eta_Sero*eligibility_trace_sero)*(1-rew_found)*(not Inhib)
        else:
            #change due to dopamine or sustained weights at the end of the trial
            w_tot[0:N_action,0:N_pc_ca1]=w_tot[0:N_action,0:N_pc_ca1]*(1-rew_found)+(w_tot_old+eta_DA*eligibility_trace)*rew_found

        #weights limited between lower and upper bounds
        w_tot[np.where(w_tot[:,0:N_pc_ca1]>w_max)]=w_max
        w_tot[np.where(w_tot[:,0:N_pc_ca1]<w_min)]=w_min

        #store weights before the beginning of next trial (for updates in case reward is found)
        w_tot_old = np.copy(w_tot[0:N_action,0:N_pc_ca1])

        #calculate policy
        ac =np.dot(np.squeeze(actions),(np.multiply(w_tot_old,w_walls[:,0:N_pc_ca1]))/a0) #vector of preferred actions according to the weights
        ac[:,np.unique(np.sort(np.reshape(sides, (np.max(sides.shape)*4, 1),order='F'))).astype(int).tolist()]=0 #do not count actions AT the boundaries (just for plotting)

        ##save median_distance
        median_distance[trial] = np.median(median_tr)

        ## plot
        if plot_flag:
            
            ax1.set_title('Trial '+str(trial))
            #display trajectory of the agent in each trial
            f3 =ax1.plot(store_pos[trial, : ,0], store_pos[trial, : ,1]) #trajectory
            ax1.plot(starting_position[0],starting_position[1],'r',marker='o',markersize=5) #starting point

            #display action neurons firing rates (activity bump)
            pos = ax2.imshow(
                firing_rate_store[:,:,trial],cmap='Blues', interpolation='none',aspect='auto')
            #colorbar
            if trial==1:
                fig.colorbar(pos, ax=ax2)
            #display weights over the open field, averaged over action neurons
            w_plot = np.mean(w_tot[:,0:N_pc_ca1],axis=0) #use weights as they were at the beginning of the trial
            w_plot = np.reshape(w_plot,(int(np.sqrt(N_pc_ca1)),int(np.sqrt(N_pc_ca1))))
            pos2 = ax3.imshow(w_plot,cmap='Reds_r',origin='lower', interpolation='none',aspect='auto')
            #set(gca,'YDir','normal')
            if trial==1:
                fig.colorbar(pos2, ax=ax3)

            #plot policy as a vector field
            #filter zero values
            ac_norm=np.max(np.linalg.norm(ac,axis=0))
            f4=ax4.quiver(pc_ca1[:,0], pc_ca1[:,1], ac[0,:].T/ac_norm, ac[1,:].T/ac_norm)
            #time.sleep(1.0)
            fig.canvas.draw()
            plt.pause(0.00001)
            f3.pop(0).remove()
            f4.remove()
            pos2.remove()
            t_end = T_max


    activities['ca1'].append(ca1_activities)
    activities['ca3'].append(ca3_activities)
    activities['ac'].append(ac_activities)

    return episode, rewarding_trials,\
        quadrant_map,median_distance,time_reward,time_reward2,time_reward_old,\
        store_pos, activities, w_ca1_initial, w_ca1


if __name__ == '__main__':

    results = main()
