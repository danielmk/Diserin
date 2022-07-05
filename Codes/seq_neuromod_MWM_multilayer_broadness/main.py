"""Received from Carlos as working baseline.
"""

import sys
sys.path.extend(['../', './Codes/'])

from numba import jit, cuda
import os
import numpy as np
import time
import multiprocessing
import pickle
import psutil
from tqdm import tqdm
import time

from layers import *
from plot_functions import *
from utils import *
from plasticity_models import *
from environment import *

parameter_file = sys.argv[-1]
exec("from {} import *".format(parameter_file))

def main():

    start = time.time()

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

        print("Done! Simulation time: {:.2f} minutes.".format((time.time()-start)/60))

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

    environment = MWM(bounds_x, bounds_y, c, r_goal, dx, 
                      obstacle, obstacle_bounds_x, obstacle_bounds_y,
                      obstacle_2, obstacle_bounds_x_2, obstacle_bounds_y_2)

    CA3 = CA3_layer(bounds_x, bounds_y, space_pc, offset_ca3, rho_pc, sigma_pc_ca3)

    CA1 = CA1_layer(bounds_x, bounds_y, space_pc, offset_ca1, rho_pc, sigma_pc_ca1,
                    tau_m_ca1, tau_s_ca1, eps0_ca1, chi_ca1, rho0_ca1, theta_ca1, delta_u_ca1, ca3_scale, CA3.N,
                    w_min_ca1, w_max_ca1, w_ca1_init, max_init, sigma_init,
                    smooth_firing, tau_gamma_ca1, v_gamma_ca1)

    AC  = Action_layer(N_action, tau_m, tau_s, eps0, chi,
                       rho0, theta, delta_u, tau_gamma, 
                       v_gamma, CA1.N,
                       a0, psi, w_minus, w_plus,
                       weight_decay_ac, base_weight_ac, w_min, w_max, fixed_step)

    bcm = BCM(CA1.N, memory_factor, weight_decay, base_weight)
    
    plasticity_AC = Plasticity_AC(AC.N, CA1.N, A_DA, tau_DA, tau_e, A_Sero, tau_Sero, tau_e_sero, A_DA, tau_DA)

    ## initialise variables
    store_pos = np.zeros([trials, T_max,2]) # stores trajectories (for plotting)
    initial_weights = {'CA1': CA1.w_ca3.copy(),
                       'AC': AC.w_ca1.copy()}

    if save_activity or plot_flag:  

        firing_rate_store_AC = np.zeros([AC.N, T_max, trials]) #stores firing rates action neurons (for plotting)
        firing_rate_store_CA1 = np.zeros([CA1.N, T_max, trials])
        firing_rate_store_CA3 = np.zeros([CA3.N, T_max, trials])


    ## initialize plot open field
    if plot_flag:

        fig= initialize_plots(CA1, CA3)

        update_plots(fig, 0, store_pos, None,
                     firing_rate_store_AC, firing_rate_store_CA1,
                     firing_rate_store_CA3, CA3, CA1, AC)
        fig.show()
          
    
    ######################## START SIMULATION ######################################
    
    t_episode = 0 # counter ms

    for trial in range(trials):

        starting_position = get_starting_position(starting_position_option)

        position = starting_position.copy()
        t_trial = 0

        print('Episode:', episode, 'Trial:', trial, flush=True)

        for t_trial in tqdm(range(T_max)):                    

            # store variables for plotting/saving
            store_pos[trial, t_trial, :] = position

            if save_activity or plot_flag:
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
                
                update = bcm.get_update(CA3.firing_rates, CA1.firing_rates, CA1.w_ca3, use_sum=False)
                CA1.update_weights(eta_bcm * update)

            plasticity_AC.update_traces(CA1.firing_rates, AC.instantaneous_firing_rates)

            ## position update

            position, wall_hit, reward_found = environment.update_position(position, a)
            
            if Acetylcholine and wall_hit:
                
                AC.update_weights(-eta_ACh*plasticity_AC.release_ACh(CA1.firing_rates, AC.instantaneous_firing_rates))

            if reward_found:

                rewarding_trials[trial] = 1
                rewarding_times[trial] = t_trial
                
                if not ever_rewarded_flag:

                    ever_rewarded_flag = True
                    print('First reward,episode',episode,'trial', trial)

                break

            t_episode  += 1

        ## update weights - end of trial

        # change due to serotonin or dopamine
        if Dopamine and reward_found:

            AC.update_weights(eta_DA*plasticity_AC.trace_DA)

        if Serotonine and not reward_found:

            AC.update_weights(-eta_Sero*plasticity_AC.trace_5HT)

        ## plot
        if plot_flag:
            
            update_plots(fig,trial, store_pos, starting_position,
                         firing_rate_store_AC, firing_rate_store_CA1,
                         firing_rate_store_CA3, CA3, CA1, AC)

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
    
    return returns
     
if __name__ == '__main__':

    main()
