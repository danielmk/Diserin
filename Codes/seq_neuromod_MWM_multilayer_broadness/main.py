import sys
sys.path.extend(['./', '../', './Codes/'])

from numba import jit, cuda
import os
import numpy as np
import time
import multiprocessing
import pickle
import psutil
from tqdm import tqdm
import time
import yaml

from layers import *
from plot_functions import *
from utils import *
from plasticity_models import *
from environment import *

parameter_file = sys.argv[-1]

with open(parameter_file, 'r') as stream:
    try:
        conf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

options = conf['OPTIONS']

def main():

    start = time.time()

    results = []

    if conf['num_agents']==1:
        
        results.append(episode_run(0))

    else:

        pool = multiprocessing.Pool(os.cpu_count() - 1)

        for episode in range(0, conf['num_agents']):
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

    with open(conf['output_name']+'.pickle', 'wb') as myfile:

        pickle.dump((conf,results), myfile)

    
def episode_run(episode):

    # different random seed for each pool
    np.random.seed(conf['random_seed'] + episode)

    # flag to print first rewarding trial
    ever_rewarded_flag = False

    #Results to be exported for each episode
    rewarding_trials = np.zeros(conf['num_trials'])
    rewarding_times  = np.zeros(conf['num_trials']) - 1
    
    print('Initiated episode:',episode)

    ## Place cells positions

    environment = MWM(conf['GEOMETRY'])

    CA3 = CA3_layer(conf['CA3'])

    CA1 = CA1_layer(CA3.N, conf['CA1'])

    AC  = Action_layer(CA1.N, conf['AC'])

    plasticity_CA1 = BCM(CA1.N, conf['CA1'])

    plasticity_AC = Plasticity_AC(AC.N, CA1.N, conf['AC'])

    ## initialise variables
    store_pos = np.zeros([conf['num_trials'], conf['T_max'], 2]) # stores trajectories (for plotting)
    initial_weights = {'CA1': CA1.SRM0_model.W.copy(),
                       'AC': AC.neuron_model.W.copy()}

    if options['save_activity'] or options['plot_flag']:  

        firing_rate_store_AC = np.zeros([AC.N,   conf['T_max'], conf['num_trials']]) #stores firing rates action neurons (for plotting)
        firing_rate_store_CA1 = np.zeros([CA1.N, conf['T_max'], conf['num_trials']])
        firing_rate_store_CA3 = np.zeros([CA3.N, conf['T_max'], conf['num_trials']])


    ## initialize plot open field
    if options['plot_flag']:

        fig= initialize_plots(CA1, CA3, environment)

        update_plots(fig, 0, store_pos, None,
                     firing_rate_store_AC, firing_rate_store_CA1,
                     firing_rate_store_CA3, CA3, CA1, AC, environment)
        fig.show()
          
    
    ######################## START SIMULATION ######################################
    
    t_episode = 0 # counter ms

    for trial in range(conf['num_trials']):

        starting_position = get_starting_position(conf['GEOMETRY']['starting_position_option'])

        position = starting_position.copy()
        t_trial = 0

        print('Episode:', episode, 'Trial:', trial, flush=True)

        for t_trial in tqdm(range(conf['T_max'])):                    

            # store variables for plotting/saving
            store_pos[trial, t_trial, :] = position

            if options['save_activity'] or options['plot_flag']:
                firing_rate_store_CA3[:,t_trial,trial] = CA3.firing_rates
                firing_rate_store_CA1[:,t_trial,trial] = CA1.firing_rates
                firing_rate_store_AC[:,t_trial,trial] = AC.firing_rates 

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
            if CA1.alpha!=0. and conf['CA1']['plasticity_ON']:
                
                update = plasticity_CA1.get_update(CA3.firing_rates, CA1.firing_rates, 
                                                   use_sum=conf['CA1']['use_sum'], weights=CA1.SRM0_model.W)
                CA1.update_weights(update)

            plasticity_AC.update_traces(CA1.firing_rates, AC.firing_rates)

            ## position update

            position, wall_hit, reward_found = environment.update_position(position, a)
                        
            if conf['AC']['Acetylcholine'] and wall_hit:
                
                AC.update_weights(plasticity_AC.release_ACh(CA1.firing_rates, AC.firing_rates))

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
        if conf['AC']['Dopamine'] and reward_found:

            AC.update_weights(plasticity_AC.trace_DA)

        if conf['AC']['Serotonine'] and not reward_found:

            AC.update_weights(plasticity_AC.trace_5HT)

        ## plot
        if options['plot_flag']:
            
            update_plots(fig,trial, store_pos, starting_position,
                         firing_rate_store_AC, firing_rate_store_CA1,
                         firing_rate_store_CA3, CA3, CA1, AC, environment)

    returns = { 'episode':episode,  
                'rewarding_trials':rewarding_trials, 
                'rewarding_times': rewarding_times,
                'trajectories': store_pos,
                'initial_weights': initial_weights,
                'final_weights': {'CA1': CA1.SRM0_model.w_ca3,
                                  'AC' : AC.neuron_model.W}}
    
    if conf['save_activity']:

        returns['activities'] = {'CA3': firing_rate_store_CA3,
                                 'CA1': firing_rate_store_CA1,
                                 'AC': firing_rate_store_AC}
    
    return returns
     
if __name__ == '__main__':

    main()
