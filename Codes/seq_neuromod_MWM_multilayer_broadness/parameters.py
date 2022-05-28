import numpy as np

builtin_variables = set(globals())

"""Random seed"""
random_seed = 41 # 455 for blow ups

""" Main options """
jobID = 'results' #ID of the JOB, the results will be saved as 'jobID.pickle'
episodes = 1 # number of agents
trials = 40 # number of trials for each agent
plot_flag = True
changepos = False
Sero = False
Activ = False # Cyclic serotonin potentiation
Inhib = False # Cyclic serotonin inhibition
offset_ca1 = False
offset_ca3 = True
save_activities = False # Whether to store in the results.pickle file also the activitiy history for the layers. Note: this is very memory consuming, 15GB for about 20 episodes
save_w_ca1 = True
save_pos = False
ACh_flag = False #acetylcholine flag if required for comparisons
walls_punish = False

""" Learning rates """
eta_DA = 0.01 # Learning rate for dopamine
eta_Sero = 0.01 # Learning rate for serotonin
eta_ACh = 2e-2 #learning rate acetylcholine (if active)

"""STDP parameters"""
A_DA = 1 # STDP magnitude of dopamine
A_Sero = 1 # STDP magnitude of serotonin

tau_Sero = 10 # Time constant for the STDP window of serotonin
tau_DA = 10 # Time constant for the STDP window of dopamine

A_pre_post=A_DA   #amplitude pre-post window
A_post_pre=A_DA   #amplitude post-pre window
tau_pre_post= tau_DA   #time constant pre-post window
tau_post_pre= tau_DA   #time constant post-pre window
tau_e= 2*10**3 #time constant eligibility trace

A_pre_post_sero=A_Sero   #amplitude pre-post window for serotonin
A_post_pre_sero=0   #amplitude post-pre window for serotonin
tau_pre_post_sero= tau_Sero   #time constant pre-post window for serotonin
tau_post_pre_sero= tau_Sero   #time constant post-pre window for serotonin
tau_e_sero= 5*10**3 #time constant eligibility trace for serotonin


""" Time and Geometry parameters """
T_max = 15*10**3 #maximum time trial
delay_post_reward = 300

# Reward and agent positions
starting_position = np.array([0.,0.]) #starting position
c = np.array([-1.5,-1.5]) #centre reward 1
r_goal = 0.3 # radius goal area
c2 = np.array([1.5,1.5]) #centre reward 2
r_goal2 = 0.3 # radius goal area 2

# Space cells parameters
space_pc = 0.4 #place cells separation distance
bounds_x = np.array([-2,2]) #bounds open field, x axis
bounds_y = np.array([-2,2]) #bounds open field, y axis
rho_pc=400*10**(-3) #maximum firing rate place cells, according to Poisson
sigma_pc_ca3 = 0.4
sigma_pc_ca1 = 0.4 # (increase this for broadness)


""" Action neurons parameters"""
eps0 = 20 #scaling constant epsp
tau_m = 20 #membrane time constant
tau_s = 5 #synaptic time rise epsp
chi = -5 #scaling constant refractory effect
rho0 = 60*10**(-3) #scaling rate
theta = 16 #threshold
delta_u = 2 #escape noise
N_action = 40 #number action neurons

# action selection
tau_gamma = 50 #raise time convolution action selection
v_gamma = 20 #decay time convolution action selection

psi = 20 # the higher, the more narrow the range of excitation
w_minus = -300 # (consider decreasing it)
w_plus = 100

a0=.08 # action?

dx = 0.01 # length of bouncing back from walls


""" CA3 parameters"""
ca3_scale = 0. # To what extent does CA1 receive CA3 input? (between 0 and 1)

"""BCM parameters"""
theta_bcm = 'sliding' # double, the value of bcm threshold. You could also put the string "sliding", and mean activity of u_ca1 would be used
epsilon_bcm = 1e-4 # weight decay strength
eta_bcm = 1e-3

""" Other parameters"""
## Feed-forward weights parameters
w_max=3 #upper bound feed-forward proto-weights and weights
w_min=1 #.pwer bound feed-forward weights


""" Dict with all previous parameters, for saving configuration """
# collect all global variables in a dict
descriptor = globals().copy()

# remove builtin variables
descriptor.pop('builtin_variables')
for key in builtin_variables:
    descriptor.pop(key)


