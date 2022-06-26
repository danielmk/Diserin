import numpy as np

builtin_variables = set(globals())

"""Random seed"""
random_seed = 40

""" Main options """

jobID = 'results' #ID of the JOB, the results will be saved as 'jobID.pickle'
episodes = 1 # number of agents
trials = 40 # number of trials for each agent
plot_flag = True
save_activity = False
BCM_ON = True

Dopamine = True
Serotonine = False
Acetylcholine = True

offset_ca1 = False
offset_ca3 = False

obstacle = True
obstacle_2 = True

"""Weights boundaries"""
w_max = 3
w_min = 1

""" Learning rates """
eta_DA = 0.2 # Learning rate for dopamine
eta_Sero = 0.01 # Learning rate for serotonin
eta_ACh = 0.01 #learning rate acetylcholine (if active)

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

bounds_x = np.array([-2,2]) #bounds open field, x axis
bounds_y = np.array([-2,2]) #bounds open field, y axis
c = np.array([-1.5,-1.5]) #centre reward 1
r_goal = 0.3 # radius goal area

starting_position_option = 'upper-right' # option: 'origin', 'random'

obstacle_bounds_x = np.array([-1., -0.8 ])
obstacle_bounds_y = np.array([-2, 0. ])

obstacle_bounds_x_2 = np.array([0.8, 1 ])
obstacle_bounds_y_2 = np.array([0., 2. ])


"""Space cells parameters"""
space_pc = 0.4 #place cells separation distance

rho_pc = 0.4 #maximum firing rate place cells, according to Poisson
sigma_pc_ca3 = 0.4
sigma_pc_ca1 = 0.4 # (increase this for broadness)


"""CA1 parameters"""
# Weights
w_min_ca1 = 0
w_max_ca1 = 3

w_ca1_init = 'convolutional' # option: 'convolutional', 'uniform', 'identity'
max_init = 2. # needed just with convolutional opiton
sigma_init = 1.5 # needed just with convolutional opiton

# SRM0
eps0_ca1 = 20*5
tau_m_ca1 = 20
tau_s_ca1 = 5
chi_ca1 = -5/10
rho0_ca1 = 0.06#scaling rate
theta_ca1 = 16
delta_u_ca1 = 2 #    /20 with the identity


""" Action neurons parameters"""
N_action = 40 #number action neurons

eps0 = 20 #scaling constant epsp
tau_m = 20 #membrane time constant
tau_s = 5 #synaptic time rise epsp
chi = -5 #scaling constant refractory effect

rho0 = 0.06 #scaling rate
theta = 16 #threshold
delta_u = 5 #escape noise

weight_decay_ac = 0.
base_weight_ac = 2.

# action selection
tau_gamma = 20 # decay time for right tale
v_gamma = 5 # rise time for left tale
psi = 20 # the higher, the more narrow the range of excitation
w_minus = -300 # (consider decreasing it)
w_plus = 100

a0= 0.08 # action
fixed_step = 0.003 # None for not fixed step (velocity with 0.003 = 0.3m/s)
dx = 0.01 # length of bouncing back from walls


""" CA3 parameters"""
ca3_scale = 1.  # To what extent does CA1 receive CA3 input? (between 0 and 1)


"""BCM parameters"""
memory_factor = 0.99
weight_decay = 0.
base_weight = 2.
eta_bcm = 1e-2

""" Dict with all previous parameters, for saving configuration """
# collect all global variables in a dict
descriptor = globals().copy()

# remove builtin variables
descriptor.pop('builtin_variables')
for key in builtin_variables:
    descriptor.pop(key)


