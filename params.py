#!/usr/bin/env python3

import numpy as np
from project_utils_2 import *
import pydot

#changable
mu1 = 0.3
mu2 = 0.3
T = 10
to_rotate = np.pi
finding_friction = False

#fixed params
gravity = np.array([0, 0, -9.8])

###########   KUKA
q0_init = [-np.pi/2, -np.pi/2, 0.0, -np.pi/2, 0, 0.0, 0]
q_dot_min = -1*np.ones(7)
q_dot_max = 1*np.ones(7)

q_ddot_min = -1*np.ones(7)
q_ddot_max = 1*np.ones(7)
q_ddot_low = -0.25*np.ones(7)
q_ddot_upp = 0.25*np.ones(7)

tau_min = -70.0*np.ones(7)
tau_max = 70.0*np.ones(7)
total_iter = 0

##########   OBJECT
m_obj = 0.05
len_obj = 0.1
M_obj = m_obj*np.eye(3)
I_obj = m_obj*len_obj**2/6 * np.eye(3)

###########  KUKA-OBJECT

mu = 2*mu1*mu2/(mu1+mu2)
mu_ = mu/np.sqrt(2)
factor = 0#0.01 # friction safety margin
contact_pts_wrt_slab = [
    np.array([ 0.05, -0.05, 0.025]),
    np.array([-0.05, -0.05, 0.025]),
    np.array([ 0.05,  0.05, 0.025]),
    np.array([-0.05,  0.05, 0.025])
]

estimated_contact_pts = np.array([
    np.array([ 0.05, -0.05, 0.025]),
    np.array([-0.05, -0.05, 0.025]),
    np.array([ 0.05,  0.05, 0.025]),
    np.array([-0.05,  0.05, 0.025])
])

box_com_wrt_slab = np.array([0, 0, 0.025+len_obj/2])

iiwa_diagram, iiwa_plant = Iiwa_plant(mu1)
auto_diff_plant = iiwa_plant.ToAutoDiffXd()
auto_diff_context = auto_diff_plant.CreateDefaultContext()
iiwa = auto_diff_plant.GetModelInstanceByName('iiwa')

########## OPTIMIZATION PARAMETERS
ss = np.linspace(0, 1, T+1)
nq = auto_diff_plant.num_positions()
nf = 3 #no of contact forces
delta_s = 1/T

simulate_traj = True
plot_results = True
debug_fric = False

if(not finding_friction):
    init_opt_variables = True
    add_constraints = True
    set_initial_guess = True
else:
    init_opt_variables = False
    add_constraints = False
    set_initial_guess = False
