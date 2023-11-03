#!/usr/bin/env python3

from params import *
import params
from utils import get_q, get_dq_ds, get_d2q_ds2
from manipulator_equations import manipulator_equations
from object_equations import object_equations
import numpy as np
from pydrake.all import MathematicalProgram, eq
from project_utils_2 import *
from cost_fn import cost_func

if init_opt_variables:
    prog = MathematicalProgram()
    q_s = np.array([get_q(i) for i in ss])
    dq_ds = get_dq_ds(q_s)
    d2q_ds2 = get_d2q_ds2(dq_ds)
    # Variables
    # h = prog.NewContinuousVariables(T, name = 'h')
    f1 = prog.NewContinuousVariables(rows=T, cols=nf, name='f1')
    f2 = prog.NewContinuousVariables(rows=T, cols=nf, name='f2')
    f3 = prog.NewContinuousVariables(rows=T, cols=nf, name='f3')
    f4 = prog.NewContinuousVariables(rows=T, cols=nf, name='f4')

    S_dot = prog.NewContinuousVariables(T+1, name = 'S_dot')
    S_ddot = prog.NewContinuousVariables(T, name = 'S_ddot')

    q = prog.NewContinuousVariables(rows=T+1, cols=nq, name = 'q')
    q_dot = prog.NewContinuousVariables(rows=T+1, cols=nq, name = 'q_dot')
    q_ddot= prog.NewContinuousVariables(rows=T, cols=nq, name = 'q_ddot')

    tau = prog.NewContinuousVariables(rows=T, cols=nq, name = 'tau')

    lin_acc = prog.NewContinuousVariables(rows=T, cols=3, name = 'lin_acc')
    ang_acc = prog.NewContinuousVariables(rows=T, cols=3, name = 'ang_acc')

    total_force = prog.NewContinuousVariables(rows=T, cols=3, name = 'tot_force')


if add_constraints:
    # S_dot positive constraint
    for i in range(T+1):
        prog.AddLinearConstraint(S_dot[i] >= 0)

    # Interpolation Constraints
    for i in range(T):
        prog.AddConstraint(S_ddot[i] == (S_dot[i+1]**2 - S_dot[i]**2)/(2*delta_s))

    # q constraint
    for i in range(T+1):
        prog.AddConstraint(eq(q_s[i], q[i]))


    # q_dot constraint
    for i in range(T+1):
        prog.AddConstraint(eq(q_dot[i],dq_ds[i]*S_dot[i]))
        prog.AddBoundingBoxConstraint(q_dot_min, q_dot_max, q_dot[i])


    # q_ddot constraint
    for i in range(T):
        q_ddot_i = d2q_ds2[i]*S_dot[i]**2 + dq_ds[i]*S_ddot[i]
        prog.AddConstraint(eq(q_ddot_i, q_ddot[i]))
        prog.AddBoundingBoxConstraint(q_ddot_min - q_ddot_low,
                                      q_ddot_max - q_ddot_upp, q_ddot[i])

    # initial and final constraint
    prog.AddConstraint(eq(q_dot[0], 0.0))
    prog.AddConstraint(eq(q_dot[T], 0.0))

    # tau constraint
    for i in range(T):
        prog.AddBoundingBoxConstraint(tau_min, tau_max, tau[i])

    # dynamics constraints
    for t in range(T):
        vars = np.concatenate((q[t+1], q_dot[t+1], q_ddot[t], f1[t],
                               f2[t], f3[t], f4[t], tau[t]))
        prog.AddConstraint(manipulator_equations, lb=[-1e-6]*7, ub=[1e-6]*7, vars=vars)



    lower_limit = 0# 1e-2

    # friction cone constraints
    for t in range(T):
        prog.AddConstraint(f1[t, 2] >= lower_limit)
        prog.AddConstraint(f2[t, 2] >= lower_limit)
        prog.AddConstraint(f3[t, 2] >= lower_limit)
        prog.AddConstraint(f4[t, 2] >= lower_limit)

        prog.AddConstraint(f1[t, 0] +factor<= mu_*f1[t, 2])
        prog.AddConstraint(f2[t, 0] +factor<= mu_*f2[t, 2])
        prog.AddConstraint(f3[t, 0] +factor<= mu_*f3[t, 2])
        prog.AddConstraint(f4[t, 0] +factor<= mu_*f4[t, 2])

        prog.AddConstraint(f1[t, 0] >= factor-mu_*f1[t, 2])
        prog.AddConstraint(f2[t, 0] >= factor-mu_*f2[t, 2])
        prog.AddConstraint(f3[t, 0] >= factor-mu_*f3[t, 2])
        prog.AddConstraint(f4[t, 0] >= factor-mu_*f4[t, 2])


        prog.AddConstraint(f1[t, 1] +factor<= mu_*f1[t, 2])
        prog.AddConstraint(f2[t, 1] +factor<= mu_*f2[t, 2])
        prog.AddConstraint(f3[t, 1] +factor<= mu_*f3[t, 2])
        prog.AddConstraint(f4[t, 1] +factor <= mu_*f4[t, 2])

        prog.AddConstraint(f1[t, 1] >= factor-mu_*f1[t, 2])
        prog.AddConstraint(f2[t, 1] >= factor-mu_*f2[t, 2])
        prog.AddConstraint(f3[t, 1] >= factor-mu_*f3[t, 2])
        prog.AddConstraint(f4[t, 1] >= factor-mu_*f4[t, 2])

        prog.AddConstraint(f1[t, 0] == f3[t, 0])
        prog.AddConstraint(f1[t, 1] == f3[t, 1])
        prog.AddConstraint(f2[t, 0] == f4[t, 0])
        prog.AddConstraint(f2[t, 1] == f4[t, 1])


    # object equations
    for t in range(T):

        vars = np.concatenate((q[t+1], q_dot[t+1], q_ddot[t], f1[t], f2[t],
                               f3[t], f4[t]))
        prog.AddConstraint(object_equations, lb=[-1e-6]*6, ub=[1e-6]*6, vars=vars, description = 'object_constraint'+str(t))

    prog.AddCost(cost_func, S_dot)


if set_initial_guess:
    lst1 = [0]*nq
    lst2 = [0]*nq
    dq_ds_min = np.abs(np.amin(dq_ds, axis = 0))
    dq_ds_max = np.abs(np.amax(dq_ds, axis = 0))
    dq_ds_min_lb = [0]*nq
    dq_ds_max_lb = [0]*nq
    for i in range(nq):
        dq_ds_min_lb[i] = max(dq_ds_min[i], 1e-5)
        dq_ds_max_lb[i] = max(dq_ds_max[i], 1e-5)

        lst1[i] = np.abs(q_dot_max[i]/dq_ds_max_lb[i])
        lst2[i] = np.abs(q_dot_min[i]/dq_ds_min_lb[i])

    s_dot_0 = min(min(lst1), min(lst2))


    initial_guess = np.zeros(prog.num_vars())
    prog.SetDecisionVariableValueInVector(S_dot, [s_dot_0]*(T+1), initial_guess)
    prog.SetDecisionVariableValueInVector(S_ddot, [0]*T, initial_guess)

    prog.SetDecisionVariableValueInVector(q, q_s, initial_guess)
    prog.SetDecisionVariableValueInVector(q_dot, dq_ds*s_dot_0, initial_guess)

    q_ddot_guess = d2q_ds2*s_dot_0**2
    prog.SetDecisionVariableValueInVector(q_ddot,  q_ddot_guess, initial_guess)

    prog.SetDecisionVariableValueInVector(f1, 0.01*np.ones((T, nf)), initial_guess)
    prog.SetDecisionVariableValueInVector(f2, 0.01*np.ones((T, nf)), initial_guess)
    prog.SetDecisionVariableValueInVector(f3, 0.01*np.ones((T, nf)), initial_guess)
    prog.SetDecisionVariableValueInVector(f4, 0.01*np.ones((T, nf)), initial_guess)
