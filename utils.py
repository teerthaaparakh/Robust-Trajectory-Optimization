from params import *
import numpy as np
from pydrake.all import PiecewisePolynomial
import pandas as pd
import altair as alt
import plotly.express as px
from manipulation import running_as_notebook
from project_utils_2 import *


def get_contact_points(plant, plant_context):
    pts = []
    contact_results = plant.get_contact_results_output_port().Eval(plant_context)

    slab_frame = plant.GetBodyByName('slab').body_frame()
    X_WS = slab_frame.CalcPoseInWorld(plant_context)

    for i in range(contact_results.num_point_pair_contacts()):
        contact_info = contact_results.point_pair_contact_info(i)
        body_A = contact_info.bodyA_index()
        body_B = contact_info.bodyB_index()
        point = contact_info.contact_point()
        contact_in_slab_frame = X_WS.inverse() @ point
        pts.append(contact_in_slab_frame)
#         print(contact_in_slab_frame)

    return pts

def run_setup(simulator, end_time, speed = 0.5):
    if running_as_notebook:
        simulator.set_target_realtime_rate(speed)
    simulator.AdvanceTo(end_time if running_as_notebook else 0.1)


def get_q(s):
    q = q0_init.copy()
    q[0] = q[0] + s*to_rotate
    return q


def get_dq_ds(q_s):
    dq_ds = np.zeros_like(q_s)
    interval_size = 1/(q_s.shape[0] - 1)
    dq_ds[1:] = (q_s[1:, :] - q_s[:-1, :])/interval_size
    dq_ds[0] = dq_ds[-1]
    return dq_ds


def get_d2q_ds2(dq_ds):
    l = dq_ds.shape[0]
    nq = 7
    interval_size = 1/(dq_ds.shape[0] - 1)
    d2q_ds2 = np.zeros((l-1, nq))
    d2q_ds2 = (dq_ds[1:, :] - dq_ds[:-1, :])/interval_size
    return d2q_ds2


def convert_from_expr(x):
    x_eval = np.zeros_like(x)
    rows = len(x)
    cols = len(x[0])
    for i in range(rows):
        for j in range(cols):
            x_eval[i,j] = x[i,j].Evaluate()

    return x_eval

def get_trajectories(S_dot_val, S_ddot_val, s_dot_new = None, num_steps = 200):
    T_new = num_steps
    s_new = np.linspace(0, 1, T_new+1)
    # S_dot_val = result.GetSolution(S_dot)
    # S_ddot_val = result.GetSolution(S_ddot)

    if (s_dot_new is None):
        S_dot_new = np.interp(s_new, ss, S_dot_val)
    else:
        S_dot_new = np.interp(s_new, ss, s_dot_new)
    S_ddot_new = np.interp(np.linspace(0, 1, T_new), ss[:-1], S_ddot_val)

    t_val = np.zeros(T_new+1)

    for k in range(1, T_new+1):
        t_val[k] = t_val[k-1] + (2*1/T_new)/(S_dot_new[k] + S_dot_new[k-1])

    qs = np.array([get_q(i) for i in s_new])

    #velocity
    qs_dot = np.zeros_like(qs)
    dq_ds = get_dq_ds(qs)
    for i in range(T_new+1):
        qs_dot[i] = dq_ds[i]*S_dot_new[i]

    #acceleration
    qs_ddot = np.zeros((T_new,7))
    d2q_ds2 = get_d2q_ds2(dq_ds)
    for i in range(T_new):
        qs_ddot[i] = d2q_ds2[i]*(S_dot_new[i])**2 + dq_ds[i]*S_ddot_new[i]


    qs_new= np.insert(qs, 0, qs[0,:], axis = 0)
    qdot_new= np.insert(qs_dot, 0, qs_dot[0,:], axis = 0)
    #acceleration
    qddot_new= np.insert(qs_ddot, 0, np.zeros(7), axis = 0)


    ts = np.zeros(T_new+2)
    ts[1:] = 1.0 + np.array(t_val)

    #acceleration
    ts_ddot = np.zeros(T_new+1)
    ts_ddot[1:] = 1.0 + np.array(t_val[:-1])

    q_traj = PiecewisePolynomial.FirstOrderHold(ts, qs_new.T)
    q_dot_traj = PiecewisePolynomial.FirstOrderHold(ts, qdot_new.T)
    q_ddot_traj = PiecewisePolynomial.ZeroOrderHold(ts_ddot, qddot_new.T)

    return q_traj, q_dot_traj, q_ddot_traj, ts


# def plot_fn(q_traj, q_dot_traj, q_ddot_traj):
#     # , _, _ = get_trajectories(result)
#     data = dataframe(q_traj, q_traj.get_segment_times(), ['q1','q2', 'q3', 'q4', 'q5', 'q6', 'q7'])
#     alt.vconcat(alt.Chart(data).mark_line().encode(x='t', y='q1').properties(height=80).show(),)
#
#     data = dataframe(q_dot_traj, q_dot_traj.get_segment_times(), ['q1_dot','q2_dot', 'q3_dot', 'q4_dot', 'q5_dot', 'q6_dot', 'q7_dot'])
#     alt.vconcat(alt.Chart(data).mark_line().encode(x='t', y='q1_dot').properties(height=80).show(),)
#
#     data = dataframe(q_ddot_traj, q_ddot_traj.get_segment_times(), ['q1_ddot','q2_ddot', 'q3_ddot', 'q4_ddot', 'q5_ddot', 'q6_ddot', 'q7_ddot'])
#     alt.vconcat(alt.Chart(data).mark_line().encode(x='t', y='q1_ddot').properties(height=80).show(),)
