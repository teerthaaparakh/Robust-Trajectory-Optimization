from params import *
import pydrake
from pydrake.all import JacobianWrtVariable
import numpy as np
from project_utils_2 import *

def object_equations(vars):
    q_k = vars[:7]
    q_dot_k = vars[7:14]
    q_ddot_k = vars[14:14+7]
    f1_k = vars[21:21+3]
    f2_k = vars[24:24+3]
    f3_k = vars[27:27+3]
    f4_k = vars[30:30+3]

    auto_diff_plant.SetPositions(auto_diff_context, iiwa, q_k)
    auto_diff_plant.SetVelocities(auto_diff_context, iiwa, q_dot_k)

    slab_frame = auto_diff_plant.GetBodyByName('slab').body_frame()
    slab_orientation = slab_frame.CalcPoseInWorld(auto_diff_context).rotation()
    ground_frame = auto_diff_plant.world_frame()

    Js_V_ABp = auto_diff_plant.CalcJacobianSpatialVelocity(
                    auto_diff_context,
                    JacobianWrtVariable(0),
                    slab_frame,
                    box_com_wrt_slab,
                    ground_frame,
                    ground_frame)
    AsBias_ABp = auto_diff_plant.CalcBiasSpatialAcceleration(
                    auto_diff_context,
                    JacobianWrtVariable(1),
                    slab_frame,
                    box_com_wrt_slab,
                    ground_frame,
                    ground_frame)

    torques = [None]*4

    pi_slab = contact_pts_wrt_slab[0]
    com_slab = box_com_wrt_slab
    direction_world = slab_orientation @ (pi_slab - com_slab)
    torques[0] = np.cross(direction_world, f1_k)

    pi_slab = contact_pts_wrt_slab[1]
    com_slab = box_com_wrt_slab
    direction_world = slab_orientation @ (pi_slab - com_slab)
    torques[1] = np.cross(direction_world, f2_k)

    pi_slab = contact_pts_wrt_slab[2]
    com_slab = box_com_wrt_slab
    direction_world = slab_orientation @ (pi_slab - com_slab)
    torques[2] = np.cross(direction_world, f3_k)

    pi_slab = contact_pts_wrt_slab[3]
    com_slab = box_com_wrt_slab
    direction_world = slab_orientation @ (pi_slab - com_slab)
    torques[3] = np.cross(direction_world, f4_k)

    net_torque = torques[0] + torques[1] + torques[2] + torques[3]
    linear_acc_term = M_obj.dot(Js_V_ABp.dot(q_ddot_k)[3:] + AsBias_ABp.translational())\
            - (f1_k + f2_k + f3_k + f4_k + M_obj.dot(gravity))
    ang_acc_term = I_obj.dot(Js_V_ABp.dot(q_ddot_k)[:3] + AsBias_ABp.rotational())-net_torque
    return np.concatenate(([linear_acc_term, ang_acc_term]), axis = 0)
