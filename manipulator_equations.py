from params import *
import pydrake
from pydrake.all import JacobianWrtVariable
import numpy as np
from project_utils_2 import *



def manipulator_equations(vars):

    q_k = vars[:7]
    q_dot_k = vars[7:14]
    q_ddot_k = vars[14:14+7]

    f1_k = vars[21:21+3]
    f2_k = vars[24:24+3]
    f3_k = vars[27:27+3]
    f4_k = vars[30:30+3]

    tau_k = vars[33:33+7]

    auto_diff_plant.SetPositions(auto_diff_context, iiwa, q_k)
    auto_diff_plant.SetVelocities(auto_diff_context, iiwa, q_dot_k)

    M = auto_diff_plant.CalcMassMatrixViaInverseDynamics(auto_diff_context)
    Cv = auto_diff_plant.CalcBiasTerm(auto_diff_context)
    tauG = auto_diff_plant.CalcGravityGeneralizedForces(auto_diff_context)
    slab_frame = auto_diff_plant.GetBodyByName('slab').body_frame()
    ground_frame = auto_diff_plant.world_frame()

    # compute Jacobian matrix
    Js = [None]*4
    Js_value = [None]*4
    for i in range(4):
        Js[i] = auto_diff_plant.CalcJacobianTranslationalVelocity(
                 auto_diff_context,
                 JacobianWrtVariable(0),
                 slab_frame,
                 contact_pts_wrt_slab[i],
                 ground_frame,
                 ground_frame)

        Js_value[i] = np.array(pydrake.autodiffutils.ExtractValue(auto_diff_plant.CalcJacobianTranslationalVelocity(
                 auto_diff_context,
                 JacobianWrtVariable(0),
                 slab_frame,
                 contact_pts_wrt_slab[i],
                 ground_frame,
                 ground_frame)))

    jacob_fric = Js[0].T.dot(f1_k) + Js[1].T.dot(f2_k) + \
                Js[2].T.dot(f3_k) + Js[3].T.dot(f4_k)
    return (M.dot(q_ddot_k) + Cv - tauG - jacob_fric - tau_k)
