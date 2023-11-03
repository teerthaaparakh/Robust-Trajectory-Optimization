from params import *
from initialize import *
import numpy as np



def debug_friction_value(lin_acc_val, q_ddot_val, q_dot_val, total_force_val, q_val):
    lin_acc_val = result.GetSolution(lin_acc)

    q_ddot_val = result.GetSolution(q_ddot)
    q_dot_val = result.GetSolution(q_dot)
    total_force_val = result.GetSolution(total_force)
    q_val = result.GetSolution(q)
    q_val[:,0]+=np.pi

    theta_cap = 0.3995*q_ddot_val[:,0]*0.05
    r_cap = -0.3995*q_dot_val[1:,0]**2*0.05
    f_x = r_cap*np.cos(q_val[1:,0])-theta_cap*np.sin(q_val[1:,0])
    f_y = r_cap*np.sin(q_val[1:,0])+theta_cap*np.cos(q_val[1:,0])

    print(f_x - total_force_val[:,0], f_y - total_force_val[:,1])


def debug_friction_norm(f1, f2, f3, f4, mu = None):
    if isinstance(mu, type(None)):
        mu = mu

    # total_force_val = result.GetSolution(total_force)
    f1 = result.GetSolution(f1)
    f2 = result.GetSolution(f2)
    f3 = result.GetSolution(f3)
    f4 = result.GetSolution(f4)

    error_f1 = (mu*f1[:,2])**2 - (f1[:,0]**2 + f1[:,1]**2) + 1e-6
    error_f2 = (mu*f2[:,2])**2 - (f2[:,0]**2 + f2[:,1]**2) + 1e-6
    error_f3 = (mu*f3[:,2])**2 - (f3[:,0]**2 + f3[:,1]**2) + 1e-6
    error_f4 = (mu*f4[:,2])**2 - (f4[:,0]**2 + f4[:,1]**2) + 1e-6

    a1 = error_f1 < (2*factor**2+2*factor*f1[:,0]+2*f1[:,1]*factor)
    a2 = error_f2 < (2*factor**2+2*factor*f2[:,0]+2*f2[:,1]*factor)
    a3 = error_f3 < (2*factor**2+2*factor*f3[:,0]+2*f3[:,1]*factor)
    a4 = error_f4 < (2*factor**2+2*factor*f4[:,0]+2*f4[:,1]*factor)

    if(sum(a1)+sum(a2)+sum(a3)+sum(a4))>0:
        return False
    else:
        return True
