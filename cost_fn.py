from params import *

def cost_func(S_dot_var):
        cost = 0
        for i in range(T):
            cost = cost + 2*delta_s*1/(S_dot_var[i] + S_dot_var[i+1])
        return cost
