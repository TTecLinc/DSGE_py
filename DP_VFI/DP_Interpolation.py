# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 08:19:26 2020

@author: Peilin Yang
"""

import numpy as np
import scipy.optimize as opt
import scipy.interpolate as intpl
import matplotlib.pyplot as plt
from time import sleep

# Set up the parameters
beta = 0.9
gamma = 2.2
W_min = 0.1
W_max = 10.0
W_size = 30
W_vec = np.linspace(W_min, W_max, W_size)
V_t = np.log(W_vec)

def util_CRRA(W, W_pr, gamma):
    # Define CRRA utility function
    c = W - W_pr
    util = (c ** (1 - gamma) - 1) / (1 - gamma)
    
    return util

def neg_V(W_pr, *args):
    W_init, util, V_t_interp, gamma, beta = args
    Vtp1 = util(W_init, W_pr, gamma) + beta * V_t_interp(W_pr)
    neg_Vtp1 = -Vtp1
    
    return neg_Vtp1

# Do the interpolation outside the minimization because it
# doesn't change for different W' cake sizes
V_t_interp = intpl.interp1d(W_vec, V_t, kind='cubic',
                            fill_value='extrapolate')

W_vec_lots = np.linspace(0.5* W_min, 1.5 * W_max, 1000)
plt.plot(W_vec_lots, V_t_interp(W_vec_lots), label='interpolated func.')
plt.scatter(W_vec, V_t, color='red', label='discrete values')
plt.title('Interpolated function for discrete $V_t(W)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Value function $V_t(W)$')
plt.legend(loc='center right')


#--------------------------------------------------------------------------------
# Solve： VFI

V_init = np.log(W_vec)
V_new = V_init.copy()

VF_iter = 0
VF_dist = 10
VF_maxiter = 30
VF_mindist = 1e-3

while (VF_iter < VF_maxiter) and (VF_dist > VF_mindist):
    # sleep(0.3)
    VF_iter += 1
    V_init = V_new.copy()
    V_new = np.zeros(W_size)
    psi_vec = np.zeros(W_size)
    
    V_init_interp = intpl.interp1d(W_vec, V_init, kind='cubic',
                                   fill_value='extrapolate')

    for W_ind in range(W_size):
        W_init = W_vec[W_ind]
        V_args = (W_init, util_CRRA, V_init_interp, gamma, beta)
        results_all = opt.minimize_scalar(neg_V, bounds=(1e-10, W_init - 1e-10),
                                          args=V_args, method='bounded')
        V_new[W_ind] = -results_all.fun
        psi_vec[W_ind] = results_all.x
    
    VF_dist = ((V_init - V_new) ** 2).sum()
    print('VF_iter=', VF_iter, ', VF_dist=', VF_dist)
    
# Plot the resulting policy function
plt.figure()
plt.plot(W_vec, psi_vec, label='Policy function $W^{pr}=\psi(W)$')
plt.plot(W_vec, W_vec, color='black', linewidth=1, linestyle='--',
         label='45-degree line')
plt.title('Cake size tomorrow policy function $\psi(W)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Policy function $W^{pr}=\psi(W)$')
plt.legend()

# Plot the equilibrium value function
plt.figure()
plt.plot(W_vec, V_new)
plt.title('Equilibrium value function $V(W)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Value function $V(W)$')

#------------------------------------------------------------------------------------------
# Stochastic Utility

eps_vec = np.array([-1.40, -0.55, 0.0, 0.55, 1.4])
eps_prob = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
eps_size = eps_vec.shape[0]

def neg_V_iid(W_pr, *args):
    W, eps, util, Exp_V_t_interp, gamma, beta = args
    Vtp1 = np.exp(eps) * util(W, W_pr, gamma) + beta * Exp_V_t_interp(W_pr)
    neg_Vtp1 = -Vtp1
    
    return neg_Vtp1
#------------------------------------------------------------------------------------------
# Solve E DP

V_init = np.zeros((W_size, eps_size))
V_new = V_init.copy()

VF_iter = 0
VF_dist = 10
VF_maxiter = 30
VF_mindist = 1e-3

while (VF_iter < VF_maxiter) and (VF_dist > VF_mindist):
    VF_iter += 1
    V_init = V_new.copy()
    V_new = np.zeros((W_size, eps_size))
    psi_mat = np.zeros((W_size, eps_size))
    
    # Integrate out eps_pr from V_init
    Exp_V = V_init @ eps_prob.reshape((eps_size, 1))
    
    # Interpolate expected value function
    Exp_V_interp = intpl.interp1d(W_vec, Exp_V.flatten(), kind='cubic',
                                   fill_value='extrapolate')

    for eps_ind in range(eps_size):
        for W_ind in range(W_size):
            W = W_vec[W_ind]
            eps = eps_vec[eps_ind]
            V_args = (W, eps, util_CRRA, Exp_V_interp, gamma, beta)
            results1 = opt.minimize_scalar(neg_V_iid, bounds=(1e-10, W - 1e-10),
                                           args=V_args, method='bounded')
            V_new[W_ind, eps_ind] = -results1.fun
            psi_mat[W_ind, eps_ind] = results1.x
    
    VF_dist = ((V_init - V_new) ** 2).sum()
    print('VF_iter=', VF_iter, ', VF_dist=', VF_dist)
    
V_1 = V_new.copy()
psi_1 = psi_mat.copy()

plt.figure()
# Plot the resulting policy function
plt.plot(W_vec, psi_1[:, 0], label='$\epsilon_1$')
plt.plot(W_vec, psi_1[:, 1], label='$\epsilon_2$')
plt.plot(W_vec, psi_1[:, 2], label='$\epsilon_3$')
plt.plot(W_vec, psi_1[:, 3], label='$\epsilon_4$')
plt.plot(W_vec, psi_1[:, 4], label='$\epsilon_5$')
plt.plot(W_vec, W_vec, color='black', linewidth=1, linestyle='--',
         label='45-degree line')
plt.title('Cake size tomorrow policy function' +
          '$W^{pr}=\psi(W,\epsilon)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Policy function $W^{pr}=\psi(W,\epsilon)$')
plt.legend()

plt.figure()

# Plot the resulting value function
plt.plot(W_vec, V_1[:, 0], label='$\epsilon_1$')
plt.plot(W_vec, V_1[:, 1], label='$\epsilon_2$')
plt.plot(W_vec, V_1[:, 2], label='$\epsilon_3$')
plt.plot(W_vec, V_1[:, 3], label='$\epsilon_4$')
plt.plot(W_vec, V_1[:, 4], label='$\epsilon_5$')
plt.title('Value function $V(W,\epsilon)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Value function $V(W,\epsilon)$')
plt.legend()

#---------------------------------------------------------------------------------
# Problem 3

trans_mat = np.array([[0.40, 0.28, 0.18, 0.10, 0.04],
                      [0.20, 0.40, 0.20, 0.13, 0.07],
                      [0.10, 0.20, 0.40, 0.20, 0.10],
                      [0.07, 0.13, 0.20, 0.40, 0.20],
                      [0.04, 0.10, 0.18, 0.28, 0.40]])

V_init = np.zeros((W_size, eps_size))
V_new = V_init.copy()

VF_iter = 0
VF_dist = 10
VF_maxiter = 70
VF_mindist = 1e-3

while (VF_iter < VF_maxiter) and (VF_dist > VF_mindist):
    VF_iter += 1
    V_init = V_new.copy()
    V_new = np.zeros((W_size, eps_size))
    psi_mat = np.zeros((W_size, eps_size))
    
    

    for eps_ind in range(eps_size):
        # Calculate expected value, integrate out epsilon prime
        trans_mat_ind = trans_mat[eps_ind, :]
        Exp_V = V_init @ trans_mat_ind.reshape((eps_size, 1))
        
        # Interpolate expected value function
        Exp_V_interp = intpl.interp1d(W_vec, Exp_V.flatten(), kind='cubic',
                                      fill_value='extrapolate')
        for W_ind in range(W_size):
            W = W_vec[W_ind]
            eps = eps_vec[eps_ind]
            V_args = (W, eps, util_CRRA, Exp_V_interp, gamma, beta)
            results1 = opt.minimize_scalar(neg_V_iid, bounds=(1e-10, W - 1e-10),
                                           args=V_args, method='bounded')
            V_new[W_ind, eps_ind] = -results1.fun
            psi_mat[W_ind, eps_ind] = results1.x
    
    VF_dist = ((V_init - V_new) ** 2).sum()
    print('VF_iter=', VF_iter, ', VF_dist=', VF_dist)
    
V_2 = V_new.copy()
psi_2 = psi_mat.copy()

plt.figure()
# Plot the resulting policy function
plt.plot(W_vec, psi_2[:, 0], label='$\epsilon_1$')
plt.plot(W_vec, psi_2[:, 1], label='$\epsilon_2$')
plt.plot(W_vec, psi_2[:, 2], label='$\epsilon_3$')
plt.plot(W_vec, psi_2[:, 3], label='$\epsilon_4$')
plt.plot(W_vec, psi_2[:, 4], label='$\epsilon_5$')
plt.plot(W_vec, W_vec, color='black', linewidth=1, linestyle='--',
         label='45-degree line')
plt.title('Cake size tomorrow policy function' +
          '$W^{pr}=\psi(W,\epsilon)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Policy function $W^{pr}=\psi(W,\epsilon)$')
plt.legend()
plt.show()
plt.figure()
# Plot the resulting value function
plt.plot(W_vec, V_2[:, 0], label='$\epsilon_1$')
plt.plot(W_vec, V_2[:, 1], label='$\epsilon_2$')
plt.plot(W_vec, V_2[:, 2], label='$\epsilon_3$')
plt.plot(W_vec, V_2[:, 3], label='$\epsilon_4$')
plt.plot(W_vec, V_2[:, 4], label='$\epsilon_5$')
plt.title('Value function $V(W,\epsilon)$')
plt.xlabel('Cake size $W$')
plt.ylabel('Value function $V(W,\epsilon)$')
plt.legend()