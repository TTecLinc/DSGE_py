# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:22:06 2020

@author: Peilin Yang
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
from LinApp_FindSS import LinApp_FindSS
from LinApp_Deriv import LinApp_Deriv
from LinApp_Solve import LinApp_Solve
from LinApp_SSL import LinApp_SSL

def Modeldefs(Xp, X, Y, Z, *params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns explicitly defined
    values for consumption, gdp, wages, real interest rates, and transfers
    
    Inputs are:
        Xp: value of capital in next period
        X: value of capital this period
        Y: value of labor this period
        Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Y: GDP
        w: wage rate
        r: rental rate on capital
        T: transfer payments
        c: consumption
        u: utiity
    '''
    
    # unpack input vectors
    kp = Xp
    k = X
    ell = Y
    z = Z
    
    # find definintion values
    Y = k**alpha*(np.exp(z)*ell)**(1-alpha)
    w = (1-alpha)*Y/ell
    r = alpha*Y/k
    c = (w*ell + (r - delta)*k) + k - kp
    
    if gamma == 1.0:                  # u(t)
        u = np.log(c) - chi*ell**(1+theta)/(1+theta)
    else:
        u = c**(1-gamma)/(1-gamma) - chi*ell**(1+theta)/(1+theta)
    if c<0 or np.isnan(u) or np.isinf(u):
        u = -1.0E+99
    return Y, w, r, c, u

def Modeldyn(theta0, *params):
    '''
    This function takes vectors of endogenous and exogenous state variables
    along with a vector of 'jump' variables and returns values from the
    characterizing Euler equations.
    
    Inputs are:
        theta: a vector containng (Xpp, Xp, X, Yp, Y, Zp, Z) where:
            Xpp: value of capital in two periods
            Xp: value of capital in next period
            X: value of capital this period
            Yp: value of labor in next period
            Y: value of labor this period
            Zp: value of productivity in next period
            Z: value of productivity this period
        params: list of parameter values
    
    Output are:
        Euler: a vector of Euler equations written so that they are zero at the
            steady state values of X, Y & Z.  This is a 2x1 numpy array. 
    '''
    
    # unpack theat0
    (Xpp, Xp, X, Yp, Y, Zp, Z) = theta0
    
    # find definitions for now and next period
    Y, w, r, c, u = Modeldefs(Xp, X, Y, Z, params)
    Yp, wp, rp, cp, up = Modeldefs(Xpp, Xp, Yp, Zp, params)
    
    # Evaluate Euler equations
    ell = Y
    E1 = (c**(-gamma)*w) / (chi*ell**theta) - 1
    E2 = (c**(-gamma)) / (beta*cp**(-gamma)*(1 + rp - delta)) - 1
    
    
    return np.array([E1, E2])

# set parameter values
alpha = .35
beta = .99
gamma = 2.5
delta = .08
chi = 10.
theta = 2.
rho = .9
sigma = .02

# make parameter list to pass to functions
params = (alpha, beta, gamma, delta, chi, theta, rho, sigma)

# set LinApp parameters
Zbar = np.array([0.])
nx = 1
ny = 1
nz = 1
logX = 0
Sylv = 0

# take a guess for steady state values of k and ell
guessXY = np.array([.1, .25])

# find the steady state values using LinApp_FindSS
XYbar = LinApp_FindSS(Modeldyn, params, guessXY, Zbar, nx, ny)
(kbar, ellbar) = XYbar
print ('XYbar: ', XYbar)

# set up steady state input vector
theta0 = np.array([kbar, kbar, kbar, ellbar, ellbar, 0., 0.])

# check SS solution
check = Modeldyn(theta0, params)
print ('check: ', check)
if np.max(np.abs(check)) > 1.E-6:
    print ('Have NOT found steady state')
    
Ybar, wbar, rbar, cbar, ubar = Modeldefs(kbar, kbar, ellbar, 0, params)
print ('Ybar: ', Ybar)
print ('wbar: ', wbar)
print ('rbar: ', rbar)
print ('cbar: ', cbar)
print ('ubar: ', ubar)

# find the derivatives matrices
[AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM] = \
    LinApp_Deriv(Modeldyn, params, theta0, nx, ny, nz, logX)
print('FF: ', FF)
print('GG: ', GG)
print('HH: ', HH)
print('LL: ', LL)
print('MM: ', MM)

# set value for NN    
NN = rho
    
# find the policy and jump function coefficients
PP, QQ, RR, SS = \
    LinApp_Solve(AA,BB,CC,DD,FF,GG,HH,JJ,KK,LL,MM,NN,Zbar,Sylv)
print ('PP:',PP)
print ('QQ', QQ)
print ('RR:',RR)
print ('SS', SS)



#------------------------------------------------------------------------------------------------------
#Simulate
# generate a history of Z's
T = 1000  # number of observations to generate
# initialize Z
zhist = np.zeros((T,1))
# generate a history of epsilon shocks with mean zero and variance sigma-squared
epshist = np.random.randn(T,1)*sigma
# iteratively generate Z's starting with the first value of Z = 0
for t in range (1,T):
    zhist[t,:] = abs(rho*zhist[t-1,:] + epshist[t,:])
# generate a history of k's
kbar2 = np.array([[kbar]])
takelogs = False
khist, temp = LinApp_SSL(kbar2, zhist, kbar2, takelogs, PP, QQ, RR, SS)
# initialize histories of all other variables
yhist = np.zeros(T)
rhist = np.zeros(T)
whist = np.zeros(T)
chist = np.zeros(T)
ihist = np.zeros(T)
uhist = np.zeros(T)
# set initial values noting we started at the steady state
#Ybar, wbar, rbar, cbar, ubar
yhist[0] = Ybar
rhist[0] = rbar
whist[0] = wbar
chist[0] = cbar
uhist[0] = ubar
# compute values for all other time periods
for t in range(1, T-1):
    yhist[t],  whist[t],rhist[t], chist[t], uhist[t] = \
        Modeldefs(khist[t-1], khist[t], zhist[t,:], *params)

# plot data
t = range(0, T)
plt.plot(t, khist, label='k')
plt.plot(t, zhist, label='z')
plt.plot(t, yhist, label='y')
plt.plot(t, rhist, label='r')
plt.plot(t, whist, label='w')
plt.plot(t, chist, label='c')
#plt.plot(t, uhist, label='u')
plt.xlabel('time')
plt.legend(loc=9, ncol=4, bbox_to_anchor=(0., 1.02, 1., .102))
plt.show()