# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:29:55 2020

@author: Peilin Yang
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rouwen import rouwen 

# * represent the parameters will be input in form of tuple: NO NEED to first define var name
# ** is dictionary 

# Define it as State Variables

def modeldefs(Xm, Xn, Z, *mparams):
    # unpack sets
    k = Xm                            # k(t-1)
    kp = Xn                           # k(t)
    z = Z                             # z(t)
    
    # find variables
    y = k**alpha*np.exp((1-alpha)*z)  # y(t)
    r = alpha*y/k                     # r(t)
    w = (1-alpha)*y                   # w(t)
    c = w + (1+r-delta)*k - kp        # c(t)
    i = y - c                         # i(t)
    
    # we need to judge if c>0, but it is a form of log
    
    if gamma == 1.0:                  # u(t)
        u = np.log(c)
    else:
        u = (c**(1-gamma)-1)/(1-gamma)
    if np.isnan(u) or np.isinf(u):
        u = -1.0E+99
    return y, r, w, c, i, u

alpha = .33
beta = .95
gamma = 2.5
delta = .08
rho = .9
sigma = .02
mparams = (alpha, beta, gamma, delta, rho, sigma)

# find the steady state
rbar = 1/beta + delta - 1
kbar = (alpha/rbar)**(1/(1-alpha))
print(kbar)

#We can get the steady states for all the other variables using our definintions function

zbar = 0
ybar, rbar, wbar, cbar, ibar, ubar = modeldefs(kbar, kbar, zbar, *mparams)
print('ybar: ', ybar)
print('rbar: ', rbar)
print('wbar: ', wbar)
print('cbar: ', cbar)
print('ibar: ', ibar)
print('ubar: ', ubar)

#--------------------------------------------------------------------------------------------------------------------------------------------

#Value Function Iteration

# set up grid for k
keps = .01

# Define State Space
klow = keps*kbar
khigh = (2-keps)*kbar
knpts = 31
kgrid = np.linspace(klow, khigh, num = knpts)

# set up Markov approximation of AR(1) process using Rouwenhorst method
spread = 2.  # number of standard deviations above and below 0
znpts = 11
zstep = 4.*spread*sigma/(znpts-1)
# Markov transition probabilities, current z in cols, next z in rows
Pimat, zgrid = rouwen(rho, 0., zstep, znpts)

VF = np.zeros((knpts, znpts))
VFnew = np.zeros((knpts, znpts))
PF = np.zeros((knpts, znpts))

ccrit = 1.0E-4
maxit = 1000
damp = 1.
dist = 1.0E+99
iters = 0

while (dist > ccrit) and (iters < maxit):
    # Set all VF 0
    VFnew.fill(0.0)
    iters += 1
    for i in range (0, knpts):
        for j in range(0, znpts):
            maxval = -1.0E+98
            for m in range(0, knpts):
                # get current period utility
                yout, rat, wag, con, inv, u =  \
                    modeldefs(kgrid[i], kgrid[m], zgrid[j], *mparams)
                # get expected value
                val = 0.
                for n in range (0, znpts):
                    # sum over all possible value of z(t+1) with Markov probs
                    val = val + u + beta*Pimat[n, j]*VF[m, n]
                    # if this exceeds previous maximum do replacements
                if val > maxval:
                    maxval = val
                    VFnew[i, j] = val
                    PF[i, j] = kgrid[m]
    # dist = np.mean(np.abs(VF - VFnew))
    # amax: the max value of the array
    dist = np.amax(np.abs(VF - VFnew))
    print('iteration: ', iters, 'distance: ', dist)
    VF = damp*VFnew + (1-damp)*VF

print('Converged after', iters, 'iterations')
print('Policy function at (', int((knpts-1)/2), ',', int((znpts-1)/2), ') should be', \
    kgrid[int((knpts-1)/2)], 'and is', PF[int((knpts-1)/2), int((znpts-1)/2)])

#----------------------------------------------------------------------------------------------
# Interploation the VF
# GET THE POLICY FUNCTION

# fit a polynomial
# create meshgrid
zmesh, kmesh = np.meshgrid(zgrid, kgrid)

Y = PF.flatten()

X = np.ones(knpts*znpts)

temp = kmesh.flatten()
X = np.vstack((X,temp))

temp = kmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh**3
temp = temp.flatten()
X = np.vstack((X,temp))

temp = zmesh.flatten()
X = np.vstack((X,temp))

temp = zmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

temp = zmesh**3
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh*zmesh
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh**2*zmesh
temp = temp.flatten()
X = np.vstack((X,temp))

temp = kmesh*zmesh**2
temp = temp.flatten()
X = np.vstack((X,temp))

XtX = np.dot(X,np.transpose(X))
XtY = np.dot(X,Y)
coeffs = np.dot(np.linalg.inv(XtX),XtY)

PFpoly = np.zeros((knpts, znpts))
for i in range(0, knpts):
    for j in range(0, znpts):
        PFpoly[i,j] = np.dot(np.stack((1, kgrid[i], kgrid[i]**2, kgrid[i]**3, \
        zgrid[j], zgrid[j]**2, zgrid[j]**3, \
        kgrid[i]*zgrid[j], kgrid[i]**2*zgrid[j], kgrid[i]*zgrid[j]**2)),coeffs)
        
# calcuate R-squared
Rsq = 1 - np.sum((PF-PFpoly)**2)/np.sum(PF**2)
print('R-squared', Rsq)

# check polynomial
check = np.dot(np.stack((1, kbar, kbar**2, kbar**3, \
        zbar, zbar**2, zbar**3, kbar*zbar, \
        kbar**2*zbar, kbar*zbar**2)),coeffs)
print('Value of Phi(kbar) from polynomial: ', check)
print('kbar: ', kbar)
print('Percent difference: ', check/kbar - 1.)
diff = check - kbar
print('Absolute diffence: ', diff)


# perform simulation
T = 1000  # number of periods to simulate
kstart = kbar # starting value for simulation
# initialize variable histories
epshist = np.random.randn(T)*sigma
khist= np.zeros(T+1)
zhist = np.zeros(T+1)
yhist = np.zeros(T)
rhist= np.zeros(T)
whist = np.zeros(T)
chist = np.zeros(T)
ihist = np.zeros(T)
uhist = np.zeros(T)
khist[0] = kstart
for t in range(0, T):
    # perform simulation with polynomial fit
    khist[t+1] = np.dot(np.stack((1, khist[t], khist[t]**2, khist[t]**3, \
        zhist[t], zhist[t]**2, zhist[t]**3, khist[t]*zhist[t], \
        khist[t]**2*zhist[t], khist[t]*zhist[t]**2)),coeffs)
    zhist[t+1] = zhist[t]*rho + epshist[t]   
    yhist[t], rhist[t], whist[t], chist[t], ihist[t], uhist[t] = \
        modeldefs(khist[t], khist[t+1], zhist[t], *mparams)   
# remove final k & z
khist = khist[0:T]
zhist = zhist[0:T]

#-----------------------------------------------------------------------------------------------
#Simulate the Economy

# perform simulation
T = 1000  # number of periods to simulate
kstart = kbar # starting value for simulation
# initialize variable histories
epshist = np.random.randn(T)*sigma
khist= np.zeros(T+1)
zhist = np.zeros(T+1)
yhist = np.zeros(T)
rhist= np.zeros(T)
whist = np.zeros(T)
chist = np.zeros(T)
ihist = np.zeros(T)
uhist = np.zeros(T)
khist[0] = kstart
for t in range(0, T):
    # perform simulation with polynomial fit
    # z(t) is given 
    # according to the policy function we know k(t+1)
    
    khist[t+1] = np.dot(np.stack((1, khist[t], khist[t]**2, khist[t]**3, \
        zhist[t], zhist[t]**2, zhist[t]**3, khist[t]*zhist[t], \
        khist[t]**2*zhist[t], khist[t]*zhist[t]**2)),coeffs)
    zhist[t+1] = zhist[t]*rho + epshist[t]   
    yhist[t], rhist[t], whist[t], chist[t], ihist[t], uhist[t] = \
        modeldefs(khist[t], khist[t+1], zhist[t], *mparams)   
# remove final k & z
khist = khist[0:T]
zhist = zhist[0:T]

plt.figure()
# plot data
t = range(0, T)
plt.plot(t, khist, label='k')
plt.plot(t, zhist, label='z')
plt.plot(t, yhist, label='y')
plt.plot(t, rhist, label='r')
plt.plot(t, whist, label='w')
plt.plot(t, chist, label='c')
plt.plot(t, ihist, label='i')
plt.plot(t, uhist, label='u')
plt.xlabel('time')
plt.legend(loc=9, ncol=4, bbox_to_anchor=(0., 1.02, 1., .102))
plt.show()

# plot value function and transition function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, VF)
ax.view_init(30, 150)
plt.title('Value Function')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PF)
ax.view_init(30, 150)
plt.title('Transition Function Grid')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(kmesh, zmesh, PFpoly)
ax.view_init(30, 150)
plt.title('Transition Function Polynomial')
plt.xlabel('k(t)')
plt.ylabel('z(t)')
plt.show()