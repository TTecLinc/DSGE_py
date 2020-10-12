# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:32:03 2020

@author: Peilin Yang
"""

import numpy as np
import matplotlib.pyplot as plt

beta=0.95
T=3-1

S=np.array([100,110,121])
C=np.array([60,66,73])



def find_nearest(array, value):

    array = np.asarray(array)

    idx = (np.abs(array - value)).argmin()

    return array[idx]

def u_1(c):
    a=1e-6
    return (1-np.exp(-a*c))/a

def u_2(c):
    a=1e-8
    return (1-np.exp(-a*c))/a

#-----------------------------------------------------------------------------------
# Discrete Space

# Share Space
N_h=50
# Max Asset Share
max_h=N_h
h_space=np.linspace(0,max_h,N_h)
# Asset Price Space

MaxF1=400
MinF1=25
P1_price_space=[0.9,0.1]
P2_price_space=[0.5,0.5]
P1_change_space=[2,1/2]
P2_change_space=[4,1/4]
# Min Space distance is the step length
SizeP1_space=int((MaxF1-MinF1)/25)+1
P1_space=np.linspace(MinF1,MaxF1,SizeP1_space)

MaxF2=80
MinF2=5
SizeP2_space=int((MaxF2-MinF2)/5)+1
P2_space=np.linspace(MinF2,MaxF2,SizeP2_space)


# Insurance State Space
N_BP=3
B_P_space=np.linspace(0,200,N_BP)

V_space=np.zeros((T+1,N_BP,T+1,N_h,N_h,SizeP1_space,SizeP2_space))
#Policy Function
PP1_space=np.zeros((T+1,N_BP,T+1,N_h,N_h,SizeP1_space,SizeP2_space))
PP2_space=np.zeros((T+1,N_BP,T+1,N_h,N_h,SizeP1_space,SizeP2_space))
#------------------------------------------------------------------------------------------------------
# Last term equals to 3
last_term=3-1
for I_p in range(N_BP):
    #Last term value function
    for f1 in range(N_h):
        for f2 in range(N_h):
            # now_price
            for now_price1 in range(SizeP1_space):
                for now_price2 in range(SizeP2_space):
                    # minus term T-1 share
                    D=h_space[f1]*P1_space[now_price1]+h_space[f2]*P2_space[now_price2]+B_P_space[I_p]/0.9
                    
                    V_space[last_term,I_p,T,f1,f2,now_price1,now_price2]=u_2(D)
                    #print(V_space[f1,f2,p1,p2])
                    # Find the position
                    # P2_space[p[0][0]]
                    #p=np.argwhere(P2_space==65)
                    
                    #p=l.index(8)
                    
    #---------------------------------------------------------------------------
    #f1: current share
    for f1 in range(N_h):
        print(f1)
        for f2 in range(N_h):
            # Future Price: p1 p2
            for now_price1 in range(SizeP1_space):
                for now_price2 in range(SizeP2_space):
                    for future_price1 in range(2):
                        for future_price2 in range(2):
                            #Find Max
                            Current_value=-1e10
                            
                            #f1_p: future policy
                            for f1_p in range(N_h):
                                for f2_p in range(N_h):
                                    #Asset future
                                    A_prime=h_space[f1_p]*P1_space[future_price1]+h_space[f2_p]*P2_space[future_price2]
                                    A=h_space[f1]*P1_space[now_price1]+h_space[f2]*P2_space[now_price2]
                                    
                                    # minus term T-1 share
                                    D=S[T-1]-C[T-1]+A-A_prime
                                    
                                    if D<0:
                                        Current_value_find=-np.inf
                                        break
                                    
                                    if P1_space[now_price1]>MaxF1/2 or P1_space[now_price1]<MinF1*2 \
                                            or P2_space[now_price2]>MaxF2/4 or P2_space[now_price2]<MinF2*4:
                                                Current_value_find=-np.inf
                                                break
                                    Future_Price1=P1_space[now_price1]*P1_change_space[future_price1]
                                    Future_Price2=P2_space[now_price2]*P2_change_space[future_price2]
                                    
                                    
                                    #Find the position of Probability
                                    Probability1=np.argwhere(P1_space==Future_Price1)
                                    
                                    Probability2=np.argwhere(P2_space==Future_Price2)
                                    
                                    # If nan in two tree : break
                                    if np.shape(Probability1)[0]==0 or np.shape(Probability2)[0]==0:
                                        Current_value_find=-np.inf
                                        break
                                    Current_value_find=u_1(D)+beta*P1_price_space[future_price1]*P2_price_space[future_price2]*V_space[last_term,I_p,T,f1,f2,Probability1[0][0],Probability2[0][0]]
                                    
                                if Current_value_find>Current_value:
                                    Current_value=Current_value_find
                                    PP1_space[last_term,I_p,T-1,f1,f2,now_price1,now_price2]=h_space[f1_p]
                                    PP2_space[last_term,I_p,T-1,f1,f2,now_price1,now_price2]=h_space[f2_p]
                            V_space[last_term,I_p,T-1,f1,f2,now_price1,now_price2]=Current_value
                            
                            #Inspect the process
                            #print("Asset Policy term 2: ",PP1_space[last_term,I_p,T-1,f1,f2,now_price1,now_price2],PP2_space[last_term,I_p,T-1,f1,f2,now_price1,now_price2])
    
    #---------------------------------------------------------------------------
    # The first term
    for f1 in range(N_h):
        print(f1)
        for f2 in range(N_h):
            # Future Price: p1 p2
            for now_price1 in range(SizeP1_space):
                for now_price2 in range(SizeP2_space):
                    for future_price1 in range(2):
                        for future_price2 in range(2):
                            #Find Max
                            Current_value=-1e10
                            
                            #f1_p: future policy
                            for f1_p in range(N_h):
                                for f2_p in range(N_h):
                                    #Asset future
                                    A_prime=h_space[f1_p]*P1_space[future_price1]+h_space[f2_p]*P2_space[future_price2]
                                    A=h_space[f1]*P1_space[now_price1]+h_space[f2]*P2_space[now_price2]
                                    
                                    # minus term T-1 share
                                    D=S[T-1]-C[T-1]+A-A_prime
                                    
                                    if D<0:
                                        Current_value_find=-np.inf
                                        break
                                    
                                    if P1_space[now_price1]>MaxF1/4 or P1_space[now_price1]<MinF1*4 \
                                            or P2_space[now_price2]>MaxF2/16 or P2_space[now_price2]<MinF2*16:
                                                Current_value_find=-np.inf
                                                break
                                    
                                    Future_Price1=P1_space[now_price1]*P1_change_space[future_price1]
                                    Future_Price2=P2_space[now_price2]*P2_change_space[future_price2]
                                    
                                    
                                    #Find the position of Probability
                                    Probability1=np.argwhere(P1_space==Future_Price1)
                                    
                                    Probability2=np.argwhere(P2_space==Future_Price2)
                                    
                                    Current_value_find=u_1(D)+beta*P1_price_space[future_price1]*P2_price_space[future_price2]*V_space[last_term,I_p,T-1,f1,f2,Probability1[0][0],Probability2[0][0]]
                                    
                                    # If nan in two tree : break
                                    if np.shape(Probability1)[0]==0 or np.shape(Probability2)[0]==0:
                                        Current_value_find=-np.inf
                                        break
                                    
                                if Current_value_find>Current_value:
                                    Current_value=Current_value_find
                                    PP1_space[last_term,I_p,T-2,f1,f2,now_price1,now_price2]=h_space[f1_p]
                                    PP2_space[last_term,I_p,T-2,f1,f2,now_price1,now_price2]=h_space[f2_p]
                            V_space[last_term,I_p,T-2,f1,f2,now_price1,now_price2]=Current_value
                            
                            #Inspect the process
                            #print("Asset Policy term 1: ",PP1_space[last_term,I_p,T-2,f1,f2,now_price1,now_price2],PP2_space[last_term,I_p,T-2,f1,f2,now_price1,now_price2])
#------------------------------------------------------------------------------------------------------
# Last term equals to 2


#Policy Path
Path_asset1=np.zeros(T+1)
Path_asset2=np.zeros(T+1)
Path_asset1[0]=2
Path_asset2[0]=5
Initial_Price1=100
Initial_Price2=20
for I_p in range(N_BP):
    for i in range(2):
        a1=int(Path_asset1[i]-1)
        a2=int(Path_asset2[i]-1)
        now_price1=np.argwhere(P1_space==Initial_Price1)
        now_price2=np.argwhere(P2_space==Initial_Price2)
        for timess in range(2):
            Path_asset1[i+1]=PP1_space[last_term,I_p,i,f1,f2,now_price1,now_price2]
                    
            Path_asset2[i+1]=PP2_space[last_term,I_p,i,f1,f2,now_price1,now_price2]
plt.plot(Path_asset1,label='Asset1')
plt.plot(Path_asset2,label='Asset2')
plt.legend()
