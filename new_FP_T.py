import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import os
import time
import scipy.integrate as integrate
start_time = time.time()
sns.set()

step_space = 0.025
step_time  = 0.001
step_vel   = np.sqrt(2*step_time)+0.0001#0.04
print(step_vel)
xmax       = 8.0
vmax       = 6.0
tmax       = 1.0
numx       = int(2*xmax/step_space)
numv       = int(2*vmax/step_vel  )
numt       = int(tmax/step_time   )
list_x     = np.linspace(-xmax,+xmax,num=numx)
list_v     = np.linspace(-vmax,+vmax,num=numv)
list_t     = np.linspace(    0,+tmax,num=numt)

def potential_duffing(x,t):
    if x<-1:
        return (1/2.)*((x+2)**2)
    elif x>1:
        return (1/2.)*((x-2)**2)
    else:
        return 1-(1/2.)*(x**2)
def derivate_potential(x,t):
    if x<-1:
        return (x+2)
    elif x>1:
        return (x-2)
    else:
        return (-x)

def initial_density(x,v):
    Q = np.pi/2.0
    return (1/Q)*np.exp(-(2/1.)*(x**2+v**2))

def M_v(v):
    return (1/np.sqrt(2*np.pi))*np.exp(-(v**2)/2)

max_Uprim = np.max([derivate_potential(xx,0) for xx in list_x])*step_time/step_vel
max_vdt   = vmax*step_time/step_space
print()
print('Transport part stability condition     : '+str(max(max_Uprim,max_vdt)<1))
print('Fokker-Planck part stability condition : '+str(step_time < (step_vel**2)/2))
print()


density_over_time = np.zeros((numt,numx,numv))
for xind in range(numx):
    for vind in range(numv):
        density_over_time[0][xind][vind] = initial_density(list_x[xind],list_v[vind])

density_integrated_v = np.zeros((numt,numx))
normalisation_time   = np.zeros(numt)


for tind in range(numt):
    start_time = time.time()
    density_fixed_x = np.zeros((numx,numv))
    for xind in range(numx):
        for vind in range(numv):
            if vind == 0 or vind==numv-1:
                pass
            else:
                A1 = (1/(2*(step_vel**2)))*(1+(M_v(list_v[vind])/M_v(list_v[vind-1]) ) ) + (1/step_vel)*(derivate_potential(list_x[xind],list_t[tind])-np.abs(derivate_potential(list_x[xind],list_t[tind])))/2
                A2 = (1/(2*(step_vel**2)))*(1+(M_v(list_v[vind])/M_v(list_v[vind+1]) ) ) - (1/step_vel)*(derivate_potential(list_x[xind],list_t[tind])+np.abs(derivate_potential(list_x[xind],list_t[tind])))/2

                density_fixed_x[xind][vind] = density_over_time[tind][xind][vind-1]*step_time*A1+density_over_time[tind][xind][vind+1]*step_time*A2
    density_fixed_v = np.zeros((numv,numx))
    for vind in range(numv):
        for xind in range(numx):
            if xind == 0 or xind==numx-1:
                pass
            else:
                B1 = (1/step_space)*(list_v[vind]-np.abs(list_v[vind]))/2
                B2 = (1/step_space)*(list_v[vind]+np.abs(list_v[vind]))/2
                density_fixed_v[vind][xind] = density_over_time[tind][xind-1][vind]*step_time*B1-density_over_time[tind][xind+1][vind]*step_time*B2
    density_fixed_xv = np.zeros((numx,numv))
    for xind in range(numx):
        for vind in range(numv):
            if xind == 0 or xind==numx-1 or vind == 0 or vind==numv-1:
                pass
            else:
                cache = (1/(2*(step_vel**2)))*(-2- ((M_v(list_v[vind+1])+M_v(list_v[vind-1]))/M_v(list_v[vind])))
                C = 1+ step_time*(cache+(1/step_space)*np.abs(list_v[vind])+(1/step_vel)*np.abs(derivate_potential(list_x[xind],list_t[tind])) )
                density_fixed_xv[xind][vind] = density_over_time[tind][xind][vind]*C
                density_over_time[tind+1][xind][vind] = density_fixed_x[xind][vind]+density_fixed_v[vind][xind]+density_fixed_xv[xind][vind]
    partial = np.zeros(numx)
    for xind in range(numx):
        partial[xind] = integrate.simpson(density_over_time[tind][xind],dx=step_vel)
    normalisation_time[tind] = integrate.simpson(partial,dx=step_space)#np.sum(density_over_time[tind])*step_space*step_vel
    elapsed_time = time.time()-start_time
    for xind in range(numx):
        density_integrated_v[tind][xind] = np.sum(density_over_time[tind][xind])*step_vel
    plt.plot(list_x,density_integrated_v[tind])
    plt.plot(list_x,[potential_duffing(xx,list_t[tind]) for xx in list_x])
    plt.title(str('t=')+str(tind*step_time)+'   norm='+str(normalisation_time[tind]))
    plt.xlim(-5,+5)
    plt.ylim(0,5)
    plt.savefig(str(tind))
    plt.clf()
    print('step'+str(tind)+' : '+str(normalisation_time[tind])+'     T(s) : '+str(elapsed_time))








































#
