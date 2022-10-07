import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import pandas as pd
import seaborn as sns
import random
import os
import os.path
from os import path
import time
import scipy.integrate as integrate
import glob
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from tqdm import trange
from numba import njit

num_cores  = multiprocessing.cpu_count()
plot       = 0
step_space = 0.1
step_time  = 0.001
step_vel   = 0.1
xmax       = 7.0
vmax       = 6.0
tmax       = 10
numx       = int(2*xmax/step_space)
numv       = int(2*vmax/step_vel  )
numt       = int(tmax/step_time   )
list_x     = np.linspace(-xmax,+xmax,num=numx)
list_v     = np.linspace(-vmax,+vmax,num=numv)
list_t     = np.linspace(    0,+tmax,num=numt)
gamma      = 1
diff_coef  = 1
B          = 1
Omega      = 1
x0         = 0
name_fold  =str('dt='+str(step_time)+'_dx='+str(step_space)+'_dv='+str(step_vel)+'_D='+str(diff_coef)+'_gamma='+str(gamma)+'_x0='+str(x0))
data       = {'x':list_x}
df         = pd.DataFrame(data)

if path.exists(name_fold) == True:
    pass
else:
    os.mkdir(name_fold)

files = glob.glob(str(name_fold)+'/*')
for f in files:
    os.remove(f)

def potential_duffing(x,t):
    return -(1/2.)*((x)**2)+(1/16.)*((x)**4)+3+B*x*np.cos(Omega*t)

def derivate_potential(x,t):
    return -(-1*x+(1/4.)*x**3)-B*np.cos(Omega*t)

def initial_density(x,v):
    Q = np.pi/25.0
    return (1/Q)*np.exp(-(25/1.)*((x-x0)**2+v**2))

def M_v(v):
    return (1/np.sqrt(2*np.pi*diff_coef))*np.exp(-(v**2)/ (2*diff_coef))

max_Uprim = np.max([derivate_potential(xx,0) for xx in list_x])*step_time/step_vel
max_vdt   = vmax*step_time/step_space

print()
print('Transport part stability condition     : '+str(max(max_Uprim,max_vdt)<1))
print('Fokker-Planck part stability condition : '+str(step_time < (step_vel**2)/2))
print()

density0             = np.zeros((numx,numv))
density_integrated_v = np.zeros((numt,numx))
normalisation_time   = np.zeros(numt)
for xind in range(numx):
    for vind in range(numv):
        density0[xind][vind] = initial_density(list_x[xind],list_v[vind])

def job(xind):
    densityxv = np.zeros(numv)
    for vind in range(numv):
        if xind == 0 or xind==numx-1 or vind == 0 or vind==numv-1:
            pass
        else:
            L1         = +1+(M_v(list_v[vind])/M_v(list_v[vind-1]))
            L2         = -2-((M_v(list_v[vind+1])+M_v(list_v[vind-1]))/M_v(list_v[vind]))
            L3         = +1+(M_v(list_v[vind])/M_v(list_v[vind+1]))
            liouvilian = (1/(2*(step_vel**2)))*(density0[xind][vind-1]*L1+density0[xind][vind]*L2+density0[xind][vind+1]*L3)
            TRANSP1    = (1/step_space)*((list_v[vind]+np.abs(list_v[vind]))/2)*( density0[xind  ][vind]-density0[xind-1][vind] )
            TRANSP2    = (1/step_space)*((list_v[vind]-np.abs(list_v[vind]))/2)*( density0[xind+1][vind]-density0[xind  ][vind]   )
            TRANSP3    = (1/step_vel)*((derivate_potential(list_x[xind],list_t[tind])+np.abs(derivate_potential(list_x[xind],list_t[tind])))/2)*( density0[xind][vind  ]-density0[xind][vind-1] )
            TRANSP4    = (1/step_vel)*((derivate_potential(list_x[xind],list_t[tind])-np.abs(derivate_potential(list_x[xind],list_t[tind])))/2)*( density0[xind][vind+1]-density0[xind][vind  ] )
            densityxv[vind] = gamma*step_time*liouvilian+density0[xind][vind]-step_time*(TRANSP1+TRANSP2+TRANSP3+TRANSP4)
    return densityxv

for tind in range(numt-1):
    start_time = time.time()
    density1  = np.zeros((numx,numv))
    density1  = Parallel(n_jobs=num_cores)(delayed(job)(xind) for xind in  range(numx))
    density0  = np.array(density1)
    partial   = np.zeros(numx)
    for xind in range(numx):
        partial[xind] = integrate.simps(density1[xind],dx=step_vel)
    normalisation_time[tind] = integrate.simps(partial,dx=step_space)
    elapsed_time             = time.time()-start_time
    for xind in range(numx):
        density_integrated_v[tind][xind] = np.sum(density1[xind])*step_vel
    print('step'+str(tind)+' : %1.6f'%(normalisation_time[tind])+'     T(s) : %1.6f'%(elapsed_time)+'     max : %1.6f'%(max(density_integrated_v[tind])))
    if normalisation_time[tind] > 1.2 or normalisation_time[tind]<0.8:
        break
    df['t=%1.5f'%(tind*step_time)] = density_integrated_v[tind]

df.to_csv(str(name_fold)+'/rho.csv')
