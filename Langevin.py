import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
import pandas as pd
import seaborn as sns
import random as rd
import os
import time
import scipy.integrate as integrate
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool
from tqdm import trange
from sklearn.neighbors import KernelDensity
import imageio
import glob
sns.set()
files = glob.glob('output/*')
for f in files:
    os.remove(f)

def potential_duffing(x,t):
    return -(1/2.)*((x)**2)+(1/16.)*((x)**4)+3#+B*x*np.cos(Omega*t)

def force(x,t):
    return -(-1*x+(1/4.)*x**3)#-B*np.cos(Omega*t)

def initial_density(x,v):
    Q = np.pi/25.0
    return (1/Q)*np.exp(-(25/1.)*((x)**2+v**2))

def kde_sklearn(x,x_grid,bandwidth,**kwargs):
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:,np.newaxis])
    log_pdf = kde_skl.score_samples(x_grid[:,np.newaxis])
    return np.exp(log_pdf)

plot                 = 0
num_cores            = multiprocessing.cpu_count()
B                    = 1
Omega                = np.pi
step_space           = 0.1
step_time            = 0.001#0.01
gamma                = 1
diff_coef            = 1
xmax                 = 7.0
tmax                 = 10
numx                 = int(2*xmax/step_space)
numt                 = int(tmax/step_time   )
list_x               = np.linspace(-xmax,+xmax,num=numx)
list_t               = np.linspace(0    ,+tmax,num=numt)
total_data           = 50*numx
histogram_data       = np.zeros((numt,total_data))
vinit                = np.random.normal(0,1/np.sqrt(50),total_data)
xinit                = np.random.normal(0,1/np.sqrt(50),total_data)
data                 = {'x':list_x}
df                   = pd.DataFrame(data)
def job(cond):
    histogram_data_t     = np.zeros(numt)
    initial_condition_v  = vinit[cond]
    initial_condition_x  = xinit[cond]
    velocity_particle    = np.zeros(numt)
    position_particle    = np.zeros(numt)
    velocity_particle[0] = initial_condition_v
    position_particle[0] = initial_condition_x
    for tind in range(numt-1):
        Z = rd.random()*2-1
        velocity_particle[tind+1]  = velocity_particle[tind]*(1-step_time*gamma)+force(position_particle[tind],list_t[tind])*step_time+np.sqrt(2*diff_coef*step_time)*Z
        position_particle[tind+1]  = position_particle[tind]+step_time*velocity_particle[tind+1]
        histogram_data_t[tind] = position_particle[tind]
    return histogram_data_t

histogram_data = np.zeros((numt,total_data))
parallel_job   = Parallel(n_jobs=num_cores)(delayed(job)(g) for g in trange(total_data))
for cc in range(total_data):
    for tt in range(numt-1):
         histogram_data[tt][cc] = parallel_job[cc][tt]

for tind in range(numt-1):
    rho = kde_sklearn(histogram_data[tind],list_x,0.05)
    df['t=%1.5f'%(tind*step_time)] = rho
'''
def job2(tind):
    rho = kde_sklearn(histogram_data[tind],list_x,0.05)
    df['t=%1.5f'%(tind*step_time)] = rho
    if plot == 1:
        plt.plot(list_x,rho)
        plt.plot(list_x,[potential_duffing(list_x[xx],list_t[tind]) for xx in range(numx)])
        plt.hist(histogram_data[tind],100,range=(-xmax,+xmax),density=True)
        plt.ylim(-1,3)
        plt.savefig('output/'+str(tind)+'.png')
        plt.clf()
    else:
        pass
'''
df.to_csv('output/rho.csv')

#parallel_job   = Parallel(n_jobs=num_cores)(delayed(job2)(g) for g in trange(numt-1))

if plot == 1:
    filenames = []
    for i in range(numt-1):
        filenames.append('output/'+str(i)+'.png')
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('output/movie.gif', images)
else:
    pass
