"""
Created on Tue Mar 18 14:23:50 2025

@author: C_Gab
"""
#%%  Necessary functions and third-party libraries
import math
import random   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from priors import *
from ADO_Experiment import *

#%% Models: define your ground truth model 

#for example here we propose diferent kind of models: exponential, logistic and linear

k=100 # number of likelihood bins
n=10 # number of design
d=[0,1,2,3,4,5,6,7,8,9] # design points
D=len(d)
expan=d[-1]
# true=np.array([10,9.9,9,8,4,1,0.4,0.1,0.11,0.1])/10
# true=np.array([10,10,9,7,5,3,2,1,0.01,0])/10 # soft logis model
# true=np.array([10,10,9.9,9,5,3,1,0.01,0,0])/10 # saturated logis model
true=np.array([1 , 0.6 , 0.3  , 0.1, 0.05 , 0. , 0.0 , 0.0 ,0,0]) # saturated exponential model
# true=np.linspace(1,0,10) # linear model
# true=np.array([1 , 0.6 , 0.4 , 0.25, 0.15 , 0.05 , 0.02 , 0.0 ,0,0]) # soft exponential model
true = 1 - true # to obtain an ascending monotony (which will then be reversed)
true=np.array([0.1       , 0.48362192, 0.70372631, 0.83001196, 0.90246878,
        0.94404113, 0.96789341, 0.98157873, 0.98943073, 0.99393585])
plt.plot(d,true,'o')


#%%% Heatmaps visualization of the prior


exp=Experiment() # generate an instance of the model
exp.generate(D,k) 

# initial prior and ground truth model
prior = Prior() 
p = prior.set_prior_gauss(expan,D,k) # calculate the prior based on gaussian curves, to use a non informative prior coment this line
exp.set_prior(p)
plot = exp.p.T
plot[plot<1e-10] = 1e-10 # for a better visualization
plot = np.log10(plot) 
plt.imshow(plot, cmap='hot', origin='upper',extent=[d[0], d[-1], 0, 1], aspect="auto")  
plt.scatter(np.linspace(0 + 0.5, D-0.5,D)*expan/D,1 - true ,s = 100,c='blue')
plt.plot(np.linspace(0 + 0.5, D-0.5, D)*expan/D,1- true ,c='blue',label='Ground-Truth')
plt.xlabel('Design',fontsize=14)
plt.ylabel('Probability',fontsize=14)
plt.legend(fontsize=12)
plt.show()


#%% Simulation of the ground truth model

list_d_ado=[] # store the designs seleceted to create an histogram


trials=300 # number of trials

for i in range (trials):
    
    '''
    The first `if` implements a full sweep over all design points initially.
    `n_first_loops` sets how many times this sweep should occur.

    '''
    
    n_first_loops=2
    ff=[range(D)[mm % D] for mm in range(n_first_loops*D)] # a flat list with the design ordered
    if i<n_first_loops*D:
        
        d1=ff[i]
    else:       
        d1=exp.ADOchoose() # if the first sweep finish, the algorithm choose a design
        
    list_d_ado.append(d1)
    aux=np.random.random()

    if aux<=true[d1]: # simulate the response using the ground truth
        y=1
    else:
        y=0
    exp.update(d1,y) # updates the model to generate a new design on the next trial in base of the result 'y'
    if i%20==0:
       plt.title(f'Trials{i}')
       plot = exp.p.T
       plot[plot<1e-10] = 1e-10
       plot = np.log10(plot)
       plt.imshow(plot, cmap='hot', origin='upper',extent=[d[0], d[-1], 0, 1], aspect="auto")  
       plt.scatter(np.linspace(0 + 0.5, D-0.5,D)*expan/D,1- true ,s = 100,c='blue')
       plt.plot(np.linspace(0 + 0.5, D-0.5, D)*expan/D,1- true ,c='blue',label='Ground-Truth')
       plt.xlabel('Design',fontsize=14)
       plt.ylabel('Probability',fontsize=14)
       plt.legend(fontsize=12)
       plt.show()



# now we simulate the same situation but using a random choose

exp=Experiment() # create a new instance of the model to reset the variables
exp.generate(D,k)

list_d_ran=[]
trials=300
for i in range (trials):
    d1=exp.RANDchoose()
    list_d_ran.append(d1)
    aux=np.random.random()

    if aux<=true[d1]:
        y=1
    else:
        y=0
    exp.update(d1,y)



# now we can compare both algorithms based on the design distribution by plotting histograms.
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1]})

# ado algorithm
axs[0].hist(list_d_ado, bins=range(11), color='blue', alpha=0.7, edgecolor='black',align='left')
axs[0].set_ylabel('# of ocurrences')
axs[0].set_title('ADO-algorithm',fontsize=14)
axs[0].set_xticks(d)
# random algorithm
axs[1].hist(list_d_ran, bins=range(11), color='red', alpha=0.7, edgecolor='black',align='left')
axs[1].set_ylabel('# of ocurrences')
axs[1].set_title('RANDOM-algorithm',fontsize=14)
axs[1].set_xticks(d)

plt.xticks(d)
plt.xlabel('Design')
plt.ylabel('# of ocurrences')
plt.legend()

#%% simulations and MSE analysis

'''
In this cell, we perform several simulations of 300 trials each and calculate the MSE 
with respect to the ground truth curves for each of the studied cases: non-informative prior,
Gaussian prior, and random.
'''


exp=Experiment()
repetitions=50 # number of simulations
trials=300

##### ado algorithm with non informative prior


list_totals_mse_ado=[]
list_totals_y=[]
totals_d_ado=[]
exp.generate(n, k)

p_prior = exp.p.copy() 
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()
print('Non informative')

for rep in range(repetitions):
    exp.reset()
    exp.set_prior_mat(p_prior, mat_prior, invmat_prior)    
    if rep % (repetitions/10)==0:
        print(rep)
    list_d_ni=[]
    list_mse_freq_ado=[]
    list_y=[]
    corrects=np.zeros_like(d)
    totals=np.zeros_like(d)

    for i in range (trials):
        # initial sweep
        n_first_loops=2
        ff=[range(D)[mm % D] for mm in range(n_first_loops*D)]
        if i<n_first_loops*D:
            d1=ff[i]
        else:       
            d1=exp.ADOchoose()
        list_d_ni.append(d1)
        aux=np.random.random()
        if aux<=true[d1]: # simulate the response using the ground truth
            y=1
            corrects[d1]+=1
        else:
            y=0
        exp.update(d1,y)
        list_y.append(y)
        totals[d1]+=1
        estimation_ado=corrects/(totals+eps)
        mse_freq=np.mean((estimation_ado-true)**2)
        list_mse_freq_ado.append(mse_freq)
        
    list_totals_mse_ado.append(list_mse_freq_ado)
    totals_d_ado.append(list_d_ni)
    list_totals_y.append(list_y)

##### random algorithm
print ('random')
list_totals_mse_random=[]
exp.generate(n, k)
   
totals_d_random=[]
for repp in range(repetitions):
    exp.reset()
    list_mse_freq_random=[]
    corrects=np.zeros_like(d)
    totals=np.zeros_like(d)
    listas_d_r=[]
    for ii in range (trials):
        n_first_loops=2
        ff=[range(D)[mm % D] for mm in range(n_first_loops*D)]
        if i<n_first_loops*D:
            
            d1=ff[i]
        else:       
            d1=exp.RANDchoose()
        listas_d_r.append(d1)
        aux=np.random.random()
        if aux<=true[d1]:
            y=1
            corrects[d1]+=1
        else:
            y=0
            corrects[d1]+=0
        exp.update(d1,y)
        totals[d1]+=1
        estimation_freq=corrects/(totals+eps)
        mse_freq=np.mean((estimation_freq-true)**2)
        list_mse_freq_random.append(mse_freq)    
    list_totals_mse_random.append(list_mse_freq_random)
    totals_d_random.append(listas_d_r)

mean_ado_ni=np.mean(list_totals_mse_ado,axis=0)
std_ado_ni=np.std(list_totals_mse_ado,axis=0)
mean_random=np.mean(list_totals_mse_random,axis=0)  
std_random=np.std(list_totals_mse_random,axis=0)

# plt.title('Random')
# plt.hist(totals_d_random)
# plt.show()
# plt.title('Ado - Non informative')
# plt.hist(totals_d_ado)
# plt.show()


##### ado algorithm with gaussian prior

print('Gaussian')

exp=Experiment()
exp.generate(D,k)

list_totals_mse_ado_g=[]
list_d_ado=[]
list_totals_y_g=[]

prior = Prior() 
p = prior.set_prior_gauss(expan,n,k) # calcualte the prior based on gaussian curves
exp.set_prior(p) # set the bins priors as the one calculated above 

p_prior = exp.p.copy() 
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()

totals_d_ado_g=[]
for rep in range(repetitions):
    exp.reset()
    exp.set_prior_mat(p_prior, mat_prior, invmat_prior)    
    if rep % (repetitions/10)==0:
        print(rep)
    list_d_g=[]
    list_mse_freq_ado=[]
    list_y_g=[]

    corrects=np.zeros_like(d)
    totals=np.zeros_like(d)

    for i in range (trials):
        n_first_loops=2
        ff=[range(D)[mm % D] for mm in range(n_first_loops*D)]
        if i<n_first_loops*D:
            
            d1=ff[i]
        else:       
            d1=exp.ADOchoose()
        list_d_g.append(d1)
        aux=np.random.random()
        if aux<=true[d1]: # simulate the response using the ground truth
            y=1
            corrects[d1]+=1
        else:
            y=0
        exp.update(d1,y)
        totals[d1]+=1
        list_y_g.append(y)
        estimation_freq=corrects/(totals+eps)
        mse_freq=np.mean((estimation_freq-true)**2)
        list_mse_freq_ado.append(mse_freq)
    list_totals_mse_ado_g.append(list_mse_freq_ado)
    totals_d_ado_g.append(list_d_g)
    list_totals_y_g.append(list_y_g)

mean_ado_g=np.mean(list_totals_mse_ado_g,axis=0)
std_ado_g=np.std(list_totals_mse_ado_g,axis=0)


# plt.title('Ado - Gaussian')
# plt.hist(totals_d_ado_g)
# plt.show()

# the 0.434 factor corresponds to the correct log transformation for the error bars

plt.errorbar(range(trials), np.log10(mean_ado_ni), yerr=0.434* std_ado_ni/(np.sqrt(repetitions)), label='ADO-Non Informative',c='blue')
plt.errorbar(range(trials), np.log10(mean_random), yerr=0.434* std_random/(np.sqrt(repetitions)), label='Random',c='orange')
plt.errorbar(range(trials), np.log10(mean_ado_g), yerr=0.434* std_ado_g/(np.sqrt(repetitions)), label='ADO-Gaussian',c='green')

plt.xlabel('Trial', fontsize=14)
plt.ylabel('log(MSE)', fontsize=14)
plt.legend()
plt.show()

#%%histograms time evolution (set an emergent figure window )

data =totals_d_ado_g[16]

# timelaps
num_intervals = 30
interval_length = len(data)//num_intervals

def update_hist(num, data, bins, patch):
    
    patch[0].remove()  
    patch[0]=plt.title(f'Trials: {num * interval_length}')
    patch[0] = plt.hist(data[:num * interval_length], bins=bins, edgecolor='black',alpha=0.75,color='blue')[2]

fig, ax = plt.subplots()
bins = range(12)
ax.set_xlim(min(data), max(data)+1)

patch = [ax.hist(data[:interval_length], bins=bins, alpha=0.75)[2]]

# Crete the animation
ani = animation.FuncAnimation(fig, update_hist, fargs=(data, bins, patch)
                              ,frames=num_intervals, repeat=False)
ani.save('animacion.mp4', writer='ffmpeg')
plt.show()

