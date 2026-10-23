"""
Created on Oct 22 12:25:58 2026 -script to build Figure 4

@author: C_Gab
"""

from ADO_Experiment import *
from priors import *

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


expan = 1000 # the final time
n = 10 # number of design points
k = 100 # number of likelihood points

print("Initializing experiment")
print("")

exp = Experiment()
exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
aux = np.linspace(0, expan, n) # design points

prior = Prior() 
p = prior.set_prior_exp(expan,n,k) # calcualte the prior based on exponential curves, shown in Figure 1b
exp.set_prior(p) # set the bins priors as the one calculated above 

p_prior = exp.p.copy() 
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()


true = 1-0.9* np.exp(-aux*0.03) #fixed ground truth


trials = 300
ind =  320 # number of runs used to average the MSE vs. trial curve
xs = np.zeros(shape=(trials+1, 2,ind)) # to store each trial number
MSEs = np.zeros(shape=(trials+1,2,ind)) # to store the MSE vs trials values
infoGained = np.zeros(shape=(trials+1,2,ind)) # to store the information gained vs trials values
designs = []
results = []

for jj in range(2): # to perform the simulations first choosing designs according to our method and then randomly
    if jj==0:
        print("simulating ADO experiment:")
        print("")
    else:
        print("simulating random design experiment:")
        print("")
    for ii in range(ind): 
        print("\rrun number:",ii, end='', flush=True)
        exp.reset()
        exp.set_prior_mat(p_prior, mat_prior, invmat_prior)      
        x = [0]
        y = []
        estimated = exp.values*exp.p
        estimated = estimated.sum(axis = 1) # estimated model at the begining
        err = estimated - true
        aux_err = np.mean(err*err) # error at the begining
        MSE = [aux_err]
        info = [exp.infoProgress()]
        for i in range(trials):  
            if jj == 0: 
                aux1 = exp.ADOchoose()  # select the design using our ADO algorithm
                designs.append(aux1)
            else:
                aux1 = exp.RANDchoose() # select a random design
                
            aux2 = random.random() # choose a random number between 1 and 0
            if aux2 <= true[aux1]: # select if the result is 0 or 1 according to the probability given by the ground truth
                exp.update(aux1,1)
            else:
                exp.update(aux1,0)  
            results.append(aux2 <= true[aux1])
            x.append(i+1)
            estimated = exp.values*exp.p
            estimated = estimated.sum(axis = 1)
            # estimated=exp.values[np.argmax(exp.p,axis=1)]
            err = estimated - true
            aux_err = np.mean(err**2)
            MSE.append(aux_err)
            info.append(exp.infoProgress())
        aux1 = exp.ADOchoose()  
        designs.append(aux1)    
        MSEs[:,jj,ii] = MSE
        xs[:,jj,ii] = x
        infoGained[:,jj,ii] = info
### Save data

# np.save('MSE_data.npy', MSEs)
# np.save('infoGained.npy', infoGained)
# np.save('xs.npy',xs)

# MSEs = np.load('MSE_data.npy', allow_pickle=True)
# xs = np.load('xs.npy', allow_pickle=True)


#### Visualization 300 trials : same as fig 4 on paper
colors = ["b", "r"]
stratName = ["NADO","Random"]
# plot MSE vs trial
fig = plt.figure(2)
ax = fig.add_subplot(111)
for i in range(2):
    ax.errorbar(np.arange(trials+1), np.log10(MSEs[:,i].mean(axis = 1)), yerr=0.434* MSEs[:,i].std(axis = 1)/MSEs[:,i].mean(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    ax.set_xlabel('trial', fontsize=16)
    ax.set_ylabel('log(MSE)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    
# optional
target_log_mse = -3.25
mean_nado = np.log10(MSEs[:,0].mean(axis=1))
mean_random = np.log10(MSEs[:,1].mean(axis=1))
trial_nado = np.where((mean_nado <= target_log_mse) & (np.arange(len(mean_nado)) > 25))[0][0]
trial_random = np.where((mean_random <= target_log_mse) & (np.arange(len(mean_random)) > 25))[0][0]
ax.axhline(target_log_mse, xmin=0, xmax=350,color='k', linestyle=':', linewidth=2)
ymin = ax.get_ylim()[0]
ax.vlines(trial_nado, ymin=ymin-1, ymax=target_log_mse,color='k', linestyles=':', linewidth=2)
ax.vlines(trial_random, ymin=ymin-1, ymax=target_log_mse,colors='k', linestyles=':', linewidth=2)

ax.scatter(trial_nado, target_log_mse, color=colors[0], edgecolor='k', s=60, zorder=5)
ax.scatter(trial_random, target_log_mse, color=colors[1], edgecolor='k', s=60, zorder=5)
ax.set_ylim([ymin-0.05,ax.get_ylim()[1]])

trial_star = 175
ax.vlines(trial_star, ymin=-2, ymax=-1.5,color='k', linestyles='--', linewidth=3)
ax.vlines(trial_star, ymin=ymin-1, ymax=-2.8,color='k', linestyles='--', linewidth=3)

ax_inset = fig.add_axes([0.56,0.56,0.29,0.27])

mse_ratio = MSEs[:,0].mean(axis=1) / MSEs[:,1].mean(axis=1)

ax_inset.plot(mse_ratio, color='k')
ratio_star = mse_ratio[trial_star]
ax_inset.plot(trial_star, ratio_star, marker='s', markersize=0, color='m', linestyle='None', label=f'{ratio_star:.2f}')
ax_inset.vlines(trial_star,ymin=0.25,ymax=1.25,linestyle='--',color='k')
# ax.plot(trial_star, -4, marker='*', markersize=10, color='m', linestyle='None', label=f'{ratio_star:.2f}')

ax_inset.set_xlabel("trial", fontsize=12)
ax_inset.set_ylabel(r"$MSE_{NADO} / MSE_{Ran}$",fontsize=11)

ax_inset.set_xticks([0,150,300])
ax_inset.set_yticks([0.35,0.7,1.05])
ax_inset.set_ylim([0.25,1.25])

ax_inset.tick_params(axis='both', labelsize=10)

    
# plot information gain vs trial    
plt.figure(20)
for i in range(2):
    plt.errorbar(np.arange(trials+1), infoGained[:,i].mean(axis = 1), yerr=infoGained[:,i].std(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('trial', fontsize=16)
    plt.ylabel('info gained', fontsize=16)
    plt.legend(stratName,fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
