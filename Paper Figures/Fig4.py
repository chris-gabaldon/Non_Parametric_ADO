# -*- coding: utf-8 -*-

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

true = 1 - 0.9* np.exp(-aux*0.005) #ground truth

trials = 3000
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
        print("run number:",ii)
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
 
#### Visualization
colors = ["green", "blue"]
stratName = ["ADO","Random"]
# plot MSE vs trial
plt.figure(2)
for i in range(2):
    plt.errorbar(xs[:,i].mean(axis = 1), np.log10(MSEs[:,i].mean(axis = 1)), yerr=0.434* MSEs[:,i].std(axis = 1)/MSEs[:,i].mean(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.legend(stratName)
    
# plot information gain vs trial    
plt.figure(20)
for i in range(2):
    plt.errorbar(xs[:,i].mean(axis = 1), infoGained[:,i].mean(axis = 1), yerr=infoGained[:,i].std(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('InfoGained', fontsize=14)
    plt.legend(stratName)
