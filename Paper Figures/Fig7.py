# -*- coding: utf-8 -*-
from ADO_Experiment import *
from priors import *

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


expan = 1000 # final time
n = 10 # number of design points
k = 100 # number of likelihood points

print("Initializing experiment")
print("")

exp = Experiment()
exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
aux = np.linspace(0, expan, n) # design points

prior = Prior() 
p = prior.set_prior_gauss(expan,n,k)
exp.set_prior(p) # set the bins priors as the one calculated above 

p_prior = exp.p.copy() 
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()

visualization_trials=[29,299,2999] #trials in wich we plot the heatmap (Figure 7- top panels)

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
        print('\rRealization number: ' + str(ii), end='', flush=True)
        exp.reset()
        exp.set_prior_mat(p_prior, mat_prior, invmat_prior)   
        
        true=exp.getRandomModel2() #generate a ground truth curve enabled by the heatmap discretization
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
            err = estimated - true
            aux_err = np.mean(err**2)
            MSE.append(aux_err)
            info.append(exp.infoProgress())
            
            if ii==0 and i in visualization_trials: #visualization of heatmaps (Figure 7- top panels)
                plt.figure(1) 
                plot = exp.p.T
                plot[plot<1e-10] = 1e-10
                plot = np.log10(plot)
                plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
                plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
                plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
                plt.xlabel('Design (time, offer, etc.) (a.u.)', fontsize=14)
                plt.ylabel('Probability', fontsize=14)
                plt.title(f'{i+1} trials')
                plt.show()
                   
        aux1 = exp.ADOchoose()  
        designs.append(aux1)    
        MSEs[:,jj,ii] = MSE
        xs[:,jj,ii] = x
        infoGained[:,jj,ii] = info
 
#### Visualization
colors = ["blue", "red"]
stratName = ["NADO","Random"]
# plot MSE vs trial
plt.figure(2)
for i in range(2):
    plt.errorbar(xs[:300,i].mean(axis = 1), np.log10(MSEs[:300,i].mean(axis = 1)), yerr=0.434* MSEs[:300,i].std(axis = 1)/MSEs[:300,i].mean(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('trial', fontsize=16)
    plt.ylabel('MSE', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(stratName,fontsize=14)
    
# plot information gain vs trial    
plt.figure(20)
for i in range(2):
    plt.errorbar(xs[:,i].mean(axis = 1), infoGained[:,i].mean(axis = 1), yerr=infoGained[:,i].std(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('trial', fontsize=16)
    plt.ylabel('information gained', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(stratName,fontsize=14)

