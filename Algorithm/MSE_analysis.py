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

true=np.array([1 , 0.6 , 0.3  , 0.1, 0.05 , 0. , 0.0 , 0.0 ,0,0]) # saturated exponential model

true = 1 - true # to obtain an ascending monotony (which will then be reversed)

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


#%% Simulation of the ground truth model with for example non informative prior

list_d_ado=[] # store the designs seleceted to create an histogram


trials=300 # number of trials

for i in range (trials):
    
      
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
with respect to the ground truth curves 
'''


expan = 1000 # the final time
n = 10 # number of design points
k = 100 # number of likelihood points

print("Initializing experiment")
print("")

exp = Experiment()
exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
aux = np.linspace(0, expan, n) # design points


true = 1 - 0.9* np.exp(-aux*0.005) #ground truth

trials = 300
ind =  50 # number of runs used to average the MSE vs. trial curve
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


