# -*- coding: utf-8 -*-
from ADO_Experiment import *
from priors import *

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

list_trials=[30,300,3000]
for trials in list_trials:
    expan = 1000 # the final time
    n = 10 # number of design points
    k = 100 # number of likelihood points
    
    print("Initializing experiment")
    print("")
    
    exp = Experiment()
    exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
    aux =  np.array(range(50,1000,100)) # design points
    
    prior = Prior() 
    p = prior.set_prior_exp(expan,n,k) # calcualte the prior based on exponential curves, shown in Figure 1b
    exp.set_prior(p) # set the bins priors as the one calculated above 
    
    p_prior = exp.p.copy() 
    mat_prior = exp.mat.copy()
    invmat_prior = exp.invmat.copy()
    
    true = 1 - 0.9* np.exp(-aux*0.005) #ground truth
    
    
    ind =  1 # number of runs used to average the MSE vs. trial curve
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
                err = estimated - true
                aux_err = np.mean(err**2)
                MSE.append(aux_err)
                info.append(exp.infoProgress())
            aux1 = exp.ADOchoose()  
            designs.append(aux1)    
            MSEs[:,jj,ii] = MSE
            xs[:,jj,ii] = x
            infoGained[:,jj,ii] = info
     
    # plot posterior and ground truth
    plt.figure(4) 
    plot = exp.p.T
    plot[plot<1e-10] = 1e-10
    plot = np.log10(plot)
    plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
    plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
    plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
    plt.xticks(aux)
    plt.xlabel('Design (time, offer, etc.) (a.u.)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'{trials} trials')
    plt.show()
    plt.title(f'{trials} trials')
    vals, counts = np.unique(aux[designs], return_counts=True)
    plt.bar(vals, counts, width=75, align='center',edgecolor='k',color='blue')  # ajustá el width si hace falta
    plt.xticks(vals)
    plt.xlabel('Design')
    plt.ylabel('Ocurrences')
    plt.show()
