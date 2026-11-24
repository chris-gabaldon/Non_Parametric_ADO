# -*- coding: utf-8 -*-

from ADO_Experiment import *
from priors import *

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

list_trials=[3000]
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
    
    true = 1 - 1.2* np.exp(-aux*0.0065) #ground truth 
    
    
    ind =  1 # number of runs 
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
            y = []

            for i in range(trials):  
                if jj == 0: 
                    aux1 = exp.ADOchoose()
                else:
                    aux1 = exp.RANDchoose()
                    
                designs.append(aux1)  # Guardar en ambos casos (jj=0 y jj=1)
                    
                aux2 = random.random()
                if aux2 <= true[aux1]:
                    exp.update(aux1, 1)
                else:
                    exp.update(aux1, 0)  
                results.append(aux2 <= true[aux1])
    
                if i in [29, 299, 2999] and jj==0:
                    # 1. Plot posterior and ground truth
                    plt.figure() 
                    
                    plot = exp.p.T.copy()  
                    plot[plot < 1e-10] = 1e-10
                    plot = np.log10(plot)
                    
                    plt.imshow(plot, cmap='hot', origin='upper', extent=[0, expan, 0, 1], aspect="auto")
                    x_grid = np.linspace(0.5, n - 0.5, n) * expan / n
                    plt.scatter(x_grid, 1 - true, s=100, c='blue')
                    plt.plot(x_grid, 1 - true, c='blue')
                    plt.xticks(aux)
                    plt.xlabel('Design (time, offer, etc.) (a.u.)', fontsize=14)
                    plt.ylabel('Probability', fontsize=14)
                    plt.title(f'{i + 1} trials')
                    plt.show()
                
                    # 2. Histogram of the selected designs
                    plt.figure()
                    counts = np.bincount(designs, minlength=len(aux))

                    plt.bar(aux, counts, width=75, align='center', edgecolor='k', color='blue')
                    plt.xticks(aux)
                    plt.xlim(aux[0] - 50, aux[-1] + 50) 
                    plt.xlabel('Design')
                    plt.ylabel('Occurrences')
                    plt.title(f'{i + 1} trials')
                    plt.show()
