# -*- coding: utf-8 -*-
      

from ADO_Experiment import *
from priors import *

import random
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats

def get_ground_truth(expan, n, k, curve_type): # this function generate the ground-truth curves from each parametric form, to run the simulations
    aux = np.linspace(0, expan, n)
    if curve_type == "exponential":
        a_exp = np.random.beta(2, 1)
        b_exp = np.random.beta(1, 80)
        true = 1 - a_exp* np.exp(-aux*b_exp) 
        
    elif curve_type == "powerlaw":
        a_exp = np.random.beta(2, 1)
        b_exp = np.random.beta(1, 4)
        true = 1 - a_exp*(aux + 1)**-b_exp

    elif curve_type == "logistic":

        slope = 0.1*random.random()
        center = -100 + 200*random.random()
        true = 1/(1 + np.exp(-slope*(aux-center)))
        
    elif curve_type == "quadratic":
        a=random.random()
        b=random.random()
        true = a + b*(2*aux/expan)**2 
        true[true>1] = 1 

    return true

def MSEvsK(k, trials, ind, expan, curve_type): # trials is the number of trials and ind is the number of times the experiment is run
    
    if curve_type=='logistic': # to be consistent with the prior constructed in priors.py
        expan=300
    else:
        expan = 1000
    exp = Experiment()
    exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
    aux = np.linspace(0, expan, n) # design points
    
    if curve_type=='logistic':
        prior = Prior() 
        p = prior.set_prior_logis(expan,n,k) 
        exp.set_prior(p) # set the bins priors as the one calculated above 
        
    elif curve_type=='powerlaw':
        prior = Prior() 
        p = prior.set_prior_pow(expan,n,k)
        exp.set_prior(p) # set the bins priors as the one calculated above 
       
        
    elif curve_type=='quadratic':
        prior = Prior() 
        p = prior.set_prior_cuadra(expan,n,k)
        exp.set_prior(p) 

        
    elif curve_type=='exponential':
        prior = Prior() 
        p = prior.set_prior_exp(expan,n,k)
        exp.set_prior(p) # set the bins priors as the one calculated above 
        
    elif curve_type=='gaussian':
        prior = Prior() 
        p = prior.set_prior_gauss(expan,n,k)
        exp.set_prior(p) # set the bins priors as the one calculated above 


    print(curve_type)
    print('\nSetting prior using k =', k)


    p_prior = exp.p.copy()
    mat_prior = exp.mat.copy()
    invmat_prior = exp.invmat.copy()
    
    MSEs = np.zeros(shape=(trials,ind))
    designs = np.zeros(shape=(trials,ind))
    results = np.zeros(shape=(trials,ind))
    trueS = np.zeros(shape=(n,ind))
    
    print('\nSimulation using k equal to',k, 'and', ind, 'realizations of', trials, 'trials')
    for ii in range(ind):
        print('\rRealization number: ' + str(ii), end='', flush=True)
        exp.reset()
        exp.set_prior_mat(p_prior, mat_prior, invmat_prior)
        
        if curve_type=='gaussian': #gaussian: in this particular case we sample one of the curves enabled by the chosen discretization
            true=exp.getRandomModel2()
        else:
            true = get_ground_truth(expan, n, k, curve_type)
        
        
        trueS[:,ii] = true # save for later use                          

        MSE = []
        design = []
        result = []

        for i in range(trials):  
            aux1 = exp.ADOchoose()             
            aux2 = random.random()
            
            if aux2 <= true[aux1]:
                exp.update(aux1,1)
            else:
                exp.update(aux1,0)
                
            design.append(aux1)
            result.append(aux2 <= true[aux1])         
            estimated = exp.values*exp.p
            estimated = estimated.sum(axis = 1)
            err = estimated - true
            aux_err = np.mean(err*err)
            MSE.append(aux_err)
        MSEs[:,ii] = MSE
        designs[:,ii] = design
        results[:,ii] = result
            
    ####### run new simulations with k 200 but using a random strategy
    if k==200:
     
        aux = np.linspace(0, expan, n)
        
        exp = Experiment()
        exp.generate(n,k)
    
        MSEs2 = np.zeros(shape=(trials,ind)) #random MSE
    
        print('\nSimulation usigin k = 200 and random strategie')
        for ii in range(ind):
            print('\rRealization number: ' + str(ii), end='', flush=True)
            exp.reset()
            exp.set_prior_mat(p_prior, mat_prior, invmat_prior)
            true = trueS[:,ii]      
    
            MSE = []
        
            for i in range(trials):           
                aux1 = exp.RANDchoose()
                aux2=random.random()
                
                if aux2 <= true[aux1]:
                    exp.update(aux1,1)
                else:
                    exp.update(aux1,0)
                    
                estimated = exp.values*exp.p
                estimated = estimated.sum(axis = 1)
                err = estimated - true
                aux_err = np.mean(err*err)
                MSE.append(aux_err)
                
            MSEs2[:,ii] = MSE   
        
    if k == 200:
        return (MSEs2.mean(axis=1)[-1],
                MSEs2.std(axis=1)[-1] / np.sqrt(ind),
                MSEs.mean(axis=1)[-1],
                MSEs.std(axis=1)[-1] / np.sqrt(ind))
    else:
        # only if k=200 we save the random measuraments
        return ([], [],  
                MSEs.mean(axis=1)[-1],
                MSEs.std(axis=1)[-1] / np.sqrt(ind))
    
ind = 320
trials = 300
n = 10
ks = [2,5,10,20,40,100,200]
prior_types=['gaussian','logistic','powerlaw','quadratic','gaussian','exponential']

for curve_type in prior_types:
    MSEmeans = []
    MSEstds = []
    MSEmeans2 = []
    MSEstds2 = []
    
    print(curve_type)
    expan=1000
    if curve_type== 'logistic':
        expan=300

    for k in ks:
        print('\n\nStarting simulation using k =', k)
    
        simulation = MSEvsK(k, trials, ind, expan,curve_type)
    
        MSEmeans2.append(simulation[0])
        MSEstds2.append(simulation[1])
        MSEmeans.append(simulation[2])
        MSEstds.append(simulation[3])  
    
    # MSE vs k 
    plt.figure(1)
    MSErandom = MSEmeans2[-1]
    MSErandomStd= MSEstds2[-1]
    
    y_random = np.log10(MSErandom)
    y_low = np.log10(MSErandom - MSErandomStd)
    y_high = np.log10(MSErandom + MSErandomStd)
    
    plt.axhline(y=y_random, color='k', linestyle='--', label='Random')
    plt.axhspan(y_low, y_high, color='gray', alpha=0.2, linewidth=0)
      
    plt.title(curve_type)
    
    plt.errorbar(ks[2:],np.log10(MSEmeans)[2:],yerr=0.434 * np.array(MSEstds[2:]) / np.array(MSEmeans[2:]),fmt='-o',label='NADO',color='blue')
    plt.legend(fontsize=14)
    plt.xlabel('number of $K$ points in the likelihood grid',fontsize=14)
    plt.ylabel('log (MSE)',fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    
    plt.show()
    
    # np.save(f'mse_nado_{curve_type}.npy',
    #     {   'list_k': ks,
    #         'mse_nado':MSEmeans,
    #         'error_nado': MSEstds,
    #         'mse_random': MSErandom,
    #         'error_random': MSErandomStd})
    
    # data = np.load('mse_nado_{curve_type}.npy', allow_pickle=True).item()
    
    # ks = data['list_k']
    # MSEmeans = data['mse_nado']
    # MSEstds = data['error_nado']
    # MSErandom = data['mse_random']
    # MSErandomStd = data['error_random']

