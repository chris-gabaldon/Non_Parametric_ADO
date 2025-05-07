# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:28:27 2025

@author: C_Gab
"""

from ADO_Experiment import *
import random
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
from priors import *



list_priors=['exp','pow','logis','cuadra','gauss']
for prior_type in list_priors: 
    print('Prior type=',prior_type)
    
    def generate_trueS(realizations,prior_type):
        print("\nGenerating truths")
        global expan
        global n
        delta = expan/n
        aux = np.array([delta/2+i*delta for i in range(n)])
        trueS = np.zeros(shape=(n,realizations))
        
        for i in range(realizations):
    
            
            if prior_type=='exp':
                a_exp = np.random.beta(2, 1)
                b_exp = np.random.beta(1, 80)
                true = 1 - a_exp* np.exp(-aux*b_exp)
                
            elif prior_type=='pow':
                a_pow = np.random.beta(2, 1)
                b_pow = np.random.beta(1, 4)
                true = 1 - (a_pow*(aux + 1)**-b_pow)
            
            elif prior_type=='logis':
                true = 1/(1 + np.exp(-0.1*random.random()*(aux-400 +  100*random.random()))) #logis
            
            elif prior_type=='cuadra':
                true =   random.random()*(aux/expan)**2 #cuadra
            # true[true>1] = 1
            
            elif prior_type=='gauss':
    
                mu = 0 
                sigma = np.random.uniform(expan/10, expan/5) 
                true = 1-np.exp(-0.5 * ((aux - mu) / sigma) ** 2)  
        # true = true / np.max(true)  # Normalización para garantizar que los valores estén entre 0 y 1
                
            
            trueS[:,i] = true # save for later use                          
            
        return trueS
    
    
    def MSEvsK(k, trials, realizations, trueS,prior_type): # repeats is the number of trials and realizations is the number of times the experiment is run
        global expan
        global n
        
        exp = Experiment()
        exp.generate(n,k)
    
        print('\nSetting prior using k =', k)
    
        if prior_type=='logis':
            prior = Prior() 
            p = prior.set_prior_logis(expan,n,k) 
            exp.set_prior(p) # set the bins priors as the one calculated above 
        elif prior_type=='pow':
            prior = Prior() 
            p = prior.set_prior_pow(expan,n,k)
            exp.set_prior(p) # set the bins priors as the one calculated above 
            
        elif prior_type=='cuadra':
            prior = Prior() 
            p = prior.set_prior_cuadra(expan,n,k)
            exp.set_prior(p) 
    
        elif prior_type=='exp':
            prior = Prior() 
            p = prior.set_prior_exp(expan,n,k)
            exp.set_prior(p) 
            
        elif prior_type=='gauss':
            prior = Prior() 
            exp.generate(n,k) # non informative prior
            
        p_prior = exp.p.copy() 
        mat_prior = exp.mat.copy()
        invmat_prior = exp.invmat.copy()
        MSEs = np.zeros(shape=(trials,realizations))
        designs = np.zeros(shape=(trials,realizations))
        results = np.zeros(shape=(trials,realizations))
        

        for ii in range(realizations):
            # print('\rRealization number: ' + str(ii), end='', flush=True)
            exp.reset()
            exp.set_prior_mat(p_prior.copy(), mat_prior.copy(), invmat_prior.copy())
            
            true = trueS[:, ii]
    
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
            
        return MSEs.mean(), MSEs.std()/np.sqrt(realizations), designs, results
    
    ##################  Simulation 
    # prior with k 200
    print("\nGenerating prior with k 200")
    expan = 1000
    n = 10
    
    ### 
    
    MSEmeans = []
    MSEstds = []
    MSEmeans_freADO = []
    MSEstds_freADO  = []
    
    All_results = []
    All_designs = []
    
    trials = 300
    realizations = 320
    ks = [5,10,20,40,100,200]
    trueS = generate_trueS(realizations,prior_type)
    
    # ks = range(5,66,20)
    for k in ks:
        print('\nSimulating for k=', k)
        simulation = MSEvsK(k,trials,realizations, trueS,prior_type)
        MSEmeans.append(simulation[0])    
        MSEstds.append(simulation[1])
        # MSEmeans_freADO.append(simulation[2])    
        # MSEstds_freADO.append(simulation[3])
        All_designs.append(simulation[2])
        All_results.append(simulation[3])
        
        print( k, simulation[0], simulation[1]) 
    
    colors = ['blue','green']
    plt.figure(2)
    plt.title(f'prior ={prior_type}')
    plt.errorbar(ks, MSEmeans,MSEstds,c=colors[0])
    
    
    # =============================================================================
    # #simulte random experiment  with k=200
    # =============================================================================
    expan = 1000 # the final time
    n = 10 # number of design points

    
    
    exp = Experiment()
    exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
    aux = np.linspace(0, expan, n) # design points
    
    p_prior = exp.p.copy() 
    mat_prior = exp.mat.copy()
    invmat_prior = exp.invmat.copy()
    
    
    trials = 300
    ind =  320 # number of runs used to average the MSE vs. trial curve
    xs = np.zeros(shape=(trials+1, 2,ind)) # to store each trial number
    MSEs = np.zeros(shape=(trials+1,2,ind)) # to store the MSE vs trials values
    designs = []
    results = []
    
    jj=1 # to make a random simulation with the same prior
    if jj==0:
        print("simulating ADO experiment:")
        print("")
    else:
        print("simulating random design experiment:")
        print("")
    for ii in range(ind): 
        true = trueS[:, ii]
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
  
    mse_random=np.mean([MSEs[:,1][-1]])
    error_mse_random=np.std(MSEs[:,1][-1])/np.sqrt(ind)
    plt.axhline(mse_random,linestyle='--',color='k',label='Random K=200')
    plt.axhline(mse_random-error_mse_random,color='grey')
    plt.axhline(mse_random+error_mse_random,color='grey')
    plt.show()
