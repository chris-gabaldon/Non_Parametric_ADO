# -*- coding: utf-8 -*-
"""
Created on Sun May  4 16:28:27 2025

@author: C_Gab
"""
from ADO_Experiment import *
import random
import matplotlib.pyplot as plt

import numpy as np
from priors import *


# =============================================================================
# ## functions
# =============================================================================

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
             true = 1/(1 + np.exp(-0.1*random.random()*(aux-200 +  100*random.random()))) #logis
         
         elif prior_type=='cuadra':
             true =   random.random()*(aux/expan)**2 #cuadra
             true[true>1] = 1
         
         elif prior_type=='gauss':
 
             mu = 0 
             sigma = np.random.uniform(expan/10, expan/5) 
             true = 1-np.exp(-0.5 * ((aux - mu) / sigma) ** 2)  
             true = true / np.max(true)  # Normalización para garantizar que los valores estén entre 0 y 1
             
         
         trueS[:,i] = true # save for later use                          
         
     return trueS

def MSEvsK(k, trials, realizations, trueS): # repeats is the number of trials and realizations is the number of times the experiment is run
    global expan
    global n
    
    exp = Experiment()
    exp.generate(n,k)

    print('\nSetting prior using k =', k)

    if prior_type=='logis':
        prior = Prior() 
        expan=300
        p = prior.set_prior_logis(expan,n,k) 
        exp.set_prior(p) # set the bins priors as the one calculated above 
    elif prior_type=='pow':
        expan=1000
        prior = Prior() 
        p = prior.set_prior_pow(expan,n,k)
        exp.set_prior(p) # set the bins priors as the one calculated above 
        
    elif prior_type=='cuadra':
        expan=1000
        prior = Prior() 
        p = prior.set_prior_cuadra(expan,n,k)
        exp.set_prior(p) 

    elif prior_type=='exp':
        expan=1000
        prior = Prior() 
        p = prior.set_prior_exp(expan,n,k)
        exp.set_prior(p) 
        
    elif prior_type=='gauss':
        expan=1000
        prior = Prior() 
        p=prior.set_prior_gauss(expan,n,k)
        
    
    print('\nSetting prior using k =', k)

    p_prior = exp.p.copy()
    mat_prior = exp.mat.copy()
    invmat_prior = exp.invmat.copy()
    
    MSEs = np.zeros(shape=(trials,realizations))
    designs = np.zeros(shape=(trials,realizations))
    results = np.zeros(shape=(trials,realizations))
    
    MSEs3 = np.zeros(realizations)
    for ii in range(realizations): # represents the meassuraments with a lower k
        print('\rRealization number: ' + str(ii), end='', flush=True)
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

        # estimate and error with freADO
        number_of_designs = np.array([len(designs[:,ii][designs[:,ii] == j])  for j in range(n)])
        number_of_positive_results = np.array([sum(results[:,ii][designs[:,ii] == j])  for j in range(n)])
        
        number_of_positive_results[number_of_designs==0] = 1
        number_of_designs[number_of_designs==0] = 1
        estimated_fre_ADO = number_of_positive_results/number_of_designs
        
        err = estimated_fre_ADO - true
        aux_err = np.mean(err*err)
        MSEs3[ii] = aux_err
        
    # ####### run new simulations with k 200 and saved desings and results        
    MSEs2 = np.zeros(shape=(trials,realizations))    
    print("\nGenerating simulations with obtained results")
    exp = Experiment()
    exp.generate(n,200)

    print('\nSimulation usigin k = 200 and saved results')
    for ii in range(realizations):
        print('\rRealization number: ' + str(ii), end='', flush=True)
        exp.reset()
        exp.set_prior_mat(p_prior2.copy(), mat_prior2.copy(), invmat_prior2.copy()) # prior2 has k=200 and do not change with iterations of k
        true = trueS[:,ii]      
        MSE = []
    
        for i in range(trials):           
            aux1 = designs[i,ii]
            result = results[i,ii]
            exp.update(aux1.astype(int),result.astype(int))        
            estimated = exp.values*exp.p
            estimated = estimated.sum(axis = 1)
            err = estimated - true
            aux_err = np.mean(err*err)
            MSE.append(aux_err)
            
        MSEs2[:,ii] = MSE  
        
    return MSEs2.mean(axis = 1)[-1], MSEs2.std(axis = 1)[-1]/np.sqrt(realizations)\
       ,MSEs3.mean(), MSEs3.std()/np.sqrt(realizations), designs, results
      


# =============================================================================
# #simulation
# =============================================================================


list_priors=['exp','pow','logis','cuadra','gauss']
# list_priors=['logis']

for prior_type in list_priors: 
    
    print('Prior type=',prior_type)
    if prior_type=='logis':
        expan=300
    else:
        expan = 1000
    n = 10 

    k_prov = 200    
    
    # generate an instance only for creating the prior2 with k=200 and making a copy that will be used in the function MSEvsK
    exp = Experiment()
    exp.generate(n,k_prov)

    if prior_type=='logis':
        prior = Prior() 
        expan=300
        p = prior.set_prior_logis(expan,n,k_prov) 
        exp.set_prior(p) # set the bins priors as the one calculated above 
    elif prior_type=='pow':
        prior = Prior() 
        p = prior.set_prior_pow(expan,n,k_prov)
        exp.set_prior(p) # set the bins priors as the one calculated above 
        
    elif prior_type=='cuadra':
        prior = Prior() 
        p = prior.set_prior_cuadra(expan,n,k_prov)
        exp.set_prior(p) 
 
    elif prior_type=='exp':
        prior = Prior() 
        p = prior.set_prior_exp(expan,n,k_prov)
        exp.set_prior(p) 
        
    elif prior_type=='gauss':
        prior = Prior() 
        p=prior.set_prior_gauss(expan,n,k_prov) # non informative prior
        exp.set_prior(p)
        
    p_prior2 = exp.p.copy()
    mat_prior2 = exp.mat.copy()
    invmat_prior2 = exp.invmat.copy()

    ### 
    # ks = [5,10] 0
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

    for k in ks:
        print('\nSimulating for k=', k)
        simulation = MSEvsK(k,trials,realizations, trueS)
        MSEmeans.append(simulation[0])    
        MSEstds.append(simulation[1])
        # MSEmeans_freADO.append(simulation[2])    
        # MSEstds_freADO.append(simulation[3])
        All_designs.append(simulation[2])
        All_results.append(simulation[3])
        
        print( k, simulation[0], simulation[1]) 

    plt.figure(2)
    plt.errorbar(ks, MSEmeans,MSEstds,c='blue')

    
    # =============================================================================
    # #simulte random experiment  with k=200
    # =============================================================================
    
    
    
    if prior_type=='logis': # to be consistent with the logistic prior constructed in priors.py
        expan=300
    else:
        expan = 1000
    n = 10 # number of design points
    k=200
    
    
    exp = Experiment()
    exp.generate(n,k) # generate the bins priors considering all monotonic curves equally probable
    aux = np.linspace(0, expan, n) # design points
    
    
    
    trials = 300
    ind =  realizations # number of runs used to average the MSE vs. trial curve
    xs = np.zeros(shape=(trials+1, 2,ind)) # to store each trial number
    MSEs_rand = np.zeros(shape=(trials+1,2,ind)) # to store the MSE vs trials values
    designs = []
    results = []
    
    jj=1 # to make a random simulation with the same prior

    print("simulating random design experiment:")

    for ii in range(ind): 
        true = trueS[:, ii]
        print("run number:",ii)
        exp.reset()
        exp.set_prior_mat(p_prior2, mat_prior2, invmat_prior2)       # use the same prior with fixed k=200 likelihood bins
        x = [0]
        y = []
        estimated = exp.values*exp.p
        estimated = estimated.sum(axis = 1) # estimated model at the begining
        err = estimated - true
        aux_err = np.mean(err*err) # error at the begining
        MSE = [aux_err]
        info = [exp.infoProgress()]
        for i in range(trials):  
 
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
            aux_err = np.mean(err*err)
            MSE.append(aux_err)
            info.append(exp.infoProgress())
        aux1 = exp.RANDchoose()  
        designs.append(aux1)    
        MSEs_rand[:,jj,ii] = MSE
        xs[:,jj,ii] = x
    
    mse_random=np.mean([MSEs_rand[:,1][-1]])
    error_mse_random=np.std(MSEs_rand[:,1][-1])/np.sqrt(ind)
    plt.axhline(mse_random,linestyle='--',color='k',label='Random K=200')
    plt.axhline(mse_random-error_mse_random,color='grey')
    plt.axhline(mse_random+error_mse_random,color='grey')
    
    filename = f"mse_resultados_{prior_type}.txt"

    with open(filename, "w") as f:
        f.write("k\tMSE\tError\n")
        for k, mse, err in zip(ks, MSEmeans, MSEstds):
            f.write(f"{k}\t{mse}\t{err}\n")
        f.write("\n# MSE random\n")
        f.write(f"MSE_random\t{mse_random}\n")
        f.write(f"Error_random\t{error_mse_random}\n")
        
    
    plt.show()


