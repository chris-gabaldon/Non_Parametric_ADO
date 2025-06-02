from ADO_Experiment import *
import random
import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
from priors import *


# =============================================================================
# ## functions
# =============================================================================

def generate_trueS(n,realizations,prior_type): 
     print("\nGenerating truths")
     global expan
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
             true = random.random() + random.random()*(2*aux/expan)**2 #cuadra
             true[true>1] = 1
         
         elif prior_type=='gauss':
             mu = 0 
             sigma = np.random.uniform(expan/10, expan/5) 
             true = 1-np.exp(-0.5 * ((aux - mu) / sigma) ** 2)  
             true = true / np.max(true)  # Normalización para garantizar que los valores estén entre 0 y 1
             
         
         trueS[:,i] = true # save for later use                          
         
     return trueS
 
def MSEvsn(n, trials, realizations, trueS,prior_type): # repeats is the number of trials and realizations is the number of times the experiment is run
    global expan
    k=100
    exp = Experiment()
    exp.generate(n,k)

    print('\nSetting prior using n =', n)

    if prior_type=='logis':
        expan=300
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
    MSEs_frec = np.zeros(realizations)

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
        # estimate and error with freADO
        number_of_designs = np.array([len(designs[:,ii][designs[:,ii] == j])  for j in range(n)])
        number_of_positive_results = np.array([sum(results[:,ii][designs[:,ii] == j])  for j in range(n)])
        
        number_of_positive_results[number_of_designs==0] = 1
        number_of_designs[number_of_designs==0] = 1
        estimated_fre_ADO = number_of_positive_results/number_of_designs
        
        err = estimated_fre_ADO - true
        aux_err = np.mean(err*err)
        MSEs_frec[ii] = aux_err
        
    return MSEs.mean(axis = 1)[-1], MSEs.std(axis = 1)[-1]/np.sqrt(realizations)\


# =============================================================================
# simulation
# =============================================================================

list_priors=['exp','pow','logis','cuadra','gauss']


for prior_type in list_priors: 
    print('Prior type=',prior_type)
    if prior_type=='logis': # to be consistent with the logistic prior constructed in prior.py 
        expan=300
    else:
        expan = 1000
    k = 100
    
    ### 
    
    MSEmeans = []
    MSEstds = []

    
    All_results = []
    All_designs = []
    
    trials = 300
    realizations = 320
    ns = [2,5,10,20]


    for n in ns:
        print('\nSimulating for n=', n)
        trueS = generate_trueS(n,realizations,prior_type)
        
        simulation = MSEvsn(n,trials,realizations, trueS,prior_type)
        MSEmeans.append(simulation[0])    
        MSEstds.append(simulation[1])

        # All_designs.append(simulation[2])
        # All_results.append(simulation[3])
        
        print( n, simulation[0], simulation[1]) 

    plt.figure(2)
    plt.title(f'prior ={prior_type}')
    plt.errorbar(ns, MSEmeans,MSEstds,c='blue')
    filename = f"mse_resultados_{prior_type}.txt"

    with open(filename, "w") as f:
        f.write("n\tMSE\tError\n")
        for n, mse, err in zip(ns, MSEmeans, MSEstds):
            f.write(f"{n}\t{mse}\t{err}\n")
   
    plt.show()
    
    
    
