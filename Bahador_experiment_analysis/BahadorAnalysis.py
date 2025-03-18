# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:32:30 2025

@author: C_Gab
"""

# %% Necessary functions and third-party libraries
from ADO_Experiment import *
from priors import *
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats
#%% Construction of the data subj array

#the array data_subj contains many rows as the number of participants, and each row contains the designs (d) employed and their respective results (y)
n = 5 # number of design points used in the experiment
k = 100 # number of likelihood bins
eps=1e-100
data_subj=[]
datafile = 'BahadorData_1.csv'
n_subj=12# Filter out the first participants since they exhibited different number of trials
list_probs_per_subject=[] 
for subject in range(n_subj):
    exp1_data = np.genfromtxt(datafile, delimiter=',')
    exp1_data = exp1_data[200*subject:200 + 200*subject ]
    correct = exp1_data[:,2]==exp1_data[:,3]
    coherences = np.unique(exp1_data[:,1])
    coh = exp1_data[:,1]
    expan=coherences[-1]
    data_subj.append([correct,coh])
    All_Data =  [np.mean(correct[exp1_data[:,1] == coherences[i]]) for i in range(len(coherences))]
    list_probs_per_subject.append(All_Data)
    
# uncomment the following lines to plot the ground truth of the experiment -- note the monotonic trend

# d=np.sort(np.array(list(set(data_subj[1][1])))) # Vector of design points
# mean_prob=np.mean(list_probs_per_subject,axis=0)
# plt.scatter(d,mean_prob)
# plt.xlabel('Design')
# plt.ylabel('Prob')

#%% Data analysis

num_subj=len(data_subj)
trials=max([len(data_subj[i][0]) for i in range(num_subj)])
num_subj=len(data_subj)
d=np.sort(np.array(list(set(data_subj[1][1])))) # Vector of design points

D=len(d) 


'''
The following lists will store information from each trial, 
including the MSEs for each participant (for each method), 
the designs used, and the estimates obtained from the model.
'''

list_total_trials=[]
list_mses_ado_fre=[]
list_mses_ran_fre=[]
results_mses_ado=[]
results_mses_ran=[]
desgins_mses_ado=[]
desgins_mses_ran=[]
list_estimated_ado_fre=[]
list_estimated_ran_fre=[]


'''
Iterating over each participant and each trial. 
First, using ADO, measurements are selected one by one 
(in the original experiment order) based on the chosen design points .
Then, once saturation occurs, trials are repeated using the original order of the parameters selected in the experiment (but with the same number of trials where ADO saturates).
'''

for suj in range(num_subj):
    data_individual_subj=data_subj[suj]
    results_i=[]
    design_i=[]
    for i in range(len(d)):
        ind=np.where(data_individual_subj[1]==d[i])[0]
        results_design_i=[data_individual_subj[0][c] for c in ind]
        designs_i=[data_individual_subj[1][c] for c in ind]
        results_i.append(results_design_i)
        design_i.append(designs_i)

    true_i=np.array(list_probs_per_subject[suj]) # ground truth for an individual subject
    
    exp=Experiment() # initialize an ADO experiment
    exp.generate(D, k)
        
    counter=np.zeros(D) # uuxiliary variable for iterate the trials in the same order as the original trials
    res_counter=np.zeros(D)
    
    result_ado_1=[]
    design_ado_1=[]
    mse_ado_fre_1=[]

    list_mses_ado_fre.append(mse_ado_fre_1)
    for i in range(trials):
        '''
        The first `if` implements a full sweep over all design points initially.
        `n_first_loops` sets how many times this sweep should occur.
        '''
        n_first_loops=3
        ff=[range(D)[mm % D] for mm in range(n_first_loops*D)]
        if i<n_first_loops*D:
            
            d1=ff[i]
        else:       
            d1=exp.ADOchoose()
        
        result=results_i[d1][int(counter[d1])]
        exp.update(d1,result)
        res_counter[d1]+=result
        counter[d1]+=1
        result_ado_1.append(result)
        design_ado_1.append(d[d1])
        
        '''
       Estimations and MSE calculations:
       If `1 - true_i` exists, it means data was inverted 
       to ensure decreasing monotonic curves instead of increasing ones.
       
       '''
        
        estimated_fre=res_counter/(counter+eps)
        mse_f=np.mean((estimated_fre-(true_i))**2) # calculate the mse (eps is used to avoid divisions by zero)
        mse_ado_fre_1.append(mse_f)
        
        if i%int((trials*0.3))==0:
            print(f'ado_trial={i}/{trials},,,subject:{suj}/{num_subj}')
        
        if counter[d1]>=len(results_i[d1]): # if there is not available data, break the loop and pass to the pseudorandom loop
            print(f'saturation at design d={d1} ,using {i} trials')
            list_estimated_ado_fre.append(estimated_fre)
            break
        
            
    results_mses_ado.append(result_ado_1)
    desgins_mses_ado.append(design_ado_1)
    
    
    '''
    Pseudorandom Experiment
    '''
    
    design_ran_1=[]
    result_ran_1=[]
    mse_ran_fre_1=[]

    exp=Experiment() # create a new instance of Experiment, reset the old variables of the model
    exp.generate(D, k)
        
    total_trials=int(counter.sum()) # the number of trials before the ado loop finish
    list_total_trials.append(total_trials)
    counter_2=np.zeros(D)
    res_counter_2=np.zeros(D)
    
    for jj in range(total_trials):
        d1=list(d).index(data_individual_subj[1][jj]) # the parameter 'choose' follows the experiment original order
        result=data_individual_subj[0][jj]
        exp.update(d1,result)
        counter_2[d1]+=1
        res_counter_2[d1]+=result
        result_ran_1.append(result)
        design_ran_1.append(d[d1])
        
        
        estimated_fre= res_counter_2/(counter_2+eps) # calculate the mse
        mse_f=np.mean((estimated_fre-(true_i))**2)
        mse_ran_fre_1.append(mse_f)
        
        
        if jj==0:
            print(f'random_trial={i}/{trials},,,subject:{suj}/{num_subj}'+'\n')
            
    
    list_estimated_ran_fre.append(estimated_fre)

    results_mses_ran.append(result_ran_1)
    desgins_mses_ran.append(design_ran_1)
    
    list_mses_ran_fre.append(mse_ran_fre_1)
   
last_mse_ado_fre=[]
last_mse_ran_fre=[]

for i in range(num_subj):
    last_mse_ado_fre.append(list_mses_ado_fre[i][-1])
    last_mse_ran_fre.append(list_mses_ran_fre[i][-1])

min_subj_ni=min([len(list_mses_ado_fre[i]) for i in range(num_subj)]) # number of the participant's trials 

'''
Naturally, each participant will have a different number of trials because the algorithm saturated
at different trials. Therefore, we select the minimum possible trial so that all data points have
the same number of participants.
'''

# #####################
# #gaussian prior (same analisys but diferent prior for the ADO algorithm)
# #####################

num_subj=len(data_subj)
trials=max([len(data_subj[i][0]) for i in range(num_subj)])
num_subj=len(data_subj)
d=np.sort(np.array(list(set(data_subj[5][1])))) # Vector of design points



list_total_trials=[]
list_mses_ado_fre_gauss=[]
results_mses_ado=[]
desgins_mses_ado=[]
list_estimated_ado_fre_gauss=[]


for suj in range(num_subj):
    data_individual_subj=data_subj[suj]
    results_i=[]
    design_i=[]
    for i in range(len(d)):
        ind=np.where(data_individual_subj[1]==d[i])[0]
        results_design_i=[data_individual_subj[0][c] for c in ind]
        designs_i=[data_individual_subj[1][c] for c in ind]
        results_i.append(results_design_i)
        design_i.append(designs_i)

    true_i=np.array(list_probs_per_subject[suj]) # ground truth for an individual subject
    
    exp=Experiment() # initialize an ADO experiment
    exp.generate(D, k)
    prior = Prior() # generate an instance of Prior
    p = prior.set_prior_gauss(expan,n,k) # generate a gaussian prior
    exp.set_prior(p) # set the bins priors as the one calculated above 
    
    counter=np.zeros(D) #Auxiliary variable for iterate the trials in the same order as the original trials
    res_counter=np.zeros(D)
    
    result_ado_1=[]
    design_ado_1=[]
    
    mse_ado_fre_1=[]

    list_mses_ado_fre_gauss.append(mse_ado_fre_1)
    for i in range(trials):
        n_first_loops=3
        ff=[range(D)[mm % D] for mm in range(n_first_loops*D)]
        if i<n_first_loops*D:
            
            d1=ff[i]
        else:       
            d1=exp.ADOchoose()
        
        result=results_i[d1][int(counter[d1])]
        exp.update(d1,result)
        res_counter[d1]+=result
        counter[d1]+=1
        result_ado_1.append(result)
        design_ado_1.append(d[d1])
        
        estimated_fre=res_counter/(counter+eps)
        mse_f=np.mean((estimated_fre-(true_i))**2)
        mse_ado_fre_1.append(mse_f)
        
        if i%int((trials*0.3))==0:
            print(f'ado_trial={i}/{trials},,,subject:{suj}/{num_subj}')
        
        if counter[d1]>=len(results_i[d1]): # if there is not available data, break the loop 
            list_estimated_ado_fre_gauss.append(estimated_fre)
            break
        
            
    results_mses_ado.append(result_ado_1)
    desgins_mses_ado.append(design_ado_1)
    
    

min_subj_gauss=min([len(list_mses_ado_fre_gauss[i]) for i in range(num_subj)]) # number of the minimum participant's trials 

list_mses_ado_fre_g=[list_mses_ado_fre_gauss[i][:min_subj_gauss] for i in range(num_subj)]
list_mses_ado_fre_ni=[list_mses_ado_fre[i][:min_subj_ni] for i in range(num_subj)]
list_mses_ado_fre_ran=[list_mses_ran_fre[i][:min_subj_ni] for i in range(num_subj)]

# logarithmic transformation for the means and error bars

std_mses_random=np.std(list_mses_ado_fre_ran,axis=0)/np.sqrt(len(list_mses_ado_fre_ran))
std_mses_ni=np.std(list_mses_ado_fre_ni,axis=0)/np.sqrt(len(list_mses_ado_fre_ni))
std_mses_g=np.std(list_mses_ado_fre_g,axis=0)/np.sqrt(len(list_mses_ado_fre_g))

mean_tr_ni=np.log10(np.mean(list_mses_ado_fre_ni, axis=0))
mean_tr_g=np.log10( np.mean(list_mses_ado_fre_g, axis=0))
mean_tr_r=np.log10(np.mean(list_mses_ado_fre_ran, axis=0))

rand_sup=np.abs(np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random)-mean_tr_r)
rand_inf=np.abs(-np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random)+mean_tr_r)

ni_sup=np.abs(np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni)-mean_tr_ni)
ni_inf=np.abs(-np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni)+mean_tr_ni)

g_sup=np.abs(np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g)-mean_tr_g)
g_inf=np.abs(-np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g)+mean_tr_g)

# plot mse vs trial
plt.errorbar(range(len(list_mses_ado_fre_ni[0])), np.log10(np.mean(list_mses_ado_fre_ni, axis=0)),yerr=[ni_inf,ni_sup],label='ADO (NI prior)',c='b')
plt.errorbar(range(len(list_mses_ado_fre_g[0])),np.log10( np.mean(list_mses_ado_fre_g, axis=0)),yerr=[g_inf,g_sup],label='ADO (Gaussian prior)',c='c')
plt.errorbar(range(len(list_mses_ado_fre_ran[0])), np.log10(np.mean(list_mses_ado_fre_ran, axis=0)),yerr=[rand_inf,rand_sup],label='Pseudorandom ',c='r')

plt.xlabel('Trials')
plt.ylabel('log(MSE)')

plt.legend()