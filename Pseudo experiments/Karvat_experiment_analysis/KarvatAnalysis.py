# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:23:57 2025

@author: C_Gab
"""
#%%  Necessary functions and third-party libraries
import math
import random   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from ADO_Experiment import *
from priors import *

#%% Create an instance of - Sensory drive 

# Set the path where the Experiment and Prior classes, and the experiment data are stored
path ='path' # 
data_high_low_contrast = pd.read_csv('data_high_low_contrat.csv')


num_subj = 22 # number of subjects who performed the experimental task

# Contrast level: 1 (high contrast) or 0.5 (low contrast)
contrast = 1
k=100

dataframes_participants = {} # create separate dataframes for each participant
participantes = data_high_low_contrast['subject'].unique()

for p in participantes:
    dataframes_participants[p] = data_high_low_contrast[data_high_low_contrast['subject'] == p]
    

dataframes_participants = pd.concat(dataframes_participants.values(), ignore_index=True) # combine all participant DataFrames into a single dataframe


grupos_participants_contrast = dataframes_participants.groupby(['subject', 'contrast']) # group data by subject and contrast level

datos_contrast_1 = []
for i in range(num_subj):
    part_i = grupos_participants_contrast.get_group((i + 1, 1)).drop(columns=['rt', 'n_flashes', 'contrast', 'subject'])
    array_res = part_i.values[:, 1]
    array_d = part_i.values[:, 0]
    datos_contrast_1.append([array_res, array_d])

datos_contrast_05 = []
for i in range(num_subj):
    part_i = grupos_participants_contrast.get_group((i + 1, 0.5)).drop(columns=['rt', 'n_flashes', 'contrast', 'subject'])
    array_res = part_i.values[:, 1]
    array_d = part_i.values[:, 0]
    datos_contrast_05.append([array_res, array_d])

if contrast == 0.5:
    data_subj = datos_contrast_05
    color = 'orangered'
else:
    data_subj = datos_contrast_1
    color = 'b'

d = np.sort(np.array(list(set(data_subj[0][1])))) # extract unique sorted values for the stimulus parameter 'd'

eps = 1e-100  # small value to avoid division by zero

# compute probability of not detecting two flashes for each participant ## frequentist groundtruth
list_probs_per_subject = []

for i in range(num_subj):
    parcial_list = []
    list_probs_per_subject.append(parcial_list)
    for j in range(len(d)):
        ind = np.where(data_subj[i][1] == d[j])[0]
        results = [data_subj[i][0][k] for k in ind]
        ratio = np.sum(results) / (len(ind) + eps)
        parcial_list.append(ratio)
        


list_probs = np.mean(np.array(list_probs_per_subject), axis=0)

# Scatter plot of the mean probabilities
plt.scatter(d, list_probs, label=f'Average across subjects - Contrast {contrast}', c=color)
plt.xlabel('Gap (ms)', fontsize=12)
plt.ylabel('Probability of NOT detecting two flashes', fontsize=12)
plt.grid()
plt.legend()
plt.show()

# Plot individual probability curves for each subject
for curve in list_probs_per_subject:
    plt.plot(d, curve)
plt.xlabel('Gap (ms)', fontsize=12)
plt.ylabel('Probability of NOT detecting two flashes', fontsize=12)
plt.grid()
plt.show()


def generate_prob_per_subject_type_bay(data_subj, prior_type):
    d = np.sort(np.array(list(set(data_subj[0][1]))))
    num_subj = len(data_subj)
    list_probs_per_subject_type = []
    for subj in range(num_subj):
        data_individual_subj = data_subj[subj]
        D,k = len(d),100
        exp = Experiment() # initialize an ADO experiment.
        if prior_type == 2:
            exp.generate(D, k)
            prior = Prior() 
            p = prior.set_prior_gauss(d[-1],D,k) # Calculate the prior based on exponential curves, shown in Figure 1b.
            exp.set_prior(p)
        else:
            exp.generate(D,k)
        total_trials = len(data_individual_subj[0])
        for jj in range(total_trials):
            if jj >= len(data_individual_subj[1]):
                break
            d1 = list(d).index(data_individual_subj[1][jj])
            result = data_individual_subj[0][jj]
            exp.update(d1,result)
            estimated_bay = (exp.values*exp.p).sum(axis=1)
        estimated_bay = (exp.values*exp.p).sum(axis=1)
        list_probs_per_subject_type.append(estimated_bay)
    return list_probs_per_subject_type

'''
These lists will store the lists of Bayesian probability 
curves for each experimental group.
'''
list_probs_per_subject_bay_ni = generate_prob_per_subject_type_bay(data_subj,1)
list_probs_per_subject_bay_g = generate_prob_per_subject_type_bay(data_subj,2)

ground_truth_bay_ni=np.mean(list_probs_per_subject_bay_ni,axis=0)
ground_truth_bay_g=np.mean(list_probs_per_subject_bay_g,axis=0)
    

#%% Data analysis 
k=100
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
list_mses_ado_bay=[]
list_mses_ran_fre=[]
list_mses_ran_bay=[]
results_mses_ado=[]
results_mses_ran=[]
desgins_mses_ado_ni=[]
desgins_mses_ran=[]
list_estimated_ado_fre=[]
list_estimated_ran_fre=[]
list_estimated_ado_bay=[]
list_estimated_ran_bay=[]


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
    true_bay=list_probs_per_subject_bay_ni[suj]
    
    exp=Experiment() # initialize an ADO experiment
    exp.generate(D, k)
        
    counter=np.zeros(D) # uuxiliary variable for iterate the trials in the same order as the original trials
    res_counter=np.zeros(D)
    
    result_ado_1=[]
    design_ado_1=[]
    mse_ado_fre_1=[]
    mse_ado_bay_1=[]

    list_mses_ado_fre.append(mse_ado_fre_1)
    list_mses_ado_bay.append(mse_ado_bay_1)
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
        estimated_bay = (exp.values*exp.p).sum(axis=1)
        estimated_fre=res_counter/(counter+eps)
        
        mse_bay=np.mean((estimated_bay-(true_bay))**2)
        mse_f=np.mean((estimated_fre-(true_i))**2) # calculate the mse (eps is used to avoid divisions by zero)
        
        mse_ado_fre_1.append(mse_f)
        mse_ado_bay_1.append(mse_bay)
        
        if i%int((trials*0.3))==0:
            print(f'ado_trial={i}/{trials},,,subject:{suj}/{num_subj}')
        
        if counter[d1]>=len(results_i[d1]): # if there is not available data, break the loop and pass to the pseudorandom loop
            print(f'saturation at design d={d1} ,using {i} trials')
            list_estimated_ado_fre.append(estimated_fre)
            list_estimated_ado_bay.append(estimated_bay)  
            break
        
            
    results_mses_ado.append(result_ado_1)
    desgins_mses_ado_ni.append(design_ado_1)
    
    
    '''
    Pseudorandom Experiment
    '''
    
    design_ran_1=[]
    result_ran_1=[]
    mse_ran_fre_1=[]
    mse_ran_bay_1=[]

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
        
        estimated_bay = (exp.values*exp.p).sum(axis=1)
        estimated_fre= res_counter_2/(counter_2+eps) # calculate the mse
        
        mse_bay=np.mean((estimated_bay-(true_bay))**2)
        mse_f=np.mean((estimated_fre-(true_i))**2)
        
        mse_ran_bay_1.append(mse_bay)
        mse_ran_fre_1.append(mse_f)
        
        
        if jj==0:
            print(f'random_trial={i}/{trials},,,subject:{suj}/{num_subj}'+'\n')
            
    
    list_estimated_ran_fre.append(estimated_fre)
    list_estimated_ran_bay.append(estimated_bay)
    results_mses_ran.append(result_ran_1)
    desgins_mses_ran.append(design_ran_1)
    
    list_mses_ran_fre.append(mse_ran_fre_1)
    list_mses_ran_bay.append(mse_ran_bay_1)
    
last_mse_ado_fre=[]
last_mse_ran_fre=[]
last_mse_ado_bay=[]
last_mse_ran_bay=[]

for i in range(num_subj):
    last_mse_ado_fre.append(list_mses_ado_fre[i][-1])
    last_mse_ran_fre.append(list_mses_ran_fre[i][-1])
    last_mse_ado_bay.append(list_mses_ado_bay[i][-1])
    last_mse_ran_bay.append(list_mses_ran_bay[i][-1])

min_subj_ni_fre=min([len(list_mses_ado_fre[i]) for i in range(num_subj)]) # number of the participant's trials 
min_subj_ni_bay=min([len(list_mses_ado_bay[i]) for i in range(num_subj)]) 
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


list_mses_ran_bay_gauss=[]
list_total_trials=[]
list_mses_ado_fre_gauss=[]
list_mses_ado_bay_gauss=[]
results_mses_ado=[]
desgins_mses_ado=[]
list_estimated_ado_fre_gauss=[]
list_estimated_ado_bay_gauss=[]
list_estimated_ran_bay_gauss=[]
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
    true_bay=list_probs_per_subject_bay_g[suj]
    
    exp=Experiment() # initialize an ADO experiment
    exp.generate(D, k)
    prior = Prior() # generate an instance of Prior
    p = prior.set_prior_gauss(expan,D,k) # generate a gaussian prior
    exp.set_prior(p) # set the bins priors as the one calculated above 
    
    counter=np.zeros(D) #Auxiliary variable for iterate the trials in the same order as the original trials
    res_counter=np.zeros(D)
    
    result_ado_1=[]
    design_ado_1=[]
    
    mse_ado_fre_1=[]
    mse_ado_bay_1=[]

    list_mses_ado_fre_gauss.append(mse_ado_fre_1)
    list_mses_ado_bay_gauss.append(mse_ado_bay_1)
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
        
        estimated_bay = (exp.values*exp.p).sum(axis=1)
        estimated_fre=res_counter/(counter+eps)
        
        mse_bay=np.mean((estimated_fre-(true_bay))**2)
        mse_f=np.mean((estimated_fre-(true_i))**2)
        
        mse_ado_bay_1.append(mse_bay)
        mse_ado_fre_1.append(mse_f)
        
        if i%int((trials*0.3))==0:
            print(f'ado_trial={i}/{trials},,,subject:{suj}/{num_subj}')
        
        if counter[d1]>=len(results_i[d1]): # if there is not available data, break the loop 
            list_estimated_ado_fre_gauss.append(estimated_fre)
            list_estimated_ado_bay_gauss.append(estimated_bay)
            break
        
            
    results_mses_ado.append(result_ado_1)
    desgins_mses_ado.append(design_ado_1)
    
    '''
    Pseudorandom Experiment
    '''
    
    design_ran_1=[]
    result_ran_1=[]
    mse_ran_fre_1=[]
    mse_ran_bay_1=[]

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
        
        estimated_bay = (exp.values*exp.p).sum(axis=1)
        estimated_fre= res_counter_2/(counter_2+eps) # calculate the mse
        
        mse_bay=np.mean((estimated_bay-(true_bay))**2)
        mse_f=np.mean((estimated_fre-(true_i))**2)
        
        mse_ran_bay_1.append(mse_bay)
        mse_ran_fre_1.append(mse_f)
        
        
        if jj==0:
            print(f'random_trial={i}/{trials},,,subject:{suj}/{num_subj}'+'\n')
            
    
    list_estimated_ran_bay_gauss.append(estimated_bay)
    results_mses_ran.append(result_ran_1)
    desgins_mses_ran.append(design_ran_1)
    
    list_mses_ran_fre.append(mse_ran_fre_1)
    list_mses_ran_bay_gauss.append(mse_ran_bay_1)
    
last_mse_ado_fre=[]
last_mse_ran_fre=[]
last_mse_ado_bay=[]
last_mse_ran_bay=[]

for i in range(num_subj):
    last_mse_ado_fre.append(list_mses_ado_fre[i][-1])
    last_mse_ran_fre.append(list_mses_ran_fre[i][-1])
    last_mse_ado_bay.append(list_mses_ado_bay[i][-1])
    last_mse_ran_bay.append(list_mses_ran_bay_gauss[i][-1])


min_subj_gauss_fre=min([len(list_mses_ado_fre_gauss[i]) for i in range(num_subj)]) # number of the minimum participant's trials 
min_subj_gauss_bay=min([len(list_mses_ado_bay_gauss[i]) for i in range(num_subj)]) 

list_mses_ado_fre_g=[list_mses_ado_fre_gauss[i][:min_subj_gauss_fre] for i in range(num_subj)]
list_mses_ado_fre_ni=[list_mses_ado_fre[i][:min_subj_ni_fre] for i in range(num_subj)]
list_mses_ado_fre_ran=[list_mses_ran_fre[i][:min_subj_ni_fre] for i in range(num_subj)]

list_mses_ado_bay_g=[list_mses_ado_bay_gauss[i][:min_subj_gauss_bay] for i in range(num_subj)]
list_mses_ado_bay_ni=[list_mses_ado_bay[i][:min_subj_ni_bay] for i in range(num_subj)]
list_mses_ado_bay_ran=[list_mses_ran_bay[i][:min_subj_ni_bay] for i in range(num_subj)]

# logarithmic transformation for the means and error bars

std_mses_random_fre=np.std(list_mses_ado_fre_ran,axis=0)/np.sqrt(len(list_mses_ado_fre_ran))
std_mses_ni_fre=np.std(list_mses_ado_fre_ni,axis=0)/np.sqrt(len(list_mses_ado_fre_ni))
std_mses_g_fre=np.std(list_mses_ado_fre_g,axis=0)/np.sqrt(len(list_mses_ado_fre_g))

std_mses_random_bay=np.std(list_mses_ado_bay_ran,axis=0)/np.sqrt(len(list_mses_ado_bay_ran))
std_mses_ni_bay=np.std(list_mses_ado_bay_ni,axis=0)/np.sqrt(len(list_mses_ado_bay_ni))
std_mses_g_bay=np.std(list_mses_ado_bay_g,axis=0)/np.sqrt(len(list_mses_ado_bay_g))

mean_tr_ni_fre=np.log10(np.mean(list_mses_ado_fre_ni, axis=0))
mean_tr_g_fre=np.log10( np.mean(list_mses_ado_fre_g, axis=0))
mean_tr_r_fre=np.log10(np.mean(list_mses_ado_fre_ran, axis=0))

mean_tr_ni_bay=np.log10(np.mean(list_mses_ado_bay_ni, axis=0))
mean_tr_g_bay=np.log10( np.mean(list_mses_ado_bay_g, axis=0))
mean_tr_r_bay=np.log10(np.mean(list_mses_ado_bay_ran, axis=0))

rand_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random_fre)-mean_tr_r_fre)
rand_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random_fre)+mean_tr_r_fre)

rand_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_ran, axis=0)+std_mses_random_bay)-mean_tr_r_bay)
rand_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_ran, axis=0)+std_mses_random_bay)+mean_tr_r_bay)

ni_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni_fre)-mean_tr_ni_fre)
ni_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni_fre)+mean_tr_ni_fre)

ni_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_ni, axis=0)+std_mses_ni_bay)-mean_tr_ni_bay)
ni_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_ni, axis=0)+std_mses_ni_bay)+mean_tr_ni_bay)

g_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g_fre)-mean_tr_g_fre)
g_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g_fre)+mean_tr_g_fre)

g_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_g, axis=0)+std_mses_g_bay)-mean_tr_g_bay)
g_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_g, axis=0)+std_mses_g_bay)+mean_tr_g_bay)

# plot mse vs trial
plt.title('Frequentist inference')
plt.errorbar(range(len(list_mses_ado_fre_ni[0])), np.log10(np.mean(list_mses_ado_fre_ni, axis=0)),yerr=[ni_inf_fre,ni_sup_fre],label='ADO (NI prior) fre',c='b')
plt.errorbar(range(len(list_mses_ado_fre_g[0])),np.log10( np.mean(list_mses_ado_fre_g, axis=0)),yerr=[g_inf_fre,g_sup_fre],label='ADO (Cuadra prior)',c='g')
plt.errorbar(range(len(list_mses_ado_fre_ran[0])), np.log10(np.mean(list_mses_ado_fre_ran, axis=0)),yerr=[rand_inf_fre,rand_sup_fre],label='Pseudorandom fre',c='r')

plt.xlabel('Trials')
plt.ylabel('log(MSE)')
plt.legend()

# plt.show()
# plot mse vs trial
plt.title('Bayesian inference')
plt.errorbar(range(len(list_mses_ado_bay_ni[0])), np.log10(np.mean(list_mses_ado_bay_ni, axis=0)),yerr=[ni_inf_bay,ni_sup_bay],label='ADO (NI prior) bay',c='c')
plt.errorbar(range(len(list_mses_ado_bay_g[0])),np.log10( np.mean(list_mses_ado_bay_g, axis=0)),yerr=[g_inf_bay,g_sup_bay],label='ADO (Cuadra prior)',c='lightgreen')
plt.errorbar(range(len(list_mses_ado_bay_ran[0])), np.log10(np.mean(list_mses_ado_bay_ran, axis=0)),yerr=[rand_inf_bay,rand_sup_bay],label='Pseudorandom bay ',c='m')

plt.xlabel('Trials')
plt.ylabel('log(MSE)')
plt.legend()

#%% Logarithmic fit to obtain the PSE point through different algorithms

from scipy.optimize import curve_fit

# Define the logistic function
def logistic(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

list_errors=[np.std(list_probs_per_subject,axis=0)/np.sqrt(22),np.std(list_probs_per_subject_bay_ni,axis=0)/np.sqrt(22),np.std(list_probs_per_subject_bay_g,axis=0)/np.sqrt(22),np.std(list_estimated_ado_fre, axis=0)/np.sqrt(22),np.std(list_estimated_ran_fre, axis=0)/np.sqrt(num_subj),np.std(list_estimated_ado_fre_gauss, axis=0)/np.sqrt(num_subj),np.std(list_estimated_ado_bay, axis=0)/np.sqrt(22),np.std(list_estimated_ran_bay, axis=0)/np.sqrt(num_subj),np.std(list_estimated_ado_bay_gauss, axis=0)/np.sqrt(num_subj),np.std(list_estimated_ran_bay_gauss, axis=0)/np.sqrt(num_subj)]

list_y_data = [list_probs,ground_truth_bay_ni,  ground_truth_bay_g ,np.mean(list_estimated_ado_fre, axis=0), np.mean(list_estimated_ran_fre, axis=0),np.mean(list_estimated_ado_fre_gauss, axis=0),np.mean(list_estimated_ado_bay, axis=0), np.mean(list_estimated_ran_bay, axis=0),np.mean(list_estimated_ado_bay_gauss, axis=0),np.mean(list_estimated_ran_bay_gauss, axis=0)]
labels = ['Ground truth fre', 'Ground Truth NI','Ground Truth G','ADO-NI fre', 'Random fre','ADO-G fre','ADO-NI bay', 'Random bay NI','ADO-G bay','Random bay G']
colors = ['k','brown','grey','b', 'r', 'g','c','m','lightgreen','pink']
k
for i, y_data in enumerate(list_y_data):

    x_data = d
    
    # logistic fit
    initial_guesses = [1, np.median(x_data), 1, 0]  # L, x0, k, b
    params, covariance = curve_fit(logistic, x_data, y_data, p0=initial_guesses)
    
    # parameters
    L, x0, k, b = params
    
    # PSE value is for definition the x0 param 
    PSE = x0
    
    # calculate the PSE error
    perr = np.sqrt(np.diag(covariance))
    PSE_err = perr[1]

    print(f"El valor de PSE es: {PSE:.2f} ± {PSE_err:.2f}")

    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = logistic(x_fit, *params)
    
    # plot
    plt.errorbar(x_data, y_data,yerr=list_errors[i] ,label=f'{labels[i]}', fmt='x',color=colors[i])
    
    plt.plot(x_fit, y_fit,  color=colors[i])
    plt.axvline(PSE, color=colors[i], linestyle='--', label=f'PSE = {PSE:.2f} ± {PSE_err:.2f}')
    plt.xlabel('Design')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.show()


#%% #histogram

def mean_histogram_with_error(data_list, bins):
    histograms = [np.histogram(data, bins=bins)[0] for data in data_list]  # Obtener frecuencias por participante
    histograms = np.array(histograms)
    mean_counts = np.mean(histograms, axis=0)
    std_errors = np.std(histograms, axis=0, ddof=1) / np.sqrt(len(data_list))  # Error estándar
    return mean_counts, std_errors
bins=np.array(range(0,7*10,7))

mean_counts_NI, error_NI = mean_histogram_with_error(desgins_mses_ado_ni, bins)
mean_counts_Gaussian, error_Gaussian = mean_histogram_with_error(desgins_mses_ado, bins)
mean_counts_Random, error_Random = mean_histogram_with_error(desgins_mses_ran, bins)

width = (bins[1] - bins[0]) * 0.2

plt.figure(figsize=(8, 6))
x = bins[:-1]

plt.bar(x - width, mean_counts_NI, yerr=error_NI, width=width, color='blue', 
        label='ADO (NI Prior)', alpha=0.7, edgecolor='k', capsize=5)
plt.bar(x, mean_counts_Gaussian, yerr=error_Gaussian, width=width, color='green', 
        label='ADO (Gaussian Prior)', alpha=0.7, edgecolor='k', capsize=5)
plt.bar(x + width, mean_counts_Random, yerr=error_Random, width=width, color='red', 
        label='Pseudorandom', alpha=0.7, edgecolor='k', capsize=5)

plt.xticks(d,fontsize=12)
plt.legend(fontsize=12)
plt.xlabel('Design : Temporal gap (ms)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.tight_layout()
plt.show()
