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
            p = prior.set_prior_cuadra(d[-1],D,k) # Calculate the prior based on exponential curves, shown in Figure 1b.
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
list_mses_ado_bay=[]
list_mses_ran_fre=[]
list_mses_ran_bay=[]
results_mses_ado=[]
results_mses_ran=[]
desgins_mses_ado=[]
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
        n_first_loops=2
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
# #cuadratic prior (same analisys but diferent prior for the ADO algorithm)
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
    p = prior.set_prior_cuadra(expan,D,k) # generate a gaussian prior
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
        n_first_loops=2
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
list_mses_ado_bay_ran_gauss=[list_mses_ran_bay_gauss[i][:min_subj_gauss_bay] for i in range(num_subj)]

# logarithmic transformation for the means and error bars

std_mses_random_fre=np.std(list_mses_ado_fre_ran,axis=0)/np.sqrt(len(list_mses_ado_fre_ran))
std_mses_ni_fre=np.std(list_mses_ado_fre_ni,axis=0)/np.sqrt(len(list_mses_ado_fre_ni))
std_mses_g_fre=np.std(list_mses_ado_fre_g,axis=0)/np.sqrt(len(list_mses_ado_fre_g))

std_mses_random_bay=np.std(list_mses_ado_bay_ran,axis=0)/np.sqrt(len(list_mses_ado_bay_ran))
std_mses_random_bay_gauss=np.std(list_mses_ado_bay_ran_gauss,axis=0)/np.sqrt(len(list_mses_ado_bay_ran_gauss))
std_mses_ni_bay=np.std(list_mses_ado_bay_ni,axis=0)/np.sqrt(len(list_mses_ado_bay_ni))
std_mses_g_bay=np.std(list_mses_ado_bay_g,axis=0)/np.sqrt(len(list_mses_ado_bay_g))

mean_tr_ni_fre=np.log10(np.mean(list_mses_ado_fre_ni, axis=0))
mean_tr_g_fre=np.log10( np.mean(list_mses_ado_fre_g, axis=0))
mean_tr_r_fre=np.log10(np.mean(list_mses_ado_fre_ran, axis=0))

mean_tr_ni_bay=np.log10(np.mean(list_mses_ado_bay_ni, axis=0))
mean_tr_g_bay=np.log10( np.mean(list_mses_ado_bay_g, axis=0))
mean_tr_r_bay=np.log10(np.mean(list_mses_ado_bay_ran, axis=0))
mean_tr_r_bay_gauss=np.log10(np.mean(list_mses_ado_bay_ran_gauss, axis=0))

rand_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random_fre)-mean_tr_r_fre)
rand_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_ran, axis=0)+std_mses_random_fre)+mean_tr_r_fre)

rand_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_ran, axis=0)+std_mses_random_bay)-mean_tr_r_bay)
rand_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_ran, axis=0)+std_mses_random_bay)+mean_tr_r_bay)
rand_sup_bay_gauss=np.abs(np.log10(np.mean(list_mses_ado_bay_ran_gauss, axis=0)+std_mses_random_bay_gauss)-mean_tr_r_bay_gauss)
rand_inf_bay_gauss=np.abs(-np.log10(np.mean(list_mses_ado_bay_ran_gauss, axis=0)+std_mses_random_bay_gauss)+mean_tr_r_bay_gauss)

ni_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni_fre)-mean_tr_ni_fre)
ni_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_ni, axis=0)+std_mses_ni_fre)+mean_tr_ni_fre)

ni_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_ni, axis=0)+std_mses_ni_bay)-mean_tr_ni_bay)
ni_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_ni, axis=0)+std_mses_ni_bay)+mean_tr_ni_bay)

g_sup_fre=np.abs(np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g_fre)-mean_tr_g_fre)
g_inf_fre=np.abs(-np.log10(np.mean(list_mses_ado_fre_g, axis=0)+std_mses_g_fre)+mean_tr_g_fre)

g_sup_bay=np.abs(np.log10(np.mean(list_mses_ado_bay_g, axis=0)+std_mses_g_bay)-mean_tr_g_bay)
g_inf_bay=np.abs(-np.log10(np.mean(list_mses_ado_bay_g, axis=0)+std_mses_g_bay)+mean_tr_g_bay)




fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Subplot 1: Frequentist inference
axes[1].set_title('Frequentist inference', fontsize=14)
axes[1].errorbar(range(len(list_mses_ado_fre_ni[0])), np.log10(np.mean(list_mses_ado_fre_ni, axis=0)), 
                 yerr=[ni_inf_fre, ni_sup_fre], label='ADO (NI prior) ', c='b')
axes[1].errorbar(range(len(list_mses_ado_fre_g[0])), np.log10(np.mean(list_mses_ado_fre_g, axis=0)), 
                 yerr=[g_inf_fre, g_sup_fre], label='ADO (Quadratic prior)', c='g')
axes[1].errorbar(range(len(list_mses_ado_fre_ran[0])), np.log10(np.mean(list_mses_ado_fre_ran, axis=0)), 
                 yerr=[rand_inf_fre, rand_sup_fre], label='Pseudorandom', c='r')
axes[1].set_xlabel('Trials', fontsize=14)
axes[0].set_ylabel('log(MSE)', fontsize=14)
axes[1].tick_params(axis='both', labelsize=14)
axes[1].legend(fontsize=12)

# Subplot 2: Bayesian inference
axes[0].set_title('Bayesian inference', fontsize=14)
axes[0].errorbar(range(len(list_mses_ado_bay_ni[0])), np.log10(np.mean(list_mses_ado_bay_ni, axis=0)), 
                 yerr=[ni_inf_bay, ni_sup_bay], label='ADO (NI prior) ', c='b')
axes[0].errorbar(range(len(list_mses_ado_bay_g[0])), np.log10(np.mean(list_mses_ado_bay_g, axis=0)), 
                 yerr=[g_inf_bay, g_sup_bay], label='ADO (Cuadra prior)', c='g')
axes[0].errorbar(range(len(list_mses_ado_bay_ran[0])), np.log10(np.mean(list_mses_ado_bay_ran, axis=0)), 
                 yerr=[rand_inf_bay, rand_sup_bay], label='Pseudorandom (NI prior)', c='r')
axes[0].errorbar(range(len(list_mses_ado_bay_ran_gauss[0])), np.log10(np.mean(list_mses_ado_bay_ran_gauss, axis=0)), 
                 yerr=[rand_inf_bay_gauss, rand_sup_bay_gauss], label='Pseudorandom (Quadratic prior)', c='m')
axes[0].set_xlabel('Trials', fontsize=14)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].legend(fontsize=12)

plt.tight_layout()
plt.show()

