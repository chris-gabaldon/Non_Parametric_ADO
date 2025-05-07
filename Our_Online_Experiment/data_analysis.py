#%% All necessary third-party libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ADO_Experiment import *
from priors import*

#%% Loading the experimental data into dataframes

df_session = pd.read_csv('df_session_final.csv')       # Dataframe containing information about each participant's session.
df_trial = pd.read_csv('df_trial_final.csv')           # Dataframe containing information about each trial.

#%% Data Filtering

# Filter participants with over 249 responses in df_trial.
participant_counts = df_trial['participant_id'].value_counts()
valid_participants = participant_counts[participant_counts >= 250].index

# Filter valid participants in df_session and df_trial.
valid_sessions = df_session[df_session['participant_id'].isin(valid_participants)]
valid_trials = df_trial[df_trial['participant_id'].isin(valid_participants)]
valid_trials = valid_trials[['trial_id', 'd1', 'response', 'participant_id', 'ado_type','stimulus']]
valid_trials.loc[:, 'response'] = valid_trials['response'].apply(lambda x: 0 if x == '<' else 1)    # Updating the response column: if the value is '<', it is replaced with 0; otherwise, it is replaced with 1.
valid_trials = valid_trials.groupby('participant_id').tail(250)   # For participants that restarted the experiment.

# Filter by prior_type
gaussian_sessions = valid_sessions[valid_sessions['prior_type'] == 1] # prior type is the variable that link the experiment type : 1=> gaussian prior , 2=> non_imformative prior
ni_sessions = valid_sessions[valid_sessions['prior_type'] == 2]

# Function to shift d1, set the first value to '4' and delete the last row, this is beacouse the first trial was always 4 but not computed in the online experiment
def modify_d1(df):
    df['d1'] = df['d1'].shift(1)                                  # Move all rows in d1 one position down.
    df.iloc[0, df.columns.get_loc('d1')] = 4                      # Set the first value of d1 to 4.
    return df.iloc[:-1].reset_index(drop=True)                    # Delete the last row and reset the index

def filter_and_modify_trials(df):
    df = df.groupby('participant_id').apply(modify_d1).reset_index(drop=True)
    return df

valid_trials = filter_and_modify_trials(valid_trials)

# Get participant_id from ni_sessions and gaussian_sessions.
ni_participants = ni_sessions['participant_id'].unique()
g_participants = gaussian_sessions['participant_id'].unique()

'''
Function to filter participants based on their ado_type and 
their presence in a list of participant_ids. 
'''
def process_trials(df, participants, ado_type):
    trials = df[df['participant_id'].isin(participants) & (df['ado_type'] == ado_type)]
    return trials

# Using the function for ni_trials and g_trials with ado_type '10'.
ni_trials = process_trials(valid_trials, ni_participants, '10')
participant_counts_ni = ni_trials['participant_id'].value_counts()
ado_ni_participants = ni_trials['participant_id'].unique()

g_trials = process_trials(valid_trials, g_participants, '10')
ado_g_participants = g_trials['participant_id'].unique()
participant_counts_g = g_trials['participant_id'].value_counts()

# Creating and modifying random_trials.
random_trials_ni = valid_trials[valid_trials['participant_id'].isin(ni_participants) & (valid_trials['ado_type'] == '20')]
random_trials_g = valid_trials[valid_trials['participant_id'].isin(g_participants) & (valid_trials['ado_type'] == '20')]
random_trials = pd.concat([random_trials_ni, random_trials_g])
random_participants = random_trials['participant_id'].unique()

# Construction of data_subj and average ground truth curves.
data_subj = valid_trials.groupby('participant_id').apply(lambda x: [x['response'].values, x['d1'].values,x['participant_id'].values]).tolist()
num_subj = len(data_subj)

for i in range(num_subj):
    for j, valor in enumerate(data_subj[i][0]):
        indx = np.where(data_subj[i][1] < 0)[0]
        if j in indx:
            data_subj[i][0][j] = 1 - valor

d = np.sort(np.array(list(set(data_subj[0][1]))))
d_plotted = [20,25, 35, 41, 44, 47, 53, 56, 59, 65, 75,80]
list_probs_per_subject = []

# Iterating over all valid subjects to construct the Ground Truth Curve.
for i in range(num_subj):
    parcial_list = []
    list_probs_per_subject.append(parcial_list)
    for j in range(len(d)):
        ind = np.where(data_subj[i][1] == d[j])[0]
        results = [data_subj[i][0][k] for k in ind]
        ratio = np.sum(results) / (len(ind) + eps)
        parcial_list.append(ratio)

# Average values and curve plotting.
list_probs = np.mean(np.array(list_probs_per_subject), axis=0)
stds_subjects=np.std(np.array(list_probs_per_subject), axis=0)/np.sqrt(num_subj)
plt.errorbar(d_plotted, 1-list_probs, yerr=stds_subjects,label = 'Mean across all subjects', color = 'orangered')
plt.xlabel('Design: List average', fontsize=12)
plt.xticks(d_plotted)
plt.ylabel('Probability of response $avg$ < $50$', fontsize=12)
plt.grid()
plt.legend()
plt.show()

        
#%% Frequentist ground truth calculation

'''
These lists will store information from each participant
according to their experimental group, in the following 
order: answer (0 -less- or 1 -greater-), the design used 
and the participant_id (for each trial).
'''

data_subj_g = []            # Group 1: gaussian prior.
data_subj_ni = []           # Group 2: non-informative prior.
data_subj_random = []       # Group 3: random.

# Distribute data_subj in the appropriate lists.
for subj in range(len(data_subj)):
    participant_id = data_subj[subj][2][0]     
    if participant_id in ado_ni_participants:
        data_subj_ni.append(data_subj[subj])
    elif participant_id in ado_g_participants:
        data_subj_g.append(data_subj[subj])
    elif participant_id in random_participants:
        data_subj_random.append(data_subj[subj])

# Printing the number of participants in each experimental group.
print("NI Trials:", len(data_subj_ni))
print("G Trials:", len(data_subj_g))
print("Random Trials:", len(data_subj_random))

'''
This function generates a list of probability curves for a 
list of subjects with the characteristics of the data_subj 
ones defined above.
'''
def generate_prob_per_subject_type(data_subj):
    d = np.sort(np.array(list(set(data_subj[0][1]))))
    num_subj=len(data_subj)
    list_probs_per_subject_type = []
    for i in range(num_subj):
        parcial_list = []
        list_probs_per_subject_type.append(parcial_list)
        for j in range(len(d)):
            ind = np.where(data_subj[i][1] == d[j])[0]
            results = [data_subj[i][0][k] for k in ind]
            ratio = np.sum(results) / (len(ind) + eps)
            parcial_list.append(ratio)
    return list_probs_per_subject_type

'''
These lists will store the lists of probability curves for
each experimental group.
'''
list_probs_per_subject_ni = generate_prob_per_subject_type(data_subj_ni)
list_probs_per_subject_g = generate_prob_per_subject_type(data_subj_g)
list_probs_per_subject_random = generate_prob_per_subject_type(data_subj_random)

# Calculating the mean of each list of probability curves.
means_ni = np.mean(np.array(list_probs_per_subject_ni), axis=0)
means_g = np.mean(np.array(list_probs_per_subject_g), axis=0)
means_random = np.mean(np.array(list_probs_per_subject_random), axis=0)



# Plotting the proportion of correct responses.

plt.plot(d_plotted,means_ni,label='ni')
plt.plot(d_plotted,means_g,label='g')
plt.plot(d_plotted,means_random,label='random')
plt.legend()
plt.show()

for participant in range(len(list_probs_per_subject_ni)):
    subj=data_subj_ni[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_ni[participant],label=f'NI-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment NI')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


for participant in range(len(list_probs_per_subject_g)):
    subj=data_subj_g[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_g[participant],label=f'G-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment Gaussian')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
for participant in range(len(list_probs_per_subject_random)):
    subj=data_subj_random[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_random[participant],label=f'R-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment random')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% Bayesian ground truth calculation

'''
The following function generates a list of probability curves, 
estimating trought the bayesian posterior heatmap, for a list of 
subjects with the characteristics of the data_subj ones 
defined above.
'''

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
            estimated_bay = exp.values[np.argmax(exp.p,axis=1)]
        estimated_bay = exp.values[np.argmax(exp.p,axis=1)]   
        list_probs_per_subject_type.append(estimated_bay)
    return list_probs_per_subject_type

'''
These lists will store the lists of Bayesian probability 
curves for each experimental group.
'''
list_probs_per_subject_bay_ni = generate_prob_per_subject_type_bay(data_subj_ni,1)
list_probs_per_subject_bay_g = generate_prob_per_subject_type_bay(data_subj_g,2)
list_probs_per_subject_bay_random = generate_prob_per_subject_type_bay(data_subj_random,1)
list_probs_per_subject_bay_random_g = generate_prob_per_subject_type_bay(data_subj_random,2)

# Calculating the mean of each list of Bayesian probability curves.
means_ni = np.mean(np.array(list_probs_per_subject_bay_ni), axis=0)
means_g = np.mean(np.array(list_probs_per_subject_bay_g), axis=0)
means_random = np.mean(np.array(list_probs_per_subject_bay_random), axis=0)

# Plotting the proportion of correct responses.

plt.plot(d_plotted,means_ni,label='ni')
plt.plot(d_plotted,means_g,label='g')
plt.plot(d_plotted,means_random,label='random')
plt.legend()

for participant in range(len(list_probs_per_subject_ni)):
    subj=data_subj_ni[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_ni[participant],label=f'NI-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


for participant in range(len(list_probs_per_subject_g)):
    subj=data_subj_g[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_g[participant],label=f'G-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

for participant in range(len(list_probs_per_subject_random)):
    subj=data_subj_random[participant][2][0]
    plt.plot(d_plotted,list_probs_per_subject_random[participant],label=f'R-{subj}')
plt.xlabel('d1')
plt.ylabel('Proportion Correct')
plt.title('Proportion Correct by Participant and Treatment')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% MSE analysis + heatmaps

'''
The following function calculates the MSEs of an 
experimental group as a function of trials using
both the Bayesian and frequentist inferences.
'''

def calculate_mse(data_subj, list_probs_fre,list_probs_bay ,d, prior_type):
    num_subj = len(data_subj)
    total_trials = 200
    eps = 1e-100,
    list_mses_ran_fre = []
    list_mses_ran_bay = []
    results_mses_ran = []
    desgins_mses_ran = []
    list_estimated_ran_fre = []
    list_estimated_ran_bay = []
    for suj in range(num_subj):
        for i in range(num_subj):
            for j, valor in enumerate(data_subj[i][0]):
                indx = np.where(data_subj[i][1]<0)[0]
                if j in indx:
                    data_subj[i][0][j] = 1-valor

        d = np.sort(np.array(list(set(data_subj[0][1]))))
        data_individual_subj = data_subj[suj]
        true_i_fre = np.array(list_probs_fre[suj])
        true_i_bay = np.array(list_probs_bay[suj])
        D,k=len(d),100
        exp=Experiment()  # Initialize an ADO experiment.
        
        if prior_type==2: # Prior type = 2 --->gaussian.
            exp.generate(D, k)
            prior = Prior() 
            p = prior.set_prior_gauss(d[-1],D,k) # Calculate the prior based on exponential curves, shown in Figure 1b.
            exp.set_prior(p)                     # Set the bins priors as the one calculated above.
        else:
            exp.generate(D, k)
            
        fre_ADO = np.zeros(len(d))
        
        design_ran_1 = []
        result_ran_1 = []
        mse_ran_fre_1 = []

        counter_2 = np.zeros(len(d))
        res_counter_2 = np.zeros(len(d))
        mse_ran_bay_1=[]

        
        for jj in range(total_trials):
            if jj >= len(data_individual_subj[1]):
                break
            d1 = list(d).index(data_individual_subj[1][jj])
            result = data_individual_subj[0][jj]
            counter_2[d1] += 1
            res_counter_2[d1] += result
            result_ran_1.append(result)
            design_ran_1.append(d[d1])
            fre_ADO[int(d1)] += result
            exp.update(d1,result)
            estimated_bay=exp.values[np.argmax(exp.p,axis=1)]
            estimated_fre = res_counter_2 / (counter_2 + eps)
            mse_f = (np.mean((estimated_fre - true_i_fre) ** 2))
            mse_b=(np.mean((estimated_bay - true_i_bay) ** 2))
            mse_ran_fre_1.append(mse_f)
            mse_ran_bay_1.append(mse_b)
            
        list_estimated_ran_bay.append(estimated_bay)
        list_estimated_ran_fre.append(estimated_fre)
        results_mses_ran.append(result_ran_1)
        desgins_mses_ran.append(design_ran_1)
        list_mses_ran_fre.append(mse_ran_fre_1)
        list_mses_ran_bay.append(mse_ran_bay_1)

    return list_mses_ran_fre,list_mses_ran_bay

# Calculating the MSE as a function of trials for each experimental group.
d = np.unique(np.concatenate([ni_trials['d1'], g_trials['d1'], random_trials['d1']]))
list_mses_ran_fre_ni,list_mses_ran_bay_ni = calculate_mse(data_subj_ni, list_probs_per_subject_ni,list_probs_per_subject_bay_ni, d,1) #ado_type1=ni , 2gaussian
list_mses_ran_fre_g ,list_mses_ran_bay_g= calculate_mse(data_subj_g, list_probs_per_subject_g,list_probs_per_subject_bay_g, d,2)
list_mses_ran_fre_random,list_mses_ran_bay_random= calculate_mse(data_subj_random, list_probs_per_subject_random,list_probs_per_subject_bay_random, d,1)
list_mses_ran_fre_random,list_mses_ran_bay_random_g= calculate_mse(data_subj_random, list_probs_per_subject_random,list_probs_per_subject_bay_random_g, d,2)

std_mses_random=np.std(list_mses_ran_fre_random,axis=0)/np.sqrt(len(list_mses_ran_fre_random))
std_mses_ni=np.std(list_mses_ran_fre_ni,axis=0)/np.sqrt(len(list_mses_ran_fre_ni))
std_mses_g=np.std(list_mses_ran_fre_g,axis=0)/np.sqrt(len(list_mses_ran_fre_g))

mean_tr_ni=np.log10(np.mean(list_mses_ran_fre_ni, axis=0))
mean_tr_g=np.log10( np.mean(list_mses_ran_fre_g, axis=0))
mean_tr_r=np.log10(np.mean(list_mses_ran_fre_random, axis=0))

rand_sup=np.abs(np.log10(np.mean(list_mses_ran_fre_random, axis=0)+std_mses_random)-mean_tr_r)
rand_inf=np.abs(-np.log10(np.mean(list_mses_ran_fre_random, axis=0)+std_mses_random)+mean_tr_r)

ni_sup=np.abs(np.log10(np.mean(list_mses_ran_fre_ni, axis=0)+std_mses_ni)-mean_tr_ni)
ni_inf=np.abs(-np.log10(np.mean(list_mses_ran_fre_ni, axis=0)+std_mses_ni)+mean_tr_ni)

g_sup=np.abs(np.log10(np.mean(list_mses_ran_fre_g, axis=0)+std_mses_g)-mean_tr_g)
g_inf=np.abs(-np.log10(np.mean(list_mses_ran_fre_g, axis=0)+std_mses_g)+mean_tr_g)




## bayesian estimation curves
std_mses_random_b=np.std(list_mses_ran_bay_random,axis=0)/np.sqrt(len(list_mses_ran_bay_random))
std_mses_random_b_g=np.std(list_mses_ran_bay_random_g,axis=0)/np.sqrt(len(list_mses_ran_bay_random_g))
std_mses_ni_b=np.std(list_mses_ran_bay_ni,axis=0)/np.sqrt(len(list_mses_ran_bay_ni))
std_mses_g_b=np.std(list_mses_ran_bay_g,axis=0)/np.sqrt(len(list_mses_ran_bay_g))

mean_tr_ni_b=np.log10(np.mean(list_mses_ran_bay_ni, axis=0))
mean_tr_g_b=np.log10( np.mean(list_mses_ran_bay_g, axis=0))
mean_tr_r_b=np.log10(np.mean(list_mses_ran_bay_random, axis=0))
mean_tr_r_b_g=np.log10(np.mean(list_mses_ran_bay_random_g, axis=0))

rand_sup_b=np.abs(np.log10(np.mean(list_mses_ran_bay_random, axis=0)+std_mses_random_b)-mean_tr_r_b)
rand_inf_b=np.abs(-np.log10(np.mean(list_mses_ran_bay_random, axis=0)+std_mses_random_b)+mean_tr_r_b)
rand_sup_b_g=np.abs(np.log10(np.mean(list_mses_ran_bay_random_g, axis=0)+std_mses_random_b_g)-mean_tr_r_b_g)
rand_inf_b_g=np.abs(-np.log10(np.mean(list_mses_ran_bay_random_g, axis=0)+std_mses_random_b_g)+mean_tr_r_b_g)


ni_sup_b=np.abs(np.log10(np.mean(list_mses_ran_bay_ni, axis=0)+std_mses_ni_b)-mean_tr_ni_b)
ni_inf_b=np.abs(-np.log10(np.mean(list_mses_ran_bay_ni, axis=0)+std_mses_ni_b)+mean_tr_ni_b)

g_sup_b=np.abs(np.log10(np.mean(list_mses_ran_bay_g, axis=0)+std_mses_g_b)-mean_tr_g_b)
g_inf_b=np.abs(-np.log10(np.mean(list_mses_ran_bay_g, axis=0)+std_mses_g_b)+mean_tr_g_b)




fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Subplot 1: Bayesian estimations
axes[0].set_title('Bayesian estimation', fontsize=14)
axes[0].errorbar(range(len(list_mses_ran_bay_ni[0])), np.log10(np.mean(list_mses_ran_bay_ni, axis=0)),
                  yerr=[ni_inf_b, ni_sup_b], label='ADO (NI prior) ', c='b')
axes[0].errorbar(range(len(list_mses_ran_bay_g[0])), np.log10(np.mean(list_mses_ran_bay_g, axis=0)),
                  yerr=[g_inf_b, g_sup_b], label='ADO (Gaussian prior) ', c='green')
axes[0].errorbar(range(len(list_mses_ran_bay_random[0])), np.log10(np.mean(list_mses_ran_bay_random, axis=0)),
                  yerr=[rand_inf_b, rand_sup_b], label='Random  (NI prior)', c='r')
axes[0].errorbar(range(len(list_mses_ran_bay_random_g[0])), np.log10(np.mean(list_mses_ran_bay_random_g, axis=0)),
                  yerr=[rand_inf_b_g, rand_sup_b_g], label='Random (Gaussian prior)', c='m')
axes[0].set_xlabel('Trials', fontsize=14)
axes[0].set_ylabel('log(MSE)', fontsize=14)
axes[0].tick_params(axis='both', labelsize=14)
axes[0].legend(fontsize=12)

# Subplot 2: Frequentist estimations
axes[1].set_title('Frequentist estimation', fontsize=14)
axes[1].errorbar(range(len(list_mses_ran_fre_ni[0])), np.log10(np.mean(list_mses_ran_fre_ni, axis=0)),
                  yerr=[ni_inf, ni_sup], label='ADO (NI prior)', c='b')
axes[1].errorbar(range(len(list_mses_ran_fre_g[0])), np.log10(np.mean(list_mses_ran_fre_g, axis=0)),
                  yerr=[g_inf, g_sup], label='ADO (Gaussian prior)', c='g')
axes[1].errorbar(range(len(list_mses_ran_fre_random[0])), np.log10(np.mean(list_mses_ran_fre_random, axis=0)),
                  yerr=[rand_inf, rand_sup], label='Random', c='r')
axes[1].set_xlabel('Trials', fontsize=14)
axes[1].tick_params(axis='both', labelsize=14)
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.show()


#%%histogram

def plot_grouped_histograms_absolute(df_list, labels, colors, d_plotted):
    bins = np.arange(len(d_plotted) + 1) - 0.5  
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (df, label, color) in enumerate(zip(df_list, labels, colors)):
        participants = df['participant_id'].unique()
        N = len(participants)
        histograms = []

        for participant in participants:
            values = df[df['participant_id'] == participant]['d1'].astype(int)
            hist, _ = np.histogram(values, bins=bins)
            histograms.append(hist)

        histograms = np.array(histograms)
        mean_hist = np.mean(histograms, axis=0)
        std_error = np.std(histograms, axis=0, ddof=1) / np.sqrt(N)

        offset = (i - 1) * width
        positions = np.arange(len(d_plotted)) + offset

        ax.bar(positions, mean_hist, yerr=std_error, width=width,
               label=f'{label} (N={N})', color=color, alpha=0.8,
               edgecolor='black', capsize=5)

    ax.set_xlabel('Design : Average of the list', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xticks(np.arange(len(d_plotted)))
    ax.set_xticklabels(d_plotted, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


plot_grouped_histograms_absolute(
    df_list=[ni_trials, g_trials, random_trials],
    labels=['ADO (NI Prior)', 'ADO (Gaussian Prior)', 'Random'],
    colors=['blue', 'green', 'red'],d_plotted=d_plotted)














