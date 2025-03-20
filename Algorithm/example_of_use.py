# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 19:57:03 2025

@author: C_Gab
"""

# Step by step tutorial for using the ado algorithm with a gaussian prior in your own experiment.

# step 1 : import the Experiment and Prior classes.
from priors import *
from ADO_Experiment import *

# step 2: generate an instance of the experiment.
exp= Experiment()

# step 3: set the prior , if non informative prior is needed, skip this step.
prior = Prior() 
p = prior.set_prior_gauss(expan,D,k) # calculate the prior based on gaussian curves, to use a non informative prior coment this line
exp.set_prior(p)

# step 4: define your discretization and generate the model , set the n and k values appropiated for your experiment.
n=10 # number of design points of your experiment
k=100 # number of likelihood bins
exp.generate(n, k)

# step 5 : the algorithm selects a design point 'd' to meassure.
d=exp.ADOchoose()

# step 6 : meassure in this design point, and update the model using the result of the meassure (if the experiment has binary results y=1 or y=0).
y=1 # or may be y=0 
exp.update(d,y)

# step 7 : iterate step 5 and 6 as many times as neccesary. For a new experiment (for example a new participant repeat the step 4).


    
    
