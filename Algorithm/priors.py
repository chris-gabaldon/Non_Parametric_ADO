import math
import random   
import numpy as np
import scipy.stats
import random

# this files defines a class called Prior that you can use to create 
# priors based on exponential, power law, quadratic, logistic or gaussian models

class Prior:
    def __init__(self):
        self.n = 10
        self.k = 100

    def set_prior_exp(self,expan,n,k): # create a prior based on exponential curves
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 4*400 # this  number is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these numbers be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        a_grid = np.linspace(0, 1, ab_n)
        dist_a = scipy.stats.beta(2, 1)
        a_prob = dist_a.pdf(a_grid)/ab_n
        
        b_grid = np.linspace(0, 1, ab_n)
        dist_b = scipy.stats.beta(1, 80)
        b_prob = dist_b.pdf(b_grid)/ab_n
        for i in range(ab_n):
            for ii in range(ab_n):
                true = (k-1) - (k-1)*  ( a_grid[i]* np.exp(-b_grid[ii]*aux ) )                                   
                for iii in range(n):
                    p[iii,round(true[iii])] += a_prob[i]*b_prob[ii]
        
        p = n * p/p.sum()
        
        return p
    
    
    def set_prior_pow(self,expan,n,k): # create a prior based on power law curves
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 400# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        a_grid = np.linspace(0, 1, ab_n)
        dist_a = scipy.stats.beta(2, 1)
        a_prob = dist_a.pdf(a_grid)/ab_n
        
        b_grid = np.linspace(0, 1, ab_n)
        dist_b = scipy.stats.beta(1, 4)
        b_prob = dist_b.pdf(b_grid)/ab_n
        for i in range(ab_n):
            for ii in range(ab_n):
                true = (k-1) - (k-1)*(a_grid[i]*(aux + 1)**-b_grid[ii])
                for iii in range(n):
                    p[iii,round(true[iii])] += a_prob[i]*b_prob[ii]
        
        p = n * p/p.sum()
        
        return p      

    def set_prior_logis(self,expan,n,k): # create a prior based on logistic curves
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 40000# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        for i in range(ab_n):
            true = (k-1)/(1 + np.exp(-0.1*random.random()*(aux-100 +  200*random.random())))
            for iii in range(n):
                p[iii,round(true[iii])] += 1/ab_n        
        p = n * p/p.sum()
        
        return p  
    
    
    def set_prior_cuadra(self,expan,n,k): # create a prior based on quadratic curves
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 40000# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        for i in range(ab_n):
            # true = (k-1)*random.random() + (k-1)*random.random()*(aux/expan)**2
            true = (k-1)*random.random() + (k-1)*random.random()*(2*aux/expan)**2
            true[true>(k-1)] = k-1
            for iii in range(n):
                p[iii,round(true[iii])] += 1/ab_n        
        p = n * p/p.sum()
        
        return p  
    
    
    def set_prior_gauss(self,expan,n,k): # create a prior based gaussian distributions at each design point
        p = np.zeros((n,k))
        aux = np.linspace(0, expan, n)
        means = k * (1 - aux/expan)
        x = np.linspace(1, k, k)
        
        for i in range(n):
            sigma = k/5
            p[i,:] = 1./((sigma)* math.sqrt(2*math.pi))*np.exp(-(0.5/((sigma)**2))*np.power((x - means[i]), 2.))
            # p[i,:] = gaussian(x, k/10, means[i])
            p[i,:] = np.flip(p[i,:]/p[i,:].sum())
        
        return p 
    
    
