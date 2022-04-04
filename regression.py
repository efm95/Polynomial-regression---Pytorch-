import numpy as np 
import torch 
import matplotlib.pyplot as plt
from torch.cuda import random

import torch.nn as nn 
import torch.optim as optim


def p(x):
    out = (x ** 3)+ 2*(x ** 2)-(4*x) - 8
    return out
    
    
def create_dataset(w_star,x_range ,sample_size, sigma, seed=None):
    """
    Documentation: 
        w_star      = array of true weights 
        x_range     = range within the data should be sampled
        sample_size = size (# rows) of the final sample  
        sigma       = standard deviation for noise (noramlly distributed) on y sampling, the higher, the more disperse the y's are 
        seed        = setting the seed for reprudicitility 
    """
    random_state = np.random.RandomState(seed)

    x = random_state.uniform(x_range[0],x_range[1],(sample_size))
    X = np.zeros((sample_size,w_star.shape[0]))

    for i in range(sample_size):
        X[i,0] = 1
        for j in range(1,w_star.shape[0]):
            X[i,j] = x[i]**j

    y = X.dot(w_star)
    if sigma >0:
        y+=random_state.normal(0.0,sigma,sample_size)
    
    return X, y
