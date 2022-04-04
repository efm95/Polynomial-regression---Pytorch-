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

def poly_regr(X_train,y_train,X_val,y_val,learning_rate,num_steps,degree=3):

    """
    Documentation:
        X             = N*p array (or N*(p+1) if bias set to TRUE) - X training
        y             = 1-D array - y training 
        X_val         = N*p array (or N*(p+1) if bias set to TRUE) - X validation 
        y_val         = 1-D array - y validation 
        learning_rate = int - learning rate for stochastic gradient descent
        num_steps     = int - number of iterations
        degree        = polynomial degree
    """

    #num_samples = X_train.shape[0]

    DEVICE = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

    model = nn.Linear(degree+1,1,bias=False)
    model.to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    
    X = X_train.reshape(X_train.shape[0], degree+1) # shape expected by nn.Linear
    X = torch.from_numpy(X) # convert to torch.tensor
    X = X.float() # convert to float32 (from numpy double).
    X = X.to(DEVICE) # copy data to GPU.
    y = torch.from_numpy(y_train.reshape((y_train.shape[0], 1))).float().to(DEVICE)

    X_val = torch.from_numpy(X_val.reshape((X_val.shape[0], degree+1))).float().to(DEVICE)
    y_val = torch.from_numpy(y_val.reshape((y_val.shape[0], 1))).float().to(DEVICE)

    loss_vec = np.zeros(num_steps)
    val_vec = np.zeros(num_steps)

    for step in range(num_steps):
    
        model.train() # systematic: put model in 'training' mode.
        optimizer.zero_grad() # systematic: start step w/ zero gradient.
    
        y_ = model(X) # do prediction using the current model.
        loss = loss_fn(y_, y) # compute error.
        loss_vec[step] = loss
        print(f"Step {step}: train loss: {loss}") # running train loss

        loss.backward() # compute gradients.
        optimizer.step() # update parameters
    
        # Eval on validation set
        model.eval() # systematic: put model in 'eval' mode.
        with torch.no_grad():
            y_ = model(X_val)
            val_loss = loss_fn(y_, y_val)
            val_vec[step] = val_loss
        print(f"Step {step}: val loss: {val_loss}")
    
    return model.weight, loss_vec, val_vec
