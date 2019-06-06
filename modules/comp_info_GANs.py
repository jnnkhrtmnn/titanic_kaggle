# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:36:54 2019

@author: JannikHartmann
"""

# first; make up some training data

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random

# Data params; for creating some dummy "real" data

#mu = [0,0]
#cov = [[1,-0.99],
#       [-0.99,1]]

#n = 10000
# X should then contain the acutal "real" data
#X = np.random.multivariate_normal(mean=mu, cov=cov, size=(n))
#Xx = np.random.dirichlet(alpha=[0.2,0.2], size=n)


import os
wd = "C:/Users/JannikHartmann/Desktop/ECB DG-S SAT"

# set directory to code rep
os.chdir(wd)

df_name = "GANs_X_y_features"

import pandas as pd
y_X = pd.read_csv(df_name, index_col=0)

y = y_X["y"]
X = y_X.drop(["y"], axis=1)

ind_x = y_X.index

X.index = pd.Index(range(len(X)))
X_cat = X.select_dtypes(include=[object])

onehotlabels = pd.get_dummies(X_cat, columns=["lgl_frm", "ecnmc_actvty"], prefix=["lgl_frm", "ecnmc_actvty"])

X = X.select_dtypes(include=['float64'])

X = X.join(onehotlabels)
X.index = ind_x.values

# select which category to use
X = X[y=='S124']

X_colnames = X.columns
X = X.as_matrix()

# ##### DATA: Target data and generator input data

# get a sample (with replacement) from real data
def get_d_sampler(n,x):
    return  torch.Tensor(random.choices(x,k=n))


# Generate some uniform dist fake data; also called noise prior;
# another common choice is plain gaussian noise; think about the support
# before and after transforming through G!!!!
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  

# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, hidden_size)
        self.map4 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        x = self.f(self.map3(x))
        return self.f(self.map4(x))


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))

def extract(v):
    return v.data.storage().tolist()


def train():
    # Model parameters
    g_input_size = 3698      # Random noise dimension coming into generator, per output vector
    g_hidden_size = 5000    # Generator complexity
    g_output_size = 3698     # Size of generated output vector; has to match d_input_size

    d_input_size = 3698 # dimensions of training data    
    d_hidden_size = 100    # Discriminator complexity
    d_output_size = 1     # Single dimension for 'real' vs. 'fake' classification
    
    minibatch_size = 200 #number of samples to draw

    d_learning_rate = 2e-2
    g_learning_rate = 2e-2
    sgd_momentum = 0.9

    num_epochs = 500
    d_burnin = 100
    print_interval = 10
    #d_steps = 10
    #g_steps = 5

    dfe, dre, ge = 0, 0, 0
    d_real_data, d_fake_data, g_fake_data = None, None, None
    d_real_error_series = [None] * num_epochs
    d_fake_error_series = [None] * num_epochs
    g_error_series = [None] * num_epochs

    discriminator_activation_function = torch.sigmoid
    generator_activation_function = torch.sigmoid #tanh #relu
    
    gi_sampler = get_generator_input_sampler()
    
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
    
    D = Discriminator(input_size=d_input_size,
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)
    
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

    for epoch in range(num_epochs):
        #for d_index in range(d_steps):
        if (epoch==0):    
            # get generator input from uni dist in correct sizes
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            
            for burnin_index in range(d_burnin):
                
                D.zero_grad()
                d_real_data = Variable(get_d_sampler(n=minibatch_size, x=X))
                d_real_decision = D(d_real_data)
                d_real_error = criterion(d_real_decision, Variable(torch.ones([1,minibatch_size])).t())
                d_real_error.backward()
                d_fake_data = G(d_gen_input).detach()  
                d_fake_decision = D(d_fake_data)
                d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,minibatch_size])).t())  
                d_fake_error.backward()
                d_optimizer.step()     
                dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        else:
            
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real; sample REAL data here
            d_real_data = Variable(get_d_sampler(n=minibatch_size, x=X))
            # decide on probs of being real/fake
            d_real_decision = D(d_real_data)
            # calculate error, ones = real
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1,minibatch_size])).t())
            # compute/store gradients, but don't change params
            d_real_error.backward()

            #  1B: Train D on fake
            # get generator input from uni dist in correct sizes
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            # transform random input by function G
            # detach to avoid training G on these labels
            d_fake_data = G(d_gen_input).detach()  
            d_fake_decision = D(d_fake_data)
            # zeros = fake
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1,minibatch_size])).t())  
            d_fake_error.backward()
            # Only optimizes D's parameters; changes based on stored gradients from backward()
            d_optimizer.step()     

            # d real error and d fake error
            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]
            
            d_real_error_series[epoch] = dre
            d_fake_error_series[epoch] = dfe

        #for g_index in range(g_steps):
            # 2. Train G on D's response
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            # Train G to pretend it's genuine
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1,minibatch_size])).t())  

            g_error.backward()
            # Only optimizes G's parameters
            g_optimizer.step()  
            # extract generative error
            ge = extract(g_error)[0]
            
            g_error_series[epoch] = ge

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err) " % (epoch, dre, dfe, ge))
            
    values = g_fake_data.detach().numpy()
    #g_cov = np.cov(values[:,0],values[:,1])
    #print(" G-covariance: %s" % (g_cov))
    #g_mean = [np.mean(values[:,0]), np.mean(values[:,1])]
    #print(" mean: %s" % (g_mean))
    
    from matplotlib import pyplot
    pyplot.figure(figsize=(14,7))
    pyplot.plot(d_real_error_series)
    pyplot.plot(d_fake_error_series)
    pyplot.plot(g_error_series)

    pyplot.gca().legend(('d_real_error','d_fake_error', 'g_error'))
    pyplot.show()    
    
    return values
    
    
###############################################################################


fakes = train()

# binary cross entropy loss; helps for interpeting errors
# y = 1
# p =0.8
# cel = -(y*np.log(p)+(1-y)*np.log(1-p))
# cel
     
# next for validation: get tfidf matrix words; compare non-zero entries
# in faked data to corresponding tokens and see, if it makes sense

fakes_df = pd.DataFrame(fakes)
fakes_df.columns = X_colnames

# put zero where values under threshold
threshold  = 0.01
fakes_df[fakes_df < threshold] = 0

# get token lists where fake data is above threshold
fakes_df_token_lists = [None] * fakes_df.shape[0]
for i in range(fakes_df.shape[0]):
    x = (fakes_df.loc[i,:] != 0)
    x = x[x==True]
    fakes_df_token_lists[i] = list(x.index)

# see examples
print(fakes_df_token_lists[0])
# --> issue: having multiple lgl_frm or ecnmc_actvty codes does not make sense






    
    
