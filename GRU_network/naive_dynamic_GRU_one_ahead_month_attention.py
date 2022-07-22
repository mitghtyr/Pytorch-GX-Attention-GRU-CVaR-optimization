######### Naive-GRU  ###############
## exe with [nohup python -u GRU_network/naive_dynamic_GRU_one_ahead_month_attention.py>data/gru_save/naive_print_one_ahead_month_attn.log >&1 &] in env
## file_to_read_in: "data/daily_return_ratio.csv"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import os
import math
import random
from datetime import datetime


import torch
import itertools
import numbers
import torch.utils.data as utils
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal as Mvnorm
from torch.utils.data import Dataset

from scipy.stats import mvn


import time

np.random.seed(817)
torch.manual_seed(817)
random.seed(817)

Dow_file="data/daily_return_ratio.csv"
df=pd.read_csv(Dow_file)


data = torch.tensor(df.iloc[:,1:30].transpose().values.astype("float32"))
datets = pd.to_datetime(df.Date)

[nrow,ncol] = data.shape


def create_multi_ahead_samples_roll_single_month(ts, lookBack, lookAhead=1):
    # standardization
    mu = torch.mean(ts)
    sigma = torch.sqrt(torch.var(ts))

    ts1 = (ts-mu)/sigma
   
    interval = 21

    dataX, dataY = [], []
    ncol = np.int(np.floor((ts.shape[0]-lookBack-interval)/lookAhead))
    
    
    for i in range(ncol):
        
        history_seq = ts1[(i*lookAhead): (i*lookAhead + lookBack)]
        # future monthly return
        future_seq = torch.prod(ts[(i*lookAhead + lookBack): (interval+(i)*lookAhead + lookBack)]+1)-1
        
        # daily return as input
        history_seq = torch.stack([history_seq,
                             (history_seq-torch.mean(history_seq))*(history_seq-torch.mean(history_seq))])
        
        
        
        dataX.append(history_seq)
        dataY.append(future_seq)
    
    dataX=torch.stack(dataX)
    dataY=torch.stack(dataY)
    mu = torch.mean(dataY)
    sigma = torch.sqrt(torch.var(dataY))
    dataY = (dataY-mu)/sigma
    
    return dataX, dataY,mu,sigma



lookBack=200
lookAhead=1

# create tensor for the Market factor, daily input X and monthly return Y
dataX_M,dataY_M,mu_M,sigma_M = create_multi_ahead_samples_roll_single_month(data[0,:], lookBack, lookAhead) 

# (train+valid) : test=(0.48+0.12) : 0.4
rate1 = 0.6 
rate2 = 0.8

train_size = int((dataY_M.shape[0]) * rate1 * rate2)
valid_size = int((dataY_M.shape[0]) * rate1*(1-rate2))
test_size = dataY_M.shape[0] - train_size-valid_size

interval = 21
# test set: evaluation per 21 days (per month)
jump_ind = torch.tensor(np.arange(1,np.int(np.floor(test_size/interval))+1))
jump_ind = train_size+valid_size+jump_ind*interval
    
#### validation separation: time-order or random split
# random-select valid : 1;
# time-seq valid: 0;
random_v = 0
if random_v==1:
    #### random-select valid
    ind_M = torch.randperm(train_size+valid_size)
        
else:
    ## time-seq valid
    ind_M = torch.arange(train_size+valid_size)
    
    
trainX_M, trainY_M = dataX_M[ind_M[0:train_size],:,:],  dataY_M[ind_M[0:train_size]]
validX_M, validY_M = dataX_M[ind_M[train_size:train_size+valid_size],:,:],  dataY_M[ind_M[train_size:train_size+valid_size]]
testX_M, testY_M = dataX_M[jump_ind,:,:],  dataY_M[jump_ind]

# learning rate
lr = 1e-3
# compare with the best model every 30 steps
steps= 30
K=28
lookBack=200
lookAhead=1

from torch.distributions import Normal
class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim,lookAhead, layerNum):

        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lookAhead = lookAhead
        self.layerNum = layerNum
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
        ## GRU cell
        self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                                num_layers=self.layerNum, dropout=0.0,
                                 batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim*self.lookAhead)
## Attention layer
class Attention(nn.Module):
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (1,lag,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,lag,H)
        :return
            attention energies in seq length (B,1,lag)
            
        denote lag as T
        '''
        max_len = encoder_outputs.size(1)
        hns = hidden.repeat(max_len,1,1).transpose(0,1) # B*T*H
        energy = F.tanh(self.attn(torch.cat([hns, encoder_outputs], 2))) # [B*T*H]+[B*T*H] ->[B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        attn_energies  = torch.bmm(v,energy).squeeze(1)  # [B*T]
        

        
        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax [B*1*T]
       
## Network for the market factor fitting: MNN
class GRUModel_M_month(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        super(GRUModel_M_month, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        self.attention = Attention(hiddenNum)
 
    def forward(self, x):

        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        energy = self.attention(hn, rnnOutput) # B*1*T
        context = energy.bmm(rnnOutput).squeeze(1)  # (B,1,T)*(B*T*H)->(B,1,H)->(B*H)
  
        
        fcOutput = self.fc(context)
        # ahpla_m is controled in the interval [-1,1]
        fcOutput[:,0:self.lookAhead] = 1*(2*self.sigmoid(fcOutput[:,0:self.lookAhead])-1)
        # alphai is controled in the interval [0,10]
        fcOutput[:,self.lookAhead:(2*self.lookAhead)] = (10*self.sigmoid(fcOutput[:,self.lookAhead:(2*self.lookAhead)]))
        
        
        
        return fcOutput
    
    def loss(self, x, y):
        batchSize = y.shape[0]
        # alpha_M,  beta_M,
        thetapred = x
        alpha_M, beta_M = torch.split(thetapred, [self.lookAhead,self.lookAhead], dim=1)
        
        # Z_M tilde
        ZM = (y[:].view(alpha_M.shape)-alpha_M)/beta_M

        ## loglikelihood
        m = Normal(torch.zeros(batchSize),torch.ones(batchSize))
        loglik = m.log_prob(ZM.reshape([-1,1]))
        loglik -= torch.log((beta_M).reshape([-1,1]))
        loss = -loglik

        return loss      
## Network for the individual stock fitting: SNN
class GRUModel_S_month(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        super(GRUModel_S_month, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        self.attention = Attention(hiddenNum)
 
    def forward(self, x):

        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))
        
        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        energy = self.attention(hn, rnnOutput) # B*1*T
        context = energy.bmm(rnnOutput).squeeze(1)  # (B,1,T)*(B*T*H)->(B,1,H)->(B*H)
  
        
        fcOutput = self.fc(context)
        # alphai is controled in the interval [-1,1]
        fcOutput[:,0:self.lookAhead] = 1*(2*self.sigmoid(fcOutput[:,0:self.lookAhead])-1)
        #  beta_i,gamma_i are controled in the interval [-10,10],[0,20]
        fcOutput[:,self.lookAhead:(2*self.lookAhead)] = 10*(2*self.sigmoid(fcOutput[:,self.lookAhead:(2*self.lookAhead)])-1)
        fcOutput[:,(2*self.lookAhead):(3*self.lookAhead)] = (10*self.sigmoid(fcOutput[:,(2*self.lookAhead):(3*self.lookAhead)]))

        return fcOutput
    
    def loss(self, x, y,ZM):
        batchSize = y.shape[0]
        #  alpha_i,  beta_i,gamma_i
        thetapred = x
        alpha_i, beta_i, gamma = torch.split(thetapred, [self.lookAhead, self.lookAhead,self.lookAhead], dim=1)

        # Z_i tilde
        Zi = (y[:].view(alpha_i.shape)-alpha_i-beta_i*ZM)/gamma

        ## loglikelihood
        m = Normal(torch.zeros(batchSize),torch.ones(batchSize))
        loglik = m.log_prob(Zi.reshape([-1,1]))
        loglik -= torch.log((gamma).reshape([-1,1]))
        loss = -loglik
        return loss          
    
def train_M(net,trainX,validX, trainY,validY, lr, hidden_num=64, epoch=20, max_epochs_stop=2,steps=5):

    trainX = torch.transpose(trainX,1,2)
    #trainY = torch.transpose(trainY,1,2)
    validX = torch.transpose(validX,1,2)
    #validY = torch.transpose(validY,1,2)
    
    trainloss = []
    validloss = []

    optimizer = optim.RMSprop(net.parameters(), lr=lr,momentum=0.2)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps= steps


    for i in range(epoch):
        # train step
        net = net.train()
        trainX, trainY = Variable(trainX), Variable(trainY)
            
        optimizer.zero_grad()
        thetapred= net.forward(trainX)

        losst = net.loss(thetapred,  trainY).mean()
        losst.backward()
        optimizer.step() 
        
        # valid step
        with torch.no_grad():
                # Set to evaluation mode
            net.eval()
            validX, validY = Variable(validX), Variable(validY)

            thetapred=net.forward(validX)
            lossv = net.loss(thetapred, validY).mean()

        if lossv.item() < valid_loss_min:
            torch.save(net.state_dict(), "data/gru_save/histmodel.pt")
            bias = np.abs(lossv.item()-valid_loss_min)
            valid_loss_min = lossv.item()
                
            # Track improvement
            epochs_no_improve = 0
            best_epoch = i

            #print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())

        if((i+1)%(accumulation_steps))==0:
            print("%d epoch is finished!" % (i+1)) 
            if (lossv.item() > valid_loss_min) | (bias<1e-5):
                epochs_no_improve += 1
                net.load_state_dict(torch.load("data/gru_save/histmodel.pt"))
                
                    # Trigger early stopping
                print("%d epoch falls in" % (i+1))
                
                print(
                    f'\nvalidloss  is: {lossv.item():.5f}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}. epoch without improvement is {epochs_no_improve}'
                       )
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                        )

                    net.optimizer = optimizer
                    return net, trainloss,validloss
            
                
    net.optimizer = optimizer
    return net,trainloss,validloss


def train_S(net,trainX,validX, trainY,validY, ZM_T,ZM_V, lr, hidden_num=64, epoch=20, max_epochs_stop=2,steps=5):

    trainX = torch.transpose(trainX,1,2)
    #trainY = torch.transpose(trainY,1,2)
    validX = torch.transpose(validX,1,2)
    #validY = torch.transpose(validY,1,2)
    
    trainloss = []
    validloss = []

    optimizer = optim.RMSprop(net.parameters(), lr=lr,momentum=0.2)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps= steps

    for i in range(epoch):
        # train step
        net = net.train()
        trainX, trainY = Variable(trainX), Variable(trainY)
            
        optimizer.zero_grad()
        thetapred= net.forward(trainX)

        losst = net.loss(thetapred,  trainY,ZM_T).mean()
        losst.backward()

        optimizer.step() 
        
        # valid step
        with torch.no_grad():
                # Set to evaluation mode
            net.eval()
            validX, validY = Variable(validX), Variable(validY)

            thetapred=net.forward(validX)
            lossv = net.loss(thetapred, validY,ZM_V).mean()

        if lossv.item() < valid_loss_min:
            torch.save(net.state_dict(), "data/gru_save/s_histmodel.pt")
            bias = np.abs(lossv.item()-valid_loss_min)
            valid_loss_min = lossv.item()
                
            # Track improvement
            epochs_no_improve = 0
            best_epoch_no_improve=0
            best_epoch = i

            #print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())
                
           
            
        if((i+1)%(accumulation_steps))==0:
            print("%d epoch is finished!" % (i+1)) 
            if (lossv.item() > valid_loss_min) | (bias<1e-5):
                epochs_no_improve += 1
                net.load_state_dict(torch.load("data/gru_save/s_histmodel.pt"))

                # Trigger early stopping
                print("%d epoch falls in" % (i+1))
                print(
                    f'\nvalidloss  is: {lossv.item():.5f}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}. epoch without improvement is {epochs_no_improve}'
                       )
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                        )

                    net.optimizer = optimizer
                    return net, trainloss,validloss
            
                
    net.optimizer = optimizer
    return net,trainloss,validloss

### Fit the Market GRU network
net = GRUModel_M_month(inputDim=2, hiddenNum=2*2*lookAhead, outputDim=2,lookAhead=lookAhead, layerNum=1)
net,trainloss,validloss=train_M(net,trainX_M,validX_M, trainY_M,validY_M, 
                                                lr, hidden_num=2*2*lookAhead, epoch=2500, max_epochs_stop=5,steps=20)

### output the monthly prediction for market volatility (beta_M) and alpha_M
testX_M = torch.transpose(testX_M,1,2)
testX_M,testY_M = Variable(testX_M), Variable(testY_M)
        
net.eval()   
paraM=net(testX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead,lookAhead], dim=1)
param_matrix_M_alpha_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0],1]))  
param_matrix_M_beta_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0],1]))  
param_matrix_M_alpha_step_one_ahead_month.iloc[:,0] = (mu_M+sigma_M*alpha_M).detach().numpy().reshape(-1,1)
param_matrix_M_beta_step_one_ahead_month.iloc[:,0] = (sigma_M*beta_M).detach().numpy().reshape(-1,1)

param_matrix_M_alpha_step_one_ahead_month.to_csv("data/gru_save/param_matrix_M_alpha_step_one_ahead_month_time_att.csv")
param_matrix_M_beta_step_one_ahead_month.to_csv("data/gru_save/param_matrix_M_beta_step_one_ahead_month_time_att.csv")

# Recover hidden Z_M
trainX_M = torch.transpose(trainX_M,1,2)
#trainY = torch.transpose(trainY,1,2)
validX_M = torch.transpose(validX_M,1,2)
trainX_M, trainY_M = Variable(trainX_M), Variable(trainY_M)
# ZM_t for training
paraM=net(trainX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead,lookAhead], dim=1)
ZM_T = (trainY_M[:].view(alpha_M.shape)-alpha_M)/beta_M
ZM_T = ZM_T.detach()
## ZM_t for validation
validX_M, validY_M = Variable(validX_M), Variable(validY_M)
paraM=net(validX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead,lookAhead], dim=1)
ZM_V = (validY_M[:].view(alpha_M.shape)-alpha_M)/beta_M
ZM_V = ZM_V.detach()

K=28
param_matrix_S_alpha_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0],K]))
param_matrix_S_beta_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0],K]))
param_matrix_S_gamma_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0],K]))
predY_matrix = pd.DataFrame(np.zeros([jump_ind.shape[0],K]))

lookBack=200
lookAhead=1

for k in range(K):
    # tenser for the individual stock
    dataX,dataY,mu,sigma = create_multi_ahead_samples_roll_single_month(data[k+1,:], lookBack, lookAhead) 

    trainX, trainY= dataX[ind_M[0:train_size],:,:],  dataY[ind_M[0:train_size]]
    validX, validY = dataX[ind_M[train_size:train_size+valid_size],:,:],  dataY[ind_M[train_size:train_size+valid_size]]
    testX, testY = dataX[jump_ind,:,:],  dataY[jump_ind] 
    
    # train the SNN
    netS = GRUModel_S_month(inputDim=2, hiddenNum=2*3*lookAhead, outputDim=3,lookAhead=lookAhead, layerNum=1)
    netS,trainloss,validloss=train_S(netS,trainX,validX, trainY,validY, ZM_T,ZM_V,
                                                lr, hidden_num=2*3*lookAhead, epoch=2500, max_epochs_stop=5,steps=20)
    testX = torch.transpose(testX,1,2)    
    #testY = torch.transpose(testY,1,2)
    testX,testY = Variable(testX), Variable(testY)
            
    ## prediction for monthly stock return: alpha_S,beta_S,gamma_S
    netS.eval()   
    thetapred = netS(testX)
    alpha_S, beta_S,gamma = torch.split(thetapred, [lookAhead,lookAhead,lookAhead], dim=1)
    
    param_matrix_S_alpha_step_one_ahead_month.iloc[:,k] = (mu+sigma*alpha_S).detach().numpy().reshape(-1,1)
    param_matrix_S_beta_step_one_ahead_month.iloc[:,k] = (sigma*beta_S).detach().numpy().reshape(-1,1)
    param_matrix_S_gamma_step_one_ahead_month.iloc[:,k] = (sigma*gamma).detach().numpy().reshape(-1,1)
    predY_matrix.iloc[:,k] = (testY*sigma+mu).detach().numpy().reshape(-1,1)



param_matrix_S_alpha_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_alpha_step_one_ahead_month_time_att.csv")
param_matrix_S_beta_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_beta_step_one_ahead_month_time_att.csv")
param_matrix_S_gamma_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_gamma_step_one_ahead_month_time_att.csv")

predY_matrix.to_csv("data/gru_save/predY_matrix_month.csv")
    
  
