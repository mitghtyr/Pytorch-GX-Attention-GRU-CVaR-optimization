######### GX-GRU  ###############
### exe with [nohup python -u GRU_network/gx_dynamic_GRU_one_ahead_month_attn.py>GRU_network/gx_print_one_ahead_month_attn.log >&1 &] in env
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

Dow_file = "data/daily_return_ratio.csv"
df = pd.read_csv(Dow_file)

data = torch.tensor(df.iloc[:, 1:30].transpose().values.astype("float32"))
datets = pd.to_datetime(df.Date)

[nrow, ncol] = data.shape

def create_multi_ahead_samples_roll_single_month(ts, lookBack, lookAhead=1):
    # standardization
    mu = torch.mean(ts)

    sigma = torch.sqrt(torch.var(ts))

    ts1 = (ts - mu) / sigma

    interval = 21
    dataX, dataY = [], []
    ncol = np.int(np.floor((ts.shape[0] - lookBack - interval) / lookAhead))

    for i in range(ncol):
        history_seq = ts1[(i * lookAhead): (i * lookAhead + lookBack)]
        future_seq = torch.prod(ts[(i * lookAhead + lookBack): (interval + (i) * lookAhead + lookBack)] + 1) - 1

        history_seq = torch.stack([history_seq,
                                   (history_seq - torch.mean(history_seq)) * (history_seq - torch.mean(history_seq))])

        dataX.append(history_seq)
        dataY.append(future_seq)

    dataX = torch.stack(dataX)
    dataY = torch.stack(dataY)
    mu = torch.mean(dataY)
    sigma = torch.sqrt(torch.var(dataY))
    dataY = (dataY - mu) / sigma

    return dataX, dataY, mu, sigma


lookBack = 200
lookAhead = 1
# create tensor for the Market factor, daily input X and monthly return Y
dataX_M, dataY_M, mu_M, sigma_M = create_multi_ahead_samples_roll_single_month(data[0, :], lookBack, lookAhead)

# (train+valid) : test=(0.48+0.12) : 0.4
rate1 = 0.6
rate2 = 0.8

train_size = int((dataY_M.shape[0]) * rate1 * rate2)
valid_size = int((dataY_M.shape[0]) * rate1 * (1 - rate2))
test_size = dataY_M.shape[0] - train_size - valid_size

interval = 21
# test set: evaluation per 21 days (per month)
jump_ind = torch.tensor(np.arange(1, np.int(np.floor(test_size / interval)) + 1))
jump_ind = train_size + valid_size + jump_ind * interval

#### validation separation: time-order or random split
# random-select valid : 1;
# time-seq valid: 0;
random_v = 0
if random_v == 1:
    #### random-select valid
    ind_M = torch.randperm(train_size + valid_size)

else:
    ##time-seq valid
    ind_M = torch.arange(train_size + valid_size)

trainX_M, trainY_M = dataX_M[ind_M[0:train_size], :, :], dataY_M[ind_M[0:train_size]]
validX_M, validY_M = dataX_M[ind_M[train_size:train_size + valid_size], :, :], dataY_M[
    ind_M[train_size:train_size + valid_size]]
testX_M, testY_M = dataX_M[jump_ind, :, :], dataY_M[jump_ind]

# learning rate
lr = 1e-3
# compare with the best model every 30 steps
steps = 30
K = 28
lookBack = 200
lookAhead = 1


def gx(x, u, v, A):
    '''
    :param x: latent variable
    :param u: upper tail parameter
    :param v: lower tail parameter
    :param A: constant, s.t. A=4
    :return: output of gx transformation
    '''
    return x * ((torch.exp(torch.log(u) * x) + torch.exp(-torch.log(v) * x)) / A + 1)


def divideinverse(Y, u, v, A):
    # inverse function of gx transformation, with divide algorithm
    Xleft = (Y < 0) * Y
    Xright = (Y >= 0) * Y

    index = torch.range(0, Y.nelement() - 1, dtype=int)
    for i in range(0, 1000):
        Xnl = Xleft[index]
        Xnr = Xright[index]
        Yn = Y[index]

        Xm = (Xnl + Xnr) / 2
        Ym = gx(Xm, u, v, A)
        ifp = Ym >= Yn
        ifn = Ym < Yn
        Xleft[index] = ifp * Xnl + ifn * Xm
        Xright[index] = ifp * Xm + ifn * Xnr

        Xm = (Xleft[index] + Xright[index]) / 2
        Ym = gx(Xm, u, v, A)
        index = index[(torch.abs(Ym - Yn) > 1e-12)]
        if index.nelement() == 0:
            break

    return (Xleft + Xright) / 2


def ggrad(x, u, v):
    # gradient of gx function with respect to x
    y = (torch.exp(torch.log(u) * x) + torch.exp(-torch.log(v) * x)) / 4 + 1
    y += x * ((torch.log(u) * torch.exp(torch.log(u) * x) - torch.log(v) * torch.exp(-torch.log(v) * x)) / 4)
    return y

########## outo-gradient cell for the back-propagation in GX-GRU #######

class fixginverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, u, v):

        output = divideinverse(y, u, v, A=4)
        ctx.save_for_backward(y, u, v, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        y, u, v, result = ctx.saved_variables
        #y = torch.clip(y, min=-50, max=50)
        grad_y = grad_u = grad_v = None

        grad_y = grad_output / ggrad(result, u, v)
        #if ctx.needs_input_grad[1]:
        grad_u = - grad_y * torch.square(result) * torch.exp(torch.log(u) * (result - 1)) / 4
        #if ctx.needs_input_grad[2]:
        grad_v =  grad_y * torch.square(result) * torch.exp(torch.log(v) * (-result - 1)) / 4
        #if 1 - ctx.needs_input_grad[0]:
            #grad_y = None

        return grad_y, grad_u, grad_v
class fixginverse_M(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, u, v):
        output = divideinverse(y, u, v, A=4)
        ctx.save_for_backward(y, u, v, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        y, u, v, result = ctx.saved_variables
        #y = torch.clip(y, min=-50, max=50)
        grad_y = grad_u = grad_v = None

        grad_y = grad_output / ggrad(result, u, v)
        #if ctx.needs_input_grad[1]:
        grad_u = - grad_y * torch.square(result) * torch.exp(torch.log(u) * (result - 1)) / 4
        #if ctx.needs_input_grad[2]:
        grad_v =  grad_y * torch.square(result) * torch.exp(torch.log(v) * (-result - 1)) / 4
        #if 1 - ctx.needs_input_grad[0]:
        grad_y = None

        return grad_y, grad_u, grad_v


class ginverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, u, v):

        output = divideinverse(y, u, v, A=4)
        ctx.save_for_backward(y, u, v, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y, u, v, result = ctx.saved_variables

        grad_y = grad_u = grad_v = None

        #if ctx.needs_input_grad[0]:
        grad_y = grad_output / ggrad(result, u, v)

        return grad_y, grad_u, grad_v


from torch.distributions import Normal

class BaseModel(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        super(BaseModel, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lookAhead = lookAhead
        self.fixpara = nn.Parameter(torch.ones(outputDim))
        self.layerNum = layerNum
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        ## GRU cell
        self.cell = nn.GRU(input_size=self.inputDim, hidden_size=self.hiddenNum,
                           num_layers=self.layerNum, dropout=0.0,
                           batch_first=True, )
        print(self.cell)
        self.fc = nn.Linear(self.hiddenNum, self.outputDim * self.lookAhead)

## Attention layer
class Attention(nn.Module):
    def __init__(self, hidden_size):
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
        hns = hidden.repeat(max_len, 1, 1).transpose(0, 1)  # B*T*H
        energy = F.tanh(self.attn(torch.cat([hns, encoder_outputs], 2)))  # [B*T*H]+[B*T*H] ->[B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        attn_energies = torch.bmm(v, energy).squeeze(1)  # [B*T]

        return F.softmax(attn_energies).unsqueeze(1)  # normalize with softmax [B*1*T]


# output the fixed uv for the market factor: MNN & FIX-OPTIM
class FixGRUModel_M(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        #super(FixGRUModel_M, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        super(FixGRUModel_M, self).__init__()
        self.fixpara = nn.Parameter(torch.ones(outputDim))
        self.sigmoid = nn.Sigmoid()
        self.lookAhead = lookAhead
    def forward(self, x):
        return (2 * self.sigmoid(self.fixpara) + 1)

    def loss(self, x, uv, y):
        batchSize = y.shape[0]
        thetapred = x
        # u_i,v_M,
        fixuv = uv
        # alpha_M,  beta_M,
        alpha_M, beta_M = torch.split(thetapred, [self.lookAhead, self.lookAhead], dim=1)
        ## loglikelihood
        m = Normal(torch.zeros(batchSize), torch.ones(batchSize))

        # Z_M tilde
        YM = (y[:].view(alpha_M.shape) - alpha_M) / beta_M
        ZM = fixginverse_M.apply(YM[:, 0], fixuv[0], fixuv[1])

        # gpartial
        gpzm = torch.abs(ggrad(ZM, fixuv[0], fixuv[1]))

        loglik = m.log_prob(ZM)
        loglik -= torch.log(gpzm)
        loglik -= torch.log(beta_M[:, 0])

        loss = -loglik
        
        ###############constraint the whole uv not too close to 1################
        # Z_M tilde
        YM_0 = y[:].view(alpha_M.shape) 
        ZM_0 = fixginverse_M.apply(YM_0[:, 0], fixuv[0], fixuv[1])
        ZM_0 = (ZM_0-torch.mean(ZM_0))/torch.std(ZM_0)

        # gpartial
        gpzm_0 = torch.abs(ggrad(ZM_0, fixuv[0], fixuv[1]))

        loglik_0 = m.log_prob(ZM_0)
        loglik_0 -= torch.log(gpzm_0)

        #loglik -= torch.log(beta_M[:, 0])

        loss_0 = -loglik_0
        return loss+loss_0


# output the fixed uv
class FixGRUModel_S(nn.Module):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        # super(FixGRUModel_S, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        super(FixGRUModel_S, self).__init__()
        self.fixpara = nn.Parameter(torch.ones(outputDim))
        self.sigmoid = nn.Sigmoid()
        self.lookAhead = lookAhead
    def forward(self, x):
        return (2 * self.sigmoid(self.fixpara) + 1)

    def loss(self, x, uv, y, ZM):
        batchSize = y.shape[0]

        # alpha_M, alpha_i, beta_M, beta_i,gamma_i
        thetapred = x
        # u_i,v_M, u_Mi,v_Mi,u_i, v_i
        fixuv = uv

        alpha_i, beta_i, gamma = torch.split(thetapred, [self.lookAhead, self.lookAhead, self.lookAhead], dim=1)

        ## loglikelihood
        m = Normal(torch.zeros(batchSize), torch.ones(batchSize))

        gzM = gx(ZM.view(gamma.shape), fixuv[0], fixuv[1], 4)
        Yi = (y[:].view(alpha_i.shape) - alpha_i - beta_i * gzM) / gamma

        Zi = fixginverse.apply(Yi[:, 0], fixuv[2], fixuv[3])

        gpzi = torch.abs(ggrad(Zi, fixuv[2], fixuv[3]))

        loglik = m.log_prob(Zi)

        loglik -= torch.log(gpzi)

        loglik -= torch.log(gamma[:, 0])

        loss = -loglik
        
         ###############constraint the whole uv not too close to 1################
        y0 = y[:].view(alpha_i.shape)-torch.mean(y[:].view(alpha_i.shape))
        fixbeta =torch.dot(gzM[:,0],y0[:,0])/torch.dot(gzM[:,0],gzM[:,0])
        Yi = (y0 - fixbeta*gzM) 

        Zi = fixginverse.apply(Yi[:, 0], fixuv[2], fixuv[3])
        Zi = (Zi-torch.mean(Zi))/torch.std(Zi)

        gpzi = torch.abs(ggrad(Zi, fixuv[2], fixuv[3]))

        loglik_0 = m.log_prob(Zi)

        loglik_0 -= torch.log(gpzi)

        #loglik_0 -= torch.log(gamma[:, 0])

        loss_0 = -loglik_0
        
        return loss+loss_0


class GRUModel_M(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        super(GRUModel_M, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        self.attention = Attention(hiddenNum)

    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))

        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        energy = self.attention(hn, rnnOutput)  # B*1*T
        context = energy.bmm(rnnOutput).squeeze(1)  # (B,1,T)*(B*T*H)->(B,1,H)->(B*H)

        fcOutput = self.fc(context)
        # ahpla_m is controled in the interval [-1,1]
        fcOutput[:, 0:self.lookAhead] = 1 * (2 * self.sigmoid(fcOutput[:, 0:self.lookAhead]) - 1)
        # alphai is controled in the interval [0,10]
        fcOutput[:, self.lookAhead:(2 * self.lookAhead)] = (
                    10 * self.sigmoid(fcOutput[:, self.lookAhead:(2 * self.lookAhead)]))

        return fcOutput

    def loss(self, x, uv, y):
        batchSize = y.shape[0]

        # alpha_M, alpha_i, beta_M, beta_i,gamma_i
        thetapred = x
        fixuv = uv

        alpha_M, beta_M = torch.split(thetapred, [self.lookAhead, self.lookAhead], dim=1)
        ## loglikelihood
        m = Normal(torch.zeros(batchSize), torch.ones(batchSize))

        # Z_M tilde
        YM = (y[:].view(alpha_M.shape) - alpha_M) / beta_M
        #ZM = fixginverse.apply(YM[:, 0], fixuv[0], fixuv[1])
        ZM = ginverse.apply(YM[:, 0], fixuv[0], fixuv[1])
        # gpartial
        gpzm = torch.abs(ggrad(ZM, fixuv[0], fixuv[1]))

        loglik = m.log_prob(ZM)
        loglik -= torch.log(gpzm)
        loglik -= torch.log(beta_M[:, 0])

        loss = -loglik
        return loss


class GRUModel_S(BaseModel):

    def __init__(self, inputDim, hiddenNum, outputDim, lookAhead, layerNum):
        super(GRUModel_S, self).__init__(inputDim, hiddenNum, outputDim, lookAhead, layerNum)
        self.attention = Attention(hiddenNum)

    def forward(self, x):
        batchSize = x.size(0)
        h0 = Variable(torch.zeros(self.layerNum * 1, batchSize, self.hiddenNum))

        rnnOutput, hn = self.cell(x, h0)
        hn = hn.view(batchSize, self.hiddenNum)
        energy = self.attention(hn, rnnOutput)  # B*1*T
        context = energy.bmm(rnnOutput).squeeze(1)  # (B,1,T)*(B*T*H)->(B,1,H)->(B*H)

        fcOutput = self.fc(context)
        # alphai is controled in the interval [-1,1]
        fcOutput[:, 0:self.lookAhead] = 1 * (2 * self.sigmoid(fcOutput[:, 0:self.lookAhead]) - 1)

        #  beta_i,gamma_i are controled in the interval [-10,10],[0,20]
        fcOutput[:, self.lookAhead:(2 * self.lookAhead)] = 10 * (
                    2 * self.sigmoid(fcOutput[:, self.lookAhead:(2 * self.lookAhead)]) - 1)
        fcOutput[:, (2 * self.lookAhead):(3 * self.lookAhead)] = (
                    10 * self.sigmoid(fcOutput[:, (2 * self.lookAhead):(3 * self.lookAhead)]))

        return fcOutput

    def loss(self, x, uv, y, ZM):
        batchSize = y.shape[0]

        # alpha_M, alpha_i, beta_M, beta_i,gamma_i
        thetapred = x
        fixuv = uv

        alpha_i, beta_i, gamma = torch.split(thetapred, [self.lookAhead, self.lookAhead, self.lookAhead], dim=1)

        ## loglikelihood
        m = Normal(torch.zeros(batchSize), torch.ones(batchSize))

        gzM = gx(ZM.view(gamma.shape), fixuv[0], fixuv[1], 4)
        Yi = (y[:].view(alpha_i.shape) - alpha_i - beta_i * gzM) / gamma
        #Zi = fixginverse.apply(Yi[:, 0], fixuv[2], fixuv[3])
        Zi = ginverse.apply(Yi[:, 0], fixuv[2], fixuv[3])
        gpzi = torch.abs(ggrad(Zi, fixuv[2], fixuv[3]))

        loglik = m.log_prob(Zi)

        loglik -= torch.log(gpzi)
        loglik -= torch.log(gamma[:, 0])

        loss = -loglik
        return loss


def train_fix_M(net, trainX, validX, trainY, validY, thetapredt, thetapredv, lr, hidden_num=64, epoch=20,
                max_epochs_stop=2, steps=5):
    trainloss = []
    validloss = []

    # 都和single时候一样
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.2)
    # 步长规划器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps = steps

    best_epoch_no_improve = 0

    for i in range(epoch):

        # train step
        net.train()

        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)

        optimizer.zero_grad()
        uv = net.forward(trainX)

        losst = net.loss(thetapredt, uv, trainY).mean()

        uv = net.forward(validX)

        lossv = net.loss(thetapredv, uv, validY).mean()
        lossvt = losst + lossv

        lossvt.backward()
        optimizer.step()

        if lossvt.item() < valid_loss_min:
            # torch.save(net.state_dict(), "ctsun/.jupyter/fix_M_histmodel.pt")
            bias = np.abs(lossvt.item() - valid_loss_min)
            valid_loss_min = lossvt.item()

            # Track improvement
            epochs_no_improve = 0
            best_epoch_no_improve = 0
            best_epoch = i

            # print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())

        if ((i + 1) % (accumulation_steps)) == 0:
            print("%d epoch is finished!" % (i + 1))
            if (lossvt.item() > valid_loss_min) | (bias < 1e-5):
                epochs_no_improve += 1
                print("%d epoch falls in!" % (i + 1))
                print(" validloss is %f" %(lossv.item()))
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                    )

                    net.optimizer = optimizer
                    return net, trainloss, validloss

    net.optimizer = optimizer
    return net, trainloss, validloss


def train_fix_S(net, trainX, validX, trainY, validY, ZM_T, ZM_V, thetapredt, thetapredv, lr, hidden_num=64, epoch=20,
                max_epochs_stop=2, steps=5):
    trainloss = []
    validloss = []

    # 都和single时候一样
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.2)
    # 步长规划器
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps = steps

    best_epoch_no_improve = 0

    for i in range(epoch):

        # train step
        net.train()

        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)

        optimizer.zero_grad()
        uv = net.forward(trainX)

        losst = net.loss(thetapredt, uv, trainY, ZM_T).mean()
        uv = net.forward(validX)

        lossv = net.loss(thetapredv, uv, validY, ZM_V).mean()
        lossvt = losst + lossv

        lossvt.backward()
        optimizer.step()

        # valid step

        if lossvt.item() < valid_loss_min:
            # torch.save(net.state_dict(), "ctsun/.jupyter/fix_S_histmodel.pt")
            bias = np.abs(lossvt.item() - valid_loss_min)
            valid_loss_min = lossvt.item()

            # Track improvement
            epochs_no_improve = 0
            best_epoch_no_improve = 0
            best_epoch = i

            # print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())

        if ((i + 1) % (accumulation_steps)) == 0:
            print("%d epoch is finished!" % (i + 1))
            if (lossvt.item() > valid_loss_min) | (bias < 1e-5):
                epochs_no_improve += 1

                print("%d epoch falls in!" % (i + 1))
                print(" validloss is %f" % (lossv.item()))
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                    )

                    net.optimizer = optimizer
                    return net, trainloss, validloss

    net.optimizer = optimizer
    return net, trainloss, validloss


def train_M(net, trainX, validX, trainY, validY, uvt, uvv, lr, hidden_num=64, epoch=20, max_epochs_stop=2, steps=5):
    trainloss = []
    validloss = []

    # 都和single时候一样
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.2)
    # 步长规划器
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps = steps

    best_epoch_no_improve = 0

    for i in range(epoch):

        # train step
        net = net.train()

        trainX, trainY = Variable(trainX), Variable(trainY)

        optimizer.zero_grad()
        thetapred = net.forward(trainX)

        losst = net.loss(thetapred, uvt, trainY).mean()

        losst.backward()
        optimizer.step()

        # valid step

        with torch.no_grad():
            # Set to evaluation mode
            net.eval()

            validX, validY = Variable(validX), Variable(validY)

            thetapred = net.forward(validX)

            lossv = net.loss(thetapred, uvv, validY).mean()

            # if ((i+1)%12)==0:
            # torch.save(net.state_dict(), "histmodelpre.pt")

        if lossv.item() < valid_loss_min:
            torch.save(net.state_dict(), "M_histmodel.pt")
            bias = np.abs(lossv.item() - valid_loss_min)
            valid_loss_min = lossv.item()

            # Track improvement
            epochs_no_improve = 0
            best_epoch_no_improve = 0
            best_epoch = i

            # print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())

        if ((i + 1) % (accumulation_steps)) == 0:
            print("%d epoch is finished!" % (i + 1))
            if (lossv.item() > valid_loss_min) | (bias < 1e-5):
                epochs_no_improve += 1
                net.load_state_dict(torch.load("M_histmodel.pt"))

                # 根据目前loss和历史最优loss的差距调整步长
                #bias2 = np.abs(lossv.item()-valid_loss_min)
                #scheduler.step(max(bias,bias2))

                # Trigger early stopping
                print("%d epoch falls in" % (i + 1))

                print(
                    f'\nvalidloss  is: {lossv.item():.5f}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}. epoch without improvement is {epochs_no_improve}'
                )
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                    )

                    net.optimizer = optimizer
                    return net, trainloss, validloss

    net.optimizer = optimizer
    return net, trainloss, validloss


def train_S(net, trainX, validX, trainY, validY, ZM_T, ZM_V, uvt, uvv, lr, hidden_num=64, epoch=20, max_epochs_stop=2,
            steps=5):
    trainloss = []
    validloss = []

    # 都和single时候一样
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=0.2)
    # 步长规划器
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)
    valid_loss_min = np.Inf
    epochs_no_improve = 0
    accumulation_steps = steps

    best_epoch_no_improve = 0

    for i in range(epoch):

        # train step
        net = net.train()

        trainX, trainY = Variable(trainX), Variable(trainY)

        optimizer.zero_grad()
        thetapred = net.forward(trainX)

        losst = net.loss(thetapred, uvt, trainY, ZM_T).mean()

        losst.backward()
        optimizer.step()

        # valid step

        with torch.no_grad():
            # Set to evaluation mode
            net.eval()

            validX, validY = Variable(validX), Variable(validY)

            thetapred = net.forward(validX)

            lossv = net.loss(thetapred, uvv, validY, ZM_V).mean()

            # if ((i+1)%12)==0:
            # torch.save(net.state_dict(), "histmodelpre.pt")

        if lossv.item() < valid_loss_min:
            torch.save(net.state_dict(), "histmodel.pt")
            bias = np.abs(lossv.item() - valid_loss_min)
            valid_loss_min = lossv.item()

            # Track improvement
            epochs_no_improve = 0
            best_epoch_no_improve = 0
            best_epoch = i

            # print("%d epoch is finished!" % (i+1))
            trainloss.append(losst.item())
            validloss.append(lossv.item())

        if ((i + 1) % (accumulation_steps)) == 0:
            print("%d epoch is finished!" % (i + 1))
            if (lossv.item() > valid_loss_min) | (bias < 1e-5):
                epochs_no_improve += 1
                net.load_state_dict(torch.load("histmodel.pt"))

                # 根据目前loss和历史最优loss的差距调整步长
                #bias2 = np.abs(lossv.item()-valid_loss_min)
                #scheduler.step(max(bias,bias2))

                # Trigger early stopping
                print("%d epoch falls in" % (i + 1))

                print(
                    f'\nvalidloss  is: {lossv.item():.5f}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}. epoch without improvement is {epochs_no_improve}'
                )
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {i}. Best epoch: {best_epoch} with loss: {valid_loss_min:.5f}'
                    )

                    net.optimizer = optimizer
                    return net, trainloss, validloss

    net.optimizer = optimizer
    return net, trainloss, validloss


def itertrain_M(trainX, validX, trainY, validY, lookAhead,
                lr, hidden_numfix=10, hidden_num=20, epochfix=500, epoch=2500,
                max_epochs_stop_fix=8, max_epochs_stop=8, stepsfix=10, steps=30, maxk=20):
    trainX = torch.transpose(trainX, 1, 2)
    # trainY = torch.transpose(trainY,1,2)
    validX = torch.transpose(validX, 1, 2)
    # validY = torch.transpose(validY,1,2)

    k = 0
    lossvk = []

    fixuvt = torch.ones(2,requires_grad=False)
    fixuvv = torch.ones(2,requires_grad=False)
    optimuv = fixuvt
    optimloss = np.inf
    no_improve =0

    net_int = None

    net_int = GRUModel_M(inputDim=2, hiddenNum=hidden_num, outputDim=2, lookAhead=lookAhead, layerNum=1)
    torch.save(net_int.state_dict(), "histmodel_iter.pt")
    netfix_int = None

    netfix_int = FixGRUModel_M(inputDim=2, hiddenNum=hidden_numfix, outputDim=2, lookAhead=lookAhead, layerNum=1)
    torch.save(netfix_int.state_dict(), "histmodel_iter_fix.pt")
    while (k <= maxk):
        net = None

        net = GRUModel_M(inputDim=2, hiddenNum=hidden_num, outputDim=2, lookAhead=lookAhead, layerNum=1)
        net.load_state_dict(torch.load("histmodel_iter.pt"))
        net, trainloss, validloss = train_M(net, trainX, validX, trainY, validY,
                                            fixuvt, fixuvv,
                                            epoch=epoch, lr=lr,
                                            hidden_num=hidden_num, max_epochs_stop=max_epochs_stop,
                                            steps=steps)
        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)
        net.eval()
        thetapredt = net(trainX).detach()
        thetapredv = net(validX).detach()



        netfix = None

        netfix = FixGRUModel_M(inputDim=2, hiddenNum=hidden_numfix, outputDim=2, lookAhead=lookAhead, layerNum=1)
        netfix.load_state_dict(torch.load("histmodel_iter_fix.pt"))
        netfix, trainlossfix, validlossfix = train_fix_M(netfix, trainX, validX, trainY, validY,
                                                         thetapredt, thetapredv,
                                                         epoch=epochfix, lr=1e-1,
                                                         hidden_num=hidden_numfix, max_epochs_stop=max_epochs_stop_fix,
                                                         steps=stepsfix)
        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)
        netfix.eval()


        if validloss[-1]<=optimloss:
            no_improve=0
            fixuvt = netfix(trainX).detach()
            fixuvv = netfix(validX).detach()
            loss2= np.abs(optimloss-validloss[-1])
            optimloss = validloss[-1]
            optimuv = fixuvt
            torch.save(net.state_dict(), "histmodel_iter.pt")
            torch.save(netfix.state_dict(), "histmodel_iter_fix.pt")
            lossvk.append(validlossfix[-1])
            k += 1
        else:
            fixuvt=optimuv
            fixuvv = optimuv
            net.load_state_dict(torch.load("histmodel_iter.pt"))
            netfix.load_state_dict(torch.load("histmodel_iter_fix.pt"))
            loss2 = np.abs(optimloss-validloss[-1])
            no_improve +=1
            print("Back to optimal model!" )


        print("%d iteration is finished!" % (k ))


        if k >= 3:
            biasloss = np.abs(validlossfix[-1]-validloss[-1])
            #print("the loss between iters is  %f !" % (biasloss))

            if biasloss < 1e-3 and loss2< 1e-2:

                break

        if no_improve >= 3:
            break

    return netfix, net, lossvk, trainloss, validloss, trainlossfix, validlossfix


def itertrain_S(trainX, validX, trainY, validY, ZM_t, ZM_v, lookAhead, input_dim,
                lr, hidden_numfix=10, hidden_num=20, epochfix=500, epoch=2500,
                max_epochs_stop_fix=8, max_epochs_stop=8, stepsfix=10, steps=30, maxk=20):
    trainX = torch.transpose(trainX, 1, 2)
    # trainY = torch.transpose(trainY,1,2)
    validX = torch.transpose(validX, 1, 2)
    # validY = torch.transpose(validY,1,2)

    k = 0
    lossvk = []

    fixuvt = torch.ones(4,requires_grad=False)
    fixuvv = torch.ones(4,requires_grad=False)
    optimuv = fixuvt
    optimloss = np.inf
    no_improve=0
    net_int = None

    net_int = GRUModel_S(inputDim=input_dim, hiddenNum=hidden_num, outputDim=3, lookAhead=lookAhead, layerNum=1)
    torch.save(net_int.state_dict(), "histmodel_iter.pt")

    netfix_int = None

    netfix_int = FixGRUModel_S(inputDim=input_dim, hiddenNum=hidden_numfix, outputDim=4, lookAhead=lookAhead,
                           layerNum=1)
    torch.save(netfix_int.state_dict(), "histmodel_iter_fix.pt")
    while (k <= maxk):
        net = None

        net = GRUModel_S(inputDim=input_dim, hiddenNum=hidden_num, outputDim=3, lookAhead=lookAhead, layerNum=1)
        net.load_state_dict(torch.load("histmodel_iter.pt"))

        net, trainloss, validloss = train_S(net, trainX, validX, trainY, validY, ZM_t, ZM_v,
                                            fixuvt, fixuvv,
                                            epoch=epoch, lr=lr,
                                            hidden_num=hidden_num, max_epochs_stop=max_epochs_stop,
                                            steps=steps)
        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)
        thetapredt = net(trainX).detach()
        thetapredv = net(validX).detach()

        netfix = None

        netfix = FixGRUModel_S(inputDim=input_dim, hiddenNum=hidden_numfix, outputDim=4, lookAhead=lookAhead,
                               layerNum=1)
        netfix.load_state_dict(torch.load("histmodel_iter_fix.pt"))
        netfix, trainlossfix, validlossfix = train_fix_S(netfix, trainX, validX, trainY, validY, ZM_t, ZM_v,
                                                         thetapredt, thetapredv,
                                                         epoch=epochfix, lr=1e-1,
                                                         hidden_num=hidden_numfix, max_epochs_stop=max_epochs_stop_fix,
                                                         steps=stepsfix)
        trainX, trainY = Variable(trainX), Variable(trainY)
        validX, validY = Variable(validX), Variable(validY)
        if validlossfix[-1] <= optimloss:
            no_improve = 0
            fixuvt = netfix(trainX).detach()
            fixuvv = netfix(validX).detach()
            loss2 = np.abs(optimloss - validlossfix[-1])
            optimloss = validlossfix[-1]
            optimuv = fixuvt
            torch.save(net.state_dict(), "histmodel_iter.pt")
            torch.save(netfix.state_dict(), "histmodel_iter_fix.pt")
            lossvk.append(validlossfix[-1])
            k += 1
        else:
            fixuvt = optimuv
            fixuvv = optimuv
            net.load_state_dict(torch.load("histmodel_iter.pt"))
            netfix.load_state_dict(torch.load("histmodel_iter_fix.pt"))
            loss2 = np.abs(optimloss - validlossfix[-1])
            no_improve += 1
            print("%d iteration is back to optimal model!" % (k))

        print("%d iteration is finished!" % (k ))
        if k >= 3:
            biasloss = np.abs(validlossfix[-1] - validloss[-1])
            # print("the loss between iters is  %f !" % (biasloss))

            if biasloss < 1e-3 and loss2 < 1e-2:
                break

        if no_improve >= 3:
            break


    return netfix, net, lossvk, trainloss, validloss, trainlossfix, validlossfix


lookAhead = 1
lr=1e-3
#net = GRUModel_M(inputDim=2, hiddenNum=4, outputDim=2,lookAhead=1, layerNum=1)
#netfix = FixGRUModel_M(inputDim=2, hiddenNum=4, outputDim=2,lookAhead=1, layerNum=1)
#net.load_state_dict(torch.load("ctsun/.jupyter/M_histmodel.pt"))
#netfix.load_state_dict(torch.load("ctsun/.jupyter/fix_M_histmodel.pt"))
netfix, net, lossvk, trainloss, validloss, trainlossfix, validlossfix = itertrain_M(trainX_M, validX_M, trainY_M,
                                                                                    validY_M, lookAhead,
                                                                                    lr, hidden_numfix=4,
                                                                                    hidden_num=2 * 2 * lookAhead,
                                                                                    epochfix=2000, epoch=2500,
                                                                                    max_epochs_stop_fix=5,
                                                                                    max_epochs_stop=10, stepsfix=10,
                                                                                    steps=30, maxk=5)

testX_M = torch.transpose(testX_M, 1, 2)
# testY = torch.transpose(testY,1,2)

testX_M, testY_M = Variable(testX_M), Variable(testY_M)

net.eval()
netfix.eval()
uv_M = netfix(testX_M)
paraM = net(testX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead, lookAhead], dim=1)

param_matrix_M_alpha_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0], 1]))
param_matrix_M_beta_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0], 1]))
param_matrix_M_alpha_step_one_ahead_month.iloc[:, 0] = (mu_M + sigma_M * alpha_M).detach().numpy().reshape(-1, 1)
param_matrix_M_beta_step_one_ahead_month.iloc[:, 0] = (sigma_M * beta_M).detach().numpy().reshape(-1, 1)

param_matrix_M_alpha_step_one_ahead_month.to_csv("param_matrix_M_alpha_step_one_ahead_month_time_gx_att.csv")
param_matrix_M_beta_step_one_ahead_month.to_csv("param_matrix_M_beta_step_one_ahead_month_time_gx_att.csv")

print("Market finished!")
net.eval()
netfix.eval()

trainX_M = torch.transpose(trainX_M, 1, 2)
# trainY = torch.transpose(trainY,1,2)
validX_M = torch.transpose(validX_M, 1, 2)

trainX_M, trainY_M = Variable(trainX_M), Variable(trainY_M)
validX_M, validY_M = Variable(validX_M), Variable(validY_M)

uv_M = netfix(trainX_M)
paraM = net(trainX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead, lookAhead], dim=1)

ZM_T = (trainY_M[:].view(alpha_M.shape) - alpha_M) / beta_M
ZM_T = fixginverse.apply(ZM_T[:, 0], uv_M[0], uv_M[1])
ZM_T = ZM_T.detach()

uv_M = netfix(validX_M)
paraM = net(validX_M)
alpha_M, beta_M = torch.split(paraM, [lookAhead, lookAhead], dim=1)
ZM_V = (validY_M[:].view(alpha_M.shape) - alpha_M) / beta_M
ZM_V = fixginverse.apply(ZM_V[:, 0], uv_M[0], uv_M[1])
ZM_V = ZM_V.detach()

K = 28
UV_matrix_gx = pd.DataFrame(np.zeros([K + 1, 4]))
param_matrix_S_alpha_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0], K]))
param_matrix_S_beta_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0], K]))
param_matrix_S_gamma_step_one_ahead_month = pd.DataFrame(np.zeros([jump_ind.shape[0], K]))
predY_matrix = pd.DataFrame(np.zeros([jump_ind.shape[0], K]))

UV_matrix_gx.iloc[0, 0:2] = uv_M.detach().numpy()

lookBack = 200
lookAhead = 1

input_dim = 2
# input_dim=5
for k in range(K):
    # input dim =2
    dataX, dataY, mu, sigma = create_multi_ahead_samples_roll_single_month(data[k + 1, :], lookBack, lookAhead)
    # input dim =5
    # dataX,dataY,mu,sigma = create_multi_ahead_samples_roll_bivariate_month(data[[0,k+1],:], lookBack, lookAhead)

    trainX, trainY = dataX[ind_M[0:train_size], :, :], dataY[ind_M[0:train_size]]
    validX, validY = dataX[ind_M[train_size:train_size + valid_size], :, :], dataY[
        ind_M[train_size:train_size + valid_size]]
    testX, testY = dataX[jump_ind, :, :], dataY[jump_ind]

    netfix_S, net_S, lossvk, trainloss, validloss, trainlossfix, validlossfix = itertrain_S(trainX, validX, trainY,
                                                                                            validY,
                                                                                            ZM_T, ZM_V, lookAhead,
                                                                                            input_dim,
                                                                                            lr, hidden_numfix=6,
                                                                                            hidden_num=2 * 3 * lookAhead,
                                                                                            epochfix=2000,
                                                                                            epoch=2500,
                                                                                            max_epochs_stop_fix=5,
                                                                                            max_epochs_stop=10,
                                                                                            stepsfix=10, steps=30,
                                                                                            maxk=5)
    testX = torch.transpose(testX, 1, 2)
    # testY = torch.transpose(testY,1,2)

    testX, testY = Variable(testX), Variable(testY)

    net_S.eval()
    netfix_S.eval()
    uv_S = netfix_S(testX)
    UV_matrix_gx.iloc[k + 1, :] = uv_S.detach().numpy()
    thetapred = net_S(testX)
    alpha_S, beta_S, gamma = torch.split(thetapred, [lookAhead, lookAhead, lookAhead], dim=1)

    param_matrix_S_alpha_step_one_ahead_month.iloc[:, k] = (mu + sigma * alpha_S).detach().numpy().reshape(-1, 1)
    param_matrix_S_beta_step_one_ahead_month.iloc[:, k] = (sigma * beta_S).detach().numpy().reshape(-1, 1)
    param_matrix_S_gamma_step_one_ahead_month.iloc[:, k] = (sigma * gamma).detach().numpy().reshape(-1, 1)

    predY_matrix.iloc[:, k] = (testY * sigma + mu).detach().numpy().reshape(-1, 1)
    print(
        f'\n The {k+1}th stock is finished, remaining {28-k-1} stocks.'
    )
if random_v == 1:
    # input dim =2
    # param_matrix_S_alpha_step_one_ahead_month.to_csv("param_matrix_S_alpha_step_one_ahead_month_gx.csv")
    # input dim =5
    param_matrix_S_alpha_step_one_ahead_month.to_csv("param_matrix_S_alpha_step_one_ahead_month_gx_5.csv")
    param_matrix_S_beta_step_one_ahead_month.to_csv("param_matrix_S_beta_step_one_ahead_month_gx_5.csv")
    param_matrix_S_gamma_step_one_ahead_month.to_csv("param_matrix_S_gamma_step_one_ahead_month_gx_5.csv")

    # predY_matrix.to_csv("predY_matrix_month_gx.csv")
    # input_dim=2
    # UV_matrix_gx.to_csv("UV_matrix_gx_month_5.csv")
    # input_dim=5
    UV_matrix_gx.to_csv("UV_matrix_gx_month_5.csv")

else:
    # input dim =2
    # param_matrix_S_alpha_step_one_ahead_month.to_csv("param_matrix_S_alpha_step_one_ahead_month_time_gx.csv")
    # input dim =5
    param_matrix_S_alpha_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_alpha_step_one_ahead_month_time_gx_att.csv")
    param_matrix_S_beta_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_beta_step_one_ahead_month_time_gx_att.csv")
    param_matrix_S_gamma_step_one_ahead_month.to_csv("data/gru_save/param_matrix_S_gamma_step_one_ahead_month_time_gx_att.csv")

    # predY_matrix.to_csv("predY_matrix_month_time_gx.csv")
    # input dim =2
    # UV_matrix_gx.to_csv("UV_matrix_gx_month_time.csv")
    # input dim =5
    UV_matrix_gx.to_csv("data/gru_save/UV_matrix_gx_month_time_att.csv")