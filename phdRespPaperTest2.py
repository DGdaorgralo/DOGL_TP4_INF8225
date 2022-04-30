

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:33:14 2022

@author: daorg
"""

"""
import numpy as np
import torch

np.random.seed(2)

T = 20
L = 1000
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float64')
torch.save(data, open('traindatawavw.pt', 'wb'))
"""

class SequenceOneChannel(nn.Module):
    def __init__(self):
        super(SequenceOneChannel , self).__init__()
        self.n_hidden = 31
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
        # print("Input to forward (LSTM)", input.shape)
        outputs = []
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        # print("last Input_t to LSTMcell", input_t.shape)
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class SequenceOneChannel1(nn.Module):
    def __init__(self):
        super(SequenceOneChannel1 , self).__init__()
        self.n_hidden = 15
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
        # print("Input to forward (LSTM)", input.shape)
        outputs = []
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        # print("last Input_t to LSTMcell", input_t.shape)
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class SequenceOneChannel3(nn.Module):
    def __init__(self):
        super(SequenceOneChannel3 , self).__init__()
        self.n_hidden = 15
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
        # print("Input to forward (LSTM)", input.shape)
        outputs = []
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t3 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t3 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs += [output]
        # print("last Input_t to LSTMcell", input_t.shape)
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            output = self.linear(h_t3)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
class SequenceOneChannel5(nn.Module):
    def __init__(self):
        super(SequenceOneChannel5 , self).__init__()
        self.n_hidden = 51
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm4 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm5 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
        # print("Input to forward (LSTM)", input.shape)
        outputs = []
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t3 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t3 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t4 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t4 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t5 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t5 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm5(h_t4, (h_t5, c_t5))
            output = self.linear(h_t3)
            outputs += [output]
        # print("last Input_t to LSTMcell", input_t.shape)
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))
            h_t5, c_t5 = self.lstm3(h_t4, (h_t5, c_t5))
            output = self.linear(h_t3)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class SequenceMultiChannel(nn.Module):
    def __init__(self):
        super(SequenceMultiChannel , self).__init__()
        self.n_hidden = 31
        self.n_feat = 1024
        self.lstm1 = nn.LSTMCell(self.n_feat, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, self.n_feat)

    def forward(self, input, future = 0):
        # print("Input to forward (LSTM)", input.shape)
        outputs = []
        outputs1 = torch.empty(input.size(0), 0, self.n_feat)
        h_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.n_hidden, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            input_t = input_t.squeeze(1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            # print("size output", output.shape)
            outputs += [output]
            outputs1 = torch.cat((outputs1, output.unsqueeze(1)), 1)
        # print("size outputs1", outputs1.shape)
        # print("last Input_t to LSTMcell", input_t.shape)
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
            outputs1 = torch.cat((outputs1, output.unsqueeze(1)), 1)
        outputs = torch.cat(outputs, dim=1)
        return outputs1#outputs

# for input_t in input.split(1, dim=1):
#     input_t = input_t.squeeze(1)
#     print("last Input_t to LSTMcell", input_t.shape)
#     break

def training_loop0(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target):
    for i in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimiser.step(closure)
        with torch.no_grad():
            future = 2
            pred = model(test_input, future=future)
            # use all pred samples, but only go to 999
            loss = loss_fn(pred[:, :-future], test_target)
            y = pred.detach().numpy()
        # draw figures
        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] # 999
        def draw(yi, colour):
            plt.plot(np.arange(n), yi[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        plt.savefig("predict%d.png"%i, dpi=200)
        plt.close()
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))

def training_loop1(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target):
    train_loss = []
    val_loss = []
    for i in range(n_epochs):
        def closure():
            optimiser.zero_grad()
            out = model(train_input)
            # print("Size predOutput:", out.shape)
            # print("Size trainTarget:", train_target.shape)
            loss = loss_fn(out, train_target)
            loss.backward()
            # print('loss:', loss.item())
            return loss
        optimiser.step(closure)
        with torch.no_grad():
            future = 2
            pred = model(test_input, future=future)
            # use all pred samples, but only go to 999
            # print("size pred", pred.shape)
            # print("size test_target", test_target.shape)
            loss = loss_fn(pred[:, :-future], test_target)
            y = pred.detach().numpy()
        # draw figures
        if i == 0 or i%9 == 0:
            plt.figure(figsize=(12,6))
            # plt.title(f"Step {i+1}")
            plt.title("Step: {}, Loss: {}".format(i, loss))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            n = train_input.shape[1] # 49
            def draw(y_original, y_pred, colour):
                plt.plot(np.arange(n), y_original[:n], colour, linewidth=2.0)
                plt.plot(np.arange(n), y_pred[:n], colour+":", linewidth=2.0)
                # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
            draw(test_target[0], y[0], 'r')
            draw(test_target[1], y[1], 'b')
            draw(test_target[2], y[2], 'g')
            plt.savefig("t1nPredict%d.png"%i, dpi=200)
            plt.close()
        # print the loss
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Train Loss: {}".format(i, loss_print))
        train_loss += [loss_print]
        out = model(test_input)
        loss_print = loss_fn(out, test_target)
        print("Step: {}, Test Loss: {}".format(i, loss_print))        
        val_loss += [loss_print]
        if val_loss[-1]>0.1  or train_loss[-1]>0.1:
            break
        
        
    plt.figure(figsize=(12,6))
    # plt.title(f"Step {i+1}")
    plt.title("Loss (N train: {}, N val: {})".format(train_input.shape[0], test_input.shape[0]))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(np.arange(i), train_loss[:i], 'b', linewidth=2.0)
    plt.plot(np.arange(i), val_loss[:i], 'b'+":", linewidth=2.0)
    plt.savefig("TrainingLoss.png", dpi=200)
    plt.close()


#---
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=15, help='steps to run')
opt = parser.parse_args()
opt.steps = 30 #47#37 #---
# set random seed to 0
np.random.seed(0)
torch.manual_seed(0)
# load data and make training set
# data = torch.load('traindatawavw.pt')#

"""
traindataFp1.pt : FP spectra, from 55 sites, only 1 cam pixel (50 time-steps) [source Fp50C1M.mat]
            test Loss: 4597203.734938095
traindataFp1n.pt  : FP spectra, from 55 sites, only 1 cam pixel (50 time-steps) Normalized with MinMaxScaler [source Fp50C1M.mat]
            test Loss: 0.0008523097699205046

traindataFp1024.pt:  FP spectra, from 55 sites, 1024 cam pixel (50 time-steps)  [source Fp50C1024.mat]
            test Loss: 4841708.46318737
traindataFp1024n.pt:  FP spectra, from 55 sites, 1024 cam pixel (50 time-steps) Normalized with MinMaxScaler [source Fp50C1024.mat]
            test Loss: 0.005942544753121238

"""
nst = 3 #number of samples for test
channels = 1024
# max_int_cam = 62780
from sklearn.preprocessing import MinMaxScaler#--- 
normStrategy = 'throughTime' #'throughTime'  #'throughSamples'


if channels == 1:
    # data = torch.load('traindataFp1.pt')#traindataFp1#---
    data = torch.load('traindataFp1.pt')#traindataFp1#---
    # data = data / max_int_cam #not working test Loss: 0.0011788546711502719
    scaler1 = MinMaxScaler(feature_range=(-1, 1))#--- 
    if normStrategy == 'throughSamples':
        data = scaler1.fit_transform(data)#--- 
    if normStrategy == 'throughTime':
        data = scaler1.fit_transform(torch.transpose(torch.from_numpy(data),0,1))
        data = np.transpose(data)#torch.transpose(torch.from_numpy(data),0,1)
    input = torch.from_numpy(data[nst:, :-1])
    target = torch.from_numpy(data[nst:, 1:])
    test_input = torch.from_numpy(data[:nst, :-1])
    test_target = torch.from_numpy(data[:nst, 1:])
    seq = SequenceOneChannel()
    seq.double()
else:
    data = torch.load('traindataFp1024.pt')#traindataFp1#---
    #data = data / max_int_cam #seems (but not) working test Loss: 0.0012530317884813476
    scaler1 = MinMaxScaler(feature_range=(-1, 1))#--- 
    if normStrategy == 'throughSamples':
        for i in range(data.shape[2]):
            # scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
            # ppp =torch.tensor(scaler.fit_transform(ddT[:,:,i]))
            d = data[:,:,i]
            d = scaler1.fit_transform(d)
            d =torch.tensor(d)
            # d=torch.unsqueeze(torch.tensor(d), 2)
            data[:,:,i] = d#--- 
        # data = scaler1.fit_transform(data)#--- 
    if normStrategy == 'throughTime':
        for i in range(data.shape[2]):
            d = data[:,:,i]
            d = scaler1.fit_transform(torch.transpose(d,0,1))
            d = np.transpose(d)#torch.transpose(torch.from_numpy(data),0,1)
            d =torch.tensor(d)
            data[:,:,i] = d#--- 
        # data = scaler1.fit_transform(torch.transpose(torch.from_numpy(data),0,1))
        # data = np.transpose(data)#torch.transpose(torch.from_numpy(data),0,1)
    input = data[nst:, :-1, :]
    target = data[nst:, 1:, :]
    test_input = data[:nst, :-1, :]
    test_target = data[:nst, 1:, :]
    seq = SequenceMultiChannel()
    seq.double()

# from sklearn.preprocessing import MinMaxScaler#---    
# scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
# data = scaler.fit_transform(data)#--- 

# input = torch.from_numpy(data[nst:, :-1])
# target = torch.from_numpy(data[nst:, 1:])
# test_input = torch.from_numpy(data[:nst, :-1])
# test_target = torch.from_numpy(data[:nst, 1:])
# build the model
# seq = SequenceOneChannel()
# seq.double()
criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)#lr=0.8
# optimizer = optim.Adam(seq.parameters(), lr=0.8)#lr=0.001


n_epochs = opt.steps
model = seq
optimiser = optimizer
loss_fn = criterion
train_input = input
train_target = target
#, test_input, test_target
training_loop1(n_epochs, model, optimiser, loss_fn, 
                  train_input, train_target, test_input, test_target)
def toTest(test_input=test_input, test_target=test_target, scaler=scaler1, ns=normStrategy):    
    #test
    with torch.no_grad():
        # future = 1
        # predF = model(test_input, future=future)
        # # use all pred samples, but only go to 999
        # loss = loss_fn(predF[:, :-future], test_target)
        # y = predF.detach().numpy()
        pred = model(test_input)
        loss = loss_fn(pred, test_target)
        print(pred.shape)
        # print("lossF: {}, lossP: {}".format(loss, lossP))
    if True:
        if normStrategy == 'throughSamples':
            temp_f = torch.cat((torch.zeros(test_target.shape[0],1), test_target),1)#temp with a buffer in the last pos
            s_test_target = scaler.inverse_transform(temp_f)
            temp_f = torch.cat((torch.zeros(test_target.shape[0],1), pred),1)#temp with a buffer in the last pos
            s_pred = scaler.inverse_transform(temp_f)
        if normStrategy == 'throughTime':
            temp_f = torch.cat((test_target, torch.zeros((data.shape[0]-test_target.shape[0]),test_target.shape[1])),0)
            s_test_target = scaler.inverse_transform(torch.transpose(temp_f,0,1))
            s_test_target = torch.transpose(torch.tensor(s_test_target), 0,1)
            temp_f = torch.cat((pred, torch.zeros((data.shape[0]-test_target.shape[0]),49)),0)
            s_pred = scaler.inverse_transform(torch.transpose(temp_f,0,1))
            s_pred = torch.transpose(torch.tensor(s_pred), 0,1)   
        # 
        
        n = train_input.shape[1] # 49
        def drawP(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[:n], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Normalized 501 pixel signal Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        drawP(test_target[0], pred[0], 'r')
        drawP(test_target[1], pred[1], 'b')
        drawP(test_target[2], pred[2], 'g')
        plt.savefig("xxNormTestPredicted1.png", dpi=200)
        plt.close()
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("De-normalized 501 pixel signal Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        drawP(s_test_target[0], s_pred[0], 'r')
        drawP(s_test_target[1], s_pred[1], 'b')
        drawP(s_test_target[2], s_pred[2], 'g')
        plt.savefig("xxDe-normTestPredicted1.png", dpi=200)
        plt.close()        
    # draw figures
    if False:
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] # 49
        def draw(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[:n], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        draw(test_target[0], pred[0], 'r')
        draw(test_target[1], pred[1], 'b')
        draw(test_target[2], pred[2], 'g')
        plt.savefig("__testPredicted1024.png", dpi=200)
        plt.close()
        # print the loss
    print("test Loss: {}".format(loss))
    return test_target, pred
# s_true, s_pred = toTest()
     
#---  
"""
# data = torch.load('traindataFp1.pt')#traindataFp1#---
data = torch.load('traindataFp1.pt')#traindataFp1#---
# data = data / max_int_cam #not working test Loss: 0.0011788546711502719
scaler1 = MinMaxScaler(feature_range=(-1, 1))#--- 
if normStrategy == 'throughSamples':
    data = scaler1.fit_transform(data)#--- 
if normStrategy == 'throughTime':
    data = scaler1.fit_transform(torch.transpose(torch.from_numpy(data),0,1))
    data = np.transpose(data)#torch.transpose(torch.from_numpy(data),0,1)
# input = torch.from_numpy(data[nst:, :-1])
# target = torch.from_numpy(data[nst:, 1:])
test_input = torch.from_numpy(data[:, :-1])
test_target = torch.from_numpy(data[:, 1:])
ss_true, ss_pred = toTest()
# toTest(test_input=test_input, test_target=test_target, scaler=scaler1, ns=normStrategy)
"""

#---
    
# toTest()
def toTest1(test_input=test_input, test_target=test_target):    
    #test
    with torch.no_grad():
        # future = 1
        # predF = model(test_input, future=future)
        # # use all pred samples, but only go to 999
        # loss = loss_fn(predF[:, :-future], test_target)
        # y = predF.detach().numpy()
        pred = model(test_input)
        loss = loss_fn(pred, test_target)
        # print("lossF: {}, lossP: {}".format(loss, lossP))
    # draw figures
    if True:
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Loss_: {}".format(loss))
        plt.xlabel("x pixel")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] # 49
        def drawP(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[:n, 501], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[:n, 501], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        drawP(test_target[0], pred[0], 'r')
        drawP(test_target[1], pred[1], 'b')
        drawP(test_target[2], pred[2], 'g')
        # n = train_input.shape[2] # 1024
        # def draw(y_original, y_pred, colour):
        #     for i in range(train_input.shape[1]):#49
        #         plt.plot(np.arange(n), y_original[i,:n], colour, linewidth=2.0)
        #         plt.plot(np.arange(n), y_pred[i,:n], colour+":", linewidth=2.0)
        #     # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        # draw(test_target[0], pred[0], 'r')
        # draw(test_target[1], pred[1], 'b')
        # draw(test_target[2], pred[2], 'g')
        plt.savefig("zPredicted1024plotpixel501.png", dpi=200)
        plt.close()
        #---
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Loss_: {}".format(loss))
        plt.xlabel("x pixel")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[2] # 49
        def drawS(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[10, :], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[10, :], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        drawS(test_target[0], pred[0], 'r')
        drawS(test_target[1], pred[1], 'b')
        drawS(test_target[2], pred[2], 'g')
        plt.savefig("zPredicted1024plot10pred.png", dpi=200)
        plt.close()
        print("_test Loss: {}".format(loss))
        # print the loss
        return test_target, pred
# s_true, s_pred = toTest1()

"""
s_true, s_pred = toTest1()
dataO = torch.load('traindataFp1024.pt')#traindataFp1#---
dataOtt = dataO[:nst, 1:, :]#test target
#data = data / max_int_cam #seems (but not) working test Loss: 0.0012530317884813476
scaler1 = MinMaxScaler(feature_range=(-1, 1))#--- 
if normStrategy == 'throughSamples':
    for i in range(dataOtt.shape[2]):
        # scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
        # ppp =torch.tensor(scaler.fit_transform(ddT[:,:,i]))
        d = dataOtt[:,:,i]
        d = scaler1.fit_transform(d)
        tt = s_true[:,:,i]
        temp_f = torch.cat((torch.zeros(tt.shape[0],1), tt),1)#temp with a buffer in the last pos
        s_tt = scaler1.inverse_transform(temp_f)
        # d =torch.tensor(d)
        # # d=torch.unsqueeze(torch.tensor(d), 2)
        # data[:,:,i] = d#--- 
    # data = scaler1.fit_transform(data)#--- 
if normStrategy == 'throughTime':
    for i in range(data.shape[2]):
        d = data[:,:,i]
        d = scaler1.fit_transform(torch.transpose(d,0,1))
        d = np.transpose(d)#torch.transpose(torch.from_numpy(data),0,1)
        d =torch.tensor(d)
        data[:,:,i] = d#--- 
    # data = scaler1.fit_transform(torch.transpose(torch.from_numpy(data),0,1))
    # data = np.transpose(data)#torch.transpose(torch.from_numpy(data),0,1)
input = data[nst:, :-1, :]
target = data[nst:, 1:, :]
test_input = data[:nst, :-1, :]
test_target = data[:nst, 1:, :]
seq = SequenceMultiChannel()
seq.double()
"""

def toTest2(test_input=test_input, test_target=test_target, scaler=scaler1, ns=normStrategy):    
    #test
    with torch.no_grad():
        # future = 1
        # predF = model(test_input, future=future)
        # # use all pred samples, but only go to 999
        # loss = loss_fn(predF[:, :-future], test_target)
        # y = predF.detach().numpy()
        pred = model(test_input)
        loss = loss_fn(pred, test_target)
        # print("lossF: {}, lossP: {}".format(loss, lossP))
    if False:
        if normStrategy == 'throughSamples':
            for i in range(data.shape[2]):
                # d = data[:,:,i]
                # d = scaler1.fit_transform(d)
                # d =torch.tensor(d)
                # data[:,:,i] = d#--- 
                tt = test_target[:,:,i]
                temp_f = torch.cat((torch.zeros(tt.shape[0],1), tt),1)#temp with a buffer in the last pos
                s_tt = scaler.inverse_transform(temp_f)
            temp_f = torch.cat((torch.zeros(test_target.shape[0],1), test_target),1)#temp with a buffer in the last pos
            s_test_target = scaler.inverse_transform(temp_f)
            temp_f = torch.cat((torch.zeros(test_target.shape[0],1), pred),1)#temp with a buffer in the last pos
            s_pred = scaler.inverse_transform(temp_f)
        if normStrategy == 'throughTime':
            temp_f = torch.cat((test_target, torch.zeros((data.shape[0]-test_target.shape[0]),test_target.shape[1])),0)
            s_test_target = scaler.inverse_transform(torch.transpose(temp_f,0,1))
            s_test_target = torch.transpose(torch.tensor(s_test_target), 0,1)
            temp_f = torch.cat((pred, torch.zeros((data.shape[0]-test_target.shape[0]),49)),0)
            s_pred = scaler.inverse_transform(torch.transpose(temp_f,0,1))
            s_pred = torch.transpose(torch.tensor(s_pred), 0,1)   
        # 
        
        n = train_input.shape[1] # 49
        def drawP(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[:n], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Normalized 501 pixel signal Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        drawP(test_target[0], pred[0], 'r')
        drawP(test_target[1], pred[1], 'b')
        drawP(test_target[2], pred[2], 'g')
        plt.savefig("xxNormTestPredicted1.png", dpi=200)
        plt.close()
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("De-normalized 501 pixel signal Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        drawP(s_test_target[0], s_pred[0], 'r')
        drawP(s_test_target[1], s_pred[1], 'b')
        drawP(s_test_target[2], s_pred[2], 'g')
        plt.savefig("xxDe-normTestPredicted1.png", dpi=200)
        plt.close()        
    # draw figures
    if False:
        plt.figure(figsize=(12,6))
        # plt.title(f"Step {i+1}")
        plt.title("Loss_: {}".format(loss))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] # 49
        def draw(y_original, y_pred, colour):
            plt.plot(np.arange(n), y_original[:n], colour, linewidth=2.0)
            plt.plot(np.arange(n), y_pred[:n], colour+":", linewidth=2.0)
            # plt.plot(np.arange(n, n+future), yi[n:], colour+":", linewidth=2.0)
        draw(test_target[0], pred[0], 'r')
        draw(test_target[1], pred[1], 'b')
        draw(test_target[2], pred[2], 'g')
        plt.savefig("__testPredicted1024.png", dpi=200)
        plt.close()
        # print the loss
    print("test Loss: {}".format(loss))
    return test_target, pred

#---

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--steps', type=int, default=15, help='steps to run')
#     opt = parser.parse_args()
#     opt.steps = 20 #---
#     # set random seed to 0
#     np.random.seed(0)
#     torch.manual_seed(0)
#     # load data and make training set
#     # data = torch.load('traindatawavw.pt')#
#     data = torch.load('traindataFp1.pt')#traindataFp1#---
#     from sklearn.preprocessing import MinMaxScaler#---    
#     scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
#     data = scaler.fit_transform(data)#--- 
#     nst = 3 #number of samples for test
#     input = torch.from_numpy(data[nst:, :-1])
#     target = torch.from_numpy(data[nst:, 1:])
#     test_input = torch.from_numpy(data[:nst, :-1])
#     test_target = torch.from_numpy(data[:nst, 1:])
#     # build the model
#     seq = SequenceOneChannel()
#     seq.double()
#     criterion = nn.MSELoss()
#     # use LBFGS as optimizer since we can load the whole data to train
#     optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#     #begin to train
#     for i in range(opt.steps):
#         print('STEP: ', i)
#         def closure():
#             optimizer.zero_grad()
#             out = seq(input)
#             loss = criterion(out, target)
#             print('loss:', loss.item())
#             loss.backward()
#             return loss
#         optimizer.step(closure)
#         # begin to predict, no need to track gradient here
#         with torch.no_grad():
#             future = 10
#             pred = seq(test_input, future=future)
#             loss = criterion(pred[:, :-future], test_target)
#             print('test loss:', loss.item())
#             y = pred.detach().numpy()
#         # draw the result
#         plt.figure(figsize=(30,10))
#         plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
#         plt.xlabel('x', fontsize=20)
#         plt.ylabel('y', fontsize=20)
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         def draw(yi, color):
#             plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
#             plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
#         tt = test_target.detach().numpy()
#         tt = tt[2]
#         # draw(tt[0], )
#         plt.plot(np.arange(1,input.size(1)+1), tt[:input.size(1)], 'k', linewidth = 3.0)
#         plt.plot(np.arange(0,input.size(1)), y[2][:input.size(1)], 'r', linewidth = 3.0)
#         # draw(y[0], 'r')
#         # draw(y[1], 'g')
#         # draw(y[2], 'b')
#         if i==0 or i==9 or i==19:
#             plt.savefig('predict%d.pdf'%i)
#             plt.close()
#---



