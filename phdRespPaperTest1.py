

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
        self.n_hidden = 15
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
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
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    opt.steps = 20 #---
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    # data = torch.load('traindatawavw.pt')#
    data = torch.load('traindataFp1.pt')#traindataFp1#---
    from sklearn.preprocessing import MinMaxScaler#---    
    scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
    data = scaler.fit_transform(data)#--- 
    nst = 3 #number of samples for test
    input = torch.from_numpy(data[nst:, :-1])
    target = torch.from_numpy(data[nst:, 1:])
    test_input = torch.from_numpy(data[:nst, :-1])
    test_target = torch.from_numpy(data[:nst, 1:])
    # build the model
    seq = SequenceOneChannel()
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(opt.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 10
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        tt = test_target.detach().numpy()
        tt = tt[2]
        # draw(tt[0], )
        plt.plot(np.arange(1,input.size(1)+1), tt[:input.size(1)], 'k', linewidth = 3.0)
        plt.plot(np.arange(0,input.size(1)), y[2][:input.size(1)], 'r', linewidth = 3.0)
        # draw(y[0], 'r')
        # draw(y[1], 'g')
        # draw(y[2], 'b')
        if i==0 or i==9 or i==19:
            plt.savefig('predict%d.pdf'%i)
            plt.close()
#---
