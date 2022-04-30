#gen data


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

# import numpy as np
# import h5py
# f = h5py.File('\phdPython\Fp50C1.mat','r')
# data = f.get('data/variable1')
# data = np.array(data) # For converting to a NumPy array

import numpy as np
from scipy.io import loadmat
import torch

max_int_cam = 62780

a = loadmat('Fp50C1Mex.mat')
d = np.array(a['dM']).astype('float64')
# from sklearn.preprocessing import MinMaxScaler#---    
# scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
# d = scaler.fit_transform(d)#--- 
torch.save(d, open('traindataFp1ex.pt', 'wb'))

# data = torch.load('Fp50C1.mat')
ii = torch.from_numpy(d[3:, :-1])
# target = torch.from_numpy(data[3:, 1:])
# test_input = torch.from_numpy(data[:3, :-1])
# test_target = torch.from_numpy(data[:3, 1:])

#---
aa = loadmat('Fp50C1024ex.mat')
dd = aa['data']
# dd = np.array(aa['data']).astype('float64')
ddT = torch.zeros(0, dd[0][0].shape[1], dd[0][0].shape[0] )
for i in dd:
    print(i[0].shape)
    temp = torch.transpose(torch.tensor(np.array(i[0]).astype('float64')), 0, 1)
    print(temp.shape)
    temp = torch.unsqueeze(temp, 0)
    ddT = torch.cat((ddT, temp), 0)
for i in range(ddT.shape[2]):
    # scaler = MinMaxScaler(feature_range=(-1, 1))#--- 
    # ppp =torch.tensor(scaler.fit_transform(ddT[:,:,i]))
    ppp =torch.tensor(ddT[:,:,i])
    # ppp=torch.unsqueeze(ppp, 1)
    ddT[:,:,i] = ppp#--- 
torch.save(ddT, open('traindataFp1024ex.pt', 'wb'))
