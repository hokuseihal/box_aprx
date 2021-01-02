import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kfac import KFAC
from model.FC_Resnet import *
from model.function import log_upper_standerdize
from model.mish import Mish
class InputParam(nn.Module):
    def __init__(self, data,optimizable=64):
        super(InputParam, self).__init__()
        self.optimazable_data=nn.Parameter(torch.tensor(data[:optimizable]))
        self.freeze_data=torch.tensor(data[optimizable:],requires_grad=False)
    def getdata(self):
        return torch.cat([self.optimazable_data,self.freeze_data]).unsqueeze(0).to(device)

def estimate():
    def okng(output):
        return F.relu(output-standardizer(torch.tensor(1.))).mean()
    model.eval()
    for idx in range(est_epochs):
        output = model(inputparam.getdata())[-1]
        loss = okng(output)
        loss.backward()
        data_optimizer.step()
        data_optimizer.zero_grad()
        print(idx,f'est, {loss.item()}')
        if loss<esp:
            print('optimazation is finished.')
            print(inputparam.getdata())
            exit()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default=0,type=int,help='set integer, 0:fc_resnet18,, 1:fc_resnet34,, 2:fc_resnet50,, 3:fc_resnet101, 4:fc_resnet151, 5:fc_resnet304, 6:resnet304_b')
    parser.add_argument('--standardize',default='sigmoid')
    parser.add_argument('--activation',default='relu')
    parser.add_argument('--modelpath',required=True)
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'

    if args.standardize=='log':
        standardizer=log_upper_standerdize
    elif args.standardize=='sigmoid':
        standardizer=torch.sigmoid
    else:
        standardizer=lambda x :x
    if args.activation=='relu':
        activation=nn.ReLU()
    elif args.activation=='mish':
        activation=Mish()

    models=[fc_resnet18,fc_resnet34,fc_resnet50,fc_resnet101,fc_resnet152,fc_resnet304,fc_resnet304_batch]
    model=models[args.model](fn_activate=activation,batchnorm=True).to(device)
    model.load_state_dict(torch.load(args.modelpath))
    est_epochs=1000
    esp=1e-3
    data=np.array([3.02891190e-01, 3.13605135e-01, 5.48877447e-01, 6.40108695e-01,
       4.72233627e-01, 6.98364247e-01, 8.19522781e-01, 4.77520763e-01,
       4.61205840e-01, 9.42414997e-01, 8.02712917e-01, 7.03027491e-01,
       7.91286242e-01, 7.79482202e-01, 3.02572374e-01, 7.42047824e-01,
       4.14928773e-01, 1.34446199e-01, 5.52252838e-02, 7.34882363e-01,
       8.94737765e-01, 2.02604544e-01, 5.53732799e-01, 3.44702161e-01,
       5.33443705e-01, 7.27564771e-01, 3.71085315e-01, 2.51645876e-01,
       9.83150703e-01, 6.41157965e-01, 1.59041993e-01, 5.88446718e-01,
       3.71765966e-02, 2.67708992e-01, 6.26030412e-01, 4.11626518e-01,
       4.48336908e-01, 2.57647894e-02, 5.74232972e-01, 3.27238703e-01,
       1.46239538e-01, 3.89189404e-01, 5.64929751e-02, 3.59313987e-01,
       4.62790378e-01, 7.05257858e-02, 6.96000574e-01, 4.50132263e-01,
       8.66143177e-01, 8.98682113e-01, 2.76011399e-02, 2.77274380e-01,
       7.58162853e-01, 3.76959103e-01, 6.74018987e-01, 9.75906362e-01,
       4.74307515e-01, 8.58860262e-01, 9.49846831e-01, 1.45080025e-01,
       3.13088141e-01, 2.11500904e-01, 7.49924754e-01, 1.43118398e-01,
       2.37830487e-02, 2.39908427e-01, 8.36780616e-01, 7.97915887e-01,
       7.64211796e-01, 3.47701383e-01, 8.12091031e-01, 3.27014336e-01,
       5.39187838e-01, 5.93593277e-01, 7.98873882e-01, 9.64946370e-01,
       3.18545940e-01, 2.51421932e-01, 4.89778150e-01, 7.78014024e-01,
       8.39472311e-01, 3.90316814e-01, 3.23382593e-01, 4.97166879e-02,
       3.49937993e-01, 1.93316607e-01, 4.81165986e-01, 9.18935722e-01,
       8.39098511e-01, 2.36477937e-01, 2.49511963e-01, 7.59650162e-01,
       5.27866536e-01, 7.33184172e-01, 2.51333102e-01, 8.14070140e-01,
       5.28644627e-02, 2.22253847e-01, 7.84305194e-01, 9.49397372e-01,
       3.28507312e-01, 4.93637434e-01, 9.89217677e-01, 2.20508526e-01,
       1.29226287e-01, 1.75203984e-02, 9.38795393e-01, 4.37718462e-01,
       4.24773988e-01, 3.99781693e-05, 1.47536321e-01, 8.00882947e-01,
       7.93773425e-02, 5.79908493e-01, 5.40433333e-01, 2.80219106e-03],dtype=np.float32)
    # data=np.random.random(data.shape).astype(np.float32)
    inputparam = InputParam(data)
    data_optimizer = torch.optim.Adam(inputparam.parameters(),lr=1)
    estimate()