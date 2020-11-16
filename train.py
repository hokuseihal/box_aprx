import argparse
import glob
import os
import pickle
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import addvalue, save
from model.FC_Resnet import fc_resnet18,fc_resnet34,fc_resnet50,fc_resnet101,fc_resnet152


class InputParam(nn.Module):
    def __init__(self, data):
        super(InputParam, self).__init__()
        self.optimizable=np.zeros(56,dtype=np.bool)
        self.optimizable[3:14]=True
        self.data=torch.stack([nn.Parameter(torch.rand(1),requires_grad=self.optimizable[i]) for i in range(56)])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.data = glob.glob(f'{root}/[!error]*.pkl')
        self.thresh = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        with open(self.data[item], 'rb') as f:
            data, target = pickle.load(f)
        target[target > self.thresh] = self.thresh
        return data.astype(np.float32), target.astype(np.float32)


def operate(phase):
    if phase=='train':
        model.train()
        loader=trainloader
    else:
        model.eval()
        loader=valloader

    for idx,(data,target) in enumerate(loader):
        with torch.set_grad_enabled(phase=='train'):
            all_output=model(data.to(device))
            loss=sum([lossf(output,target.to(device)) for output in all_output])/len(all_output)
            if phase=='train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'{e}:{idx}/{len(loader)}, {loss.item():.4f},{phase}')
            addvalue(writer,f'loss:{phase}',loss.item(),e)

def estimate():
    def okng(output):
        return F.relu(output-1).mean()
    model.eval()
    inputparam = InputParam(data)
    data_optimizer = torch.optim.Adam(inputparam.parameters())
    for idx in range(est_epochs):
        output = model(inputparam.data)
        loss = okng(output)
        loss.backward()
        data_optimizer.step()
        data_optimizer.zero_grad()

    print(inputparam.data)

if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    parser=argparse.ArgumentParser()
    args=parser.parse_args()
    model=fc_resnet18().to(device)
    writer={}
    batchsize=1024
    dataset=Dataset('J:/data3')
    traindataset,valdataset=torch.utils.data.random_split(dataset,[dsize:=int(len(dataset)*0.8),len(dataset)-dsize])
    trainloader=torch.utils.data.DataLoader(traindataset,batch_size=batchsize,num_workers=cpu_count(),shuffle=True)
    valloader=torch.utils.data.DataLoader(valdataset,batch_size=batchsize,num_workers=cpu_count(),shuffle=True)
    optimizer=torch.optim.Adam(model.parameters())
    lossf=nn.MSELoss()
    esp=1
    epochs=1000
    est_epochs=200
    savefolder=f'data/FC_152'
    os.makedirs(savefolder,exist_ok=True)
    data=torch.rand(116)
    for e in range(epochs):
        # operate('train')
        # operate('val')
        # save(model,savefolder,writer,f'')
        estimate()