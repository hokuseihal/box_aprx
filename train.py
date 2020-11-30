import argparse
import glob
import os
import pickle
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from test.datatest import datatest
from core import addvalue, save
from model.FC_Resnet import fc_resnet18,fc_resnet34,fc_resnet50,fc_resnet101,fc_resnet152


class InputParam(nn.Module):
    def __init__(self, data,device,optimizable=64):
        super(InputParam, self).__init__()
        self.optimazable_data=nn.Parameter(torch.tensor(data[:optimizable]))
        self.freeze_data=torch.tensor(data[optimizable:])
        self.data=torch.cat([self.optimazable_data,self.freeze_data]).unsqueeze(0).to(device)
    def forward(self):
        return -F.threshold(F.relu(self.data),-1,-1)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root,thresh=np.inf):
        datatest(root)
        _data=[]
        with open(f'{root}/.okdata') as f:
            lines=f.readlines()
        for i in range(len(lines)//2):
            p,m=lines[i*2:(i+1)*2]
            p=p.strip()
            m=np.float(m)
            if thresh>m:
                _data.append(p)
                print(p,m)
        self.data = _data
        self.thresh = thresh

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        with open(self.data[item], 'rb') as f:
            data, target = pickle.load(f)
        if self.thresh:
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
    inputparam = InputParam(data,device=device)
    data_optimizer = torch.optim.Adam(inputparam.parameters(),lr=0.1)
    for idx in range(est_epochs):
        output = model(inputparam.data)[-1]
        loss = okng(output)
        loss.backward()
        data_optimizer.step()
        data_optimizer.zero_grad()
        print(idx,f'est, {loss.item()}')
        if loss<esp:
            print('optimazation is finished.')
            print(inputparam.data)
            exit()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--linux',default=False,action='store_true')
    parser.add_argument('--thresh',default=np.inf,type=float)
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=fc_resnet18().to(device)
    writer={}
    esp=1e-3
    batchsize=1024
    dataset=Dataset('J:/data3',args.thresh) if not args.linux else Dataset('../data/doboku/box_aprx/data3',args.thresh)
    traindataset,valdataset=torch.utils.data.random_split(dataset,[dsize:=int(len(dataset)*0.8),len(dataset)-dsize])
    trainloader=torch.utils.data.DataLoader(traindataset,batch_size=batchsize,num_workers=cpu_count(),shuffle=True)
    valloader=torch.utils.data.DataLoader(valdataset,batch_size=batchsize,num_workers=cpu_count(),shuffle=True)
    optimizer=torch.optim.Adam(model.parameters())
    lossf=nn.MSELoss()
    epochs=100
    est_epochs=200
    savefolder=f'data/tmp'
    os.makedirs(savefolder,exist_ok=True)
    data=torch.rand(116)
    for e in range(epochs):
        operate('train')
        operate('val')
        save(model,savefolder,writer,f'')
    estimate()