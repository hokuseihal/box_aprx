import torch
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import glob
import pickle
import numpy as np
from core import addvalue,save
from multiprocessing import cpu_count
class ResBlock(nn.Module):
    def __init__(self,feature):
        super(ResBlock,self).__init__()
        self.fc1=nn.Linear(feature,feature)
        self.fc2=nn.Linear(feature,feature)

    def forward(self,x):
        _x=x
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=_x+x
        x=F.relu(x)
        return x

class Model(nn.Module):
    def __init__(self,num_layer,feature,num_model,fn_activate=nn.ReLU()):
        super(Model, self).__init__()
        self.infc=nn.Linear(116,feature)
        self.fc=nn.ModuleList([nn.Sequential(*[ResBlock(feature) for _ in range(num_layer)]) for _ in range(num_model)])
        self.outfc=nn.ModuleList([nn.Linear(feature,20) for _ in range(num_model)])
        self.activate=fn_activate
        self.num_model=num_model
    def forward(self,x):
        x=self.activate(self.infc(x))
        ret=[]
        for i in range(self.num_model):
            x=self.fc[i](x)
            ret.append(self.outfc[i](x))
        return ret

class InputParam(nn.Module):
    def __init__(self,data):
        super(InputParam,self).__init__()
        self.data=data

class Dataset(torch.utils.data.Dataset):
    def __init__(self,root):
        self.data=glob.glob(f'{root}/[!error]*.pkl')
        self.thresh=5
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        with open(self.data[item],'rb') as f:
            data,target=pickle.load(f)
        target[target>self.thresh]=self.thresh
        return data.astype(np.float32),target.astype(np.float32)

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
            loss=sum([lossf(output,target.to(device)) for output in all_output])
            if phase=='train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'{e}:{idx}/{len(loader)}, {loss.item():.4f},{phase}')
            addvalue(writer,f'loss:{phase}',loss.item(),e)

def estimate():
    model.eval()
    # data=loaddata()
    inputparam=InputParam(data)
    data_optimizer=torch.optim.Adam(inputparam.parameters())
    for idx in range(est_epochs):
        output=model(inputparam.data)
        loss=F.relu(output-1).mean()
        loss.backward()
        data_optimizer.step()
        data_optimizer.zero_grad()

    print(inputparam.data)

if __name__=='__main__':
    device='cuda' if torch.cuda.is_available() else 'cpu'
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature',type=int,default=128)
    parser.add_argument('--num_layer',type=int,default=3)
    parser.add_argument('--num_model',type=int,default=1)
    args=parser.parse_args()
    feature=args.feature
    num_layer=args.num_layer
    num_model=args.num_model
    print(f'{device=},{feature=},{num_layer=},{num_model=}')
    model=Model(feature=feature,num_layer=num_layer,num_model=num_model).to(device)
    writer={}
    batchsize=512
    dataset=Dataset('../data/doboku/box_aprx/data3')
    traindataset,valdataset=torch.utils.data.random_split(dataset,[dsize:=int(len(dataset)*0.8),len(dataset)-dsize])
    trainloader=torch.utils.data.DataLoader(traindataset,batch_size=batchsize,num_workers=cpu_count())
    valloader=torch.utils.data.DataLoader(valdataset,batch_size=batchsize,num_workers=cpu_count())
    optimizer=torch.optim.Adam(model.parameters())
    lossf=nn.MSELoss()
    esp=1
    epochs=1000
    est_epochs=200
    savefolder=f'data/{num_model}_{num_layer}_{feature}'
    os.makedirs(savefolder,exist_ok=True)
    data=torch.rand(116)
    for e in range(epochs):
        operate('train')
        operate('val    ')
        save(model,savefolder,writer,f'{feature=},{num_layer=},{num_model=}')