import argparse
import os
import tensorflow as tf
import numpy as np
import torch
from model.function import log_upper_standerdize
import torch.nn as nn
import torch.nn.functional as F
from core import addvalue, save
from model.FC_Resnet import fc_resnet18,fc_resnet34,fc_resnet50,fc_resnet101,fc_resnet152,fc_resnet304


class InputParam(nn.Module):
    def __init__(self, data,device,optimizable=64):
        super(InputParam, self).__init__()
        self.optimazable_data=nn.Parameter(torch.tensor(data[:optimizable]))
        self.freeze_data=torch.tensor(data[optimizable:])
        self.data=torch.cat([self.optimazable_data,self.freeze_data]).unsqueeze(0).to(device)
    def forward(self):
        return -F.threshold(F.relu(self.data),-1,-1)

def parse_batch_example(example):
    features = tf.io.parse_example(example, features={
        "x": tf.io.FixedLenFeature([116], dtype=tf.float32),
        "y": tf.io.FixedLenFeature([20], dtype=tf.float32)
    })
    x = features["x"]
    y = features["y"]
    return x, y


def operate(phase):
    if phase=='train':
        model.train()
        loader=tf.data.TFRecordDataset(f'{datafolder}/train.tfrecord').batch(batchsize).map(parse_batch_example)
    else:
        model.eval()
        loader=tf.data.TFRecordDataset(f'{datafolder}/val.tfrecord').batch(batchsize).map(parse_batch_example)

    for idx,(data,target) in enumerate(iter(loader)):
        data=torch.from_numpy(data.numpy()).to(device)
        target=torch.from_numpy(target.numpy()).to(device)
        if args.standardize=='log':
            target=log_upper_standerdize(target)
        with torch.set_grad_enabled(phase=='train'):
            all_output=model(data.to(device))
            loss=sum([lossf(output,target.to(device)) for output in all_output])/len(all_output)
            if phase=='train':
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f'{e}:{idx}, {loss.item():.4f},{phase}')
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--linux',default=False,action='store_true')
    parser.add_argument('--round_thresh',default=np.inf,type=float)
    parser.add_argument('--cut_thresh',default=np.inf,type=float)
    parser.add_argument('--model',default=0,type=int,help='set integer, 0:fc_resnet18,, 1:fc_resnet34,, 2:fc_resnet50,, 3:fc_resnet101, 4:fc_resnet151, 5:fc_resnet304')
    parser.add_argument('--batchsize',default=256,type=int)
    parser.add_argument('--renew_dataset',default=False,action='store_true')
    parser.add_argument('--standardize',default='log')
    args=parser.parse_args()
    # device='cuda' if torch.cuda.is_available() else 'cpu'
    device='cuda'
    models=[fc_resnet18,fc_resnet34,fc_resnet50,fc_resnet101,fc_resnet152,fc_resnet304]
    model=models[args.model]().to(device)
    writer={}
    esp=1e-3
    batchsize=args.batchsize
    datafolder='D:/data3' if not args.linux else '../data/doboku'
    optimizer=torch.optim.Adam(model.parameters())
    lossf=nn.MSELoss()
    epochs=100
    est_epochs=200
    savefolder='data/'+"_".join([f'{k}={args.__dict__[k]}' for k in args.__dict__])
    os.makedirs(savefolder,exist_ok=True)
    data=torch.rand(116)
    for e in range(epochs):
        operate('train')
        operate('val')
        save(model,savefolder,writer,args.__dict__,f'')
    # estimate()