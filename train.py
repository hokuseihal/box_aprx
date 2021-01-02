import argparse
import os

import numpy as np
import tensorflow as tf

from core import addvalue, save
from kfac import KFAC
from model.FC_Resnet import *
from model.function import log_upper_standerdize
from model.mish import Mish


def parse_batch_example(example):
    features = tf.io.parse_example(example, features={
        "x": tf.io.FixedLenFeature([116], dtype=tf.float32),
        "y": tf.io.FixedLenFeature([20], dtype=tf.float32)
    })
    x = features["x"]
    y = features["y"]
    return x, y


def operate(phase):
    if phase == 'train':
        model.train()
        loader = tf.data.TFRecordDataset(f'{datafolder}/train.tfrecord').batch(batchsize).map(parse_batch_example)
    else:
        model.eval()
        loader = tf.data.TFRecordDataset(f'{datafolder}/val.tfrecord').batch(batchsize).map(parse_batch_example)

    for idx, (data, target) in enumerate(iter(loader)):
        data = torch.from_numpy(data.numpy()).to(device)
        target = torch.from_numpy(target.numpy()).to(device)
        target = standardizer(target)
        with torch.set_grad_enabled(phase == 'train'):
            all_output = model(data.to(device))
            loss = sum([lossf(output, target.to(device)) for output in all_output]) / len(all_output)
            if phase == 'train':
                loss.backward()
                preconditioner.step()
                optimizer.step()
                optimizer.zero_grad()
            print(f'{e}:{idx}, {loss.item():.4f},{phase}')
            addvalue(writer, f'loss:{phase}', loss.item(), e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--linux',default=False,action='store_true')
    parser.add_argument('--round_thresh', default=np.inf, type=float)
    parser.add_argument('--cut_thresh', default=np.inf, type=float)
    parser.add_argument('--model', default=0, type=int,
                        help='set integer, 0:fc_resnet18,, 1:fc_resnet34,, 2:fc_resnet50,, 3:fc_resnet101, 4:fc_resnet151, 5:fc_resnet304, 6:resnet304_b')
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--renew_dataset', default=False, action='store_true')
    parser.add_argument('--standardize', default='sigmoid')
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--nonbatchnorm', default=False, action='store_true')
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--postactivator', default='sigmoid')
    args = parser.parse_args()
    # device='cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'mish':
        activation = Mish()
    models = [fc_resnet18, fc_resnet34, fc_resnet50, fc_resnet101, fc_resnet152, fc_resnet304, fc_resnet304_batch]
    writer = {}
    esp = 1e-3
    batchsize = args.batchsize
    datafolder = '/opt/data/doboku'
    if args.postactivator == 'sigmoid':
        postactivator = torch.sigmoid
    else:
        postactivator = lambda x: x

    model = models[args.model](fn_activate=activation, batchnorm=not args.nonbatchnorm, postactivator=postactivator).to(
        device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 1e-3)

    if args.standardize == 'log':
        standardizer = log_upper_standerdize
    elif args.standardize == 'sigmoid':
        standardizer = torch.sigmoid
    else:
        standardizer = lambda x: x
    preconditioner = KFAC(model, 0.1)
    lossf = nn.MSELoss()
    epochs = 100
    est_epochs = 200
    savefolder = f'data/{args.savefolder}'
    os.makedirs(savefolder, exist_ok=True)
    data = torch.rand(116)
    for e in range(epochs):
        operate('train')
        operate('val')
        save(model, savefolder, writer, args.__dict__, f'')
