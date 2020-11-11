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





class InputParam(nn.Module):
    def __init__(self, data):
        super(InputParam, self).__init__()
        self.data = data


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


def train():
    model.train()
    for idx, (data, target) in enumerate(dataloader):
        all_output = model(data.to(device))
        loss = sum([lossf(output, target.to(device)) for output in all_output])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'{e}:{idx}/{len(dataloader)}, {loss.item():.4f}')
        addvalue(writer, f'loss:train', loss.item(), e)


def estimate():
    model.eval()
    # data=loaddata()
    inputparam = InputParam(data)
    data_optimizer = torch.optim.Adam(inputparam.parameters())
    for idx in range(est_epochs):
        output = model(inputparam.data)
        loss = F.relu(output - 1).mean()
        loss.backward()
        data_optimizer.step()
        data_optimizer.zero_grad()

    print(inputparam.data)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=int)
    parser.add_argument('--num_layer', type=int)
    parser.add_argument('--num_model', type=int)
    args = parser.parse_args()
    feature = args.feature
    num_layer = args.num_layer
    num_model = args.num_model
    print(f'{device=},{feature=},{num_layer=},{num_model=}')
    model = Model(feature=feature, num_layer=num_layer, num_model=num_model).to(device)
    writer = {}
    batchsize = 512
    dataloader = torch.utils.data.DataLoader(Dataset('../data/doboku/box_aprx/data3'), batch_size=batchsize,
                                             num_workers=cpu_count())
    optimizer = torch.optim.Adam(model.parameters())
    lossf = nn.MSELoss()
    esp = 1
    epochs = 1000
    est_epochs = 200
    savefolder = f'data/{num_model}_{num_layer}_{feature}'
    os.makedirs(savefolder, exist_ok=True)
    for e in range(epochs):
        train()
        save(model, savefolder, writer, f'{feature=},{num_layer=},{num_model=}')
