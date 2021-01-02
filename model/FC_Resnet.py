import torch
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, feature,batchnorm=False):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(feature, feature)
        self.fc2 = nn.Linear(feature, feature)
        self.bn1=nn.BatchNorm1d(feature)
        self.bn2=nn.BatchNorm1d(feature)
        self.batchnorm=batchnorm

    def forward(self, x):
        _x = x
        x = self.fc1(x)
        if self.batchnorm:
            x=self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.batchnorm:
            x=self.bn2(x)
        x = F.relu(x)
        x = _x + x
        x = F.relu(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, feature, fn_act=nn.ReLU(),batchnorm=False):
        super(BottleNeck, self).__init__()
        self.fcs = nn.ModuleList([nn.Linear(feature, feature) for _ in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(feature) for _ in range(3)])
        self.fn_act = fn_act
        self.batchnorm=batchnorm
    def forward(self, x):
        _x = x
        x = self.fcs[0](x)
        if self.batchnorm:
            x=self.bns[0](x)
        x=self.fn_act(x)
        x = self.fcs[1](x)
        if self.batchnorm:
            x=self.bns[1](x)
        x=self.fn_act(x)
        x = self.fcs[2](x)
        if self.batchnorm:
            x=self.bns[2](x)
        return self.fn_act(x + _x)


class FC_ResNet(nn.Module):
    def __init__(self, block, layers=(1, 2, 3, 4), features=(128, 128, 256, 512), fn_activate=nn.ReLU(),flow_loss=False,batchnorm=True,postactivator=lambda x:x):
        super(FC_ResNet, self).__init__()
        self.infc = nn.Linear(116, features[0])
        self.fcs = nn.ModuleList(
            [nn.Sequential(
                *[block(feature=features[idx],batchnorm=batchnorm) for _ in range(l)]
            ) for idx, l in enumerate(layers)]
        )
        self.changer=nn.ModuleList([nn.Linear(in_feature,out_feature) for in_feature,out_feature in zip(features[:-1],features[1:])])
        self.outfc = nn.Linear(features[-1], 20)
        self.activate = fn_activate
        self.flow_loss=flow_loss
        self.postactivator=postactivator
    def forward(self, x):
        x = self.activate(self.infc(x))
        for idx,fc in enumerate(self.fcs):
            x = fc(x)
            if idx<3:
                x=self.changer[idx](x)
        x=self.outfc(x)
        x=self.postactivator(x)
        return [x]


def fc_resnet18(**kwargs):
    return FC_ResNet(layers=[2, 2, 2, 2], block=BasicBlock,**kwargs)


def fc_resnet34(**kwargs):
    return FC_ResNet(layers=[3, 4, 6, 3], block=BasicBlock,**kwargs)


def fc_resnet50(**kwargs):
    return FC_ResNet(layers=[3, 4, 6, 3], block=BottleNeck,**kwargs)


def fc_resnet101(**kwargs):
    return FC_ResNet(layers=[3, 4, 23, 3], block=BottleNeck,**kwargs)


def fc_resnet152(**kwargs):
    return FC_ResNet(layers=[3, 8, 36, 3], block=BottleNeck,**kwargs)

def fc_resnet304(**kwargs):
    return FC_ResNet(layers=[6,16,73,6],block=BottleNeck,features=(256,256,512,1024),**kwargs)

def fc_resnet304_batch(**kwargs):
    return FC_ResNet(layers=[6,16,73,6],block=BottleNeck,features=(256,256,512,1024),**kwargs)

if __name__=='__main__':
    model=fc_resnet304().cuda()
    print(model)
    optimizer=torch.optim.Adam(model.parameters())
    output=model(torch.randn(32,116).cuda())
    loss=output[-1].mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()