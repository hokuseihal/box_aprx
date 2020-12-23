import torch.nn as nn

class Bert(nn.Module):
    def __init__(self,num_transformer,d_model,nhead,in_fearure,out_feature):
        super(Bert,self).__init__()
        self.in_fc=nn.Linear(in_fearure,d_model)
        self.transformers=nn.Sequential(
            *[nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead),num_layers=6) for _ in range(num_transformer)]
        )
        self.out_fc=nn.Linear(d_model,out_feature)
    def forward(self,x):
        x=x.unsqueeze(0)
        x=self.in_fc(x)
        x=self.transformers(x)
        x=self.out_fc(x)
        return [x.squeeze(0)]

def Bertbase(in_feature=116,out_feature=20):
    return Bert(num_transformer=12,d_model=768,nhead=12,in_fearure=in_feature,out_feature=out_feature)

def Bertlarge(in_feature=116,out_feature=20):
    return Bert(num_transformer=24,d_model=1024,nhead=16,in_fearure=in_feature,out_feature=out_feature)