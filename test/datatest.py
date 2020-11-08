import pickle
import glob
root='../data/doboku/box_aprx/data3'
datapathes=glob.glob(f'{root}/[!error]*.pkl')

for path in datapathes:
    with open(path,'rb') as f:
        data=pickle.load(f)
        print(data[0].shape,data[1].shape)
        assert data[0].shape[0]==116,f'{path},{data}'
        assert data[1].shape[0]==20
