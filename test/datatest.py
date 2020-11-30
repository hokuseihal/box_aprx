import pickle
import glob
import os
root='../data/doboku/box_aprx/data3'
def datatest(root):
    memopath='.okdata'
    if os.path.exists(f'{root}/{memopath}'):
        with open(f'{root}/{memopath}') as f:
            okdata=[l.strip() for l in f.readlines()]
    else:
        okdata=[]
    datapathes=glob.glob(f'{root}/[!error]*.pkl')
    checkdatapathes=list(set(datapathes)-set(okdata[::2]))
    ma=[]
    for path in checkdatapathes:
        with open(path,'rb') as f:
            data=pickle.load(f)
            if (data[0].shape[0]==116) and (data[1].shape[0]==20):
                ma.append(data[1].max())
                print("OK",path,ma[-1])
            else:
                print("NG",path)
                os.remove(path)

    with open(f'{root}/.okdata','a') as f:
        for p,m in zip(checkdatapathes,ma):
            f.write(f'{p}\n{str(m)}\n')
if __name__=="__main__":
    datatest(root)
