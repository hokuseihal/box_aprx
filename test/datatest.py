import pickle
import glob
import os
root='../data/doboku/box_aprx/data3'
def datatest(root):
    if os.path.exists(f'{root}/.okdata'):
        with open(f'{root}/.okdata') as f:
            okdata=[l.strip() for l in f.readlines()]
    else:
        okdata=set()
    datapathes=glob.glob(f'{root}/[!error]*.pkl')
    datapathes=list(set(datapathes)-set(okdata))
    for path in datapathes:
        with open(path,'rb') as f:
            data=pickle.load(f)
            if (data[0].shape[0]==116) and (data[1].shape[0]==20):
                print("OK",path)
            else:
                print("NG",path)
                os.remove(path)

    with open(f'{root}/.okdata','a') as f:
        f.writelines("\n".join(datapathes))
if __name__=="__main__":
    datatest(root)
