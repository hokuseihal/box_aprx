import pickle
import glob
import os
root='../data/doboku/box_aprx/data3'
def datatest(root):
    datapathes=glob.glob(f'{root}/[!error]*.pkl')
    for path in datapathes:
        with open(path,'rb') as f:
            data=pickle.load(f)
            if (data[0].shape[0]==116) and (data[1].shape[0]==20):
                print("OK",path)
            else:
                print("NG",path)
                os.remove(path)

if __name__=="__main__":
    datatest(root)