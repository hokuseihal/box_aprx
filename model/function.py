import numpy as np

def log_upper_standerdize(x):
    x[x>np.e]=np.log(x[x>np.e])+np.e-1
    return x

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.plot(log_upper_standerdize(np.linspace(0,10,1000)))
    plt.show()