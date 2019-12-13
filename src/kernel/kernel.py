import numpy as np
def kernel_uniform_2D(x,y):
    return np.ones(x.shape)

def kernel_uniform_1D(x):
    return np.ones(x.shape)

def kernel_uniform_3D(x,y,z):
    return np.ones(x.shape)

def kernel_Gauss_2D(x,y,sigma):
    k = 1/(2*np.pi*sigma**2)*np.exp(-(np.power(x,2)+np.power(y,2))/sigma**2)
    return k