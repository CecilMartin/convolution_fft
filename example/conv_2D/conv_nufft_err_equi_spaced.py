# Test convolution for 2D
# %%
from __future__ import division, print_function
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import convolution_fft as conv
import kernel.kernel as kernel_mod
import numpy as np
import multiprocessing
import pyfftw
import sys
import scipy.io as sio

# import copy

#%%
nj = 16
Lx, Ly = [10, 10]

periodic = np.array([0, 0])
sigma = 1
L = [Lx, Ly]


def kernel_handle(x, y): return kernel_mod.kernel_Gauss_2D(x, y, sigma)


repeat = 8
num = 8
err = np.zeros((num, repeat))

for i in range(num):
    nx = ny = int(round(np.sqrt(nj)))* (2**i)
    for j in range(repeat):
        x = (np.linspace(0, nx-1, nx)+0.5)*Lx/nx
        y = (np.linspace(0, ny-1, ny)+0.5)*Ly/ny
        x,y = np.meshgrid(x,y,indexing = 'ij')
        x = x.flatten()
        y = y.flatten()
        scalar = np.array([x, y])
        x_grid = np.linspace(0, nx-1, nx)*Lx/nx
        y_grid = np.linspace(0, ny-1, ny)*Ly/ny

        kernel = conv.kernel_evaluate(
            [x_grid, y_grid], kernel_handle, periodic, L, 'ij')

        a = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            0, 1), flags=('FFTW_MEASURE', ))

        a[:][:] = kernel
        kernel_hat1D = (fft_object()).copy()*Lx/nx*Ly/ny
        kernel_hat = np.array([kernel_hat1D, kernel_hat1D])
        source_strenth = np.ones(nx*ny)
        num_modes = [2*nx, 2*ny]


        v = conv.vel_convolution_nufft(
            scalar, source_strenth, num_modes, L, kernel_hat=kernel_hat)
        v_direct = conv.vel_direct_convolution(
            scalar, source_strenth, kernel_handle, L, periodic)
        # print(np.isclose(v[:,0],v_direct))
        err[i, j] = np.sqrt(np.sum(np.power(v[:, 0]-v_direct, 2))) / \
            np.sqrt(np.sum(v_direct**2))


# %%

save_fn = 'err.mat'

save_array = err

sio.savemat(save_fn,{'err':save_array})

# %%
