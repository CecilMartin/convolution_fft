# Test convolution for 2D
# %%
from __future__ import division, print_function
import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import pyfftw
import multiprocessing
# import copy
import numpy as np
import kernel.kernel as kernel_mod
import convolution_fft as conv

#%%
# 2D NUFFT based-particles
# Randomly distributed particle
# Constant source_strenth with constant kernel
# kernel_flag = 2
nj = 16
Lx, Ly = [10, 10]
eps = 1e-8
nx = ny = int(np.ceil(np.sqrt(nj)*5))
x = (np.random.rand(nj))*Lx
y = (np.random.rand(nj))*Ly
scalar = np.array([x,y])
periodic = np.array([1, 0])
kernel = np.ones([2*nx, 2*ny])*(1+2*periodic[0])*(1+2*periodic[1])

a = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))

a[:][:] = kernel
kernel_hat1D = (fft_object()).copy()*Lx/nx*Ly/ny
kernel_hat = np.array([kernel_hat1D,kernel_hat1D])
source_strenth = np.ones(nj)
num_modes = [2*nx, 2*ny]
L = [Lx, Ly]

v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, eps = eps, kernel_hat = kernel_hat)
print(np.allclose(v, np.ones([nj, 2]) * nj * (1+2*periodic[0])*(1+2*periodic[1])))
v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel_mod.kernel_uniform_2D, L, periodic)
print(np.allclose(v[:,0],v_direct))
#%%
# 2D NUFFT based-particles
# Uniformly distributed particle
# Constant source_strenth with constant kernel
# kernel_flag = 2
nj = 64
Lx, Ly = [10, 10]
nx = ny = 8
eps = 1e-8
x = (np.linspace(0, nx-1, nx)+0.5)*Lx/nx
y = (np.linspace(0, ny-1, ny)+0.5)*Ly/ny
x,y = np.meshgrid(x,y,indexing = 'ij')
x = x.flatten()
y = y.flatten()
scalar = np.array([x,y])
periodic = np.array([1, 0])
kernel = np.ones([2*nx, 2*ny])*(1+2*periodic[0])*(1+2*periodic[1])

a = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))

a[:][:] = kernel
kernel_hat1D = (fft_object()).copy()*Lx/nx*Ly/ny
kernel_hat = np.array([kernel_hat1D,kernel_hat1D])
source_strenth = np.ones(nj)
num_modes = [2*nx, 2*ny]
L = [Lx, Ly]

v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, eps = eps, kernel_hat = kernel_hat)
print(np.allclose(v, np.ones([nj, 2]) * nj * (1+2*periodic[0])*(1+2*periodic[1])))

v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel_mod.kernel_uniform_2D, L, periodic)
print(np.allclose(v[:,0],v_direct))

#%%
# 2D NUFFT based-particles
# Uniformly distributed particle
# Constant source_strenth with Gaussian kernel
# kernel_flag = 2
nj = 64
Lx, Ly = [10, 10]
nx = ny = 8
eps = 1e-8
x = (np.linspace(0, nx-1, nx)+0.5)*Lx/nx
y = (np.linspace(0, ny-1, ny)+0.5)*Ly/ny
x,y = np.meshgrid(x,y,indexing = 'ij')
x = x.flatten()
y = y.flatten()
scalar = np.array([x,y])
periodic = np.array([1, 0])
sigma = 1
L = [Lx, Ly]
x_grid = np.linspace(0,nx-1,nx)*Lx/nx
y_grid = np.linspace(0,ny-1,ny)*Ly/ny
kernel_handle = lambda x, y: kernel_mod.kernel_Gauss_2D(x,y,sigma)
kernel = conv.kernel_evaluate([x_grid,y_grid], kernel_handle, periodic, L, 'ij')

a = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))

a[:][:] = kernel
kernel_hat1D = (fft_object()).copy()*Lx/nx*Ly/ny
kernel_hat = np.array([kernel_hat1D,kernel_hat1D])
source_strenth = np.ones(nj)
num_modes = [2*nx, 2*ny]


v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, eps = eps, kernel_hat = kernel_hat)
v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)
# print(np.allclose(v[:,0],v_direct))
print(np.sqrt(np.sum(np.power(v[:,0]-v_direct,2)))/np.sqrt(np.sum(v_direct**2)))


#%%
# 2D NUFFT based-particles
# Randomly distributed particle
# Constant source_strenth with Gaussian kernel
# kernel_flag = 2

nj = 16
Lx, Ly = [10, 10]
eps = 1e-8
nx = ny = int(np.ceil(np.sqrt(nj)*5))
x = (np.random.rand(nj))*Lx
y = (np.random.rand(nj))*Ly
scalar = np.array([x,y])
periodic = np.array([0, 0])
sigma = 1
L = [Lx, Ly]
x_grid = np.linspace(0,nx-1,nx)*Lx/nx
y_grid = np.linspace(0,ny-1,ny)*Ly/ny
kernel_handle = lambda x, y: kernel_mod.kernel_Gauss_2D(x,y,sigma)
kernel = conv.kernel_evaluate([x_grid,y_grid], kernel_handle, periodic, L, 'ij')

a = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))

a[:][:] = kernel
kernel_hat1D = (fft_object()).copy()*Lx/nx*Ly/ny
kernel_hat = np.array([kernel_hat1D,kernel_hat1D])
source_strenth = np.ones(nj)
num_modes = [2*nx, 2*ny]


v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, eps = eps,  kernel_hat = kernel_hat)
v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)
# print(np.isclose(v[:,0],v_direct))
print(np.sqrt(np.sum(np.power(v[:,0]-v_direct,2)))/np.sqrt(np.sum(v_direct**2)))

#%%
