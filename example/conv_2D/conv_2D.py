# Test convolution for 2D
# %%
from __future__ import division, print_function
import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import pyfftw
import multiprocessing
# import copy
import numpy as np
import kernel.kernel as kernel
import convolution_fft as conv

# %%
nx, ny = [4, 4]
Lx, Ly = [10, 10]
dx, dy = [Lx/nx, Ly/ny]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
periodic = np.array([1, 0])
method = 1
if method == 1:
    a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
    # Save efforts by knowing that a is real
    b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
    # Real to complex FFT Over the both axes
    fft_object = pyfftw.FFTW(a, b, axes=(
        0, 1), flags=('FFTW_MEASURE', ))
    ifft_object = pyfftw.FFTW(b, a, axes=(
        0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
else:
    raise Exception('Only Method = 1 is supported now.')

scalar = np.ones([ny, nx])

v = conv.vel_convolution_fft(scalar, L=[Lx,Ly], x=[x,y], kernel_handle=kernel.kernel_uniform_2D,
                             periodic=periodic, fft_object=fft_object, ifft_object=ifft_object)

# judge whether this test is successful
print(np.allclose(v,np.ones([ny,nx])*ny*nx*(1+2*periodic[0])*(1+2*periodic[1])))


# %%
# 1D

if method == 1:
    a = pyfftw.empty_aligned((2*nx), dtype='float64')
    # Save efforts by knowing that a is real
    b = pyfftw.empty_aligned((nx+1), dtype='complex128')
    # Real to complex FFT Over the both axes
    fft_object = pyfftw.FFTW(a, b, axes=(
        -1,), flags=('FFTW_MEASURE', ))
    ifft_object = pyfftw.FFTW(b, a, axes=(
        -1,), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
else:
    raise Exception('Only Method = 1 is supported now.')

scalar = np.ones(nx)

v = conv.vel_convolution_fft(scalar, L=[Lx], x=[x], kernel_handle=kernel.kernel_uniform_1D,
                             periodic=periodic[1:], fft_object=fft_object, ifft_object=ifft_object)

# judge whether this test is successful
print(np.allclose(v, np.ones(nx) *
                  nx*(1+2*periodic[1])))


# %%
# 2D
nx, ny = [4, 4]
Lx, Ly = [10, 10]
dx, dy = [Lx/nx, Ly/ny]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
scalar = np.ones([ny, nx])
periodic = np.array([1, 0])
kernel = np.ones([ny*2, nx*2])*(1+2*periodic[0])*(1+2*periodic[1])
v = conv.vel_convolution_fft(scalar, kernel=kernel)
print(np.allclose(v, np.ones([ny, nx])*ny *
                  nx*(1+2*periodic[0])*(1+2*periodic[1])))


#%%
# 3D
nx, ny, nz = [4, 4, 4]
Lx, Ly, Lz= [10, 10, 10]
dx, dy, dz = [Lx/nx, Ly/ny, Lz/nz]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
scalar = np.ones([ny, nx, nz])
periodic = np.array([1, 0, 2])
kernel = np.ones([ny*2, nx*2, nz*2])*(1+2*periodic[0])*(1+2*periodic[1])*(1+2*periodic[2])
v = conv.vel_convolution_fft(scalar, kernel=kernel)
print(np.allclose(v, np.ones([ny, nx, nz])*ny *
                  nx*nz*(1+2*periodic[0])*(1+2*periodic[1])*(1+2*periodic[2])))

#%%
# 2D NUFFT based-particles
nj = 16
Lx, Ly = [10, 10]
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

v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, kernel_hat = kernel_hat)
print(np.allclose(v, np.ones([nj, 2]) * nj * (1+2*periodic[0])*(1+2*periodic[1])))
v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel.kernel_uniform_2D, L, periodic)
print(np.allclose(v,v_direct))
#%%
# 2D NUFFT based-particles
nj = 64
Lx, Ly = [10, 10]
nx = ny = 8
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

v = conv.vel_convolution_nufft(scalar, source_strenth, num_modes, L, kernel_hat = kernel_hat)
print(np.allclose(v, np.ones([nj, 2]) * nj * (1+2*periodic[0])*(1+2*periodic[1])))

#%%
