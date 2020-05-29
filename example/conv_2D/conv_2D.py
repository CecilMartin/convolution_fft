# Test convolution for 2D
# %%
from __future__ import division, print_function
import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import pyfftw
# import copy
import numpy as np
import kernel.kernel as kernel_mod
import convolution_fft as conv

# %%
# Convolution of constant scalar with constant kernel in 2D
# kernel_flag = 1
nx, ny = [4, 4]
Lx, Ly = [10, 10]
dx, dy = [Lx/nx, Ly/ny]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
periodic = np.array([1, 0])
method = 1
if method == 1:
    fft_object, ifft_object = kernel_mod.create_fftw_plan([nx,ny])
    a = fft_object.input_array
    b = fft_object.output_array
else:
    raise Exception('Only Method = 1 is supported now.')
scalar = np.ones([ny, nx])

kernel_hat = kernel_mod.kernel_fft_evaluate(L=[Lx,Ly], x=[x,y], kernel_handle=kernel_mod.kernel_uniform_2D,
                             periodic=periodic, fft_object=fft_object, ifft_object=ifft_object)
v = conv.vel_convolution_fft(scalar, kernel_hat, fft_object=fft_object, ifft_object=ifft_object)

# judge whether this test is successful
print(np.allclose(v,np.ones([ny,nx])*ny*nx*(1+2*periodic[0])*(1+2*periodic[1])))


# %%
# Convolution of constant scalar with constant kernel in 1D
# kernel_flag = 1

if method == 1:
    fft_object, ifft_object = kernel_mod.create_fftw_plan([nx])
    a = fft_object.input_array
    b = fft_object.output_array
else:
    raise Exception('Only Method = 1 is supported now.')

scalar = np.ones(nx)

kernel_hat = kernel_mod.kernel_fft_evaluate(L=[Lx], x=[x], kernel_handle=kernel_mod.kernel_uniform_1D,
                             periodic=periodic[1:], fft_object=fft_object, ifft_object=ifft_object)
v = conv.vel_convolution_fft(scalar, kernel_hat, fft_object=fft_object, ifft_object=ifft_object)

# judge whether this test is successful
print(np.allclose(v, np.ones(nx) *
                  nx*(1+2*periodic[1])))


# %%
# Convolution of constant scalar with constant kernel in 2D
# kernel_flag = 0
nx, ny = [4, 4]
Lx, Ly = [10, 10]
dx, dy = [Lx/nx, Ly/ny]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
scalar = np.ones([ny, nx])
periodic = np.array([1, 0])
kernel = np.ones([ny*2, nx*2])*(1+2*periodic[0])*(1+2*periodic[1])
kernel_hat = kernel_mod.kernel_fft_evaluate(kernel = kernel)

v = conv.vel_convolution_fft(scalar, kernel_hat)
print(np.allclose(v, np.ones([ny, nx])*ny *
                  nx*(1+2*periodic[0])*(1+2*periodic[1])))


#%%
# Convolution of constant scalar with constant kernel in 3D
# kernel_flag = 0
nx, ny, nz = [4, 4, 4]
Lx, Ly, Lz= [10, 10, 10]
dx, dy, dz = [Lx/nx, Ly/ny, Lz/nz]
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
scalar = np.ones([ny, nx, nz])
periodic = np.array([1, 0, 2])
kernel = np.ones([ny*2, nx*2, nz*2])*(1+2*periodic[0])*(1+2*periodic[1])*(1+2*periodic[2])
kernel_hat = kernel_mod.kernel_fft_evaluate(kernel = kernel)
v = conv.vel_convolution_fft(scalar, kernel_hat)
print(np.allclose(v, np.ones([ny, nx, nz])*ny *
                  nx*nz*(1+2*periodic[0])*(1+2*periodic[1])*(1+2*periodic[2])))


#%%
