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
import convolution_fft_z_trans_variant as conv

# %%
# Test of convolution of constant scalar wiht constant kernel in 3D
# kernel_flag = 1
nx, ny, nz = [4, 4, 2]
Lx, Ly, Lz = [10, 10, 5]
dx, dy, dz = [Lx/nx, Ly/ny, Lz/nz]
x = (np.linspace(0, nx-1, nx)-0.5)*dx
y = (np.linspace(0, ny-1, ny)-0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
periodic = np.array([1, 0])
method = 1
if method == 1:
    a = pyfftw.empty_aligned((2*ny, 2*nx, nz), dtype='float64')
    # Save efforts by knowing that a is real
    b = pyfftw.empty_aligned((2*ny, nx+1, nz), dtype='complex128')
    # Real to complex FFT Over the both axes
    fft_object = pyfftw.FFTW(a, b, axes=(
        0, 1), flags=('FFTW_MEASURE', ))
    ifft_object = pyfftw.FFTW(b, a, axes=(
        0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
else:
    raise Exception('Only Method = 1 is supported now.')

scalar = np.ones((ny, nx, nz))

v = conv.vel_convolution_fft_z_trans_variant(scalar, L=[Lx,Ly,Lz], x=[x,y,z], kernel_handle=kernel_mod.kernel_uniform_3D,
                             periodic=periodic, fft_object=fft_object, ifft_object=ifft_object)

# judge whether this test is successful
print(np.allclose(v,np.ones([ny,nx,nz])*ny*nx*nz*(1+2*periodic[0])*(1+2*periodic[1])))


# %%
# Test of convolution of constant scalar wiht constant kernel in 3D
# kernel_flag = 0 
nx, ny, nz = [4, 4, 2]
Lx, Ly, Lz = [10, 10, 5]
dx, dy, dz = [Lx/nx, Ly/ny, Lz/nz]
x = (np.linspace(0, nx-1, nx)-0.5)*dx
y = (np.linspace(0, ny-1, ny)-0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
z = (np.linspace(0, nz-1, nz)+0.5)*dz
scalar = np.ones((ny, nx, nz))
periodic = np.array([1, 0])
method = 1
kernel = np.ones((nz, ny*2, nx*2, nz))*(1+2*periodic[0])*(1+2*periodic[1])
v = conv.vel_convolution_fft_z_trans_variant(scalar, kernel=kernel)
print(np.allclose(v,np.ones([ny,nx,nz])*ny*nx*nz*(1+2*periodic[0])*(1+2*periodic[1])))


#%%
# Test of convolution of constant scalar wiht constant kernel in 3D
# kernel_flag =2
nx, ny, nz = [4, 4, 2]
Lx, Ly, Lz = [10, 10, 5]
dx, dy, dz = [Lx/nx, Ly/ny, Lz/nz]
x = (np.linspace(0, nx-1, nx)-0.5)*dx
y = (np.linspace(0, ny-1, ny)-0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
z = (np.linspace(0, nz-1, nz)+0.5)*dz
scalar = np.ones((ny, nx, nz))
periodic = np.array([1, 0])
method = 1
a = pyfftw.empty_aligned((2*ny, 2*nx, nz), dtype='float64')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((2*ny, nx+1, nz), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))
ifft_object = pyfftw.FFTW(b, a, axes=(
    0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))

kernel_hat = conv.kernel_hat_z_trans_variant([x,y,z], kernel_mod.kernel_uniform_3D, periodic, [Lx,Ly,Lz], fft_object)
v = conv.vel_convolution_fft_z_trans_variant(scalar, kernel_hat=kernel_hat, fft_object = fft_object, ifft_object = ifft_object)
print(np.allclose(v,np.ones([ny,nx,nz])*ny*nx*nz*(1+2*periodic[0])*(1+2*periodic[1])))

#%%
