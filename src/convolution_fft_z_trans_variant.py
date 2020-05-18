# This function is to deal with situation specifically where z component of the kernel_handle is not transtationally invariant.
from __future__ import division, print_function
import pyfftw
import multiprocessing
import numpy as np
# from numba import jit
import sys
sys.path.append('~/Cecil/Projects/convolution_fft/src') # Donev: Use relative paths like ./ not absolute. Put absolute paths in main not here
# sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
# import copy


import convolution_fft as conv

# Donev: This should use numba
def kernel_evaluate_z_trans_variant(x, kernel_handle, periodic, L):
    """ Evaluate kernel on given grid
    1, 2, 3D are accepted.
    The kernel is translationally variant in z-direction
    """
    dim = len(x)
    assert dim == 3, "Dim should be 3!"
    Lx, Ly, Lz = L[:]
    x, y, z = x[:]
    nx = x.size
    ny = y.size
    nz = z.size
    dz = Lz/nz
    kernel = np.zeros((nz, 2*ny, 2*nx, nz)) 
    # Donev: This array should be allocated once and only once, not each time step because it is very large
    x_d = np.concatenate((x, x-Lx), axis=None)
    y_d = np.concatenate((y, y-Ly), axis=None)
    # z_d = np.concatenate((z, z-Lz), axis=None)
    periodic_flag = ~(periodic == 0)
    for k in range(nz):
        zf = z[k]-dz/2
        grid_x, grid_y, grid_z = np.meshgrid(x_d, y_d, zf-z)
        scalar_d = kernel_handle(grid_x, grid_y, grid_z)
        if periodic_flag[0]:
            for i in range(periodic[0]):
                if periodic_flag[1]:
                    for j in range(periodic[1]):
                        scalar_d = scalar_d + \
                            kernel_handle(grid_x+(i+1)*Lx,
                                          grid_y+(j+1)*Ly, grid_z)
                        scalar_d = scalar_d + \
                            kernel_handle(grid_x+(i+1)*Lx,
                                          grid_y-(j+1)*Ly, grid_z)
                        scalar_d = scalar_d + \
                            kernel_handle(grid_x-(i+1)*Lx,
                                          grid_y+(j+1)*Ly, grid_z)
                        scalar_d = scalar_d + \
                            kernel_handle(grid_x-(i+1)*Lx,
                                          grid_y-(j+1)*Ly, grid_z)
                else:
                    scalar_d = scalar_d + \
                        kernel_handle(grid_x+(i+1)*Lx, grid_y, grid_z)
                    scalar_d = scalar_d + \
                        kernel_handle(grid_x-(i+1)*Lx, grid_y, grid_z)
        else:
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    scalar_d = scalar_d + \
                        kernel_handle(grid_x, grid_y+(j+1)*Ly, grid_z)
                    scalar_d = scalar_d + \
                        kernel_handle(grid_x, grid_y-(j+1)*Ly, grid_z)
        kernel[k, :, :, :] = scalar_d
    return kernel


def kernel_hat_z_trans_variant(x, kernel_handle, periodic, L, fft_object):
    """ Fourier transform of kernel on given grid
    1, 2, 3D are accepted.
    This kernel is translationally variant in z.
    """
    dim = len(x)
    assert dim == 3, "Dim should be 3!"
    nx = x[0].size
    ny = x[1].size
    nz = x[2].size
    kernel = kernel_evaluate_z_trans_variant(x, kernel_handle, periodic, L)
    kernel_hat = np.zeros((nz, 2*ny, nx+1, nz), dtype=complex)
    a = fft_object.input_array
    for i in range(nz):
        a[:] = kernel[i, :, :, :]
        kernel_hat[i, :, :, :] = (fft_object())
    return kernel_hat

def vel_convolution_fft_z_trans_variant(scalar, method=1, *args, **kwargs):
    """ Compute convolution of given 'scalar' with kernel, 
    dimension of 1, 2, 3 is accepted. 
    """
    if ('kernel_hat' in kwargs):
        # User provides with Fourier transform of the kernel
        kernel_hat = kwargs['kernel_hat']
        assert isinstance(kernel_hat, np.ndarray)
    else:
        raise Exception("FT of the kernel should be given")    

    Nshape = scalar.shape
    dim = len(Nshape)
    assert dim == 3, "Dim should be 3!"

    [ny, nx, nz] = Nshape[:]
    if method == 1:
        scalar_d = np.zeros((2*ny, 2*nx, nz))
        scalar_d[:ny, :nx, :nz] = scalar
        # fftw objects, either a global object or created at every call
        if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
            fft_object = kwargs['fft_object']
            ifft_object = kwargs['ifft_object']
            a = fft_object.input_array
            b = fft_object.output_array
        else:
            a = pyfftw.empty_aligned((2*ny, 2*nx, nz), dtype='float64') # Double zero-pad in x and y to do method 1
            # Save efforts by knowing that a is real
            b = pyfftw.empty_aligned(
                (2*ny, nx+1, nz), dtype='complex128')
            # Real to complex FFT Over the both axes
            fft_object = pyfftw.FFTW(a, b, axes=(
                0, 1), flags=('FFTW_MEASURE', ))
            ifft_object = pyfftw.FFTW(b, a, axes=(
                0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
        
        # Fourier transform of the scalar
        a[:][:][:] = scalar_d
        scalar_d_hat = (fft_object()).copy()
        v = np.zeros((ny, nx, nz))
        for i in range(nz):
            b[:][:][:] = scalar_d_hat * kernel_hat[i, :, :, :]
            tmp = (ifft_object()).copy()
            tmp = tmp[:ny, :nx, :nz]
            v[:, :, i] = tmp.sum(axis=2)
        return v
    else:
        raise Exception("Method %i has not been implemented!" % method)
