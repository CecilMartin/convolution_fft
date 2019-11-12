# Test convolution for 2D

# %%
from __future__ import division, print_function
import sys
sys.path.append('../../src')
import convolution_fft as conv
import kernel.kernel
import numpy as np
import copy
import multiprocessing
import pyfftw


nx, ny = [4, 4]
x = np.linspace(0, nx-1, nx)
y = np.linspace(0, ny-1, ny)
periodic = 1
Lx, Ly = [10, 10]
dx, dy = [Lx/nx, Ly/ny]
a = pyfftw.empty_aligned((ny, nx), dtype='float64')
# Save efforts by knowing that a is real
b = pyfftw.empty_aligned((ny, nx//2+1), dtype='complex128')
# Real to complex FFT Over the both axes
fft_object = pyfftw.FFTW(a, b, axes=(
    0, 1), flags=('FFTW_MEASURE', ))
ifft_object = pyfftw.FFTW(b, a, axes=(
    0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
v = conv.vel_convolution_fft(scalar, dx, dy, x, y, kernel.kernel_uniform_2D,
                             periodic, method=1, fft_object=fft_object, ifft_object=ifft_object)


# %%
