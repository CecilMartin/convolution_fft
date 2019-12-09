from __future__ import division, print_function
import numpy as np
import finufftpy 
# import copy
import multiprocessing
import pyfftw
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
# pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
# precision = 'single'
precision = 'double'
# Only double precision is implemented for now

def real(x):
    if precision == 'single':
        return np.float32(x)
    else:
        return np.float64(x)


def vel_convolution_fft(scalar, method=1, *args, **kwargs):
    if 'kernel' in kwargs:
        kernel = kwargs['kernel']
        kernel_flag = 0  # doubled kernel is provided
        assert isinstance(kernel, np.ndarray)
    elif ('kernel_handle' in kwargs) & ('L' in kwargs) & ('x' in kwargs) & ('periodic' in kwargs):
        kernel_handle = kwargs['kernel_handle']
        L = kwargs['L']
        x = kwargs['x']
        periodic = kwargs['periodic']
        kernel_flag = 1  # given kernel handle
        assert callable(kernel_handle)
    elif ('kernel_hat' in kwargs):
        kernel_hat = kwargs['kernel_hat']
        kernel_flag = 2  # given fft of kernel
        assert isinstance(kernel_hat, np.ndarray)

    Nshape = scalar.shape
    dim = len(Nshape)
    if dim == 1:
        nx = Nshape[0]
        if method == 1:
            scalar_d = np.zeros(2*nx)
            scalar_d[:nx] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
                a = fft_object.input_array
                b = fft_object.output_array
            else:
                a = pyfftw.empty_aligned((2*nx), dtype='float64')
                # Save efforts by knowing that a is real
                b = pyfftw.empty_aligned((nx+1), dtype='complex128')
                # Real to complex FFT Over the both axes
                fft_object = pyfftw.FFTW(a, b, axes=(
                    -1,), flags=('FFTW_MEASURE', ))
                ifft_object = pyfftw.FFTW(b, a, axes=(
                    -1,), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
            if kernel_flag == 0:
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 2:
                pass

            a[:] = scalar_d
            scalar_d_hat = (fft_object()).copy()

            b[:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:nx]
    elif dim == 2:
        [ny, nx] = Nshape[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx])
            scalar_d[:ny, :nx] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
                a = fft_object.input_array
                b = fft_object.output_array
            else:
                a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
                # Save efforts by knowing that a is real
                b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
                # Real to complex FFT Over the both axes
                fft_object = pyfftw.FFTW(a, b, axes=(
                    0, 1), flags=('FFTW_MEASURE', ))
                ifft_object = pyfftw.FFTW(b, a, axes=(
                    0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
            if kernel_flag == 0:
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 2:
                pass

            a[:][:] = scalar_d
            scalar_d_hat = (fft_object()).copy()

            b[:][:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx]

    elif dim == 3:
        [ny, nx, nz] = Nshape[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx, 2*nz])
            scalar_d[:ny, :nx, :nz] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
                a = fft_object.input_array
                b = fft_object.output_array
            else:
                a = pyfftw.empty_aligned((2*ny, 2*nx, 2*nz), dtype='float64')
                # Save efforts by knowing that a is real
                b = pyfftw.empty_aligned(
                    (2*ny, 2*nx, nz+1), dtype='complex128')
                # Real to complex FFT Over the both axes
                fft_object = pyfftw.FFTW(a, b, axes=(
                    0, 1, 2), flags=('FFTW_MEASURE', ))
                ifft_object = pyfftw.FFTW(b, a, axes=(
                    0, 1, 2), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
            if kernel_flag == 0:
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 2:
                pass

            a[:][:][:] = scalar_d
            scalar_d_hat = (fft_object()).copy()

            b[:][:][:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx, :nz]
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return 0


def vel_convolution_nufft(scalar, source_strenth, num_modes, L,  eps=1e-8, method=1, *args, **kwargs):
    if('kernel_hat' in kwargs):
        kernel_hat = kwargs['kernel_hat']
        assert isinstance(kernel_hat, np.ndarray)
    else:
        raise Exception(
            'kernel_hat, fourier transform of kernel should be given')

    Nshape = scalar.shape
    dim = Nshape[0]
    assert dim == 2, "Dim should be 2!"
    nx, ny = num_modes[:]
    if method == 1:
        xj, yj = scalar
        nj = len(xj)
        xmin = xj.min()
        xmax = xj.max()
        ymin = yj.min()
        ymax = yj.max()
        Lx, Ly = L
        assert ((xmax-xmin) <= Lx) & ((ymax-ymin) <= Ly), "All particles should be limieted to given [Lx,Ly] box!"
        # double zero-padding the delta function and scale it into [-pi,pi)
        xj = (xj-(xmax+xmin)/2)/Lx*np.pi
        yj = (yj-(ymax+ymin)/2)/Ly*np.pi
        fk = np.zeros(num_modes, dtype=np.complex128, order='F')
        # particle locations change at every iteration, no need to reuse fftw object
        ret = finufftpy.nufft2d1(xj, yj, source_strenth, -1, eps, nx, ny, fk, modeord=1)
        assert ret == 0, "NUFFT not successful!"
        v_hat = np.zeros((nx, ny, 2), order='F', dtype=np.complex128)
        v_hat[:, :, 0] = fk * kernel_hat[0]
        v_hat[:, :, 1] = fk * kernel_hat[1]
        v = np.zeros((nj, 2), order='F', dtype=np.complex128)
        ret = finufftpy.nufft2d2many(xj, yj, v, 1, eps, v_hat, modeord=1)
        assert ret == 0, "NUFFT not successful!"
        v = np.real(v)/(2*Lx*2*Ly)  # Real?
        return v

def vel_direct_convolution(scalar, source_strenth, kernel_handle, L, periodic):
    Np = len(source_strenth)
    dim = len(L)
    v = np.zeros(Np)
    if dim == 2:
        x, y = scalar[:]
        for i in range(Np):
            for j in range(Np):
                kernel_index_x, kernel_index_y = np.meshgrid(np.linspace(-periodic[0],periodic[0],2*periodic[0]+1),np.linspace(-periodic[1],periodic[1],2*periodic[1]+1), indexing = 'ij')
                xd = x[i]-x[j]-L[0]*kernel_index_x
                yd = y[i]-y[j]-L[1]*kernel_index_y
                kernel = kernel_handle(xd,yd).sum()
                v[i] += kernel * source_strenth[j]
    else:
        raise Exception("Not implemented yet!")
    return v
        
    
    
def kernel_evaluate(x, kernel, periodic, L):
    dim = len(L)
    if dim == 1:
        Lx = L[0]
        x = x[0]
        x_d = np.concatenate((x, x-Lx), axis=None)
        periodic_flag = ~(periodic == 0)
        scalar_d = kernel(x_d)
        if periodic_flag[0]:
            for i in range(periodic[0]):
                scalar_d = scalar_d + kernel(x_d+(i+1)*Lx)
                scalar_d = scalar_d + kernel(x_d-(i+1)*Lx)
        return scalar_d
    if dim == 2:
        Lx, Ly = L[:]
        x, y = x[:]
        x_d = np.concatenate((x, x-Lx), axis=None)
        y_d = np.concatenate((y, y-Ly), axis=None)
        grid_x, grid_y = np.meshgrid(x_d, y_d)
        periodic_flag = ~(periodic == 0)
        scalar_d = kernel(grid_x, grid_y)
        if periodic_flag[0]:
            for i in range(periodic[0]):
                if periodic_flag[1]:
                    for j in range(periodic[1]):
                        scalar_d = scalar_d + \
                            kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly)
                        scalar_d = scalar_d + \
                            kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly)
                        scalar_d = scalar_d + \
                            kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly)
                        scalar_d = scalar_d + \
                            kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly)
                else:
                    scalar_d = scalar_d + kernel(grid_x+(i+1)*Lx, grid_y)
                    scalar_d = scalar_d + kernel(grid_x-(i+1)*Lx, grid_y)
        else:
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    scalar_d = scalar_d + kernel(grid_x, grid_y+(j+1)*Ly)
                    scalar_d = scalar_d + kernel(grid_x, grid_y-(j+1)*Ly)
        return scalar_d
    elif dim == 3:
        Lx, Ly, Lz = L[:]
        x, y, z = x[:]
        x_d = np.concatenate((x, x-Lx), axis=None)
        y_d = np.concatenate((y, y-Ly), axis=None)
        z_d = np.concatenate((z, z-Lz), axis=None)
        grid_x, grid_y, grid_z = np.meshgrid(x_d, y_d, z_d)
        periodic_flag = ~(periodic == 0)
        scalar_d = kernel(grid_x, grid_y, grid_z)
        if periodic_flag[0]:
            for i in range(periodic[0]):
                if periodic_flag[1]:
                    for j in range(periodic[1]):
                        if periodic_flag[2]:
                            for k in range(periodic[2]):
                                scalar_d += kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                    grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                                scalar_d += kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                    grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                                scalar_d += kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                    grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                                scalar_d += kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                    grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                        else:
                            scalar_d += kernel(grid_x+(i+1)
                                               * Lx, grid_y+(j+1)*Ly, grid_z)
                            scalar_d += kernel(grid_x+(i+1)
                                               * Lx, grid_y-(j+1)*Ly, grid_z)
                            scalar_d += kernel(grid_x-(i+1)
                                               * Lx, grid_y+(j+1)*Ly, grid_z)
                            scalar_d += kernel(grid_x-(i+1)
                                               * Lx, grid_y-(j+1)*Ly, grid_z)
                else:
                    if periodic_flag[2]:
                        for k in range(periodic[2]):
                            scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                                grid_x+(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                                grid_x-(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                    else:
                        scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z)
                        scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z)
        else:
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    if periodic_flag[2]:
                        for k in range(periodic[2]):
                            scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                    else:
                        scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z) + \
                            kernel(grid_x, grid_y-(j+1)*Ly, grid_z)
            else:
                if periodic_flag[2]:
                    for k in range(periodic[2]):
                        scalar_d += kernel(grid_x, grid_y, grid_z+(k+1)*Lz) + \
                            kernel(grid_x, grid_y, grid_z-(k+1)*Lz)
        return scalar_d
    else:
        raise Exception('%iD is not implemented yet!' % dim)
