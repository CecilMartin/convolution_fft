import numpy as np
from numba import jit, prange
import multiprocessing
import pyfftw
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
# pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
from numba.pycc import CC
cc = CC('kernel_module')


@jit(nopython = True)
# @cc.export('kernel_uniform_2D','f8[:,:](f8[:,:],f8[:,:])')
def kernel_uniform_2D(x,y):
    return np.ones(x.shape)

@jit(nopython = True)
def kernel_uniform_1D(x):
    return np.ones(x.shape)


@jit(nopython = True)
def kernel_uniform_3D(x,y,z):
    return np.ones(x.shape)

@jit(nopython = True)
# @jit(nopython = True, parallel = True)
def kernel_Gauss_2D(x,y,sigma):
    k = 1/(2*np.pi*sigma**2)*np.exp(-(np.power(x,2)+np.power(y,2))/sigma**2)
    return k

# Donev: I would prefer if you use the kernels in mobility_numba.py and just modify that code
# In that code the loop is over pairs of particles. But, you can rewrite this so that:
# 1. The actual kernel code just evaluates the kernel for two points (no loops, simple function that could also be written in C)
#    You don't have to rewrite the kernels, just extract them from inside mobility numba loops and move them to a routine
#    Later we can try to move this to C or Fortran
# 2. The loops over pairs or particles, or over grid points, are in numba, parallel, and they call the kernel from point #1
# 3. One can also use CUDA via numba for example instead of openMP parallel loops. The kernel functon from #1 should not need to change
@jit(nopython = True)
def mobilityUFRPY_x_dir_1(rx, ry): # Donev: Where is this function actually used? Can't find it in the code.
    # rx, ry = rx/a, ry/a.
    # r>=2
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    invr = 1 / r
    invr2 = invr * invr
    c1 = 1+2/(3*r2)
    c2 = (1 - 2 * invr2) * invr2
    Mxx = (c1 + c2*rx*rx) * invr
    return Mxx

@jit(nopython = True)
def mobilityUFRPY_x_dir_2(rx, ry):
    # rx, ry = rx/a, ry/a.
    # r<2
    fourOverThree = 4/3
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    invr = 1 / r
    c1 = fourOverThree * (1 - 0.28125 * r) #9/32 = 0.28125
    c2 = fourOverThree * 0.09375 * invr #3/32 = 0.09375
    Mxx = c1 + c2 * rx*rx
    return Mxx



     
# Donev: Numba will accelerate this greatly. Even though we only need to do this once, it is nice to make it more efficient and it should be easy
# Remember that we need to sum over many periodic images sometimes so this should be somewhat efficient not plain python loops    
# @jit(nopython = True) # A function wrapper is needed here since some functions in numpy are not well supported
# Donev: I think this can be handled best by always making the arrays be 3d
def kernel_evaluate(grids, kernel, periodic, L, indexing = 'xy'):
    """ Evaluate kernel on given grid
    1, 2, 3D are accepted.
    """
    dim = len(L)
    if dim == 1:
        Lx = L[0]
        x = grids[0]
        return __kernel_evaluate_1D(Lx, np.concatenate((x, x-Lx), axis=None), kernel, periodic)
    if dim == 2:
        Lx, Ly = L[:]
        x, y = grids[:]
        x_d = np.concatenate((x, x-Lx), axis=None)  # Numba has problem with (x,x-Lx)
        y_d = np.concatenate((y, y-Ly), axis=None)
        grid_x, grid_y = np.meshgrid(x_d, y_d, indexing = indexing)
        return __kernel_evaluate_2D(Lx, Ly, grid_x, grid_y, kernel, periodic)
    elif dim == 3:
        Lx, Ly, Lz = L[:]
        x, y, z = grids[:]
        x_d = np.concatenate((x, x-Lx), axis=None) # Numba
        y_d = np.concatenate((y, y-Ly), axis=None)
        z_d = np.concatenate((z, z-Lz), axis=None)
        grid_x, grid_y, grid_z = np.meshgrid(x_d, y_d, z_d, indexing = indexing)
        return __kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic)
    else:
        raise Exception('Only 1,2,3 dimension(s) are supported!')


@jit(nopython = True)
def __kernel_evaluate_1D(Lx, x_d, kernel, periodic):
    periodic_flag = ~(periodic == 0)
    scalar_d = kernel(x_d)
    if periodic_flag[0]:
        for i in range(periodic[0]):
            scalar_d = scalar_d + kernel(x_d+(i+1)*Lx) # Donev: Will numba inline kernel ? # Zhe: The test turns out it will.
            scalar_d = scalar_d + kernel(x_d-(i+1)*Lx)
    return scalar_d


@jit(nopython = True)
def __kernel_evaluate_2D(Lx, Ly, grid_x, grid_y, kernel, periodic):
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



@jit(nopython = True)
def __kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic):
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

def kernel_fft_evaluate(method = 1, indexing = 'xy', *arg, **kwargs):
    """ Compute Fourier transform of kenrel. user can provide with kernel evaluated at grids,
    or kernel function handle and grid. It's recommended that user provides with FFTW objects
    so that it could be used among iterations.
    """
    if 'kernel' in kwargs:
        # User provides with evaluated kernel 
        kernel = kwargs['kernel']
        kernel_flag = 0  # doubled kernel is provided
        assert isinstance(kernel, np.ndarray)
        Nshape = [i//2 for i in kernel.shape]
    elif ('kernel_handle' in kwargs) & ('L' in kwargs) & ('x' in kwargs) & ('periodic' in kwargs):
        # User provides with hernel function handle.
        kernel_handle = kwargs['kernel_handle']
        L = kwargs['L']
        x = kwargs['x']
        periodic = kwargs['periodic']
        kernel_flag = 1  # given kernel handle
        assert callable(kernel_handle)
        Nshape = [len(i) for i in x]
    else:
        raise Exception('Please provide kernel or kernel function')
    dim = len(Nshape)
    if dim == 1:
        # One dimension 
        nx = Nshape[0]
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                # fftw objects is passed as input, it's used globally
                fft_object = kwargs['fft_object']
            else:
                # fftw object is created every time this function is called, 
                # suitable for situation where Foutier modes change at every time step
                fft_object = create_fftw_plan([nx],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                # kernel needs to  be evaluated
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
    elif dim == 2:
        # two dimension
        [ny, nx] = Nshape[:] # TODO: indexing of arrays
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                fft_object = kwargs['fft_object']
            else:
                fft_object = create_fftw_plan([nx,ny],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
    elif dim == 3:
        # three dimension
        [ny, nx, nz] = Nshape[:]
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                fft_object = kwargs['fft_object']
            else:
                fft_object = create_fftw_plan([nx,ny,nz],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return kernel_hat


def create_fftw_plan(n, ifft_flag = True):
    """Create Pyfftw plan
    n, array of [nx[,ny[,nz]]], 1,2,3D are accepted
    ifft_flag = True, whether to create ifft object at the same time. Notice that, to save memory, this use
    input array a of fft object as output array, it uses output array b of fft objects as input array.
    Users should be cautious about that this sharing array issue.
    """
    dim = len(n)
    if dim == 1:
        nx = n[0]
        a = pyfftw.empty_aligned((2*nx), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned((nx+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            -1,), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                -1,), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    elif dim == 2:
        nx, ny = n[:]
        a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            0, 1), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    elif dim == 3:
        nx, ny, nz = n[:]
        a = pyfftw.empty_aligned((2*ny, 2*nx, 2*nz), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned(
            (2*ny, 2*nx, nz+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            0, 1, 2), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                0, 1, 2), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    else:
        raise Exception('It work only in 1,2,3D!')
    if ifft_flag:
        return fft_object, ifft_object
    else:
        return fft_object
        
if __name__ == "__main__":
    cc.compile()
        

    