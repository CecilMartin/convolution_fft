# A FFT-based convolution method for mix-periodic Boundary Condition

## Author: [Zhe Chen](https://github.com/CecilMartin)

This library is aimed to developing a fast fft-based method to compute convolution with mix-boundary condition, i.e. some dimension is periodic while the other is not. Basically, it uses a trick to double zerod-padding the non-periodic dimension and then utilize FFT method to do the convolution. As for the periodic dimension, we could use periodic summation to mimic periodicity. Moreover, if analytic fourier transform or the kernel that being convolved is given or its asymtopic approximation function's analytic FT is given, the cost of periodic summation could be reduced or even eliminated.

However, for situation where 3D kernel is not translational invariant in z direction, naive FFT in 3D will not work since the kernel does not only depend on z-z0. Therefore, we provide another module convolution_fft_z_trans_variant that deal with such situation. Consider a system of Nx*Ny*Nz, interaction between any two layers in z direction is computed by 2D fft method, and then adds up all Nz layers' contribution to the target layer. Thus, kernel should be evaulated as Nz*Ny*Nx*Nz. Moreover, the 3 kernel options and fftw objects options are provides too. 

You could find more detail in computation and numerics in this [report](https://github.com/CecilMartin/convolution_fft/blob/master/doc/FFT_Conv.pdf).

## Features

1. Convolution in 1D, 2D and 3D is supported. Any periodicity condition is supported.

2. Analytic fourier transform of the kernel or its asymptopic approximation is supported.

3. FFTW object could be planned only once during the whole project, which only requires one-time fft planning and this object is used globally.

4. Multiprocessing of FFTW is used.

5. Double and single precision are provided. (TODO)

6. Kernel could be given as a function handle that been evaluated every time that this function was called, or as a numpy.ndarray so that evaluation of kernel is only computed one time, or even as its fft so that effort to compute its fft is saved.

7. We also provide features to deal with situation where 3D kernel is not translational invariant in z direction

## Parameters

vel_convolution_fft(scalar, method=1, *args, **kwargs)

1. scalar: This is the field that is convolved. dtype = numpy.narray, dim=1,2,3. Shape = [Nx] or [Ny,Nx] or [Ny,Nx,Nz]. 

2. Method = 1(default) or 2, representing method 1 or 2 in the report accordingly.

Moreover, there's keyword arguments that is used for features about kernel and fftw objects

For kernel, there's three options,

3. Given numpy.ndarray of the kernel, which would avoid evaluating the kernel every time the function is called.\
3.1. kernel = kernel: numpy.ndarray. It should be evaluated with double size of scalar, and periodic summation should have done.

4. Given function handle of the kernel, the kernel would be evaluated inside the routine.\
4.1. L=L: length of the computing boxes, L=[Lx] or [Lx,Ly] or [Lx,Ly,Lz].\
4.2. x=x: grid of each dimension, numpy.narray, x=[grid_x] or [grid_x,grid_y] or [grid_x,grid_y,grid_z].\
4.3. kernel_handle=kernel_handle: function handle of the kernel, kernel_handle(x(,y(,z))), multidimention of x, y and z should be supported.\
4.4. periodic=periodic: numpy.ndarray, size is dimension. Peridic BC of each dimension. 0 represents a-periodic, n>0 means it is periodic and truncated at n images on both sides of the periodic summation, i.e. [-n,...,0,...,n].  

5. Given fft of the periodic summation kernel.\
5.1. kernel_hat=kernel_hat: numpy.ndarray. It should be double the size of scalar and is fft of the periodic summation of kernel, which would save the effort to compute the fft of kernel every time the function is called.

For fftw objects, we could pass global objects to the routine so that it won't plan a object this function is called, which would cost a lot.

6. fft_object, ifft_object:  If they are not specified, the fftw object would be created every time this function is called. If they are given, the fftw objects would be used every time this function is called, which would save lots of time for planning fftw objects. Wisdom mechanism of FFTW is utilized here.

