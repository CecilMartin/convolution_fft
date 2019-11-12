# A FFT-based convolution method for mix-periodic Boundary Condition

## Author: [Zhe Chen](https://github.com/CecilMartin)

This library is aimed to developing a fast fft-based method to compute convolution with mix-boundary condition, i.e. some dimension is periodic while the other is not. Basically, it uses a trick to double zerod-padding the non-periodic dimension and then utilize FFT method to do the convolution. As for the periodic dimension, we could use periodic summation to mimic periodicity. Moreover, if analytic fourier transform or the kernel that being convolved is given or its asymtopic approximation function's analytic FT is given, the cost of periodic summation could be reduced or even eliminated.

You could find more detail in computation and numerics in this [report](https://github.com/CecilMartin/convolution_fft/blob/master/doc/FFT_Conv.pdf).

## Features

1. Convolution in 1D, 2D and 3D is supported. Any periodicity condition is supported.

2. Analytic fourier transform of the kernel or its asymptopic approximation is supported.

3. FFTW object could be planned only once during the whole project, which only requires one-time fft planning and this object is used globally.

4. Multiprocessing of FFTW is used.

5. Double and single precision are provided. (TODO)

## Parameters

vel_convolution_fft(scalar, L, x, kernel, periodic, method=1, fft_object=None, ifft_object=None, *args, **kwargs)

1. scalar: This is the field that is convolved. dtype = numpy.narray, dim=1,2,3. Shape = [Nx] or [Ny,Nx] or [Ny,Nx,Nz]. 

2. L: length of the computing boxes, L=[Lx] or [Lx,Ly] or [Lx,Ly,Lz].

3. x: grid of each dimension, numpy.narray, x=[grid_x] or [grid_x,grid_y] or [grid_x,grid_y,grid_z].

4. kernel: function handle of the kernel, kernel(x,y), multidimention of x and y should be supported.

5. Method: 1 or 2, representing method 1 or 2 in the report accordingly.

6. fft_object, ifft_object: pyfftw object, default is None. If they are not specified, the fftw object would be created every time this function is called. If they are given, the fftw objects would be used every time this function is called, which would save lots of time for planning fftw objects.