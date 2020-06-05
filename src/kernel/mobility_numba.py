from __future__ import division, print_function
import numpy as np
import scipy.sparse

# Try to import numba
try:
  from numba import njit(fastmath = True), prange
except ImportError:
  print('numba not found')

# Donev: If numba has an "inline" tag add this tag to this function to encourage inlining it inside loops or other functions
@njit(fastmath = True)
def no_wall_mobility_trans_trans_numba(r_vectors, eta, a):
    '''
    Returns the mobility at the blob level to the force
    on the blobs. Mobility for particles in an unbounded domain, it uses
    the standard RPY tensor.

    This function uses numba.
    '''
    M = np.zeros((3,3))
    fourOverThree = 4.0 / 3.0 # Donev: I think the code will be clearer and equally fast/slow if you put 4.0/3.0 in the code
    # A good compiler with optimization on will evaluate all floating-point constants at compile time anyway
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)


    # Compute vector between particles i and j
    # Normalize distance with hydrodynamic radius
    rx = r_vectors[0] * inva
    ry = r_vectors[1] * inva
    rz = r_vectors[2] * inva

    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)

    if r > 2:
        invr = 1.0 / r
        invr2 = invr * invr
        c1 = 1.0 + 2.0 / (3.0 * r2)
        c2 = (1.0 - 2.0 * invr2) * invr2
        M[0,0] = (c1 + c2*rx*rx) * invr
        M[0,1] = (c2*rx*ry) * invr
        M[0,2] = (c2*rx*rz) * invr
        M[1,1] = (c1 + c2*ry*ry) * invr
        M[1,2] = (c2*ry*rz) * invr
        M[2,2] = (c1 + c2*rz*rz) * invr
    else:
        # Deal with r->0+
        eps = 2.220446049250313e-16 # np.finfo(float).eps # Donev: Why are you not using np.finfo(float).eps? In particular is this always double precision?
        # Donev: In general hard-coded values like 2.22e-16 are not a good idea.
        r_hat = np.sqrt((rx+eps)**2+(ry+eps)**2+(rz+eps)**2)
        rx_hat = rx/r_hat
        ry_hat = ry/r_hat
        rz_hat = rz/r_hat
        c1 = fourOverThree * (1.0 - 0.28125 * r)  # 9/32 = 0.28125 Exactly Donev: I think just using 9.0/32.0 is better to make it clear. This makes no real difference for speed
        # 3/32 = 0.09375 # Donev: I would rather use 3.0/32.0 for clarity
        c2 = fourOverThree * 0.09375 * r
        M[0,0] = c1 + c2 * rx_hat * rx_hat
        M[0,1] =      c2 * rx_hat * ry_hat
        M[0,2] =      c2 * rx_hat * rz_hat
        M[1,1] = c1 + c2 * ry_hat * ry_hat
        M[1,2] =      c2 * ry_hat * rz_hat
        M[2,2] = c1 + c2 * rz_hat * rz_hat
    
    M[1,0] = M[0,1]
    M[2,0] = M[0,2]
    M[2,1] = M[1,2]
    return M*norm_fact_f

# Donev: Also add inline flag here and all other kernel functions
@njit(fastmath = True)
def single_wall_mobility_trans_trans_numba(r_vectors, h, eta, a):
    ''' 
    Returns the translational mobility at the blob level to the force 
    on the blobs. Mobility for particles above a single wall.  

    This function uses numba.
    '''
    
    # Donev added documentation/comment:
    # Compute RPY without a wall first and then add wall corrections to it
    # The numba compiler will hopefully inline this anyway:
    M = no_wall_mobility_trans_trans_numba(r_vectors, eta, a)
    
    # Compute wall-corrections using Swan-Brady:
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a)

    rx = r_vectors[0] * inva
    ry = r_vectors[1] * inva
    rz = (r_vectors[2] + 2*h) * inva  # Reciprocal in z

    hj = h * inva
    h_hat = hj / rz
    invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
    ex = rx * invR
    ey = ry * invR
    ez = rz * invR
    invR3 = invR * invR * invR
    invR5 = invR3 * invR * invR

    fact1 = -(3.0*(1.0+2.0*h_hat*(1.0-h_hat)*ez*ez) * invR + 2.0 *
              (1.0-3.0*ez*ez) * invR3 - 2.0*(1.0-5.0*ez*ez) * invR5) / 3.0
    fact2 = -(3.0*(1.0-6.0*h_hat*(1.0-h_hat)*ez*ez) * invR - 6.0 *
              (1.0-5.0*ez*ez) * invR3 + 10.0*(1.0-7.0*ez*ez) * invR5) / 3.0
    fact3 = ez * (3.0*h_hat*(1.0-6.0*(1.0-h_hat)*ez*ez) * invR - 6.0 *
                  (1.0-5.0*ez*ez) * invR3 + 10.0*(2.0-7.0*ez*ez) * invR5) * 2.0 / 3.0
    fact4 = ez * (3.0*h_hat*invR - 10.0*invR5) * 2.0 / 3.0
    fact5 = -(3.0*h_hat*h_hat*ez*ez*invR + 3.0*ez*ez *
              invR3 + (2.0-15.0*ez*ez)*invR5) * 4.0 / 3.0

    M[0, 0] += (fact1 + fact2 * ex*ex)*norm_fact_f
    M[0, 1] += (fact2 * ex*ey)*norm_fact_f
    M[0, 2] += (fact2 * ex*ez + fact3 * ex)*norm_fact_f
    M[1, 0] += (fact2 * ey*ex)*norm_fact_f
    M[1, 1] += (fact1 + fact2 * ey*ey)*norm_fact_f
    M[1, 2] += (fact2 * ey*ez + fact3 * ey)*norm_fact_f
    M[2, 0] += (fact2 * ez*ex + fact4 * ex)*norm_fact_f
    M[2, 1] += (fact2 * ez*ey + fact4 * ey)*norm_fact_f
    M[2, 2] += (fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5)*norm_fact_f
    return M



@njit(fastmath = True)
def no_wall_mobility_trans_rot_numba(r_vectors, eta, a):
    ''' 
    Returns the mobility translation-rotation at the blob level to the torque 
    on the blobs. Mobility for particles in an unbounded domain, it uses
    the standard RPY tensor.  

    This function uses numba.
    '''
    M = np.zeros((3,3))
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)

    # Normalize distance with hydrodynamic radius
    rx = r_vectors[0] * inva
    ry = r_vectors[1] * inva
    rz = r_vectors[2] * inva
    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)
    r3 = r2*r
    if r >= 2:
        invr3 = 1.0 / r3
        # M[0,0] = 0
        M[0,1] = rz * invr3
        M[0,2] = -ry * invr3
        # M[1,1] = 0
        M[1,2] = rx * invr3
        # M[2,2] = 0
    else:
        c1 = 0.5 * (1.0 - 0.375 * r)  # 3/8 = 0.375
        # Mxx = 0
        M[0,1] = c1 * rz
        M[0,2] = -c1 * ry
        # Myy = 0
        M[1,2] = c1 * rx
        # M[2,2] = 0

    M[1,0] = -M[0,1]
    M[2,0] = -M[0,2]
    M[2,1] = -M[1,2]

    return M*norm_fact_f


@njit(fastmath = True)
def single_wall_mobility_trans_rot_numba(r_vectors, h, eta, a):
    ''' 
    Returns the product of the mobility translation-rotation at the blob level to the torque 
    on the blobs. Mobility for particles on top of an infinite wall.  

    This function uses numba.
    '''
    
    # Compute mobility without wall:
    M = no_wall_mobility_trans_rot_numba(r_vectors, eta, a)
    
    # Now add Swan-Brady wall correction:
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**2)


    # Normalize distance with hydrodynamic radius
    rx = -r_vectors[0] * inva
    ry = -r_vectors[1] * inva
    rz = (r_vectors[2]+2*h) * inva
    hj = (r_vectors[2]+h) * inva 

    h_hat = hj / rz
    invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
    invR2 = invR * invR
    invR4 = invR2 * invR2
    ex = rx * invR
    ey = ry * invR
    ez = rz * invR

    fact1 = invR2
    fact2 = (6.0 * h_hat * ez*ez * invR2 +
    (1.0-10.0 * ez*ez) * invR4) * 2.0
    fact3 = -ez * (3.0 * h_hat * invR2 - 5.0 * invR4) * 2.0
    fact4 = -ez * (h_hat * invR2 - invR4) * 2.0

    M[0,0] -= -fact3*ex*ey*norm_fact_f
    M[0,1] -= (-fact1*ez + fact3*ex*ex - fact4)*norm_fact_f
    M[0,2] -= fact1*ey*norm_fact_f
    M[1,0] -= (fact1*ez - fact3*ey*ey + fact4)*norm_fact_f
    M[1,1] -= fact3*ex*ey*norm_fact_f
    M[1,2] -= -fact1*ex*norm_fact_f
    M[2,0] -= (-fact1*ey - fact2*ey - fact3*ey*ez)*norm_fact_f
    M[2,1] -= (fact1*ex + fact2*ex + fact3*ex*ez)*norm_fact_f
    return M



@njit(fastmath = True)
def no_wall_mobility_rot_rot_numba(r_vectors, eta, a):
    ''' 
    Returns the mobility rotation-rotation at the blob level to the torque 
    on the blobs. Mobility for particles in an unbounded domain, it uses
    the standard RPY tensor.  

    This function uses numba.
    '''
    # Variables
    M = np.zeros((3,3))
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**3)

    # Normalize distance with hydrodynamic radius
    rx = r_vectors[0] * inva
    ry = r_vectors[1] * inva
    rz = r_vectors[2] * inva

    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)
    r3 = r2*r


    if r >= 2:
        invr2 = 1.0 / r2
        invr3 = 1.0 / r3
        c1 = -0.5
        c2 = 1.5 * invr2
        M[0,0] = (c1 + c2*rx*rx) * invr3
        M[0,1] = (c2*rx*ry) * invr3
        M[0,2] = (c2*rx*rz) * invr3
        M[1,1] = (c1 + c2*ry*ry) * invr3
        M[1,2] = (c2*ry*rz) * invr3
        M[2,2] = (c1 + c2*rz*rz) * invr3
    else:
        # 27/32 = 0.84375, 5/64 = 0.078125
        c1 = (1.0 - 0.84375 * r + 0.078125 * r3)
        c2 = 0.28125 * r - 0.046875 * r3       # 9/32 = 0.28125, 3/64 = 0.046875 # Donev: I think you don't need this
        eps = 2.220446049250313e-16 # np.finfo(float).eps # Donev: I don't like hard-wired constants
        r_hat = np.sqrt((rx+eps)**2+(ry+eps)**2+(rz+eps)**2)
        rx_hat = rx/r_hat
        ry_hat = ry/r_hat
        rz_hat = rz/r_hat
        M[0,0] = c1 + c2 * rx_hat * rx_hat
        M[0,1] =      c2 * rx_hat * ry_hat
        M[0,2] =      c2 * rx_hat * rz_hat
        M[1,1] = c1 + c2 * ry_hat * ry_hat
        M[1,2] =      c2 * ry_hat * rz_hat
        M[2,2] = c1 + c2 * rz_hat * rz_hat

    M[1,0] = M[0,1]
    M[2,0] = M[0,2]
    M[2,1] = M[1,2]
    return M*norm_fact_f


@njit(fastmath = True)
def single_wall_mobility_rot_rot_numba(r_vectors, h, eta, a):
    ''' 
    Returns the product of the mobility translation-rotation at the blob level to the torque 
    on the blobs. Mobility for particles in an unbounded domain, it uses
    the standard RPY tensor.  

    This function uses numba.
    '''
    
    # Compute mobility without wall:    
    M = no_wall_mobility_rot_rot_numba(r_vectors, eta, a)

    # Now add Swan-Brady wall correction:
    inva = 1.0 / a
    norm_fact_f = 1.0 / (8.0 * np.pi * eta * a**3)

    rx = r_vectors[0] * inva
    ry = r_vectors[1] * inva
    rz = (r_vectors[2] + 2*h) * inva  # Reciprocal in z

    invR = 1.0 / np.sqrt(rx*rx + ry*ry + rz*rz)
    invR3 = invR * invR * invR
    ex = rx * invR
    ey = ry * invR
    ez = rz * invR

    fact1 = ((1.0 - 6.0*ez*ez) * invR3) * 0.5
    fact2 = -(9.0 * invR3) / 6.0
    fact3 = (3.0 * invR3 * ez)
    fact4 = (3.0 * invR3)

    M[0,0] += (fact1 + fact2 * ex*ex + fact4 * ey*ey)*norm_fact_f
    M[0,1] += (fact2 - fact4) * ex*ey * norm_fact_f
    M[0,2] += fact2 * ex*ez * norm_fact_f
    M[1,0] += (fact2 - fact4) * ex*ey*norm_fact_f
    M[1,1] += (fact1 + fact2 * ey*ey + fact4 * ex*ex)*norm_fact_f
    M[1,2] += fact2 * ey*ez * norm_fact_f
    M[2,0] += (fact2 * ez*ex + fact3 * ex)*norm_fact_f
    M[2,1] += (fact2 * ez*ey + fact3 * ey)*norm_fact_f
    M[2,2] += (fact1 + fact2 * ez*ez + fact3 * ez)*norm_fact_f
    return M
