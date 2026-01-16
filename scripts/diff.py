"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

diff module
Provides functions to compute gradients, divergences, curls of scalar and vector fields defined
on the hierarchy of AMR simulations.

Created by David Vallés and enriched by Marco Molina Pradillo.
"""

import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline
import scripts.utils as tools

# On patch functions (arr_) #

@njit(fastmath=True)
def arr_diff_x(arr):
    nx = arr.shape[0]
    difference = np.zeros_like(arr)
    difference[1:nx-1,:,:] = arr[2:nx,:,:] - arr[0:nx-2,:,:]
    # Second order extrapolation at the boundary
    difference[0,:,:] = 4*arr[1,:,:] - 3*arr[0,:,:] - arr[2,:,:]
    difference[nx-1,:,:] = 3*arr[nx-1,:,:] + arr[nx-3,:,:] - 4*arr[nx-2,:,:]
    return difference

@njit(fastmath=True)
def arr_periodic_diff_x(arr):
    '''
    Computes the three-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    nx = arr.shape[0]
    difference = np.zeros_like(arr)
    difference[1:nx-1, :, :] = arr[2:nx, :, :] - arr[0:nx-2, :, :]  # The denominator is multiplied by 2*dx in the following functions
    difference[0, :, :] = arr[1, :, :] - arr[-1, :, :]  # Periodic boundary condition
    difference[nx-1, :, :] = arr[0, :, :] - arr[nx-2, :, :]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_diff_x_5_stencil(arr):
    '''
    Computes the five-point stencil derivative in this coordinate direction.

    Author: Marco Molina
    '''
    # The denominator is multiplied by 12*dx in the following functions
    nx = arr.shape[0]
    difference = np.zeros_like(arr)
    difference[2:nx-2,:,:] = -arr[4:nx,:,:] + 8*arr[3:nx-1,:,:] - 8*arr[1:nx-3,:,:] + arr[0:nx-4,:,:] 
    # Second order central difference at the second to last boundary
    difference[1,:,:] = 6*(arr[2,:,:] - arr[0,:,:]) 
    difference[nx-2,:,:] = 6*(arr[nx-1,:,:] - arr[nx-3,:,:]) 
    # Second order extrapolation at the boundary
    difference[0,:,:] = 6*(4*arr[1,:,:] - 3*arr[0,:,:] - arr[2,:,:]) 
    difference[nx-1,:,:] = 6*(3*arr[nx-1,:,:] + arr[nx-3,:,:] - 4*arr[nx-2,:,:]) 
    return difference

@njit(fastmath=True)
def arr_periodic_diff_x_5_stencil(arr):
    '''
    Computes the five-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    nx = arr.shape[0]
    difference = np.zeros_like(arr)
    difference[2:nx-2, :, :] = -arr[4:nx, :, :] + 8 * arr[3:nx-1, :, :] - 8 * arr[1:nx-3, :, :] + arr[0:nx-4, :, :]  # The denominator is multiplied by 12*dx in the following functions
    difference[1, :, :] = -arr[3, :, :] + 8 * arr[2, :, :] - 8 * arr[0, :, :] + arr[-1, :, :]  # Periodic boundary condition
    difference[nx-2, :, :] = -arr[0, :, :] + 8 * arr[nx-1, :, :] - 8 * arr[nx-3, :, :] + arr[nx-4, :, :]  # Periodic boundary condition
    difference[0, :, :] = -arr[2, :, :] + 8 * arr[1, :, :] - 8 * arr[-1, :, :] + arr[-2, :, :]  # Periodic boundary condition
    difference[nx-1, :, :] = -arr[1, :, :] + 8 * arr[0, :, :] - 8 * arr[nx-2, :, :] + arr[nx-3, :, :]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_diff_y(arr):
    ny = arr.shape[1]
    difference = np.zeros_like(arr)
    difference[:,1:ny-1,:] = arr[:,2:ny,:] - arr[:,0:ny-2,:]
    # Second order extrapolation at the boundary
    difference[:,0,:] = 4*arr[:,1,:] - 3*arr[:,0,:] - arr[:,2,:]
    difference[:,ny-1,:] = 3*arr[:,ny-1,:] + arr[:,ny-3,:] - 4*arr[:,ny-2,:]
    return difference

@njit(fastmath=True)
def arr_periodic_diff_y(arr):
    '''
    Computes the three-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    ny = arr.shape[1]
    difference = np.zeros_like(arr)
    difference[:, 1:ny-1, :] = arr[:, 2:ny, :] - arr[:, 0:ny-2, :]  # The denominator is multiplied by 2*dx in the following functions
    difference[:, 0, :] = arr[:, 1, :] - arr[:, -1, :]  # Periodic boundary condition
    difference[:, ny-1, :] = arr[:, 0, :] - arr[:, ny-2, :]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_diff_y_5_stencil(arr):
    '''
    Computes the five-point stencil derivative in this coordinate direction.

    Author: Marco Molina
    '''
    # The denominator is multiplied by 12*dx in the following functions
    ny = arr.shape[1]
    difference = np.zeros_like(arr)
    difference[:,2:ny-2,:] = -arr[:,4:ny,:] + 8*arr[:,3:ny-1,:] - 8*arr[:,1:ny-3,:] + arr[:,0:ny-4,:] 
    # Second order central difference at the second to last boundary
    difference[:,1,:] = 6*(arr[:,2,:] - arr[:,0,:]) 
    difference[:,ny-2,:] = 6*(arr[:,ny-1,:] - arr[:,ny-3,:]) 
    # Second order extrapolation at the boundary
    difference[:,0,:] = 6*(4*arr[:,1,:] - 3*arr[:,0,:] - arr[:,2,:]) 
    difference[:,ny-1,:] = 6*(3*arr[:,ny-1,:] + arr[:,ny-3,:] - 4*arr[:,ny-2,:])
    return difference

@njit(fastmath=True)
def arr_periodic_diff_y_5_stencil(arr):
    '''
    Computes the five-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    ny = arr.shape[1]
    difference = np.zeros_like(arr)
    difference[:, 2:ny-2, :] = -arr[:, 4:ny, :] + 8 * arr[:, 3:ny-1, :] - 8 * arr[:, 1:ny-3, :] + arr[:, 0:ny-4, :]  # The denominator is multiplied by 12*dx in the following functions
    difference[:, 1, :] = -arr[:, 3, :] + 8 * arr[:, 2, :] - 8 * arr[:, 0, :] + arr[:, -1, :]  # Periodic boundary condition
    difference[:, ny-2, :] = -arr[:, 0, :] + 8 * arr[:, ny-1, :] - 8 * arr[:, ny-3, :] + arr[:, ny-4, :]  # Periodic boundary condition
    difference[:, 0, :] = -arr[:, 2, :] + 8 * arr[:, 1, :] - 8 * arr[:, -1, :] + arr[:, -2, :]  # Periodic boundary condition
    difference[:, ny-1, :] = -arr[:, 1, :] + 8 * arr[:, 0, :] - 8 * arr[:, ny-2, :] + arr[:, ny-3, :]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_diff_z(arr):
    nz = arr.shape[2]
    difference = np.zeros_like(arr)
    difference[:,:,1:nz-1] = arr[:,:,2:nz] - arr[:,:,0:nz-2]
    # Second order extrapolation at the boundary
    difference[:,:,0] = 4*arr[:,:,1] - 3*arr[:,:,0] - arr[:,:,2]
    difference[:,:,nz-1] = 3*arr[:,:,nz-1] + arr[:,:,nz-3] - 4*arr[:,:,nz-2]
    return difference

@njit(fastmath=True)
def arr_periodic_diff_z(arr):
    '''
    Computes the three-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    nz = arr.shape[2]
    difference = np.zeros_like(arr)
    difference[:, :, 1:nz-1] = arr[:, :, 2:nz] - arr[:, :, 0:nz-2]  # The denominator is multiplied by 2*dx in the following functions
    difference[:, :, 0] = arr[:, :, 1] - arr[:, :, -1]  # Periodic boundary condition
    difference[:, :, nz-1] = arr[:, :, 0] - arr[:, :, nz-2]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_diff_z_5_stencil(arr):
    '''
    Computes the five-point stencil derivative in this coordinate direction.

    Author: Marco Molina
    '''
    # The denominator is multiplied by 12*dx in the following functions
    nz = arr.shape[2]
    difference = np.zeros_like(arr)
    difference[:,:,2:nz-2] = -arr[:,:,4:nz] + 8*arr[:,:,3:nz-1] - 8*arr[:,:,1:nz-3] + arr[:,:,0:nz-4] 
    # Second order central difference at the second to last boundary
    difference[:,:,1] = 6*(arr[:,:,2] - arr[:,:,0]) 
    difference[:,:,nz-2] = 6*(arr[:,:,nz-1] - arr[:,:,nz-3]) 
    # Second order extrapolation at the boundary
    difference[:,:,0] = 6*(4*arr[:,:,1] - 3*arr[:,:,0] - arr[:,:,2]) 
    difference[:,:,nz-1] = 6*(3*arr[:,:,nz-1] + arr[:,:,nz-3] - 4*arr[:,:,nz-2]) 
    return difference

@njit(fastmath=True)
def arr_periodic_diff_z_5_stencil(arr):
    '''
    Computes the five-point stencil derivative with periodic boundary conditions in this coordinate direction.

    Author: Marco Molina
    '''
    nz = arr.shape[2]
    difference = np.zeros_like(arr)
    difference[:, :, 2:nz-2] = -arr[:, :, 4:nz] + 8 * arr[:, :, 3:nz-1] - 8 * arr[:, :, 1:nz-3] + arr[:, :, 0:nz-4]  # The denominator is multiplied by 12*dx in the following functions
    difference[:, :, 1] = -arr[:, :, 3] + 8 * arr[:, :, 2] - 8 * arr[:, :, 0] + arr[:, :, -1]  # Periodic boundary condition
    difference[:, :, nz-2] = -arr[:, :, 0] + 8 * arr[:, :, nz-1] - 8 * arr[:, :, nz-3] + arr[:, :, nz-4]  # Periodic boundary condition
    difference[:, :, 0] = -arr[:, :, 2] + 8 * arr[:, :, 1] - 8 * arr[:, :, -1] + arr[:, :, -2]  # Periodic boundary condition
    difference[:, :, nz-1] = -arr[:, :, 1] + 8 * arr[:, :, 0] - 8 * arr[:, :, nz-2] + arr[:, :, nz-3]  # Periodic boundary condition
    return difference

@njit(fastmath=True)
def arr_gradient(arr, dx, stencil=3):
    if stencil==3:
        den = np.float32(1/(2*dx))
        return den*arr_diff_x(arr), den*arr_diff_y(arr), den*arr_diff_z(arr)
    elif stencil==5:
        den = np.float32(1/(12*dx))
        return den*arr_diff_x_5_stencil(arr), den*arr_diff_y_5_stencil(arr), den*arr_diff_z_5_stencil(arr)
    
@njit(fastmath=True)
def arr_periodic_gradient(arr, dx, stencil=3):
    if stencil==3:
        den = np.float32(1/(2*dx))
        return den*arr_periodic_diff_x(arr), den*arr_periodic_diff_y(arr), den*arr_periodic_diff_z(arr)
    elif stencil==5:
        den = np.float32(1/(12*dx))
        return den*arr_periodic_diff_x_5_stencil(arr), den*arr_periodic_diff_y_5_stencil(arr), den*arr_periodic_diff_z_5_stencil(arr)

@njit(fastmath=True)
def arr_gradient_magnitude(arr, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*np.sqrt(arr_diff_x(arr)**2 + arr_diff_y(arr)**2 + arr_diff_z(arr)**2)
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*np.sqrt(arr_diff_x_5_stencil(arr)**2 + arr_diff_y_5_stencil(arr)**2 + arr_diff_z_5_stencil(arr)**2)
    
@njit(fastmath=True)
def arr_periodic_gradient_magnitude(arr, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*np.sqrt(arr_periodic_diff_x(arr)**2 + arr_periodic_diff_y(arr)**2 + arr_periodic_diff_z(arr)**2)
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*np.sqrt(arr_periodic_diff_x_5_stencil(arr)**2 + arr_periodic_diff_y_5_stencil(arr)**2 + arr_periodic_diff_z_5_stencil(arr)**2)

@njit(fastmath=True)
def arr_divergence(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return (arr_diff_x(arr_x) + arr_diff_y(arr_y) + arr_diff_z(arr_z))*den
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return (arr_diff_x_5_stencil(arr_x) + arr_diff_y_5_stencil(arr_y) + arr_diff_z_5_stencil(arr_z))*den
    
@njit(fastmath=True)
def arr_periodic_divergence(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1 / (2 * dx))
        return (arr_periodic_diff_x(arr_x) + arr_periodic_diff_y(arr_y) + arr_periodic_diff_z(arr_z)) * den
    elif stencil == 5:
        den = np.float32(1 / (12 * dx))
        return (arr_periodic_diff_x_5_stencil(arr_x) + arr_periodic_diff_y_5_stencil(arr_y) + arr_periodic_diff_z_5_stencil(arr_z)) * den

@njit(fastmath=True)
def arr_curl(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*(arr_diff_y(arr_z) - arr_diff_z(arr_y)), den*(arr_diff_z(arr_x) - arr_diff_x(arr_z)), den*(arr_diff_x(arr_y) - arr_diff_y(arr_x))
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*(arr_diff_y_5_stencil(arr_z) - arr_diff_z_5_stencil(arr_y)), den*(arr_diff_z_5_stencil(arr_x) - arr_diff_x_5_stencil(arr_z)), den*(arr_diff_x_5_stencil(arr_y) - arr_diff_y_5_stencil(arr_x))

@njit(fastmath=True)
def arr_periodic_curl(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*(arr_periodic_diff_y(arr_z) - arr_periodic_diff_z(arr_y)), den*(arr_periodic_diff_z(arr_x) - arr_periodic_diff_x(arr_z)), den*(arr_periodic_diff_x(arr_y) - arr_periodic_diff_y(arr_x))
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*(arr_periodic_diff_y_5_stencil(arr_z) - arr_periodic_diff_z_5_stencil(arr_y)), den*(arr_periodic_diff_z_5_stencil(arr_x) - arr_periodic_diff_x_5_stencil(arr_z)), den*(arr_periodic_diff_x_5_stencil(arr_y) - arr_periodic_diff_y_5_stencil(arr_x))

@njit(fastmath=True)
def arr_curl_magnitude(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*np.sqrt((arr_diff_y(arr_z) - arr_diff_z(arr_y))**2 +
                        (arr_diff_z(arr_x) - arr_diff_x(arr_z))**2 + 
                        (arr_diff_x(arr_y) - arr_diff_y(arr_x))**2)
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*np.sqrt((arr_diff_y_5_stencil(arr_z) - arr_diff_z_5_stencil(arr_y))**2 +
                        (arr_diff_z_5_stencil(arr_x) - arr_diff_x_5_stencil(arr_z))**2 + 
                        (arr_diff_x_5_stencil(arr_y) - arr_diff_y_5_stencil(arr_x))**2)

@njit(fastmath=True)
def arr_periodic_curl_magnitude(arr_x, arr_y, arr_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*np.sqrt((arr_periodic_diff_y(arr_z) - arr_periodic_diff_z(arr_y))**2 +
                        (arr_periodic_diff_z(arr_x) - arr_periodic_diff_x(arr_z))**2 + 
                        (arr_periodic_diff_x(arr_y) - arr_periodic_diff_y(arr_x))**2)
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*np.sqrt((arr_periodic_diff_y_5_stencil(arr_z) - arr_periodic_diff_z_5_stencil(arr_y))**2 +
                        (arr_periodic_diff_z_5_stencil(arr_x) - arr_periodic_diff_x_5_stencil(arr_z))**2 + 
                        (arr_periodic_diff_x_5_stencil(arr_y) - arr_periodic_diff_y_5_stencil(arr_x))**2)

@njit(fastmath=True)
def arr_u_nabla_phi(arrphi, arru_x, arru_y, arru_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*(arru_x*arr_diff_x(arrphi) + arru_y*arr_diff_y(arrphi) + arru_z*arr_diff_z(arrphi))
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*(arru_x*arr_diff_x_5_stencil(arrphi) + arru_y*arr_diff_y_5_stencil(arrphi) + arru_z*arr_diff_z_5_stencil(arrphi))
    
@njit(fastmath=True)
def arr_periodic_u_nabla_phi(arrphi, arru_x, arru_y, arru_z, dx, stencil=3):
    if stencil == 3:
        den = np.float32(1/(2*dx))
        return den*(arru_x*arr_periodic_diff_x(arrphi) + arru_y*arr_periodic_diff_y(arrphi) + arru_z*arr_periodic_diff_z(arrphi))
    elif stencil == 5:
        den = np.float32(1/(12*dx))
        return den*(arru_x*arr_periodic_diff_x_5_stencil(arrphi) + arru_y*arr_periodic_diff_y_5_stencil(arrphi) + arru_z*arr_periodic_diff_z_5_stencil(arrphi))

@njit(fastmath=True)
def arr_u_nabla_v(arrv_x, arrv_y, arrv_z, arru_x, arru_y, arru_z, dx, stencil=3):
    if stencil == 3:
        return arr_u_nabla_phi(arrv_x, arru_x, arru_y, arru_z, dx), arr_u_nabla_phi(arrv_y, arru_x, arru_y, arru_z, dx), arr_u_nabla_phi(arrv_z, arru_x, arru_y, arru_z, dx)
    elif stencil == 5:
        return arr_u_nabla_phi(arrv_x, arru_x, arru_y, arru_z, dx, stencil=5), arr_u_nabla_phi(arrv_y, arru_x, arru_y, arru_z, dx, stencil=5), arr_u_nabla_phi(arrv_z, arru_x, arru_y, arru_z, dx, stencil=5)

@njit(fastmath=True)
def arr_periodic_u_nabla_v(arrv_x, arrv_y, arrv_z, arru_x, arru_y, arru_z, dx, stencil=3):
    if stencil == 3:
        return arr_periodic_u_nabla_phi(arrv_x, arru_x, arru_y, arru_z, dx), arr_periodic_u_nabla_phi(arrv_y, arru_x, arru_y, arru_z, dx), arr_periodic_u_nabla_phi(arrv_z, arru_x, arru_y, arru_z, dx)
    elif stencil == 5:
        return arr_periodic_u_nabla_phi(arrv_x, arru_x, arru_y, arru_z, dx, stencil=5), arr_periodic_u_nabla_phi(arrv_y, arru_x, arru_y, arru_z, dx, stencil=5), arr_periodic_u_nabla_phi(arrv_z, arru_x, arru_y, arru_z, dx, stencil=5)

# Fields funtions #

def gradient(field, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the gradient of a scalar field defined on the AMR hierarchy of
    grids.

    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        grad_x: a list of numpy arrays, each one containing the x-component of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy
        grad_y: idem for the y-component
        grad_z: idem for the z-component

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    grad_x = []
    grad_y = []
    grad_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum() + 1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                gx, gy, gz = arr_periodic_gradient(field[ipatch], resolution[ipatch], stencil=5)
            else:
                # Refined levels: use regular boundaries with extrapolation
                gx, gy, gz = arr_gradient(field[ipatch], resolution[ipatch], stencil=stencil)
        else:
            gx, gy, gz = 0, 0, 0
        grad_x.append(gx)
        grad_y.append(gy)
        grad_z.append(gz)

    return grad_x, grad_y, grad_z

def divergence(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the divergence of a vector field defined on the AMR hierarchy of
    grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of 
                the vector field defined on the corresponding grid of the AMR 
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        div: a list of numpy arrays, each one containing the divergence of the
                vector field defined on the corresponding grid of the AMR    
                hierarchy 

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    div = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                div.append(arr_periodic_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=5))
            else:
                # Refined levels: use regular boundaries with extrapolation
                div.append(arr_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil))
        else:
            div.append(0)

    return div

def curl(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the curl of a vector field defined on the AMR hierarchy of
    grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        curl_x: a list of numpy arrays, each one containing the x-component of
                the curl of the vector field defined on the corresponding grid  
                of the AMR hierarchy
        curl_y: idem for the y-component
        curl_z: idem for the z-component

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    curl_x = []
    curl_y = []
    curl_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                cx,cy,cz = arr_periodic_curl(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=5)
            else:
                # Refined levels: use regular boundaries with extrapolation
                cx,cy,cz = arr_curl(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil)
        else:
            cx,cy,cz = 0,0,0
        curl_x.append(cx)
        curl_y.append(cy)
        curl_z.append(cz)

    return curl_x, curl_y, curl_z

def curl_magnitude(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the magnitude of the curl of a vector field defined on the 
    AMR hierarchy of grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        curl_mag: a list of numpy arrays, each one containing the magnitude of
                the curl of the vector field defined on the corresponding grid 
                of the AMR hierarchy

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    curl_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                curl_mag.append(arr_periodic_curl_magnitude(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=5))
            else:
                # Refined levels: use regular boundaries with extrapolation
                curl_mag.append(arr_curl_magnitude(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil))
        else:
            curl_mag.append(0)

    return curl_mag

def gradient_magnitude(field, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the magnitude of the gradient of a scalar field defined on 
    the AMR hierarchy of grids.

    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        grad_mag: a list of numpy arrays, each one containing the magnitude of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy

    Author: David Vallés
    '''  
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    grad_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                grad_mag.append(arr_periodic_gradient_magnitude(field[ipatch], resolution[ipatch], stencil=5))
            else:
                # Refined levels: use regular boundaries with extrapolation
                grad_mag.append(arr_gradient_magnitude(field[ipatch], resolution[ipatch], stencil=stencil))
        else:
            grad_mag.append(0)

    return grad_mag

def periodic_gradient(field, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the gradient of a scalar field defined on the AMR hierarchy of
    grids with periodic boundary conditions.
    
    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5
        
    Returns:
        grad_x: a list of numpy arrays, each one containing the x-component of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy
        grad_y: idem for the y-component
        grad_z: idem for the z-component
        
    Author: Marco Molina
    '''
    levels = tools.create_vector_levels(npatch)
    resolution = dx / 2**levels
    grad_x = []
    grad_y = []
    grad_z = []
    
    if kept_patches is None:
        kept_patches = np.ones(npatch.sum() + 1, dtype=bool)
        
    for ipatch in prange(npatch.sum() + 1):
        if kept_patches[ipatch]:
            gx, gy, gz = arr_periodic_gradient(field[ipatch], resolution[ipatch], stencil=stencil)
        else:
            gx, gy, gz = 0, 0, 0
        grad_x.append(gx)
        grad_y.append(gy)
        grad_z.append(gz)
        
    return grad_x, grad_y, grad_z

def periodic_divergence(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3,):
    '''
    Computes the divergence of a vector field defined on the AMR hierarchy of
    grids with periodic boundary conditions.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of 
                the vector field defined on the corresponding grid of the AMR 
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use, either 3 (3-point) or 5 (5-point)

    Returns:
        div: a list of numpy arrays, each one containing the divergence of the
                vector field defined on the corresponding grid of the AMR    
                hierarchy 

    Author: Marco Molina
    '''
    levels = tools.create_vector_levels(npatch)
    resolution = dx / 2**levels
    div = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum() + 1, dtype=bool)

    for ipatch in prange(npatch.sum() + 1):
        if kept_patches[ipatch]:
            div.append(arr_periodic_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil))
        else:
            div.append(0)

    return div

def periodic_curl(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the curl of a vector field defined on the AMR hierarchy of
    grids with periodic boundary conditions.
    
    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5
        
    Returns:
        curl_x: a list of numpy arrays, each one containing the x-component of
                the curl of the vector field defined on the corresponding grid  
                of the AMR hierarchy
        curl_y: idem for the y-component
        curl_z: idem for the z-component
        
    Author: Marco Molina
    '''
    levels = tools.create_vector_levels(npatch)
    resolution = dx / 2**levels
    curl_x = []
    curl_y = []
    curl_z = []
    
    if kept_patches is None:
        kept_patches = np.ones(npatch.sum() + 1, dtype=bool)
        
    for ipatch in prange(npatch.sum() + 1):
        if kept_patches[ipatch]:
            cx, cy, cz = arr_periodic_curl(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil)
        else:
            cx, cy, cz = 0, 0, 0
        curl_x.append(cx)
        curl_y.append(cy)
        curl_z.append(cz)
        
    return curl_x, curl_y, curl_z

def periodic_curl_magnitude(field_x, field_y, field_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the magnitude of the curl of a vector field defined on the 
    AMR hierarchy of grids with periodic boundary conditions.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        curl_mag: a list of numpy arrays, each one containing the magnitude of
                the curl of the vector field defined on the corresponding grid 
                of the AMR hierarchy

    Author: Marco Molina
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    curl_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            curl_mag.append(arr_periodic_curl_magnitude(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil))
        else:
            curl_mag.append(0)

    return curl_mag

def periodic_gradient_magnitude(field, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes the magnitude of the gradient of a scalar field defined on 
    the AMR hierarchy of grids with periodic boundary conditions.

    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        grad_mag: a list of numpy arrays, each one containing the magnitude of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy

    Author: Marco Molina
    '''  
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    grad_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            grad_mag.append(arr_periodic_gradient_magnitude(field[ipatch], resolution[ipatch], stencil=stencil))
        else:
            grad_mag.append(0)

    return grad_mag

def directional_derivative_scalar_field(sfield, ufield_x, ufield_y, ufield_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes (\vb{u} \cdot \nabla) \phi, where \vb{u} is a vector field and
    \phi is a scalar field, defined on the AMR hierarchy of grids.

    Args:
        sfield: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        ufield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        ufield_y: idem for the y-component
        ufield_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        u_nabla_phi: a list of numpy arrays, each one containing the result of
                the operation (\vb{u} \cdot \nabla) \phi defined on the
                corresponding grid of the AMR hierarchy

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    u_nabla_phi = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                u_nabla_phi.append(arr_periodic_u_nabla_phi(sfield[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch], stencil=5))
            else:
                # Refined levels: use regular boundaries with extrapolation
                u_nabla_phi.append(arr_u_nabla_phi(sfield[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch], stencil=stencil))
        else:
            u_nabla_phi.append(0)

    return u_nabla_phi

def periodic_directional_derivative_scalar_field(sfield, ufield_x, ufield_y, ufield_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes (\vb{u} \cdot \nabla) \phi, where \vb{u} is a vector field and
    \phi is a scalar field, defined on the AMR hierarchy of grids with periodic
    boundary conditions.

    Args:
        sfield: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        ufield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        ufield_y: idem for the y-component
        ufield_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        u_nabla_phi: a list of numpy arrays, each one containing the result of
                the operation (\vb{u} \cdot \nabla) \phi defined on the
                corresponding grid of the AMR hierarchy

    Author: Marco Molina
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    u_nabla_phi = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            u_nabla_phi.append(arr_periodic_u_nabla_phi(sfield[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch], stencil=stencil))
        else:
            u_nabla_phi.append(0)

    return u_nabla_phi

def directional_derivative_vector_field(vfield_x, vfield_y, vfield_z, ufield_x, ufield_y, ufield_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes (\vb{u} \cdot \nabla) \vb{v}, where \vb{u} and \vb{v} are vector
    fields defined on the AMR hierarchy of grids.

    Args:
        vfield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        vfield_y: idem for the y-component
        vfield_z: idem for the z-component
        ufield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        ufield_y: idem for the y-component
        ufield_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        u_nabla_v: a list of numpy arrays, each one containing the result of
                the operation (\vb{u} \cdot \nabla) \vb{v} defined on the
                corresponding grid of the AMR hierarchy
    
    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    u_nabla_v_x = []
    u_nabla_v_y = []
    u_nabla_v_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            if ipatch == 0:
                # Level 0: use periodic boundaries
                ux,uy,uz = arr_periodic_u_nabla_v(vfield_x[ipatch], vfield_y[ipatch], vfield_z[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch],
                                        stencil=5)
            else:
                # Refined levels: use regular boundaries with extrapolation
                ux,uy,uz = arr_u_nabla_v(vfield_x[ipatch], vfield_y[ipatch], vfield_z[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch],
                                        stencil=stencil)
        else:
            ux,uy,uz = 0,0,0
        u_nabla_v_x.append(ux)
        u_nabla_v_y.append(uy)
        u_nabla_v_z.append(uz)

    return u_nabla_v_x, u_nabla_v_y, u_nabla_v_z

def periodic_directional_derivative_vector_field(vfield_x, vfield_y, vfield_z, ufield_x, ufield_y, ufield_z, dx, npatch, kept_patches=None, stencil=3):
    '''
    Computes (\vb{u} \cdot \nabla) \vb{v}, where \vb{u} and \vb{v} are vector
    fields defined on the AMR hierarchy of grids with periodic boundary conditions.

    Args:
        vfield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        vfield_y: idem for the y-component
        vfield_z: idem for the z-component
        ufield_x: a list of numpy arrays, each one containing the x-component
                of the vector field defined on the corresponding grid of the AMR
                hierarchy
        ufield_y: idem for the y-component
        ufield_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.
        stencil: the stencil to use for the derivatives, either 3 or 5

    Returns:
        u_nabla_v: a list of numpy arrays, each one containing the result of
                the operation (\vb{u} \cdot \nabla) \vb{v} defined on the
                corresponding grid of the AMR hierarchy
    
    Author: Marco Molina
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    u_nabla_v_x = []
    u_nabla_v_y = []
    u_nabla_v_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    for ipatch in prange(npatch.sum()+1):
        if kept_patches[ipatch]:
            ux,uy,uz = arr_periodic_u_nabla_v(vfield_x[ipatch], vfield_y[ipatch], vfield_z[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch],
                                        stencil=stencil)
        else:
            ux,uy,uz = 0,0,0
        u_nabla_v_x.append(ux)
        u_nabla_v_y.append(uy)
        u_nabla_v_z.append(uz)

    return u_nabla_v_x, u_nabla_v_y, u_nabla_v_z

def rhs_energy(t, f_term):
    '''
    Evaluate dE/dt for the magnetic energy ODE at time `t`.
    
    Args:
        - t : float. Current time.
        - f_term : callable. Source-term integrand returning ∫_V f(B, v) dV at the queried time.
        
    Returns:
        - float. Time derivative dE/dt.
        
    Author: Marco Molina
    '''
    return 2.0 * f_term(t)

def rk4_step(t, dt, E, rho_b, f_term):
    '''
    Advance the magnetic energy one Runge–Kutta 4 step.
    
    Args:
        - t : float. Current time.
        - dt : float. Integration step size.
        - E : float. Magnetic energy at `t`.
        - rho_b : object. Background-density wrapper supplying `value` and `derivative`.
        - f_term : callable. Source-term integrand as in `rhs_energy`.
        
    Returns:
        - float. Updated magnetic energy at time `t + dt`.
        
    Author: Marco Molina
    '''
    k1 = rhs_energy(t,         f_term)
    k2 = rhs_energy(t + dt/2., f_term)
    k3 = rhs_energy(t + dt/2., f_term)
    k4 = rhs_energy(t + dt,    f_term)
    return (E/rho_b(t)) + (dt/6.) * (k1 + 2*k2 + 2*k3 + k4)

class TabulatedSignal:
    '''
    Hold a time-sampled signal and expose value/derivative callables.
    
    Args:
        - t : array_like. Monotonic time samples.
        - y : array_like. Signal values at `t`.
        - kind : {"cubic", "linear"}, optional. Interpolation scheme; cubic uses `CubicSpline`, linear relies on
                `numpy.interp` and a gradient-based derivative
                
    Returns:
        - value : callable. Interpolated signal value at arbitrary times.
        - derivative : callable. Interpolated signal derivative at arbitrary times.

    Author: Marco Molina
    '''                      
    def __init__(self, t, y, kind="cubic"):
        if kind == "cubic":
            self._spline = CubicSpline(t, y)
            self.value = self._spline
            self.derivative = self._spline.derivative()
        else:
            self._t = np.asarray(t, dtype=np.float64)
            self._y = np.asarray(y, dtype=np.float64)
            self._dy = np.gradient(self._y, self._t)
        if kind != "cubic":
            self.value = lambda x: np.interp(x, self._t, self._y)
            self.derivative = lambda x: np.interp(x, self._t, self._dy)

def integrate_energy(t, E0, rho_b_samples, f_samples):
    '''
    Integrate the magnetic energy ODE over the supplied time grid.

    Args:
        - t : array_like. Monotonic time samples.
        - E0 : float. Initial magnetic energy at `t[0]`.
        - rho_b_samples : array_like. Background density values matching `t`.
        - f_samples : array_like. Source term samples matching `t`.

    Returns:
        - E : array_like. Integrated magnetic energy at every entry of `t`.
        
    Author: Marco Molina
    '''
    rho_b = TabulatedSignal(t, rho_b_samples, kind="cubic")
    f_term = TabulatedSignal(t, f_samples, kind="cubic")
    E = np.empty_like(t, dtype=np.float64)
    E[0] = E0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        E[i+1] = rho_b_samples[i+1] * rk4_step(t[i], dt, E[i], rho_b.value, f_term.value)
    return E


@njit(fastmath=True)
def arr_parent_diff_x_at_boundary(field, field_parent, 
                                ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                dx_parent, resolution_ratio, nghost=0, stencil=3):
    '''
    Computes df/dx at patch boundaries using parent derivatives.
    Uses parent spacing directly to avoid over-scaling on refined grids.
    
    Args:
        field: child patch field array (shape can include buffer)
        field_parent: parent patch field array
        ipatch_in_parent, jpatch_in_parent, kpatch_in_parent: child's starting position in parent (0-based)
        dx_parent: parent cell size
        resolution_ratio: dx_parent / dx_child (typically 2)
        nghost: number of ghost cells if field has buffer
        stencil: 3 or 5 point stencil
        
    Returns:
        difference: df/dx at X-boundaries only (rest is zero), same shape as field
        
    Author: Marco Molina
    '''
    difference = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    # Real patch boundaries (excluding buffer if present)
    i_start = nghost
    i_end = nx - nghost - 1
    j_start = nghost
    j_end = ny - nghost - 1
    k_start = nghost
    k_end = nz - nghost - 1
    
    # Left X-boundary (i = i_start)
    for j in range(j_start, j_end + 1):
        for k in range(k_start, k_end + 1):
            i_local = 0
            j_local = j - j_start
            k_local = k - k_start
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if ip > 0 and ip < field_parent.shape[0] - 1 and jp > 0 and jp < field_parent.shape[1] - 1 and kp > 0 and kp < field_parent.shape[2] - 1:
                if stencil == 3:
                    difference[i_start, j, k] = (field_parent[ip + 1, jp, kp] - field_parent[ip - 1, jp, kp]) / (2.0 * dx_parent)
                else:
                    if ip > 1 and ip < field_parent.shape[0] - 2:
                        difference[i_start, j, k] = (-field_parent[ip + 2, jp, kp] + 8.0 * field_parent[ip + 1, jp, kp] - 8.0 * field_parent[ip - 1, jp, kp] + field_parent[ip - 2, jp, kp]) / (12.0 * dx_parent)
    
    # Right X-boundary (i = i_end)
    for j in range(j_start, j_end + 1):
        for k in range(k_start, k_end + 1):
            i_local = i_end - i_start
            j_local = j - j_start
            k_local = k - k_start
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if ip > 0 and ip < field_parent.shape[0] - 1 and jp > 0 and jp < field_parent.shape[1] - 1 and kp > 0 and kp < field_parent.shape[2] - 1:
                if stencil == 3:
                    difference[i_end, j, k] = (field_parent[ip + 1, jp, kp] - field_parent[ip - 1, jp, kp]) / (2.0 * dx_parent)
                else:
                    if ip > 1 and ip < field_parent.shape[0] - 2:
                        difference[i_end, j, k] = (-field_parent[ip + 2, jp, kp] + 8.0 * field_parent[ip + 1, jp, kp] - 8.0 * field_parent[ip - 1, jp, kp] + field_parent[ip - 2, jp, kp]) / (12.0 * dx_parent)
    
    return difference

@njit(fastmath=True)
def arr_parent_diff_y_at_boundary(field, field_parent, 
                                ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                dx_parent, resolution_ratio, nghost=0, stencil=3):
    '''
    Computes df/dy at patch boundaries using parent derivatives.
    
    Args:
        field: child patch field array (shape can include buffer)
        field_parent: parent patch field array
        ipatch_in_parent, jpatch_in_parent, kpatch_in_parent: child's starting position in parent (0-based)
        dx_parent: parent cell size
        resolution_ratio: dx_parent / dx_child (typically 2)
        nghost: number of ghost cells if field has buffer
        stencil: 3 or 5 point stencil
        
    Returns:
        difference: df/dy at Y-boundaries only (rest is zero), same shape as field
        
    Author: Marco Molina
    '''
    difference = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    i_start = nghost
    i_end = nx - nghost - 1
    j_start = nghost
    j_end = ny - nghost - 1
    k_start = nghost
    k_end = nz - nghost - 1
    
    # Front Y-boundary (j = j_start)
    for i in range(i_start, i_end + 1):
        for k in range(k_start, k_end + 1):
            i_local = i - i_start
            j_local = 0
            k_local = k - k_start
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if jp > 0 and jp < field_parent.shape[1] - 1 and ip > 0 and ip < field_parent.shape[0] - 1 and kp > 0 and kp < field_parent.shape[2] - 1:
                if stencil == 3:
                    difference[i, j_start, k] = (field_parent[ip, jp + 1, kp] - field_parent[ip, jp - 1, kp]) / (2.0 * dx_parent)
                else:
                    if jp > 1 and jp < field_parent.shape[1] - 2:
                        difference[i, j_start, k] = (-field_parent[ip, jp + 2, kp] + 8.0 * field_parent[ip, jp + 1, kp] - 8.0 * field_parent[ip, jp - 1, kp] + field_parent[ip, jp - 2, kp]) / (12.0 * dx_parent)
    
    # Back Y-boundary (j = j_end)
    for i in range(i_start, i_end + 1):
        for k in range(k_start, k_end + 1):
            i_local = i - i_start
            j_local = j_end - j_start
            k_local = k - k_start
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if jp > 0 and jp < field_parent.shape[1] - 1 and ip > 0 and ip < field_parent.shape[0] - 1 and kp > 0 and kp < field_parent.shape[2] - 1:
                if stencil == 3:
                    difference[i, j_end, k] = (field_parent[ip, jp + 1, kp] - field_parent[ip, jp - 1, kp]) / (2.0 * dx_parent)
                else:
                    if jp > 1 and jp < field_parent.shape[1] - 2:
                        difference[i, j_end, k] = (-field_parent[ip, jp + 2, kp] + 8.0 * field_parent[ip, jp + 1, kp] - 8.0 * field_parent[ip, jp - 1, kp] + field_parent[ip, jp - 2, kp]) / (12.0 * dx_parent)
    
    return difference

@njit(fastmath=True)
def arr_parent_diff_z_at_boundary(field, field_parent, 
                                ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                dx_parent, resolution_ratio, nghost=0, stencil=3):    
    '''
    Computes df/dz at patch boundaries using parent derivatives.
    
    Args:
        field: child patch field array (shape can include buffer)
        field_parent: parent patch field array
        ipatch_in_parent, jpatch_in_parent, kpatch_in_parent: child's starting position in parent (0-based)
        dx_parent: parent cell size
        resolution_ratio: dx_parent / dx_child (typically 2)
        nghost: number of ghost cells if field has buffer
        stencil: 3 or 5 point stencil
        
    Returns:
        difference: df/dz at Z-boundaries only (rest is zero), same shape as field
        
    Author: Marco Molina
    '''
    difference = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    i_start = nghost
    i_end = nx - nghost - 1
    j_start = nghost
    j_end = ny - nghost - 1
    k_start = nghost
    k_end = nz - nghost - 1
    
    # Bottom Z-boundary (k = k_start)
    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            i_local = i - i_start
            j_local = j - j_start
            k_local = 0
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if kp > 0 and kp < field_parent.shape[2] - 1 and ip > 0 and ip < field_parent.shape[0] - 1 and jp > 0 and jp < field_parent.shape[1] - 1:
                if stencil == 3:
                    difference[i, j, k_start] = (field_parent[ip, jp, kp + 1] - field_parent[ip, jp, kp - 1]) / (2.0 * dx_parent)
                else:
                    if kp > 1 and kp < field_parent.shape[2] - 2:
                        difference[i, j, k_start] = (-field_parent[ip, jp, kp + 2] + 8.0 * field_parent[ip, jp, kp + 1] - 8.0 * field_parent[ip, jp, kp - 1] + field_parent[ip, jp, kp - 2]) / (12.0 * dx_parent)
    
    # Top Z-boundary (k = k_end)
    for i in range(i_start, i_end + 1):
        for j in range(j_start, j_end + 1):
            i_local = i - i_start
            j_local = j - j_start
            k_local = k_end - k_start
            
            ip = int(ipatch_in_parent + i_local // int(resolution_ratio))
            jp = int(jpatch_in_parent + j_local // int(resolution_ratio))
            kp = int(kpatch_in_parent + k_local // int(resolution_ratio))
            
            if kp > 0 and kp < field_parent.shape[2] - 1 and ip > 0 and ip < field_parent.shape[0] - 1 and jp > 0 and jp < field_parent.shape[1] - 1:
                if stencil == 3:
                    difference[i, j, k_end] = (field_parent[ip, jp, kp + 1] - field_parent[ip, jp, kp - 1]) / (2.0 * dx_parent)
                else:
                    if kp > 1 and kp < field_parent.shape[2] - 2:
                        difference[i, j, k_end] = (-field_parent[ip, jp, kp + 2] + 8.0 * field_parent[ip, jp, kp + 1] - 8.0 * field_parent[ip, jp, kp - 1] + field_parent[ip, jp, kp - 2]) / (12.0 * dx_parent)
    
    return difference

@njit(fastmath=True)
def arr_divergence_with_parent_boundaries(arr_x, arr_y, arr_z, dx,
                                        arr_x_parent, arr_y_parent, arr_z_parent, dx_parent,
                                        ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                        nghost=0, stencil=3):
    '''
    Computes divergence with parent-based derivatives at boundaries.
    Combines standard divergence in interior with parent-based at boundaries.
    
    Args:
        arr_x, arr_y, arr_z: child patch vector field components (can include buffer)
        dx: child cell size
        arr_x_parent, arr_y_parent, arr_z_parent: parent patch vector field components
        dx_parent: parent cell size
        ipatch_in_parent, jpatch_in_parent, kpatch_in_parent: child position in parent (0-based)
        nghost: number of ghost cells if arrays have buffer
        stencil: 3 or 5 point stencil
        
    Returns:
        div: divergence array, same shape as input arrays
        
    Author: Marco Molina
    '''
    # Standard divergence on full array
    div = arr_divergence(arr_x, arr_y, arr_z, dx, stencil=stencil)
    
    # Resolution ratio (used only to locate child cells in parent)
    res_ratio = dx_parent / dx
    
    # Parent-based derivatives at boundaries (already divided by dx_parent)
    diff_x_parent = arr_parent_diff_x_at_boundary(arr_x, arr_x_parent, 
                                                  ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                                  dx_parent, res_ratio, nghost, stencil)
    diff_y_parent = arr_parent_diff_y_at_boundary(arr_y, arr_y_parent,
                                                  ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                                  dx_parent, res_ratio, nghost, stencil)
    diff_z_parent = arr_parent_diff_z_at_boundary(arr_z, arr_z_parent,
                                                  ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                                                  dx_parent, res_ratio, nghost, stencil)
    
    # Divergence from parent at boundaries (no extra dx scaling needed)
    div_parent = diff_x_parent + diff_y_parent + diff_z_parent
    
    # Average standard and parent divergence at boundaries
    nx, ny, nz = arr_x.shape
    i_start = nghost
    i_end = nx - nghost - 1
    j_start = nghost
    j_end = ny - nghost - 1
    k_start = nghost
    k_end = nz - nghost - 1
    
    # X-boundaries
    div[i_start, j_start:j_end+1, k_start:k_end+1] = 0.5 * (div[i_start, j_start:j_end+1, k_start:k_end+1] + 
                                                            div_parent[i_start, j_start:j_end+1, k_start:k_end+1])
    div[i_end, j_start:j_end+1, k_start:k_end+1] = 0.5 * (div[i_end, j_start:j_end+1, k_start:k_end+1] + 
                                                          div_parent[i_end, j_start:j_end+1, k_start:k_end+1])
    
    # Y-boundaries
    div[i_start:i_end+1, j_start, k_start:k_end+1] = 0.5 * (div[i_start:i_end+1, j_start, k_start:k_end+1] + 
                                                            div_parent[i_start:i_end+1, j_start, k_start:k_end+1])
    div[i_start:i_end+1, j_end, k_start:k_end+1] = 0.5 * (div[i_start:i_end+1, j_end, k_start:k_end+1] + 
                                                          div_parent[i_start:i_end+1, j_end, k_start:k_end+1])
    
    # Z-boundaries
    div[i_start:i_end+1, j_start:j_end+1, k_start] = 0.5 * (div[i_start:i_end+1, j_start:j_end+1, k_start] + 
                                                            div_parent[i_start:i_end+1, j_start:j_end+1, k_start])
    div[i_start:i_end+1, j_start:j_end+1, k_end] = 0.5 * (div[i_start:i_end+1, j_start:j_end+1, k_end] + 
                                                          div_parent[i_start:i_end+1, j_start:j_end+1, k_end])
    
    return div


def divergence_with_parent_boundaries(field_x, field_y, field_z, dx, npatch, 
                                    patchpare, patchnx, patchny, patchnz,
                                    patchrx, patchry, patchrz,
                                    buffer_active=False, nghost=1,
                                    kept_patches=None, stencil=3):
    '''
    Computes divergence of vector field with parent-based derivatives at boundaries.
    High-level function that iterates over patches.
    
    Implements Fortran-like approach: boundaries use rescaled parent derivatives.
    When buffer_active=True, averages buffer-based and parent-based results at real boundaries.
    
    Args:
        field_x, field_y, field_z: vector field components (list of arrays per patch)
        dx: coarsest grid cell size
        npatch: number of patches per level
        patchpare: parent patch indices
        patchnx, patchny, patchnz: patch dimensions (without buffer)
        patchrx, patchry, patchrz: patch positions in parent (cell indices, 0-based)
        buffer_active: whether fields have ghost buffer zones
        nghost: number of ghost cells if buffer_active=True
        kept_patches: boolean array for patches to process (None = all)
        stencil: derivative stencil (3 or 5)
    
    Returns:
        div: list of divergence arrays per patch
    
    Author: Marco Molina
    '''
    levels = tools.create_vector_levels(npatch)
    resolution = dx / 2**levels
    div = []
    
    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)
    
    for ipatch in range(npatch.sum()+1):
        if not kept_patches[ipatch]:
            div.append(0)
            continue
        
        if ipatch == 0:
            # Level 0: periodic boundaries (no parent)
            div_patch = arr_periodic_divergence(field_x[ipatch], field_y[ipatch], 
                                            field_z[ipatch], resolution[ipatch], stencil=5)
            div.append(div_patch)
            continue
        
        # Refined patches: use parent for boundaries
        parent_idx = int(patchpare[ipatch])
        
        if parent_idx >= 0 and parent_idx < len(field_x):
            # Valid parent exists
            ipatch_in_parent = int(patchrx[ipatch])
            jpatch_in_parent = int(patchry[ipatch])
            kpatch_in_parent = int(patchrz[ipatch])
            
            ng = nghost if buffer_active else 0
            
            div_patch = arr_divergence_with_parent_boundaries(
                field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch],
                field_x[parent_idx], field_y[parent_idx], field_z[parent_idx], resolution[parent_idx],
                ipatch_in_parent, jpatch_in_parent, kpatch_in_parent,
                nghost=ng, stencil=stencil
            )
            div.append(div_patch)
        else:
            # No valid parent: use standard divergence
            div_patch = arr_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], 
                                        resolution[ipatch], stencil=stencil)
            div.append(div_patch)
    
    return div