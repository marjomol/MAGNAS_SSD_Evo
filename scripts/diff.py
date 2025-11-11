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
                gx, gy, gz = arr_periodic_gradient(field[ipatch], resolution[ipatch], stencil=stencil)
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
                div.append(arr_periodic_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil))
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
                cx,cy,cz = arr_periodic_curl(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil)
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
                curl_mag.append(arr_periodic_curl_magnitude(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch], stencil=stencil))
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
                grad_mag.append(arr_periodic_gradient_magnitude(field[ipatch], resolution[ipatch], stencil=stencil))
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
                u_nabla_phi.append(arr_periodic_u_nabla_phi(sfield[ipatch], ufield_x[ipatch], ufield_y[ipatch], ufield_z[ipatch], resolution[ipatch], stencil=stencil))
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
                                        stencil=stencil)
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