"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

buffer module
Provides functions to create and manage data buffers for their proper AMR differential treatment by the diff.py module.

Created by Marco Molina Pradillo and Òscar Monllor.
"""

import numpy as np
from numba import njit, prange
from numba.typed import List
import scripts.utils as tools

@njit(fastmath=True)
def TSC_kernel(x, h):
    """
    Triangular-Shaped Cloud (TSC) kernel for interpolation.
    
    Args:
        x: distance from cell center
        h: cell size
        
    Returns:
        Weight for TSC interpolation (between 0 and 0.75)
        
    Author: Òscar Monllor
    """
    ax = abs(x) / h
    if ax <= 0.5:
        w = 0.75 - ax**2
    elif ax > 0.5 and ax <= 1.5:
        w = 0.5 * (1.5 - ax)**2
    else:
        w = 0.0
    return w

@njit(fastmath=True)
def TSC_interpolation(ipatch, i_patch, j_patch, k_patch,
                    x0, y0, z0,
                    x, y, z,
                    pres,nmax,
                    patch_nx, patch_ny, patch_nz,
                    patch_field, level_res,
                    patches_pare, patches_nx, patches_ny, patches_nz,
                    patches_x, patches_y, patches_z,
                    patches_rx, patches_ry, patches_rz, patches_levels,
                    coarse_field, arr_fields):
    """
    Performs TSC interpolation for a point, handling boundaries via parent patch interpolation.
    
    Args:
        ipatch: current patch index
        i_patch, j_patch, k_patch: cell indices in current patch
        x0, y0, z0: center coordinates of the cell (i_patch, j_patch, k_patch)
        x, y, z: target coordinates for interpolation
        pres: current patch cell size
        nmax: base level grid size
        patch_nx, patch_ny, patch_nz: current patch dimensions
        patch_field: field data for current patch
        level_res: resolution (cell size) for each level
        patches_pare: parent patch indices
        patches_nx, patches_ny, patches_nz: dimensions of all patches
        patches_x, patches_y, patches_z: cell positions in parent coordinates
        patches_rx, patches_ry, patches_rz: physical positions (center of first cell)
        patches_levels: refinement level of each patch
        coarse_field: level 0 field data
        arr_fields: list of all patch fields
        
    Returns:
        Interpolated value at (x, y, z)
        
    Author: Òscar Monllor
    """ 

    # 3x3x3 stencil for TSC interpolation
    tsc_array = np.zeros((3, 3, 3), dtype=np.float32)
    l0 = patches_levels[ipatch]

    # If the 3x3x3 stencil is entirely inside the l patch, just copy values
    if (i_patch < patch_nx-1 and i_patch > 0 and
        j_patch < patch_ny-1 and j_patch > 0 and
        k_patch < patch_nz-1 and k_patch > 0):

        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    tsc_array[di+1, dj+1, dk+1] = patch_field[i_patch + di, j_patch + dj, k_patch + dk]

    # If at boundary, some stencil cells from l level need interpolation from parent patches (l-1, l-2, ...)
    else:
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    ii = i_patch + di
                    jj = j_patch + dj
                    kk = k_patch + dk

                    # If inside current patch, use direct value
                    if (ii < patch_nx and ii >= 0 and
                        jj < patch_ny and jj >= 0 and
                        kk < patch_nz and kk >= 0):
                        tsc_array[di+1, dj+1, dk+1] = patch_field[ii, jj, kk]

                    # Otherwise, need to get value from parent (granparent or ancestor)
                    else:
                        ipatch2 = ipatch
                        ii2 = ii
                        jj2 = jj
                        kk2 = kk

                        proper_bounds = False
                        while not proper_bounds:
                            ipare = patches_pare[ipatch2]
                            pare_l = patches_levels[ipare]
                            
                            if pare_l > 0:
                                pare_nx = patches_nx[ipare]
                                pare_ny = patches_ny[ipare]
                                pare_nz = patches_nz[ipare]
                                pare_field = arr_fields[ipare]
                            else:
                                # Level 0 - use periodic boundaries
                                pare_nx = nmax
                                pare_ny = nmax
                                pare_nz = nmax
                                pare_field = coarse_field

                            # Convert child cell indices to parent cell indices
                            # patches_x/y/z are the starting indices in parent coordinates
                            imin_pare = patches_x[ipatch2]
                            jmin_pare = patches_y[ipatch2]
                            kmin_pare = patches_z[ipatch2]

                            # Map child cell to parent cell (refinement factor = 2)
                            ii_pare = imin_pare + ii2 // 2
                            jj_pare = jmin_pare + jj2 // 2
                            kk_pare = kmin_pare + kk2 // 2
                            
                            # Check if we have proper stencil bounds in parent (need ±1 cells)
                            if (ii_pare < pare_nx - 1 and ii_pare > 0 and
                                jj_pare < pare_ny - 1 and jj_pare > 0 and 
                                kk_pare < pare_nz - 1 and kk_pare > 0):
                                proper_bounds = True

                            # Exit condition: reached level 0 or found proper bounds
                            if ipare == 0 or proper_bounds:
                                break
                            else:
                                # Go up another level
                                ipatch2 = ipare
                                ii2 = ii_pare
                                jj2 = jj_pare
                                kk2 = kk_pare

                        # If even level 0 has no proper bounds, use 0th order (nearest neighbor)
                        if not proper_bounds:
                            # Apply periodic boundary conditions for level 0
                            if pare_l == 0:
                                ii_pare = ii_pare % nmax
                                jj_pare = jj_pare % nmax
                                kk_pare = kk_pare % nmax
                            tsc_array[di+1, dj+1, dk+1] = pare_field[ii_pare, jj_pare, kk_pare]
                        
                        #pare patch properly inside bounds -> interpolate from pare values
                        else:
                            # Parent cell size
                            hx = level_res[pare_l]

                            # Center coordinates of parent cell containing the child cell
                            x0_pare = patches_rx[ipare] + (ii_pare - 0.5) * hx
                            y0_pare = patches_ry[ipare] + (jj_pare - 0.5) * hx
                            z0_pare = patches_rz[ipare] + (kk_pare - 0.5) * hx

                            # Child cell center to interpolate (outside current patch)
                            x_interp = x0 + di * pres
                            y_interp = y0 + dj * pres
                            z_interp = z0 + dk * pres

                            # TSC interpolation using parent values
                            value = 0.0
                            for di2 in range(-1, 2):
                                x1 = x0_pare + di2 * hx
                                wx = TSC_kernel(x_interp - x1, hx)
                                for dj2 in range(-1, 2):
                                    y1 = y0_pare + dj2 * hx
                                    wy = TSC_kernel(y_interp - y1, hx)
                                    for dk2 in range(-1, 2):
                                        z1 = z0_pare + dk2 * hx
                                        wz = TSC_kernel(z_interp - z1, hx)

                                        value += pare_field[ii_pare + di2, jj_pare + dj2, kk_pare + dk2] * wx * wy * wz

                            tsc_array[di+1, dj+1, dk+1] = value

    # Final TSC interpolation using the filled 3x3x3 stencil
    value = 0.0
    for di in range(-1, 2):
        x1 = x0 + di * pres
        wx = TSC_kernel(x - x1, pres)
        for dj in range(-1, 2):
            y1 = y0 + dj * pres
            wy = TSC_kernel(y - y1, pres)
            for dk in range(-1, 2):
                z1 = z0 + dk * pres
                wz = TSC_kernel(z - z1, pres)

                value += tsc_array[di+1, dj+1, dk+1] * wx * wy * wz

    return value


def fill_ghost_buffer(ipatch, buffered_patch, nghost,
                    patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz,
                    patchpare, levels, level_res,
                    field, nmax):
    """
    Fills ghost cells of a single patch using TSC interpolation from parent patches.
    
    Args:
        ipatch: current patch index
        buffered_patch: array with ghost cells (interior already filled)
        nghost: number of ghost cells on each side
        patchnx, patchny, patchnz: dimensions of all patches
        patchx, patchy, patchz: cell indices in parent
        patchrx, patchry, patchrz: physical positions of patches
        patchpare: parent patch indices
        levels: refinement levels
        level_res: cell sizes for each level
        field: list of all patch fields
        nmax: base level grid size
        
    Returns:
        buffered_patch: buffered_patch with ghost cells filled
        
    Author: Marco Molina
    """
    
    nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
    pres = level_res[ipatch]
    
    # Patch origin (lower-left-front corner, not cell center)
    # patchrx is the center of the first cell at parent resolution
    patch_x0 = patchrx[ipatch] - pres / 2
    patch_y0 = patchry[ipatch] - pres / 2
    patch_z0 = patchrz[ipatch] - pres / 2
    
    # Process all cells including ghost cells
    for i in prange(nx + 2*nghost):
        for j in range(ny + 2*nghost):
            for k in range(nz + 2*nghost):
                
                # Skip interior cells (already filled)
                if (i >= nghost and i < nx + nghost and
                    j >= nghost and j < ny + nghost and
                    k >= nghost and k < nz + nghost):
                    continue
                
                # Local coordinates (can be negative for ghost cells)
                i_local = i - nghost
                j_local = j - nghost
                k_local = k - nghost
                
                # Physical coordinates of this ghost cell center
                x_ghost = patch_x0 + (i_local + 0.5) * pres
                y_ghost = patch_y0 + (j_local + 0.5) * pres
                z_ghost = patch_z0 + (k_local + 0.5) * pres
                
                # Find which patch cell this ghost cell "belongs to" for indexing
                i_patch = i_local
                j_patch = j_local
                k_patch = k_local
                
                # Cell center coordinates (for reference in TSC_interpolation)
                x0 = patch_x0 + (i_patch + 0.5) * pres
                y0 = patch_y0 + (j_patch + 0.5) * pres
                z0 = patch_z0 + (k_patch + 0.5) * pres
                
                # Use TSC interpolation
                buffered_patch[i, j, k] = TSC_interpolation(
                    ipatch, i_patch, j_patch, k_patch,
                    x0, y0, z0,
                    x_ghost, y_ghost, z_ghost,
                    pres, nmax,
                    nx, ny, nz,
                    field[ipatch], level_res,
                    patchpare, patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz, levels,
                    field[0], field
                )
    
    return buffered_patch


def add_ghost_buffer(field, npatch,
                    patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz, patchpare,
                    size, nmax, nghost=1, kept_patches=None):
    """
    Adds ghost buffer cells to all patches in an AMR field using TSC interpolation.
    
    Args:
        field: list of numpy arrays, each containing a patch of the AMR field
        npatch: number of patches per level (from read_grids)
        patchnx, patchny, patchnz: dimensions of each patch
        patchx, patchy, patchz: cell indices in parent (patchx, patchy, patchz from read_grids)
        patchrx, patchry, patchrz: physical positions of patches
        patchpare: parent patch index for each patch (pare from read_grids)
        size: simulation box size
        nmax: number of cells at base level
        nghost: number of ghost cells to add on each side (default: 1)
        kept_patches: boolean array indicating which patches to process
        
    Returns:
        buffered_field: list of numpy arrays with ghost cells added
        
    Author: Marco Molina
    """
    
    levels = tools.create_vector_levels(npatch)
    level_res = size / nmax / (2.0 ** levels)
    
    if kept_patches is None:
        kept_patches = np.ones(len(field), dtype=bool)
    
    buffered_field = []
    
    # Convert to numba List for njit compatibility
    # Build a homogeneous typed List: replace non-array entries with zero arrays
    field_list = List()
    # determine fallback dtype from the first real array
    fallback_dtype = None
    for f in field:
        if isinstance(f, np.ndarray):
            fallback_dtype = f.dtype
            break
    if fallback_dtype is None:
        fallback_dtype = np.float32

    for idx in range(len(field)):
        f = field[idx]
        if isinstance(f, np.ndarray):
            field_list.append(f)
        else:
            # create placeholder array with the expected patch shape so numba types stay consistent
            nx_i = patchnx[idx]
            ny_i = patchny[idx]
            nz_i = patchnz[idx]
            placeholder = np.zeros((nx_i, ny_i, nz_i), dtype=fallback_dtype, order='F')
            field_list.append(placeholder)
    
    for ipatch in range(len(field)):
        if not kept_patches[ipatch]:
            # keep same convention as readers: use scalar 0 for patches outside region
            buffered_field.append(0)
            continue
            
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Create patch with ghost cells
        buffered_patch = np.zeros((nx + 2*nghost, ny + 2*nghost, nz + 2*nghost), 
                                dtype=field[ipatch].dtype, order='F')
        
        # Copy interior cells first (to avoid race conditions)
        buffered_patch[nghost:nx+nghost, nghost:ny+nghost, nghost:nz+nghost] = field[ipatch][:,:,:]
        
        # Fill ghost cells using TSC interpolation
        buffered_patch = fill_ghost_buffer(
            ipatch, buffered_patch, nghost,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz,
            patchpare, levels, level_res,
            field_list, nmax
        )
        
        buffered_field.append(buffered_patch)
    
    return buffered_field


def ghost_buffer_buster(buffered_field, patchnx, patchny, patchnz, nghost=1, kept_patches=None):
    """
    Removes ghost buffer cells from all patches, returning only the interior data.
    
    Args:
        buffered_field: list of numpy arrays with ghost buffer cells
        patchnx, patchny, patchnz: dimensions of each patch (original, without ghost buffer cells)
        nghost: number of ghost cells on each side (must match that used in add_ghost_buffer)
        kept_patches: boolean array indicating which patches were processed
        
    Returns:
        field: list of numpy arrays without ghost buffer cells in original dimensions
        
    Author: Marco Molina
    """
    
    if kept_patches is None:
        kept_patches = np.ones(len(buffered_field), dtype=bool)
    
    field = []
    
    for ipatch in range(len(buffered_field)):
        if not kept_patches[ipatch] or isinstance(buffered_field[ipatch], (int, float)):
            # return scalar 0 for outside-region patches to match readers' convention
            field.append(0)
            continue
        
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Extract interior cells only (removing ghost cells)
        interior_patch = buffered_field[ipatch][nghost:nx+nghost, 
                                                nghost:ny+nghost, 
                                                nghost:nz+nghost].copy()
        
        field.append(interior_patch)
    
    return field

def inplace_ghost_buffer_buster(buffered_field, patchnx, patchny, patchnz, nghost=1, kept_patches=None):
    """
    Removes ghost buffer cells in-place, modifying the list but creating new arrays.
    More memory efficient than ghost_buffer_buster for large fields.
    
    Args:
        buffered_field: list of numpy arrays with ghost buffer cells. Will be modified in-place.
        patchnx, patchny, patchnz: dimensions of each patch (original, without ghost buffer cells)
        nghost: number of ghost buffer cells on each side (must match that used in add_ghost_buffer)
        kept_patches: boolean array indicating which patches were processed
        
    Returns:
        None (modifies buffered_field in-place)
        
    Author: Marco Molina
    """
    
    if kept_patches is None:
        kept_patches = np.ones(len(buffered_field), dtype=bool)
    
    for ipatch in range(len(buffered_field)):
        if not kept_patches[ipatch] or isinstance(buffered_field[ipatch], (int, float)):
            continue
        
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Extract interior cells and replace in list
        buffered_field[ipatch] = buffered_field[ipatch][nghost:nx+nghost, 
                                                        nghost:ny+nghost, 
                                                        nghost:nz+nghost].copy()