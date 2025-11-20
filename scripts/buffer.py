"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

buffer module
Provides functions to create and manage data buffers for their proper AMR differential treatment by the diff.py module.

Created by Marco Molina Pradillo and Òscar Monllor.
"""

import numpy as np
from numba import njit, prange
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
def SPH_kernel(r, h):
    """
    3D cubic-spline SPH kernel (Monaghan). Support radius = 2*h.
    Returns W(|r|,h) (not multiplied by mass/density).
    Normalisation constant uses 1/(pi*h^3) for 3D.
    
    Args:
        r: distance from cell center
        h: smoothing length (cell size)
        
    Returns:
        Weight for SPH interpolation (between 0 and ~0.318/h^3)
    
    Author: Marco Molina
    """
    q = r / h
    sigma = 1.0 / (np.pi * h**3)   # 3D normalisation
    if q < 0.0:
        q = -q
    if q <= 1.0:
        return sigma * (1.0 - 1.5*q*q + 0.75*q*q*q)
    elif q <= 2.0:
        return sigma * 0.25 * (2.0 - q)**3
    else:
        return 0.0

@njit(fastmath=True)
def TSC_interpolation(ipatch, i_patch, j_patch, k_patch,
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
    l0 = patches_levels[ipatch]

    ipatch2 = ipatch
    ii2 = i_patch
    jj2 = j_patch
    kk2 = k_patch

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
    

    # Parent cell size
    hx = level_res[pare_l]

    # Center coordinates of parent cell containing the child cell
    x0_pare = patches_rx[ipare] + (ii_pare - 0.5) * hx
    y0_pare = patches_ry[ipare] + (jj_pare - 0.5) * hx
    z0_pare = patches_rz[ipare] + (kk_pare - 0.5) * hx

    # TSC interpolation using parent values
    value = 0.
    for di2 in range(-1, 2):
        x1 = x0_pare + di2 * hx
        wx = TSC_kernel(x - x1, hx)
        for dj2 in range(-1, 2):
            y1 = y0_pare + dj2 * hx
            wy = TSC_kernel(y - y1, hx)
            for dk2 in range(-1, 2):
                z1 = z0_pare + dk2 * hx
                wz = TSC_kernel(z - z1, hx)

                value += pare_field[ii_pare + di2, jj_pare + dj2, kk_pare + dk2] * wx * wy * wz

    return value

@njit(fastmath=True)
def SPH_interpolation(ipatch, i_patch, j_patch, k_patch,
                        x, y, z,
                        pres, nmax,
                        patch_nx, patch_ny, patch_nz,
                        patch_field, level_res,
                        patches_pare, patches_nx, patches_ny, patches_nz,
                        patches_x, patches_y, patches_z,
                        patches_rx, patches_ry, patches_rz, patches_levels,
                        coarse_field, arr_fields):
    """
    SPH-style interpolation at point (x,y,z) using parent/ancestor cells.
    Similar parent-walk as TSC_interpolation, but uses cubic-spline SPH kernel
    with support 2*h. We gather a small parent stencil (±2) and compute
    weighted average (weights normalized).
    
    Args:
        ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        x, y, z: coordinates of the interpolation point
        pres: pressure (unused here, but kept for consistency)
        nmax: maximum number of cells in level 0
        patch_nx, patch_ny, patch_nz: dimensions of the current patch
        patch_field: field values in the current patch
        level_res: resolution at each refinement level
        patches_pare, patches_nx, patches_ny, patches_nz: parent patch info
        patches_x, patches_y, patches_z: patch starting indices
        patches_rx, patches_ry, patches_rz: patch reference coordinates
        patches_levels: refinement levels of patches
        coarse_field: level 0 field data
        arr_fields: list of all patch fields
        
    Returns:
        Interpolated value at (x, y, z)
        
    Author: Marco Molina
    """
    # Find a parent patch that provides a sufficiently large stencil (like TSC)
    l0 = patches_levels[ipatch]
    ipatch2 = ipatch
    ii2 = i_patch
    jj2 = j_patch
    kk2 = k_patch

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
            pare_nx = nmax
            pare_ny = nmax
            pare_nz = nmax
            pare_field = coarse_field

        imin_pare = patches_x[ipatch2]
        jmin_pare = patches_y[ipatch2]
        kmin_pare = patches_z[ipatch2]

        ii_pare = imin_pare + ii2 // 2
        jj_pare = jmin_pare + jj2 // 2
        kk_pare = kmin_pare + kk2 // 2

        # need margin of 2 cells for cubic SPH support (support 2*h)
        if (ii_pare < pare_nx - 2 and ii_pare > 1 and
            jj_pare < pare_ny - 2 and jj_pare > 1 and
            kk_pare < pare_nz - 2 and kk_pare > 1):
            proper_bounds = True

        if ipare == 0 or proper_bounds:
            break
        else:
            ipatch2 = ipare
            ii2 = ii_pare
            jj2 = jj_pare
            kk2 = kk_pare

    # parent cell size
    hx = level_res[pare_l]

    # center of parent cell
    x0_pare = patches_rx[ipare] + (ii_pare - 0.5) * hx
    y0_pare = patches_ry[ipare] + (jj_pare - 0.5) * hx
    z0_pare = patches_rz[ipare] + (kk_pare - 0.5) * hx

    # accumulate weighted sum and normalisation
    wsum = 0.0
    value = 0.0

    # support radius = 2*h -> need indices di in [-2,2]
    for di2 in range(-2, 3):
        for dj2 in range(-2, 3):
            for dk2 in range(-2, 3):
                ii = ii_pare + di2
                jj = jj_pare + dj2
                kk = kk_pare + dk2
                # boundary safety
                if ii < 0 or ii >= pare_nx or jj < 0 or jj >= pare_ny or kk < 0 or kk >= pare_nz:
                    continue
                xc = patches_rx[ipare] + (ii - 0.5) * hx
                yc = patches_ry[ipare] + (jj - 0.5) * hx
                zc = patches_rz[ipare] + (kk - 0.5) * hx
                dx = x - xc
                dy = y - yc
                dz = z - zc
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                w = SPH_kernel(r, hx)
                if w > 0.0:
                    value += pare_field[ii, jj, kk] * w
                    wsum += w

    if wsum > 0.0:
        return value / wsum
    else:
        # fallback to direct parent cell value if no weight (rare)
        return pare_field[ii_pare, jj_pare, kk_pare]

@njit(fastmath=True)
def fill_ghost_buffer(ipatch, buffered_patches, nghost,
                    patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz,
                    patchpare, levels, level_res,
                    fields, nmax, interpol='TSC'):
    """
    Fills ghost cells of a single patch using TSC interpolation from parent patches.
    If the ghost cell lies inside a brother patch (same refinement level), copy the
    value directly from that brother instead of using parent interpolation.

    Args:
        ipatch: current patch index
        buffered_patches: list of list of arrays with ghost cells (interior already filled) for each field
        nghost: number of ghost cells on each side
        patchnx, patchny, patchnz: dimensions of all patches
        patchx, patchy, patchz: cell indices in parent
        patchrx, patchry, patchrz: physical positions of patches
        patchpare: parent patch indices
        levels: refinement levels for each patch
        level_res: cell sizes for each level
        fields: list of lists of all fields patches
        nmax: base level grid size
        interpol: interpolation method ('TSC' or 'SPH')
        
    Returns:
        buffered_patches: list of list of arrays with ghost cells filled
        
    Author: Marco Molina
    """
    
    nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
    pres = level_res[ipatch]
    my_level = levels[ipatch]
    
    # Upper and lower indices for the current level patches
    low_patch_index = (levels < my_level).sum()
    high_patch_index = (levels <= my_level).sum()
    
    # Patch origin (lower-left-front corner, not cell center)
    # patchrx is the center of the first cell at parent resolution
    patch_x0 = patchrx[ipatch] - pres / 2
    patch_y0 = patchry[ipatch] - pres / 2
    patch_z0 = patchrz[ipatch] - pres / 2
    
    # Per-patch discovered-brothers cache (append-only, small)
    ## Tune cache_size to expected number of nearby brothers (power of two good)
    cache_size = 32
    bro_list = np.full(cache_size, -1, np.int64)      # stored brother indices
    bro_bx0 = np.zeros(cache_size, dtype=np.float64)
    bro_by0 = np.zeros(cache_size, dtype=np.float64)
    bro_bz0 = np.zeros(cache_size, dtype=np.float64)
    bro_bnx = np.zeros(cache_size, dtype=np.int64)
    bro_bny = np.zeros(cache_size, dtype=np.int64)
    bro_bnz = np.zeros(cache_size, dtype=np.int64)
    bro_pres = np.zeros(cache_size, dtype=np.float64)
    bro_count = 0
    
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
                x_ghost = patch_x0 + i_local * pres
                y_ghost = patch_y0 + j_local * pres
                z_ghost = patch_z0 + k_local * pres
                
                # First we try to find a brother patch at the same level that contains this point
                brother_found = False
                
                ## Try discovered brothers first (cheap)
                for bi in range(bro_count):
                    ip = bro_list[bi]
                    bx0 = bro_bx0[bi]; by0 = bro_by0[bi]; bz0 = bro_bz0[bi]
                    bnx = bro_bnx[bi]; bny = bro_bny[bi]; bnz = bro_bnz[bi]
                    pres_b = bro_pres[bi]
                    
                    # Check inclusion
                    if (x_ghost >= bx0 and x_ghost < bx0 + bnx * pres_b and
                        y_ghost >= by0 and y_ghost < by0 + bny * pres_b and
                        z_ghost >= bz0 and z_ghost < bz0 + bnz * pres_b):
                        
                        # Compute integer indices inside brother patch
                        ib = int((x_ghost - bx0) / pres_b)
                        jb = int((y_ghost - by0) / pres_b)
                        kb = int((z_ghost - bz0) / pres_b)
                        
                        # For some edge cases
                        if ib < 0: ib = 0
                        if ib > bnx - 1: ib = bnx - 1
                        if jb < 0: jb = 0
                        if jb > bny - 1: jb = bny - 1
                        if kb < 0: kb = 0
                        if kb > bnz - 1: kb = bnz - 1
                        
                        # Copy value from brother patch (nearest-cell copy)
                        for f in range(len(fields)):
                            buffered_patches[f][i, j, k] = fields[f][ip][ib, jb, kb]
                        
                        brother_found = True
                        break

                if brother_found:
                    continue
                
                ## Full scan fallback (when not covered by discovered brothers)
                for ip in range(low_patch_index, high_patch_index):
                    if ip == ipatch:
                        continue
                    
                    # Brother patch cell size and origin
                    bnx, bny, bnz = patchnx[ip], patchny[ip], patchnz[ip]
                    pres_b = level_res[ip]
                    bx0 = patchrx[ip] - pres_b / 2
                    by0 = patchry[ip] - pres_b / 2
                    bz0 = patchrz[ip] - pres_b / 2

                    # Check inclusion
                    if (x_ghost >= bx0 and x_ghost < bx0 + bnx * pres_b and
                        y_ghost >= by0 and y_ghost < by0 + bny * pres_b and
                        z_ghost >= bz0 and z_ghost < bz0 + bnz * pres_b):
                        
                        # Compute integer indices inside brother patch
                        ib = int((x_ghost - bx0) / pres_b)
                        jb = int((y_ghost - by0) / pres_b)
                        kb = int((z_ghost - bz0) / pres_b)
                        
                        # For some edge cases
                        if ib < 0: ib = 0
                        if ib > bnx - 1: ib = bnx - 1
                        if jb < 0: jb = 0
                        if jb > bny - 1: jb = bny - 1
                        if kb < 0: kb = 0
                        if kb > bnz - 1: kb = bnz - 1

                        # Copy value from brother patch (nearest-cell copy)
                        for f in range(len(fields)):
                            buffered_patches[f][i, j, k] = fields[f][ip][ib, jb, kb]

                        # Record discovered brother if cache not full and not already present
                        already = False
                        for bi in range(bro_count):
                            if bro_list[bi] == ip:
                                already = True
                                break
                        if (not already) and (bro_count < cache_size):
                            idx = bro_count
                            bro_list[idx] = ip
                            bro_bx0[idx] = bx0
                            bro_by0[idx] = by0
                            bro_bz0[idx] = bz0
                            bro_bnx[idx] = bnx
                            bro_bny[idx] = bny
                            bro_bnz[idx] = bnz
                            bro_pres[idx] = pres_b
                            bro_count += 1
                            
                        brother_found = True
                        break

                if brother_found:
                    continue
                
                ## If not borther, use interpolation from parents (choose method)
                for f in range(len(fields)):
                    if interpol == 'TSC':
                        buffered_patches[f][i, j, k] = TSC_interpolation(
                            ipatch, i_local, j_local, k_local,
                            x_ghost, y_ghost, z_ghost,
                            pres, nmax,
                            nx, ny, nz,
                            fields[f][ipatch], level_res,
                            patchpare, patchnx, patchny, patchnz,
                            patchx, patchy, patchz,
                            patchrx, patchry, patchrz, levels,
                            fields[f][0], fields[f])
                    else:
                        buffered_patches[f][i, j, k] = SPH_interpolation(
                            ipatch, i_local, j_local, k_local,
                            x_ghost, y_ghost, z_ghost,
                            pres, nmax,
                            nx, ny, nz,
                            fields[f][ipatch], level_res,
                            patchpare, patchnx, patchny, patchnz,
                            patchx, patchy, patchz,
                            patchrx, patchry, patchrz, levels,
                            fields[f][0], fields[f])
    
    return buffered_patches


def add_ghost_buffer(fields, npatch,
                    patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz, patchpare,
                    size, nmax, nghost=1, interpol='TSC', bitformat=np.float32, kept_patches=None):
    """
    Adds ghost buffer cells to all patches in an AMR field using TSC interpolation.
    Accepts `fields` as a list of fields (each field is a list of per-patch arrays).
    Converts each field to a numba.typed.List of arrays (placeholders for missing patches)
    so njit functions receive homogeneous typed lists.
    
    Args:
        fields: list of list of numpy arrays, each containing fields with each array being a patch of the AMR field
        npatch: number of patches per level (from read_grids)
        patchnx, patchny, patchnz: dimensions of each patch
        patchx, patchy, patchz: cell indices in parent (patchx, patchy, patchz from read_grids)
        patchrx, patchry, patchrz: physical positions of patches
        patchpare: parent patch index for each patch (pare from read_grids)
        size: simulation box size
        nmax: number of cells at base level
        nghost: number of ghost cells to add on each side (default: 1)
        interpol: interpolation method ('TSC' or 'SPH')
        bitformat: data type for the fields (default is np.float32)
        kept_patches: boolean array indicating which patches to process
        
    Returns:
        buffered_fields: list of list of numpy arrays with ghost cells added
        
    Author: Marco Molina
    """
    
    if interpol not in ['TSC', 'SPH']:
        raise ValueError("Interpolation method must be 'TSC' or 'SPH'")
    
    levels = tools.create_vector_levels(npatch)
    level_res = size / nmax / (2.0 ** levels)
    
    if kept_patches is None:
        kept_patches = np.ones(len(fields[0]), dtype=bool)
    
    # Convert each field (list of per-patch items) into a numba.typed.List of arrays
    
    fields_numba = []
    for fset in fields:
        L = tools.python_to_numba_list(fset, patchnx, patchny, patchnz, fallback_dtype=bitformat, order='F')
        fields_numba.append(L)

    buffered_fields = [[] for _ in range(len(fields))]
        
    for ipatch in range(len(fields[0])):
        if not kept_patches[ipatch]:
            # keep same convention as readers: use scalar 0 for patches outside region
            [buffered_fields[f].append(0) for f in range(len(fields))]
            continue
            
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Create patch with ghost cells
        buffered_patches = [np.zeros((nx + 2*nghost, ny + 2*nghost, nz + 2*nghost), 
                                dtype=bitformat, order='F') for _ in range(len(fields))]
            
        for f in range(len(fields)):
            src = fields[f][ipatch]
            if isinstance(src, np.ndarray):
                buffered_patches[f][nghost:nx+nghost, nghost:ny+nghost, nghost:nz+nghost] = src[:,:,:]
            else:
                # source was placeholder scalar 0: leave interior zeros (or could copy from placeholder)
                pass
        
        # Fill ghost cells using TSC interpolation
        buffered_patches = fill_ghost_buffer(
            ipatch, buffered_patches, nghost,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz,
            patchpare, levels, level_res,
            fields_numba, nmax, interpol=interpol
        )
        
        [buffered_fields[f].append(buffered_patches[f]) for f in range(len(fields))]
    
    return buffered_fields


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