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
    '''
    Triangular-Shaped Cloud (TSC) kernel for interpolation.
    
    Args:
        x: distance from cell center
        h: cell size
        
    Returns:
        Weight for TSC interpolation (between 0 and 0.75)
        
    Author: Òscar Monllor
    '''
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
    '''
    3D cubic-spline SPH kernel (Monaghan). Support radius = 2*h.
    Returns W(|r|,h) (not multiplied by mass/density).
    Normalisation constant uses 1/(pi*h^3) for 3D.
    
    Args:
        r: distance from cell center
        h: smoothing length (cell size)
        
    Returns:
        Weight for SPH interpolation (between 0 and ~0.318/h^3)
    
    Author: Marco Molina
    '''
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
def direct_parent_access(ipatch, i_patch, j_patch, k_patch,
                        x, y, z,
                        pres, nmax,
                        patch_nx, patch_ny, patch_nz,
                        patch_field, level_res,
                        patches_pare, patches_nx, patches_ny, patches_nz,
                        patches_x, patches_y, patches_z,
                        patches_rx, patches_ry, patches_rz, patches_levels,
                        coarse_field, arr_fields):
    '''
    Direct parent access: accesses parent cell value directly without interpolation.
    Equivalent to MASCLET-B 2020 Fortran method: goes up hierarchy until finds parent with sufficient bounds,
    then returns parent cell value directly without any interpolation.
    Most conservative and accurate method for avoiding artificial gradients.
    
    Args:
        - ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        - x, y, z: coordinates of the interpolation point
        - pres: pressure (unused here, but kept for consistency)
        - nmax: maximum number of cells in level 0
        - patch_nx, patch_ny, patch_nz: dimensions of the current patch
        - patch_field: field values in the current patch
        - level_res: resolution at each refinement level
        - patches_pare, patches_nx, patches_ny, patches_nz: parent patch info
        - patches_x, patches_y, patches_z: patch starting indices
        - patches_rx, patches_ry, patches_rz: patch reference coordinates
        - patches_levels: refinement levels of patches
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
    
    Indexing notes:
        - Python uses 0-based indexing
        - patches_x, patches_y, patches_z are already converted to 0-based in readers.py (line 238-240)
        - Refinement factor = 2 (child cell i maps to parent cell i//2)
        - No +1 offset needed because already 0-based
        
    Returns:
        - Value at the parent cell directly above the target cell
    
    Author: Marco Molina
    '''
    l0 = patches_levels[ipatch]
    ipatch2 = ipatch
    ii2 = i_patch
    jj2 = j_patch
    kk2 = k_patch

    proper_bounds = False
    while not proper_bounds:
        ipare = patches_pare[ipatch2]
        pare_l = patches_levels[ipare]
        
        # If parent is also refined, go up one more level
        if pare_l > 0:
            ii2 = ii2 // 2
            jj2 = jj2 // 2
            kk2 = kk2 // 2
            ipatch2 = ipare
        else:
            # Found a non-refined parent
            break

        # Get parent patch dimensions
        pare_nx = patches_nx[ipatch2]
        pare_ny = patches_ny[ipatch2]
        pare_nz = patches_nz[ipatch2]

        # Parent patch starting indices (0-based, already from readers.py)
        imin_pare = patches_x[ipatch2]
        jmin_pare = patches_y[ipatch2]
        kmin_pare = patches_z[ipatch2]

        # Map child cell to parent coordinates
        ii_pare = imin_pare + ii2 // 2
        jj_pare = jmin_pare + jj2 // 2
        kk_pare = kmin_pare + kk2 // 2
        
        # Check if within parent bounds
        if (ii_pare >= 0 and ii_pare < pare_nx and
            jj_pare >= 0 and jj_pare < pare_ny and 
            kk_pare >= 0 and kk_pare < pare_nz):
            proper_bounds = True

        if ipare == 0 or proper_bounds:
            break

    # Get parent patch dimensions (for final access)
    ipare = patches_pare[ipatch2]
    pare_nx = patches_nx[ipare]
    pare_ny = patches_ny[ipare]
    pare_nz = patches_nz[ipare]

    # Get parent patch starting indices
    imin_pare = patches_x[ipare]
    jmin_pare = patches_y[ipare]
    kmin_pare = patches_z[ipare]

    # Final mapping to parent (0-based)
    ii_pare = imin_pare + ii2 // 2
    jj_pare = jmin_pare + jj2 // 2
    kk_pare = kmin_pare + kk2 // 2
    
    # Clamp to valid range (defensive, should not be needed)
    if ii_pare < 0: ii_pare = 0
    if ii_pare >= pare_nx: ii_pare = pare_nx - 1
    if jj_pare < 0: jj_pare = 0
    if jj_pare >= pare_ny: jj_pare = pare_ny - 1
    if kk_pare < 0: kk_pare = 0
    if kk_pare >= pare_nz: kk_pare = pare_nz - 1
    
    # Access parent field directly (no interpolation)
    pare_field = arr_fields[ipare]
    return float(pare_field[ii_pare, jj_pare, kk_pare])


@njit(fastmath=True)
def nearest_interpolation(ipatch, i_patch, j_patch, k_patch,
                        x, y, z,
                        pres, nmax,
                        patch_nx, patch_ny, patch_nz,
                        patch_field, level_res,
                        patches_pare, patches_nx, patches_ny, patches_nz,
                        patches_x, patches_y, patches_z,
                        patches_rx, patches_ry, patches_rz, patches_levels,
                        coarse_field, arr_fields):
    '''
    Nearest-neighbor with parent frontier validation.
    Copies parent value directly, but ensures parent cell has enough neighbors (not frontier).
    If parent is frontier, recursively climbs AMR hierarchy until finding a valid non-frontier parent.
    
    This implements the PARENT method: skips interpolation and directly uses parent divergence
    values, but only for parent cells that aren't themselves too close to boundaries.
    
    Frontier detection:
    - For 3-point stencil: cell within 1 of boundary is frontier (needs interior neighbors)
    - For 5-point stencil: cell within 2 of boundary is frontier
    - Stencil radius hardcoded to 1 (works for 3-point stencil)
    
    Args:
        - ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        - x, y, z: coordinates of the interpolation point
        - pres: pressure (unused here, but kept for consistency)
        - nmax: maximum number of cells in level 0
        - patch_nx, patch_ny, patch_nz: dimensions of the current patch
        - patch_field: field values in the current patch
        - level_res: resolution at each refinement level
        - patches_pare, patches_nx, patches_ny, patches_nz: parent patch info
        - patches_x, patches_y, patches_z: patch starting indices
        - patches_rx, patches_ry, patches_rz: patch reference coordinates
        - patches_levels: refinement levels of patches
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
    
    Indexing notes:
        - Python uses 0-based indexing  
        - patches_x/y/z are 0-based (converted in readers.py)
        - Refinement factor = 2
        - Stencil radius: 1 for 3-point stencil
        
    Returns:
        - Value from valid (non-frontier) parent cell
    
    Author: Marco Molina
    '''
    # Stencil radius - hardcoded for 3-point stencil (need 1 neighbor on each side)
    stencil_radius = 1
    
    l0 = patches_levels[ipatch]
    ipatch_curr = ipatch
    ii_curr = i_patch
    jj_curr = j_patch
    kk_curr = k_patch
    
    max_iterations = 20  # Safety limit to prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        ipare = patches_pare[ipatch_curr]
        pare_l = patches_levels[ipare]
        
        # Get parent patch dimensions and field
        if pare_l > 0:
            pare_nx = patches_nx[ipare]
            pare_ny = patches_ny[ipare]
            pare_nz = patches_nz[ipare]
            pare_field = arr_fields[ipare]
        else:
            # Level 0 - use coarse field
            pare_nx = nmax
            pare_ny = nmax
            pare_nz = nmax
            pare_field = coarse_field
        
        # Get parent patch starting indices
        imin_pare = patches_x[ipatch_curr]
        jmin_pare = patches_y[ipatch_curr]
        kmin_pare = patches_z[ipatch_curr]
        
        # Map child cell to parent cell coordinates (refinement factor = 2)
        ii_pare = imin_pare + ii_curr // 2
        jj_pare = jmin_pare + jj_curr // 2
        kk_pare = kmin_pare + kk_curr // 2
        
        # Check if parent cell has enough neighbors (is NOT frontier)
        # For 3-point stencil: need at least 1 neighbor on each side (ii_pare > 0 and ii_pare < pare_nx - 1)
        is_parent_valid = (ii_pare > stencil_radius - 1 and ii_pare < pare_nx - stencil_radius and
                           jj_pare > stencil_radius - 1 and jj_pare < pare_ny - stencil_radius and
                           kk_pare > stencil_radius - 1 and kk_pare < pare_nz - stencil_radius)
        
        if is_parent_valid:
            # Found a valid parent cell with proper bounds - use its value
            return float(pare_field[ii_pare, jj_pare, kk_pare])
        
        # Parent doesn't have proper bounds - try next coarser level
        if ipare == 0:
            # Reached level 0
            # Clamp indices to valid range for level 0
            ii_pare_clamped = max(0, min(ii_pare, pare_nx - 1))
            jj_pare_clamped = max(0, min(jj_pare, pare_ny - 1))
            kk_pare_clamped = max(0, min(kk_pare, pare_nz - 1))
            return float(pare_field[ii_pare_clamped, jj_pare_clamped, kk_pare_clamped])
        
        # Go to next coarser level
        ipatch_curr = ipare
        ii_curr = ii_pare
        jj_curr = jj_pare
        kk_curr = kk_pare
        iteration += 1
    
    # Fallback: return zero (shouldn't reach here in normal operation)
    return 0.0

@njit(fastmath=True)
def linear_interpolation(ipatch, i_patch, j_patch, k_patch,
                        x, y, z,
                        pres, nmax,
                        patch_nx, patch_ny, patch_nz,
                        patch_field, level_res,
                        patches_pare, patches_nx, patches_ny, patches_nz,
                        patches_x, patches_y, patches_z,
                        patches_rx, patches_ry, patches_rz, patches_levels,
                        coarse_field, arr_fields):
    '''
    Linear extrapolation from nearest two parent cells in each direction.
    More conservative than TSC but smoother than nearest-neighbor.
    
    Args:
        - ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        - x, y, z: coordinates of the interpolation point
        - pres: pressure (unused here, but kept for consistency)
        - nmax: maximum number of cells in level 0
        - patch_nx, patch_ny, patch_nz: dimensions of the current patch
        - patch_field: field values in the current patch
        - level_res: resolution at each refinement level
        - patches_pare, patches_nx, patches_ny, patches_nz: parent patch info
        - patches_x, patches_y, patches_z: patch starting indices
        - patches_rx, patches_ry, patches_rz: patch reference coordinates
        - patches_levels: refinement levels of patches
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
    
    Indexing notes:
        - Python 0-based indexing throughout
        - patches_x/y/z are 0-based (readers.py line 238: patchx.append(this_x - 1))
        - patches_rx/ry/rz are physical coordinates (center of FIRST cell in parent coords)
        - Refinement factor = 2, so ii2 // 2 maps child to parent
        - NO +1 offset needed in Python (unlike Fortran which uses 1-based)
        
    Returns:
        - Interpolated value at (x, y, z)
    
    Author: Marco Molina
    '''
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
        
        # For linear, need 2 cells in each direction for interpolation
        if (ii_pare < pare_nx - 1 and ii_pare > 0 and
            jj_pare < pare_ny - 1 and jj_pare > 0 and 
            kk_pare < pare_nz - 1 and kk_pare > 0):
            proper_bounds = True

        if ipare == 0 or proper_bounds:
            break
        else:
            ipatch2 = ipare
            ii2 = ii_pare
            jj2 = jj_pare
            kk2 = kk_pare
    
    hx = level_res[pare_l]
    
    # Center of parent cell
    x0_pare = patches_rx[ipare] + (ii_pare - 0.5) * hx
    y0_pare = patches_ry[ipare] + (jj_pare - 0.5) * hx
    z0_pare = patches_rz[ipare] + (kk_pare - 0.5) * hx

    # Linear interpolation in each direction separately
    # Note: proper_bounds check guarantees safe indices (ii_pare > 0 and ii_pare < pare_nx - 1, etc.)
    # x-direction
    if x < x0_pare:
        v0 = pare_field[ii_pare - 1, jj_pare, kk_pare]
        v1 = pare_field[ii_pare, jj_pare, kk_pare]
        x0 = x0_pare - hx
        x1 = x0_pare
    else:
        v0 = pare_field[ii_pare, jj_pare, kk_pare]
        v1 = pare_field[ii_pare + 1, jj_pare, kk_pare]
        x0 = x0_pare
        x1 = x0_pare + hx
    
    if abs(x1 - x0) > 1e-10:
        val_x = v0 + (v1 - v0) * (x - x0) / (x1 - x0)
    else:
        val_x = v0
    
    # y-direction
    if y < y0_pare:
        v0 = pare_field[ii_pare, jj_pare - 1, kk_pare]
        v1 = pare_field[ii_pare, jj_pare, kk_pare]
        y0 = y0_pare - hx
        y1 = y0_pare
    else:
        v0 = pare_field[ii_pare, jj_pare, kk_pare]
        v1 = pare_field[ii_pare, jj_pare + 1, kk_pare]
        y0 = y0_pare
        y1 = y0_pare + hx
    
    if abs(y1 - y0) > 1e-10:
        val_y = v0 + (v1 - v0) * (y - y0) / (y1 - y0)
    else:
        val_y = v0
    
    # z-direction
    if z < z0_pare:
        v0 = pare_field[ii_pare, jj_pare, kk_pare - 1]
        v1 = pare_field[ii_pare, jj_pare, kk_pare]
        z0 = z0_pare - hx
        z1 = z0_pare
    else:
        v0 = pare_field[ii_pare, jj_pare, kk_pare]
        v1 = pare_field[ii_pare, jj_pare, kk_pare + 1]
        z0 = z0_pare
        z1 = z0_pare + hx
    
    if abs(z1 - z0) > 1e-10:
        val_z = v0 + (v1 - v0) * (z - z0) / (z1 - z0)
    else:
        val_z = v0
    
    # Average the three directional interpolations
    return (val_x + val_y + val_z) / 3.0

@njit(fastmath=True)
def trilinear_interpolation(ipatch, i_patch, j_patch, k_patch,
                            x, y, z,
                            pres, nmax,
                            patch_nx, patch_ny, patch_nz,
                            patch_field, level_res,
                            patches_pare, patches_nx, patches_ny, patches_nz,
                            patches_x, patches_y, patches_z,
                            patches_rx, patches_ry, patches_rz, patches_levels,
                            coarse_field, arr_fields):
    '''
    Trilinear interpolation using the 8 vertices of a parent cell cube.
    Standard trilinear interpolation: finds the parent cell containing the point,
    then uses the 8 corner values to compute weighted average based on relative position.
    
    The interpolation formula is:
        F(x,y,z) = F000*(1-dx)*(1-dy)*(1-dz) + F100*dx*(1-dy)*(1-dz) +
                   F010*dy*(1-dx)*(1-dz)     + F001*dz*(1-dx)*(1-dy) +
                   F110*dx*dy*(1-dz)         + F101*dx*dz*(1-dy) +
                   F011*dy*dz*(1-dx)         + F111*dx*dy*dz
    
    where dx, dy, dz are normalized distances [0,1] from the lower corner of the cell.
    
    Args:
        - ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        - x, y, z: coordinates of the interpolation point
        - pres: current patch cell size (unused but kept for consistency)
        - nmax: base level grid size
        - patch_nx, patch_ny, patch_nz: current patch dimensions
        - patch_field: field data for current patch
        - level_res: resolution (cell size) for each level
        - patches_pare: parent patch indices
        - patches_nx, patches_ny, patches_nz: dimensions of all patches
        - patches_x, patches_y, patches_z: cell positions in parent coordinates (0-based)
        - patches_rx, patches_ry, patches_rz: physical positions (center of first cell)
        - patches_levels: refinement level of each patch
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
        
    Indexing notes:
        - All indices are 0-based (Python convention)
        - patches_x/y/z are 0-based from readers.py
        - Refinement factor = 2
        - Trilinear requires access to 8 corners, so parent cell needs +1 in each direction
        
    Returns:
        - Interpolated value at (x, y, z)
        
    Author: Marco Molina
    '''
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
        imin_pare = patches_x[ipatch2]
        jmin_pare = patches_y[ipatch2]
        kmin_pare = patches_z[ipatch2]

        # Map child cell to parent cell (refinement factor = 2)
        ii_pare = imin_pare + ii2 // 2
        jj_pare = jmin_pare + jj2 // 2
        kk_pare = kmin_pare + kk2 // 2
        
        # Check if we have proper bounds for trilinear (need cell + 1 in each direction)
        if (ii_pare < pare_nx - 1 and ii_pare >= 0 and
            jj_pare < pare_ny - 1 and jj_pare >= 0 and 
            kk_pare < pare_nz - 1 and kk_pare >= 0):
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

    # Center coordinates of parent cell (following TSC_interpolation pattern)
    # patches_rx/ry/rz provide reference coordinates (see TSC implementation)
    x0_pare = patches_rx[ipare] + (ii_pare - 0.5) * hx
    y0_pare = patches_ry[ipare] + (jj_pare - 0.5) * hx
    z0_pare = patches_rz[ipare] + (kk_pare - 0.5) * hx
    
    # For trilinear interpolation, we need normalized distances within the cell
    # The cell extends from (ii_pare-0.5)*hx to (ii_pare+0.5)*hx around the center
    # Normalized position within the cell [0,1]:
    #   0 corresponds to lower face (ii_pare cell center - 0.5*hx)
    #   1 corresponds to upper face (ii_pare+1 cell center - 0.5*hx = ii_pare cell center + 0.5*hx)
    
    # However, for trilinear, we interpolate between vertices at cell centers
    # Cell center ii_pare is at x0_pare
    # Cell center ii_pare+1 is at x0_pare + hx
    # So normalized distance from ii_pare center to ii_pare+1 center:
    dx = (x - x0_pare) / hx
    dy = (y - y0_pare) / hx
    dz = (z - z0_pare) / hx
    
    # Clamp to [0, 1] for safety (prevents extrapolation)
    if dx < 0.0: dx = 0.0
    if dx > 1.0: dx = 1.0
    if dy < 0.0: dy = 0.0
    if dy > 1.0: dy = 1.0
    if dz < 0.0: dz = 0.0
    if dz > 1.0: dz = 1.0

    # Get the 8 vertex values (cell centers forming the cube)
    # F000 = value at (ii_pare, jj_pare, kk_pare)
    # F100 = value at (ii_pare+1, jj_pare, kk_pare), etc.
    F000 = pare_field[ii_pare,     jj_pare,     kk_pare    ]
    F100 = pare_field[ii_pare + 1, jj_pare,     kk_pare    ]
    F010 = pare_field[ii_pare,     jj_pare + 1, kk_pare    ]
    F001 = pare_field[ii_pare,     jj_pare,     kk_pare + 1]
    F110 = pare_field[ii_pare + 1, jj_pare + 1, kk_pare    ]
    F101 = pare_field[ii_pare + 1, jj_pare,     kk_pare + 1]
    F011 = pare_field[ii_pare,     jj_pare + 1, kk_pare + 1]
    F111 = pare_field[ii_pare + 1, jj_pare + 1, kk_pare + 1]

    # Trilinear interpolation formula (standard formula)
    value = (F000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz) +
             F100 * dx * (1.0 - dy) * (1.0 - dz) +
             F010 * dy * (1.0 - dx) * (1.0 - dz) +
             F001 * dz * (1.0 - dx) * (1.0 - dy) +
             F110 * dx * dy * (1.0 - dz) +
             F101 * dx * dz * (1.0 - dy) +
             F011 * dy * dz * (1.0 - dx) +
             F111 * dx * dy * dz)

    return value

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
    '''
    Performs TSC interpolation for a point, handling boundaries via parent patch interpolation.
    
    Args:
        - ipatch: current patch index
        - i_patch, j_patch, k_patch: cell indices in current patch
        - x, y, z: target coordinates for interpolation
        - pres: current patch cell size
        - nmax: base level grid size
        - patch_nx, patch_ny, patch_nz: current patch dimensions
        - patch_field: field data for current patch
        - level_res: resolution (cell size) for each level
        - patches_pare: parent patch indices
        - patches_nx, patches_ny, patches_nz: dimensions of all patches
        - patches_x, patches_y, patches_z: cell positions in parent coordinates
        - patches_rx, patches_ry, patches_rz: physical positions (center of first cell)
        - patches_levels: refinement level of each patch
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
        
    Indexing notes:
        - All indices are 0-based (Python convention)
        - patches_x/y/z are 0-based from readers.py
        - Refinement factor = 2
        - No index offset adjustments needed (Fortran had +1 for 1-based, we don't)
        
    Returns:
        - Interpolated value at (x, y, z)
        
    Author: Òscar Monllor
    ''' 

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
    '''
    SPH-style interpolation at point (x,y,z) using parent/ancestor cells.
    Similar parent-walk as TSC_interpolation, but uses cubic-spline SPH kernel
    with support 2*h. We gather a small parent stencil (±2) and compute
    weighted average (weights normalized).
    
    Args:
        - ipatch, i_patch, j_patch, k_patch: indices of the patch and cell
        - x, y, z: coordinates of the interpolation point
        - pres: pressure (unused here, but kept for consistency)
        - nmax: maximum number of cells in level 0
        - patch_nx, patch_ny, patch_nz: dimensions of the current patch
        - patch_field: field values in the current patch
        - level_res: resolution at each refinement level
        - patches_pare, patches_nx, patches_ny, patches_nz: parent patch info
        - patches_x, patches_y, patches_z: patch starting indices
        - patches_rx, patches_ry, patches_rz: patch reference coordinates
        - patches_levels: refinement levels of patches
        - coarse_field: level 0 field data
        - arr_fields: list of all patch fields
        
    Indexing notes:
        - All indices are 0-based (Python convention)
        - patches_x/y/z are 0-based from readers.py
        - Refinement factor = 2
        - No index offset adjustments needed (Fortran had +1 for 1-based, we don't)
        
    Returns:
        - Interpolated value at (x, y, z)
        
    Author: Marco Molina
    '''
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
                    fields, nmax, interpol='TSC', use_siblings=True):
    '''
    Fills ghost cells of a single patch using interpolation from parent patches.
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
        interpol: interpolation method - 'TSC', 'SPH', 'LINEAR', 'TRILINEAR' or 'NEAREST'
        use_siblings: if True, check for sibling patches before parent interpolation (default: True)
        
    Returns:
        buffered_patches: list of list of arrays with ghost cells filled
        
    Author: Marco Molina
    '''
    
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
                if nghost > 0:
                    if (i >= nghost and i < nx + nghost and
                        j >= nghost and j < ny + nghost and
                        k >= nghost and k < nz + nghost):
                        continue
                else:
                    # Parent mode: only fill boundary cells on the patch
                    if (i > 0 and i < nx - 1 and
                        j > 0 and j < ny - 1 and
                        k > 0 and k < nz - 1):
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
                # (only if use_siblings is enabled)
                brother_found = False
                
                if use_siblings:
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
                
                ## If not brother, use interpolation from parents (choose method)
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
                    elif interpol == 'SPH':
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
                    elif interpol == 'LINEAR':
                        buffered_patches[f][i, j, k] = linear_interpolation(
                            ipatch, i_local, j_local, k_local,
                            x_ghost, y_ghost, z_ghost,
                            pres, nmax,
                            nx, ny, nz,
                            fields[f][ipatch], level_res,
                            patchpare, patchnx, patchny, patchnz,
                            patchx, patchy, patchz,
                            patchrx, patchry, patchrz, levels,
                            fields[f][0], fields[f])
                    elif interpol == 'TRILINEAR':
                        buffered_patches[f][i, j, k] = trilinear_interpolation(
                            ipatch, i_local, j_local, k_local,
                            x_ghost, y_ghost, z_ghost,
                            pres, nmax,
                            nx, ny, nz,
                            fields[f][ipatch], level_res,
                            patchpare, patchnx, patchny, patchnz,
                            patchx, patchy, patchz,
                            patchrx, patchry, patchrz, levels,
                            fields[f][0], fields[f])
                    elif interpol == 'NEAREST':
                        buffered_patches[f][i, j, k] = nearest_interpolation(
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
                    size, nmax, nghost=1, interpol='TSC', use_siblings=True, bitformat=np.float32, kept_patches=None):
    '''
    Adds ghost buffer cells to all patches in an AMR field using interpolation from parent patches.
    If the ghost cell lies inside a brother patch (same refinement level), copies directly instead of interpolating.
    
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
        interpol: interpolation method - 
                'TSC' (Triangular-Shaped Cloud, high order, creates nghost ghost cells)
                'SPH' (smoothed particle hydrodynamics, high order, creates nghost ghost cells)
                'LINEAR' (linear extrapolation, conservative, creates nghost ghost cells)
                'TRILINEAR' (trilinear interpolation using 8-vertex cube, standard method, creates nghost ghost cells)
                'NEAREST' (nearest-neighbor copy, most conservative, creates nghost ghost cells)
                
                        Note: parent mode (nghost=0) differs from extra buffer addition in that it only fills frontier cells:
                                - parent mode with TSC/SPH/LINEAR/TRILINEAR/NEAREST uses the chosen interpolation
                                    to fill only boundary cells with valid (non-frontier) parent values.
        use_siblings: if True, check for sibling patches before parent interpolation (default: True).
                                        Set to False in parent mode to avoid frontier contamination
        bitformat: data type for the fields (default is np.float32)
        kept_patches: boolean array indicating which patches to process
        
    Returns:
        buffered_fields: list of list of numpy arrays with ghost cells added
        
    Author: Marco Molina
    '''
    
    if interpol not in ['TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST']:
        raise ValueError("Interpolation method must be 'TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST'")
    
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
        
        # Skip buffer creation for level 0 (base level with periodic boundary conditions)
        # Level 0 always has exactly 1 patch at index 0
        if ipatch == 0:
            # Level 0 should not have ghost buffer; return as-is without padding
            for f in range(len(fields)):
                src = fields[f][ipatch]
                if isinstance(src, np.ndarray):
                    buffered_fields[f].append(src.copy())
                else:
                    buffered_fields[f].append(src)
            continue
            
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Create patch with ghost cells (only for refinement levels > 0)
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
            fields_numba, nmax, interpol=interpol, use_siblings=use_siblings
        )
        
        [buffered_fields[f].append(buffered_patches[f]) for f in range(len(fields))]
    
    return buffered_fields


def blend_patch_boundaries(base_fields, parent_fields,
                          patchnx, patchny, patchnz,
                          boundary_width=1, kept_patches=None):
    '''
    Blend patch boundary cells between two field lists.

    Args:
        base_fields: list of per-patch arrays (buffered-then-busted result)
        parent_fields: list of per-patch arrays (parent-filled result)
        patchnx, patchny, patchnz: per-patch dimensions
        boundary_width: number of cells from the edge to blend
        kept_patches: optional mask for valid patches

    Returns:
        blended_fields: list of per-patch arrays with blended boundaries

    Author: Marco Molina
    '''
    if kept_patches is None:
        kept_patches = np.ones(len(base_fields), dtype=bool)

    bw = int(max(0, boundary_width))
    if bw == 0:
        return [bf if isinstance(bf, np.ndarray) else bf for bf in base_fields]

    blended_fields = []
    for ipatch in range(len(base_fields)):
        if not kept_patches[ipatch] or isinstance(base_fields[ipatch], (int, float)):
            blended_fields.append(0)
            continue
        if ipatch == 0:
            src = base_fields[ipatch]
            blended_fields.append(src.copy() if isinstance(src, np.ndarray) else src)
            continue

        base = base_fields[ipatch]
        parent = parent_fields[ipatch]
        if not isinstance(base, np.ndarray) or not isinstance(parent, np.ndarray):
            blended_fields.append(base if isinstance(base, np.ndarray) else 0)
            continue

        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        bw_eff = min(bw, max(1, min(nx, ny, nz) // 2))

        if bw_eff * 2 >= min(nx, ny, nz):
            blended = 0.5 * (base + parent)
            blended_fields.append(blended)
            continue

        blended = base.copy()
        blended[:bw_eff, :, :] = 0.5 * (base[:bw_eff, :, :] + parent[:bw_eff, :, :])
        blended[-bw_eff:, :, :] = 0.5 * (base[-bw_eff:, :, :] + parent[-bw_eff:, :, :])
        blended[:, :bw_eff, :] = 0.5 * (base[:, :bw_eff, :] + parent[:, :bw_eff, :])
        blended[:, -bw_eff:, :] = 0.5 * (base[:, -bw_eff:, :] + parent[:, -bw_eff:, :])
        blended[:, :, :bw_eff] = 0.5 * (base[:, :, :bw_eff] + parent[:, :, :bw_eff])
        blended[:, :, -bw_eff:] = 0.5 * (base[:, :, -bw_eff:] + parent[:, :, -bw_eff:])

        blended_fields.append(blended)

    return blended_fields


def ghost_buffer_buster(buffered_field, patchnx, patchny, patchnz, nghost=1, kept_patches=None):
    '''
    Removes ghost buffer cells from all patches, returning only the interior data.
    
    Args:
        buffered_field: list of numpy arrays with ghost buffer cells
        patchnx, patchny, patchnz: dimensions of each patch (original, without ghost buffer cells)
        nghost: number of ghost cells on each side (must match that used in add_ghost_buffer)
        kept_patches: boolean array indicating which patches were processed
        
    Returns:
        field: list of numpy arrays without ghost buffer cells in original dimensions
        
    Author: Marco Molina
    '''
    
    if kept_patches is None:
        kept_patches = np.ones(len(buffered_field), dtype=bool)
    
    field = []
    
    for ipatch in range(len(buffered_field)):
        if not kept_patches[ipatch] or isinstance(buffered_field[ipatch], (int, float)):
            # return scalar 0 for outside-region patches to match readers' convention
            field.append(0)
            continue
        
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Level 0 (base patch at index 0) has no ghost buffer and is returned as-is
        if ipatch == 0:
            interior_patch = buffered_field[ipatch].copy() if isinstance(buffered_field[ipatch], np.ndarray) else buffered_field[ipatch]
        else:
            # All other patches have ghost buffer; extract interior cells only
            interior_patch = buffered_field[ipatch][nghost:nx+nghost, 
                                                    nghost:ny+nghost, 
                                                    nghost:nz+nghost].copy()
        
        field.append(interior_patch)
    
    return field

def inplace_ghost_buffer_buster(buffered_field, patchnx, patchny, patchnz, nghost=1, kept_patches=None):
    '''
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
    '''
    
    if kept_patches is None:
        kept_patches = np.ones(len(buffered_field), dtype=bool)
    
    for ipatch in range(len(buffered_field)):
        if not kept_patches[ipatch] or isinstance(buffered_field[ipatch], (int, float)):
            continue
        
        nx, ny, nz = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
        
        # Level 0 (base patch at index 0) has no ghost buffer; skip for other patches
        if ipatch != 0:
            # Extract interior cells and replace in list for all non-level-0 patches
            buffered_field[ipatch] = buffered_field[ipatch][nghost:nx+nghost, 
                                                            nghost:ny+nghost, 
                                                            nghost:nz+nghost].copy()


def debug_buffer_passthrough(fields, npatch,
                            patchnx, patchny, patchnz,
                            patchx, patchy, patchz,
                            patchrx, patchry, patchrz, patchpare,
                            size, nmax, nghost=1, interpol='TSC', use_siblings=True, 
                            bitformat=np.float32, kept_patches=None, verbose=True):
    '''
    Debug function to verify that add_ghost_buffer + ghost_buffer_buster pipeline
    does not modify the interior cell values of fields.
    
    Uses the provided fields directly, passes them through the buffer add/remove 
    pipeline, and verifies that interior cells are identical after the round trip.
    
    Args:
        fields: list of list of numpy arrays containing the actual field data to test
        All other args: same as add_ghost_buffer
        verbose: print detailed results (default: True)
        
    Returns:
        debug_report: dictionary with test results:
            - 'passed': boolean, True if all patches passed the test
            - 'total_patches': number of patches tested
            - 'failed_patches': list of patch indices that failed
            - 'max_error': maximum absolute difference found
            - 'error_details': dictionary with per-patch error information
            
    Author: Marco Molina
    '''
    
    if verbose:
        print("\n" + "="*70)
        print("DEBUG: Buffer Pass-Through Test")
        print("="*70)
    
    # Store original interior values (deep copy)
    original_data = []
    for field_set in fields:
        field_original = []
        for ipatch, patch in enumerate(field_set):
            if isinstance(patch, np.ndarray):
                field_original.append(patch.copy())
            else:
                field_original.append(0)
        original_data.append(field_original)
    
    # Pass through buffer pipeline
    buffered_fields = add_ghost_buffer(fields, npatch,
                                    patchnx, patchny, patchnz,
                                    patchx, patchy, patchz,
                                    patchrx, patchry, patchrz, patchpare,
                                    size, nmax, nghost=nghost, interpol=interpol, 
                                    use_siblings=use_siblings, bitformat=bitformat, 
                                    kept_patches=kept_patches)
    
    # Remove buffer
    final_fields = []
    for buffered_field in buffered_fields:
        final_field = ghost_buffer_buster(buffered_field, patchnx, patchny, patchnz, 
                                        nghost=nghost, kept_patches=kept_patches)
        final_fields.append(final_field)
    
    # Compare original and final interior values
    debug_report = {
        'passed': True,
        'total_patches': len(fields[0]),
        'failed_patches': [],
        'max_error': 0.0,
        'error_details': {}
    }
    
    for field_idx in range(len(fields)):
        for ipatch in range(len(fields[field_idx])):
            if kept_patches is not None and not kept_patches[ipatch]:
                continue
            
            original = original_data[field_idx][ipatch]
            final = final_fields[field_idx][ipatch]
            
            if isinstance(original, np.ndarray) and isinstance(final, np.ndarray):
                # Compare interior cells
                diff = np.abs(original - final)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                debug_report['max_error'] = max(debug_report['max_error'], max_diff)
                
                # Use relative tolerance if values are non-zero, absolute otherwise
                max_val = np.max(np.abs(original))
                tolerance = 1e-10 * max_val if max_val > 1e-15 else 1e-15
                
                if max_diff > tolerance:
                    debug_report['passed'] = False
                    debug_report['failed_patches'].append(ipatch)
                    debug_report['error_details'][ipatch] = {
                        'field_index': field_idx,
                        'max_error': max_diff,
                        'mean_error': mean_diff,
                        'shape': original.shape
                    }
    
    if verbose:
        print(f"Total patches tested: {debug_report['total_patches']}")
        print(f"Passed: {debug_report['passed']}")
        if debug_report['failed_patches']:
            print(f"Failed patches: {debug_report['failed_patches']}")
            print(f"Error details:")
            for ipatch, details in debug_report['error_details'].items():
                print(f"  Patch {ipatch}: max_error={details['max_error']:.2e}, "
                    f"mean_error={details['mean_error']:.2e}, shape={details['shape']}")
        else:
            print("All patches passed!")
        print(f"Maximum error found: {debug_report['max_error']:.2e}")
        print("="*70 + "\n")
    
    return debug_report