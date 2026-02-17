"""
Debugging utilities for analyzing and visualizing the AMR patch structure, buffer effects and divergence calculations
in the uniform grid visualization of AMR magnetic dynamo simulations.

Usage:
    python debug_buffer_visual.py <simulation_name> <snapshot_it>
    
Example:
    python scripts/debug.py cluster_B_low_res_paper_2020 1050
    
Authors: Marco Molina
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so `scripts.*` imports work
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from the scripts package
try:
    import scripts.utils as utils
    import scripts.diff as diff
    import scripts.buffer as buff
    import scripts.readers as reader
    from scripts.plot_fields import distribution_check, scan_animation_3D
    from config import IND_PARAMS as ind_params
    from config import OUTPUT_PARAMS as out_params
    from config import DEBUG_PARAMS as debug_params
    from config import SCAN_PLOT_PARAMS as scan_plot_params
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure this script is run from the MAGNAS_SSD_Evo/scripts directory")
    sys.exit(1)


# Use log_message from utils
log_message = utils.log_message
build_terminal_log_filename = utils.build_terminal_log_filename
redirect_output_to_file = utils.redirect_output_to_file


def _format_array_short(arr, max_elements=8):
    '''
    Format array for printing, showing only first and last few elements if too long.
    
    Args:
        arr: numpy array to format
        max_elements: maximum number of elements to show before truncating
        
    Returns:
        formatted string
    '''
    if len(arr) <= max_elements:
        return str(arr)
    
    n_show = max_elements // 2
    first_part = arr[:n_show]
    last_part = arr[-n_show:]
    
    first_str = ' '.join([f'{x:.6e}' if abs(x) < 0.01 or abs(x) > 1000 else f'{x:.4f}' for x in first_part])
    last_str = ' '.join([f'{x:.6e}' if abs(x) < 0.01 or abs(x) > 1000 else f'{x:.4f}' for x in last_part])
    
    return f"[{first_str} ... {last_str}] ({len(arr)} values)"


def analyze_patch_positions(grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, 
                           grid_patchny, grid_patchnz, pare, grid_npatch,
                           dir_params, suspicious_threshold=15.0, verbose=True):
    '''
    Analyze patch positions and identify suspicious patches in boundary regions.
    
    Checks for refined patches that exist in suspicious regions (e.g., high Z coordinates)
    which may indicate issues in AMR structure or data loading.
    
    Args:
        grid_patchrx, grid_patchry, grid_patchrz: Patch position arrays (Mpc)
        grid_patchnx, grid_patchny, grid_patchnz: Patch size arrays (cells)
        pare: Parent patch array
        grid_npatch: Number of patches at each level
        dir_params: Directory containing simulation parameters
        suspicious_threshold: Threshold for flagging suspicious patch positions (Mpc, default=15.0)
        verbose: Print diagnostic output
        
    Returns:
        diagnostics: Dictionary with identified suspicious patches and statistics
        
    Author: Marco Molina
    '''
    
    # Create vector of refinement levels for each patch
    vector_levels_temp = utils.create_vector_levels(grid_npatch)
    
    keep_count = len(grid_patchrx)
    
    # Check all refined patches (level > 0)
    suspicious_patches = []
    for ipatch in range(1, keep_count):  # Skip level 0 (ipatch=0)
        patch_level = vector_levels_temp[ipatch]
        if patch_level > 0:
            # Get patch physical extent
            patch_rx = grid_patchrx[ipatch]
            patch_ry = grid_patchry[ipatch]
            patch_rz = grid_patchrz[ipatch]
            
            # Check if patch center is in suspicious region
            if abs(patch_rz) > suspicious_threshold:
                suspicious_patches.append({
                    'ipatch': ipatch,
                    'level': patch_level,
                    'rx': patch_rx,
                    'ry': patch_ry,
                    'rz': patch_rz,
                    'nx': grid_patchnx[ipatch],
                    'ny': grid_patchny[ipatch],
                    'nz': grid_patchnz[ipatch],
                    'parent': pare[ipatch]
                })
            
            if abs(patch_ry) > suspicious_threshold:
                suspicious_patches.append({
                    'ipatch': ipatch,
                    'level': patch_level,
                    'rx': patch_rx,
                    'ry': patch_ry,
                    'rz': patch_rz,
                    'nx': grid_patchnx[ipatch],
                    'ny': grid_patchny[ipatch],
                    'nz': grid_patchnz[ipatch],
                    'parent': pare[ipatch]
                })
                
            if abs(patch_rx) > suspicious_threshold:
                suspicious_patches.append({
                    'ipatch': ipatch,
                    'level': patch_level,
                    'rx': patch_rx,
                    'ry': patch_ry,
                    'rz': patch_rz,
                    'nx': grid_patchnx[ipatch],
                    'ny': grid_patchny[ipatch],
                    'nz': grid_patchnz[ipatch],
                    'parent': pare[ipatch]
                })
    
    diagnostics = {
        'suspicious_count': len(suspicious_patches),
        'suspicious_patches': suspicious_patches,
        'threshold': suspicious_threshold
    }
    
    if verbose:
        if suspicious_patches:
            print(f"\n⚠️  Found {len(suspicious_patches)} suspicious refined patches with |x,y,z| > {suspicious_threshold} Mpc:")
            for i, p in enumerate(suspicious_patches[:10]):  # Show first 10
                print(f"  Patch {p['ipatch']:4d}: Level={p['level']}, pos=({p['rx']:7.3f}, {p['ry']:7.3f}, {p['rz']:7.3f}) Mpc, "
                      f"size=({p['nx']:2d},{p['ny']:2d},{p['nz']:2d}), parent={p['parent']}")
            if len(suspicious_patches) > 10:
                print(f"  ... and {len(suspicious_patches) - 10} more")
                
            # Show statistics
            z_coords = [p['rz'] for p in suspicious_patches]
            print(f"\n  Z-coordinate statistics for suspicious patches:")
            print(f"    Min Z: {min(z_coords):7.3f} Mpc")
            print(f"    Max Z: {max(z_coords):7.3f} Mpc")
            print(f"    Mean Z: {np.mean(z_coords):7.3f} Mpc")
            
            # Check if they form a strip
            y_coords = [p['ry'] for p in suspicious_patches]
            print(f"  Y-coordinate statistics:")
            print(f"    Min Y: {min(y_coords):7.3f} Mpc")
            print(f"    Max Y: {max(y_coords):7.3f} Mpc")
            print(f"    Mean Y: {np.mean(y_coords):7.3f} Mpc")
        else:
            print(f"✓ No suspicious patches found with |x,y,z| > {suspicious_threshold} Mpc")
        
        # Condensed dump: show patch structure summary only (detailed cell dump disabled by default)
        if verbose and dir_params:
            nmax_param, _, _, size_param = reader.read_parameters(load_nma=True, load_npalev=False,
                                                      load_nlevels=False, load_namr=False,
                                                      load_size=True, path=dir_params)
            base_dx = size_param / nmax_param
            print("\nDIAGNOSTIC: Patch structure summary")
            print(f"  Total patches: {keep_count}")
            print(f"  Base cell size (level 0): {base_dx:.6f} Mpc")
            # Show first 10 and last 10 patches as summary
            for ipatch in range(1, min(11, keep_count)):
                level_patch = vector_levels_temp[ipatch]
                dx = base_dx / (2 ** level_patch)
                nx = grid_patchnx[ipatch]
                ny = grid_patchny[ipatch]
                nz = grid_patchnz[ipatch]
                x0 = grid_patchrx[ipatch]
                y0 = grid_patchry[ipatch]
                z0 = grid_patchrz[ipatch]
                print(f"  Patch {ipatch:4d} (L{level_patch}): origin=({x0:.6f}, {y0:.6f}, {z0:.6f}) Mpc, "
                      f"cell_size={dx:.6f} Mpc, dims=({nx},{ny},{nz})")
            if keep_count > 11:
                print(f"    .")
                print(f"    .")
                print(f"    .")
            for ipatch in range(max(1, keep_count - 10), keep_count):
                level_patch = vector_levels_temp[ipatch]
                dx = base_dx / (2 ** level_patch)
                nx = grid_patchnx[ipatch]
                ny = grid_patchny[ipatch]
                nz = grid_patchnz[ipatch]
                x0 = grid_patchrx[ipatch]
                y0 = grid_patchry[ipatch]
                z0 = grid_patchrz[ipatch]
                print(f"  Patch {ipatch:4d} (L{level_patch}): origin=({x0:.6f}, {y0:.6f}, {z0:.6f}) Mpc, "
                      f"cell_size={dx:.6f} Mpc, dims=({nx},{ny},{nz})")
    
    return diagnostics


def _prepare_patch_metadata(data, verbose=True):
    '''
    Shared initialization: extract and validate patch metadata from data dictionary.

    Args:
        data: output dictionary from load_data containing patch metadata
        verbose: print debug information

    Returns:
        dict containing:
            - npatch: number of patches per level (from data)
            - patchnx, patchny, patchnz: patch dimensions
            - patchrx, patchry, patchrz: patch positions
            - patchpare: parent patch information
            - kept_patches: mask of valid patches (bool array)
            - patch_count: total number of patches
            - levels: level for each patch
            - npatch_adj: adjusted npatch for unigrid convention
            - max_level_found: deepest AMR level present

    Author: Marco Molina
    '''

    kept_patches = data.get('clus_kp', None)
    npatch = data['grid_npatch']
    patchnx = data['grid_patchnx']
    patchny = data['grid_patchny']
    patchnz = data['grid_patchnz']
    patchrx = data['grid_patchrx']
    patchry = data['grid_patchry']
    patchrz = data['grid_patchrz']
    patchpare = data['grid_pare']

    patch_count_raw = len(patchnx)
    if kept_patches is None:
        kept_patches = np.ones(patch_count_raw, dtype=bool)
    else:
        kept_patches = np.asarray(kept_patches, dtype=bool)
        if kept_patches.size != patch_count_raw and verbose:
            log_message(
                f"Kept_patches size {kept_patches.size} mismatches patch count {patch_count_raw}; "
                f"adjusting mask",
                tag='meta',
                level=2
            )
        if kept_patches.size > patch_count_raw:
            kept_patches = kept_patches[:patch_count_raw]
        elif kept_patches.size < patch_count_raw:
            pad_len = patch_count_raw - kept_patches.size
            kept_patches = np.concatenate([kept_patches, np.ones(pad_len, dtype=bool)])

    patch_count = min(
        patch_count_raw,
        len(patchny), len(patchnz), len(patchrx), len(patchry), len(patchrz), len(patchpare),
        kept_patches.size
    )
    if patch_count != patch_count_raw and verbose:
        log_message(f"Truncating patch arrays to {patch_count} (had {patch_count_raw})", tag='meta', level=2)

    patchnx = patchnx[:patch_count]
    patchny = patchny[:patch_count]
    patchnz = patchnz[:patch_count]
    patchrx = patchrx[:patch_count]
    patchry = patchry[:patch_count]
    patchrz = patchrz[:patch_count]
    patchpare = patchpare[:patch_count]
    kept_patches = kept_patches[:patch_count]

    levels = utils.create_vector_levels(npatch)
    if levels.size > patch_count:
        levels = levels[:patch_count]
    elif levels.size < patch_count:
        pad_len = patch_count - levels.size
        levels = np.concatenate([levels, np.full(pad_len, levels[-1] if levels.size else 0)])

    max_level_found = int(np.max(levels)) if levels.size > 0 else 0
    min_size = max(max_level_found + 1, 5)
    npatch_adj = np.zeros(min_size, dtype=int)

    for level_idx in range(min_size):
        npatch_adj[level_idx] = int(np.sum(levels == level_idx))

    npatch_adj[0] = 0
    target_sum = max(patch_count - 1, 0)
    current_sum = int(np.sum(npatch_adj))
    if current_sum != target_sum:
        if current_sum > target_sum:
            excess = current_sum - target_sum
            for lvl in range(len(npatch_adj) - 1, 0, -1):
                if excess <= 0:
                    break
                take = min(excess, npatch_adj[lvl])
                npatch_adj[lvl] -= take
                excess -= take
        else:
            npatch_adj[-1] += (target_sum - current_sum)

    if verbose:
        log_message(f"npatch_adj: {npatch_adj} (max_level={max_level_found}, total_patches={len(levels)})", tag='meta', level=2)

    return {
        'npatch': npatch,
        'patchnx': patchnx,
        'patchny': patchny,
        'patchnz': patchnz,
        'patchrx': patchrx,
        'patchry': patchry,
        'patchrz': patchrz,
        'patchpare': patchpare,
        'kept_patches': kept_patches,
        'patch_count': patch_count,
        'levels': levels,
        'npatch_adj': npatch_adj,
        'max_level_found': max_level_found
    }


def _build_color_field_from_levels(levels, patchnx, patchny, patchnz, kept_patches,
                                   bitformat=np.float32, use_buffer=False, buffered_field=None,
                                   verbose=True):
    '''
    Shared color field creation: builds a field where each cell has its patch's level value.

    Args:
        levels: array with level for each patch
        patchnx, patchny, patchnz: patch dimensions
        kept_patches: mask of valid patches
        bitformat: data type for output
        use_buffer: if True, use buffered_field instead of creating from levels
        buffered_field: pre-buffered field (used when use_buffer=True)
        verbose: print progress

    Returns:
        color_field: list of 3D arrays, one per patch

    Author: Marco Molina
    '''

    patch_count = len(patchnx)
    color_field = []

    if use_buffer and buffered_field is not None:
        color_field = buffered_field
        if verbose:
            log_message("Using pre-computed buffered color field", tag='visual', level=2)
    else:
        for ipatch in range(patch_count):
            if kept_patches[ipatch]:
                patch_shape = (patchnx[ipatch], patchny[ipatch], patchnz[ipatch])
                patch_field = np.full(patch_shape, float(levels[ipatch]), dtype=bitformat, order='F')
                color_field.append(patch_field)
            else:
                patch_shape = (patchnx[ipatch], patchny[ipatch], patchnz[ipatch])
                patch_field = np.zeros(patch_shape, dtype=bitformat, order='F')
                color_field.append(patch_field)

        if verbose:
            log_message(
                f"Created color field from {patch_count} patches (levels range: "
                f"{np.min(levels):.0f}-{np.max(levels):.0f})",
                tag='visual',
                level=2
            )

    return color_field


def _check_buffer_level_values(buffered_field, levels, nghost, use_siblings, max_report=10, verbose=True):
    """
    Validate ghost buffer values for level-encoded fields.

    Expected behavior (when siblings are disabled):
    - Ghost cells should not exceed the parent level, i.e. <= (level - 1).

    With siblings enabled, ghost cells may match the same level.
    
    Author: Marco Molina
    """
    expected_tag = "siblings-allowed" if use_siblings else "parents-only"
    
    if verbose:
        log_message(f"Ghost layer validation ({expected_tag})", tag="buffer-check", level=0)
        log_message(f"Checking {len(buffered_field)} patches with nghost={nghost}", tag="buffer-check", level=1)
    
    violations = []
    good_samples = []

    for ipatch, arr in enumerate(buffered_field):
        if ipatch == 0:
            continue
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            continue
        if min(arr.shape) <= 2 * nghost:
            continue

        level = int(levels[ipatch])
        expected_max = level if use_siblings else max(level - 1, 0)

        # Get interior for comparison
        interior = arr[nghost:-nghost, nghost:-nghost, nghost:-nghost]
        interior_val = float(np.mean(interior))
        
        faces = [
            arr[:nghost, :, :],
            arr[-nghost:, :, :],
            arr[:, :nghost, :],
            arr[:, -nghost:, :],
            arr[:, :, :nghost],
            arr[:, :, -nghost:]
        ]

        ghost_max = max(float(np.max(f)) for f in faces)
        ghost_min = min(float(np.min(f)) for f in faces)
        bad = sum(int(np.sum(f > expected_max + 1e-6)) for f in faces)

        if bad > 0:
            violations.append((ipatch, level, expected_max, interior_val, ghost_min, ghost_max, bad))
        else:
            good_samples.append((ipatch, level, expected_max, interior_val, ghost_min, ghost_max))

    if verbose:
        log_message(f"Violations: {len(violations)} patches", tag="buffer-check", level=1)
        if violations:
            for row in violations[:max_report]:
                ipatch, level, expected_max, interior_val, ghost_min, ghost_max, bad = row
                log_message(
                    f"✗ ipatch={ipatch:3d} L={level} interior={interior_val:.1f} "
                    f"ghost=[{ghost_min:.2f}, {ghost_max:.2f}] expected_max={expected_max} bad_cells={bad}",
                    tag="buffer-check", level=2
                )
        
        log_message(f"Valid samples: {len(good_samples)} patches", tag="buffer-check", level=1)
        
        # Show first 10 and last 10 samples with ellipsis if more than 20
        if len(good_samples) > 0:
            if len(good_samples) <= 2 * max_report:
                # Show all if 20 or fewer
                for row in good_samples:
                    ipatch, level, expected_max, interior_val, ghost_min, ghost_max = row
                    log_message(
                        f"✓ ipatch={ipatch:3d} L={level} interior={interior_val:.1f} "
                        f"ghost=[{ghost_min:.2f}, {ghost_max:.2f}] expected_max={expected_max}",
                        tag="buffer-check", level=2
                    )
            else:
                # Show first 10
                for row in good_samples[:max_report]:
                    ipatch, level, expected_max, interior_val, ghost_min, ghost_max = row
                    log_message(
                        f"✓ ipatch={ipatch:3d} L={level} interior={interior_val:.1f} "
                        f"ghost=[{ghost_min:.2f}, {ghost_max:.2f}] expected_max={expected_max}",
                        tag="buffer-check", level=2
                    )
                # Ellipsis
                log_message("  .", level=2)
                log_message("  .", level=2)
                log_message("  .", level=2)
                # Show last 10
                for row in good_samples[-max_report:]:
                    ipatch, level, expected_max, interior_val, ghost_min, ghost_max = row
                    log_message(
                        f"✓ ipatch={ipatch:3d} L={level} interior={interior_val:.1f} "
                        f"ghost=[{ghost_min:.2f}, {ghost_max:.2f}] expected_max={expected_max}",
                        tag="buffer-check", level=2
                    )


def _diagnose_cell_patch_leakage(cell_patch, fine_coordinates, vertices_patches, 
                                  patchnx, patchny, patchnz, levels, 
                                  up_to_level, uniform_cellsize, max_report=20, verbose=True):
    """
    Check if cell_patch assignments are spatially consistent.
    
    For each cell (i,j,k) assigned to patch ipatch, verify that the coordinate
    (x,y,z) actually falls within the spatial bounds of that patch.
    
    Args:
        cell_patch: 3D array mapping uniform grid cells to patch indices
        fine_coordinates: tuple of (x_coords, y_coords, z_coords) for uniform grid
        vertices_patches: patch boundaries [ipatch, 0:6] = [xmin, xmax, ymin, ymax, zmin, zmax]
        patchnx, patchny, patchnz: patch dimensions
        levels: refinement level for each patch
        up_to_level: target refinement level
        uniform_cellsize: cell size of uniform grid
        max_report: max number of leaks to print
        verbose: print diagnostic information
        
    Returns:
        dict with diagnostic information
        
    Author: Marco Molina
    """
    if verbose:
        log_message("Cell-patch spatial consistency check", tag="cell-patch", level=0)
    
    # Sample cells from the uniform grid (check every Nth cell to avoid slowdown)
    sample_stride = max(1, cell_patch.shape[0] // 50)  # ~50 samples per dimension
    
    leaks = []
    total_checked = 0
    
    for i in range(0, cell_patch.shape[0], sample_stride):
        for j in range(0, cell_patch.shape[1], sample_stride):
            for k in range(0, cell_patch.shape[2], sample_stride):
                ipatch = abs(cell_patch[i, j, k])
                if ipatch == 0:
                    continue
                
                total_checked += 1
                
                # Get cell coordinates
                x = fine_coordinates[0][i]
                y = fine_coordinates[1][j]
                z = fine_coordinates[2][k]
                
                # Get patch bounds
                xmin, xmax = vertices_patches[ipatch, 0], vertices_patches[ipatch, 1]
                ymin, ymax = vertices_patches[ipatch, 2], vertices_patches[ipatch, 3]
                zmin, zmax = vertices_patches[ipatch, 4], vertices_patches[ipatch, 5]
                
                # Check if coordinate is within bounds (with small tolerance for rounding)
                tol = uniform_cellsize * 0.01
                x_ok = (x >= xmin - tol) and (x <= xmax + tol)
                y_ok = (y >= ymin - tol) and (y <= ymax + tol)
                z_ok = (z >= zmin - tol) and (z <= zmax + tol)
                
                if not (x_ok and y_ok and z_ok):
                    leak_info = {
                        'cell': (i, j, k),
                        'coord': (x, y, z),
                        'ipatch': ipatch,
                        'level': int(levels[ipatch]),
                        'bounds': ((xmin, xmax), (ymin, ymax), (zmin, zmax)),
                        'violations': []
                    }
                    
                    if not x_ok:
                        leak_info['violations'].append(f"X: {x:.4f} not in [{xmin:.4f}, {xmax:.4f}]")
                    if not y_ok:
                        leak_info['violations'].append(f"Y: {y:.4f} not in [{ymin:.4f}, {ymax:.4f}]")
                    if not z_ok:
                        leak_info['violations'].append(f"Z: {z:.4f} not in [{zmin:.4f}, {zmax:.4f}]")
                    
                    leaks.append(leak_info)
    
    if verbose:
        log_message(f"Checked {total_checked} cells (stride={sample_stride})", tag="cell-patch", level=1)
        log_message(
            f"Found {len(leaks)} spatial leaks ({100*len(leaks)/max(total_checked,1):.2f}%)",
            tag="cell-patch", level=1
        )
        
        if leaks:
            log_message(f"Showing first {min(len(leaks), max_report)} leaks:", tag="cell-patch", level=1)
            for idx, leak in enumerate(leaks[:max_report]):
                log_message(f"Leak #{idx+1}:", tag="cell-patch", level=2)
                log_message(
                    f"Cell: {leak['cell']} → ipatch={leak['ipatch']} (L{leak['level']})",
                    tag="cell-patch", level=3
                )
                log_message(
                    f"Coord: ({leak['coord'][0]:.4f}, {leak['coord'][1]:.4f}, {leak['coord'][2]:.4f})",
                    tag="cell-patch", level=3
                )
                for violation in leak['violations']:
                    log_message(f"✗ {violation}", tag="cell-patch", level=3)
        else:
            log_message("✓ No spatial leaks detected - all cells correctly assigned", tag="cell-patch", level=1)
        
        # Level distribution analysis
        log_message("Level distribution:", tag="cell-patch", level=1)
        for level in range(up_to_level + 1):
            patches_at_level = np.sum(levels == level)
            cells_at_level = np.sum(np.isin(np.abs(cell_patch), np.where(levels == level)[0]))
            log_message(
                f"L{level}: {patches_at_level} patches, {cells_at_level} cells assigned",
                tag="cell-patch", level=2
            )
    
    return {
        'total_checked': total_checked,
        'leaks': leaks,
        'leak_rate': len(leaks) / max(total_checked, 1)
    }


def _project_field_to_uniform_grid(field, npatch_adj, patchnx, patchny, patchnz,
                                   patchrx, patchry, patchrz, size, nmax, region_box,
                                   up_to_level, kept_patches, interpolate='DIRECT', verbose=True,
                                   clean_output=False, check_cell_patch=False):
    '''
    Shared projection: converts AMR patch field to uniform grid.

    Args:
        field: list of 3D patch arrays
        npatch_adj: adjusted npatch for unigrid convention
        patchnx, patchny, patchnz: patch dimensions
        patchrx, patchry, patchrz: patch positions
        size: simulation box size
        nmax: base grid resolution
        region_box: tuple (xmin, xmax, ymin, ymax, zmin, zmax); None uses full box
        up_to_level: target refinement level
        kept_patches: mask of valid patches
        interpolate (bool or str): 
            - True or 'TRILINEAR': trilinear interpolation (may cause artifacts at boundaries)
            - False or 'NGP': Nearest Grid Point with position calculations
            - 'DIRECT': direct copy from most refined patch (recommended for debug, avoids spurious bands)
        verbose: print progress
        clean_output: if True, replace NaN/inf/outliers with zeros ("sweep under the rug").
                     if False (default), keep raw values to diagnose errors.
        check_cell_patch: if True, run diagnostic to detect spatial leakage in cell_patch assignments

    Returns:
        uniform_volume: 3D numpy array on uniform grid

    Author: Marco Molina
    '''

    if region_box is None:
        half = size / 2
        region_box = (-half, half, -half, half, -half, half)

    if verbose:
        log_message(
            f"Projecting to uniform grid up_to_level={up_to_level}, interpolate={interpolate}",
            tag='visual',
            level=1
        )

    uniform_volume = utils.unigrid(
        field=field,
        box_limits=region_box,
        up_to_level=up_to_level,
        npatch=npatch_adj,
        patchnx=patchnx,
        patchny=patchny,
        patchnz=patchnz,
        patchrx=patchrx,
        patchry=patchry,
        patchrz=patchrz,
        size=size,
        nmax=nmax,
        interpolate=interpolate,
        verbose=False,
        kept_patches=kept_patches,
        return_coords=check_cell_patch
    )
    
    # Run cell_patch diagnostic if requested
    if check_cell_patch:
        uniform_volume, fine_coordinates, cell_patch, vertices_patches = uniform_volume
        
        levels = utils.create_vector_levels(npatch_adj)
        uniform_cellsize = size / nmax / 2 ** up_to_level
        
        _diagnose_cell_patch_leakage(
            cell_patch, fine_coordinates, vertices_patches,
            patchnx, patchny, patchnz, levels, up_to_level, uniform_cellsize,
            verbose=verbose
        )

    # Optional cleaning: replace problematic values with zeros
    # Default behavior (clean_output=False) preserves raw data to diagnose errors
    if clean_output:
        uniform_volume = np.where(
            (~np.isfinite(uniform_volume)) | (np.abs(uniform_volume) < 1e-3) | (np.abs(uniform_volume) > 10),
            0.0,
            uniform_volume
        )

    if verbose:
        finite_mask = np.isfinite(uniform_volume)
        log_message(f"Uniform grid shape: {uniform_volume.shape}", tag='visual', level=2)
        if np.any(finite_mask):
            log_message(
                f"Uniform grid range: {np.min(uniform_volume[finite_mask]):.1f} "
                f"to {np.max(uniform_volume[finite_mask]):.1f}",
                tag='visual',
                level=2
            )
        else:
            log_message("Uniform grid: all values cleaned", tag='visual', level=2)

    return uniform_volume


def build_patch_level_volume(data, size, nmax, region_box=None, up_to_level=None,
                             bitformat=np.float32, verbose=True, use_buffer=False,
                             nghost=1, interpol='NEAREST', use_siblings=True,
                             clean_output=False, interp_mode='DIRECT',
                             buffer_level_check=False, check_cell_patch=False):
    '''
    Creates a uniform grid volume showing patch AMR levels, with optional buffer inflation.
    Useful for diagnosing whether artifacts come from data vs buffer creation.

    Args:
        data: output dictionary from load_data containing patch metadata and kept_patches mask
        size: simulation box size
        nmax: base grid resolution
        region_box: tuple (xmin, xmax, ymin, ymax, zmin, zmax) in box coordinates; None uses full box
        up_to_level: target refinement level for the uniform grid; None -> deepest available level
        bitformat: dtype for the synthetic field
        verbose: print progress
        use_buffer: if True, apply buffer pipeline before projection
        nghost: ghost cells to generate for the visualization
        interpol: buffer interpolation mode
        use_siblings: whether sibling copying is allowed in buffering
        clean_output: if True, replace NaN/inf/outliers with zeros ("sweep under the rug").
                     if False (default), keep raw values to diagnose errors.
        interp_mode: interpolation mode for unigrid projection ('DIRECT', 'NGP', or 'TRILINEAR').
                    'DIRECT' (default) avoids spurious artifacts from zero-division in base grid.

    Returns:
        uniform_volume: 3D numpy array on a uniform grid showing patch levels

    Author: Marco Molina
    '''

    meta = _prepare_patch_metadata(data, verbose=verbose)

    npatch = meta['npatch']
    patchnx = meta['patchnx']
    patchny = meta['patchny']
    patchnz = meta['patchnz']
    patchrx = meta['patchrx']
    patchry = meta['patchry']
    patchrz = meta['patchrz']
    patchpare = meta['patchpare']
    kept_patches = meta['kept_patches']
    patch_count = meta['patch_count']
    levels = meta['levels']
    npatch_adj = meta['npatch_adj']
    max_level_found = meta['max_level_found']

    if up_to_level is None:
        up_to_level = max_level_found

    if verbose:
        mode_label = "with buffer" if use_buffer else "no buffer"
        log_message(f"Building patch level volume ({mode_label}) up to level {up_to_level}", tag='visual', level=1)

    color_field = _build_color_field_from_levels(
        levels, patchnx, patchny, patchnz, kept_patches,
        bitformat=bitformat, use_buffer=False, verbose=verbose
    )

    if use_buffer:
        if verbose and patch_count > 0:
            log_message(
                f"[DEBUG] Base patch (ipatch=0): {'ACCEPTED' if kept_patches[0] else 'REJECTED'}, "
                f"level={levels[0] if kept_patches[0] else 'N/A'}",
                level=2
            )

        buffered = buff.add_ghost_buffer(
            [color_field], npatch,
            patchnx, patchny, patchnz,
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            patchrx, patchry, patchrz, patchpare,
            size=size, nmax=nmax, nghost=nghost, interpol=interpol,
            use_siblings=use_siblings, bitformat=bitformat, kept_patches=kept_patches
        )[0]

        if buffer_level_check:
            _check_buffer_level_values(buffered, levels, nghost, use_siblings)

        dx_levels = size / nmax / (2.0 ** levels)
        patchnx_ext = patchnx.copy()
        patchny_ext = patchny.copy()
        patchnz_ext = patchnz.copy()
        patchrx_ext = patchrx.copy()
        patchry_ext = patchry.copy()
        patchrz_ext = patchrz.copy()

        for ipatch in range(1, len(patchnx)):
            patchnx_ext[ipatch] += 2 * nghost
            patchny_ext[ipatch] += 2 * nghost
            patchnz_ext[ipatch] += 2 * nghost
            # patchrx is center of first INTERIOR cell at coord patchrx
            # Ghost cell center is at patchrx - 1.5*dx (see fill_ghost_buffer)
            # So patchrx_ext must be center of first ghost cell = patchrx - (nghost + 0.5)*dx
            patchrx_ext[ipatch] -= (nghost + 0.5) * dx_levels[ipatch]
            patchry_ext[ipatch] -= (nghost + 0.5) * dx_levels[ipatch]
            patchrz_ext[ipatch] -= (nghost + 0.5) * dx_levels[ipatch]

        if verbose:
            log_message("Building uniform grid for buffer visualization...", tag="visual", level=1)
            log_message(
                f"[DEBUG] Color field range: "
                f"{min([np.min(f) if isinstance(f, np.ndarray) else 0 for f in buffered if isinstance(f, np.ndarray)]):.0f} "
                f"to {max([np.max(f) if isinstance(f, np.ndarray) else 0 for f in buffered if isinstance(f, np.ndarray)]):.0f}",
                level=2
            )
            log_message(f"[DEBUG] Levels per patch: {levels[:10] if len(levels) > 10 else levels}...", level=2)
            log_message(f"[DEBUG] Number of level 0 patches: {np.sum(levels == 0)}", level=2)
            log_message(f"[DEBUG] Number of refined patches (level>0): {np.sum(levels > 0)}", level=2)
            log_message(f"[DEBUG] npatch_adj: {npatch_adj}", level=2)
            log_message(
                f"[DEBUG] Building unigrid with up_to_level={up_to_level}, interp_mode={interp_mode}",
                level=2
            )

        uniform_volume = _project_field_to_uniform_grid(
            buffered, npatch_adj,
            patchnx_ext, patchny_ext, patchnz_ext,
            patchrx_ext, patchry_ext, patchrz_ext,
            size, nmax, region_box, up_to_level, kept_patches,
            interpolate=interp_mode, verbose=verbose, clean_output=clean_output,
            check_cell_patch=check_cell_patch
        )
    else:
        uniform_volume = _project_field_to_uniform_grid(
            color_field, npatch_adj,
            patchnx, patchny, patchnz,
            patchrx, patchry, patchrz,
            size, nmax, region_box, up_to_level, kept_patches,
            interpolate=interp_mode, verbose=verbose, clean_output=clean_output,
            check_cell_patch=check_cell_patch
        )

    if verbose:
        if use_buffer:
            unique_colors = np.unique(uniform_volume[np.isfinite(uniform_volume)])
            log_message(
                f"[DEBUG] Unique colors in volume: {_format_array_short(unique_colors)}",
                level=2
            )
            bottom_unique = np.unique(uniform_volume[:, :, 0])
            log_message(f"[DEBUG] Bottom boundary (z=0) unique values: {_format_array_short(bottom_unique)}", level=2)
            top_unique = np.unique(uniform_volume[:, :, -1])
            log_message(f"[DEBUG] Top boundary (z=-1) unique values: {_format_array_short(top_unique)}", level=2)
            left_unique = np.unique(uniform_volume[0, :, :])
            log_message(f"[DEBUG] Left boundary (x=0) unique values: {_format_array_short(left_unique)}", level=2)
            right_unique = np.unique(uniform_volume[-1, :, :])
            log_message(f"[DEBUG] Right boundary (x=-1) unique values: {_format_array_short(right_unique)}", level=2)
        else:
            unique_vals = np.unique(uniform_volume)
            unique_vals = unique_vals[np.isfinite(unique_vals)]
            log_message(f"Unique levels in raw volume: {_format_array_short(unique_vals)}", tag='visual', level=2)

    return uniform_volume


def compare_divergence_methods(clus_Bx, clus_By, clus_Bz, grid_npatch, clus_kp,
                               patchnx, patchny, patchnz,
                               patchx, patchy, patchz,
                               patchrx, patchry, patchrz, patchpare,
                               grid_irr, dir_params, size, nmax,
                               use_buffer=False, nghost=1, interpol='TSC',
                               parent_interpol=None,
                               use_siblings=True, blend=False, parent_mode=False, stencil=3, verbose=True):
    '''
    Compare divergence calculation methods for magnetic fields.

    Computes divergence with three approaches and reports statistics:
    1) Pipeline method: diff.divergence (with/without buffer based on use_buffer)
    2) Periodic method: centered differences using np.roll
    3) Inline method: centered differences with boundary extrapolation

    Args:
        clus_Bx, clus_By, clus_Bz: magnetic field components
        grid_npatch: number of patches per level
        clus_kp: mask for valid patches
        patchnx, patchny, patchnz: patch dimensions
        patchx, patchy, patchz: patch coordinates  
        patchrx, patchry, patchrz: patch positions
        patchpare: parent patch indices
        grid_irr: snapshot index (for logging)
        dir_params: directory containing simulation parameters
        size: simulation box size
        nmax: base grid resolution
        use_buffer: if True, use buffer for pipeline method
        nghost: number of ghost cells
        interpol: interpolation method for buffer
        use_siblings: use sibling patches
        verbose: print diagnostic output

    Returns:
        results: dictionary with per-method statistics

    Author: Marco Molina
    '''
    dx = size / nmax
    if parent_interpol is None:
        parent_interpol = interpol
    parent_use_siblings = False
    
    parent_active = bool(parent_mode) and not blend

    if verbose:
        if use_buffer and parent_active:
            buffer_mode = "parent frontier fill"
        else:
            buffer_mode = "with buffer" if use_buffer else "without buffer (extrapolation)"
        log_message(f'Divergence Methods Comparison (snapshot {grid_irr}, {buffer_mode})', tag="divergence", level=1)
        if use_buffer and (parent_active or blend):
            log_message(
                f'Params: buffer_interpol={interpol}, parent_interpol={parent_interpol}, '
                f'nghost={nghost}, use_siblings={use_siblings}, stencil={stencil}, blend={blend}',
                tag="divergence",
                level=2
            )
        else:
            log_message(
                f'Params: interpol={interpol}, nghost={nghost}, use_siblings={use_siblings}, '
                f'stencil={stencil}, blend={blend}',
                tag="divergence",
                level=2
            )
        log_message(f'Grid: nmax={nmax}, size={size}, dx={dx:.3e}', tag="divergence", level=2)

    def _stats_from_div(div_list, label):
        div_flat = np.concatenate([
            np.abs(div_list[p]).flatten() for p in range(npatch_count) if bool(clus_kp[p])
        ])
        stats = {
            'max': np.max(div_flat) if div_flat.size else 0.0,
            'min': np.min(div_flat) if div_flat.size else 0.0,
            'mean': np.mean(div_flat) if div_flat.size else 0.0,
            'median': np.median(div_flat) if div_flat.size else 0.0,
            'std': np.std(div_flat) if div_flat.size else 0.0,
            'p95': np.percentile(div_flat, 95) if div_flat.size else 0.0,
            'p99': np.percentile(div_flat, 99) if div_flat.size else 0.0,
            'cells': div_flat.size
        }
        if verbose:
            log_message(f'Snap {grid_irr} ({label}): |∇·B| stats over {stats["cells"]} cells', tag="DEBUG", level=2)
            log_message(
                f'  min={stats["min"]:.6e}, max={stats["max"]:.6e}, mean={stats["mean"]:.6e}, std={stats["std"]:.6e}',
                tag="DEBUG",
                level=2
            )
            log_message(
                f'  median={stats["median"]:.6e}, p95={stats["p95"]:.6e}, p99={stats["p99"]:.6e}',
                tag="DEBUG",
                level=2
            )
        return stats, div_flat

    npatch_count = grid_npatch.sum() + 1
    if clus_kp is None:
        clus_kp = np.ones(npatch_count, dtype=bool)

    B_magnitude_flat = np.concatenate([
        np.sqrt(clus_Bx[p]**2 + clus_By[p]**2 + clus_Bz[p]**2).flatten()
        for p in range(npatch_count) if bool(clus_kp[p])
    ])
    mean_B_magnitude = np.mean(B_magnitude_flat) if B_magnitude_flat.size else 0.0
    levels = utils.create_vector_levels(grid_npatch)
    resolution = dx / (2 ** np.array(levels))
    mean_B_magnitude_per_dx = mean_B_magnitude / np.mean(resolution)

    blend_boundary = 1 if stencil == 3 else 2
    effective_nghost = 0 if parent_active else nghost
    if blend and effective_nghost == 0:
        effective_nghost = blend_boundary

    if use_buffer:
        if effective_nghost > 0:
            # Pipeline method WITH buffer
            buffered = buff.add_ghost_buffer(
                [clus_Bx, clus_By, clus_Bz], grid_npatch,
                patchnx, patchny, patchnz,
                patchx, patchy, patchz,
                patchrx, patchry, patchrz, patchpare,
                size=size, nmax=nmax, nghost=effective_nghost,
                interpol=interpol, use_siblings=use_siblings,
                kept_patches=clus_kp
            )
            Bx_buff, By_buff, Bz_buff = buffered
            div_pipeline_buffered = diff.divergence(Bx_buff, By_buff, Bz_buff,
                                                   dx, grid_npatch, clus_kp, stencil=stencil)
            div_pipeline = buff.ghost_buffer_buster(div_pipeline_buffered, patchnx, patchny, patchnz,
                                                   nghost=effective_nghost, kept_patches=clus_kp)

            if blend:
                div_parent = diff.divergence(clus_Bx, clus_By, clus_Bz,
                                              dx, grid_npatch, clus_kp, stencil=stencil)
                div_parent = buff.add_ghost_buffer(
                    [div_parent], grid_npatch,
                    patchnx, patchny, patchnz,
                    patchx, patchy, patchz,
                    patchrx, patchry, patchrz, patchpare,
                    size=size, nmax=nmax, nghost=0,
                    interpol=parent_interpol, use_siblings=parent_use_siblings,
                    kept_patches=clus_kp
                )[0]
                div_pipeline = buff.blend_patch_boundaries(
                    div_pipeline, div_parent,
                    patchnx, patchny, patchnz,
                    boundary_width=blend_boundary, kept_patches=clus_kp
                )
                pipeline_stats, div_B_flat = _stats_from_div(div_pipeline, 'pipeline method (blend)')
            else:
                pipeline_stats, div_B_flat = _stats_from_div(div_pipeline, 'pipeline method (with buffer)')
        else:
            # Pipeline method with parent frontier fill
            div_pipeline = diff.divergence(clus_Bx, clus_By, clus_Bz,
                                           dx, grid_npatch, clus_kp, stencil=stencil)
            div_pipeline = buff.add_ghost_buffer(
                [div_pipeline], grid_npatch,
                patchnx, patchny, patchnz,
                patchx, patchy, patchz,
                patchrx, patchry, patchrz, patchpare,
                size=size, nmax=nmax, nghost=0,
                interpol=parent_interpol, use_siblings=parent_use_siblings,
                kept_patches=clus_kp
            )[0]
            pipeline_stats, div_B_flat = _stats_from_div(div_pipeline, 'pipeline method (parent fill)')
    else:
        # Pipeline method WITHOUT buffer (extrapolation)
        div_pipeline = diff.divergence(clus_Bx, clus_By, clus_Bz,
                                       dx, grid_npatch, clus_kp, stencil=stencil)
        pipeline_stats, div_B_flat = _stats_from_div(div_pipeline, 'pipeline method (extrapolation)')

    if mean_B_magnitude > 0:
        if verbose:
            log_message(
                f'Snap {grid_irr} (pipeline method): Mean |∇·B| / |B| = '
                f'{np.mean(div_B_flat) / mean_B_magnitude_per_dx:.6e}',
                tag="DEBUG",
                level=2
            )
        pipeline_ratio = np.mean(div_B_flat) / mean_B_magnitude_per_dx
    else:
        if verbose:
            log_message(
                f'Snap {grid_irr} (pipeline method): Mean |B| = 0, '
                f'cannot compute |∇·B| / |B| ratio.',
                tag="DEBUG",
                level=2
            )
        pipeline_ratio = None
        
    def divergence_inline(Bx, By, Bz, dx_local, npatch, kp):
        levels_local = utils.create_vector_levels(npatch)
        resolution_local = dx_local / (2 ** np.array(levels_local))
        div = []
        div_x = []
        div_y = []
        div_z = []

        if kp is None:
            kp = np.ones(npatch.sum() + 1, dtype=bool)

        for p in range(npatch.sum() + 1):
            if not bool(kp[p]):
                div.append(0)
                continue

            Bx_p = Bx[p]
            By_p = By[p]
            Bz_p = Bz[p]
            res_p = resolution_local[p]

            dBx_dx = np.zeros_like(Bx_p)
            dBy_dy = np.zeros_like(By_p)
            dBz_dz = np.zeros_like(Bz_p)

            dBx_dx[1:-1, :, :] = (Bx_p[2:, :, :] - Bx_p[0:-2, :, :]) / (2 * res_p)
            dBy_dy[:, 1:-1, :] = (By_p[:, 2:, :] - By_p[:, 0:-2, :]) / (2 * res_p)
            dBz_dz[:, :, 1:-1] = (Bz_p[:, :, 2:] - Bz_p[:, :, 0:-2]) / (2 * res_p)

            dBx_dx[0, :, :] = (4 * Bx_p[1, :, :] - 3 * Bx_p[0, :, :] - Bx_p[2, :, :]) / (2 * res_p)
            dBx_dx[-1, :, :] = (3 * Bx_p[-1, :, :] + Bx_p[-3, :, :] - 4 * Bx_p[-2, :, :]) / (2 * res_p)
            dBy_dy[:, 0, :] = (4 * By_p[:, 1, :] - 3 * By_p[:, 0, :] - By_p[:, 2, :]) / (2 * res_p)
            dBy_dy[:, -1, :] = (3 * By_p[:, -1, :] + By_p[:, -3, :] - 4 * By_p[:, -2, :]) / (2 * res_p)
            dBz_dz[:, :, 0] = (4 * Bz_p[:, :, 1] - 3 * Bz_p[:, :, 0] - Bz_p[:, :, 2]) / (2 * res_p)
            dBz_dz[:, :, -1] = (3 * Bz_p[:, :, -1] + Bz_p[:, :, -3] - 4 * Bz_p[:, :, -2]) / (2 * res_p)

            div_p = dBx_dx + dBy_dy + dBz_dz
            div.append(div_p)
            div_x.append(dBx_dx)
            div_y.append(dBy_dy)
            div_z.append(dBz_dz)

        return div, div_x, div_y, div_z

    div_inline, div_Bx3, div_By3, div_Bz3 = divergence_inline(
        clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp
    )
    inline_stats, div_B_flat3 = _stats_from_div(div_inline, 'inline method')

    if mean_B_magnitude > 0:
        if verbose:
            log_message(
                f'Snap {grid_irr} (inline method): Mean |∇·B| / |B| = '
                f'{np.mean(div_B_flat3) / mean_B_magnitude_per_dx:.6e}',
                tag="DEBUG",
                level=2
            )
        inline_ratio = np.mean(div_B_flat3) / mean_B_magnitude_per_dx
    else:
        if verbose:
            log_message(
                f'Snap {grid_irr} (inline method): Mean |B| = 0, '
                f'cannot compute |∇·B| / |B| ratio.',
                tag="DEBUG",
                level=2
            )
        inline_ratio = None

    def divergence_periodic(Bx, By, Bz, dx_local, npatch, kp):
        levels_local = utils.create_vector_levels(npatch)
        resolution_local = dx_local / (2 ** np.array(levels_local))
        div = []
        div_x = []
        div_y = []
        div_z = []

        if kp is None:
            kp = np.ones(npatch.sum() + 1, dtype=bool)

        for p in range(npatch.sum() + 1):
            if not bool(kp[p]):
                div.append(0)
                continue

            Bx_p = Bx[p]
            By_p = By[p]
            Bz_p = Bz[p]
            res_p = resolution_local[p]

            dBx_dx = (np.roll(Bx_p, -1, axis=0) - np.roll(Bx_p, 1, axis=0)) / (2 * res_p)
            dBy_dy = (np.roll(By_p, -1, axis=1) - np.roll(By_p, 1, axis=1)) / (2 * res_p)
            dBz_dz = (np.roll(Bz_p, -1, axis=2) - np.roll(Bz_p, 1, axis=2)) / (2 * res_p)

            div_p = dBx_dx + dBy_dy + dBz_dz
            div.append(div_p)
            div_x.append(dBx_dx)
            div_y.append(dBy_dy)
            div_z.append(dBz_dz)

        return div, div_x, div_y, div_z

    div_periodic, div_Bx, div_By, div_Bz = divergence_periodic(
        clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp
    )
    periodic_stats, div_B_flat2 = _stats_from_div(div_periodic, 'periodic method')

    if mean_B_magnitude > 0:
        if verbose:
            log_message(
                f'Snap {grid_irr} (periodic method): Mean |∇·B| / |B| = '
                f'{np.mean(div_B_flat2) / mean_B_magnitude_per_dx:.6e}',
                tag="DEBUG",
                level=2
            )
        periodic_ratio = np.mean(div_B_flat2) / mean_B_magnitude_per_dx
    else:
        if verbose:
            log_message(
                f'Snap {grid_irr} (periodic method): Mean |B| = 0, '
                f'cannot compute |∇·B| / |B| ratio.',
                tag="DEBUG",
                level=2
            )
        periodic_ratio = None

    return {
        'pipeline': {
            'stats': pipeline_stats,
            'ratio': pipeline_ratio
        },
        'periodic': {
            'stats': periodic_stats,
            'ratio': periodic_ratio
        },
        'inline': {
            'stats': inline_stats,
            'ratio': inline_ratio
        },
        'parent_interpol': parent_interpol,
        'diagnostics': {
            'dx': dx,
            'mean_B_magnitude': mean_B_magnitude,
            'mean_B_magnitude_per_dx': mean_B_magnitude_per_dx,
            'div_Bx': div_Bx,
            'div_By': div_By,
            'div_Bz': div_Bz,
            'div_Bx_inline': div_Bx3,
            'div_By_inline': div_By3,
            'div_Bz_inline': div_Bz3
        }
    }


def diagnose_buffer_vs_extrapolation(clus_Bx, clus_By, clus_Bz, grid_npatch, clus_kp,
                                     patchnx, patchny, patchnz,
                                     patchx, patchy, patchz,
                                     patchrx, patchry, patchrz, patchpare,
                                     grid_irr, size, nmax, dx, nghost=1,
                                     interpol='TSC', stencil=3, use_siblings=True, blend=False,
                                     parent_interpol=None, parent_mode=False, verbose=True):
    '''
    MAIN DIAGNOSTIC: Compare buffer vs. extrapolation divergence in detail.
    
     This function compares:
     1. Divergence WITHOUT buffer (pure extrapolation at patch boundaries)
     2. Divergence WITH buffer (ghost cells interpolated, then divergence calculated)
         or parent frontier fill when nghost=0
    
    Args:
        clus_Bx, clus_By, clus_Bz: magnetic field components
        grid_npatch: number of patches per level
        clus_kp: mask for valid patches
        patchnx, patchny, patchnz: patch dimensions
        patchx, patchy, patchz: patch coordinates
        patchrx, patchry, patchrz: patch positions
        patchpare: parent patch indices
        grid_irr: snapshot index
        size: simulation box size
        nmax: base grid resolution
        dx: cell size
        nghost: number of ghost cells per side
        interpol: interpolation method from IND_PARAMS
        stencil: stencil size from IND_PARAMS
        use_siblings: use sibling patches from IND_PARAMS
        verbose: print diagnostic output
        
    Returns:
        results: dictionary with detailed comparison metrics
        
    Author: Marco Molina
    '''
    
    if parent_interpol is None:
        parent_interpol = interpol
    parent_use_siblings = False

    parent_active = bool(parent_mode) and not blend

    if verbose:
        log_message(f'Buffer vs Extrapolation Comparison (snapshot {grid_irr})', tag="buffer_comp", level=1)
        mode = "parent frontier fill" if parent_active else "buffer"
        if parent_active or blend:
            log_message(
                f'Using: buffer_interpol={interpol}, parent_interpol={parent_interpol}, '
                f'stencil={stencil}, use_siblings={use_siblings}, nghost={nghost} ({mode})',
                tag="buffer_comp",
                level=2
            )
        else:
            log_message(
                f'Using: interpol={interpol}, stencil={stencil}, use_siblings={use_siblings}, nghost={nghost} ({mode})',
                tag="buffer_comp",
                level=2
            )
    
    # Calculate divergence WITHOUT buffer (baseline: extrapolation)
    if verbose:
        log_message(f'Computing divergence WITHOUT buffer (extrapolation)...', tag="buffer_comp", level=2)
    
    div_no_buffer = diff.divergence(clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp, stencil=stencil)
    stats_no_buffer = _extract_divergence_stats(div_no_buffer, clus_kp, grid_npatch,
                                                verbose=verbose, tag="buffer_comp",
                                                label=f"No buffer (stencil-{stencil})")
    
    # Calculate divergence WITH buffer or parent frontier fill
    blend_boundary = 1 if stencil == 3 else 2
    effective_nghost = 0 if parent_active else nghost
    if blend and effective_nghost == 0:
        effective_nghost = blend_boundary

    if effective_nghost > 0:
        if verbose:
            label_tag = "buffer" if not blend else "buffer+parent blend"
            log_message(f'Computing divergence WITH {label_tag} ({interpol} interpolation)...', tag="buffer_comp", level=2)
        
        # Step 1: Add ghost cells
        buffered = buff.add_ghost_buffer(
            [clus_Bx, clus_By, clus_Bz], grid_npatch,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz, patchpare,
            size=size, nmax=nmax, nghost=effective_nghost,
            interpol=interpol, use_siblings=use_siblings,
            kept_patches=clus_kp
        )
        
        Bx_buff, By_buff, Bz_buff = buffered
        
        # Step 2: Calculate divergence on buffered fields
        div_with_buffer_buffered = diff.divergence(Bx_buff, By_buff, Bz_buff, dx, grid_npatch, clus_kp, stencil=stencil)
        
        # Step 3: Remove ghost cells from divergence result using ghost_buffer_buster
        div_with_buffer = buff.ghost_buffer_buster(div_with_buffer_buffered, patchnx, patchny, patchnz, 
                                                   nghost=effective_nghost, kept_patches=clus_kp)

        if blend:
            div_parent = buff.add_ghost_buffer(
                [div_no_buffer], grid_npatch,
                patchnx, patchny, patchnz,
                patchx, patchy, patchz,
                patchrx, patchry, patchrz, patchpare,
                size=size, nmax=nmax, nghost=0,
                interpol=parent_interpol, use_siblings=parent_use_siblings,
                kept_patches=clus_kp
            )[0]
            div_with_buffer = buff.blend_patch_boundaries(
                div_with_buffer, div_parent,
                patchnx, patchny, patchnz,
                boundary_width=blend_boundary, kept_patches=clus_kp
            )
            label = f"Blend (stencil-{stencil})"
        else:
            label = f"With buffer (stencil-{stencil})"
    else:
        if verbose:
            log_message(
                f'Computing divergence with parent frontier fill ({parent_interpol})...',
                tag="buffer_comp",
                level=2
            )
        div_with_buffer = buff.add_ghost_buffer(
            [div_no_buffer], grid_npatch,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz, patchpare,
            size=size, nmax=nmax, nghost=0,
            interpol=parent_interpol, use_siblings=parent_use_siblings,
            kept_patches=clus_kp
        )[0]
        label = f"Parent fill ({parent_interpol}, stencil-{stencil})"
    
    # Now div_with_buffer has same dimensions as div_no_buffer
    stats_with_buffer = _extract_divergence_stats(div_with_buffer, clus_kp, grid_npatch,
                                                  verbose=verbose, tag="buffer_comp",
                                                  label=label)
    
    # Compute differences
    if verbose:
        log_message('Comparison summary (with buffer/parent vs no-buffer):', tag="buffer_comp", level=2)
        delta_max = stats_with_buffer['max'] - stats_no_buffer['max']
        delta_mean = stats_with_buffer['mean'] - stats_no_buffer['mean']
        ratio_max = stats_with_buffer['max'] / stats_no_buffer['max'] if stats_no_buffer['max'] > 0 else np.inf
        ratio_mean = stats_with_buffer['mean'] / stats_no_buffer['mean'] if stats_no_buffer['mean'] > 0 else np.inf
        
        log_message(f'  Δmax = {delta_max:+.6e} (ratio={ratio_max:.3f}x)', tag="buffer_comp", level=2)
        log_message(f'  Δmean = {delta_mean:+.6e} (ratio={ratio_mean:.3f}x)', tag="buffer_comp", level=2)
        
        if ratio_max > 1.5:
            log_message(f'  ⚠ Buffer increases max divergence by {(ratio_max-1)*100:.1f}%', tag="buffer_comp", level=2)
        elif ratio_max < 0.8:
            log_message(f'  ✓ Buffer reduces max divergence by {(1-ratio_max)*100:.1f}%', tag="buffer_comp", level=2)
        else:
            log_message(f'  ≈ Buffer has minimal impact on divergence', tag="buffer_comp", level=2)
    
    results = {
        'snapshot': grid_irr,
        'interpol': interpol,
        'parent_interpol': parent_interpol,
        'stencil': stencil,
        'use_siblings': use_siblings,
        'no_buffer': stats_no_buffer,
        'with_buffer': stats_with_buffer,
        'delta_max': stats_with_buffer['max'] - stats_no_buffer['max'],
        'delta_mean': stats_with_buffer['mean'] - stats_no_buffer['mean'],
        'ratio_max': stats_with_buffer['max'] / stats_no_buffer['max'] if stats_no_buffer['max'] > 0 else np.inf,
        'ratio_mean': stats_with_buffer['mean'] / stats_no_buffer['mean'] if stats_no_buffer['mean'] > 0 else np.inf,
    }
    
    return results


# Function removed - buffer comparison now done directly in diagnose_buffer_vs_extrapolation


def _extract_divergence_stats(div_list, kp, npatch, verbose=False, tag="div_stats", label=""):
    '''
    Extract comprehensive statistics from divergence field.
    
    Args:
        div_list: list of divergence arrays per patch
        kp: valid patches mask
        npatch: number of patches per level
        verbose: if True, print statistics to terminal
        tag: log message tag for output
        label: descriptive label for this divergence field
        
    Returns:
        stats: dictionary with divergence statistics
    '''
    npatch_count = npatch.sum() + 1
    
    if kp is None:
        kp = np.ones(npatch_count, dtype=bool)
    
    # Flatten active patches
    div_flat = []
    div_by_patch = {}
    
    for p in range(npatch_count):
        if not bool(kp[p]):
            continue
        
        div_p = np.abs(div_list[p]).flatten()
        div_flat.extend(div_p)
        
        # Per-patch stats
        div_by_patch[p] = {
            'max': np.max(div_p) if div_p.size else 0.0,
            'mean': np.mean(div_p) if div_p.size else 0.0,
            'median': np.median(div_p) if div_p.size else 0.0,
            'std': np.std(div_p) if div_p.size else 0.0,
            'p95': np.percentile(div_p, 95) if div_p.size else 0.0,
            'p99': np.percentile(div_p, 99) if div_p.size else 0.0,
        }
    
    div_flat = np.array(div_flat)
    
    stats = {
        'max': np.max(div_flat) if div_flat.size else 0.0,
        'min': np.min(div_flat) if div_flat.size else 0.0,
        'mean': np.mean(div_flat) if div_flat.size else 0.0,
        'median': np.median(div_flat) if div_flat.size else 0.0,
        'std': np.std(div_flat) if div_flat.size else 0.0,
        'p95': np.percentile(div_flat, 95) if div_flat.size else 0.0,
        'p99': np.percentile(div_flat, 99) if div_flat.size else 0.0,
        'total_cells': div_flat.size,
        'per_patch': div_by_patch,
    }
    
    if verbose:
        prefix = f"{label}: " if label else ""
        log_message(f'{prefix}max={stats["max"]:.6e}, mean={stats["mean"]:.6e}, std={stats["std"]:.6e}', 
                   tag=tag, level=2)
        log_message(f'{prefix}p95={stats["p95"]:.6e}, p99={stats["p99"]:.6e} ({stats["total_cells"]} cells)', 
                   tag=tag, level=3)
    
    return stats


def diagnose_ghost_cell_values(Bx, By, Bz, npatch, patchnx, patchny, patchnz,
                               patchx, patchy, patchz,
                               patchrx, patchry, patchrz, patchpare,
                               size, nmax, nghost=1, interpol='TSC',
                               parent_interpol=None,
                               use_siblings=True, parent_mode=False, blend=False, verbose=True):
    '''
    GHOST CELL INSPECTION: Examine buffer ghost cell values for reasonableness.
    If nghost=0 (parent mode), inspect boundary cells instead.
    
    Checks:
    - Ghost cell value ranges (are they extreme?)
    - Ghost cell smoothness (do they match interior trends?)
    - Ghost cell gradient magnitudes
    - NaN/Inf detection
    
    Args:
        Bx, By, Bz: field components (no buffer yet)
        npatch, patchnx, patchny, patchnz, ...: AMR structure
        size, nmax: simulation parameters
        nghost: number of ghost cells
        interpol: interpolation method
        use_siblings: use sibling patches
        verbose: print diagnostics
        
    Returns:
        results: dictionary with ghost cell diagnostics
        
    Author: Marco Molina
    '''
    
    if parent_interpol is None:
        parent_interpol = interpol
    parent_use_siblings = False

    use_parent_fill = bool(parent_mode) and not blend

    if verbose:
        if use_parent_fill:
            log_message(f'Boundary Cell Inspection: parent fill ({parent_interpol})', tag="ghost_cells", level=1)
        else:
            log_message(f'Ghost Cell Inspection: {interpol} interpolation', tag="ghost_cells", level=1)
    
    if not use_parent_fill:
        # Create buffered version
        buffered = buff.add_ghost_buffer(
            [Bx, By, Bz], npatch,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz, patchpare,
            size=size, nmax=nmax, nghost=nghost,
            interpol=interpol, use_siblings=use_siblings
        )
        
        Bx_buff, By_buff, Bz_buff = buffered
    else:
        buffered = buff.add_ghost_buffer(
            [Bx, By, Bz], npatch,
            patchnx, patchny, patchnz,
            patchx, patchy, patchz,
            patchrx, patchry, patchrz, patchpare,
            size=size, nmax=nmax, nghost=0,
            interpol=parent_interpol, use_siblings=parent_use_siblings
        )
        Bx_buff, By_buff, Bz_buff = buffered
    
    results = {
        'method': parent_interpol if use_parent_fill else interpol,
        'nghost': 0 if use_parent_fill else nghost,
        'patches_analyzed': 0,
        'total_nan_inf': 0,
        'value_range_global': {'min': np.inf, 'max': -np.inf}
    }
    
    # Per-patch analysis
    all_ghost_values = []
    patches_with_issues = []
    
    for ipatch in range(len(Bx_buff)):
        if ipatch == 0 or not isinstance(Bx_buff[ipatch], np.ndarray):
            continue
        
        bx_p = Bx_buff[ipatch]
        by_p = By_buff[ipatch]
        bz_p = Bz_buff[ipatch]
        
        nx, ny, nz = bx_p.shape
        
        # Extract ghost or boundary cell values from all components
        if not use_parent_fill:
            ghost_cells_all = np.concatenate([
                bx_p[0:nghost, nghost:-nghost, nghost:-nghost].flatten(),
                bx_p[-nghost:, nghost:-nghost, nghost:-nghost].flatten(),
                by_p[nghost:-nghost, 0:nghost, nghost:-nghost].flatten(),
                by_p[nghost:-nghost, -nghost:, nghost:-nghost].flatten(),
                bz_p[nghost:-nghost, nghost:-nghost, 0:nghost].flatten(),
                bz_p[nghost:-nghost, nghost:-nghost, -nghost:].flatten(),
            ])
        else:
            ghost_cells_all = np.concatenate([
                bx_p[0:1, :, :].flatten(),
                bx_p[-1:, :, :].flatten(),
                by_p[:, 0:1, :].flatten(),
                by_p[:, -1:, :].flatten(),
                bz_p[:, :, 0:1].flatten(),
                bz_p[:, :, -1:].flatten(),
            ])
        
        nan_inf_count = int(np.isnan(ghost_cells_all).sum() + np.isinf(ghost_cells_all).sum())
        
        if ghost_cells_all.size > 0:
            all_ghost_values.extend(ghost_cells_all)
            results['patches_analyzed'] += 1
            results['total_nan_inf'] += nan_inf_count
            
            patch_min = np.min(ghost_cells_all)
            patch_max = np.max(ghost_cells_all)
            
            if patch_min < results['value_range_global']['min']:
                results['value_range_global']['min'] = patch_min
            if patch_max > results['value_range_global']['max']:
                results['value_range_global']['max'] = patch_max
            
            if nan_inf_count > 0:
                patches_with_issues.append((ipatch, nan_inf_count))
    
    # Compute global statistics
    if len(all_ghost_values) > 0:
        all_ghost_values = np.array(all_ghost_values)
        results['mean'] = np.mean(all_ghost_values)
        results['std'] = np.std(all_ghost_values)
        results['median'] = np.median(all_ghost_values)
    
    # Print results to terminal
    if verbose:
        label = "Boundary cells" if use_parent_fill else "Ghost cells"
        total_patches = len(Bx_buff)
        skipped = total_patches - results["patches_analyzed"]
        log_message(
            f'{label} analyzed across {results["patches_analyzed"]}/{total_patches} patches (skipped={skipped})',
            tag="ghost_cells",
            level=2
        )
        if len(all_ghost_values) > 0:
            log_message(f'Samples analyzed: {len(all_ghost_values)} values', tag="ghost_cells", level=2)
        
        if len(all_ghost_values) > 0:
            log_message(f'Ghost cell value range: [{results["value_range_global"]["min"]:.6e}, {results["value_range_global"]["max"]:.6e}]', 
                       tag="ghost_cells", level=2)
            log_message(f'Ghost cell statistics: mean={results["mean"]:.6e}, std={results["std"]:.6e}, median={results["median"]:.6e}', 
                       tag="ghost_cells", level=2)
        
        if results['total_nan_inf'] > 0:
            nan_ratio = (results["total_nan_inf"] / max(len(all_ghost_values), 1)) * 100.0
            log_message(
                f'WARNING: {results["total_nan_inf"]} NaN/Inf values detected ({nan_ratio:.3f}% of samples)',
                tag="ghost_cells",
                level=0
            )
            if len(patches_with_issues) <= 5:
                for ipatch, count in patches_with_issues:
                    log_message(f'  Patch {ipatch}: {count} NaN/Inf values', tag="ghost_cells", level=2)
            else:
                log_message(f'  Issues found in {len(patches_with_issues)} patches', tag="ghost_cells", level=2)
        else:
            log_message(f'No NaN/Inf values detected (GOOD)', tag="ghost_cells", level=2)
    
    return results


def diagnose_divergence_spatial_distribution(div_list, npatch, kp, grid_irr,
                                            use_buffer=False, parent_mode=False, blend_mode=False, verbose=True):
    '''
    SPATIAL DISTRIBUTION: Where do high divergence values cluster?
    
    Creates heatmaps per patch showing divergence intensity.
    Useful for identifying if buffer artifacts cluster at patch boundaries.
    
    Args:
        div_list: divergence field (list of patch arrays)
        npatch: number of patches per level
        kp: valid patches mask
        grid_irr: snapshot index
        use_buffer: whether divergence was calculated with buffer
        verbose: print diagnostics
        
    Returns:
        results: dictionary with spatial clustering metrics
        
    Author: Marco Molina
    '''
    
    npatch_count = npatch.sum() + 1
    
    if kp is None:
        kp = np.ones(npatch_count, dtype=bool)
    
    if verbose:
        if use_buffer and blend_mode:
            buffer_mode = "blend (buffer + parent)"
        elif use_buffer and parent_mode:
            buffer_mode = "parent frontier fill"
        else:
            buffer_mode = "with buffer" if use_buffer else "without buffer (extrapolation)"
        log_message(f'Divergence Spatial Distribution Analysis (snapshot {grid_irr}, {buffer_mode})', tag="spatial_div", level=1)
        log_message('Boundary layer uses 2-cell thickness on each face', tag="spatial_div", level=2)
    
    # Aggregate statistics
    boundary_ratios = []
    high_div_ratios = []
    max_divs = []
    patches_analyzed = 0
    
    for p in range(npatch_count):
        if not bool(kp[p]):
            continue
        
        div_p = np.abs(div_list[p])
        patches_analyzed += 1
        
        # Find cells with maximum divergence
        max_div = np.max(div_p)
        max_divs.append(max_div)
        
        max_indices = np.where(div_p > 0.95 * max_div)
        high_div_ratio = len(max_indices[0]) / div_p.size if div_p.size > 0 else 0.0
        high_div_ratios.append(high_div_ratio)
        
        # Boundary layer analysis
        boundary_stats = _analyze_boundary_layer(div_p)
        boundary_ratios.append(boundary_stats['ratio_boundary_to_interior'])
    
    # Print aggregate results
    if verbose and patches_analyzed > 0:
        log_message(f'Analyzed {patches_analyzed}/{npatch_count} patches', tag="spatial_div", level=2)
        
        # Max divergence statistics
        max_div_global = np.max(max_divs)
        mean_max_div = np.mean(max_divs)
        log_message(f'Global max |∇·B|: {max_div_global:.6e} (mean across patches: {mean_max_div:.6e})', 
                   tag="spatial_div", level=2)
        
        # High divergence clustering
        mean_high_div_ratio = np.mean(high_div_ratios)
        max_high_div_ratio = np.max(high_div_ratios)
        log_message(
            f'High-divergence clustering (>95% of patch max): mean={mean_high_div_ratio*100:.2f}%, max={max_high_div_ratio*100:.2f}%',
                   tag="spatial_div", level=2)
        
        # Boundary vs interior analysis
        valid_ratios = [r for r in boundary_ratios if not np.isinf(r) and not np.isnan(r)]
        if len(valid_ratios) > 0:
            mean_boundary_ratio = np.mean(valid_ratios)
            median_boundary_ratio = np.median(valid_ratios)
            max_boundary_ratio = np.max(valid_ratios)
            
            log_message(f'Boundary/Interior divergence ratio: mean={mean_boundary_ratio:.2f}x, median={median_boundary_ratio:.2f}x, max={max_boundary_ratio:.2f}x', 
                       tag="spatial_div", level=2)
            
            # Interpretation
            if mean_boundary_ratio > 3.0:
                log_message(f'⚠ ARTIFACT DETECTED: High boundary concentration (ratio > 3.0) suggests edge artifacts', 
                           tag="spatial_div", level=0)
            elif mean_boundary_ratio > 1.5:
                log_message(f'Moderate boundary concentration (ratio > 1.5) - may be acceptable trade-off', 
                           tag="spatial_div", level=2)
            else:
                log_message(f'Divergence evenly distributed (ratio ≈ 1.0) - no edge artifacts detected', 
                           tag="spatial_div", level=2)
    
    results = {
        'snapshot': grid_irr,
        'patches_analyzed': patches_analyzed,
        'max_divergence_global': max_div_global if patches_analyzed > 0 else 0.0,
        'mean_boundary_ratio': mean_boundary_ratio if len(valid_ratios) > 0 else np.nan,
        'median_boundary_ratio': median_boundary_ratio if len(valid_ratios) > 0 else np.nan,
    }
    
    return results


def _analyze_boundary_layer(div_patch):
    '''
    Analyze if divergence is concentrated at patch boundaries (suggests buffer artifacts).
    
    Returns: {'interior': stats, 'boundary_1': stats, 'boundary_2': stats} where
    boundary_N is the Nth layer from the edge.
    '''
    nx, ny, nz = div_patch.shape
    
    # Define layers
    interior = div_patch[2:-2, 2:-2, 2:-2]
    boundary_1 = np.concatenate([
        div_patch[0:2, :, :].flatten(),      # left x face
        div_patch[-2:, :, :].flatten(),      # right x face
        div_patch[2:-2, 0:2, :].flatten(),   # left y face
        div_patch[2:-2, -2:, :].flatten(),   # right y face
        div_patch[2:-2, 2:-2, 0:2].flatten(), # left z face
        div_patch[2:-2, 2:-2, -2:].flatten(), # right z face
    ])
    
    return {
        'interior_mean': np.mean(interior) if interior.size > 0 else 0.0,
        'boundary_mean': np.mean(boundary_1) if boundary_1.size > 0 else 0.0,
        'ratio_boundary_to_interior': (np.mean(boundary_1) / np.mean(interior)) if (interior.size > 0 and np.mean(interior) > 0) else np.inf,
    }


def build_scan_animation_data(data, size, nmax, region_coords, nghost, use_siblings,
                              up_to_level, bitformat=np.float32, verbose=False, clean_output=False):
    '''
    Build visualization volumes for scan animation debugging.

    Args:
        data: data dictionary from load_data()
        size: simulation box size
        nmax: base grid resolution
        region_coords: region coordinates or [None]
        nghost: number of ghost cells
        use_siblings: use sibling patches
        up_to_level: AMR refinement level
        bitformat: data type for visualization field
        verbose: print debug output
        clean_output: if True, replace NaN/inf/outliers with zeros.
                     if False (default), keep raw values to diagnose errors.

    Returns:
        scan_pack: dictionary with visualization volumes and metadata
    '''
    if region_coords is None or (isinstance(region_coords, list) and len(region_coords) > 0 and region_coords[0] is None):
        region_box = None
    else:
        region_box = tuple(region_coords[1:])

    volume = build_patch_level_volume(
        data=data,
        size=size,
        nmax=nmax,
        region_box=region_box,
        up_to_level=up_to_level,
        bitformat=bitformat,
        verbose=verbose,
        use_buffer=True,
        nghost=nghost,
        interpol='NEAREST',
        use_siblings=use_siblings,
        clean_output=clean_output,
        interp_mode=debug_params.get('unigrid_interp_mode', 'DIRECT'),
        buffer_level_check=not debug_params.get('buffer_level_check', {}).get('enabled', False),
        check_cell_patch=(not debug_params.get('buffer_level_check', {}).get('check_cell_patch', False)) 
                         if debug_params.get('buffer_level_check', {}).get('enabled', False) else False
    )
    region_size = size if region_box is None else (region_box[1] - region_box[0])
    levels_for_colorbar = data.get('vector_levels', np.array([]))

    volume_no_buffer = build_patch_level_volume(
        data=data,
        size=size,
        nmax=nmax,
        region_box=region_box,
        up_to_level=up_to_level,
        bitformat=bitformat,
        verbose=verbose,
        use_buffer=False,
        clean_output=clean_output,
        interp_mode=debug_params.get('unigrid_interp_mode', 'DIRECT'),
        buffer_level_check=False,
        check_cell_patch=False
    )

    return {
        '_scan_volume': volume,
        '_scan_volume_no_buffer': volume_no_buffer,
        '_scan_region_size': region_size,
        '_scan_volume_levels': levels_for_colorbar
    }


def analyze_debug_fields(field_sources, region_coords, data, debug_params,
                         up_to_level, size, nmax, pipeline_debug_results=None,
                         scan_pack=None, verbose=False, it=None, sims=None,
                         field_list=None):
    '''
    Build debug field outputs based on configuration.

    Args:
        field_sources: list of source dictionaries (data, vectorial, induction, induction_energy)
        region_coords: region coordinates or [None]
        data: data dictionary from load_data()
        debug_params: DEBUG_PARAMS configuration
        up_to_level: AMR refinement level
        size: simulation box size
        nmax: base grid resolution
        pipeline_debug_results: optional buffer pipeline diagnostics
        scan_pack: optional scan visualization data
        verbose: print debug output
        it: snapshot index (for logging)
        sims: simulation name (for logging)
        field_list: optional list of fields to analyze

    Returns:
        debug_fields: dictionary with debug outputs
    '''
    debug_data = field_list or [
        'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2',
        'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'
    ]

    debug_fields = {
        key: _get_debug_field(key, field_sources, region_coords, data,
                              debug_params, up_to_level, size, nmax)
        for key in debug_data
    }
    debug_fields = {k: v for k, v in debug_fields.items() if v is not None}

    if pipeline_debug_results is not None:
        debug_fields['_pipeline_validation'] = pipeline_debug_results
    if scan_pack is not None:
        debug_fields.update(scan_pack)

    if verbose and it is not None and sims is not None:
        log_message(
            'Debug fields computed for iteration ' + str(it) +
            ' in simulation ' + str(sims),
            tag="DEBUG",
            level=1
        )

    return debug_fields


def analyze_metadata(data, size, nmax, verbose=True):
    """Analyze patch metadata for inconsistencies."""
    print("\n" + "="*60)
    print("METADATA ANALYSIS")
    print("="*60)
    
    meta = _prepare_patch_metadata(data, verbose=verbose)
    
    print(f"\nPatch Summary:")
    print(f"  Total patches: {meta['patch_count']}")
    print(f"  Max level found: {meta['max_level_found']}")
    print(f"  npatch_adj: {meta['npatch_adj']}")
    
    # Count patches per level
    for level in range(meta['max_level_found'] + 1):
        count = int(np.sum(meta['levels'] == level))
        print(f"  Level {level}: {count} patches")
    
    # Check kept_patches mask
    kept_count = int(np.sum(meta['kept_patches']))
    rejected_count = meta['patch_count'] - kept_count
    print(f"\nKept patches: {kept_count}, Rejected: {rejected_count}")
    
    if rejected_count > 0:
        rejected_indices = np.where(~meta['kept_patches'])[0]
        print(f"  Rejected indices (first 10): {rejected_indices[:10]}")
    
    return meta


def compare_volumes(data, size, nmax, region_box=None, up_to_level=None, 
                   nghost=1, bitformat=np.float32, verbose=True, clean_output=False,
                   interp_mode='DIRECT', buffer_level_check=False, check_cell_patch=False,
                   use_siblings=False):
    """Build both volumes and compare them."""
    print("\n" + "="*60)
    print("BUILDING COMPARISON VOLUMES")
    print("="*60)
    
    print("\n[1/2] Building raw patch level volume (no buffer)...")
    vol_raw = build_patch_level_volume(
        data, size, nmax, region_box=region_box, up_to_level=up_to_level, bitformat=bitformat,
        verbose=verbose, clean_output=clean_output, interp_mode=interp_mode,
        buffer_level_check=False, check_cell_patch=False
    )
    print(f"  Shape: {vol_raw.shape}")
    print(f"  Range: {np.nanmin(vol_raw):.1f} to {np.nanmax(vol_raw):.1f}")
    raw_unique = np.unique(vol_raw[np.isfinite(vol_raw)])
    print(f"  Unique values: {_format_array_short(raw_unique)}")
    
    print("\n[2/2] Building buffered volume...")
    vol_buffered = build_patch_level_volume(
        data, size, nmax, region_box=region_box, up_to_level=up_to_level, bitformat=bitformat,
        verbose=verbose, use_buffer=True, nghost=nghost, interpol='NEAREST',
        use_siblings=use_siblings, clean_output=clean_output, interp_mode=interp_mode,
        buffer_level_check=buffer_level_check, check_cell_patch=check_cell_patch
    )
    print(f"  Shape: {vol_buffered.shape}")
    print(f"  Range: {np.nanmin(vol_buffered):.1f} to {np.nanmax(vol_buffered):.1f}")
    buf_unique = np.unique(vol_buffered[np.isfinite(vol_buffered)])
    print(f"  Unique values: {_format_array_short(buf_unique)}")
    
    return vol_raw, vol_buffered


def analyze_spurious_band(vol_raw, vol_buffered, size, nmax, up_to_level=1):
    """Analyze the spurious cells artifact."""
    print("\n" + "="*60)
    print("SPURIOUS CELLS ANALYSIS")
    print("="*60)
    
    # Convert to physical coordinates
    half = size / 2
    x = np.linspace(-half, half, vol_raw.shape[0])
    y = np.linspace(-half, half, vol_raw.shape[1])
    z = np.linspace(-half, half, vol_raw.shape[2])
    
    # Find Z index around 20
    z_target = 20.0
    z_idx = np.argmin(np.abs(z - z_target))
    z_actual = z[z_idx]
    print(f"\nTarget Z coordinate: {z_target} Mpc")
    print(f"Actual Z index: {z_idx}, Actual Z value: {z_actual:.2f} Mpc")
    print(f"Volume shape: {vol_raw.shape} (X, Y, Z)")
    
    # Analyze XY slices at different Z positions (fast overview)
    print(f"\n{'Position':<15} {'Z [Mpc]':<12} {'Raw Max':<12} {'Buf Max':<12} {'Diff':<12}")
    print("-" * 60)
    
    z_slices = {
        'Bottom': 0,
        'Lower quarter': vol_raw.shape[2]//4,
        'Middle': vol_raw.shape[2]//2,
        'Upper quarter': 3*vol_raw.shape[2]//4,
        'Top': -1
    }
    
    for label, z_slice in z_slices.items():
        raw_slice = vol_raw[:, :, z_slice]
        buf_slice = vol_buffered[:, :, z_slice]
        
        z_val = z[z_slice]
        raw_max = np.nanmax(raw_slice) if np.any(np.isfinite(raw_slice)) else 0
        buf_max = np.nanmax(buf_slice) if np.any(np.isfinite(buf_slice)) else 0
        diff_max = np.nanmax(np.abs(buf_slice - raw_slice))
        
        print(f"{label:<15} {z_val:>11.2f} {raw_max:>11.1f} {buf_max:>11.1f} {diff_max:>11.6f}")
    
    # Detailed analysis around Z~20 (spurious band location)
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS AROUND Z ≈ 20 Mpc (Spurious Band Region)")
    print(f"{'='*60}")
    
    for z_off in [-2, -1, 0, 1, 2]:
        z_idx_test = min(max(z_idx + z_off, 0), len(z)-1)
        raw_xy = vol_raw[:, :, z_idx_test]
        buf_xy = vol_buffered[:, :, z_idx_test]
        
        z_val = z[z_idx_test]
        
        # Statistics for this XY plane
        raw_unique = np.unique(raw_xy[np.isfinite(raw_xy)])
        buf_unique = np.unique(buf_xy[np.isfinite(buf_xy)])
        
        raw_max = np.nanmax(raw_xy) if np.any(np.isfinite(raw_xy)) else 0
        buf_max = np.nanmax(buf_xy) if np.any(np.isfinite(buf_xy)) else 0
        
        diff_xy = np.abs(buf_xy - raw_xy)
        max_diff = np.nanmax(diff_xy) if np.any(np.isfinite(diff_xy)) else 0
        
        # Count spurious values (values exceeding expected max refinement level from config)
        spurious_threshold = up_to_level + 0.5  # Safety margin
        spurious_raw = np.sum((raw_xy > spurious_threshold) & np.isfinite(raw_xy))
        spurious_buf = np.sum((buf_xy > spurious_threshold) & np.isfinite(buf_xy))
        
        print(f"\nZ = {z_val:>6.2f} Mpc (index {z_idx_test}):")
        print(f"  Raw:  unique_levels = {_format_array_short(raw_unique)}, max = {raw_max:.1f}, spurious_count = {spurious_raw} (>{spurious_threshold:.1f})")
        print(f"  Buf:  unique_levels = {_format_array_short(buf_unique)}, max = {buf_max:.1f}, spurious_count = {spurious_buf} (>{spurious_threshold:.1f})")
        print(f"  Diff: max = {max_diff:.6f}")
        print(f"  Expected max level from config: ≤{up_to_level}")
        
        # If spurious values found, show their XY distribution
        if spurious_raw > 0 or spurious_buf > 0:
            print(f"  ⚠ SPURIOUS VALUES DETECTED in this Z-plane!")
            if spurious_raw > 0:
                spur_locs_raw = np.where((raw_xy > spurious_threshold) & np.isfinite(raw_xy))
                x_spur = x[spur_locs_raw[0]]
                y_spur = y[spur_locs_raw[1]]
                print(f"     Raw spurious X range: [{np.min(x_spur):.2f}, {np.max(x_spur):.2f}]")
                print(f"     Raw spurious Y range: [{np.min(y_spur):.2f}, {np.max(y_spur):.2f}]")
    
    # Global difference statistics
    print(f"\n{'='*60}")
    print(f"GLOBAL DIFFERENCE ANALYSIS")
    print(f"{'='*60}")
    
    diff = np.abs(vol_buffered - vol_raw)
    diff_finite = diff[np.isfinite(diff)]
    
    if diff_finite.size > 0:
        print(f"Max difference:  {np.max(diff_finite):.6f}")
        print(f"Mean difference: {np.mean(diff_finite):.6f}")
        print(f"Std difference:  {np.std(diff_finite):.6f}")
        
        # Locations with significant differences
        if np.max(diff_finite) > 1e-3:
            diff_locations = np.where(diff > 1e-3)
            if len(diff_locations[0]) > 0:
                print(f"\nVoxels with diff > 1e-3: {len(diff_locations[0])}")
                print(f"Sample locations (first 5):")
                for i in range(min(5, len(diff_locations[0]))):
                    ix, iy, iz = diff_locations[0][i], diff_locations[1][i], diff_locations[2][i]
                    x_pos, y_pos, z_pos = x[ix], y[iy], z[iz]
                    print(f"  ({ix:3d}, {iy:3d}, {iz:3d}) -> ({x_pos:6.2f}, {y_pos:6.2f}, {z_pos:6.2f}) Mpc: "
                          f"raw={vol_raw[ix,iy,iz]:.2f}, buffered={vol_buffered[ix,iy,iz]:.2f}")
                if len(diff_locations[0]) > 5:
                    print(f"  ... and {len(diff_locations[0]) - 5} more voxels")
        else:
            print("\n✓ No significant differences (all < 1e-3)")
    else:
        print("\n✓ Volumes are identical!")


def find_refined_bands(vol_raw, vol_buffered, size, nmax, up_to_level=1):
    """Find all regions with high refinement levels."""
    print("\n" + "="*60)
    print("HIGH REFINEMENT REGION ANALYSIS")
    print("="*60)
    
    half = size / 2
    z = np.linspace(-half, half, vol_raw.shape[2])
    
    print(" ...According to the uniform grid projection volumes")
    print(f"Expected max refinement level from config: {up_to_level}")
    
    spurious_threshold = up_to_level + 0.5  # Values exceeding expected max
    print(f"\nRaw volume - regions with level > {spurious_threshold:.1f} (spurious):")
    refined_raw = np.where(vol_raw > spurious_threshold)
    if len(refined_raw[0]) > 0:
        z_indices_raw = refined_raw[2]  # Z is 3rd dimension
        z_values = z[z_indices_raw]
        print(f"  Z range: {np.min(z_values):.2f} to {np.max(z_values):.2f} Mpc")
        print(f"  Total refined voxels: {len(z_indices_raw)}")
        print(f"  Z bins with refined voxels (first 20):")
        unique_z_indices = np.unique(z_indices_raw)
        for z_idx in unique_z_indices[:20]:
            z_val = z[z_idx]
            count = np.sum(refined_raw[2] == z_idx)
            print(f"    Z index {z_idx:3d} (Z={z_val:6.2f} Mpc): {count:5d} refined voxels")
        if len(unique_z_indices) > 20:
            print(f"    ... and {len(unique_z_indices) - 20} more Z indices with refined voxels.")
    else:
        print(f"  No spurious voxels found (all ≤ {spurious_threshold:.1f})")
    
    print(f"\nBuffered volume - regions with level > {spurious_threshold:.1f} (spurious):")
    refined_buf = np.where(vol_buffered > spurious_threshold)
    if len(refined_buf[0]) > 0:
        z_indices_buf = refined_buf[2]  # Z is 3rd dimension
        z_values = z[z_indices_buf]
        print(f"  Z range: {np.min(z_values):.2f} to {np.max(z_values):.2f} Mpc")
        print(f"  Total refined voxels: {len(z_indices_buf)}")
        print(f"  Z bins with refined voxels (first 20):")
        unique_z_indices = np.unique(z_indices_buf)
        for z_idx in unique_z_indices[:20]:
            z_val = z[z_idx]
            count = np.sum(refined_buf[2] == z_idx)
            print(f"    Z index {z_idx:3d} (Z={z_val:6.2f} Mpc): {count:5d} refined voxels")
        if len(unique_z_indices) > 20:
            print(f"    ... and {len(unique_z_indices) - 20} more Z indices with spurious voxels.")
    else:
        print(f"  No spurious voxels found (all ≤ {spurious_threshold:.1f})")


def analyze_nan_regions(vol_raw, vol_buffered, size, nmax):
    """Analyze NaN regions which may indicate spurious bands."""
    print("\n" + "="*60)
    print("NaN REGION ANALYSIS (POTENTIAL SPURIOUS BAND ARTIFACT)")
    print("="*60)
    
    nan_raw = np.isnan(vol_raw)
    nan_buf = np.isnan(vol_buffered)
    
    print(f"\nRaw volume NaN statistics:")
    print(f"  Total NaN voxels: {np.sum(nan_raw)}")
    print(f"  NaN percentage: {100 * np.sum(nan_raw) / vol_raw.size:.4f}%")
    
    print(f"\nBuffered volume NaN statistics:")
    print(f"  Total NaN voxels: {np.sum(nan_buf)}")
    print(f"  NaN percentage: {100 * np.sum(nan_buf) / vol_buffered.size:.4f}%")
    
    # Find where NaN appears
    if np.sum(nan_raw) > 0:
        nan_indices_raw = np.where(nan_raw)
        print(f"\nRaw volume - NaN locations:")
        print(f"  X range (indices): {np.min(nan_indices_raw[0])} to {np.max(nan_indices_raw[0])}")
        print(f"  Y range (indices): {np.min(nan_indices_raw[1])} to {np.max(nan_indices_raw[1])}")
        print(f"  Z range (indices): {np.min(nan_indices_raw[2])} to {np.max(nan_indices_raw[2])} ⚠ (spurious band axis)")
        
        # Convert to physical coordinates
        half = size / 2
        x_coords = np.linspace(-half, half, vol_raw.shape[0])[nan_indices_raw[0]]
        y_coords = np.linspace(-half, half, vol_raw.shape[1])[nan_indices_raw[1]]
        z_coords = np.linspace(-half, half, vol_raw.shape[2])[nan_indices_raw[2]]
        
        print(f"  X range (Mpc): {np.min(x_coords):.2f} to {np.max(x_coords):.2f}")
        print(f"  Y range (Mpc): {np.min(y_coords):.2f} to {np.max(y_coords):.2f}")
        print(f"  Z range (Mpc): {np.min(z_coords):.2f} to {np.max(z_coords):.2f} ⚠ (check for Z~20)")
    
    if np.sum(nan_buf) > 0:
        nan_indices_buf = np.where(nan_buf)
        print(f"\nBuffered volume - NaN locations:")
        print(f"  X range (indices): {np.min(nan_indices_buf[0])} to {np.max(nan_indices_buf[0])}")
        print(f"  Y range (indices): {np.min(nan_indices_buf[1])} to {np.max(nan_indices_buf[1])}")
        print(f"  Z range (indices): {np.min(nan_indices_buf[2])} to {np.max(nan_indices_buf[2])} ⚠ (spurious band axis)")
        
        # Convert to physical coordinates
        half = size / 2
        x_coords = np.linspace(-half, half, vol_buffered.shape[0])[nan_indices_buf[0]]
        y_coords = np.linspace(-half, half, vol_buffered.shape[1])[nan_indices_buf[1]]
        z_coords = np.linspace(-half, half, vol_buffered.shape[2])[nan_indices_buf[2]]
        
        print(f"  X range (Mpc): {np.min(x_coords):.2f} to {np.max(x_coords):.2f}")
        print(f"  Y range (Mpc): {np.min(y_coords):.2f} to {np.max(y_coords):.2f}")
        print(f"  Z range (Mpc): {np.min(z_coords):.2f} to {np.max(z_coords):.2f} ⚠ (check for Z~20)")
    
    # Compare NaN patterns
    print(f"\nNaN pattern comparison:")
    nan_both = np.sum(nan_raw & nan_buf)
    nan_only_raw = np.sum(nan_raw & ~nan_buf)
    nan_only_buf = np.sum(~nan_raw & nan_buf)
    
    print(f"  NaN in both: {nan_both}")
    print(f"  NaN only in raw: {nan_only_raw}")
    print(f"  NaN only in buffered: {nan_only_buf}")
    
    if nan_only_raw > 0 or nan_only_buf > 0:
        print(f"  ⚠️  DIFFERENCE DETECTED - NaN patterns differ between raw and buffered!")
        if nan_only_buf > 0:
            print(f"      Buffered has {nan_only_buf} extra NaN voxels (possible spurious region)")
            nan_diff_indices = np.where(~nan_raw & nan_buf)
            if len(nan_diff_indices[0]) > 0:
                half = size / 2
                z_diff = np.linspace(-half, half, vol_buffered.shape[2])[nan_diff_indices[2]]
                print(f"      Extra NaN Z range: {np.min(z_diff):.2f} to {np.max(z_diff):.2f} Mpc (expected ~20 for spurious band)")


def _get_debug_field(key, field_sources, region_coords, data, debug_params, up_to_level, size, nmax):
    '''
    Helper function to retrieve and process debug fields.
    
    Args:
        - key: the key of the field to retrieve
        - field_sources: list of dictionaries containing possible field sources
        - region_coords: coordinates defining the region of interest
        - data: dictionary containing grid parameters
        - debug_params: dictionary with debug configuration (structure from DEBUG_PARAMS)
        - up_to_level: level of refinement in the AMR grid
        - size: size of the grid
        - nmax: maximum number of patches in the grid
        
    Returns:
        - processed field or None if not found
        
    Author: Marco Molina
    '''
    field = None
    for source in field_sources:
        if source is None:
            continue
        if key in source.keys():
            field = source[key]
            break
    
    if field is None:
        return None
    
    # Get field_analysis configuration
    field_analysis_config = debug_params.get("field_analysis", {})
    
    # Check if field should be converted to uniform grid
    if field_analysis_config.get("uniform_grid", False):
        return utils.unigrid(
            field=field,
            box_limits=region_coords[1:] if len(region_coords) > 1 else None,
            up_to_level=up_to_level,
            npatch=data['grid_npatch'],
            patchnx=data['grid_patchnx'],
            patchny=data['grid_patchny'],
            patchnz=data['grid_patchnz'],
            patchrx=data['grid_patchrx'],
            patchry=data['grid_patchry'],
            patchrz=data['grid_patchrz'],
            size=size,
            nmax=nmax,
            interpolate=True,
            verbose=False,
            kept_patches=data['clus_kp'],
            return_coords=False
        )
    elif field_analysis_config.get("clean_field", False):
        return utils.clean_field(
            field,
            data['clus_cr0amr'],
            data['clus_solapst'],
            data['grid_npatch'],
            up_to_level=up_to_level
        )
    else:  # Return raw AMR field
        return field


def run_debug_buffer_pipeline(data, size, nmax, nghost=1, interpol='TSC', 
                            use_siblings=True, bitformat=np.float32, verbose=True):
    '''
    Runs a comprehensive debug test of the buffer add/remove pipeline without
    performing any operations that modify field values.
    
    Takes the fields, passes them through the buffer pipeline, and
    compares the final values with originals to ensure no unexpected modifications
    occur during ghost cell handling and removal.
    
    Args:
        data: dictionary containing grid data from load_data()
        size: simulation box size
        nmax: base level grid resolution
        nghost: number of ghost cells
        interpol: interpolation method
        bitformat: data type for fields
        verbose: print detailed results
        
    Returns:
        debug_results: dictionary with comprehensive test results
        
    Author: Marco Molina
    '''
    
    if verbose:
        log_message("="*80, tag="DEBUG", level=1)
        log_message("RUNNING COMPREHENSIVE DEBUG BUFFER PIPELINE TEST", tag="DEBUG", level=1)
        log_message("="*80, tag="DEBUG", level=1)
    
    debug_results = {
        'buffer_test': None,
        'divergence_test': None,
        'passed_all': True,
        'summary': {}
    }
    
    # Test 1: Buffer pass-through with magnetic and velocity fields
    if verbose:
        log_message("\n[1/2] Testing buffer pipeline (add + remove)...", tag="DEBUG", level=1)
    
    test_fields = [data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                data['clus_vx'], data['clus_vy'], data['clus_vz']]
    
    buffer_test = buff.debug_buffer_passthrough(
        test_fields, data['grid_npatch'],
        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
        data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
        data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
        data['grid_pare'],
        size=size, nmax=nmax, nghost=nghost, interpol=interpol, 
        use_siblings=use_siblings, bitformat=bitformat, 
        kept_patches=data['clus_kp'], verbose=verbose
    )
    
    debug_results['buffer_test'] = buffer_test
    if not buffer_test['passed']:
        debug_results['passed_all'] = False
    
    # Test 2: Divergence field consistency
    if verbose:
        log_message("\n[2/2] Testing divergence field consistency...", tag="DEBUG", level=1)
    
    dx = size / nmax
    divergence_test = diff.debug_divergence_no_operation(
        data['clus_Bx'], data['clus_By'], data['clus_Bz'],
        dx, data['grid_npatch'],
        kept_patches=data['clus_kp'],
        stencil=3,  # Using 3-point stencil as default
        verbose=verbose
    )
    
    debug_results['divergence_test'] = divergence_test
    if not divergence_test['passed']:
        debug_results['passed_all'] = False
    
    # Summary
    debug_results['summary'] = {
        'buffer_passed': buffer_test['passed'],
        'divergence_passed': divergence_test['passed'],
        'overall_passed': debug_results['passed_all'],
        'buffer_max_error': buffer_test['max_error'],
        'divergence_max_error': divergence_test.get('max_divergence_abs'),
        'divergence_max_div_curl': divergence_test.get('max_div_curl')
    }
    
    if verbose:
        log_message("\n" + "="*80, tag="DEBUG", level=1)
        log_message("DEBUG TEST SUMMARY", tag="DEBUG", level=1)
        log_message("="*80, tag="DEBUG", level=1)
        log_message(f"Buffer Test: {'PASSED ✓' if buffer_test['passed'] else 'FAILED ✗'}", tag="DEBUG", level=1)
        log_message(f"Divergence Test: {'PASSED ✓' if divergence_test['passed'] else 'FAILED ✗'}", tag="DEBUG", level=1)
        log_message(f"Overall: {'ALL TESTS PASSED ✓' if debug_results['passed_all'] else 'SOME TESTS FAILED ✗'}", tag="DEBUG", level=1)
        log_message("="*80 + "\n", tag="DEBUG", level=1)
    
    return debug_results


def run_debug_diagnostics(sim_name, snapshot_it, data, size, nmax, up_to_level,
                         region_coords=None, debug_params=None, verbose=True,
                         dir_params=None, field_sources=None,
                         pipeline_debug_results=None, scan_pack=None,
                         use_siblings=True, nghost=1, clean_output=False):
    '''
    Coordinator function: runs all debug diagnostics based on DEBUG_PARAMS configuration.
    
    Args:
        sim_name: simulation name
        snapshot_it: snapshot iteration
        data: data dictionary from load_data()
        size: simulation box size
        nmax: base grid resolution
        up_to_level: AMR refinement level
        region_coords: region coordinates (if any)
        debug_params: DEBUG_PARAMS from config.py
        verbose: print debug output
        clean_output: if True, replace NaN/inf/outliers with zeros in visualizations.
                     if False (default), preserve raw data to diagnose errors.
        
    Returns:
        results: dictionary with all debug results
        
    Author: Marco Molina
    '''
    
    if debug_params is None:
        debug_params = {}
    
    results = {}
    pipeline_debug_results_local = pipeline_debug_results
    scan_pack_local = scan_pack
    
    # Buffer diagnostics
    if debug_params.get("buffer", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Buffer pipeline analysis")
            print("="*70)
            log_message(f"Running buffer diagnostics...", tag="buffer", level=0)
        results['buffer'] = run_debug_buffer_pipeline(
            data, size, nmax,
            nghost=debug_params.get("buffer", {}).get("nghost", 1),
            interpol=debug_params.get("buffer", {}).get("interpol", 'TSC'),
            use_siblings=use_siblings,
            bitformat=out_params.get("bitformat", np.float32),
            verbose=debug_params.get("buffer", {}).get("verbose", False)
        )
        if pipeline_debug_results_local is None:
            pipeline_debug_results_local = results['buffer']
            

    # Patch position diagnostics
    if debug_params.get("patch_analysis", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Patch position analysis")
            print("="*70)
        if dir_params is None:
            log_message("Patch analysis requested but dir_params is missing; skipping.", tag="patch", level=0)
        else:
            if verbose:
                log_message("Running patch position diagnostics...", tag="patch", level=0)
            results['patch_analysis'] = analyze_patch_positions(
                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                data['grid_pare'], data['grid_npatch'],
                dir_params=dir_params,
                suspicious_threshold=debug_params.get("patch_analysis", {}).get("suspicious_threshold", 15.0),
                verbose=debug_params.get("patch_analysis", {}).get("verbose", False)
            )
            

    # Divergence diagnostics
    # Uses buffer settings from IND_PARAMS
    if debug_params.get("divergence", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Divergence analysis")
            print("="*70)
        if dir_params is None:
            log_message("Divergence diagnostics requested but dir_params is missing; skipping.", tag="divergence", level=0)
        else:
            if verbose:
                log_message("Running divergence diagnostics...", tag="divergence", level=0)
            
            # Use buffer settings from IND_PARAMS
            use_buffer_analysis = ind_params.get("buffer", False)
            parent_mode_flag = ind_params.get("parent", False)
            interpol_method = ind_params.get("interpol", "TRILINEAR")
            parent_interpol_method = ind_params.get("parent_interpol", interpol_method)
            blend_method = ind_params.get("blend", False)
            stencil_size = ind_params.get("stencil", 3)
            
            results['divergence'] = compare_divergence_methods(
                data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                data['grid_npatch'], data['clus_kp'],
                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                data['grid_irr'], dir_params, size, nmax,
                use_buffer=use_buffer_analysis, nghost=nghost, interpol=interpol_method,
                parent_interpol=parent_interpol_method,
                use_siblings=use_siblings, blend=blend_method, parent_mode=parent_mode_flag,
                stencil=stencil_size,
                verbose=debug_params.get("divergence", {}).get("verbose", False)
            )
    
    # Ghost Cell Inspection (Buffer Diagnostics)
    # Uses interpol method from IND_PARAMS
    if debug_params.get("ghost_cell_inspection", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Ghost cell inspection")
            print("="*70)
            log_message(f"Running ghost cell inspection...", tag="ghost_cells", level=0)
        
        if not ind_params.get("buffer", False):
            log_message("Ghost cell inspection skipped (buffer disabled in IND_PARAMS).", tag="ghost_cells", level=0)
        else:
            # Use interpolation method from IND_PARAMS
            interpol_method = ind_params.get("interpol", "TSC")
            parent_interpol_method = ind_params.get("parent_interpol", interpol_method)
            parent_mode_flag = ind_params.get("parent", False)
            blend_method = ind_params.get("blend", False)
            try:
                ghost_results = diagnose_ghost_cell_values(
                    data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                    data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                    data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                    data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                    size, nmax, nghost=nghost, interpol=interpol_method,
                    parent_interpol=parent_interpol_method,
                    use_siblings=use_siblings,
                    parent_mode=parent_mode_flag,
                    blend=blend_method,
                    verbose=debug_params.get("ghost_cell_inspection", {}).get("verbose", False)
                )
                method_tag = parent_interpol_method if parent_mode_flag and not blend_method else interpol_method
                results[f'ghost_cells_{method_tag}'] = ghost_results
            except Exception as e:
                log_message(f"ERROR in ghost cell inspection ({interpol_method}): {e}", tag="ghost_cells", level=0)
    
    # Buffer vs Extrapolation Comparison
    # Uses interpol, stencil, and use_siblings from IND_PARAMS
    if debug_params.get("buffer_vs_extrapolation", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Buffer vs extrapolation comparison")
            print("="*70)
            log_message(f"Running buffer vs extrapolation comparison...", tag="buffer_comp", level=0)
        
        if not ind_params.get("buffer", False):
            log_message("Buffer vs extrapolation skipped (buffer disabled in IND_PARAMS).", tag="buffer_comp", level=0)
        else:
            # Use parameters from IND_PARAMS
            interpol_method = ind_params.get("interpol", "TSC")
            parent_interpol_method = ind_params.get("parent_interpol", interpol_method)
            stencil_size = ind_params.get("stencil", 3)
            blend_method = ind_params.get("blend", False)
            parent_mode_flag = ind_params.get("parent", False)
            
            try:
                buffer_comp = diagnose_buffer_vs_extrapolation(
                    data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                    data['grid_npatch'], data['clus_kp'],
                    data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                    data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                    data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                    data['grid_irr'], size, nmax, size/nmax,
                    nghost=nghost, interpol=interpol_method, stencil=stencil_size, blend=blend_method,
                    parent_interpol=parent_interpol_method,
                    parent_mode=parent_mode_flag,
                    use_siblings=use_siblings,
                    verbose=debug_params.get("buffer_vs_extrapolation", {}).get("verbose", False)
                )
                results['buffer_vs_extrapolation'] = buffer_comp
            except Exception as e:
                log_message(f"ERROR in buffer vs extrapolation comparison: {e}", tag="buffer_comp", level=0)
    
    # Divergence Spatial Distribution
    # Uses buffer settings from IND_PARAMS
    if debug_params.get("divergence_spatial", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Divergence spatial distribution")
            print("="*70)
            log_message(f"Running divergence spatial distribution analysis...", tag="spatial_div", level=0)
        
        try:
            dx = size / nmax
            use_buffer_analysis = ind_params.get("buffer", False)
            blend_method = ind_params.get("blend", False)
            parent_mode_flag = ind_params.get("parent", False)
            
            if use_buffer_analysis:
                # Calculate divergence WITH buffer, parent fill, or blend (following IND_PARAMS)
                interpol_method = ind_params.get("interpol", "TSC")
                parent_interpol_method = ind_params.get("parent_interpol", interpol_method)
                stencil_size = ind_params.get("stencil", 3)
                blend_boundary = 1 if stencil_size == 3 else 2
                parent_active = parent_mode_flag and not blend_method
                effective_nghost = 0 if parent_active else nghost
                if blend_method and effective_nghost == 0:
                    effective_nghost = blend_boundary
                
                if effective_nghost > 0:
                    # Add buffer
                    buffered = buff.add_ghost_buffer(
                        [data['clus_Bx'], data['clus_By'], data['clus_Bz']], data['grid_npatch'],
                        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                        data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                        data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                        size=size, nmax=nmax, nghost=effective_nghost,
                        interpol=interpol_method, use_siblings=use_siblings,
                        kept_patches=data['clus_kp']
                    )
                    Bx_buff, By_buff, Bz_buff = buffered
                    
                    # Calculate divergence on buffered fields
                    div_field_buffered = diff.divergence(Bx_buff, By_buff, Bz_buff, dx, data['grid_npatch'], 
                                                        data['clus_kp'], stencil=stencil_size)
                    
                    # Remove buffer from divergence result
                    div_field = buff.ghost_buffer_buster(div_field_buffered, 
                                                        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                                                        nghost=effective_nghost, kept_patches=data['clus_kp'])

                    if blend_method:
                        div_parent = diff.divergence(
                            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                            dx, data['grid_npatch'], data['clus_kp'], stencil=stencil_size
                        )
                        div_parent = buff.add_ghost_buffer(
                            [div_parent], data['grid_npatch'],
                            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                            size=size, nmax=nmax, nghost=0,
                            interpol=parent_interpol_method, use_siblings=False,
                            kept_patches=data['clus_kp']
                        )[0]
                        div_field = buff.blend_patch_boundaries(
                            div_field, div_parent,
                            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                            boundary_width=blend_boundary, kept_patches=data['clus_kp']
                        )
                else:
                    div_field = diff.divergence(
                        data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                        dx, data['grid_npatch'], data['clus_kp'], stencil=stencil_size
                    )
                    div_field = buff.add_ghost_buffer(
                        [div_field], data['grid_npatch'],
                        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                        data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
                        data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
                        size=size, nmax=nmax, nghost=0,
                        interpol=parent_interpol_method, use_siblings=False,
                        kept_patches=data['clus_kp']
                    )[0]
            else:
                # Calculate divergence WITHOUT buffer (extrapolation)
                stencil_size = ind_params.get("stencil", 3)
                div_field = diff.divergence(
                    data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                    dx, data['grid_npatch'], data['clus_kp'], stencil=stencil_size
                )
            
            spatial_results = diagnose_divergence_spatial_distribution(
                div_field, data['grid_npatch'], data['clus_kp'],
                data['grid_irr'], use_buffer=use_buffer_analysis,
                parent_mode=(use_buffer_analysis and parent_mode_flag and not blend_method),
                blend_mode=bool(use_buffer_analysis and blend_method),
                verbose=debug_params.get("divergence_spatial", {}).get("verbose", False)
            )
            results['divergence_spatial'] = spatial_results
        except Exception as e:
            log_message(f"ERROR in divergence spatial distribution: {e}", tag="spatial_div", level=0)
    
    
    # Visualization diagnostics (volume comparison)
    if debug_params.get("volume_analysis", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Volume analysis")
            print("="*70)
            log_message(f"Running volume analysis diagnostics...", tag="volume", level=0)
        
        region_box = None
        if region_coords and region_coords[0] == "box":
            region_box = region_coords[1:]
        
        meta = analyze_metadata(data, size, nmax, verbose=debug_params.get("volume_analysis", {}).get("verbose", False))
        vol_raw, vol_buf = compare_volumes(
            data, size, nmax, region_box=region_box, up_to_level=up_to_level, bitformat=out_params.get("bitformat", np.float32),
            verbose=debug_params.get("volume_analysis", {}).get("verbose", False),
            clean_output=clean_output,
            interp_mode=debug_params.get('unigrid_interp_mode', 'DIRECT'),
            buffer_level_check=debug_params.get('buffer_level_check', {}).get('enabled', False),
            check_cell_patch=debug_params.get('buffer_level_check', {}).get('check_cell_patch', False),
            use_siblings=use_siblings
        )
        analyze_spurious_band(vol_raw, vol_buf, size, nmax, up_to_level=up_to_level)
        analyze_nan_regions(vol_raw, vol_buf, size, nmax)
        find_refined_bands(vol_raw, vol_buf, size, nmax, up_to_level=up_to_level)
        
        results['volume'] = {
            'metadata': meta,
            'volumes': (vol_raw, vol_buf)
        }
        
    # Scan animation diagnostics
    if debug_params.get("scan_animation", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Scan animation")
            print("="*70)
            log_message("Running scan animation diagnostics...", tag="visual", level=0)
        scan_pack_local = build_scan_animation_data(
            data=data,
            size=size,
            nmax=nmax,
            region_coords=region_coords,
            nghost=debug_params.get("scan_animation", {}).get("nghost", nghost),
            use_siblings=use_siblings,
            up_to_level=up_to_level,
            bitformat=out_params.get("bitformat", np.float32),
            verbose=debug_params.get("scan_animation", {}).get("verbose", False),
            clean_output=clean_output
        )
        results['scan_animation'] = scan_pack_local
        

    # Field analysis diagnostics
    if debug_params.get("field_analysis", {}).get("enabled", False):
        if verbose:
            print("\n" + "="*70)
            print("DIAGNOSTIC: Field analysis")
            print("="*70)
        if field_sources is None:
            log_message("Field analysis requested but field_sources is missing; skipping.", tag="field_analysis", level=0)
        else:
            if verbose:
                log_message("Running field analysis diagnostics...", tag="field_analysis", level=0)
            field_list = debug_params.get("field_analysis", {}).get("field_list", None)
            results['field_analysis'] = analyze_debug_fields(
                field_sources=field_sources,
                region_coords=region_coords,
                data=data,
                debug_params=debug_params,
                up_to_level=up_to_level,
                size=size,
                nmax=nmax,
                pipeline_debug_results=pipeline_debug_results_local,
                scan_pack=scan_pack_local,
                verbose=debug_params.get("field_analysis", {}).get("verbose", False),
                it=snapshot_it,
                sims=sim_name,
                field_list=field_list
            )
    
    return results


def validate_scan_pack(scan_pack, verbose=True):
    '''
    Validates the integrity and sanity of the scan pack dictionary.
    
    Args:
        scan_pack: Dictionary returned from run_debug_diagnostics containing scan animation data
        verbose: Print diagnostic output
        
    Returns:
        tuple (is_valid, diagnostics) where:
            - is_valid: bool indicating if scan_pack is valid
            - diagnostics: dict with validation details
            
    Author: Marco Molina
    '''
    
    diagnostics = {
        'is_none': False,
        'missing_scan_volume': False,
        'scan_volume_shape': None,
        'scan_volume_dtype': None,
        'scan_volume_has_inf': False,
        'scan_volume_has_nan': False,
        'scan_volume_has_zero': False,
        'region_size_valid': True,
        'region_size': None,
        'amr_levels_valid': True,
        'amr_levels_shape': None,
        'volume_no_buffer_valid': True,
        'has_all_required_keys': True
    }
    
    if scan_pack is None:
        diagnostics['is_none'] = True
        if verbose:
            log_message("❌ Scan pack is None", tag="sanity", level=0)
        return False, diagnostics
    
    if not isinstance(scan_pack, dict):
        if verbose:
            log_message(f"❌ Scan pack is not a dict, got {type(scan_pack)}", tag="sanity", level=0)
        diagnostics['has_all_required_keys'] = False
        return False, diagnostics
    
    # Check required keys
    required_keys = ['_scan_volume']
    missing_keys = [k for k in required_keys if k not in scan_pack]
    if missing_keys:
        diagnostics['missing_scan_volume'] = True
        diagnostics['has_all_required_keys'] = False
        if verbose:
            log_message(f"❌ Scan pack missing required keys: {missing_keys}", tag="sanity", level=0)
        return False, diagnostics
    
    # Validate _scan_volume
    volume = scan_pack['_scan_volume']
    if not isinstance(volume, np.ndarray):
        if verbose:
            log_message(f"❌ _scan_volume is not numpy array, got {type(volume)}", tag="sanity", level=0)
        return False, diagnostics
    
    diagnostics['scan_volume_shape'] = volume.shape
    diagnostics['scan_volume_dtype'] = str(volume.dtype)
    
    if volume.size == 0:
        if verbose:
            log_message("❌ _scan_volume is empty", tag="sanity", level=0)
        return False, diagnostics
    
    if not np.issubdtype(volume.dtype, np.number):
        if verbose:
            log_message(f"❌ _scan_volume has non-numeric dtype: {volume.dtype}", tag="sanity", level=0)
        return False, diagnostics
    
    # Check for inf/nan/zero
    has_inf = np.any(np.isinf(volume))
    has_nan = np.any(np.isnan(volume))
    diagnostics['scan_volume_has_inf'] = has_inf
    diagnostics['scan_volume_has_nan'] = has_nan
    
    if has_inf:
        n_inf = np.sum(np.isinf(volume))
        if verbose:
            log_message(f"⚠️  _scan_volume has {n_inf}/{volume.size} inf values", tag="sanity", level=1)
    
    if has_nan:
        n_nan = np.sum(np.isnan(volume))
        if verbose:
            log_message(f"⚠️  _scan_volume has {n_nan}/{volume.size} NaN values", tag="sanity", level=1)
    
    # Check for significant zero content
    n_zero = np.sum(volume == 0)
    if n_zero > volume.size * 0.9:
        diagnostics['scan_volume_has_zero'] = True
        if verbose:
            log_message(f"⚠️  _scan_volume is {100*n_zero/volume.size:.1f}% zeros", tag="sanity", level=1)
    
    # Validate region_size
    region_size = scan_pack.get('_scan_region_size', None)
    if region_size is not None:
        diagnostics['region_size'] = region_size
        if not isinstance(region_size, (int, float, np.number)):
            diagnostics['region_size_valid'] = False
            if verbose:
                log_message(f"❌ _scan_region_size has invalid type: {type(region_size)}", tag="sanity", level=0)
            return False, diagnostics
        if region_size <= 0:
            diagnostics['region_size_valid'] = False
            if verbose:
                log_message(f"❌ _scan_region_size is non-positive: {region_size}", tag="sanity", level=0)
            return False, diagnostics
    
    # Validate amr_levels
    amr_levels = scan_pack.get('_scan_volume_levels', None)
    if amr_levels is not None:
        if isinstance(amr_levels, np.ndarray):
            diagnostics['amr_levels_shape'] = amr_levels.shape
            if amr_levels.size == 0:
                diagnostics['amr_levels_valid'] = False
                if verbose:
                    log_message("⚠️  _scan_volume_levels is empty", tag="sanity", level=1)
        else:
            diagnostics['amr_levels_valid'] = False
            if verbose:
                log_message(f"⚠️  _scan_volume_levels is not ndarray, got {type(amr_levels)}", tag="sanity", level=1)
    
    # Validate volume_no_buffer if present
    volume_no_buffer = scan_pack.get('_scan_volume_no_buffer', None)
    if volume_no_buffer is not None:
        if not isinstance(volume_no_buffer, np.ndarray):
            diagnostics['volume_no_buffer_valid'] = False
            if verbose:
                log_message(f"⚠️  _scan_volume_no_buffer is not ndarray, got {type(volume_no_buffer)}", tag="sanity", level=1)
        elif volume_no_buffer.shape != volume.shape:
            diagnostics['volume_no_buffer_valid'] = False
            if verbose:
                log_message(f"⚠️  _scan_volume_no_buffer shape {volume_no_buffer.shape} != _scan_volume shape {volume.shape}", 
                           tag="sanity", level=1)
    
    # Do not fail on NaN/inf/zero content; we still want to plot problematic data.
    is_valid = (
        diagnostics['has_all_required_keys'] and
        diagnostics['region_size_valid'] and
        diagnostics['amr_levels_valid'] and
        diagnostics['volume_no_buffer_valid']
    )
    
    if is_valid and verbose:
        log_message(
            f"✓ Scan pack valid (warnings allowed): shape={volume.shape}, dtype={volume.dtype}, "
            f"region_size={region_size}",
            tag="sanity",
            level=0
        )
    
    return is_valid, diagnostics


def main():
    """Main debug routine."""
    # Lazy import to avoid circular dependency
    from scripts.induction_evo import (
        load_data,
        find_most_massive_halo,
        create_region,
        vectorial_quantities,
        induction_equation,
        induction_equation_energy
    )

    print("\n" + "="*80)
    print("DEBUG TOOL")
    print("="*80)

    if len(sys.argv) < 3:
        print("Usage: python debug_buffer_visual.py <simulation_name> <snapshot_it>")
        print("Example: python debug_buffer_visual.py cluster_B_low_res_paper_2020 1050")
        sys.exit(1)

    sim_name = sys.argv[1]
    snapshot_it = int(sys.argv[2])

    # Resolve config-driven parameters
    sim_index = out_params["sims"].index(sim_name) if sim_name in out_params["sims"] else 0
    level = ind_params["level"][0] if isinstance(ind_params["level"], list) else ind_params["level"]
    up_to_level = ind_params["up_to_level"][0] if isinstance(ind_params["up_to_level"], list) else ind_params["up_to_level"]
    nmax = ind_params["nmax"][0] if isinstance(ind_params["nmax"], list) else ind_params["nmax"]
    size = ind_params["size"][0] if isinstance(ind_params["size"], list) else ind_params["size"]
    dir_params = out_params["dir_params"][sim_index]

    # Prepare terminal logging if enabled
    log_filepath = None
    if out_params.get("save_terminal", False):
        log_filename = build_terminal_log_filename(sim_name, snapshot_it, ind_params, out_params)
        terminal_folder = out_params.get("terminal_folder", "terminal_output/")
        # Ensure terminal_folder ends with /
        if not terminal_folder.endswith('/'):
            terminal_folder += '/'
        log_filepath = terminal_folder + log_filename
        print(f"\n[INFO] Terminal output will be saved to: {log_filepath}\n")
    
    # Build region if configured, otherwise use full box
    region_coords = [None]
    rad_val = None
    try:
        if ind_params.get("region") in ["BOX", "SPH"]:
            coords, rad = find_most_massive_halo(
                [sim_name],
                [snapshot_it],
                ind_params["a0"],
                out_params["dir_halos"],
                out_params["dir_grids"],
                out_params["data_folder"],
                vir_kind=ind_params.get("vir_kind", 1),
                rad_kind=ind_params.get("rad_kind", 1),
                verbose=out_params.get("verbose", False)
            )
            region_coords, _ = create_region(
                [sim_name],
                [snapshot_it],
                coords,
                rad,
                size=size,
                F=ind_params.get("F", 1.0),
                reg=ind_params.get("region"),
                verbose=out_params.get("verbose", False)
            )
            region_coords = region_coords[0]
            if rad:
                rad_val = rad[0] if isinstance(rad, (list, tuple)) else rad
    except Exception as exc:
        print(f"Warning: could not build region, using full box. Reason: {exc}")
        region_coords = [None]
        rad_val = None

    # If logging to file, redirect output
    if log_filepath:
        with redirect_output_to_file(log_filepath, verbose_console=True):
            _run_debug_main_logic(
                sim_name, snapshot_it, level, up_to_level, nmax, size, 
                dir_params, region_coords, rad_val
            )
    else:
        _run_debug_main_logic(
            sim_name, snapshot_it, level, up_to_level, nmax, size, 
            dir_params, region_coords, rad_val
        )


def _run_debug_main_logic(sim_name, snapshot_it, level, up_to_level, nmax, size, 
                         dir_params, region_coords, rad_val=None):
    '''
    Core debug logic extracted for use with or without output redirection.
    '''
    # Load data for the snapshot
    from scripts.induction_evo import (
        load_data,
        vectorial_quantities,
        induction_equation,
        induction_equation_energy
    )
    
    print(f"Loading data: sim={sim_name}, it={snapshot_it}")
    data = load_data(
        sim_name,
        snapshot_it,
        ind_params["a0"],
        ind_params["H0"],
        out_params["dir_grids"],
        out_params["dir_gas"],
        dir_params,
        level,
        test=ind_params["test_params"],
        bitformat=out_params["bitformat"],
        region=region_coords,
        verbose=out_params.get("verbose", False),
        debug=False
    )

    vectorial = None
    induction = None
    induction_energy = None
    field_sources = None

    if debug_params.get("field_analysis", {}).get("enabled", False):
        dx = size / nmax
        vectorial = vectorial_quantities(
            ind_params["components"],
            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
            data['clus_vx'], data['clus_vy'], data['clus_vz'],
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            dx, stencil=ind_params.get("stencil", 3),
            verbose=out_params.get("verbose", False)
        )
        induction, _ = induction_equation(
            ind_params["components"],
            vectorial,
            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
            data['clus_vx'], data['clus_vy'], data['clus_vz'],
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            data['H'], data['a'],
            mag=ind_params.get("mag", False),
            verbose=out_params.get("verbose", False)
        )
        induction_energy = induction_equation_energy(
            ind_params["components"],
            induction,
            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
            data['clus_rho_rho_b'], data['clus_v2'],
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            verbose=out_params.get("verbose", False)
        )
        field_sources = [data, vectorial, induction, induction_energy]

    debug_results = run_debug_diagnostics(
        sim_name=sim_name,
        snapshot_it=snapshot_it,
        data=data,
        size=size,
        nmax=nmax,
        up_to_level=up_to_level,
        region_coords=region_coords,
        debug_params=debug_params,
        verbose=out_params.get("verbose", False),
        dir_params=dir_params,
        field_sources=field_sources,
        use_siblings=ind_params.get("use_siblings", True),
        nghost=ind_params.get("nghost", 1),
        clean_output=debug_params.get("clean_output", False)
    )

    debug_fields = debug_results.get("field_analysis")
    if debug_params.get("field_analysis", {}).get("enabled", False) and ind_params["components"].get("divergence", False) and debug_fields:
        levels = utils.create_vector_levels(data['grid_npatch'])
        resolution = (size / nmax) / (2 ** levels)
        inv_resolution = 1.0 / np.array(resolution)
        local_ind_params = ind_params.copy()
        local_ind_params["up_to_level"] = up_to_level

        grid_time = np.array([data['grid_time']])
        grid_zeta = np.array([data['grid_zeta']])
        if rad_val is None:
            rad_val = ind_params.get("rmin", 1.0)
            if isinstance(rad_val, (list, tuple)):
                rad_val = rad_val[0] if rad_val else 1.0

        quantities = debug_params.get("field_analysis", {}).get("quantities", [])
        for field_key, ref_key, ref_scale_val, quantity in zip(
            ['clus_B2', 'diver_B', 'MIE_diver_x', 'MIE_diver_y', 'MIE_diver_z', 'MIE_diver_B2'],
            ['clus_B2', 'clus_B', 'clus_Bx', 'clus_By', 'clus_Bz', 'clus_B2'],
            [1.0, inv_resolution, 1.0, 1.0, 1.0, 1.0],
            quantities
        ):
            if field_key not in debug_fields or ref_key not in debug_fields:
                continue
            distribution_check(
                [debug_fields[field_key]],
                quantity,
                debug_params,
                local_ind_params,
                grid_time,
                grid_zeta,
                rad_val,
                ref_field=[debug_fields[ref_key]],
                ref_scale=ref_scale_val,
                clean=debug_params.get("field_analysis", {}).get("clean_field", False),
                verbose=out_params.get("verbose", False),
                save=out_params.get("save", False),
                folder=out_params.get("image_folder", None)
            )

    scan_pack = debug_results.get("scan_animation")
    if debug_params.get("scan_animation", {}).get("enabled", False) and scan_pack:
        # Validate scan pack before processing, but always plot even if warnings exist.
        is_valid, scan_diagnostics = validate_scan_pack(scan_pack, verbose=out_params.get("verbose", True))

        if not is_valid:
            log_message("Scan pack has structural issues; plotting anyway for diagnostics.", tag="sanity", level=0)
        if out_params.get("verbose", False):
            log_message(f"Diagnostics: {scan_diagnostics}", tag="DEBUG", level=1)

        if '_scan_volume' in scan_pack:
            volume = scan_pack['_scan_volume']
            volume_no_buffer = scan_pack.get('_scan_volume_no_buffer', None)
            region_size = scan_pack.get('_scan_region_size', size)
            amr_levels = scan_pack.get('_scan_volume_levels', np.array([]))

            ind_meta = ind_params.copy()
            ind_meta['sim'] = sim_name
            ind_meta['it'] = snapshot_it
            ind_meta['zeta'] = data['grid_zeta']
            ind_meta['time'] = data['grid_time']
            ind_meta['level'] = level
            ind_meta['up_to_level'] = up_to_level
            ind_meta['buffer'] = True
            ind_meta['interpol'] = 'NEAREST'
            ind_meta['use_siblings'] = ind_params.get("use_siblings", True)
            ind_meta['stencil'] = ind_params.get("stencil", 3)

            scan_plot_params_with_amr = scan_plot_params.copy()
            scan_plot_params_with_amr['amr_levels'] = amr_levels
            scan_plot_params_with_amr['title'] = f"Buffer Assignment Scan {'Cleaned' if debug_params.get('clean_output', False) else 'Raw'}"  # Indicate if cleaned or raw

            scan_animation_3D(
                volume,
                region_size,
                scan_plot_params_with_amr,
                ind_meta,
                volume_params={'vol_idx': 0, 'reg_idx': 0},
                verbose=debug_params.get("scan_animation", {}).get("verbose", False),
                save=debug_params.get("scan_animation", {}).get("save", False),
                folder=out_params.get("image_folder", None)
            )

            if volume_no_buffer is not None:
                ind_meta_no_buffer = ind_meta.copy()
                ind_meta_no_buffer['buffer'] = False
                scan_animation_3D(
                    volume_no_buffer,
                    region_size,
                    scan_plot_params_with_amr,
                    ind_meta_no_buffer,
                    volume_params={'vol_idx': 0, 'reg_idx': 0},
                    verbose=debug_params.get("scan_animation", {}).get("verbose", False),
                    save=debug_params.get("scan_animation", {}).get("save", False),
                    folder=out_params.get("image_folder", None)
                )


if __name__ == "__main__":
    main()
