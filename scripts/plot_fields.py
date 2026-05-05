"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

plot_fields module
Contains functions to plot the magnetic field induction components or any other interesting quantities.

Created by Marco Molina Pradillo
"""

import numpy as np
import gc
import os
from datetime import datetime
import scripts.diff as diff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm, to_rgb
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from scipy import fft
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from . import spectral
import multiprocessing
import time
from time import strftime
from time import gmtime
import sys
import hashlib
from scripts.units import *
from scripts.utils import log_message

DEFAULT_PLOT_PALETTE = {
    'measured_energy': '#1f77b4',
    'induction_itemized': '#ff7f0e',
    'induction_compact': '#800020',
    'kinetic_energy': '#17becf',
    'production': '#2ca02c',
    'dissipation': '#d62728',
    'net_itemized': '#ff7f0e',
    'net_compact': '#800020',
    'efficiency': '#1f77b4',
    'density': '#2ca02c',
    'max_curve': '#111111',
    'negative_interval': '#364243',
    'percentile_cmap': 'viridis',
    'component_colors': {
        'compression': '#9467bd',
        'stretching': '#ff9896',
        'advection': '#e377c2',
        'divergence': '#c5b0d5',
        'drag': '#7f7f7f'
    }
}


def get_plot_palette(plot_params=None, induction_params=None):
    """Resolve the active plot palette from plot parameters or defaults."""
    plot_params = plot_params or {}
    induction_params = induction_params or {}

    palettes = plot_params.get('palettes') or induction_params.get('palettes') or {}
    palette_name = plot_params.get('palette_name', induction_params.get('palette_name', 'classic'))

    palette = None
    if isinstance(palettes, dict) and palettes:
        palette = palettes.get(palette_name, palettes.get('classic', next(iter(palettes.values()))))
    if not isinstance(palette, dict):
        palette = DEFAULT_PLOT_PALETTE

    resolved = DEFAULT_PLOT_PALETTE.copy()
    resolved.update({k: v for k, v in palette.items() if k != 'component_colors'})

    component_colors = DEFAULT_PLOT_PALETTE['component_colors'].copy()
    component_colors.update(palette.get('component_colors', {}))
    resolved['component_colors'] = component_colors
    return resolved


def align_cumulative_overlay_zero(ax, ax_aux, y_aux_max=None, headroom=0.05):
    """
    Align y=0 horizontally between a primary axis and a cumulative-overlay twin axis.

    The left axis defines the zero proportion. The right axis is then forced so that
    its minimum matches that same proportion, using the supplied cumulative maximum
    plus a configurable headroom.
    """
    try:
        y1_min, y1_max = ax.get_ylim()
        if not np.isfinite(y1_min) or not np.isfinite(y1_max) or y1_max == y1_min:
            return

        p = (0.0 - y1_min) / (y1_max - y1_min)
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

        if y_aux_max is None or not np.isfinite(y_aux_max):
            _, y2_max_current = ax_aux.get_ylim()
            y_aux_max = y2_max_current

        try:
            headroom = float(headroom)
        except (TypeError, ValueError):
            headroom = 0.05
        if not np.isfinite(headroom) or headroom < 0.0:
            headroom = 0.05

        y_aux_max = (1.0 + headroom) * float(y_aux_max)
        if y_aux_max <= 0.0:
            _, y2_max_current = ax_aux.get_ylim()
            y_aux_max = float(max(y2_max_current, 1e-30))

        y2_min_forced = -(p / (1.0 - p)) * y_aux_max
        ax_aux.set_ylim(y2_min_forced, y_aux_max)
        ax_aux.set_autoscale_on(False)
    except Exception:
        pass

def safe_filename(filepath, max_length=255, verbose=False):
    """
    Ensures the filename doesn't exceed the filesystem limit by shortening it intelligently.
    
    Args:
        - filepath: Full path to the file
        - max_length: Maximum allowed length for the filename (default 255 for most filesystems)
        - verbose: Print information about filename shortening
    
    Returns:
        - Safe filepath with shortened filename if necessary
    
    Author: Marco Molina
    """
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # If filename is within limit, return as is
    if len(filename) <= max_length:
        return filepath
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Calculate how much we need to shorten
    # Reserve space for extension, underscore, and 8-char hash
    available_length = max_length - len(ext) - 9
    
    if available_length < 20:
        # If still too long, use a very short name with hash
        hash_obj = hashlib.md5(name.encode())
        short_hash = hash_obj.hexdigest()[:16]
        shortened_name = f"plot_{short_hash}"
    else:
        # Keep the beginning (important info like run name) and add hash at the end
        # Try to preserve the first ~60% of available space for the start
        keep_start = int(available_length * 0.6)
        
        # Add a hash to maintain uniqueness
        hash_obj = hashlib.md5(name.encode())
        short_hash = hash_obj.hexdigest()[:8]
        
        shortened_name = f"{name[:keep_start]}_{short_hash}"
    
    new_filename = shortened_name + ext
    new_filepath = os.path.join(directory, new_filename)
    
    if verbose:
        log_message(
            f"Filename too long ({len(filename)} chars). Shortened to {len(new_filename)} chars.",
            tag="plot",
            level=1,
        )
        log_message(f"Original: {filename}", tag="plot", level=2)
        log_message(f"Shortened: {new_filename}", tag="plot", level=2)
    
    return new_filepath
        
def zoom_animation_3D(arr, size, arrow_scale = 1, units = 'Mpc', title = 'Magnetic Field Seed Zoom', verbose = True, Save = False, DPI = 300, run = '_', folder = None):
    '''
    Generates an animation of the magnetic field seed in 3D with a zoom effect. Can be used for any other 3D spacial field.
    
    Args:
        - arr: 3D array to animate
        - size: size of the array in Mpc in the x direction
        - arrow_scale: scale of the arrow in the provided units
        - units: units of the arrow scale. Can be 'Mpc' or 'kpc'
        - title: title of the animation
        - verbose: boolean to print the progress of the function
        - Save: boolean to save the animation or not
        - DPI: dots per inch in the animation
        - run: name of the run
        - folder: folder to save the animation
        
    Returns:
        - gif file with the animation
        
    Author: Marco Molina
    '''
    
    # Ensure the array is 3D
    assert arr.ndim == 3, "Input array must be 3D"
    assert arrow_scale > 0, "Arrow scale must be a positive integer"
    assert units in ['Mpc', 'kpc'], "Units must be 'Mpc' or 'kpc'"
    
    nmax, nmay, nmaz = arr.shape
    
    dx = size / nmax  # Cell size in Mpc
    
    inter = 200
    depth = 10
    col = 'red'
    
    for m in range(1, nmax//2):
        max_imdim = np.round((arrow_scale+m)/dx, 0).astype(int)
        if max_imdim <= nmax//2:
            max_frame = m
        else:
            break
    
    fig = plt.figure(figsize=(5, 5))
    
    if units == 'Mpc':
        ctou = arrow_scale/dx
    elif units == 'kpc':
        ctou = arrow_scale/(dx * 1000)
    
    def animate(frame):
        plt.clf()
        imdim = np.round((frame+arrow_scale)/dx, 0).astype(int)
        section = np.sum(arr[(nmax//2 - imdim):(nmax//2 + imdim), (nmay//2 - imdim):(nmay//2 + imdim), (nmaz//2 - depth//2):(nmaz//2 + depth//2)], axis=2)
        plt.imshow(section, cmap='viridis')
        plt.title(title)
        plt.arrow(imdim, imdim, ctou, 0, head_width=(ctou/14), head_length=(ctou/7), fc=col, ec=col)
        plt.text(imdim, imdim-arrow_scale, f'{arrow_scale} {units}', color=col)
        plt.xlabel('x cells')
        plt.ylabel('y cells')

    ani = FuncAnimation(fig, animate, frames = range(1, max_frame), interval=inter)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ani.save(data_dir + '/animation.gif', writer='pillow')
    
    if verbose == True:
        log_message('Magnetic Field Seed Zoom Animation computed', tag='plot', level=1)
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
    
        file_title = ' '.join(title.split()[:4])
        ani.save(folder + f'/{run}_{file_title}_zoom.gif', writer='pillow', dpi = DPI)
        
    return ani
        
def scan_animation_3D(arr, size, plot_params, induction_params, volume_params=None, verbose=True, save=False, folder=None):
    '''
    Generates an animation of the field in 3D with a scan effect.
    
    Args:
        - arr: 3D array to animate
        - size: size of the array in Mpc
        - plot_params: dict with study_box, depth, arrow_scale, units, interval, title, dpi, run, and optional 'amr_levels'
        - induction_params: dict with sim, it, zeta, time, level, up_to_level, buffer, interpol, stencil
        - volume_params: dict with vol_idx, reg_idx (for multi-volume scans)
        - verbose: boolean to print progress
        - save: boolean to save animation
        - folder: folder to save animation
        
    Returns:
        - gif file with the animation
        
    Author: Marco Molina
    '''
    
    assert arr.ndim == 3, "Input array must be 3D"

    # Extract plot parameters
    study_box = plot_params.get('study_box', 1.0)
    depth = plot_params.get('depth', 2)
    arrow_scale = plot_params.get('arrow_scale', 1.0)
    units = plot_params.get('units', 'Mpc')
    interval = plot_params.get('interval', 100)
    cmap = plot_params.get('cmap', 'viridis')
    projection_mode = plot_params.get('projection_mode', 'max')
    dpi = plot_params.get('dpi', 300)
    base_title = plot_params.get('title', 'Field Scan')
    run = plot_params.get('run', '_')
    
    # Extract optional AMR levels for colorbar customization (debug mode only)
    amr_levels_array = plot_params.get('amr_levels', None)
    max_level = int(np.max(amr_levels_array)) if amr_levels_array is not None and len(amr_levels_array) > 0 else None

    # Extract induction parameters for metadata
    sim = induction_params.get('sim', 'unknown')
    it = induction_params.get('it', 0)
    zeta = induction_params.get('zeta', 0.0)
    time = induction_params.get('time', 0.0)
    level = induction_params.get('level', 0)
    up_to_level = induction_params.get('up_to_level', level)
    diff_cfg = induction_params.get('differentiation', {})
    buffer = diff_cfg.get('buffer', True)
    interpol = diff_cfg.get('interpol', 'TSC')
    stencil = diff_cfg.get('stencil', 3)

    # Extract volume parameters (if scanning multiple volumes)
    vol_idx = volume_params.get('vol_idx', 0) if volume_params else 0
    reg_idx = volume_params.get('reg_idx', 0) if volume_params else 0

    # Build full title with metadata
    full_title = f"{base_title} - z: {zeta:.2f}"

    assert 0 < study_box <= 1, "Study box must be a float in (0, 1]"
    assert depth > 0, "Depth must be a positive integer"
    assert arrow_scale > 0, "Arrow scale must be a positive number"
    assert units in ['Mpc', 'kpc'], "Units must be 'Mpc' or 'kpc'"
    
    nmax, nmay, nmaz = arr.shape
    dx = size / nmax  # Cell size in Mpc
    inter = interval
    x_lsize = round(nmax//2 - nmax*study_box//2)
    x_dsize = round(nmax//2 + nmax*study_box//2)
    y_lsize = round(nmay//2 - nmay*study_box//2)
    y_dsize = round(nmay//2 + nmay*study_box//2)
    new_nmax = x_dsize - x_lsize
    col = 'red'

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.12, right=0.80, bottom=0.12, top=0.92)

    # Find the minimum and maximum values of the field among all the studied volume
    # Include all values (including 0) to properly capture the full range
    all_values = []
    for i in range(nmaz):
        frame_data = arr[x_lsize:x_dsize, y_lsize:y_dsize, i]
        all_values.extend(frame_data.flatten())  # Include ALL values, including 0

    all_values = np.array(all_values)
    # Filter out NaN/inf but keep zeros
    all_values = all_values[np.isfinite(all_values)]
    
    if all_values.size > 0:
        min_value = np.percentile(all_values, 1)    # 1st percentile
        max_value = np.percentile(all_values, 99.9) # 99.9th percentile
    else:
        min_value = 0
        max_value = 1
    
    # Ensure min < max
    if min_value >= max_value:
        if min_value <= 0:
            min_value = 0  # Allow 0 as minimum for AMR visualization
        if min_value >= max_value:
            max_value = min_value + 0.01

    # Choose normalization: BoundaryNorm for discrete AMR levels, LogNorm otherwise
    if amr_levels_array is not None and len(amr_levels_array) > 0:
        # Discrete AMR level visualization: values are simply 0, 1, 2, 3, ...
        max_level = int(np.max(amr_levels_array))
        # Create boundaries: [0, 1), [1, 2), [2, 3), ...
        boundaries = [float(i) for i in range(max_level + 2)]
        # BoundaryNorm for discrete color levels with explicit boundaries
        norm = BoundaryNorm(boundaries=boundaries, ncolors=256)
    else:
        # Logarithmic normalization for continuous field intensity
        norm = LogNorm(vmin=min_value, vmax=max_value)
    
    # Calculate arrow scale conversion
    if units == 'Mpc':
        ctou = arrow_scale / dx
    elif units == 'kpc':
        ctou = arrow_scale / (dx * 1000)

    # Create the initial image and colorbar (will be reused)
    im = None
    cbar = None
    
    def animate(frame):
        nonlocal im, cbar
        
        # Calculate depth slice bounds with boundary protection
        z_start = max(0, frame - depth//2)
        z_end = min(nmaz, frame + depth//2)
        
        # Ensure valid slice (at least 1 element)
        if z_end <= z_start:
            z_end = z_start + 1
        
        # Project along depth direction for this frame
        if projection_mode == 'min':
            section = np.min(arr[x_lsize:x_dsize, y_lsize:y_dsize, z_start:z_end], axis=2)
        elif projection_mode == 'sum':
            section = np.sum(arr[x_lsize:x_dsize, y_lsize:y_dsize, z_start:z_end], axis=2)
        else:
            # Default to max for discrete AMR levels
            section = np.max(arr[x_lsize:x_dsize, y_lsize:y_dsize, z_start:z_end], axis=2)

        # Snap to integer AMR levels and clip to valid range to avoid float speckles
        if amr_levels_array is not None and len(amr_levels_array) > 0 and max_level is not None:
            section = np.rint(section).astype(np.int16)
            section = np.clip(section, 0, max_level)
        
        # On first frame, create image and colorbar
        if im is None:
            # Transpose section: imshow expects [y, x], but our section is [x, y]
            # Also flip y axis to match physical coordinates (y from -y_max to +y_max)
            im = ax.imshow(section.T[::-1, :], cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
            cbar = fig.colorbar(im, ax=ax, fraction=0.039, pad=0.05, label='AMR Level')
            
            # Customize colorbar ticks to show AMR levels if available
            if amr_levels_array is not None and len(amr_levels_array) > 0:
                # For AMR visualization, show discrete level labels (L0, L1, L2, ...)
                max_level_ticks = int(np.max(amr_levels_array))
                tick_positions = []
                tick_labels = []
                
                # Simple color scheme: color value = level (0, 1, 2, ...)
                # Position ticks at the center of each discrete color bin
                for lvl in range(max_level_ticks + 1):
                    tick_positions.append(float(lvl) + 0.5)  # Center of bin [lvl, lvl+1)
                    tick_labels.append(f'L{lvl}')
                
                if tick_positions:
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels(tick_labels)
        else:
            # Update image data without recreating - apply same transpose and flip
            im.set_data(section.T[::-1, :])
        
        ax.set_title(full_title, fontsize=12, fontweight='bold')
        
        # Draw reference arrow (bottom-left for consistent placement after transpose)
        arrow_x = int(new_nmax * 0.08)
        arrow_y = int(new_nmax * 0.08)
        ax.arrow(arrow_x, arrow_y, ctou, 0,
            head_width=(ctou/14), head_length=(ctou/7), fc=col, ec=col)
        
        # Add arrow label
        text_x = arrow_x + ctou / 2
        text_y = arrow_y + 0.03 * new_nmax
        ax.text(text_x, text_y, f'{arrow_scale} {units}', color=col, ha='center', va='bottom', fontsize=10)
        
        # Set axis labels with physical units
        x_extent = study_box * size
        y_extent = study_box * size
        x_coords = np.linspace(-x_extent/2, x_extent/2, new_nmax)
        y_coords = np.linspace(-y_extent/2, y_extent/2, new_nmax)
        
        # Set ticks and labels in Mpc
        n_ticks = 5
        x_tick_indices = np.linspace(0, new_nmax-1, n_ticks, dtype=int)
        y_tick_indices = np.linspace(0, new_nmax-1, n_ticks, dtype=int)
        
        x_tick_labels = [f'{x_coords[i]:.1f}' for i in x_tick_indices]
        y_tick_labels = [f'{y_coords[i]:.1f}' for i in y_tick_indices]
        
        ax.set_xticks(x_tick_indices)
        ax.set_yticks(y_tick_indices)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)
        ax.tick_params(labelsize=9)
        
        ax.set_xlabel(f'x (Mpc)', fontsize=11)
        ax.set_ylabel(f'y (Mpc)', fontsize=11)

    ani = FuncAnimation(fig, animate, frames=range(nmaz), interval=inter)

    if verbose:
        log_message('Field Scan Animation computed', tag='plot', level=1)

    if save:
        if folder is None:
            folder = os.getcwd()
        os.makedirs(folder, exist_ok=True)
        # Standardized filename: include z/time to avoid overwrites across snaps
        title_slug = '_'.join(base_title.split())
        sim_info = f"L{induction_params.get('up_to_level','')}_{induction_params.get('F','')}_{induction_params.get('vir_kind','')}vir_{induction_params.get('rad_kind','')}rad_{induction_params.get('region','None')}Region"
        
        # Buffer info: distinguish between buffered, no-buffer (test), and data-only (no buffer applied)
        diff_cfg = induction_params.get('differentiation', {})
        buffer_flag = diff_cfg.get('buffer', False)
        if 'buffer' in diff_cfg and not buffer_flag:
            # Explicitly marked as no-buffer for testing
            buffer_info = 'RawLevels_NoBuf'
        elif buffer_flag:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol',''))
            buffer_info = f"Buffered_{diff_cfg.get('interpol','')}_siblings_{diff_cfg.get('use_siblings', False)}"
            if parent_flag:
                buffer_info += f"_parent_{parent_interpol}"
        else:
            buffer_info = 'NoBuffer'
        
        z_info = f"z{zeta:.3f}"
        proj_info = f"proj_{projection_mode}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run}_{title_slug}_{sim_info}_{buffer_info}_{diff_cfg.get('stencil','')}_{proj_info}_{z_info}_{timestamp}.gif"
        filepath = os.path.join(folder, filename)
        ani.save(filepath, writer='pillow', dpi=dpi)
        if verbose:
            log_message(f'Scan animation saved to {filepath}', tag='plot', level=1)


    return ani


def setup_axis(ax, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis, evolution_type, font):
        '''
        Helper function to set up axis properties
        
        Args:
            - ax: matplotlib axis object
            - x_scale: 'lin' or 'log' for x axis scale
            - y_scale: 'lin' or 'log' for y axis scale
            - xlim: [xlimo, xlimf] or None for auto
            - ylim: [ylimo, ylimf] or None for auto
            - cancel_limits: if True, ignore manual xlim/ylim; zeta axis inversion is handled by caller
            - x_axis: 'zeta' or 'years'
            - evolution_type: 'total' or 'differential'
            - font: font properties for labels
            
        Returns:
            - None (modifies ax in place)
            
        Author: Marco Molina
        '''

        if x_axis == 'years':
            ax.set_xlabel('Time (yr)', fontproperties=font)
            if evolution_type == 'total':
                ax.set_ylabel('Magnetic Energy (erg)', fontproperties=font)
            else:
                ax.set_ylabel('Magnetic Energy Induction (erg/s)', fontproperties=font)
        else:  # zeta
            ax.set_xlabel('Redshift (z)', fontproperties=font)
            if evolution_type == 'total':
                ax.set_ylabel('Magnetic Energy (erg)', fontproperties=font)
            else:
                ax.set_ylabel('Magnetic Energy Induction (erg/s)', fontproperties=font)
        
        if not cancel_limits and xlim:
            ax.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax.set_ylim(ylim[0], ylim[1])
            
        if x_scale == 'log':
            ax.set_xscale('log')
            if x_axis == 'years':
                ax.set_xlabel('Time log[yr]', fontproperties=font)
            else:
                ax.set_xlabel('Redshift log[z]', fontproperties=font)
                
        if y_scale == 'log':
            ax.set_yscale('log')
            if evolution_type == 'total':
                ax.set_ylabel('Magnetic Energy log[erg]', fontproperties=font)
            else:
                ax.set_ylabel('Magnetic Energy Induction log[erg/s]', fontproperties=font)
                
def should_plot_component(data, threshold=1e-30):
        '''
        Check if component has any non-zero values worth plotting
        
        Args:
            - data: array-like data of the component
            - threshold: minimum absolute value to consider for plotting
            
        Returns:
            - bool indicating if the component should be plotted
            
        Author: Marco Molina
        '''
        arr = np.asarray(data, dtype=float)
        # Ignore NaN/inf when deciding if a curve has meaningful values.
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return False
        return np.any(np.abs(arr) > threshold)


def plot_percentile_evolution(percentile_data, plot_params, induction_params,
                            grid_t, grid_zeta,
                            verbose=True, save=False, folder=None):
    '''
    Plot the evolution of precomputed percentile thresholds over time or redshift.

    Args:
        - percentile_data: dict with accumulated data from multiple snapshots:
            * 'percentiles': list of 2D arrays (n_levels,) per snapshot
            * 'levels': list containing the same percentile levels (or single array)
            * optional 'global_min': list of 1D arrays per snapshot
            * optional 'global_max': list of 1D arrays per snapshot
        - plot_params: dict with plotting options:
            * x_axis: 'zeta' or 'years'
            * x_scale, y_scale: 'lin' or 'log'
            * xlim, ylim: [min, max] or None
            * figure_size: [width, height]
            * line_widths: [percentile_lines, max_line]
            * alpha_fill: transparency for shaded bands
            * title: plot title
            * dpi: dots per inch
            * run: identifier for filenames
        - induction_params: dict with metadata (used for file naming)
        - grid_t: time grid
        - grid_zeta: redshift grid
        - verbose: bool
        - save: bool
        - folder: path to save plots

    Returns:
        - matplotlib Figure or None if no data
        
    Author: Marco Molina
    '''

    # Extract and combine data from list format to array format
    pct_list = percentile_data.get('percentiles', [])
    levels_list = percentile_data.get('levels', [])
    pct_plus_list = percentile_data.get('percentiles_plus', [])
    pct_minus_list = percentile_data.get('percentiles_minus', [])
    
    if not pct_list or not levels_list:
        if verbose:
            log_message('Percentile evolution: no percentile data to plot', tag='percentiles', level=1)
        return None
    
    # Determine reference levels from first non-empty entry
    ref_levels = None
    for lv in (levels_list if isinstance(levels_list, (list, tuple)) else [levels_list]):
        if isinstance(lv, (list, np.ndarray)) and len(lv) > 0:
            ref_levels = np.asarray(lv, dtype=float)
            break
    if ref_levels is None:
        if verbose:
            log_message('Percentile evolution: no valid levels found', tag='percentiles', level=1)
        return None

    # Build a list of valid indices where percentiles exist and align their order to ref_levels when possible
    valid_indices = []
    aligned_pct = []
    aligned_plus = [] if pct_plus_list else None
    aligned_minus = [] if pct_minus_list else None

    # Helper to align order based on provided per-snapshot levels
    def align_to_ref(values, snap_levels):
        vals = np.asarray(values, dtype=float)
        if snap_levels is None:
            return vals
        sl = np.asarray(snap_levels, dtype=float)
        # If identical order/values, return fast
        if vals.size == ref_levels.size and np.array_equal(sl, ref_levels):
            return vals
        # Map snapshot levels to ref order
        order = []
        for r in ref_levels:
            # find index of r in sl
            idx = np.where(sl == r)[0]
            if idx.size == 0:
                return None  # cannot align
            order.append(int(idx[0]))
        return vals[order]

    for k in range(len(pct_list)):
        p = pct_list[k]
        if p is None:
            continue
        # Determine snapshot levels for potential reordering
        snap_levels = None
        if isinstance(levels_list, (list, tuple)) and k < len(levels_list) and isinstance(levels_list[k], (list, np.ndarray)):
            snap_levels = levels_list[k]
        p_aligned = align_to_ref(p, snap_levels)
        if p_aligned is None or p_aligned.size != ref_levels.size:
            continue
        valid_indices.append(k)
        aligned_pct.append(p_aligned)
        if aligned_plus is not None and k < len(pct_plus_list) and pct_plus_list[k] is not None:
            plus_aligned = align_to_ref(pct_plus_list[k], snap_levels)
            aligned_plus.append(plus_aligned if plus_aligned is not None else np.full_like(p_aligned, np.nan))
        if aligned_minus is not None and k < len(pct_minus_list) and pct_minus_list[k] is not None:
            minus_aligned = align_to_ref(pct_minus_list[k], snap_levels)
            aligned_minus.append(minus_aligned if minus_aligned is not None else np.full_like(p_aligned, np.nan))

    if not aligned_pct:
        if verbose:
            log_message('Percentile evolution: no valid percentile rows to plot', tag='percentiles', level=1)
        return None

    # Stack to 2D arrays (n_snap_valid, n_levels)
    pct = np.vstack(aligned_pct)
    pct_plus = np.vstack(aligned_plus) if aligned_plus is not None and len(aligned_plus) == len(aligned_pct) else None
    pct_minus = np.vstack(aligned_minus) if aligned_minus is not None and len(aligned_minus) == len(aligned_pct) else None
    levels = ref_levels

    if pct.size == 0 or levels.size == 0:
        if verbose:
            log_message('Percentile evolution: no valid percentile data to plot', tag='percentiles', level=1)
        return None

    if pct.ndim != 2:
        raise ValueError(f'percentiles must form a 2D array; got shape {pct.shape}')

    n_snap, n_levels = pct.shape
    if levels.size != n_levels:
        raise ValueError(f'levels length ({levels.size}) must match percentile columns ({n_levels})')

    # Prepare x-axis (plots all snapshots like plot_integral_evolution)
    x_axis = plot_params.get('x_axis', 'zeta')
    if x_axis == 'years':
        x = np.array([grid_t[i] * time_to_yr / 1e9 for i in valid_indices], dtype=float)  # Convert to Gyr
        xlabel = 'Time (Gyr)'
    else:
        x = np.array([grid_zeta[i] for i in valid_indices], dtype=float)
        if x.size and x[-1] < 0:
            x[-1] = abs(x[-1])
        xlabel = 'Redshift (z)'

    # Sort levels ascending for consistent shading
    sort_idx = np.argsort(levels)
    levels_sorted = levels[sort_idx]
    pct_sorted = pct[:, sort_idx]
    
    # Sort error band arrays using the same indices
    if pct_plus is not None:
        pct_plus_sorted = pct_plus[:, sort_idx]
    else:
        pct_plus_sorted = None
    if pct_minus is not None:
        pct_minus_sorted = pct_minus[:, sort_idx]
    else:
        pct_minus_sorted = None

    # Matplotlib styling
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18
    })

    font = FontProperties(); font.set_size(12)
    font_title = FontProperties(); font_title.set_style('normal'); font_title.set_weight('bold'); font_title.set_size(18)
    font_legend = FontProperties(); font_legend.set_size(12)

    x_scale = plot_params.get('x_scale', 'lin')
    y_scale = plot_params.get('y_scale', 'log')
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    figure_size = plot_params.get('figure_size', [12, 6])
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    base_title = plot_params.get('title', 'Percentile Threshold Evolution')
    line_widths = plot_params.get('line_widths', [2.0, 1.5])
    alpha_fill = plot_params.get('alpha_fill', 0.20)
    palette = get_plot_palette(plot_params, induction_params)
    color_max_curve = palette.get('max_curve', DEFAULT_PLOT_PALETTE['max_curve'])
    
    # Read boundary exclusion parameters from plot_params
    exclude_boundaries = plot_params.get('exclude_boundaries', False)
    boundary_width = plot_params.get('boundary_width', 1)
    
    # Build title and subtitle separately
    title = base_title
    subtitle = None
    if exclude_boundaries and "Excl." not in base_title:
        subtitle = f"(Excl. {boundary_width}px boundary)"
    
    lw_pct = line_widths[0]
    lw_max = line_widths[1] if len(line_widths) > 1 else line_widths[0]

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Set main title
    ax.set_title(title, fontproperties=font_title, pad=30)
    
    # Add subtitle below title if needed
    if subtitle:
        # Add subtitle with smaller font below the main title
        ax.text(0.5, 1.05, subtitle, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='normal')

    colors = plt.get_cmap(palette.get('percentile_cmap', 'viridis'))(np.linspace(0.15, 0.85, levels_sorted.size))

    # Shaded bands: ±1% error bands around each percentile (if available)
    if pct_plus_sorted is not None and pct_minus_sorted is not None:
        for i in range(levels_sorted.size):
            lower = pct_minus_sorted[:, i]
            upper = pct_plus_sorted[:, i]
            ax.fill_between(x, lower, upper, color=colors[i], alpha=alpha_fill, label='_nolegend_')

    # Plot percentile curves
    label_entries = []
    label_x_frac = plot_params.get('label_x_frac', 0.90)
    for i, lvl in enumerate(levels_sorted):
        if float(lvl).is_integer():
            lbl = f'{int(lvl)}%'
        else:
            lbl = f'{lvl:.1f}%'
        ax.plot(x, pct_sorted[:, i], color=colors[i], linewidth=lw_pct, label='_nolegend_')
        
        # Store label position for later (after limits are set)
        label_idx = max(0, min(int(len(x) * label_x_frac), len(x) - 1))
        x_label = x[label_idx]
        y_label = pct_sorted[label_idx, i]
        label_entries.append((x_label, y_label, lbl, colors[i]))

    # Optional max curve
    gmax_list = percentile_data.get('global_max', None)
    if gmax_list is not None:
        # Select only valid indices if possible
        try:
            gmax = np.asarray([gmax_list[i] for i in valid_indices], dtype=float)
        except Exception:
            gmax = None
        if gmax is not None and gmax.size == x.size:
            ax.plot(x, gmax, color=color_max_curve, linewidth=lw_max, linestyle='--', label='_nolegend_')
            # Store Max label position for later
            label_idx = max(0, min(int(len(x) * label_x_frac), len(x) - 1))
            label_entries.append((x[label_idx], gmax[label_idx], 'Max', color_max_curve))
    
    if x_scale == 'log':
        ax.set_xscale('log')
        ax.set_xlabel(f'{xlabel} log', fontproperties=font)
    else:
        ax.set_xlabel(xlabel, fontproperties=font)
    
    if y_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylabel('Field amplitude log', fontproperties=font)
    else:
        ax.set_ylabel('Field amplitude', fontproperties=font)
    
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])

    ax.grid(alpha=0.3)

    if x_axis == 'zeta':
        ax.invert_xaxis()

    # Ensure labels stay inside the axes bounds
    label_margin_frac = plot_params.get('label_margin_frac', 0.03)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_min, x_max = (x0, x1) if x0 < x1 else (x1, x0)
    y_min, y_max = (y0, y1) if y0 < y1 else (y1, y0)

    def _clamp_lin(val, vmin, vmax, frac):
        margin = (vmax - vmin) * frac
        return min(max(val, vmin + margin), vmax - margin)

    def _clamp_log(val, vmin, vmax, frac):
        if val <= 0 or vmin <= 0 or vmax <= 0:
            return _clamp_lin(val, vmin, vmax, frac)
        lval = np.log10(val)
        lmin = np.log10(vmin)
        lmax = np.log10(vmax)
        margin = (lmax - lmin) * frac
        lval = min(max(lval, lmin + margin), lmax - margin)
        return 10 ** lval

    for x_label, y_label, lbl, color in label_entries:
        if not (np.isfinite(x_label) and np.isfinite(y_label)):
            continue
        if x_scale == 'log':
            x_plot = _clamp_log(x_label, x_min, x_max, label_margin_frac)
        else:
            x_plot = _clamp_lin(x_label, x_min, x_max, label_margin_frac)
        if y_scale == 'log':
            y_plot = _clamp_log(y_label, y_min, y_max, label_margin_frac)
        else:
            y_plot = _clamp_lin(y_label, y_min, y_max, label_margin_frac)
        ax.text(x_plot, y_plot, lbl, fontsize=10, color=color,
                verticalalignment='center', fontweight='bold', clip_on=True,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))

    fig.tight_layout()

    if save:
        if folder is None:
            folder = os.getcwd()

        axis_info = f"{x_axis}_{x_scale}_{y_scale}"
        limit_info = f"{xlim[0] if xlim else 'auto'}_{ylim[0] if ylim else 'auto'}_{ylim[1] if ylim else 'auto'}"
        sim_info = f"{induction_params.get('up_to_level','')}_{induction_params.get('F','')}_{induction_params.get('vir_kind','')}vir_{induction_params.get('rad_kind','')}rad_{induction_params.get('region','None')}Region"
        diff_cfg = induction_params.get('differentiation', {})
        if diff_cfg.get('buffer', False) == True:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol',''))
            buffer_info = f'Buffered_{diff_cfg.get("interpol","")}_siblings_{diff_cfg.get("use_siblings", False)}'
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
        else:
            buffer_info = 'NoBuffer'
        
        # Add boundary exclusion info to filename (use base_title to avoid duplication)
        boundary_info = f"ExclBound{boundary_width}px" if exclude_boundaries else ""

        file_title = '_'.join(base_title.split()[:3])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/{run}_{file_title}_percentile_evo_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get('stencil','')}_{boundary_info}_{timestamp}.png"
        filename = safe_filename(filename, verbose=verbose)
        fig.savefig(filename, dpi=dpi)
        if verbose:
            log_message(f'Percentile evolution plot saved as: {filename}', tag='percentiles', level=1)

    if verbose:
        lvl_str = ', '.join([str(l) for l in levels_sorted])
        log_message(f'Plotted percentile evolution for levels: {lvl_str}', tag='percentiles', level=1)

    return fig
        
        
def plot_integral_evolution(evolution_data, plot_params, induction_params,
                            grid_t, grid_zeta, rad,
                            verbose=True, save=False, folder=None):
    """
    Plot the evolution of the integrated magnetic energy and its induction components attending to the time derivative prediction from the induction equation.
    Shows total and differential evolutions simultaneously.
    
    Args:
        - evolution_data: dictionary containing the evolution data from induction_energy_integral_evolution()
                         (contains both total and differential data with _diff suffix for diff)
        - plot_params: dictionary containing plotting parameters:
            - plot_total: True to plot total (integrated) energy evolution
            - plot_differential: True to plot differential (rate of change) energy evolution
            - derivative: 'RK', 'central', 'implicit_forward', 'alpha_fit' or 'rate'
            - x_axis: 'zeta' or 'years'
            - x_scale: 'lin' or 'log'
            - y_scale: 'lin' or 'log'
            - xlim: [xlimo, xlimf] or None for auto
            - ylim: [ylimo, ylimf] or None for auto
            - cancel_limits: bool to flip the x axis (useful for zeta)
            - figure_size: [width, height]
            - line_widths: [line1, line2] for main and component lines
            - plot_type: 'raw', 'smoothed', or 'interpolated' to choose plot style
            - smoothing_sigma: sigma for Gaussian smoothing (only for 'smooth' type)
            - interpolation_points: number of points for interpolation (only for 'interpolated' type)
            - interpolation_kind: 'linear', 'cubic', or 'nearest' for interpolation method
            - volume_evolution: bool to plot volume evolution as additional figure
            - title: title for the plots (default: 'Integrated Magnetic Energy Evolution and Induction Prediction')
            - dpi: dots per inch for saved plots (default: 300)
            - run: identifier for the run (default: '_')
        - induction_params: dictionary containing simulation parameters:
            - units: energy unit conversion
            - F: size factor
            - level: refinement level
        - grid_t: time grid
        - grid_zeta: redshift grid  
        - rad: radius of the region in the last snapshot
        - verbose: bool for verbose output
        - save: bool to save plots
        - folder: folder to save plots (if None, uses current directory)
        
    Returns:
        - Dictionary with 'total' and/or 'differential' keys containing figure objects
        
    Author: Marco Molina
    """
    
    # Validate plot_params
    plot_type = plot_params.get('plot_type', 'raw')
    assert plot_type in ['raw', 'smoothed', 'interpolated'], "plot_type must be 'raw', 'smoothed', or 'interpolated'"
    assert plot_params.get('interpolation_kind', 'linear') in ['linear', 'cubic', 'nearest'], "interpolation_kind must be 'linear', 'cubic', or 'nearest'"
    assert plot_params.get('smoothing_sigma', 1.10) > 0, "smoothing_sigma must be a positive number"
    assert plot_params.get('x_axis', 'zeta') in ['zeta', 'years'], "x_axis must be 'zeta' or 'years'"
    
    # Read plot_total and plot_differential flags
    plot_total = plot_params.get('plot_total', True)
    plot_differential = plot_params.get('plot_differential', True)
    assert plot_total or plot_differential, "At least one of plot_total or plot_differential must be True"

    requested_modes = []
    if plot_total:
        requested_modes.append('total')
    if plot_differential:
        requested_modes.append('differential')

    if not plot_params.get('_internal_mode', False) and len(requested_modes) > 1:
        figures = []
        plot_volume_once = bool(plot_params.get('volume_evolution', False))
        for mode_index, mode in enumerate(requested_modes):
            mode_params = plot_params.copy()
            mode_params['evolution_type'] = mode
            mode_params['plot_total'] = mode == 'total'
            mode_params['plot_differential'] = mode == 'differential'
            mode_params['y_scale'] = 'log' if mode == 'total' else 'lin'
            mode_params['volume_evolution'] = plot_volume_once and mode_index == 0
            mode_params['_internal_mode'] = True
            figures.extend(
                plot_integral_evolution(
                    evolution_data, mode_params, induction_params,
                    grid_t, grid_zeta, rad,
                    verbose=verbose, save=save, folder=folder
                )
            )
        return figures
    
    # Extract parameters from plot_params
    derivative = plot_params['derivative']
    x_axis = plot_params['x_axis']
    if x_axis == 'zeta':
        assert len(grid_zeta) > 0, "grid_zeta must not be empty when x_axis is 'zeta'"
        assert plot_params.get('interpolation_points', 5000) > 0, "interpolation_points must be a positive integer"
    elif x_axis == 'years':
        assert len(grid_t) > 0, "grid_t must not be empty when x_axis is 'years'"
        assert plot_params.get('interpolation_points', 500) > 0, "interpolation_points must be a positive integer"
    
    evolution_type = plot_params.get('evolution_type', 'differential' if plot_differential else 'total')
    data_suffix = '_diff' if evolution_type == 'differential' else ''
    x_scale = plot_params['x_scale']
    y_scale = 'lin' if evolution_type == 'differential' else plot_params['y_scale']
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    cancel_limits = plot_params.get('cancel_limits', False)
    figure_size = plot_params.get('figure_size', [10, 8])
    line_widths = plot_params.get('line_widths', [5, 3])
    volume_evolution = plot_params.get('volume_evolution', False)
    title = plot_params.get('title', 'Integrated Magnetic Energy Evolution and Induction Prediction')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    
    # Parameters specific to plot type
    plot_type = plot_params.get('plot_type', 'raw')
    if plot_type == 'smoothed':
        smoothing_sigma = plot_params.get('smoothing_sigma', 1.10)
    elif plot_type == 'interpolated':
        interpolation_points = plot_params.get('interpolation_points', {'years': 500, 'zeta': 5000})
        interpolation_kind = plot_params.get('interpolation_kind', 'cubic')
    cumulative_headroom = plot_params.get('plot_cumulative_magnetic_energy_headroom', 0.05)
    
    # Extract induction parameters
    units = plot_params.get('units', induction_params.get('units', 1.0))
    factor_F = induction_params['F']
    region = induction_params['region']
    components_cfg = induction_params.get('components', {})
    plot_magnetic_energy = bool(components_cfg.get('magnetic_energy', True))
    plot_kinetic_energy = bool(components_cfg.get('kinetic_energy', True))
    plot_cumulative_magnetic_energy = bool(plot_params.get('plot_cumulative_magnetic_energy', False))
    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_negative_interval = palette.get('negative_interval', DEFAULT_PLOT_PALETTE['negative_interval'])
    color_measured = palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
    color_itemized = palette.get('induction_itemized', DEFAULT_PLOT_PALETTE['induction_itemized'])
    color_compact = palette.get('induction_compact', DEFAULT_PLOT_PALETTE['induction_compact'])
    color_kinetic = palette.get('kinetic_energy', DEFAULT_PLOT_PALETTE['kinetic_energy'])
    
    # Set up matplotlib parameters
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 20
    })
    
    # Define font properties
    font = FontProperties()
    font.set_style('normal')
    font.set_weight('normal')
    font.set_size(12)
    
    font_title = FontProperties()
    font_title.set_style('normal')
    font_title.set_weight('bold')
    font_title.set_size(20)
    
    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)
    
    y_title = 1.02
    line1, line2 = line_widths
    component_alpha = float(np.clip(plot_params.get('component_alpha', 0.75), 0.05, 1.0))
    component_alpha = float(np.clip(plot_params.get('component_alpha', 0.75), 0.05, 1.0))
    component_threshold = plot_params.get(
        'component_threshold',
        0.0 if evolution_type == 'differential' else induction_params.get('differentiation', {}).get('epsilon', 1e-30)
    )
    normalize_by_volume = induction_params.get('energy_evolution', {}).get('normalize_by_volume', False)
    normalized = plot_params.get(
        'normalized',
        induction_params.get('energy_evolution', {}).get('normalized', True)
    )
    
    # Prepare time and redshift arrays
    if x_axis == 'zeta':
        z = np.array([grid_zeta[i] for i in range(len(grid_zeta))])
        if z[-1] < 0:
            z[-1] = abs(z[-1])
    else: # years
        t = [grid_t[i] * time_to_yr for i in range(len(grid_t))]
    
    # Extract evolution data with units (using data_suffix for dynamic key selection)
    if evolution_type == 'differential' and units!= 1.0:
        units = units / time_to_s
    
    # Get the appropriate data arrays based on evolution_type and derivative
    if evolution_type == 'total':
        index_O, index_F = 0, len(grid_t)
        kind = 'total'
    else:  # differential
        index_O, index_F = 1, len(grid_t)
        kind = 'differential'
        
    if derivative == 'RK':
        index_o, index_f = 0, len(grid_t)
        plotid = f'RK_{kind}'
    elif derivative == 'central':
        # Central total predictor yields E_{i+1}; compare against measured support [1:]
        index_o, index_f = 1, len(grid_t)
        plotid = f'central_{kind}'
    elif derivative == 'alpha_fit':
        # Calibrated explicit predictor yields E_{i+1}; compare against measured support [1:]
        index_o, index_f = 1, len(grid_t)
        plotid = f'alpha_fit_{kind}'
    elif derivative == 'rate':
        # Rate predictor also targets the next snapshot E_{i+1}.
        index_o, index_f = 1, len(grid_t)
        plotid = f'rate_{kind}'
    elif derivative == 'implicit_forward':
        # Forward-implicit branch starts at k=1 and predicts E_{k+1} -> support [2:]
        index_o, index_f = 2, len(grid_t)
        plotid = f'implicit_forward_{kind}'
    else:
        raise ValueError(f"Unsupported derivative for evolution plotting: {derivative}")
    
    # Extract component data with units (use dynamic keys based on data_suffix)
    n1 = [units * evolution_data[f'evo_b2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_b2{data_suffix}']))]
    n0 = [units * evolution_data[f'evo_ind_b2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_ind_b2{data_suffix}']))]
    diver_work = [units * evolution_data[f'evo_MIE_diver_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_diver_B2{data_suffix}']))]
    compres_work = [units * evolution_data[f'evo_MIE_compres_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_compres_B2{data_suffix}']))]
    stretch_work = [units * evolution_data[f'evo_MIE_stretch_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_stretch_B2{data_suffix}']))]
    advec_work = [units * evolution_data[f'evo_MIE_advec_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_advec_B2{data_suffix}']))]
    drag_work = [units * evolution_data[f'evo_MIE_drag_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_drag_B2{data_suffix}']))]
    total_work = [units * evolution_data[f'evo_MIE_total_B2{data_suffix}'][i] for i in range(len(evolution_data[f'evo_MIE_total_B2{data_suffix}']))]
    kinetic_work = [units * evolution_data[f'evo_kinetic_energy{data_suffix}'][i] for i in range(len(evolution_data[f'evo_kinetic_energy{data_suffix}']))]
    
    # Volume data
    volume_phi = evolution_data['evo_volume_phi']
    volume_co = evolution_data['evo_volume_co']
    
    # Prepare data based on plot type
    if plot_type == 'smoothed':
        # Apply Gaussian smoothing
        n1_data = gaussian_filter1d(n1, sigma=smoothing_sigma)
        n0_data = gaussian_filter1d(n0, sigma=smoothing_sigma)
        diver_work_data = gaussian_filter1d(diver_work, sigma=smoothing_sigma)
        compres_work_data = gaussian_filter1d(compres_work, sigma=smoothing_sigma)
        stretch_work_data = gaussian_filter1d(stretch_work, sigma=smoothing_sigma)
        advec_work_data = gaussian_filter1d(advec_work, sigma=smoothing_sigma)
        drag_work_data = gaussian_filter1d(drag_work, sigma=smoothing_sigma)
        total_work_data = gaussian_filter1d(total_work, sigma=smoothing_sigma)
        kinetic_work_data = gaussian_filter1d(kinetic_work, sigma=smoothing_sigma)
        x_data = z if x_axis == 'zeta' else t
        plot_suffix = f'_smoothed_sigma_{smoothing_sigma}'
        
    elif plot_type == 'interpolated':
        # Use a common valid domain so all interpolated curves are aligned.
        common_i0 = max(index_O, index_o)
        common_i1 = min(index_F, index_f)
        # Create interpolations
        if x_axis == 'years':
            x_data = t
            x_new = np.linspace(min(t[common_i0:common_i1]), max(t[common_i0:common_i1]), 
                                    num=interpolation_points['years'], endpoint=True)
        else:  # zeta
            x_data = z
            x_new = np.linspace(max(z[common_i0:common_i1]), min(z[common_i0:common_i1]), 
                                    num=interpolation_points['zeta'], endpoint=True)
        
        # Create interpolation functions
        n1_interp = interp1d(x_data[common_i0:common_i1], n1[common_i0:common_i1], kind=interpolation_kind)
        n0_interp = interp1d(x_data[common_i0:common_i1], n0[common_i0:common_i1], kind=interpolation_kind)
        diver_work_interp = interp1d(x_data[common_i0:common_i1], diver_work[common_i0:common_i1], kind=interpolation_kind)
        compres_work_interp = interp1d(x_data[common_i0:common_i1], compres_work[common_i0:common_i1], kind=interpolation_kind)
        stretch_work_interp = interp1d(x_data[common_i0:common_i1], stretch_work[common_i0:common_i1], kind=interpolation_kind)
        advec_work_interp = interp1d(x_data[common_i0:common_i1], advec_work[common_i0:common_i1], kind=interpolation_kind)
        drag_work_interp = interp1d(x_data[common_i0:common_i1], drag_work[common_i0:common_i1], kind=interpolation_kind)
        total_work_interp = interp1d(x_data[common_i0:common_i1], total_work[common_i0:common_i1], kind=interpolation_kind)
        kinetic_work_interp = interp1d(x_data[common_i0:common_i1], kinetic_work[common_i0:common_i1], kind=interpolation_kind)
        
        # Use interpolated data
        x_data = x_new
        n1_data = n1_interp(x_new)
        n0_data = n0_interp(x_new)
        diver_work_data = diver_work_interp(x_new)
        compres_work_data = compres_work_interp(x_new)
        stretch_work_data = stretch_work_interp(x_new)
        advec_work_data = advec_work_interp(x_new)
        drag_work_data = drag_work_interp(x_new)
        total_work_data = total_work_interp(x_new)
        kinetic_work_data = kinetic_work_interp(x_new)
        plot_suffix = f'{interpolation_kind}_interpolated_{interpolation_points[x_axis]}_points'

        # Adjust indices for interpolated data
        index_O_plot = 0
        index_F_plot = len(n1_data)
        index_o_plot = 0
        index_f_plot = len(n0_data)
        
    else:  # raw
        # Use raw data
        n1_data = n1
        n0_data = n0
        diver_work_data = diver_work
        compres_work_data = compres_work
        stretch_work_data = stretch_work
        advec_work_data = advec_work
        drag_work_data = drag_work
        total_work_data = total_work
        kinetic_work_data = kinetic_work
        x_data = z if x_axis == 'zeta' else t
        plot_suffix = '_raw'
        
    # For raw and smooth data, use original indices
    if plot_type != 'interpolated':
        index_O_plot = index_O
        index_F_plot = index_F
        index_o_plot = index_o
        index_f_plot = index_f

    if verbose:
        axis_values = np.asarray(x_data)
        axis_label = 'z' if x_axis == 'zeta' else 't_yr'

        def _range_from_indices(yvals, i0, i1):
            ylen = len(yvals)
            upper = min(i1, len(axis_values), ylen)
            lower = max(0, min(i0, upper))
            if upper <= lower:
                return f'empty (idx=[{i0}:{i1}], len_y={ylen}, len_x={len(axis_values)})'
            return (
                f'idx=[{lower}:{upper}] '
                f'{axis_label}:[{axis_values[lower]:.6g}, {axis_values[upper-1]:.6g}] '
                f'len={upper-lower} (len_y={ylen})'
            )

        def _range_from_offset(yvals, i0, i1):
            # y[0] is mapped to x[i0], useful for derivative-predicted arrays.
            ylen = len(yvals)
            lower = max(0, min(i0, len(axis_values)))
            upper = min(i1, len(axis_values), i0 + ylen)
            if upper <= lower:
                return f'empty (idx=[{i0}:{i1}], len_y={ylen}, len_x={len(axis_values)})'
            return (
                f'idx=[{lower}:{upper}] '
                f'{axis_label}:[{axis_values[lower]:.6g}, {axis_values[upper-1]:.6g}] '
                f'len={upper-lower} (len_y={ylen}, y_offset={i0})'
            )

        # Log hierarchical debug using utils.log_message for consistent formatting
        log_message(f'Plotting debug: x_axis={x_axis}, plot_type={plot_type}, derivative={derivative}', tag='plots', level=1)
        log_message(f'  Base axis len={len(axis_values)}, range {axis_label}:[{axis_values[0]:.6g}, {axis_values[-1]:.6g}]', tag='plots', level=1)
        if plot_magnetic_energy:
            log_message(f'    Magnetic Energy (n1): {_range_from_indices(n1_data, index_O_plot, index_F_plot)}', tag='plots', level=2)
        if plot_kinetic_energy:
            log_message(f'    Kinetic Energy: {_range_from_indices(kinetic_work_data, index_O_plot, index_F_plot)}', tag='plots', level=2)
        log_message(f'    Itemized Prediction (n0): {_range_from_offset(n0_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Compact Prediction (total): {_range_from_offset(total_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Compression: {_range_from_offset(compres_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Stretching: {_range_from_offset(stretch_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Advection: {_range_from_offset(advec_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Divergence: {_range_from_offset(diver_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
        log_message(f'    Cosmic Drag: {_range_from_offset(drag_work_data, index_o_plot, index_f_plot)}', tag='plots', level=2)
    
    # Create figures list
    figures = []
    
    # Main evolution plot
    fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)

    def _slice_xy(xvals, yvals, i0, i1):
        """Return aligned x/y slices, clipped to valid bounds of both arrays."""
        nxy = min(len(xvals), len(yvals))
        i0c = max(0, min(i0, nxy))
        i1c = max(i0c, min(i1, nxy))
        return xvals[i0c:i1c], yvals[i0c:i1c]

    def _slice_xy_with_offset(xvals, yvals, i0, i1):
        """Map y[0] to x[i0] and clip by i1 and array bounds."""
        i0c = max(0, min(i0, len(xvals)))
        i1c = min(i1, len(xvals), i0 + len(yvals))
        i1c = max(i0c, i1c)
        count = i1c - i0c
        return xvals[i0c:i1c], yvals[:count]
    
    # Track which components were plotted
    components_plotted = []
    
    # Kinetic energy
    xk, yk = _slice_xy(x_data, kinetic_work_data, index_O_plot, index_F_plot)
    if plot_kinetic_energy and should_plot_component(yk, threshold=component_threshold):
        ax1.plot(xk, yk, 
            linewidth=line1, label='Kinetic Energy', color=color_kinetic)
        components_plotted.append('kinetic')
    
    # Main energy line (always plot)
    if evolution_type == 'total':
        label = 'Magnetic Energy'
    else:
        label = 'Magnetic Energy Induction'

    x1, y1 = _slice_xy(x_data, n1_data, index_O_plot, index_F_plot)
    if plot_magnetic_energy and should_plot_component(y1, threshold=component_threshold):
        ax1.plot(x1, y1, 
            linewidth=line1, label=label, color=color_measured)
        components_plotted.append('magnetic_energy')
        
    # Total work (compacted)
    xt, yt = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)
    if should_plot_component(yt, threshold=component_threshold):
        ax1.plot(xt, yt, '-', 
            linewidth=line1, label='...from Compact Induction', color=color_compact)
        components_plotted.append('total')

    # Induction prediction (plot if has data)
    x0, y0 = _slice_xy_with_offset(x_data, n0_data, index_o_plot, index_f_plot)
    if should_plot_component(y0, threshold=component_threshold):
        ax1.plot(x0, y0, '--',
            linewidth=line1, label='...from Itemize Induction', color=color_itemized)

    # Individual components with their colors
    component_configs = [
        (compres_work_data, 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
        (stretch_work_data, 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
        (advec_work_data, 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
        (diver_work_data, 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
        (drag_work_data, 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag')
    ]
    
    for data, label, color, sym in component_configs:
        xc, yc = _slice_xy_with_offset(x_data, data, index_o_plot, index_f_plot)
        if should_plot_component(yc, threshold=component_threshold):
            latex_label = rf'{label} $\Gamma_{{\mathrm{{{sym}}}}}$'
            ax1.plot(xc, yc, '--', 
                    linewidth=line2, label=latex_label, color=color)
            components_plotted.append(label.lower())

    ax1_aux = None
    yb2_cumulative = None
    if evolution_type == 'differential' and plot_cumulative_magnetic_energy:
        cumulative_units = plot_params.get('units', induction_params.get('units', 1.0))
        b2_total = cumulative_units * np.asarray(evolution_data.get('evo_b2', []), dtype=float)
        xb2, yb2 = _slice_xy(x_data, b2_total, index_O_plot, index_F_plot)
        if len(yb2) > 0 and should_plot_component(yb2, threshold=0.0):
            yb2_cumulative = np.cumsum(np.nan_to_num(yb2, nan=0.0))
            ax1_aux = ax1.twinx()
            ax1_aux.plot(
                xb2,
                yb2_cumulative,
                '-',
                linewidth=max(1.2, line2),
                color=color_measured,
                alpha=0.45,
                label='Cumulative Magnetic Energy',
                zorder=1
            )
            ax1_aux.fill_between(
                xb2,
                0.0,
                yb2_cumulative,
                color=color_measured,
                alpha=0.06,
                label='_nolegend_',
                zorder=1
            )
            norm_suffix_cum = r' ($\rho_B^{-1}$)' if normalized else ''
            ax1_aux.set_ylabel(f'Cumulative Magnetic Energy{norm_suffix_cum}', fontproperties=font, color=color_measured)
            ax1_aux.tick_params(axis='y', colors=color_measured)
            components_plotted.append('cumulative_magnetic_energy')

    if verbose and evolution_type == 'total':
        # Quantify temporal lag between measured magnetic energy and predictions.
        xm, ym = _slice_xy(x_data, n1_data, index_O_plot, index_F_plot)
        xp_item, yp_item = _slice_xy_with_offset(x_data, n0_data, index_o_plot, index_f_plot)
        xp_comp, yp_comp = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)

        # Recompute effective starts to translate local peak indices into global snapshot indices.
        nxy_meas = min(len(x_data), len(n1_data))
        start_meas = max(0, min(index_O_plot, nxy_meas))
        start_pred = max(0, min(index_o_plot, len(x_data)))

        def _peak_info(xv, yv):
            if len(yv) == 0:
                return None
            ip = int(np.nanargmax(np.asarray(yv)))
            return ip, float(xv[ip]), float(yv[ip])

        peak_m = _peak_info(xm, ym)
        peak_i = _peak_info(xp_item, yp_item)
        peak_c = _peak_info(xp_comp, yp_comp)

        if peak_m and peak_i:
            global_m = start_meas + peak_m[0]
            global_i = start_pred + peak_i[0]
            log_message(
                "Peak lag (itemized vs measured): "
                f"d_idx_local={peak_i[0]-peak_m[0]}, d_idx_global={global_i-global_m}, "
                f"d_{axis_label}={peak_i[1]-peak_m[1]:.6g}, "
                f"measured_{axis_label}={peak_m[1]:.6g}, itemized_{axis_label}={peak_i[1]:.6g}, "
                f"idx_measured_global={global_m}, idx_itemized_global={global_i}",
                tag='evolution',
                level=2,
            )
        if peak_m and peak_c:
            global_m = start_meas + peak_m[0]
            global_c = start_pred + peak_c[0]
            log_message(
                "Peak lag (compact vs measured): "
                f"d_idx_local={peak_c[0]-peak_m[0]}, d_idx_global={global_c-global_m}, "
                f"d_{axis_label}={peak_c[1]-peak_m[1]:.6g}, "
                f"measured_{axis_label}={peak_m[1]:.6g}, compact_{axis_label}={peak_c[1]:.6g}, "
                f"idx_measured_global={global_m}, idx_compact_global={global_c}",
                tag='evolution',
                level=2,
            )
    
    setup_axis(ax1, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis, evolution_type, font)
    if ax1_aux is not None:
        align_cumulative_overlay_zero(
            ax1,
            ax1_aux,
            y_aux_max=np.nanmax(yb2_cumulative) if yb2_cumulative is not None else None,
            headroom=cumulative_headroom,
        )
    norm_suffix = r' ($\rho_B^{-1}$)' if normalized else ''
    vol_suffix = r' / Volume' if normalize_by_volume else ''
    if evolution_type == 'total':
        ax1.set_ylabel(f'Magnetic Energy{norm_suffix}{vol_suffix}', fontproperties=font)
    else:
        ax1.set_ylabel(f'Magnetic Evolution{norm_suffix}{vol_suffix}', fontproperties=font)
    ax1.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    if ax1_aux is not None:
        aux_handles, aux_labels = ax1_aux.get_legend_handles_labels()
        handles = handles + aux_handles
        labels = labels + aux_labels
    ax1.legend(handles, labels, prop=font_legend, ncol=2)

    if region == 'None':
        plot_title = f'{title} - {np.round(induction_params["size"][0]/2, 1)} Mpc'
    else:
        plot_title = f'{title} - {np.round(factor_F*rad, 1)} Mpc'

    if evolution_type != 'total':
        plot_title = plot_title.replace('Evolution', 'Induction Evolution')
    ax1.set_title(plot_title, y=y_title, fontproperties=font_title)
    
    if cancel_limits and x_axis == 'zeta':
        ax1.invert_xaxis()
    fig1.tight_layout()
    figures.append(fig1)
    
    # Volume plot (optional)
    if volume_evolution:
        fig2, ax2 = plt.subplots(figsize=figure_size, dpi=dpi)
        
        if x_axis == 'years':
            ax2.set_xlabel('Time (yr)', fontproperties=font)
            ax2.plot(t, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(t, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
        else:  # zeta
            ax2.set_xlabel('Redshift (z)', fontproperties=font)
            ax2.plot(z, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(z, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
        
        ax2.set_ylabel('Integration Volume', fontproperties=font)
        ax2.legend(prop=font_legend)
        ax2.set_yscale('log')
        ax2.grid(alpha=0.3)
        
        ax2.set_title('Integrated Volume', y=y_title, fontproperties=font_title)
        fig2.tight_layout()
        figures.append(fig2)
    
    if verbose:
        log_message(f'{plot_type.capitalize()} integrated magnetic energy and induction prediction plot created', tag='evolution', level=1)
        log_message(f'Components plotted: {", ".join(components_plotted)}', tag='evolution', level=1)
        if volume_evolution:
            log_message('Volume evolution plot created', tag='evolution', level=1)
    
    # Save plots if requested
    if save:
        if folder is None:
            folder = os.getcwd()
        
        # Create filename components
        sim_info = f'{induction_params["up_to_level"]}_{induction_params["F"]}_{induction_params["vir_kind"]}vir_{induction_params["rad_kind"]}rad_{region}Region'
        axis_info = f'{x_axis}_{x_scale}_{y_scale}'
        if cancel_limits:
            limit_info = 'cancel_limits'
        else:
            limit_info = f'{xlim[0] if xlim else "auto"}_{ylim[0] if ylim else "auto"}_{ylim[1] if ylim else "auto"}'
            
        diff_cfg = induction_params.get('differentiation', {})
        if diff_cfg.get('buffer', False) == True:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol',''))
            buffer_info = f'Buffered_{diff_cfg.get("interpol", "")}_siblings_{diff_cfg.get("use_siblings", False)}'
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
        else:
            buffer_info = 'NoBuffer'
        
        # Save main plot
        file_title = '_'.join(title.split()[:3])
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename1 = f'{folder}/{run}_{file_title}_integrated_energy_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}_{plotid}{plot_suffix}.png'
        filename1 = safe_filename(filename1, verbose=verbose)
        fig1.savefig(filename1, dpi=dpi)
        
        if verbose:
            log_message(f'Main plot saved as: {filename1}', tag='evolution', level=1)
        
        # Save volume plot if created
        if volume_evolution:
            filename2 = f'{folder}/{run}_{file_title}_volume_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}_{plotid}.png'
            filename2 = safe_filename(filename2, verbose=verbose)
            fig2.savefig(filename2, dpi=dpi)
            if verbose:
                log_message(f'Volume plot saved as: {filename2}', tag='evolution', level=1)
            
            if verbose:
                print(f'Plotting... Volume plot saved as: {filename2}')
                
    if verbose:       
        for i, sim in enumerate(induction_params.get('sims', ['default'])):
            if derivative == 'RK':
                n_iter = len(induction_params.get('it', [0]))
            elif derivative == 'central' or derivative == 'alpha_fit' or derivative == 'rate':
                n_iter = len(induction_params.get('it', [0])) - 1
            elif derivative == 'implicit_forward':
                n_iter = len(induction_params.get('it', [0])) - 2
            else:
                n_iter = 0
            if n_iter > 0:
                for j in range(min(n_iter, len(n0_data))):
                    print(f'Simulation: {sim} | Iteration: {j}')
                    
                    if evolution_type == 'total':
                        print(f'Magnetic energy density in snap {i+j+1}: {n1_data[min(i+j+1, len(n1_data)-1)]}')
                        print(f'Magnetic from induction in snap {i+j+1}: {n0_data[min(i+j, len(n0_data)-1)]}')
                    else:
                        print(f'Magnetic induction in snap {i+j+1}: {n0_data[min(i+j, len(n0_data)-1)]}')
                        print(f'Predicted induction in snap {i+j+1}: {n0_data[min(i+j, len(n0_data)-1)]}')
                    
                    print(f'Divergence work: {diver_work_data[min(i+j, len(diver_work_data)-1)]}')
                    print(f'Compressive work: {compres_work_data[min(i+j, len(compres_work_data)-1)]}')
                    print(f'Stretching work: {stretch_work_data[min(i+j, len(stretch_work_data)-1)]}')
                    print(f'Advection work: {advec_work_data[min(i+j, len(advec_work_data)-1)]}')
                    print(f'Drag work: {drag_work_data[min(i+j, len(drag_work_data)-1)]}')
                    print(f'Total work (compacted): {total_work_data[min(i+j, len(total_work_data)-1)]}')
                    print(f'Kinetic energy: {kinetic_work_data[min(i+j, len(kinetic_work_data)-1)]}')
    
    return figures


def plot_production_dissipation_evolution(pd_data, plot_params, induction_params,
                                        grid_t, grid_zeta, rad=None,
                                        verbose=True, save=False, folder=None):
    '''
    Plot production/dissipation evolution from precomputed volumetric integrals.

    Args:
        - pd_data: dictionary with integrated production/dissipation arrays over snapshots
        - plot_params: dictionary with plotting options
        - induction_params: dictionary with simulation metadata
        - grid_t: time grid
        - grid_zeta: redshift grid
        - verbose: whether to print progress
        - save: whether to save figures
        - folder: output folder

    Returns:
        - list of matplotlib figure objects

    Author: Marco Molina
    '''

    required_keys = [
        'int_MIE_total_B2_prod_itemized',
        'int_MIE_total_B2_diss_itemized'
    ]
    if not all(k in pd_data for k in required_keys):
        if verbose:
            print('Production/dissipation plot skipped: required integrated keys are missing')
        return []

    x_axis = plot_params.get('x_axis', 'zeta')
    x_scale = plot_params.get('x_scale', 'lin')
    y_scale = plot_params.get('y_scale', 'log')
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    cancel_limits = plot_params.get('cancel_limits', False)
    figure_size = plot_params.get('figure_size', [12, 8])
    line_widths = plot_params.get('line_widths', [3.0, 2.0])
    title = plot_params.get('title', 'Production and Dissipation Evolution')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    plot_total_prod_diss = plot_params.get('plot_total_prod_diss', True)
    plot_absolute = plot_params.get('plot_absolute', True)
    plot_fractional = plot_params.get('plot_fractional', True)
    plot_net = plot_params.get('plot_net', False)
    units = plot_params.get('units', induction_params.get('units', 1.0))
    normalized = plot_params.get('normalized', induction_params.get('production_dissipation', {}).get('normalized', True))
    normalize_by_volume = induction_params.get('production_dissipation', {}).get('normalize_by_volume', False)
    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_prod = palette.get('production', DEFAULT_PLOT_PALETTE['production'])
    color_diss = palette.get('dissipation', DEFAULT_PLOT_PALETTE['dissipation'])
    color_itemized_net = palette.get('net_itemized', DEFAULT_PLOT_PALETTE['net_itemized'])
    color_compact_net = palette.get('net_compact', DEFAULT_PLOT_PALETTE['net_compact'])
    color_efficiency = palette.get('efficiency', DEFAULT_PLOT_PALETTE['efficiency'])
    plot_density = bool(plot_params.get('plot_density', False))
    plot_magnetic_energy = bool(plot_params.get('plot_magnetic_energy', False))
    plot_cumulative_magnetic_energy = bool(plot_params.get('plot_cumulative_magnetic_energy', False))
    cumulative_headroom = plot_params.get('plot_cumulative_magnetic_energy_headroom', 0.05)
    color_efficiency = palette.get('efficiency', DEFAULT_PLOT_PALETTE['efficiency'])
    if not isinstance(normalized, bool):
        normalized = True
    if not isinstance(normalize_by_volume, bool):
        normalize_by_volume = False
    try:
        units = float(units)
    except (TypeError, ValueError):
        units = 1.0
    epsilon = induction_params.get('differentiation', {}).get('epsilon', 1e-30)

    # Match style with latest plot functions (e.g. radial profiles)
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18
    })

    font = FontProperties()
    font.set_style('normal')
    font.set_weight('normal')
    font.set_size(12)

    font_title = FontProperties()
    font_title.set_style('normal')
    font_title.set_weight('bold')
    font_title.set_size(20)

    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)

    y_title = 1.02

    line_main = line_widths[0]
    line_comp = line_widths[1] if len(line_widths) > 1 else line_widths[0]

    if x_axis == 'years':
        x = np.array([grid_t[i] * time_to_yr for i in range(len(grid_t))], dtype=float)
        xlabel = 'Time (yr)'
    else:
        x = np.array([grid_zeta[i] for i in range(len(grid_zeta))], dtype=float)
        if x.size and x[-1] < 0:
            x[-1] = abs(x[-1])
        xlabel = 'Redshift (z)'

    if rad is None:
        if induction_params.get('region', None) == 'None':
            region_label = f'{np.round(induction_params.get("size", [0])[0] / 2)} Mpc'
        else:
            region_label = f'{np.round(induction_params.get("F", 1.0) * induction_params.get("size", [0])[0] / 2)} Mpc'
    else:
        if induction_params.get('region', None) == 'None':
            region_label = f'{np.round(induction_params.get("size", [0])[0] / 2)} Mpc'
        else:
            region_label = f'{np.round(induction_params.get("F", 1.0) * rad, 1)} Mpc'

    # Itemized totals: sum of per-term production/dissipation contributions.
    itemized_prod = units * np.asarray(pd_data['int_MIE_total_B2_prod_itemized'], dtype=float)
    itemized_diss = units * np.asarray(pd_data['int_MIE_total_B2_diss_itemized'], dtype=float)
    itemized_net = itemized_prod - itemized_diss

    compact_prod = None
    compact_diss = None
    if 'int_MIE_total_B2_prod_compact' in pd_data and 'int_MIE_total_B2_diss_compact' in pd_data:
        compact_prod = units * np.asarray(pd_data['int_MIE_total_B2_prod_compact'], dtype=float)
        compact_diss = units * np.asarray(pd_data['int_MIE_total_B2_diss_compact'], dtype=float)
    compact_net = None if compact_prod is None or compact_diss is None else (compact_prod - compact_diss)

    # Component palette used in plot_induction_radial_profiles
    component_map = [
        ('MIE_compres_B2', 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
        ('MIE_stretch_B2', 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
        ('MIE_advec_B2', 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
        ('MIE_diver_B2', 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
        ('MIE_drag_B2', 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag')
    ]

    if verbose:
        try:
            if compact_prod is not None and compact_diss is not None:
                # Primary comparison: itemized-sum totals vs compact-total split.
                rel_prod_comp = np.nanmax(np.abs(itemized_prod - compact_prod) / np.maximum(np.abs(compact_prod), epsilon))
                rel_diss_comp = np.nanmax(np.abs(itemized_diss - compact_diss) / np.maximum(np.abs(compact_diss), epsilon))
                print(f'Production/dissipation totals: itemized-sum vs compact-split (max rel diff P={rel_prod_comp:.3e}, D={rel_diss_comp:.3e}). Expected differences as compact totals internalize per-component cancellations.')
            if compact_net is not None:
                rel_net = np.nanmax(np.abs(itemized_net - compact_net) / np.maximum(np.abs(compact_net), epsilon))
                print(f'Production/dissipation net: itemized vs compact (max rel diff N={rel_net:.3e}). Expected convergent net results.')
        except Exception:
            pass

    figures = []

    def _overlay_cumulative_magnetic_energy(ax):
        if not plot_cumulative_magnetic_energy:
            return None
        if 'int_b2' not in pd_data:
            return None

        b2_snap = units * np.asarray(pd_data.get('int_b2', []), dtype=float)
        nxy = min(len(x), len(b2_snap))
        if nxy == 0:
            return None

        x_aux = np.asarray(x[:nxy], dtype=float)
        y_aux = np.asarray(b2_snap[:nxy], dtype=float)
        if not should_plot_component(y_aux, threshold=0.0):
            return None

        y_aux_cumulative = np.cumsum(np.nan_to_num(y_aux, nan=0.0))
        ax_aux = ax.twinx()
        ax_aux.plot(
            x_aux,
            y_aux_cumulative,
            '-',
            linewidth=max(1.2, line_comp),
            color=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']),
            alpha=0.45,
            label='Cumulative Magnetic Energy',
            zorder=1
        )
        ax_aux.fill_between(
            x_aux,
            0.0,
            y_aux_cumulative,
            color=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']),
            alpha=0.06,
            label='_nolegend_',
            zorder=1
        )
        norm_suffix_cum = r' ($\rho_B^{-1}$)' if normalized else ''
        ax_aux.set_ylabel(
            f'Cumulative Magnetic Energy{norm_suffix_cum}',
            fontproperties=font,
            color=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
        )
        ax_aux.tick_params(
            axis='y',
            colors=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
        )
        y_aux_max = np.nanmax(y_aux_cumulative) if len(y_aux_cumulative) > 0 else None
        align_cumulative_overlay_zero(ax, ax_aux, y_aux_max=y_aux_max, headroom=cumulative_headroom)
        return ax_aux

    def _set_combined_legend(ax, ax_aux=None):
        handles, labels = ax.get_legend_handles_labels()
        if ax_aux is not None:
            aux_handles, aux_labels = ax_aux.get_legend_handles_labels()
            handles = handles + aux_handles
            labels = labels + aux_labels
        ax.legend(handles, labels, prop=font_legend, ncol=2)

    # Absolute production/dissipation rates
    if plot_absolute:
        fig_abs, ax_abs = plt.subplots(figsize=figure_size, dpi=dpi)
        if plot_total_prod_diss:
            ax_abs.plot(x, itemized_prod, '-.', linewidth=line_main, color=color_prod, label='Total Production (itemized sum)')
            ax_abs.plot(x, itemized_diss, '-.', linewidth=line_main, color=color_diss, label='Total Dissipation (itemized sum)')
        ax_abs.plot(x, itemized_net, '--', linewidth=line_main, color=color_itemized_net, label='Net (itemized) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$')

        if compact_prod is not None and compact_diss is not None:
            if plot_total_prod_diss:
                ax_abs.plot(x, compact_prod, '-', linewidth=line_comp, color=color_prod, label='Total Production (compact)')
                ax_abs.plot(x, compact_diss, '-', linewidth=line_comp, color=color_diss, label='Total Dissipation (compact)')
        if compact_net is not None:
            ax_abs.plot(x, compact_net, '-', linewidth=line_main, color=color_compact_net, label='Net (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$')

        for prefix, label, color, sym in component_map:
            prod_key = f'int_{prefix}_prod'
            diss_key = f'int_{prefix}_diss'
            if prod_key in pd_data:
                arr_p = units * np.asarray(pd_data[prod_key], dtype=float)
                if should_plot_component(arr_p):
                    ax_abs.plot(x, arr_p, '--', linewidth=line_comp, color=color, label=rf'{label} $P_{{\mathrm{{{sym}}}}}$')
            if diss_key in pd_data:
                arr_d = units * np.asarray(pd_data[diss_key], dtype=float)
                if should_plot_component(arr_d):
                    ax_abs.plot(x, arr_d, ':', linewidth=line_comp, color=color, label=rf'{label} $D_{{\mathrm{{{sym}}}}}$')

        ax_abs.set_xlabel(xlabel, fontproperties=font)
        norm_suffix = r' ($\rho_B^{-1}$)' if normalized else ''
        vol_suffix = r' / Volume' if normalize_by_volume else ''
        ax_abs.set_ylabel(f'Integrated Production / Dissipation{norm_suffix}{vol_suffix}', fontproperties=font)
        if x_scale == 'log':
            ax_abs.set_xscale('log')
            if x_axis == 'years':
                ax_abs.set_xlabel('Time log[yr]', fontproperties=font)
            else:
                ax_abs.set_xlabel('Redshift log[z]', fontproperties=font)
        if y_scale == 'log':
            ax_abs.set_yscale('log')
            ax_abs.set_ylabel(f'Integrated Production / Dissipation log{norm_suffix}{vol_suffix}', fontproperties=font)
        if not cancel_limits and xlim:
            ax_abs.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax_abs.set_ylim(ylim[0], ylim[1])
        if cancel_limits and x_axis == 'zeta':
            ax_abs.invert_xaxis()

        ax_abs.grid(alpha=0.3)
        ax_abs.set_title(f'{title} - {region_label}', y=y_title, fontproperties=font_title)
        _set_combined_legend(ax_abs, None)
        fig_abs.tight_layout()
        figures.append(fig_abs)

    # Fractional contributions and net efficiency
    if plot_fractional:
        fig_frac, ax_frac = plt.subplots(figsize=figure_size, dpi=dpi)

        for prefix, label, color, sym in component_map:
            frac_p_key = f'int_PD_frac_{prefix}_prod'
            frac_d_key = f'int_PD_frac_{prefix}_diss'
            if frac_p_key in pd_data:
                arr_fp = np.asarray(pd_data[frac_p_key], dtype=float)
                if should_plot_component(arr_fp, threshold=epsilon):
                    ax_frac.plot(x, arr_fp, '--', linewidth=line_comp, color=color, label=rf'{label} $p_{{\mathrm{{{sym}}}}}$')
            if frac_d_key in pd_data:
                arr_fd = np.asarray(pd_data[frac_d_key], dtype=float)
                if should_plot_component(arr_fd, threshold=epsilon):
                    ax_frac.plot(x, -arr_fd, ':', linewidth=line_comp, color=color, label=rf'{label} $d_{{\mathrm{{{sym}}}}}$')

        if 'int_PD_iota' in pd_data:
            iota = np.asarray(pd_data['int_PD_iota'], dtype=float)
            ax_frac.plot(x, iota, '-', linewidth=line_main, color=color_efficiency, label=r'Net Efficiency $\iota$')

        ax_frac.set_xlabel(xlabel, fontproperties=font)
        ax_frac.set_ylabel('Fractional Contribution (+prod / -diss)', fontproperties=font)
        if x_scale == 'log':
            ax_frac.set_xscale('log')
            if x_axis == 'years':
                ax_frac.set_xlabel('Time log[yr]', fontproperties=font)
            else:
                ax_frac.set_xlabel('Redshift log[z]', fontproperties=font)
        if not cancel_limits and xlim:
            ax_frac.set_xlim(xlim[0], xlim[1])
        ax_frac.set_ylim(-1.05, 1.05)
        if cancel_limits and x_axis == 'zeta':
            ax_frac.invert_xaxis()

        ax_frac.grid(alpha=0.3)
        ax_frac.set_title(f'{title} (Fractions) - {region_label}', y=y_title, fontproperties=font_title)
        ax_frac_aux = _overlay_cumulative_magnetic_energy(ax_frac)
        _set_combined_legend(ax_frac, ax_frac_aux)
        fig_frac.tight_layout()
        figures.append(fig_frac)

    # Net contributions: per-component net and total net curves
    if plot_net:
        fig_net, ax_net = plt.subplots(figsize=figure_size, dpi=dpi)

        for prefix, label, color, sym in component_map:
            prod_key = f'int_{prefix}_prod'
            diss_key = f'int_{prefix}_diss'
            if prod_key in pd_data and diss_key in pd_data:
                arr_p = units * np.asarray(pd_data[prod_key], dtype=float)
                arr_d = units * np.asarray(pd_data[diss_key], dtype=float)
                net_i = arr_p - arr_d
                if should_plot_component(net_i, threshold=epsilon):
                    ax_net.plot(x, net_i, '--', linewidth=line_comp, color=color, label=rf'{label} $N_{{\mathrm{{{sym}}}}}$')

        ax_net.plot(x, itemized_net, '--', linewidth=line_main, color=color_itemized_net, label='Net total (itemized)')
        if compact_net is not None:
            ax_net.plot(x, compact_net, '-', linewidth=line_main, color=color_compact_net, label='Net total (compact)')

        ax_net.set_xlabel(xlabel, fontproperties=font)
        norm_suffix = r' ($\rho_B^{-1}$)' if normalized else ''
        vol_suffix = r' / Volume' if normalize_by_volume else ''
        ax_net.set_ylabel(f'Integrated Net Contribution{norm_suffix}{vol_suffix}', fontproperties=font)
        if x_scale == 'log':
            ax_net.set_xscale('log')
            if x_axis == 'years':
                ax_net.set_xlabel('Time log[yr]', fontproperties=font)
            else:
                ax_net.set_xlabel('Redshift log[z]', fontproperties=font)
        if not cancel_limits and xlim:
            ax_net.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax_net.set_ylim(ylim[0], ylim[1])
        if cancel_limits and x_axis == 'zeta':
            ax_net.invert_xaxis()

        ax_net.grid(alpha=0.3)
        ax_net.set_title(f'{title} (Net) - {region_label}', y=y_title, fontproperties=font_title)
        ax_net_aux = _overlay_cumulative_magnetic_energy(ax_net)
        _set_combined_legend(ax_net, ax_net_aux)
        fig_net.tight_layout()
        figures.append(fig_net)

    if save and figures:
        if folder is None:
            folder = os.getcwd()

        sim_info = f'{induction_params["up_to_level"]}_{induction_params["F"]}_{induction_params["vir_kind"]}vir_{induction_params["rad_kind"]}rad_{induction_params["region"]}Region'
        axis_info = f'{x_axis}_{x_scale}_{y_scale}'
        if cancel_limits:
            limit_info = 'cancel_limits'
        else:
            limit_info = f'{xlim[0] if xlim else "auto"}_{ylim[0] if ylim else "auto"}_{ylim[1] if ylim else "auto"}'
        diff_cfg = induction_params.get('differentiation', {})
        if diff_cfg.get('buffer', False) == True:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol', ''))
            buffer_info = f'Buffered_{diff_cfg.get("interpol", "")}_siblings_{diff_cfg.get("use_siblings", False)}'
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
        else:
            buffer_info = 'NoBuffer'

        base_title = '_'.join(title.split()[:4])
        units_info = f'pd_{"physical" if not normalized else "normalized"}'
        if plot_absolute and len(figures) >= 1:
            fname_abs = f'{folder}/{run}_{base_title}_prod_diss_abs_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_abs = safe_filename(fname_abs, verbose=verbose)
            figures[0].savefig(fname_abs, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation absolute plot saved as: {fname_abs}')

        abs_idx = 0 if plot_absolute else None
        frac_idx = None
        net_idx = None
        next_idx = 0
        if plot_absolute:
            abs_idx = next_idx
            next_idx += 1
        if plot_fractional:
            frac_idx = next_idx
            next_idx += 1
        if plot_net:
            net_idx = next_idx

        if frac_idx is not None and len(figures) > frac_idx:
            fname_frac = f'{folder}/{run}_{base_title}_prod_diss_frac_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_frac = safe_filename(fname_frac, verbose=verbose)
            figures[frac_idx].savefig(fname_frac, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation fractional plot saved as: {fname_frac}')

        if net_idx is not None and len(figures) > net_idx:
            fname_net = f'{folder}/{run}_{base_title}_prod_diss_net_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_net = safe_filename(fname_net, verbose=verbose)
            figures[net_idx].savefig(fname_net, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation net plot saved as: {fname_net}')

    if verbose and figures:
        log_message('Production/Dissipation evolution plots created', tag='production_dissipation', level=1)

    return figures

def plot_induction_radial_profiles(profile_data, plot_params, induction_params,
                                 grid_t, grid_zeta, rad,
                                 verbose=True, save=False, folder=None):
    """
    Plot radial profiles of magnetic energy and induction components.

    Args:
        - profile_data: dictionary with the data from induction_radial_profiles() with keys (each is a list/array over snapshots):
            'clus_b2_profile',
            'MIE_diver_B2_profile',
            'MIE_compres_B2_profile',
            'MIE_stretch_B2_profile',
            'MIE_advec_B2_profile',
            'MIE_drag_B2_profile',
            'MIE_total_B2_profile',
            'ind_b2_profile',
            'kinetic_energy_profile',
            'clus_rho_rho_b_profile'
            'profile_bin_centers': radial bin centers array
        - plot_params: dictionary with plotting parameters:
            - it_indx: iteration indexes to select snapshots
            - x_scale: 'lin' or 'log' (radial axis)
            - y_scale: 'lin' or 'log' (energy axis)
            - xlim: [xlimo, xlimf] or None
            - ylim: [ylimo, ylimf] or None
            - line_widths: [line1, line2]
            - plot_type: 'raw', 'smoothed', or 'interpolated' to choose plot style
            - smoothing_sigma: sigma for Gaussian smoothing (only for 'smooth' type)
            - interpolation_points: number of points for interpolation (only for 'interpolated' type)
            - interpolation_kind: 'linear', 'cubic', or 'nearest' for interpolation method
            - title: title for the plots (default: 'Magnetic Field Profile')
            - dpi: dots per inch for saved plots (default: 300)
            - run: identifier for filenames
        - induction_params: dictionary with simulation metadata:
            - units: energy unit conversion
            - F: size factor
            - level: refinement level
            - vir_kind: type of virial radius
            - rad_kind: type of radius used
            - size: simulation box size
            - buffer: bool indicating if buffer region is used
            - interpol: interpolation method for buffer
            - stencil: stencil type for induction calculation
            - up_to_level: maximum refinement level
            - sims: list of simulation identifiers
        - grid_t: time grid (in simulation units)
        - grid_zeta: redshift grid
        - rad: characteristic radius for normalization
        - verbose: bool for verbose output
        - save: bool to save plots
        - folder: folder to save plots (if None, uses current directory)
        
    Returns:
        - List of figure objects
        
    Author: Marco Molina
    """

    # Validate inputs / defaults
    assert plot_params.get('x_scale', 'lin') in ['lin', 'log'], "x_scale must be 'lin' or 'log'"
    assert plot_params.get('y_scale', 'log') in ['lin', 'log'], "y_scale must be 'lin' or 'log'"
    plot_type = plot_params.get('plot_type', 'raw')
    assert plot_type in ['raw', 'smoothed', 'interpolated'], "plot_type must be 'raw', 'smoothed', or 'interpolated'"
    assert plot_params.get('interpolation_kind', 'linear') in ['linear', 'cubic', 'nearest'], "interpolation_kind must be 'linear', 'cubic', or 'nearest'"
    assert plot_params.get('smoothing_sigma', 1.10) > 0, "smoothing_sigma must be a positive number"
    assert plot_params.get('it_indx', None) is not None, "it_indx must be provided in plot_params"
    assert len(plot_params['it_indx']) > 0, "it_indx must contain at least one index"

    # Extract parameters from plot_params
    it_indx = plot_params['it_indx']
    x_scale = plot_params.get('x_scale', 'lin')
    y_scale = plot_params.get('y_scale', 'log')
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    rylim = plot_params.get('rylim', None)
    dylim = plot_params.get('dylim', None)
    fixed_legend = bool(plot_params.get('fixed_legend', False))
    figure_size = plot_params.get('figure_size', [12, 8])
    line_widths = plot_params.get('line_widths', [3, 1.5])
    title = plot_params.get('title', 'Magnetic Field Radial Profiles')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    y_title = 1.1

    # Parameters specific to plot type
    if plot_type == 'smoothed':
        smoothing_sigma = plot_params.get('smoothing_sigma', 1.10)
    elif plot_type == 'interpolated':
        interpolation_points = plot_params.get('interpolation_points', 500)
        interpolation_kind = plot_params.get('interpolation_kind', 'cubic')
    profile_bin_centers = profile_data.get('profile_bin_centers', None)
    if profile_bin_centers is None:
        raise KeyError("profile_bin_centers not found in profile_data")
    # Pick the first non-empty array
    if isinstance(profile_bin_centers, (list, tuple, np.ndarray)) and len(profile_bin_centers) > 0 and not isinstance(profile_bin_centers, np.ndarray):
        pb = None
        for p in profile_bin_centers:
            if p is None:
                continue
            p_arr = np.asarray(p)
            if p_arr.size > 0:
                pb = p_arr
                break
        if pb is None:
            raise ValueError("profile_bin_centers list contains only empty entries")
        profile_bin_centers = pb
    else:
        profile_bin_centers = np.asarray(profile_bin_centers)
    # Ensure a 1D array
    profile_bin_centers = profile_bin_centers.flatten()
    
    # Extract induction parameters
    factor_F = induction_params.get('F', 1.0)
    region = induction_params.get('region', None)
    units = plot_params.get('units', induction_params.get('units', None))
    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_negative_interval = palette.get('negative_interval', DEFAULT_PLOT_PALETTE['negative_interval'])
    
    # Use string identifiers for axis types to avoid collisions when values are equal
    AXIS_MAIN = 'main'      # Induction main axis
    AXIS_ENERGY = 'energy'  # Magnetic/kinetic energy reference
    AXIS_DENSITY = 'density' # Density

    plot_density = bool(plot_params.get('plot_density', False))
    plot_magnetic_energy = bool(plot_params.get('plot_magnetic_energy', False))
    
    # Map axis types to their scaling factors
    if units is None:
        units_y_1 = 1.0
        units_y_2 = 1.0
        units_y_3 = 1.0
    elif units == energy_to_erg:
        units_y_1 = (energy_to_erg / (length_to_mpc)**3) / time_to_s
        units_y_2 = energy_to_erg / (length_to_mpc)**3
        units_y_3 = density_to_cgs
    elif units == energy_to_J:
        units_y_1 = (energy_to_J / (length_to_mpc)**3) / time_to_s
        units_y_2 = energy_to_J / (length_to_mpc)**3
        units_y_3 = density_to_sunMpc3
    else:
        units_y_1 = units_y_2 = units_y_3 = 1.0
        
        # Set up matplotlib parameters
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18
    })
    
    # Define font properties
    font = FontProperties()
    font.set_style('normal')
    font.set_weight('normal')
    font.set_size(12)
    
    font_title = FontProperties()
    font_title.set_style('normal')
    font_title.set_weight('bold')
    font_title.set_size(20)
    
    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)
    
    y_title = 1.02
    line1, line2 = line_widths
    component_alpha = float(np.clip(plot_params.get('component_alpha', 0.75), 0.05, 1.0))

    # Radial axis normalized by R_vir
    r = np.asarray(profile_bin_centers) / float(rad)
    nbins = profile_bin_centers.shape[0]
    
    t = [grid_t[i] * time_to_yr for i in it_indx]
    z = np.array([grid_zeta[i] for i in it_indx])
    if z[-1] < 0:
        z[-1] = abs(z[-1])

    # Compute plotted arrays with units (each is a list indexed by snapshots)
    # Robustly coerce scalars/missing values to per-bin arrays to avoid ndimage axis errors.
    def series_array(key, scale):
        raw = profile_data.get(key, None)
        out = []
        for i in range(len(it_indx)):
            value = 0.0
            if raw is not None:
                try:
                    value = raw[i]
                except Exception:
                    value = 0.0

            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = np.full(nbins, float(arr), dtype=float)
            else:
                arr = arr.ravel()
                if arr.size == 0:
                    arr = np.zeros(nbins, dtype=float)
                elif arr.size == 1:
                    arr = np.full(nbins, float(arr[0]), dtype=float)
                elif arr.size != nbins:
                    x_old = np.linspace(0.0, 1.0, arr.size)
                    x_new = np.linspace(0.0, 1.0, nbins)
                    arr = np.interp(x_new, x_old, arr)

            out.append(scale * arr)
        return out

    components_cfg = induction_params.get('components', {})
    plot_kinetic_energy = bool(components_cfg.get('kinetic_energy', True))

    kinetic_energy_profile = series_array('kinetic_energy_profile', units_y_2)
    clus_b2_profile = series_array('clus_b2_profile', units_y_2)
    clus_rho_rho_b_profile = series_array('clus_rho_rho_b_profile', units_y_3)
    diver_profile = series_array('MIE_diver_B2_profile', units_y_1)
    compres_profile = series_array('MIE_compres_B2_profile', units_y_1)
    stretch_profile = series_array('MIE_stretch_B2_profile', units_y_1)
    advec_profile = series_array('MIE_advec_B2_profile', units_y_1)
    drag_profile = series_array('MIE_drag_B2_profile', units_y_1)
    total_profile = series_array('MIE_total_B2_profile', units_y_1)
    ind_b2_profile = series_array('ind_b2_profile', units_y_1)
    # post_ind_b2_profile = [units_y_1 * safe_get('post_ind_b2_profile')[i] for i in range(len(it_indx))]

    
    # Prepare data based on plot type
    if plot_type == 'smoothed':
        # Apply Gaussian smoothing (operate per snapshot)
        clus_b2_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in clus_b2_profile]
        kinetic_energy_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in kinetic_energy_profile]
        clus_rho_rho_b_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in clus_rho_rho_b_profile]
        diver_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in diver_profile]
        compres_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in compres_profile]
        stretch_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in stretch_profile]
        advec_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in advec_profile]
        drag_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in drag_profile]
        total_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in total_profile]
        ind_b2_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in ind_b2_profile]
        # post_ind_b2_profile = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in post_ind_b2_profile]
        
        r_pro = r
        plot_suffix = f'smoothed_sigma_{smoothing_sigma}'

    elif plot_type == 'interpolated':
        # Create interpolations per snapshot, produce callables
        r_new = np.linspace(min(r), max(r), num=interpolation_points, endpoint=True)
        clus_b2_profile = [interp1d(r, clus_b2_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(clus_b2_profile))]
        kinetic_energy_profile = [interp1d(r, kinetic_energy_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(kinetic_energy_profile))]
        clus_rho_rho_b_profile = [interp1d(r, clus_rho_rho_b_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(clus_rho_rho_b_profile))]
        diver_profile = [interp1d(r, diver_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(diver_profile))]
        compres_profile = [interp1d(r, compres_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(compres_profile))]
        stretch_profile = [interp1d(r, stretch_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(stretch_profile))]
        advec_profile = [interp1d(r, advec_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(advec_profile))]
        drag_profile = [interp1d(r, drag_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(drag_profile))]
        total_profile = [interp1d(r, total_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(total_profile))]
        ind_b2_profile = [interp1d(r, ind_b2_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(ind_b2_profile))]
        # post_ind_b2_profile = [interp1d(r, post_ind_b2_profile[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for i in range(len(post_ind_b2_profile))]
        
        r_pro = r_new
        plot_suffix = f'{interpolation_kind}_interpolated_{interpolation_points}_points'

    else:  # raw
        r_pro = r
        plot_suffix = 'raw'

    figures = []
    
    # Components configuration: (data_list, label, color, lw, linestyle, axis_type, alpha)
    components_configs = [
        # Induction main axis
        (total_profile, '...from Compact Induction', palette.get('induction_compact', DEFAULT_PLOT_PALETTE['induction_compact']), line1, '-', AXIS_MAIN, 1.0),
        (ind_b2_profile, '...from Itemized Induction', palette.get('induction_itemized', DEFAULT_PLOT_PALETTE['induction_itemized']), line1, '--', AXIS_MAIN, 1.0),
        # (post_ind_b2_profile, '...from Post-Itemize Induction', "#ffbb78", line1, '-.', AXIS_MAIN),
        # Individual components (main axis)
        (compres_profile, r'Compression $\Gamma_{\mathrm{comp}}$', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), line2, '--', AXIS_MAIN, component_alpha),
        (stretch_profile, r'Stretching $\Gamma_{\mathrm{str}}$', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), line2, '--', AXIS_MAIN, component_alpha),
        (advec_profile, r'Advection $\Gamma_{\mathrm{adv}}$', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), line2, '--', AXIS_MAIN, component_alpha),
        (diver_profile, r'Divergence $\Gamma_{\mathrm{div}}$', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), line2, '--', AXIS_MAIN, component_alpha),
        (drag_profile, r'Cosmic Drag $\Gamma_{\mathrm{drag}}$', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), line2, '--', AXIS_MAIN, component_alpha)
    ]

    if plot_kinetic_energy:
        components_configs.insert(
            0,
            (kinetic_energy_profile, 'Kinetic Energy Density', palette.get('kinetic_energy', DEFAULT_PLOT_PALETTE['kinetic_energy']), line1, '-', AXIS_ENERGY, 1.0)
        )

    if plot_magnetic_energy:
        components_configs.insert(
            0,
            (clus_b2_profile, 'Magnetic Energy Density', palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']), line1, '-', AXIS_ENERGY, 1.0)
        )

    if plot_density:
        components_configs.insert(
            0,
            (clus_rho_rho_b_profile, 'Density', palette.get('density', DEFAULT_PLOT_PALETTE['density']), line1, '-', AXIS_DENSITY, 1.0)
        )

    # Decide which components have data (per snapshot we check existence)
    def has_nonzero(arr_or_callable, snap_idx, threshold=induction_params.get('epsilon', 1e-30)):
        # If callable (interp), assume has values (could be NaN outside range)
        if callable(arr_or_callable):
            try:
                y = arr_or_callable(r_pro)
                return np.any(~np.isnan(y)) and np.any(np.abs(y) > threshold)
            except Exception:
                return False
        # If list-like of arrays
        try:
            a = arr_or_callable[snap_idx]
            return np.any(np.abs(np.asarray(a)) > threshold)
        except Exception:
            return False
        
    # Helper to plot signed data: single continuous line with per-segment styling
    def plot_signed(ax, x, y, lw, ls, color, label, alpha=1.0, eps=induction_params.get('epsilon', 1e-30)):
        """
        Plot a single continuous line where:
            - positive intervals use linestyle `ls`
            - negative intervals use linestyle ':'
        No gaps between positive and negative segments.
        Returns a single Line2D handle for legend.
        """
        from matplotlib.collections import LineCollection
        from matplotlib.lines import Line2D
        
        y = np.asarray(y)
        x = np.asarray(x)

        # mask out NaNs
        valid = ~np.isnan(y)
        if not np.any(valid):
            return None

        xv = x[valid]
        yv = y[valid]
        yabs = np.maximum(np.abs(yv), eps)  # clamp to avoid log issues

        # Build segments with sign-dependent linestyle
        segments = []
        sign_styles = []  # True = positive (ls), False = negative (':')
        
        for i in range(len(xv) - 1):
            seg = np.array([[xv[i], yabs[i]], [xv[i+1], yabs[i+1]]])
            segments.append(seg)
            # Use the sign of the first point in the segment
            sign_styles.append(yv[i] >= 0)

        if not segments:
            return None

        # Plot base thin continuous line (no legend)
        ax.plot(xv, yabs, linestyle='-', linewidth=max(lw * 0.5, 0.3), 
            color=color, alpha=0.2 * alpha, label='_nolegend_')

        # Overlay segments with sign-dependent dashes
        lc_pos = LineCollection(
            [seg for seg, is_pos in zip(segments, sign_styles) if is_pos],
            linewidths=lw, colors=color, linestyles=ls, label='_nolegend_', alpha=alpha
        )
        lc_neg = LineCollection(
            [seg for seg, is_pos in zip(segments, sign_styles) if not is_pos],
            linewidths=lw, colors=color, linestyles=':', label='_nolegend_', alpha=alpha
        )
        
        ax.add_collection(lc_pos)
        ax.add_collection(lc_neg)

        # Return a dummy handle for legend entry
        h_legend, = ax.plot([], [], linestyle=ls, linewidth=lw, color=color, alpha=alpha, label=label)
        return h_legend

    def _auto_limits(values, scale, pad=0.10):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if scale == 'log':
            values = values[values > 0]
            if values.size == 0:
                return None
            vmin = np.log10(np.min(values))
            vmax = np.log10(np.max(values))
            span = max(vmax - vmin, 1e-6)
            return 10 ** (vmin - pad * span), 10 ** (vmax + pad * span)
        vmin = np.min(values)
        vmax = np.max(values)
        span = max(vmax - vmin, 1e-12)
        return vmin - pad * span, vmax + pad * span
    
    for snap_i in range(len(it_indx)):
        fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
        ax_energy = None
        ax_density = None
        if plot_magnetic_energy or plot_kinetic_energy:
            ax_energy = ax1.twinx()
        if plot_density:
            ax_density = ax1.twinx()
            if ax_energy is not None:
                ax_density.spines["right"].set_position(("axes", 1.12))
            ax_density.set_frame_on(True)
            ax_density.patch.set_visible(False)
            for sp in ax_density.spines.values():
                sp.set_visible(True)

        snap_z = np.abs(np.round(grid_zeta[it_indx[snap_i]], 2))
        z_text = f"{snap_z:6.2f}"
        ax1.set_title(f'{title} - z = {z_text}, $R_{{Vir}}$ = {np.round(rad,1)} Mpc', y=y_title, fontproperties=font_title)

        unique_handles = []
        unique_labels = []
        y_main_vals = []
        y_energy_vals = []
        y_density_vals = []

        from matplotlib.lines import Line2D
        neg_note = Line2D([0], [0], color=color_negative_interval, linestyle=':', linewidth=line2, label='Negative Interval')
        unique_handles.append(neg_note)
        unique_labels.append('Negative Interval')

        for (data_list, label, color, lw, ls, axis_type, alpha_curve) in components_configs:
            if not has_nonzero(data_list, snap_i):
                continue

            if callable(data_list[snap_i]):
                y = data_list[snap_i](r_pro)
            else:
                y = np.asarray(data_list[snap_i])
                if plot_type == 'interpolated' and y.size == r.size and r_pro.size != r.size:
                    y = np.interp(r_pro, r, y)

            if axis_type == AXIS_ENERGY:
                if ax_energy is None:
                    continue
                _ = plot_signed(ax_energy, r_pro, y, lw, ls, color, label, alpha=alpha_curve)
                eps = induction_params.get('epsilon', 1e-30)
                y_clean = np.asarray(y)
                y_clean = y_clean[np.isfinite(y_clean)]
                if y_clean.size > 0:
                    y_energy_vals.append(np.maximum(np.abs(y_clean), eps))
            elif axis_type == AXIS_DENSITY:
                if ax_density is None:
                    continue
                ax_density.plot(r_pro, y, linewidth=lw, linestyle=ls, color=color, alpha=alpha_curve, label=label)
                y_clean = np.asarray(y)
                y_clean = y_clean[np.isfinite(y_clean)]
                if y_clean.size > 0:
                    y_density_vals.append(y_clean)
            else:
                _ = plot_signed(ax1, r_pro, y, lw, ls, color, label, alpha=alpha_curve)
                y_clean = np.asarray(y)
                y_clean = y_clean[np.isfinite(y_clean)]
                if y_clean.size > 0:
                    eps = induction_params.get('epsilon', 1e-30)
                    y_main_vals.append(np.maximum(np.abs(y_clean), eps))

        if x_scale == 'log':
            ax1.set_xscale('log')
            ax1.set_xlabel('Radial Distance log[r/$R_{Vir}$]', fontproperties=font)
        else:
            ax1.set_xlabel('Radial Distance [r/$R_{Vir}$]', fontproperties=font)
        ax1.tick_params(axis='x', labelsize=11)

        if xlim is not None:
            ax1.set_xlim(xlim[0], xlim[1])

        if y_scale == 'log':
            ax1.set_yscale('log')
            if ax_energy is not None:
                ax_energy.set_yscale('log')
            if ax_density is not None:
                ax_density.set_yscale('log')

        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
        elif y_main_vals:
            y_main_auto = _auto_limits(np.concatenate(y_main_vals), y_scale)
            if y_main_auto is not None:
                ax1.set_ylim(y_main_auto[0], y_main_auto[1])

        if ax_energy is not None:
            if rylim is not None:
                ax_energy.set_ylim(rylim[0], rylim[1])
            elif y_energy_vals:
                y_energy_auto = _auto_limits(np.concatenate(y_energy_vals), y_scale)
                if y_energy_auto is not None:
                    ax_energy.set_ylim(y_energy_auto[0], y_energy_auto[1])

        if ax_density is not None:
            if dylim is not None:
                ax_density.set_ylim(dylim[0], dylim[1])
            elif y_density_vals:
                y_density_auto = _auto_limits(np.concatenate(y_density_vals), y_scale)
                if y_density_auto is not None:
                    ax_density.set_ylim(y_density_auto[0], y_density_auto[1])

        if units == energy_to_erg:
            ax1.set_ylabel('Induction Density (erg/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Energy Density (erg/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (g/cm³)', fontproperties=font)
        elif units == energy_to_J:
            ax1.set_ylabel('Induction Density (J/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Energy Density (J/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (M$_{\odot}$/Mpc³)', fontproperties=font)
        else:
            ax1.set_ylabel('Induction Density (arb. units / s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Energy Density (arb. units)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (arb. units)', fontproperties=font)

        ax1.grid(alpha=0.3)
        if ax_energy is not None:
            ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        if ax_density is not None:
            ax_density.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax_energy.get_legend_handles_labels() if ax_energy is not None else ([], [])
        h3, l3 = ax_density.get_legend_handles_labels() if ax_density is not None else ([], [])
        all_handles = h1 + h2 + h3
        all_labels = l1 + l2 + l3
        seen = set()
        for hh, ll in zip(all_handles, all_labels):
            if ll and not ll.startswith('_') and ll not in seen:
                seen.add(ll)
                unique_handles.append(hh)
                unique_labels.append(ll)

        if unique_handles:
            if fixed_legend:
                ax1.legend(unique_handles, unique_labels, prop=font_legend,
                           loc='lower left', bbox_to_anchor=(0.02, 0.02),
                           bbox_transform=ax1.transAxes, ncol=2, frameon=True)
            else:
                fig1.legend(unique_handles, unique_labels, prop=font_legend,
                            loc='lower left', bbox_to_anchor=(0.02, 0.02),
                            bbox_transform=ax1.transAxes, ncol=2, frameon=True)

        fig1.tight_layout()

        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
        elif y_main_vals:
            y_main_auto = _auto_limits(np.concatenate(y_main_vals), y_scale)
            if y_main_auto is not None:
                ax1.set_ylim(y_main_auto[0], y_main_auto[1])

        if ax_energy is not None:
            if rylim is not None:
                ax_energy.set_ylim(rylim[0], rylim[1])
            elif y_energy_vals:
                y_energy_auto = _auto_limits(np.concatenate(y_energy_vals), y_scale)
                if y_energy_auto is not None:
                    ax_energy.set_ylim(y_energy_auto[0], y_energy_auto[1])

        if ax_density is not None:
            if dylim is not None:
                ax_density.set_ylim(dylim[0], dylim[1])
            elif y_density_vals:
                y_density_auto = _auto_limits(np.concatenate(y_density_vals), y_scale)
                if y_density_auto is not None:
                    ax_density.set_ylim(y_density_auto[0], y_density_auto[1])

        figures.append(fig1)
        
    if verbose:
        # compute plotted labels directly and consistently
        plotted_labels = [
            label
            for (data_list, label, *_)
            in components_configs
            if any(has_nonzero(data_list, si) for si in range(len(it_indx)))
        ]
        print(f'Plotting... Radial profile plots created')
        print(f'Components plotted: {", ".join(plotted_labels)}')

    # Save the figures if requested
    if save:
        if folder is None:
            folder = os.getcwd()

        sim_info = f'{induction_params.get("up_to_level","")}_{factor_F}_{induction_params.get("vir_kind","")}vir_{induction_params.get("rad_kind","")}rad_{region}Region'
        axis_info = f'{x_scale}_{y_scale}'
        limit_info = f'{xlim[0] if xlim else "auto"}_{ylim[0] if ylim else "auto"}_{ylim[1] if ylim else "auto"}'

        diff_cfg = induction_params.get('differentiation', {})
        if diff_cfg.get('buffer', False) == True:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol',''))
            buffer_info = f'Buffered_{diff_cfg.get("interpol","")}_siblings_{diff_cfg.get("use_siblings","")}'
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
        else:
            buffer_info = 'NoBuffer'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figures):
            file_title = '_'.join(title.split()[:3])
            file_name = f'{folder}/{run}_{file_title}_induction_profile_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil","")}_{plot_suffix}_{i}_{timestamp}.png'
            file_name = safe_filename(file_name, verbose=verbose)
            fig.savefig(file_name, dpi=dpi)
            if verbose:
                print(f'Saved figure {i+1}/{len(figures)}: {file_name}')

    return figures


def plot_production_dissipation_radial_profiles(profile_data, plot_params, induction_params,
                                            grid_t, grid_zeta, rad,
                                            verbose=True, save=False, folder=None):
    """
    Plot radial profiles for production/dissipation components.
    
    Args:
        profile_data (dict): Dictionary containing radial profile data arrays.
        plot_params (dict): Dictionary of plotting parameters and options.
        induction_params (dict): Dictionary of induction calculation parameters.
        grid_t (array-like): Array of snapshot times.
        grid_zeta (array-like): Array of snapshot redshifts.
        rad (float): Virial radius for normalization.
        verbose (bool): If True, print detailed information about the plotting process.
        save (bool): If True, save the generated figures to disk.
        folder (str): Directory to save figures if `save` is True. Defaults to current working directory.
        
    Returns:
        list: A list of matplotlib Figure objects created for each snapshot.
        
    Author: Marco Molina
    """

    assert plot_params.get('x_scale', 'lin') in ['lin', 'log'], "x_scale must be 'lin' or 'log'"
    assert plot_params.get('y_scale', 'log') in ['lin', 'log'], "y_scale must be 'lin' or 'log'"
    plot_type = plot_params.get('plot_type', 'raw')
    assert plot_type in ['raw', 'smoothed', 'interpolated'], "plot_type must be 'raw', 'smoothed', or 'interpolated'"
    assert plot_params.get('interpolation_kind', 'linear') in ['linear', 'cubic', 'nearest'], "interpolation_kind must be 'linear', 'cubic', or 'nearest'"
    assert plot_params.get('smoothing_sigma', 1.10) > 0, "smoothing_sigma must be a positive number"
    assert plot_params.get('it_indx', None) is not None, "it_indx must be provided in plot_params"
    assert len(plot_params['it_indx']) > 0, "it_indx must contain at least one index"

    it_indx = plot_params['it_indx']
    x_scale = plot_params.get('x_scale', 'lin')
    y_scale = plot_params.get('y_scale', 'log')
    xlim = plot_params.get('xlim', None)
    ylim = plot_params.get('ylim', None)
    fixed_legend = bool(plot_params.get('fixed_legend', False))
    figure_size = plot_params.get('figure_size', [12, 8])
    line_widths = plot_params.get('line_widths', [3, 1.5])
    title = plot_params.get('title', 'Production and Dissipation Radial Profile')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')

    if plot_type == 'smoothed':
        smoothing_sigma = plot_params.get('smoothing_sigma', 1.10)
    elif plot_type == 'interpolated':
        interpolation_points = plot_params.get('interpolation_points', 500)
        interpolation_kind = plot_params.get('interpolation_kind', 'cubic')

    profile_bin_centers = profile_data.get('profile_bin_centers', None)
    if profile_bin_centers is None:
        raise KeyError("profile_bin_centers not found in profile_data")
    if isinstance(profile_bin_centers, (list, tuple, np.ndarray)) and len(profile_bin_centers) > 0 and not isinstance(profile_bin_centers, np.ndarray):
        pb = None
        for p in profile_bin_centers:
            if p is None:
                continue
            p_arr = np.asarray(p)
            if p_arr.size > 0:
                pb = p_arr
                break
        if pb is None:
            raise ValueError("profile_bin_centers list contains only empty entries")
        profile_bin_centers = pb
    else:
        profile_bin_centers = np.asarray(profile_bin_centers)
    profile_bin_centers = profile_bin_centers.flatten()

    factor_F = induction_params.get('F', 1.0)
    region = induction_params.get('region', None)
    units = plot_params.get('units', induction_params.get('units', None))
    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_prod = palette.get('production', DEFAULT_PLOT_PALETTE['production'])
    color_diss = palette.get('dissipation', DEFAULT_PLOT_PALETTE['dissipation'])
    color_itemized_net = palette.get('net_itemized', DEFAULT_PLOT_PALETTE['net_itemized'])
    color_compact_net = palette.get('net_compact', DEFAULT_PLOT_PALETTE['net_compact'])
    color_efficiency = palette.get('efficiency', DEFAULT_PLOT_PALETTE['efficiency'])
    color_measured_energy = palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
    color_density = palette.get('density', DEFAULT_PLOT_PALETTE['density'])
    plot_density = bool(plot_params.get('plot_density', False))
    plot_magnetic_energy = bool(plot_params.get('plot_magnetic_energy', False))

    if units is None:
        units_y = 1.0
        units_energy = 1.0
        units_density = 1.0
    elif units == energy_to_erg:
        units_y = (energy_to_erg / (length_to_mpc) ** 3) / time_to_s
        units_energy = energy_to_erg / (length_to_mpc) ** 3
        units_density = density_to_cgs
    elif units == energy_to_J:
        units_y = (energy_to_J / (length_to_mpc) ** 3) / time_to_s
        units_energy = energy_to_J / (length_to_mpc) ** 3
        units_density = density_to_sunMpc3
    else:
        units_y = 1.0
        units_energy = 1.0
        units_density = 1.0

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 20
    })

    font = FontProperties()
    font.set_style('normal')
    font.set_weight('normal')
    font.set_size(12)

    font_title = FontProperties()
    font_title.set_style('normal')
    font_title.set_weight('bold')
    font_title.set_size(17)

    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)

    line_main, line_comp = line_widths
    component_alpha = float(np.clip(plot_params.get('component_alpha', 0.75), 0.05, 1.0))
    area_alpha = float(np.clip(plot_params.get('area_alpha', 0.24), 0.0, 1.0))
    y_title = 1.05

    r = np.asarray(profile_bin_centers) / float(rad)
    nbins = profile_bin_centers.shape[0]

    def safe_get(key):
        return profile_data.get(key, [np.zeros(nbins) for _ in range(len(it_indx))])

    component_map = [
        ('MIE_compres_B2', 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
        ('MIE_stretch_B2', 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
        ('MIE_advec_B2', 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
        ('MIE_diver_B2', 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
        ('MIE_drag_B2', 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag')
    ]

    def _series(key):
        return [units_y * np.asarray(safe_get(key)[i], dtype=float) for i in range(len(it_indx))]

    itemized_prod = _series('MIE_total_B2_prod_itemized_profile')
    itemized_diss = _series('MIE_total_B2_diss_itemized_profile')
    itemized_net = _series('MIE_total_B2_net_itemized_profile')
    compact_prod = _series('MIE_total_B2_prod_compact_profile')
    compact_diss = _series('MIE_total_B2_diss_compact_profile')
    compact_net = _series('MIE_total_B2_net_compact_profile')
    clus_b2_profile = [units_energy * np.asarray(safe_get('clus_b2_profile')[i], dtype=float) for i in range(len(it_indx))]
    clus_rho_rho_b_profile = [units_density * np.asarray(safe_get('clus_rho_rho_b_profile')[i], dtype=float) for i in range(len(it_indx))]

    component_prod = {prefix: _series(f'{prefix}_prod_profile') for prefix, _, _, _ in component_map}
    component_diss = {prefix: _series(f'{prefix}_diss_profile') for prefix, _, _, _ in component_map}
    component_net = {prefix: _series(f'{prefix}_net_profile') for prefix, _, _, _ in component_map}
    component_frac_prod = {prefix: _series(f'PD_frac_{prefix}_prod_profile') for prefix, _, _, _ in component_map}
    component_frac_diss = {prefix: _series(f'PD_frac_{prefix}_diss_profile') for prefix, _, _, _ in component_map}

    if plot_type == 'smoothed':
        smooth = lambda arrs: [gaussian_filter1d(a, sigma=smoothing_sigma) for a in arrs]
        itemized_prod = smooth(itemized_prod)
        itemized_diss = smooth(itemized_diss)
        itemized_net = smooth(itemized_net)
        compact_prod = smooth(compact_prod)
        compact_diss = smooth(compact_diss)
        compact_net = smooth(compact_net)
        clus_b2_profile = smooth(clus_b2_profile)
        clus_rho_rho_b_profile = smooth(clus_rho_rho_b_profile)
        for prefix in component_prod:
            component_prod[prefix] = smooth(component_prod[prefix])
            component_diss[prefix] = smooth(component_diss[prefix])
            component_net[prefix] = smooth(component_net[prefix])
            component_frac_prod[prefix] = smooth(component_frac_prod[prefix])
            component_frac_diss[prefix] = smooth(component_frac_diss[prefix])
        r_plot = r
        plot_suffix = f'smoothed_sigma_{smoothing_sigma}'
    elif plot_type == 'interpolated':
        r_new = np.linspace(min(r), max(r), num=interpolation_points, endpoint=True)
        interp = lambda arrs: [
            interp1d(r, arrs[i], kind=interpolation_kind, bounds_error=False, fill_value=np.nan)(r_new)
            for i in range(len(arrs))
        ]
        itemized_prod = interp(itemized_prod)
        itemized_diss = interp(itemized_diss)
        itemized_net = interp(itemized_net)
        compact_prod = interp(compact_prod)
        compact_diss = interp(compact_diss)
        compact_net = interp(compact_net)
        clus_b2_profile = interp(clus_b2_profile)
        clus_rho_rho_b_profile = interp(clus_rho_rho_b_profile)
        for prefix in component_prod:
            component_prod[prefix] = interp(component_prod[prefix])
            component_diss[prefix] = interp(component_diss[prefix])
            component_net[prefix] = interp(component_net[prefix])
            component_frac_prod[prefix] = interp(component_frac_prod[prefix])
            component_frac_diss[prefix] = interp(component_frac_diss[prefix])
        r_plot = r_new
        plot_suffix = f'{interpolation_kind}_interpolated_{interpolation_points}_points'
    else:
        r_plot = r
        plot_suffix = 'raw'

    def _auto_limits(values, scale, pad=0.10):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        if scale == 'log':
            values = values[values > 0]
            if values.size == 0:
                return None
            vmin = np.log10(np.min(values))
            vmax = np.log10(np.max(values))
            span = max(vmax - vmin, 1e-6)
            return 10 ** (vmin - pad * span), 10 ** (vmax + pad * span)
        vmin = np.min(values)
        vmax = np.max(values)
        span = max(vmax - vmin, 1e-12)
        return vmin - pad * span, vmax + pad * span

    def _plot_signed(ax, x, y, lw, ls, color, label, alpha=1.0, eps=1e-30):
        from matplotlib.collections import LineCollection
        y = np.asarray(y)
        x = np.asarray(x)
        valid = ~np.isnan(y)
        if not np.any(valid):
            return
        xv = x[valid]
        yv = y[valid]
        yabs = np.maximum(np.abs(yv), eps)
        segments = []
        sign_styles = []
        for i in range(len(xv) - 1):
            segments.append(np.array([[xv[i], yabs[i]], [xv[i + 1], yabs[i + 1]]]))
            sign_styles.append(yv[i] >= 0)
        if not segments:
            return
        ax.plot(xv, yabs, linestyle='-', linewidth=max(lw * 0.5, 0.3), color=color, alpha=0.2 * alpha, label='_nolegend_')
        ax.add_collection(LineCollection([s for s, p in zip(segments, sign_styles) if p], linewidths=lw, colors=color, linestyles=ls, alpha=alpha))
        ax.add_collection(LineCollection([s for s, p in zip(segments, sign_styles) if not p], linewidths=lw, colors=color, linestyles=':', alpha=alpha))
        ax.plot([], [], linestyle=ls, linewidth=lw, color=color, alpha=alpha, label=label)

    def _blend_color(base_color, target='white', amount=0.4):
        """Blend a color towards white/black to create readable fill shades."""
        c = np.array(to_rgb(base_color), dtype=float)
        t = np.array([1.0, 1.0, 1.0], dtype=float) if target == 'white' else np.array([0.0, 0.0, 0.0], dtype=float)
        amount = float(np.clip(amount, 0.0, 1.0))
        return tuple((1.0 - amount) * c + amount * t)

    figures = []
    plot_absolute = bool(plot_params.get('plot_absolute', True))
    plot_net = bool(plot_params.get('plot_net', False))
    plot_fractional_profiles = bool(
        plot_params.get(
            'plot_fractional_profiles',
            induction_params.get('production_dissipation', {}).get('plot_fractional_profiles', False)
        )
    )

    for snap_i in range(len(it_indx)):
        fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
        ax_energy = ax.twinx() if plot_magnetic_energy else None
        ax_density = ax.twinx() if plot_density else None
        if ax_density is not None:
            if ax_energy is not None:
                ax_density.spines["right"].set_position(("axes", 1.12))
            ax_density.set_frame_on(True)
            ax_density.patch.set_visible(False)
            for sp in ax_density.spines.values():
                sp.set_visible(True)

        snap_z = np.abs(np.round(grid_zeta[it_indx[snap_i]], 2))
        z_text = f"{snap_z:6.2f}"
        ax.set_title(f'{title} - z = {z_text}, $R_{{Vir}}$ = {np.round(rad,1)} Mpc', y=y_title, fontproperties=font_title)

        plotted_vals = []
        energy_vals = []
        density_vals = []

        if plot_absolute:
            if should_plot_component(itemized_prod[snap_i]):
                ax.plot(r_plot, itemized_prod[snap_i], '-.', linewidth=line_main, color=color_prod, label='Total Production (itemized)')
                plotted_vals.append(itemized_prod[snap_i])
            if should_plot_component(itemized_diss[snap_i]):
                ax.plot(r_plot, itemized_diss[snap_i], '-.', linewidth=line_main, color=color_diss, label='Total Dissipation (itemized)')
                plotted_vals.append(itemized_diss[snap_i])
            if should_plot_component(compact_prod[snap_i]):
                ax.plot(r_plot, compact_prod[snap_i], '-', linewidth=line_comp, color=color_prod, label='Total Production (compact)')
                plotted_vals.append(compact_prod[snap_i])
            if should_plot_component(compact_diss[snap_i]):
                ax.plot(r_plot, compact_diss[snap_i], '-', linewidth=line_comp, color=color_diss, label='Total Dissipation (compact)')
                plotted_vals.append(compact_diss[snap_i])

        for prefix, label, color, sym in component_map:
            arr_p = component_prod[prefix][snap_i]
            arr_d = component_diss[prefix][snap_i]

            # Shade the area between P and D for each component: lighter when P dominates,
            # darker when D dominates.
            p_vals = np.asarray(arr_p, dtype=float)
            d_vals = np.asarray(arr_d, dtype=float)
            valid_fill = np.isfinite(p_vals) & np.isfinite(d_vals)
            if y_scale == 'log':
                valid_fill = valid_fill & (p_vals > 0.0) & (d_vals > 0.0)

            if np.any(valid_fill):
                prod_dominates = valid_fill & (p_vals >= d_vals)
                diss_dominates = valid_fill & (d_vals > p_vals)
                fill_light = _blend_color(color, target='white', amount=0.45)
                fill_dark = _blend_color(color, target='black', amount=0.22)

                ax.fill_between(
                    r_plot, p_vals, d_vals,
                    where=prod_dominates,
                    interpolate=True,
                    color=fill_light,
                    alpha=area_alpha,
                    linewidth=0.0,
                    label='_nolegend_'
                )
                ax.fill_between(
                    r_plot, p_vals, d_vals,
                    where=diss_dominates,
                    interpolate=True,
                    color=fill_dark,
                    alpha=area_alpha,
                    linewidth=0.0,
                    label='_nolegend_'
                )

            if should_plot_component(arr_p):
                ax.plot(r_plot, arr_p, '--', linewidth=line_comp, color=color, alpha=component_alpha, label=rf'{label} $P_{{\mathrm{{{sym}}}}}$')
                plotted_vals.append(arr_p)
            if should_plot_component(arr_d):
                ax.plot(r_plot, arr_d, ':', linewidth=line_comp, color=color, alpha=component_alpha, label=rf'{label} $D_{{\mathrm{{{sym}}}}}$')
                plotted_vals.append(arr_d)

        if should_plot_component(compact_net[snap_i]):
            _plot_signed(ax, r_plot, compact_net[snap_i], line_main, '-', color_compact_net, 'Net total (compact)')
            plotted_vals.append(np.maximum(np.abs(compact_net[snap_i]), 1e-30))
        if should_plot_component(itemized_net[snap_i]):
            _plot_signed(ax, r_plot, itemized_net[snap_i], line_main, '--', color_itemized_net, 'Net total (itemized)')
            plotted_vals.append(np.maximum(np.abs(itemized_net[snap_i]), 1e-30))

        if plot_net:
            for prefix, label, color, sym in component_map:
                arr_n = component_net[prefix][snap_i]
                if should_plot_component(arr_n):
                    _plot_signed(ax, r_plot, arr_n, line_comp, '--', color, rf'{label} $N_{{\mathrm{{{sym}}}}}$', alpha=component_alpha)
                    plotted_vals.append(np.maximum(np.abs(arr_n), 1e-30))

        if x_scale == 'log':
            ax.set_xscale('log')
            ax.set_xlabel('Radial Distance log[r/$R_{Vir}$]', fontproperties=font)
        else:
            ax.set_xlabel('Radial Distance [r/$R_{Vir}$]', fontproperties=font)
        ax.tick_params(axis='x', labelsize=11)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])

        if y_scale == 'log':
            ax.set_yscale('log')
            if ax_energy is not None:
                ax_energy.set_yscale('log')
            if ax_density is not None:
                ax_density.set_yscale('log')

        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        elif plotted_vals:
            auto_lim = _auto_limits(np.concatenate([np.asarray(v).ravel() for v in plotted_vals]), y_scale)
            if auto_lim is not None:
                ax.set_ylim(auto_lim[0], auto_lim[1])

        if ax_energy is not None:
            if should_plot_component(clus_b2_profile[snap_i]):
                ax_energy.plot(r_plot, clus_b2_profile[snap_i], '-', linewidth=line_main, color=color_measured_energy, label='Magnetic Energy Density')
                energy_vals.append(clus_b2_profile[snap_i])
        if ax_density is not None:
            if should_plot_component(clus_rho_rho_b_profile[snap_i]):
                ax_density.plot(r_plot, clus_rho_rho_b_profile[snap_i], '-', linewidth=line_main, color=color_density, label='Density')
                density_vals.append(clus_rho_rho_b_profile[snap_i])

        if ax_energy is not None and energy_vals:
            energy_auto = _auto_limits(np.concatenate([np.asarray(v).ravel() for v in energy_vals]), y_scale)
            if energy_auto is not None:
                ax_energy.set_ylim(energy_auto[0], energy_auto[1])
        if ax_density is not None and density_vals:
            density_auto = _auto_limits(np.concatenate([np.asarray(v).ravel() for v in density_vals]), y_scale)
            if density_auto is not None:
                ax_density.set_ylim(density_auto[0], density_auto[1])

        if units == energy_to_erg:
            ax.set_ylabel('Production / Dissipation (erg/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Magnetic Energy Density (erg/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (g/cm³)', fontproperties=font)
        elif units == energy_to_J:
            ax.set_ylabel('Production / Dissipation (J/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Magnetic Energy Density (J/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (M$_{\odot}$/Mpc³)', fontproperties=font)
        else:
            ax.set_ylabel('Production / Dissipation (arb. units)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Magnetic Energy Density (arb. units)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (arb. units)', fontproperties=font)

        ax.grid(alpha=0.3)
        if ax_energy is not None:
            ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        if ax_density is not None:
            ax_density.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        handles, labels = ax.get_legend_handles_labels()
        if ax_energy is not None:
            h_energy, l_energy = ax_energy.get_legend_handles_labels()
            handles += h_energy
            labels += l_energy
        if ax_density is not None:
            h_density, l_density = ax_density.get_legend_handles_labels()
            handles += h_density
            labels += l_density
        seen = set()
        legend_handles = []
        legend_labels = []
        for hh, ll in zip(handles, labels):
            if ll and not ll.startswith('_') and ll not in seen:
                seen.add(ll)
                legend_handles.append(hh)
                legend_labels.append(ll)
        if legend_handles:
            if fixed_legend:
                ax.legend(legend_handles, legend_labels, prop=font_legend,
                          loc='lower left', bbox_to_anchor=(0.02, 0.02),
                          bbox_transform=ax.transAxes, ncol=2, frameon=True)
            else:
                ax.legend(legend_handles, legend_labels, prop=font_legend, ncol=2)

        fig.tight_layout()
        figures.append(fig)

        if plot_fractional_profiles:
            fig_frac, ax_frac = plt.subplots(figsize=figure_size, dpi=dpi)
            ax_frac.set_title(f'{title} (Fractions) - z = {z_text}, $R_{{Vir}}$ = {np.round(rad,1)} Mpc', y=y_title, fontproperties=font_title)

            for prefix, label, color, sym in component_map:
                arr_fp = component_frac_prod[prefix][snap_i]
                arr_fd = component_frac_diss[prefix][snap_i]
                if should_plot_component(arr_fp, threshold=1e-30):
                    ax_frac.plot(r_plot, arr_fp, '--', linewidth=line_comp, color=color, label=rf'{label} $p_{{\mathrm{{{sym}}}}}$')
                if should_plot_component(arr_fd, threshold=1e-30):
                    ax_frac.plot(r_plot, -arr_fd, ':', linewidth=line_comp, color=color, label=rf'{label} $d_{{\mathrm{{{sym}}}}}$')

            prod_i = np.asarray(itemized_prod[snap_i], dtype=float)
            diss_i = np.asarray(itemized_diss[snap_i], dtype=float)
            iota_profile = np.divide(
                prod_i - diss_i,
                prod_i,
                out=np.zeros_like(prod_i),
                where=prod_i > 0
            )
            if should_plot_component(iota_profile, threshold=1e-30):
                ax_frac.plot(r_plot, iota_profile, '-', linewidth=line_main, color=color_efficiency, label=r'Net Efficiency $\iota$')

            if x_scale == 'log':
                ax_frac.set_xscale('log')
                ax_frac.set_xlabel('Radial Distance log[r/$R_{Vir}$]', fontproperties=font)
            else:
                ax_frac.set_xlabel('Radial Distance [r/$R_{Vir}$]', fontproperties=font)
            ax_frac.tick_params(axis='x', labelsize=11)
            if xlim is not None:
                ax_frac.set_xlim(xlim[0], xlim[1])

            ax_frac.set_ylim(-1.05, 1.05)
            ax_frac.set_ylabel('Fractional Contribution (+prod / -diss)', fontproperties=font)
            ax_frac.grid(alpha=0.3)
            if fixed_legend:
                ax_frac.legend(prop=font_legend,
                               loc='lower left', bbox_to_anchor=(0.02, 0.02),
                               bbox_transform=ax_frac.transAxes, ncol=2, frameon=True)
            else:
                ax_frac.legend(prop=font_legend, ncol=2)
            fig_frac.tight_layout()
            figures.append(fig_frac)

    if verbose:
        print('Plotting... Production/dissipation radial profile plots created')

    if save:
        if folder is None:
            folder = os.getcwd()

        sim_info = f'{induction_params.get("up_to_level","")}_{factor_F}_{induction_params.get("vir_kind","")}vir_{induction_params.get("rad_kind","")}rad_{region}Region'
        axis_info = f'{x_scale}_{y_scale}'
        limit_info = f'{xlim[0] if xlim else "auto"}_{ylim[0] if ylim else "auto"}_{ylim[1] if ylim else "auto"}'

        diff_cfg = induction_params.get('differentiation', {})
        if diff_cfg.get('buffer', False) == True:
            parent_flag = diff_cfg.get('parent', False)
            parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol', ''))
            buffer_info = f'Buffered_{diff_cfg.get("interpol","")}_siblings_{diff_cfg.get("use_siblings","")}'
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
        else:
            buffer_info = 'NoBuffer'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, fig in enumerate(figures):
            file_title = '_'.join(title.split()[:3])
            if plot_fractional_profiles and (i % 2 == 1):
                profile_tag = 'pd_frac_profile'
            else:
                profile_tag = 'pd_profile'
            file_name = f'{folder}/{run}_{file_title}_{profile_tag}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil","")}_{plot_suffix}_{i}_{timestamp}.png'
            file_name = safe_filename(file_name, verbose=verbose)
            fig.savefig(file_name, dpi=dpi)
            if verbose:
                print(f'Saved figure {i+1}/{len(figures)}: {file_name}')

    return figures

def distribution_check(arr, quantity, plot_params, induction_params,
                    grid_t, grid_z, rad, ref_field=None, ref_scale=1.0,
                    clean=False, verbose=True, save=False, folder=None):
    '''
    Given a 3D array field (or list of patches per snapshot), generates two separate figures per snapshot:
        - Figure 1: Analysis plots (4 subplots)
            * Histogram of field values
            * Cumulative distribution
            * Cumulative absolute percentiles
            * Cumulative relative percentiles (if ref_field provided)
        - Figure 2: Projection plots (2 subplots)
            * Max projection along XY plane
            * Max projection along XZ plane
    
    This function is especially meant to check the divergence of the magnetic field induction at each calculation step.

    Args:
        - arr: list of snapshots, where each snapshot is either:
             * A 3D numpy array (uniform grid), or
             * A list of patches (AMR). For AMR: only analysis plots are produced (no projections).
        - quantity: string with quantity name (for titles and labels)
        - plot_params: dict with:
            - it_indx: iteration indexes to select snapshots
            - bins: number of bins for histogram
            - log_scale: bool for log scale on y-axis
            - %points: number of points for percentile curves
            - subsample_fraction: fraction of cells to subsample for percentiles (0 < f <= 1)
            - central_fraction: fraction of central box to consider (0 < f <= 1)
            - title: title for the plots
            - dpi: dots per inch for saved plots
            - run: identifier for filenames
        - induction_params: dict with metadata for file naming
            - F: size factor
            - region: region name
            - vir_kind: 'r200' or 'r500'
            - rad_kind: 'rvir' or 'r200' or 'r500'
            - up_to_level: max refinement level
            - buffer: bool for buffer usage
            - interpol: interpolation method if buffer used
            - stencil: stencil type
        - grid_t: 1D array with grid time coordinates
        - grid_z: 1D array with grid z coordinates
        - rad: characteristic radius for normalization
        - ref_field: optional list/array of 3D arrays (same format as arr) for relative plot (cell-wise |arr| / |ref_field|)
                If ref_field equals arr element-wise, the relative plot will be 1 wherever ref!=0
        - ref_scale: scaling factor to apply to ref_field values (in case units differ). This can be a single float or an array with a values per patch.
        - clean: bool indicating whether clean_field was applied (for AMR grids). When True and working with patches (AMR), cells marked as 0.0 by clean_field are excluded from total count.
        - verbose: bool for verbose output
        - save: bool to save plots
        - folder: folder to save plots (if None, uses current directory)
    
    Returns:
        - Tuple of two lists:
            * figures_analysis: analysis plots (4 subplots each)
            * figures_projections: projection plots (2 subplots each or None for AMR snapshots)
            Each list contains one figure per snapshot in it_indx
        
    Author: Marco Molina
    '''
    
    # Extract parameters from plot_params
    DPI = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    title = plot_params.get('title', 'Field Check')
    bins = plot_params.get('bins', 100)
    log_scale = plot_params.get('log_scale', True)
    p_points = plot_params.get('%points', 1001)
    subsample_fraction = plot_params.get('subsample_fraction', 0.2)
    central_fraction = plot_params.get('central_fraction', 1.0)
    it_indx = plot_params.get('it_indx', [0])
    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_hist = palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
    color_cdf = palette.get('induction_itemized', DEFAULT_PLOT_PALETTE['induction_itemized'])
    color_abs_pct = palette.get('dissipation', DEFAULT_PLOT_PALETTE['dissipation'])
    color_rel_pct = component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression'])
    
    assert 0 < subsample_fraction <= 1, "subsample_fraction must be in (0,1]"
    assert 0 < central_fraction <= 1, "central_fraction must be in (0,1]"
    
    # Prepare time/redshift arrays
    t = np.array([grid_t[i] * time_to_yr for i in it_indx])
    z = np.array([grid_z[i] for i in it_indx])
    if z[-1] < 0:
        z[-1] = abs(z[-1])

    figures_analysis = []
    figures_projections = []

    def _is_patch_snapshot(snapshot):
        return isinstance(snapshot, (list, tuple))

    def _flatten_patches(patches, scales=None):
        """Flatten patches, optionally scaling each patch by corresponding scale value."""
        if not patches:
            return np.array([])
        if scales is None:
            return np.concatenate([np.asarray(p).ravel() for p in patches])
        else:
            # scales could be array-like with one value per patch
            if not isinstance(scales, (list, tuple, np.ndarray)):
                scales = [scales] * len(patches)
            scales_arr = np.atleast_1d(scales)
            # Debug: show patch and scale info
            if verbose:
                print(f"Debug _flatten_patches: {len(patches)} patches, {len(scales_arr)} scales")
                for i in range(min(5, len(patches))):  # Show first 5 patches
                    patch_shape = getattr(np.asarray(patches[i]), 'shape', 'no shape')
                    print(f"  Patch {i}: shape={patch_shape}, scale={scales_arr[i]}")
                if len(patches) > 5:
                    print(f"  ... ({len(patches)-5} more patches)")
                if len(patches) != len(scales_arr):
                    raise ValueError(f"Number of patches ({len(patches)}) != number of scales ({len(scales_arr)})")
            return np.concatenate([np.asarray(patches[i]).ravel() * scales_arr[i] for i in range(len(patches))])
    
    for snap_i in range(len(it_indx)):
        # Get current snapshot data
        current_raw = arr[it_indx[snap_i]]
        is_patches = _is_patch_snapshot(current_raw)

        # Optional reference field for this snapshot
        current_ref_raw = None
        if ref_field is not None:
            if isinstance(ref_field, (list, tuple)):
                current_ref_raw = ref_field[it_indx[snap_i]]
            else:
                current_ref_raw = ref_field
        
        current_ref_scale_raw = None
        if ref_scale is not None:
            if isinstance(ref_scale, (list, tuple, np.ndarray)):
                current_ref_scale_raw = ref_scale[it_indx[snap_i]]
            else:
                current_ref_scale_raw = ref_scale

        # Select central box if requested (only for uniform 3D arrays)
        def central_crop(a):
            if central_fraction >= 1.0:
                return a
            cx = int(a.shape[0] * central_fraction / 2)
            cy = int(a.shape[1] * central_fraction / 2)
            cz = int(a.shape[2] * central_fraction / 2)
            x0, x1 = a.shape[0]//2 - cx, a.shape[0]//2 + cx
            y0, y1 = a.shape[1]//2 - cy, a.shape[1]//2 + cy
            z0, z1 = a.shape[2]//2 - cz, a.shape[2]//2 + cz
            return a[x0:x1, y0:y1, z0:z1]

        if is_patches:
            # AMR patches: use all cells, skip projections
            flat = _flatten_patches(current_raw)
            arr_use = None  # Not used for projections
        else:
            assert hasattr(current_raw, "ndim") and current_raw.ndim == 3, f"Input array at snapshot {snap_i} must be 3D or list of patches"
            nmax, nmay, nmaz = current_raw.shape
            arr_use = central_crop(current_raw)
            flat = arr_use.flatten()

        # Prepare reference field if provided (must be done before filtering to maintain correspondence)
        ref_flat = None
        if current_ref_raw is not None:
            if is_patches:
                # For AMR: ref_scale could be array-like (one per patch)
                # Apply per-patch scaling before flattening
                ref_flat = _flatten_patches(current_ref_raw, scales=current_ref_scale_raw)
            else:
                # For uniform grids: ref_scale should be scalar
                assert current_ref_raw.shape == current_raw.shape, f"ref_field must match arr shape at snapshot {snap_i}"
                ref_use = central_crop(current_ref_raw)
                if isinstance(current_ref_scale_raw, (list, tuple, np.ndarray)):
                    ref_scale_scalar = float(np.atleast_1d(current_ref_scale_raw).flat[-1]) # Use last value if array-like
                else:
                    ref_scale_scalar = float(current_ref_scale_raw)
                ref_flat = np.abs(ref_use.flatten()) * ref_scale_scalar

        # Filter out cells marked as 0.0 by clean_field when appropriate
        # This applies when clean=True and working with patches (AMR)
        # Important: apply the same mask to both flat and ref_flat to maintain correspondence
        if clean and is_patches:
            mask = flat != 0.0
            n_filtered = np.sum(~mask)
            flat = flat[mask]
            if ref_flat is not None:
                if ref_flat.size == (mask.size):
                    ref_flat = ref_flat[mask]
                elif verbose:
                    print(f"Warning: ref_field size mismatch at snapshot {snap_i} before filtering. Ref: {ref_flat.size}, Field: {mask.size}")
            if verbose and n_filtered > 0:
                print(f"Snapshot {snap_i}: Filtered out {n_filtered} cells marked as 0.0 by clean_field. Remaining cells: {flat.size}")
        
        n_cells = flat.size

        # Subsample for percentile curves
        sub_n = int(subsample_fraction * n_cells)
        sub_idx = np.random.choice(n_cells, size=sub_n, replace=False)
        sub_vals = np.abs(flat[sub_idx])

        # Relative array if provided (cell-by-cell |field| / (|ref| * ref_scale))
        rel_vals = None
        if ref_flat is not None:
            if ref_flat.size == flat.size:
                ref_sub = np.abs(ref_flat[sub_idx])
                field_sub = np.abs(flat[sub_idx])
                nonzero = ref_sub != 0
                if np.any(nonzero):
                    rel_vals = np.zeros_like(field_sub)
                    rel_vals[nonzero] = field_sub[nonzero] / ref_sub[nonzero]
                elif verbose:
                    print(f"Warning: Reference field is zero everywhere at snapshot {snap_i}. Skipping relative plot.")
            elif verbose:
                print(f"Warning: ref_field length {ref_flat.size} != field length {flat.size} at snapshot {snap_i}. Skipping relative plot.")

        # Percentiles
        p_grid = np.linspace(0, 100, p_points)
        abs_curve = np.percentile(sub_vals, p_grid)
        rel_curve = np.percentile(rel_vals, p_grid) if rel_vals is not None else None

        snap_z = np.abs(np.round(z[snap_i], 2))
        snap_rad = np.round(rad, 1)

        # ========== FIGURE 1: ANALYSIS PLOTS (4 subplots) ==========
        fig_analysis, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        # Histogram
        axes[0].set_title(f'{quantity} Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'{quantity}', fontsize=11)
        axes[0].set_ylabel('Number of Cells', fontsize=11)
        if log_scale:
            axes[0].set_yscale('log')
        axes[0].hist(flat, bins=bins, color=color_hist, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].grid(alpha=0.3)

        # Cumulative distribution
        sorted_arr = np.sort(flat)
        cumulative = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        axes[1].set_title(f'{quantity} Cumulative Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'{quantity}', fontsize=11)
        axes[1].set_ylabel('Cumulative % of Cells', fontsize=11)
        if log_scale:
            axes[1].set_yscale('log')
        axes[1].plot(sorted_arr, cumulative * 100, color=color_cdf, linewidth=2, alpha=0.8)
        axes[1].grid(alpha=0.3)

        # Cumulative absolute percentiles
        axes[2].set_title('Cumulative Absolute |field|', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Percent of cells', fontsize=11)
        axes[2].set_ylabel(f'|{quantity}|', fontsize=11)
        axes[2].plot(p_grid, abs_curve, color=color_abs_pct, linewidth=2.5)
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

        # Cumulative relative percentiles (if available)
        if rel_curve is not None:
            axes[3].set_title('Cumulative Relative |field| / |ref|', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Percent of cells', fontsize=11)
            axes[3].set_ylabel('Relative amplitude', fontsize=11)
            axes[3].plot(p_grid, rel_curve, color=color_rel_pct, linewidth=2.5)
            axes[3].set_yscale('log')
            axes[3].grid(alpha=0.3)
        else:
            axes[3].axis('off')
            axes[3].text(0.5, 0.5, 'No reference field', 
                        ha='center', va='center', fontsize=12, 
                        transform=axes[3].transAxes, style='italic', color='gray')

        # Add snapshot info to title
        fig_analysis.suptitle(f'{title} - {quantity} Analysis - z = {snap_z:.2f}, R = {snap_rad} Mpc', 
                                fontsize=14, fontweight='bold')
        fig_analysis.tight_layout(rect=[0, 0, 1, 0.97])

        figures_analysis.append(fig_analysis)

        # ========== FIGURE 2: PROJECTION PLOTS (2 subplots) ==========
        if is_patches:
            fig_proj = None
            figures_projections.append(None)
            if verbose:
                print(f'Plotting... Distribution check for snapshot {snap_i} (z={snap_z:.2f}) plotted [AMR mode, projections skipped]')
        else:
            fig_proj, axes_proj = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
            plt.subplots_adjust(wspace=0.25)

            # Max projections along axes
            proj_xy = np.max(arr_use, axis=2)
            proj_xz = np.max(arr_use, axis=1)

            im1 = axes_proj[0].imshow(proj_xy, origin='lower', cmap='viridis')
            axes_proj[0].set_title('Max projection (XY)', fontsize=12, fontweight='bold')
            axes_proj[0].set_xlabel('X', fontsize=11)
            axes_proj[0].set_ylabel('Y', fontsize=11)
            cbar1 = fig_proj.colorbar(im1, ax=axes_proj[0], fraction=0.046, pad=0.04)
            cbar1.set_label(f'{quantity}', fontsize=10)

            im2 = axes_proj[1].imshow(proj_xz, origin='lower', cmap='viridis')
            axes_proj[1].set_title('Max projection (XZ)', fontsize=12, fontweight='bold')
            axes_proj[1].set_xlabel('X', fontsize=11)
            axes_proj[1].set_ylabel('Z', fontsize=11)
            cbar2 = fig_proj.colorbar(im2, ax=axes_proj[1], fraction=0.046, pad=0.04)
            cbar2.set_label(f'{quantity}', fontsize=10)

            # Add snapshot info to title
            fig_proj.suptitle(f'{title} - {quantity} Projections - z = {snap_z:.2f}, R = {snap_rad} Mpc', 
                                fontsize=14, fontweight='bold')
            fig_proj.tight_layout(rect=[0, 0, 1, 0.97])

            figures_projections.append(fig_proj)

            if verbose:
                print(f'Plotting... Distribution check for snapshot {snap_i} (z={snap_z:.2f}) plotted')

        # Save the figures if requested
        if save:
            if folder is None:
                folder = os.getcwd()
            sim_info = f'{induction_params.get("up_to_level","")}_{induction_params.get("F","")}_{induction_params.get("vir_kind","")}vir_{induction_params.get("rad_kind","")}rad_{induction_params.get("region","")}Region'
            if is_patches:
                sim_info += '_AMR'
            diff_cfg = induction_params.get('differentiation', {})
            if diff_cfg.get('buffer', False):
                parent_flag = diff_cfg.get('parent', False)
                parent_interpol = diff_cfg.get('parent_interpol', diff_cfg.get('interpol',''))
                buffer_info = f'Buffered_{diff_cfg.get("interpol","")}_siblings_{diff_cfg.get("use_siblings", False)}'
                if parent_flag:
                    buffer_info += f'_parent_{parent_interpol}'
            else:
                buffer_info = 'NoBuffer'

            # Save analysis figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name_analysis = f'{folder}/{run}_{title.replace(" ","_")}_{quantity}_analysis_{sim_info}_{buffer_info}_{snap_i}_{timestamp}.png'
            file_name_analysis = safe_filename(file_name_analysis, verbose=verbose)
            fig_analysis.savefig(file_name_analysis, dpi=DPI)
            if verbose:
                print(f'Saved analysis figure: {file_name_analysis}')

            # Save projection figure (only if generated)
            if not is_patches and fig_proj is not None:
                file_name_proj = f'{folder}/{run}_{title.replace(" ","_")}_{quantity}_projections_{sim_info}_{buffer_info}_{snap_i}_{timestamp}.png'
                file_name_proj = safe_filename(file_name_proj, verbose=verbose)
                fig_proj.savefig(file_name_proj, dpi=DPI)
                if verbose:
                    print(f'Saved projection figure: {file_name_proj}')

    return figures_analysis, figures_projections

        
def plot_3D_volume(arr, axis_values, log = False, subvolume_factor = 1, subsampling_step = 2, axis_step = 10, quantity = ' ', axis_title = ['x', 'y', 'z'], units = ' ', title = ' ', invert = False, verbose = True, Save = False, DPI = 300, run = '_', folder = None):
    
    
    '''
    Plots a 3D volume of a 3D array with changing opacity levels for different values.
    
    Args:
        - arr: 3D array to plot
        - axis_values: values of the axis in Mpc
        - log: boolean to apply a logarithmic scale to the values
        - subvolume_factor: factor to reduce the size of the volume
        - subsampling_step: step to subsample the volume
        - axis_step: step to show the axis values
        - quantity: quantity to plot
        - axis_title: title of the axis
        - units: units of the quantity
        - title: title of the plot
        - invert: boolean to invert the opacity levels
        - verbose: boolean to print the progress of the function
        - Save: boolean to save the plot or not
        - DPI: dots per inch in the plot
        - run: name of the run
        - folder: folder to save the plot
        
    Returns:
        - 3D plot of the volume
        
    Author: Marco Molina
    '''


    # Ensure the array is 3D
    assert arr.ndim == 3, "Input array must be 3D"
    
    # Subsampling the array
    nmax, nmay, nmaz = arr.shape

    sub_vol_x = nmax//subvolume_factor
    sub_vol_y = nmay//subvolume_factor
    sub_vol_z = nmaz//subvolume_factor
    
    arr = arr[sub_vol_x:-sub_vol_x:subsampling_step, sub_vol_y:-sub_vol_y:subsampling_step, sub_vol_z:-sub_vol_z:subsampling_step]


    # Define the plot type and the title
    if title == ' ' and quantity != ' ':
        title = f'3D Plot of {quantity}'

    if log == True:
        arr = np.log10(np.abs(arr) + 1e-30)
        titlecolorbar = f'log_10({quantity}) {units}'
    else:
        titlecolorbar = quantity + units
        

    # Extract the coordinates and values
    x, y, z = np.meshgrid(np.arange(arr.shape[0]),
                        np.arange(arr.shape[1]),
                        np.arange(arr.shape[2]), indexing='ij')

    # Normalize the values to range [0, 1] for opacity
    arr_normalized = (arr - np.min(arr) - 1e-30) / (np.max(arr) - np.min(arr) - 1e-30)

    # Invert the normalized values for opacity
    if invert == True:
        arr_normalized = 1 - arr_normalized
        
    # Custom tick values and labels
    tickvals_x = np.arange(nmax)  # Original tick values
    tickvals_y = np.arange(nmay)  # Original tick values
    tickvals_z = np.arange(nmaz)  # Original tick values

    ticktext_x = np.round(axis_values[0], 2)  # Custom tick labels for x-axis
    ticktext_y = np.round(axis_values[1], 2)  # Custom tick labels for y-axis
    ticktext_z = np.round(axis_values[2], 2)  # Custom tick labels for z-axis

    # Flatten the arrays for plotting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    arr_flat = arr.flatten()
    arr_normalized_flat = arr_normalized.flatten()

    # Get global min and max for color scale
    cmin = np.min(arr_flat)
    cmax = np.max(arr_flat)

    # Create a scatter plot with multiple traces for different opacity levels
    fig = go.Figure()

    # Define opacity levels
    opacity_levels = np.linspace(0.1, 1.0, 20)

    for i, opacity in enumerate(opacity_levels):
        mask = (arr_normalized_flat >= opacity - 0.11) & (arr_normalized_flat < opacity)
        if np.any(mask):  # Check if there are any points in the mask
            fig.add_trace(go.Scatter3d(
                x=x_flat[mask],
                y=y_flat[mask],
                z=z_flat[mask],
                mode='markers',
                name='',
                # showlegend=False,  # Hide legend for all traces
                marker=dict(
                    size=3,
                    color=arr_flat[mask],
                    colorscale='Jet',
                    opacity=opacity,  # Set opacity for this trace
                    showscale=(i == 0),  # Show colorbar only for the first trace
                    cmin=cmin,  # Set global min for color scale
                    cmax=cmax,  # Set global max for color scale
                    colorbar=dict(
                        title=titlecolorbar,
                        tickformat='.2f',  # Format tick labels to 2 decimal places
                        titlefont=dict(size=20),  # Increase colorbar title font size
                        tickfont=dict(size=15)  # Increase colorbar tick font size
                    ) if i == 0 else None
                )
            ))
    xaxis_title = axis_title[0]
    yaxis_title = axis_title[1]
    zaxis_title = axis_title[2]

    # Set labels and custom tick values
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=xaxis_title,
                tickvals=tickvals_x[0::axis_step],        
                ticktext=ticktext_x[0::axis_step],
                titlefont=dict(size=20),  # Increase z-axis title font size
                tickfont=dict(size=15)  # Increase z-axis tick font size            
            ),
            yaxis=dict(
                title=yaxis_title,
                tickvals=tickvals_y[0::axis_step],            
                ticktext=ticktext_y[0::axis_step],
                titlefont=dict(size=20),  # Increase z-axis title font size
                tickfont=dict(size=15)  # Increase z-axis tick font size            
            ),
            zaxis=dict(
                title=zaxis_title,
                tickvals=tickvals_z[0::axis_step],            
                ticktext=ticktext_z[0::axis_step],
                titlefont=dict(size=20),  # Increase z-axis title font size
                tickfont=dict(size=15)  # Increase z-axis tick font size     
            ),
            camera=dict(
                eye=dict(x=1.37, y=1.37, z=1.37),  # Set the initial camera position
                center=dict(x=0.01, y=0, z=-0.15)  # Translate the cube upwards
            )
        ),
        title=dict(
            text=title,
            font=dict(size=30)  # Increase plot title font size
        ),
        width=900,  # Increase width
        height=900  # Increase height
    )

    fig.show()
    
    if verbose == True:
        print(f'Plotting... 3D Volume Plot computed')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
            
        file_title = ' '.join(title.split()[:4])
        fig.write_image(folder + f'/{run}_{file_title}.png', width=900, height=900, scale=2, dpi = DPI)