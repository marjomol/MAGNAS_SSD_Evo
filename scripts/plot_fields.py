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
import scripts.diff as diff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
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
from scripts.units import *
        
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
        print(f'Plotting... Magnetic Field Seed Zoom Animation computed')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
    
        file_title = ' '.join(title.split()[:4])
        ani.save(folder + f'/{run}_{file_title}_zoom.gif', writer='pillow', dpi = DPI)
        
    return ani
        
def scan_animation_3D(arr, size, study_box, depth = 2, arrow_scale = 1, units = 'Mpc', title = 'Magnetic Field Seed Scan', verbose = True, Save = False, DPI = 300, run = '_', folder = None):
    '''
    Generates an animation of the magnetic field seed in 3D with a scan effect. Can be used for any other 3D spacial field.
    
    Args:
        - arr: 3D array to animate
        - size: size of the array in Mpc in the x direction
        - study_box: percentage of the box to scan centered in the middle of the scanning plane. Must be a float in (0, 1]
        - depth: depth of the scanning plane, the larger the depth the less frames the animation will have
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
    
    assert arr.ndim == 3, "Input array must be 3D"
    assert 0 < study_box <= 1, "Study box must be a float in (0, 1]"
    assert depth > 0, "Depth must be a positive integer"
    assert arrow_scale > 0, "Arrow scale must be a positive integer"
    assert units in ['Mpc', 'kpc'], "Units must be 'Mpc' or 'kpc'"
    
    nmax, nmay, nmaz = arr.shape
    
    dx = size / nmax  # Cell size in Mpc
    
    inter = 100
    x_lsize = round(nmax//2 - nmax*study_box//2)
    x_dsize = round(nmax//2 + nmax*study_box//2)
    y_lsize = round(nmay//2 - nmay*study_box//2)
    y_dsize = round(nmay//2 + nmay*study_box//2)
    new_nmax = x_dsize - x_lsize # Para definir el tamaño de la flecha de referencia
    col = 'red'
    
    fig = plt.figure(figsize=(5, 5))
    
    # Find the minimum and maximum values of the magnetic field among all the studied volume
    all_values = []
    for i in range(nmaz):
        frame_data = arr[x_lsize:x_dsize, y_lsize:y_dsize, i]
        all_values.extend(frame_data[frame_data > 0].flatten())

    all_values = np.array(all_values)
    min_value = np.percentile(all_values, 1)  # 1st percentile
    max_value = np.percentile(all_values, 99.9)  # 99.9th percentile
    print(min_value, max_value)
        
    # Create a logarithmic normalization for the color intensity and regulate the intensity of the color bar
    norm = LogNorm(vmin=min_value, vmax=max_value)
    
    if units == 'Mpc':
        ctou = arrow_scale/dx
    elif units == 'kpc':
        ctou = arrow_scale/(dx * 1000)

    def animate(frame):
        plt.clf()
        section = np.sum(arr[x_lsize:x_dsize, y_lsize:y_dsize, (frame - depth//2):(frame + depth//2)], axis=2)
        plt.imshow(section, cmap='viridis', norm=norm)
        plt.title(title)
        plt.arrow((new_nmax - 4*new_nmax//5), (new_nmax - new_nmax//10), ctou, 0, head_width=(ctou/14), head_length=(ctou/7), fc=col, ec=col)
        text_x = (new_nmax - 4*new_nmax//5) + ctou / 2  # Centered above the arrow
        text_y = (new_nmax - new_nmax//10) - 0.03 * new_nmax  # Small offset above the arrow
        plt.text(text_x, text_y, f'{arrow_scale} {units}', color=col, ha='center', va='bottom', fontsize=10)
        plt.xlabel('x cells')
        plt.ylabel('y cells')

    ani = FuncAnimation(fig, animate, frames = range(depth, nmaz), interval=inter)
    ani = FuncAnimation(fig, animate, frames = range(nmaz), interval=inter)
    
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ani.save(data_dir + '/animation.gif', writer='pillow')
    
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Scan Animation computed')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
    
        file_title = ' '.join(title.split()[:4])
        ani.save(folder + f'/{run}_{file_title}_scan.gif', writer='pillow', dpi = DPI)
        
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
            - cancel_limits: bool to flip the x axis (useful for zeta)
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
        arr = np.asarray(data)
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
            print('Percentile evolution: no percentile data to plot')
        return None
    
    # Determine reference levels from first non-empty entry
    ref_levels = None
    for lv in (levels_list if isinstance(levels_list, (list, tuple)) else [levels_list]):
        if isinstance(lv, (list, np.ndarray)) and len(lv) > 0:
            ref_levels = np.asarray(lv, dtype=float)
            break
    if ref_levels is None:
        if verbose:
            print('Percentile evolution: no valid levels found')
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
            print('Percentile evolution: no valid percentile rows to plot')
        return None

    # Stack to 2D arrays (n_snap_valid, n_levels)
    pct = np.vstack(aligned_pct)
    pct_plus = np.vstack(aligned_plus) if aligned_plus is not None and len(aligned_plus) == len(aligned_pct) else None
    pct_minus = np.vstack(aligned_minus) if aligned_minus is not None and len(aligned_minus) == len(aligned_pct) else None
    levels = ref_levels

    if pct.size == 0 or levels.size == 0:
        if verbose:
            print('Percentile evolution: no valid percentile data to plot')
        return None

    if pct.ndim != 2:
        raise ValueError(f'percentiles must form a 2D array; got shape {pct.shape}')

    n_snap, n_levels = pct.shape
    if levels.size != n_levels:
        raise ValueError(f'levels length ({levels.size}) must match percentile columns ({n_levels})')

    # Prepare x-axis (plots all snapshots like plot_integral_evolution)
    x_axis = plot_params.get('x_axis', 'zeta')
    if x_axis == 'years':
        x = np.array([grid_t[i] * time_to_yr for i in valid_indices], dtype=float)
        xlabel = 'Time (yr)'
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
    title = plot_params.get('title', 'Percentile Threshold Evolution')
    line_widths = plot_params.get('line_widths', [2.0, 1.5])
    alpha_fill = plot_params.get('alpha_fill', 0.20)
    
    lw_pct = line_widths[0]
    lw_max = line_widths[1] if len(line_widths) > 1 else line_widths[0]

    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, levels_sorted.size))

    # Shaded bands: ±1% error bands around each percentile (if available)
    if pct_plus_sorted is not None and pct_minus_sorted is not None:
        for i in range(levels_sorted.size):
            lower = pct_minus_sorted[:, i]
            upper = pct_plus_sorted[:, i]
            ax.fill_between(x, lower, upper, color=colors[i], alpha=alpha_fill, label='_nolegend_')

    # Plot percentile curves
    for i, lvl in enumerate(levels_sorted):
        if float(lvl).is_integer():
            lbl = f'{int(lvl)}%'
        else:
            lbl = f'{lvl:.1f}%'
        ax.plot(x, pct_sorted[:, i], color=colors[i], linewidth=lw_pct, label='_nolegend_')
        
        # Add label at the end of each curve
        x_end = x[-1]
        y_end = pct_sorted[-1, i]
        ax.text(x_end, y_end, f'  {lbl}', fontsize=10, color=colors[i], 
                verticalalignment='center', fontweight='bold')

    # Optional max curve
    gmax_list = percentile_data.get('global_max', None)
    if gmax_list is not None:
        # Select only valid indices if possible
        try:
            gmax = np.asarray([gmax_list[i] for i in valid_indices], dtype=float)
        except Exception:
            gmax = None
        if gmax is not None and gmax.size == x.size:
            ax.plot(x, gmax, color='#111111', linewidth=lw_max, linestyle='--', label='_nolegend_')
            # Add Max label
            ax.text(x[-1], gmax[-1], '  Max', fontsize=10, color='#111111', 
                    verticalalignment='center', fontweight='bold')
    
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

    fig.suptitle(title, fontproperties=font_title, y=0.995)

    fig.tight_layout()

    if save:
        if folder is None:
            folder = os.getcwd()

        axis_info = f"{x_axis}_{x_scale}_{y_scale}"
        limit_info = f"{xlim[0] if xlim else 'auto'}_{ylim[0] if ylim else 'auto'}_{ylim[1] if ylim else 'auto'}"
        sim_info = f"{induction_params.get('up_to_level','')}_{induction_params.get('F','')}_{induction_params.get('vir_kind','')}vir_{induction_params.get('rad_kind','')}rad_{induction_params.get('region','None')}Region"
        if induction_params.get('buffer', False) == True:
            buffer_info = f'Buffered_{induction_params.get("interpol","")}'
        else:
            buffer_info = 'NoBuffer'

        file_title = '_'.join(title.split()[:3])
        filename = f"{folder}/{run}_{file_title}_percentile_evo_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{induction_params.get('stencil','')}.png"
        fig.savefig(filename, dpi=dpi)
        if verbose:
            print(f'Percentile evolution plot saved as: {filename}')

    if verbose:
        lvl_str = ', '.join([str(l) for l in levels_sorted])
        print(f'Plotted percentile evolution for levels: {lvl_str}')

    return fig
        
        
def plot_integral_evolution(evolution_data, plot_params, induction_params,
                            grid_t, grid_zeta, rad,
                            verbose=True, save=False, folder=None):
    """
    Plot the evolution of the integrated magnetic energy and its induction components attending to the time derivative prediction from the induction equation.
    
    Args:
        - evolution_data: dictionary containing the evolution data from induction_energy_integral_evolution()
        - plot_params: dictionary containing plotting parameters:
            - evolution_type: 'total' or 'differential'
            - derivative: 'RK', 'central', 'implicit' or 'rate'
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
        - List of figure objects
        
    Author: Marco Molina
    """
    
    # Validate plot_params
    plot_type = plot_params.get('plot_type', 'raw')
    assert plot_type in ['raw', 'smoothed', 'interpolated'], "plot_type must be 'raw', 'smoothed', or 'interpolated'"
    assert plot_params.get('interpolation_kind', 'linear') in ['linear', 'cubic', 'nearest'], "interpolation_kind must be 'linear', 'cubic', or 'nearest'"
    assert plot_params.get('smoothing_sigma', 1.10) > 0, "smoothing_sigma must be a positive number"
    assert plot_params.get('x_axis', 'zeta') in ['zeta', 'years'], "x_axis must be 'zeta' or 'years'"
    
    # Extract parameters from plot_params
    evolution_type = plot_params['evolution_type']
    derivative = plot_params['derivative']
    x_axis = plot_params['x_axis']
    if x_axis == 'zeta':
        assert len(grid_zeta) > 0, "grid_zeta must not be empty when x_axis is 'zeta'"
        assert plot_params.get('interpolation points', 5000) > 0, "interpolation_points must be a positive integer"
    elif x_axis == 'years':
        assert len(grid_t) > 0, "grid_t must not be empty when x_axis is 'years'"
        assert plot_params.get('interpolation points', 500) > 0, "interpolation_points must be a positive integer"
    x_scale = plot_params['x_scale']
    y_scale = plot_params['y_scale']
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
    
    # Extract induction parameters
    units = induction_params['units']
    factor_F = induction_params['F']
    region = induction_params['region']
    
    # Set up matplotlib parameters
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
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
    font_title.set_size(24)
    
    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)
    
    y_title = 1.005
    line1, line2 = line_widths
    
    # Prepare time and redshift arrays
    if x_axis == 'zeta':
        z = np.array([grid_zeta[i] for i in range(len(grid_zeta))])
        if z[-1] < 0:
            z[-1] = abs(z[-1])
    else: # years
        t = [grid_t[i] * time_to_yr for i in range(len(grid_t))]
    
    # Extract evolution data with units
    if evolution_type == 'differential':
        units = units / time_to_s
    
    # Get the appropriate data arrays based on evolution_type and derivative
    if evolution_type == 'total':
        index_O, index_F = 0, len(grid_t)
        if derivative == 'RK':
            index_o, index_f = 0, len(grid_t)
            plotid = 'RK_total'
        elif derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_total'
        elif derivative == 'rate':
            index_o, index_f = 1, len(grid_t)
            plotid = 'rate_total'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_total'
    else:  # differential
        index_O, index_F = 1, len(grid_t)
        if derivative == 'RK':
            index_o, index_f = 0, len(grid_t)
            plotid = 'RK_differential'
        elif derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_differential'
        elif derivative == 'rate':
            index_o, index_f = 1, len(grid_t)
            plotid = 'rate_differential'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_differential'
    
    # Extract component data with units
    n1 = [units * evolution_data['evo_b2'][i] for i in range(len(evolution_data['evo_b2']))]
    n0 = [units * evolution_data['evo_ind_b2'][i] for i in range(len(evolution_data['evo_ind_b2']))]
    diver_work = [units * evolution_data['evo_MIE_diver_B2'][i] for i in range(len(evolution_data['evo_MIE_diver_B2']))]
    compres_work = [units * evolution_data['evo_MIE_compres_B2'][i] for i in range(len(evolution_data['evo_MIE_compres_B2']))]
    stretch_work = [units * evolution_data['evo_MIE_stretch_B2'][i] for i in range(len(evolution_data['evo_MIE_stretch_B2']))]
    advec_work = [units * evolution_data['evo_MIE_advec_B2'][i] for i in range(len(evolution_data['evo_MIE_advec_B2']))]
    drag_work = [units * evolution_data['evo_MIE_drag_B2'][i] for i in range(len(evolution_data['evo_MIE_drag_B2']))]
    total_work = [units * evolution_data['evo_MIE_total_B2'][i] for i in range(len(evolution_data['evo_MIE_total_B2']))]
    kinetic_work = [units * evolution_data['evo_kinetic_energy'][i] for i in range(len(evolution_data['evo_kinetic_energy']))]
    
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
        # Create interpolations
        if x_axis == 'years':
            x_data = t
            x_new = np.linspace(min(t[index_o:index_f]), max(t[index_o:index_f]), 
                                    num=interpolation_points['years'], endpoint=True)
        else:  # zeta
            x_data = z
            x_new = np.linspace(max(z[index_o:index_f]), min(z[index_o:index_f]), 
                                    num=interpolation_points['zeta'], endpoint=True)
        
        # Create interpolation functions
        n1_interp = interp1d(x_data[index_O:index_F], n1[index_O:index_F], kind=interpolation_kind)
        n0_interp = interp1d(x_data[index_o:index_f], n0, kind=interpolation_kind)
        diver_work_interp = interp1d(x_data[index_o:index_f], diver_work, kind=interpolation_kind)
        compres_work_interp = interp1d(x_data[index_o:index_f], compres_work, kind=interpolation_kind)
        stretch_work_interp = interp1d(x_data[index_o:index_f], stretch_work, kind=interpolation_kind)
        advec_work_interp = interp1d(x_data[index_o:index_f], advec_work, kind=interpolation_kind)
        drag_work_interp = interp1d(x_data[index_o:index_f], drag_work, kind=interpolation_kind)
        total_work_interp = interp1d(x_data[index_o:index_f], total_work, kind=interpolation_kind)
        kinetic_work_interp = interp1d(x_data[index_O:index_F], kinetic_work[index_O:index_F], kind=interpolation_kind)
        
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
    
    # Create figures list
    figures = []
    
    # Main evolution plot
    fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Track which components were plotted
    components_plotted = []
    
    # Kinetic energy
    if should_plot_component(kinetic_work_data):
        ax1.plot(x_data[index_O_plot:index_F_plot], kinetic_work_data[index_O_plot:index_F_plot], 
                linewidth=line1, label='Kinetic Energy', color='#17becf')
        components_plotted.append('kinetic')
    
    # Main energy line (always plot)
    if evolution_type == 'total':
        label = 'Magnetic Energy'
    else:
        label = 'Magnetic Energy Induction'
    ax1.plot(x_data[index_O_plot:index_F_plot], n1_data[index_O_plot:index_F_plot], 
            linewidth=line1, label=label, color='#1f77b4')
        
    # Total work (compacted)
    if should_plot_component(total_work_data):
        ax1.plot(x_data[index_o_plot:index_f_plot], total_work_data, '-', 
                linewidth=line1, label='...from Compact Induction', color='#d62728')
        components_plotted.append('total')

    # Induction prediction (plot if has data)
    if should_plot_component(n0_data):
        if evolution_type == 'total':
            label = '...from Itemize Induction'
        else:
            label = 'Predicted Induction'
        ax1.plot(x_data[index_o_plot:index_f_plot], n0_data, '--',
                linewidth=line1, label=label, color='#ff7f0e')

    # Individual components with their colors
    component_configs = [
        (compres_work_data, 'Compression', '#9467bd'),
        (stretch_work_data, 'Stretching', '#ff9896'),
        (advec_work_data, 'Advection', '#e377c2'),
        (diver_work_data, 'Divergence', '#c5b0d5'),
        (drag_work_data, 'Cosmic Drag', '#7f7f7f')
    ]
    
    for data, label, color in component_configs:
        if should_plot_component(data):
            ax1.plot(x_data[index_o_plot:index_f_plot], data, '--', 
                    linewidth=line2, label=label, color=color)
            components_plotted.append(label.lower())
    
    setup_axis(ax1, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis, evolution_type, font)
    ax1.legend(prop=font_legend)

    if region == 'None':
        plot_title = f'{title} - {np.round(induction_params["size"][0]/2)} Mpc'
    else:
        plot_title = f'{title} - {np.round(factor_F*rad)} Mpc'

    if evolution_type != 'total':
        plot_title = plot_title.replace('Evolution', 'Induction Evolution')
    plt.title(plot_title, y=y_title, fontproperties=font_title)
    
    if cancel_limits and x_axis == 'zeta':
        plt.gca().invert_xaxis()
    fig1.tight_layout()
    figures.append(fig1)
    
    # Volume plot (optional)
    if volume_evolution:
        fig2, ax2 = plt.subplots(figsize=figure_size, dpi=dpi)
        
        if x_axis == 'years':
            ax2.set_xlabel('Time (yr)')
            ax2.plot(t, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(t, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
        else:  # zeta
            ax2.set_xlabel('Redshift (z)')
            ax2.plot(z, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(z, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
            ax2.invert_xaxis()
        
        ax2.set_ylabel('Integration Volume')
        ax2.legend(fontsize='small')
        ax2.set_yscale('log')
        
        plt.title('Integrated Volume')
        fig2.tight_layout()
        figures.append(fig2)
    
    if verbose:
        print(f'Plotting... {plot_type.capitalize()} integrated magnetic energy and induction prediction plot created')
        print(f'Components plotted: {", ".join(components_plotted)}')
        if volume_evolution:
            print(f'Plotting... Volume evolution plot created')
    
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
            
        if induction_params['buffer'] == True:
            buffer_info = f'Buffered_{induction_params["interpol"]}'
        else:
            buffer_info = 'NoBuffer'
        
        # Save main plot
        file_title = '_'.join(title.split()[:3])
        filename1 = f'{folder}/{run}_{file_title}_integrated_energy_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{induction_params["stencil"]}_{plotid}{plot_suffix}.png'
        fig1.savefig(filename1, dpi=dpi)
        
        if verbose:
            print(f'Plotting... Main plot saved as: {filename1}')
        
        # Save volume plot if created
        if volume_evolution:
            filename2 = f'{folder}/{run}_{file_title}_volume_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{induction_params["stencil"]}_{plotid}.png'
            fig2.savefig(filename2, dpi=dpi)
            
            if verbose:
                print(f'Plotting... Volume plot saved as: {filename2}')
                
    if verbose:       
        for i, sim in enumerate(induction_params.get('sims', ['default'])):
            if derivative == 'RK':
                n_iter = len(induction_params.get('it', [0]))
            elif derivative == 'central' or derivative == 'rate':
                n_iter = len(induction_params.get('it', [0])) - 1
            else:  # implicit
                n_iter = len(induction_params.get('it', [0])) - 2
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

def plot_radial_profiles(profile_data, plot_params, induction_params,
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
    units = induction_params.get('units', None)
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
    font_title.set_size(24)
    
    font_legend = FontProperties()
    font_legend.set_style('normal')
    font_legend.set_weight('normal')
    font_legend.set_size(12)
    
    y_title = 1.05
    line1, line2 = line_widths

    # Radial axis normalized by R_vir
    r = np.asarray(profile_bin_centers) / float(rad)
    nbins = profile_bin_centers.shape[0]
    
    t = [grid_t[i] * time_to_yr for i in it_indx]
    z = np.array([grid_zeta[i] for i in it_indx])
    if z[-1] < 0:
        z[-1] = abs(z[-1])

    # Compute plotted arrays with units (each is a list indexed by snapshots)
    def safe_get(key):
        return profile_data.get(key, [np.zeros(nbins) for _ in it_indx])

    kinetic_energy_profile = [units_y_2 * safe_get('kinetic_energy_profile')[i] for i in it_indx]
    clus_b2_profile = [units_y_2 * safe_get('clus_b2_profile')[i] for i in it_indx]
    clus_rho_rho_b_profile = [units_y_3 * safe_get('clus_rho_rho_b_profile')[i] for i in it_indx]
    diver_profile = [units_y_1 * safe_get('MIE_diver_B2_profile')[i] for i in it_indx]
    compres_profile = [units_y_1 * safe_get('MIE_compres_B2_profile')[i] for i in it_indx]
    stretch_profile = [units_y_1 * safe_get('MIE_stretch_B2_profile')[i] for i in it_indx]
    advec_profile = [units_y_1 * safe_get('MIE_advec_B2_profile')[i] for i in it_indx]
    drag_profile = [units_y_1 * safe_get('MIE_drag_B2_profile')[i] for i in it_indx]
    total_profile = [units_y_1 * safe_get('MIE_total_B2_profile')[i] for i in it_indx]
    ind_b2_profile = [units_y_1 * safe_get('ind_b2_profile')[i] for i in it_indx]
    # post_ind_b2_profile = [units_y_1 * safe_get('post_ind_b2_profile')[i] for i in it_indx]

    
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
    
    # Components configuration: (data_list, label, color, lw, linestyle, unit)
    components_configs = [
        # Density (units_y_3)
        (clus_rho_rho_b_profile, 'Density Profile', "#2ca02c", line1, '-', units_y_3),
        # Energies (units_y_2)
        (kinetic_energy_profile, 'Kinetic Energy Density', "#17becf", line1, '-', units_y_2),
        (clus_b2_profile, 'Magnetic Energy Density', "#1f77b4", line1, '-', units_y_2),
        # Induction totals (units_y_1)
        (total_profile, '...from Compact Induction', "#d62728", line1, '-', units_y_1),
        (ind_b2_profile, '...from Itemize Induction', "#ff7f0e", line1, '--', units_y_1),
        # (post_ind_b2_profile, '...from Post-Itemize Induction', "#ffbb78", line1, '-.', units_y_1),
        # Individual components (units_y_1)
        (compres_profile, 'Compression', "#9467bd", line2, '--', units_y_1),
        (stretch_profile, 'Stretching', "#ff9896", line2, '--', units_y_1),
        (advec_profile, 'Advection', "#e377c2", line2, '--', units_y_1),
        (diver_profile, 'Divergence', "#c5b0d5", line2, '--', units_y_1),
        (drag_profile, 'Cosmic Drag', "#7f7f7f", line2, '--', units_y_1)
    ]

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
    def plot_signed(ax, x, y, lw, ls, color, label, eps=induction_params.get('epsilon', 1e-30)):
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
                color=color, alpha=0.2, label='_nolegend_')

        # Overlay segments with sign-dependent dashes
        lc_pos = LineCollection(
            [seg for seg, is_pos in zip(segments, sign_styles) if is_pos],
            linewidths=lw, colors=color, linestyles=ls, label='_nolegend_'
        )
        lc_neg = LineCollection(
            [seg for seg, is_pos in zip(segments, sign_styles) if not is_pos],
            linewidths=lw, colors=color, linestyles=':', label='_nolegend_'
        )
        
        ax.add_collection(lc_pos)
        ax.add_collection(lc_neg)

        # Return a dummy handle for legend entry
        h_legend, = ax.plot([], [], linestyle=ls, linewidth=lw, color=color, label=label)
        return h_legend
    
    for snap_i in range(len(it_indx)):
        fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
        ax_right = ax1.twinx()  # Second axis for induction
        ax_density = ax1.twinx()  # Third axis for density
        ax_density.spines["right"].set_position(("axes", 1.12))
        ax_density.set_frame_on(True)
        ax_density.patch.set_visible(False)
        for sp in ax_density.spines.values():
            sp.set_visible(True)

        snap_z = np.abs(np.round(grid_zeta[it_indx[snap_i]], 2))
        ax1.set_title(f'{title} - z = {snap_z:.2f}, $R_{{Vir}}$ = {np.round(rad,1)} Mpc', y=y_title, fontproperties=font_title)

        unique_handles = []
        unique_labels = []
        
        # Add one explanatory legend entry: dotted means negative interval
        from matplotlib.lines import Line2D
        neg_note = Line2D([0], [0], color="#364243", linestyle=':', linewidth=line2, label='Negative Interval')
        unique_handles.append(neg_note)
        unique_labels.append('Negative Interval')

        for (data_list, label, color, lw, ls, unit) in components_configs:
            if not has_nonzero(data_list, snap_i):
                continue

            # obtain y values depending on type
            if callable(data_list[snap_i]):
                y = data_list[snap_i](r_pro)
            else:
                y = np.asarray(data_list[snap_i])
                # if interpolation requested but we have raw arrays and r_pro is finer, interpolate on the fly
                if plot_type == 'interpolated' and y.size == r.size and r_pro.size != r.size:
                    y = np.interp(r_pro, r, y)
            
            # Choose axis: left for energy/density (units_y_2), right for induction (units_y_1), density (units_y_3)
            if unit == units_y_1:
                # unified signed plotting on right axis
                _ = plot_signed(ax_right, r_pro, y, lw, ls, color, label)
            elif unit == units_y_3:
                ax_density.plot(r_pro, y, linewidth=lw, linestyle=ls, color=color, label=label)
            else:
                ax1.plot(r_pro, y, linewidth=lw, linestyle=ls, color=color, label=label)
            
        # Axis scales and labels
        if x_scale == 'log':
            ax1.set_xscale('log')
            ax1.set_xlabel('Radial Distance log[r/$R_{Vir}$]')
        else:
            ax1.set_xlabel('Radial Distance [r/$R_{Vir}$]')

        if xlim is not None:
            ax1.set_xlim(xlim[0], xlim[1])

        # y scales
        if y_scale == 'log':
            ax1.set_yscale('log')
            ax_right.set_yscale('log')
            ax_density.set_yscale('log')

        if units == energy_to_erg:
            ax1.set_ylabel('Energy Density (erg/$Mpc^{3}$)')
            ax_right.set_ylabel('Induction Density (erg/$Mpc^{3}$/s)')
            ax_density.set_ylabel('Density (g/cm³)')
        elif units == energy_to_J:
            ax1.set_ylabel('Energy Density (J/$Mpc^{3}$)')
            ax_right.set_ylabel('Induction Density (J/$Mpc^{3}$/s)')
            ax_density.set_ylabel('Density (M$_{\odot}$/Mpc³)')
        else:
            ax1.set_ylabel('Energy Density (arb. units)')
            ax_right.set_ylabel('Induction Density (arb. units / s)')
            ax_density.set_ylabel('Density (arb. units)')

        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])
        if rylim is not None:
            ax_right.set_ylim(rylim[0], rylim[1])
        if dylim is not None:
            ax_density.set_ylim(dylim[0], dylim[1])
            
        ax_density.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        # Gather handles/labels from all axes and keep unique labels in order
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        h3, l3 = ax_density.get_legend_handles_labels()
        all_handles = h3 + h1 + h2
        all_labels = l3 + l1 + l2
        seen = set()
        for hh, ll in zip(all_handles, all_labels):
            if ll and not ll.startswith('_') and ll not in seen:
                seen.add(ll)
                unique_handles.append(hh)
                unique_labels.append(ll)

        if unique_handles and xlim is not None and ylim is not None and rylim is not None and dylim is not None:
            # Place a fixed, two-column legend at the bottom-left of the figure.
            # bbox_to_anchor is in figure coordinates when bbox_transform=fig1.transFigure.
            # ncol=2 forces two columns; adjust bbox_to_anchor if you need a different offset.
            fig1.legend(unique_handles, unique_labels, prop=font_legend,
                        loc='best', bbox_to_anchor=(0.49, 0.31),
                        bbox_transform=fig1.transFigure, ncol=2, frameon=True)
            # Slightly adjust subplot to avoid overlapping the legend or title
            fig1.subplots_adjust(top=0.90, right=0.86)
        else:
            fig1.legend(unique_handles, unique_labels, prop=font_legend, ncol=2, frameon=True)

        fig1.tight_layout()
        
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

        if induction_params.get('buffer', False) == True:
            buffer_info = f'Buffered_{induction_params.get("interpol","")}'
        else:
            buffer_info = 'NoBuffer'

        for i, fig in enumerate(figures):
            file_title = '_'.join(title.split()[:3])
            file_name = f'{folder}/{run}_{file_title}_induction_profile_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{induction_params.get("stencil","")}_{plot_suffix}_{i}.png'
            fig.savefig(file_name, dpi=dpi)
            if verbose:
                print(f'Saved {i} figure: {file_name}')

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
        axes[0].hist(flat, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[0].grid(alpha=0.3)

        # Cumulative distribution
        sorted_arr = np.sort(flat)
        cumulative = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        axes[1].set_title(f'{quantity} Cumulative Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(f'{quantity}', fontsize=11)
        axes[1].set_ylabel('Cumulative % of Cells', fontsize=11)
        if log_scale:
            axes[1].set_yscale('log')
        axes[1].plot(sorted_arr, cumulative * 100, color='#ff7f0e', linewidth=2, alpha=0.8)
        axes[1].grid(alpha=0.3)

        # Cumulative absolute percentiles
        axes[2].set_title('Cumulative Absolute |field|', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Percent of cells', fontsize=11)
        axes[2].set_ylabel(f'|{quantity}|', fontsize=11)
        axes[2].plot(p_grid, abs_curve, color='#d62728', linewidth=2.5)
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)

        # Cumulative relative percentiles (if available)
        if rel_curve is not None:
            axes[3].set_title('Cumulative Relative |field| / |ref|', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Percent of cells', fontsize=11)
            axes[3].set_ylabel('Relative amplitude', fontsize=11)
            axes[3].plot(p_grid, rel_curve, color='#9467bd', linewidth=2.5)
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
            if induction_params.get('buffer', False):
                buffer_info = f'Buffered_{induction_params.get("interpol","")}'
            else:
                buffer_info = 'NoBuffer'

            # Save analysis figure
            file_name_analysis = f'{folder}/{run}_{title.replace(" ","_")}_{quantity}_analysis_{sim_info}_{buffer_info}_{snap_i}.png'
            fig_analysis.savefig(file_name_analysis, dpi=DPI)
            if verbose:
                print(f'Saved analysis figure: {file_name_analysis}')

            # Save projection figure (only if generated)
            if not is_patches and fig_proj is not None:
                file_name_proj = f'{folder}/{run}_{title.replace(" ","_")}_{quantity}_projections_{sim_info}_{buffer_info}_{snap_i}.png'
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