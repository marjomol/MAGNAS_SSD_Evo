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


def _get_label_mode(plot_params=None):
    """Return plotting label mode: 'verbal' or 'math'."""
    mode = (plot_params or {}).get('label_mode', 'verbal')
    if isinstance(mode, str) and mode.lower() in ('verbal', 'math'):
        return mode.lower()
    return 'verbal'


def _axis_label_x(x_axis, x_scale='lin', label_mode='verbal'):
    """Build x-axis label in verbal or math mode."""
    if label_mode == 'math':
        if x_axis == 'years':
            return r'$\log_{10}(t/\mathrm{yr})$' if x_scale == 'log' else r'$t\,[\mathrm{yr}]$'
        return r'$\log_{10}(z)$' if x_scale == 'log' else r'$z$'

    if x_scale == 'log':
        return 'Time log[yr]' if x_axis == 'years' else 'Redshift log[z]'
    return 'Time (yr)' if x_axis == 'years' else 'Redshift (z)'


def _axis_label_evolution_y(evolution_type, y_scale='lin', label_mode='verbal'):
    """Build evolution y-axis label in verbal or math mode."""
    if label_mode == 'math':
        if evolution_type == 'total':
            return r'$\log_{10}(E_B)$' if y_scale == 'log' else r'$E_B$'
        return r'$\partial_t E_B$'

    if y_scale == 'log':
        return 'Magnetic Energy log[erg]' if evolution_type == 'total' else 'Magnetic Energy Induction log[erg/s]'
    return 'Magnetic Energy (erg)' if evolution_type == 'total' else 'Magnetic Energy Induction (erg/s)'

def _axis_label_pd_y(kind, y_scale='lin', label_mode='verbal'):
    """Build production/dissipation y-axis labels in verbal or math mode."""
    if label_mode == 'math':
        if kind == 'absolute':
            return r'$\log_{10}|P,\,D,\,N|$' if y_scale == 'log' else r'$P,\,D,\,N$'
        if kind == 'fractional':
            return r'$d_i,\,-p_i,\,\iota$'
        if kind == 'net':
            return r'$N\equiv P-D$'
        if kind == 'net_integral':
            return r'$\int N\,dt$'
        if kind == 'cumulative_b':
            return r'$\sum_k E_{B,k}$'

    if kind == 'absolute':
        return 'Integrated Production / Dissipation log' if y_scale == 'log' else 'Integrated Production / Dissipation'
    if kind == 'fractional':
        return 'Fractional Contribution (-diss / +prod)'
    if kind == 'net':
        return 'Integrated Net Contribution'
    if kind == 'net_integral':
        return 'Integrated Net Contribution'
    if kind == 'cumulative_b':
        return 'Cumulative Magnetic Energy'
    return ''


def _apply_norm_vol_suffix(label, normalized=False, normalize_by_volume=False, label_mode='verbal'):
    """Append normalization/volume suffix preserving the selected label style."""
    if label_mode == 'math':
        suffix = ''
        if normalized:
            suffix += r'\,\rho_B^{-1}'
        if normalize_by_volume:
            suffix += r'\,V^{-1}'
        if suffix and label.startswith('$') and label.endswith('$'):
            return label[:-1] + suffix + '$'
        return label + suffix

    norm_suffix = r' ($\rho_B^{-1}$)' if normalized else ''
    vol_suffix = r' / Volume' if normalize_by_volume else ''
    return f'{label}{norm_suffix}{vol_suffix}'


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


def _cumulative_integral_series(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    n = min(len(x), len(y))
    if n == 0:
        return np.asarray([], dtype=float)

    x = x[:n]
    y = np.nan_to_num(y[:n], nan=0.0)
    cumulative = np.zeros(n, dtype=float)
    if n > 1:
        dx = np.abs(np.diff(x))
        cumulative[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return cumulative


def _smart_legend(ax, fig, plot_params=None, font_legend=None):
    """Place legend inside or outside depending on number of entries."""
    if font_legend is None:
        from matplotlib.font_manager import FontProperties
        font_legend = FontProperties()
        font_legend.set_size(11)

    # Collect handles/labels from the requested axis and all axes in the figure
    handles, labels = ax.get_legend_handles_labels()
    for other_ax in fig.axes:
        if other_ax is ax:
            continue
        oh, ol = other_ax.get_legend_handles_labels()
        if oh:
            handles = list(handles) + list(oh)
            labels = list(labels) + list(ol)

    # Filter no-legend entries and deduplicate preserving order
    entries = []
    seen_labels = set()
    for h, l in zip(handles, labels):
        if not l or l.startswith('_'):
            continue
        if l in seen_labels:
            continue
        seen_labels.add(l)
        entries.append((h, l))
    if not entries:
        return False
    legend_outside = len(entries) > (plot_params or {}).get('legend_outside_threshold', 12)
    labels_to_plot = [l for _, l in entries]
    handles_to_plot = [h for h, _ in entries]
    if legend_outside:
        fig.legend(
            handles_to_plot,
            labels_to_plot,
            prop=font_legend,
            ncol=min(4, len(entries)),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.03),
            borderaxespad=0.0,
        )
        # remove legends from all axes to avoid duplicates
        for other_ax in fig.axes:
            try:
                if getattr(other_ax, 'legend_', None) is not None:
                    other_ax.legend_.remove()
            except Exception:
                pass
    else:
        # Add legend only to the requested axis
        ax.legend(handles_to_plot, labels_to_plot, prop=font_legend, ncol=2)
    return legend_outside

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


def load_saved_analysis(data_folder, sim_name, iterations, level, verbose=True):
    """Load NPY analysis products for one simulation and AMR level."""
    root = os.path.join(data_folder, sim_name, 'analysis_exports', f'L{int(level)}_U{int(level)}')
    groups = {
        'induction_energy_integrals': {},
        'induction_energy_profiles': {},
        'production_dissipation_profiles': {},
        'percentiles': {},
    }
    group_indices = {group_name: [] for group_name in groups}
    grid_time, grid_zeta, rho_b, radii = [], [], [], []
    integral_snapshot_values = []

    for iteration in iterations:
        snapshot_root = os.path.join(root, f'it{int(iteration):05d}')
        metadata_root = os.path.join(snapshot_root, 'metadata')
        time_path = os.path.join(metadata_root, 'grid_time.npy')
        zeta_path = os.path.join(metadata_root, 'grid_zeta.npy')
        if not (os.path.isfile(time_path) and os.path.isfile(zeta_path)):
            continue

        grid_time.append(np.load(time_path, allow_pickle=True).item())
        grid_zeta.append(np.load(zeta_path, allow_pickle=True).item())
        for name, target, default in (
            ('rho_b', rho_b, 1.0),
            ('rad', radii, 0.0),
        ):
            path = os.path.join(metadata_root, f'{name}.npy')
            target.append(np.load(path, allow_pickle=True).item() if os.path.isfile(path) else default)

        snapshot_integrals = {}
        for group_name, target in groups.items():
            group_root = os.path.join(snapshot_root, group_name)
            if not os.path.isdir(group_root):
                continue
            group_indices[group_name].append(len(grid_time) - 1)
            for filename in os.listdir(group_root):
                if filename.endswith('.npy'):
                    value = np.load(os.path.join(group_root, filename), allow_pickle=True)
                    parsed_value = value.item() if value.ndim == 0 else value
                    if group_name == 'induction_energy_integrals':
                        snapshot_integrals[filename[:-4]] = parsed_value
                    else:
                        target.setdefault(filename[:-4], []).append(parsed_value)
        integral_snapshot_values.append(snapshot_integrals)

    if not grid_time:
        return None

    integral_keys = set().union(*(snapshot.keys() for snapshot in integral_snapshot_values))
    integral_data = groups['induction_energy_integrals']
    for key in sorted(integral_keys):
        missing_count = sum(key not in snapshot for snapshot in integral_snapshot_values)
        integral_data[key] = [snapshot.get(key, 0.0) for snapshot in integral_snapshot_values]
        if missing_count and verbose:
            print(
                f'[only_plot] WARNING: integral key {key} is missing in '
                f'{missing_count}/{len(integral_snapshot_values)} valid snapshots; '
                'missing values were kept as zero.'
            )

    signed_terms = ('diver', 'compres', 'stretch', 'advec', 'drag')
    family_suffixes = ('', '_solenoidal', '_compressive')
    for suffix in family_suffixes:
        for term in signed_terms:
            signed_key = f'int_MIE_{term}_B2{suffix}'
            prod_key = f'{signed_key}_prod'
            diss_key = f'{signed_key}_diss'
            if signed_key not in integral_data or prod_key not in integral_data or diss_key not in integral_data:
                continue
            signed_values = np.asarray(integral_data[signed_key], dtype=float)
            prod_values = np.asarray(integral_data[prod_key], dtype=float)
            diss_values = np.asarray(integral_data[diss_key], dtype=float)
            if np.allclose(signed_values, 0.0) and np.any(np.abs(prod_values - diss_values) > 0.0) and verbose:
                print(
                    f'[only_plot] WARNING: cached {signed_key} is entirely zero while '
                    f'{prod_key}/{diss_key} contain non-zero values; no fallback reconstruction was applied.'
                )

        signed_key = f'int_MIE_total_B2{suffix}'
        prod_key = f'{signed_key}_prod_compact'
        diss_key = f'{signed_key}_diss_compact'
        if signed_key in integral_data and prod_key in integral_data and diss_key in integral_data:
            signed_values = np.asarray(integral_data[signed_key], dtype=float)
            prod_values = np.asarray(integral_data[prod_key], dtype=float)
            diss_values = np.asarray(integral_data[diss_key], dtype=float)
            if np.allclose(signed_values, 0.0) and np.any(np.abs(prod_values - diss_values) > 0.0) and verbose:
                print(
                    f'[only_plot] WARNING: cached {signed_key} is entirely zero while '
                    f'{prod_key}/{diss_key} contain non-zero values; no fallback reconstruction was applied.'
                )

    # P/D radial products do not export the optional measured references. Reuse
    # them from induction profiles, aligning by the global snapshot index.
    reference_keys = ('clus_b2_profile', 'clus_rho_rho_b_profile')
    induction_index_map = {
        global_index: local_index
        for local_index, global_index in enumerate(group_indices['induction_energy_profiles'])
    }
    pd_indices = group_indices['production_dissipation_profiles']
    induction_profiles = groups['induction_energy_profiles']
    pd_profiles = groups['production_dissipation_profiles']
    for key in reference_keys:
        if key in pd_profiles or key not in induction_profiles:
            continue
        aligned_values = []
        source_values = induction_profiles[key]
        for global_index in pd_indices:
            source_index = induction_index_map.get(global_index)
            aligned_values.append(source_values[source_index] if source_index is not None else None)
        pd_profiles[key] = aligned_values

    if verbose:
        print(f'[only_plot] Loaded {len(grid_time)} snapshots for {sim_name}, level {level}')
        for group_name, group_data in groups.items():
            if group_data:
                print(f'[only_plot]   {group_name}: {len(group_data)} series')
            else:
                print(f'[only_plot]   {group_name}: NOT FOUND')
    return {
        'integral': groups['induction_energy_integrals'],
        'profiles': groups['induction_energy_profiles'],
        'pd_profiles': groups['production_dissipation_profiles'],
        'percentiles': groups['percentiles'],
        'profile_indices': group_indices['induction_energy_profiles'],
        'pd_profile_indices': group_indices['production_dissipation_profiles'],
        'percentile_indices': group_indices['percentiles'],
        'grid_time': grid_time,
        'grid_zeta': grid_zeta,
        'rho_b': rho_b,
        'rad': radii[-1] if radii else 0.0,
    }


def run_only_plot(active_sims, active_it, levels, data_folder, image_folder,
                  ind_params, evo_plot_params, prod_diss_plot_params,
                  ind_prof_plot_params, pd_prof_plot_params, percentile_plot_params,
                  save=True,
                  verbose=True):
    """Render configured plots exclusively from saved NPY analysis products."""
    from scripts.induction_evo import induction_energy_integral_evolution

    energy_cfg = ind_params.get('energy_evolution', {})
    pd_cfg = ind_params.get('production_dissipation', {})
    energy_enabled = bool(energy_cfg.get('enabled', False) and (
        energy_cfg.get('plot_total', False) or energy_cfg.get('plot_differential', False)))
    pd_enabled = bool(pd_cfg.get('enabled', False) and (
        pd_cfg.get('plot_absolute', False) or pd_cfg.get('plot_fractional', False) or pd_cfg.get('plot_net', False)))
    induction_profiles_enabled = bool(energy_cfg.get('_truly_enabled', False) and energy_cfg.get('plot_profiles', False))
    pd_profiles_enabled = bool(pd_cfg.get('_truly_enabled', False) and (
        pd_cfg.get('plot_profiles', False) or pd_cfg.get('plot_fractional_profiles', False)))
    if verbose:
        print(
            '[only_plot] profile gates: '
            f'induction={induction_profiles_enabled}, '
            f'production_dissipation={pd_profiles_enabled}'
        )
    rendered = False

    for sim_name, iterations in zip(active_sims, active_it):
        for level in levels:
            saved_data = load_saved_analysis(data_folder, sim_name, iterations, level, verbose=verbose)
            if saved_data is None:
                print(f'[only_plot] No saved NPY analysis for {sim_name}, level {level}; skipping.')
                continue

            plot_ind_params = ind_params.copy()
            plot_ind_params['up_to_level'] = level
            grid_time = saved_data['grid_time']
            grid_zeta = saved_data['grid_zeta']
            radius = saved_data['rad']

            has_evolution_snapshots = len(grid_time) >= 2
            if energy_enabled and saved_data['integral'] and has_evolution_snapshots:
                print(f'[only_plot] Plotting magnetic-energy evolution: {sim_name}, level {level}')
                evolution = induction_energy_integral_evolution(
                    ind_params['components'], saved_data['integral'],
                    energy_cfg['derivative'], saved_data['rho_b'], grid_time, grid_zeta,
                    normalized=energy_cfg.get('normalized', False), verbose=verbose)
                plot_integral_evolution(
                    evolution, evo_plot_params, plot_ind_params, grid_time, grid_zeta,
                    radius, verbose=verbose, save=save, folder=image_folder)
                rendered = True
            elif energy_enabled and saved_data['integral'] and verbose:
                print(
                    f'[only_plot] Skipping magnetic-energy evolution for {sim_name}, level {level}: '
                    'at least two saved snapshots are required.'
                )

            if pd_enabled and saved_data['integral'] and has_evolution_snapshots:
                print(f'[only_plot] Plotting production/dissipation evolution: {sim_name}, level {level}')
                figures = plot_production_dissipation_evolution(
                    saved_data['integral'], prod_diss_plot_params, plot_ind_params,
                    grid_time, grid_zeta, radius, verbose=verbose, save=save, folder=image_folder)
                rendered = rendered or bool(figures)
            elif pd_enabled and saved_data['integral'] and verbose:
                print(
                    f'[only_plot] Skipping production/dissipation evolution for {sim_name}, level {level}: '
                    'at least two saved snapshots are required.'
                )

            if induction_profiles_enabled and saved_data['profiles']:
                print(f'[only_plot] Plotting induction profiles: {sim_name}, level {level}')
                profile_params = ind_prof_plot_params.copy()
                profile_params['it_indx'] = saved_data['profile_indices']
                plot_induction_radial_profiles(
                    saved_data['profiles'], profile_params, plot_ind_params,
                    grid_time, grid_zeta, radius, verbose=verbose, save=save, folder=image_folder)
                rendered = True
            elif induction_profiles_enabled:
                print(
                    f'[only_plot] Induction profiles requested but no saved '
                    f'induction_energy_profiles were found for {sim_name}, level {level}.'
                )

            if pd_profiles_enabled and saved_data['pd_profiles']:
                print(f'[only_plot] Plotting production/dissipation profiles: {sim_name}, level {level}')
                profile_params = pd_prof_plot_params.copy()
                profile_params['it_indx'] = saved_data['pd_profile_indices']
                plot_production_dissipation_radial_profiles(
                    saved_data['pd_profiles'], profile_params, plot_ind_params,
                    grid_time, grid_zeta, radius, verbose=verbose, save=save, folder=image_folder)
                rendered = True
            elif pd_profiles_enabled:
                print(
                    f'[only_plot] Production/dissipation profiles requested but no saved '
                    f'production_dissipation_profiles were found for {sim_name}, level {level}.'
                )

            if ind_params.get('percentiles', {}).get('enabled', False) and saved_data.get('percentiles'):
                print(f'[only_plot] Plotting divergence percentiles: {sim_name}, level {level}')
                plot_percentile_evolution(
                    saved_data['percentiles'], percentile_plot_params, plot_ind_params,
                    grid_time, grid_zeta, verbose=verbose, save=save, folder=image_folder)
                rendered = True

    if not rendered:
        print('[only_plot] No compatible saved analysis products were available for plotting.')
    else:
        print('[only_plot] Plotting completed without recalculating simulation data.')

        
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


def setup_axis(ax, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis, evolution_type, font, plot_params=None):
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

        label_mode = _get_label_mode(plot_params)
        ax.set_xlabel(_axis_label_x(x_axis, x_scale='lin', label_mode=label_mode), fontproperties=font)
        ax.set_ylabel(_axis_label_evolution_y(evolution_type, y_scale='lin', label_mode=label_mode), fontproperties=font)
        
        if not cancel_limits and xlim:
            ax.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax.set_ylim(ylim[0], ylim[1])
            
        if x_scale == 'log':
            ax.set_xscale('log')
            ax.set_xlabel(_axis_label_x(x_axis, x_scale='log', label_mode=label_mode), fontproperties=font)
        if y_scale == 'log':
            ax.set_yscale('log')
            ax.set_ylabel(_axis_label_evolution_y(evolution_type, y_scale='log', label_mode=label_mode), fontproperties=font)
                
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
    label_mode = _get_label_mode(plot_params)
    if x_axis == 'years':
        x = np.array([grid_t[i] * time_to_yr / 1e9 for i in valid_indices], dtype=float)  # Convert to Gyr
        xlabel = _axis_label_x('years', x_scale='lin', label_mode=label_mode).replace('yr', 'Gyr')
    else:
        x = np.array([grid_zeta[i] for i in valid_indices], dtype=float)
        if x.size and x[-1] < 0:
            x[-1] = abs(x[-1])
        xlabel = _axis_label_x('zeta', x_scale='lin', label_mode=label_mode)

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
            ax.plot(x, pct_sorted[:, i], color=colors[i], linewidth=lw_pct, label=lbl)
            if parent_flag:
                buffer_info += f'_parent_{parent_interpol}'
            ax.plot(x, pct_sorted[:, i], color=colors[i], linewidth=lw_pct)
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
            - velocity_families: optional family selector for velocity-decomposed curves
                (total, solenoidal, compressive).
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
    label_mode = _get_label_mode(plot_params)
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
            mode_params['velocity_families'] = plot_params.get('velocity_families', None)
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
    requested_velocity_families = plot_params.get('velocity_families', None)
    plot_split = bool(plot_params.get('plot_split', False))
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
    plot_integrals = bool(plot_params.get('plot_integrals', False))
    
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
    color_compact_family = plot_params.get('compact_family_color', '#a84a63')
    color_kinetic = palette.get('kinetic_energy', DEFAULT_PLOT_PALETTE['kinetic_energy'])

    def _component_enabled(component_key):
        return bool(components_cfg.get(component_key, False))

    def _should_show_mechanism_legend_item(component_key, y_values):
        if not _component_enabled(component_key):
            return False
        return y_values is not None and should_plot_component(y_values, threshold=component_threshold)
    
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

    velocity_family_order = ('total', 'solenoidal', 'compressive')
    velocity_family_suffix = {
        'total': '',
        'solenoidal': '_solenoidal',
        'compressive': '_compressive',
    }
    velocity_family_styles = plot_params.get(
        'velocity_family_styles',
        {
            'total': '-',
            'solenoidal': '--',
            'compressive': ':',
        }
    )

    def _normalize_velocity_families(raw_families):
        if raw_families is None:
            active_fams = []
            vel_cfg = induction_params.get('velocity_field', {})
            for fam in velocity_family_order:
                if vel_cfg.get(fam, False) or any(f'_{fam}' in k for k in evolution_data.keys()):
                    active_fams.append(fam)
            return active_fams or ['total']
        if isinstance(raw_families, str):
            raw_families = [raw_families]
        normalized_families = []
        for family in raw_families:
            if family in velocity_family_order and family not in normalized_families:
                normalized_families.append(family)
        return normalized_families or ['total']

    def _series_key(base_key, family):
        return f'{base_key}{velocity_family_suffix[family]}{data_suffix}'

    def _load_family_series(base_key, family):
        effective_family = plot_family_context or family
        key = _series_key(base_key, effective_family)
        values = evolution_data.get(key)
        if values is None:
            return None, key
        return units * np.asarray(values, dtype=float), key

    def _format_family_label(base_label, family):
        return base_label if family == 'total' else f'{base_label} ({family})'

    def _format_component_label(component_label, sym, family):
        return rf'{component_label} $\Gamma_{{\mathrm{{{sym}}}}}$' if family == 'total' else rf'{component_label} $\Gamma_{{\mathrm{{{sym}}}}}$ ({family})'

    def _family_axis_suffix(family):
        if not family or family == 'total':
            return ''
        if label_mode == 'math':
            if family == 'solenoidal':
                return r' - $\mathrm{Solenoidal\ Velocity\ Field}$'
            if family == 'compressive':
                return r' - $\mathrm{Compressive\ Velocity\ Field}$'
        if family == 'solenoidal':
            return ' - Solenoidal Velocity Field'
        if family == 'compressive':
            return ' - Compressive Velocity Field'
        return f' - {family.title()} Velocity Field'
    
    y_title = 1.02
    line1, line2 = line_widths
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
        
    if derivative in ['RK', 'central', 'alpha_fit', 'rate']:
        index_o, index_f = (0 if derivative == 'RK' else 1), len(grid_t)
        plotid = f'{derivative}_{kind}'
    elif derivative == 'implicit_forward':
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
    yb2_cumulative = evolution_data.get('evo_b2_cumulative')
    xb2 = z if x_axis == 'zeta' else t
    volume_phi = evolution_data.get('evo_volume_phi', [])
    volume_co = evolution_data.get('evo_volume_co', [])

    base_axis_data = z if x_axis == 'zeta' else t
    selected_velocity_families = _normalize_velocity_families(requested_velocity_families)
    figures = []
    if plot_split and len(selected_velocity_families) > 1:
        plot_volume_once = bool(plot_params.get('volume_evolution', False))
        for family_index, family in enumerate(selected_velocity_families):
            split_params = plot_params.copy()
            split_params['velocity_families'] = [family]
            split_params['plot_split'] = False
            split_params['plot_family_context'] = family
            split_params['_internal_mode'] = True
            split_params['volume_evolution'] = plot_volume_once and family_index == 0
            figures.extend(
                plot_integral_evolution(
                    evolution_data, split_params, induction_params,
                    grid_t, grid_zeta, rad,
                    verbose=verbose, save=save, folder=folder
                )
            )
        return figures
    plot_family_context = plot_params.get('plot_family_context', None)
    split_family_mode = plot_family_context is not None
    family_series_defs = {
        'n1': 'evo_b2',
        'n0': 'evo_ind_b2',
        'diver_work': 'evo_MIE_diver_B2',
        'compres_work': 'evo_MIE_compres_B2',
        'stretch_work': 'evo_MIE_stretch_B2',
        'advec_work': 'evo_MIE_advec_B2',
        'drag_work': 'evo_MIE_drag_B2',
        'total_work': 'evo_MIE_total_B2',
    }

    family_series_raw_map = {
        family: {
            series_name: _load_family_series(base_key, family)[0]
            for series_name, base_key in family_series_defs.items()
        }
        for family in selected_velocity_families
    }

    if plot_family_context:
        velocity_family_tag = f'_family_{plot_family_context}'
    else:
        velocity_family_tag = '' if selected_velocity_families == ['total'] else '_vf_' + '-'.join(selected_velocity_families)
    plotid = f'{plotid}{velocity_family_tag}'
    
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

    def _prepare_family_series_map(raw_series_map):
        if plot_type == 'smoothed':
            return {
                family: {
                    series_name: (
                        gaussian_filter1d(series_data, sigma=smoothing_sigma)
                        if series_data is not None else None
                    )
                    for series_name, series_data in series_dict.items()
                }
                for family, series_dict in raw_series_map.items()
            }

        if plot_type == 'interpolated':
            family_interpolated_map = {}
            for family, series_dict in raw_series_map.items():
                interpolated_series = {}
                for series_name, series_data in series_dict.items():
                    if series_data is None:
                        interpolated_series[series_name] = None
                        continue
                    family_interp = interp1d(
                        base_axis_data[common_i0:common_i1],
                        np.asarray(series_data, dtype=float)[common_i0:common_i1],
                        kind=interpolation_kind,
                    )
                    interpolated_series[series_name] = family_interp(x_new)
                family_interpolated_map[family] = interpolated_series
            return family_interpolated_map

        return raw_series_map

    family_series_map = _prepare_family_series_map(family_series_raw_map)
    family_summary_legend = len(selected_velocity_families) > 1

    def _family_prefix_label(label, family):
        if split_family_mode:
            return label
        return f'{family} {label}' if len(selected_velocity_families) == 1 else label

    def _family_prefix_component_label(component_label, sym, family):
        base_label = rf'{component_label} $\Gamma_{{\mathrm{{{sym}}}}}$'
        return base_label if split_family_mode else _family_prefix_label(base_label, family)

    def _family_component_label(component_label, sym):
        return rf'{component_label} $\Gamma_{{\mathrm{{{sym}}}}}$'

    def _family_compact_label(family):
        if split_family_mode and family in ('solenoidal', 'compressive'):
            return f'...from Compact {family.title()} Induction'
        return '...from Compact Induction'

    active_mechanism_legend_items = [
        ('compression', 'Integrated Compression', compres_work_data, component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression'])),
        ('stretching', 'Integrated Stretching', stretch_work_data, component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching'])),
        ('advection', 'Integrated Advection', advec_work_data, component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection'])),
        ('divergence', 'Integrated Divergence', diver_work_data, component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence'])),
        ('drag', 'Integrated Cosmic Drag', drag_work_data, component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag'])),
    ]

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
        if selected_velocity_families != ['total']:
            log_message(f'  Velocity families: {", ".join(selected_velocity_families)}', tag='plots', level=1)
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
    
    # Main evolution plot
    fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
    if family_summary_legend:
        ax1.plot([], [], color='0.25', linestyle=velocity_family_styles.get('total', '-'), linewidth=line1, label='Velocity family: total')
        ax1.plot([], [], color='0.25', linestyle=velocity_family_styles.get('solenoidal', '--'), linewidth=line1, label='Velocity family: solenoidal')
        ax1.plot([], [], color='0.25', linestyle=velocity_family_styles.get('compressive', ':'), linewidth=line1, label='Velocity family: compressive')
        for component_key, compact_label, y_values, color in active_mechanism_legend_items:
            if _should_show_mechanism_legend_item(component_key, y_values):
                ax1.plot([], [], color=color, linestyle='-', linewidth=line2, label=compact_label)
    components_plotted = []
    family_components_plotted = []

    def _slice_xy(xvals, yvals, i0, i1):
        nxy = min(len(xvals), len(yvals))
        i0c = max(0, min(i0, nxy))
        i1c = max(i0c, min(i1, nxy))
        return xvals[i0c:i1c], yvals[i0c:i1c]

    def _slice_xy_with_offset(xvals, yvals, i0, i1):
        i0c = max(0, min(i0, len(xvals)))
        i1c = min(i1, len(xvals), i0 + len(yvals))
        i1c = max(i0c, i1c)
        count = i1c - i0c
        return xvals[i0c:i1c], yvals[:count]
    
    # Kinetic energy
    xk, yk = _slice_xy(x_data, kinetic_work_data, index_O_plot, index_F_plot)
    if plot_kinetic_energy and should_plot_component(yk, threshold=component_threshold):
        ax1.plot(xk, yk, 
            linewidth=line1, label='Kinetic Energy', color=color_kinetic)
        components_plotted.append('kinetic')
    
    # Main energy line (always plot)
    xm, ym = _slice_xy(x_data, n1_data, index_O_plot, index_F_plot)
    if plot_magnetic_energy and should_plot_component(ym, threshold=component_threshold):
        lbl = 'Magnetic Energy' if evolution_type == 'total' else 'Magnetic Energy Induction'
        ax1.plot(xm, ym, linewidth=line1, label=lbl, color=color_measured)
        components_plotted.append(lbl)
        
    # Total work (compacted)
    xt, yt = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)
    split_non_total_family = split_family_mode and plot_family_context != 'total'
    if not split_non_total_family and should_plot_component(yt, threshold=component_threshold):
        ax1.plot(xt, yt, '-', 
            linewidth=line1, label='...from Compact Induction', color=color_compact)
        components_plotted.append('total')

    # Induction prediction (plot if has data)
    x0, y0 = _slice_xy_with_offset(x_data, n0_data, index_o_plot, index_f_plot)
    if not split_non_total_family and should_plot_component(y0, threshold=component_threshold):
        ax1.plot(x0, y0, '--',
            linewidth=line1, label='...from Itemize Induction', color=color_itemized)

    # Individual components with their colors
    xp_item, yp_item = _slice_xy_with_offset(x_data, n0_data, index_o_plot, index_f_plot)
    xp_comp, yp_comp = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)

    if 'total' in selected_velocity_families:
        total_family_linestyle = velocity_family_styles.get('total', '-')
        total_itemized_linestyle = '-' if family_summary_legend else '--'
        total_family_component_linestyle = '--' if split_family_mode else total_family_linestyle
        family_total_configs = [
            ('total_work', '...from Compact Induction', color_compact, '-', line1, True),
            ('n0', '...from Itemize Induction', color_itemized, total_itemized_linestyle, line1, True),
        ]
        for series_name, label, color, linestyle, linewidth, use_offset in family_total_configs:
            family_data = family_series_map.get('total', {}).get(series_name)
            if family_data is None:
                continue
            if use_offset:
                x_family, y_family = _slice_xy_with_offset(x_data, family_data, index_o_plot, index_f_plot)
            else:
                x_family, y_family = _slice_xy(x_data, family_data, index_O_plot, index_F_plot)
            if not should_plot_component(y_family, threshold=component_threshold):
                continue
            ax1.plot(
                x_family,
                y_family,
                linestyle=linestyle,
                linewidth=linewidth,
                label=_family_prefix_label(label, 'total') if not family_summary_legend else '_nolegend_',
                color=color,
            )
            components_plotted.append(_family_prefix_label(label, 'total'))
        component_configs = [
            ('compression', compres_work_data, 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
            ('stretching', stretch_work_data, 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
            ('advection', advec_work_data, 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
            ('divergence', diver_work_data, 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
            ('drag', drag_work_data, 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag')
        ]
        for component_key, data, label, color, sym in component_configs:
            if not _component_enabled(component_key):
                continue
            xc, yc = _slice_xy_with_offset(x_data, data, index_o_plot, index_f_plot)
            if should_plot_component(yc, threshold=component_threshold):
                lbl = rf'{label} $\Gamma_{{\mathrm{{{sym}}}}}$'
                ax1.plot(
                    xc,
                    yc,
                    linestyle=total_family_component_linestyle,
                    linewidth=line2,
                    label=_family_prefix_component_label(label, sym, 'total') if not family_summary_legend else '_nolegend_',
                    color=color,
                )
                components_plotted.append(_family_prefix_component_label(label, sym, 'total'))

    # Decomposed velocity families
    if any(family != 'total' for family in selected_velocity_families):
        family_plot_plan = [
            ('compression', 'compres_work', 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp', line2, True),
            ('stretching', 'stretch_work', 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str', line2, True),
            ('advection', 'advec_work', 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv', line2, True),
            ('divergence', 'diver_work', 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div', line2, True),
            ('drag', 'drag_work', 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag', line2, True),
        ]

        for family in selected_velocity_families:
            if family == 'total':
                continue
            family_linestyle = velocity_family_styles.get(family, '--')
            family_total_linestyle = family_linestyle if family_summary_legend else None
            family_component_linestyle = '--' if split_family_mode else family_linestyle
            family_compact_color = color_compact_family if split_family_mode else color_compact
            family_compact_label = '...from Compact Induction' if split_family_mode else _family_compact_label(family)

            if split_family_mode:
                x_total_ref, y_total_ref = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)
                if should_plot_component(y_total_ref, threshold=component_threshold):
                    ax1.plot(
                        x_total_ref,
                        y_total_ref,
                        linestyle='-',
                        linewidth=line1,
                        label='...from Compact Induction (total)',
                        color=color_compact,
                        alpha=component_alpha,
                    )
                    family_components_plotted.append('...from Compact Induction (total)')

            family_total_configs = [
                ('total_work', family_compact_label, family_compact_color, '-', line1, True),
                ('n0', '...from Itemize Induction', color_itemized, '--' if split_family_mode else '--', line1, True),
            ]
            for series_name, label, color, linestyle, linewidth, use_offset in family_total_configs:
                family_data = family_series_map.get(family, {}).get(series_name)
                if family_data is None:
                    continue
                if use_offset:
                    x_family, y_family = _slice_xy_with_offset(x_data, family_data, index_o_plot, index_f_plot)
                else:
                    x_family, y_family = _slice_xy(x_data, family_data, index_O_plot, index_F_plot)
                if not should_plot_component(y_family, threshold=component_threshold):
                    continue
                ax1.plot(
                    x_family,
                    y_family,
                    linestyle=linestyle if split_family_mode else (family_total_linestyle if family_total_linestyle is not None else linestyle),
                    linewidth=linewidth,
                    label=_family_prefix_label(label, family) if not family_summary_legend else '_nolegend_',
                    color=color,
                    alpha=component_alpha,
                )
                family_components_plotted.append(_family_prefix_label(label, family))
            for component_key, series_name, base_label, color, sym, linewidth, use_offset in family_plot_plan:
                if not _component_enabled(component_key):
                    continue
                family_data = family_series_map.get(family, {}).get(series_name)
                if family_data is None:
                    continue
                if use_offset:
                    x_family, y_family = _slice_xy_with_offset(x_data, family_data, index_o_plot, index_f_plot)
                else:
                    x_family, y_family = _slice_xy(x_data, family_data, index_O_plot, index_F_plot)
                if not should_plot_component(y_family, threshold=component_threshold):
                    continue
                family_label = _family_prefix_component_label(base_label, sym, family)
                ax1.plot(
                    x_family,
                    y_family,
                    linestyle=family_component_linestyle,
                    linewidth=linewidth,
                    label=family_label if not family_summary_legend else '_nolegend_',
                    color=color,
                    alpha=component_alpha,
                )
                family_components_plotted.append(family_label)

    ax1_aux = None
    xb2 = None
    yb2_cumulative = None
    if evolution_type == 'differential':
        cumulative_units = plot_params.get('units', induction_params.get('units', 1.0))
        b2_total = cumulative_units * np.asarray(evolution_data.get('evo_b2', []), dtype=float)
        xb2, yb2 = _slice_xy(x_data, b2_total, index_O_plot, index_F_plot)
        if len(yb2) > 0 and should_plot_component(yb2, threshold=0.0):
            yb2_cumulative = np.cumsum(np.nan_to_num(yb2, nan=0.0))
            if plot_cumulative_magnetic_energy:
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
                bcum_label = _apply_norm_vol_suffix(
                    _axis_label_pd_y('cumulative_b', label_mode=label_mode),
                    normalized=normalized,
                    normalize_by_volume=False,
                    label_mode=label_mode,
                )
                ax1_aux.set_ylabel(bcum_label, fontproperties=font, color=color_measured)
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
    
    setup_axis(ax1, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis, evolution_type, font, plot_params=plot_params)
    if ax1_aux is not None:
        align_cumulative_overlay_zero(
            ax1,
            ax1_aux,
            y_aux_max=np.nanmax(yb2_cumulative) if yb2_cumulative is not None else None,
            headroom=cumulative_headroom,
        )
    evo_base_label = _axis_label_evolution_y(evolution_type, y_scale='lin', label_mode=label_mode)
    evo_base_label = f'{evo_base_label}{_family_axis_suffix(plot_family_context)}'
    ax1.set_ylabel(
        _apply_norm_vol_suffix(
            evo_base_label,
            normalized=normalized,
            normalize_by_volume=normalize_by_volume,
            label_mode=label_mode,
        ),
        fontproperties=font,
    )
    ax1.grid(alpha=0.3)
    _smart_legend(ax1, fig1, plot_params=plot_params, font_legend=font_legend)

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
            ax2.set_xlabel(_axis_label_x('years', x_scale='lin', label_mode=label_mode), fontproperties=font)
            ax2.plot(t, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(t, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
        else:  # zeta
            ax2.set_xlabel(_axis_label_x('zeta', x_scale='lin', label_mode=label_mode), fontproperties=font)
            ax2.plot(z, volume_phi, linewidth=line1, label='Physical')
            ax2.plot(z, volume_co, linewidth=line1, label='Comoving')
            ax2.set_xscale('log')
        
        ax2.set_ylabel('Integration Volume', fontproperties=font)
        # Use smart legend for volume plot
        ax2.legend(prop=font_legend)
        legend_outside_vol = _smart_legend(ax2, fig2, plot_params=plot_params, font_legend=font_legend)
        ax2.set_yscale('log')
        ax2.grid(alpha=0.3)
        
        ax2.set_title('Integrated Volume', y=y_title, fontproperties=font_title)
        fig2.tight_layout()
        figures.append(fig2)

    # Cumulative integrals plot
    fig_integral = None
    if plot_integrals and evolution_type == 'differential':
        fig_integral, ax_integral = plt.subplots(figsize=figure_size, dpi=dpi)
        if family_summary_legend:
            ax_integral.plot([], [], color='0.25', linestyle=velocity_family_styles.get('total', '-'), linewidth=line1, label='Velocity family: total')
            ax_integral.plot([], [], color='0.25', linestyle=velocity_family_styles.get('solenoidal', '--'), linewidth=line1, label='Velocity family: solenoidal')
            ax_integral.plot([], [], color='0.25', linestyle=velocity_family_styles.get('compressive', ':'), linewidth=line1, label='Velocity family: compressive')
            for component_key, compact_label, y_values, color in active_mechanism_legend_items:
                if _should_show_mechanism_legend_item(component_key, y_values):
                    ax_integral.plot([], [], color=color, linestyle='-', linewidth=line2, label=compact_label)
        integral_components_plotted = []

        def _plot_integral_curve(source_x, source_y, label, color, linestyle='-', linewidth=None, threshold=component_threshold, use_offset=False):
            if use_offset:
                source_x, source_y = _slice_xy_with_offset(source_x, source_y, index_o_plot, index_f_plot)
            else:
                source_x, source_y = _slice_xy(source_x, source_y, index_O_plot, index_F_plot)
            if not should_plot_component(source_y, threshold=threshold):
                return
            cumulative_y = _cumulative_integral_series(source_x, source_y)
            if len(cumulative_y) == 0:
                return
            ax_integral.plot(
                source_x[:len(cumulative_y)],
                cumulative_y,
                linestyle=linestyle,
                linewidth=linewidth if linewidth is not None else line1,
                color=color,
                label=label,
            )
            integral_components_plotted.append(label if len(selected_velocity_families) > 1 else _family_prefix_label(label, selected_velocity_families[0]))

        if plot_kinetic_energy:
            _plot_integral_curve(x_data, kinetic_work_data, 'Integrated Kinetic Energy', color_kinetic, linewidth=line1)

        if plot_magnetic_energy:
            label = 'Integrated Magnetic Energy' if evolution_type == 'total' else 'Integrated Magnetic Energy Induction'
            _plot_integral_curve(x_data, n1_data, label, color_measured, linewidth=line1)

        total_itemized_linestyle = '-' if family_summary_legend else '--'
        if not split_non_total_family:
            _plot_integral_curve(x_data, total_work_data, _family_prefix_label('...from Compact Induction', 'total'), color_compact, linestyle='-', linewidth=line1, threshold=component_threshold, use_offset=True)
            _plot_integral_curve(x_data, n0_data, _family_prefix_label('...from Itemize Induction', 'total'), color_itemized, linestyle='--' if split_family_mode else total_itemized_linestyle, linewidth=line1, threshold=component_threshold, use_offset=True)
        else:
            _plot_integral_curve(x_data, total_work_data, '...from Compact Induction (total)', color_compact, linestyle='-', linewidth=line1, threshold=component_threshold, use_offset=True)

        if selected_velocity_families == ['total']:
            component_integral_configs = [
                (compres_work_data, 'Integrated Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
                (stretch_work_data, 'Integrated Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
                (advec_work_data, 'Integrated Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
                (diver_work_data, 'Integrated Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
                (drag_work_data, 'Integrated Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag'),
            ]
            for data, label, color, sym in component_integral_configs:
                _plot_integral_curve(x_data, data, _family_component_label(label, sym), color, linestyle='--', linewidth=line2, threshold=component_threshold, use_offset=True)

        if 'total' in selected_velocity_families and family_summary_legend:
            family_integral_plot_plan_total = [
                ('compres_work', 'Integrated Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp', line2, True),
                ('stretch_work', 'Integrated Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str', line2, True),
                ('advec_work', 'Integrated Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv', line2, True),
                ('diver_work', 'Integrated Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div', line2, True),
                ('drag_work', 'Integrated Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag', line2, True),
            ]
            family = 'total'
            family_linestyle = velocity_family_styles.get(family, '-')
            family_total_linestyle = family_linestyle
            for series_name, base_label, color, sym, linewidth, use_offset in family_integral_plot_plan_total:
                family_data = family_series_map.get(family, {}).get(series_name)
                if family_data is None:
                    continue
                if use_offset:
                    x_family, y_family = _slice_xy_with_offset(x_data, family_data, index_o_plot, index_f_plot)
                else:
                    x_family, y_family = _slice_xy(x_data, family_data, index_O_plot, index_F_plot)
                if not should_plot_component(y_family, threshold=component_threshold):
                    continue
                cumulative_family = _cumulative_integral_series(x_family, y_family)
                if len(cumulative_family) == 0:
                    continue
                family_label = _family_component_label(base_label, sym)
                ax_integral.plot(
                    x_family[:len(cumulative_family)],
                    cumulative_family,
                    linestyle='-' if split_family_mode else family_total_linestyle,
                    linewidth=linewidth,
                    color=color,
                    label='_nolegend_',
                )
                integral_components_plotted.append(family_label)

        if any(family != 'total' for family in selected_velocity_families):
            family_integral_plot_plan = [
                ('total', 'total_work', '...from Compact Induction', color_compact, None, line1, True),
                ('total', 'n0', '...from Itemize Induction', color_itemized, None, line1, True),
                ('compression', 'compres_work', 'Integrated Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp', line2, True),
                ('stretching', 'stretch_work', 'Integrated Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str', line2, True),
                ('advection', 'advec_work', 'Integrated Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv', line2, True),
                ('divergence', 'diver_work', 'Integrated Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div', line2, True),
                ('drag', 'drag_work', 'Integrated Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag', line2, True),
            ]

            for family in selected_velocity_families:
                if family == 'total':
                    continue
                family_linestyle = velocity_family_styles.get(family, '--')
                family_total_linestyle = family_linestyle if family_summary_legend else None
                if split_family_mode:
                    x_total_ref, y_total_ref = _slice_xy_with_offset(x_data, total_work_data, index_o_plot, index_f_plot)
                    cumulative_total_ref = _cumulative_integral_series(x_total_ref, y_total_ref)
                    if len(cumulative_total_ref) > 0 and should_plot_component(y_total_ref, threshold=component_threshold):
                        ax_integral.plot(
                            x_total_ref[:len(cumulative_total_ref)],
                            cumulative_total_ref,
                            linestyle='-',
                            linewidth=line1,
                            color=color_compact,
                            label='...from Compact Induction (total)',
                        )
                        integral_components_plotted.append('...from Compact Induction (total)')
                for component_key, series_name, base_label, color, sym, linewidth, use_offset in family_integral_plot_plan:
                    if component_key not in ('total_work', 'n0') and not _component_enabled(component_key):
                        continue
                    family_data = family_series_map.get(family, {}).get(series_name)
                    if family_data is None:
                        continue
                    if use_offset:
                        x_family, y_family = _slice_xy_with_offset(x_data, family_data, index_o_plot, index_f_plot)
                    else:
                        x_family, y_family = _slice_xy(x_data, family_data, index_O_plot, index_F_plot)
                    if not should_plot_component(y_family, threshold=component_threshold):
                        continue
                    cumulative_family = _cumulative_integral_series(x_family, y_family)
                    if len(cumulative_family) == 0:
                        continue
                    family_label = _family_prefix_label(base_label, family) if sym is None else _family_component_label(base_label, sym)
                    ax_integral.plot(
                        x_family[:len(cumulative_family)],
                        cumulative_family,
                        linestyle='--' if split_family_mode and sym is not None else (
                            family_total_linestyle if family_total_linestyle is not None else ('-' if 'Compact' in base_label else '--')
                        ),
                        linewidth=linewidth,
                        color=color_compact_family if split_family_mode and series_name == 'total_work' else color,
                        label=family_label if not family_summary_legend else '_nolegend_',
                    )
                    integral_components_plotted.append(family_label)

        setup_axis(ax_integral, x_scale, 'lin', xlim, ylim, cancel_limits, x_axis, evolution_type, font, plot_params=plot_params)
        integral_base_label = r'$\int \partial_t E_B\,dt$' if label_mode == 'math' else 'Cumulative Integrated Contribution'
        integral_base_label = f'{integral_base_label}{_family_axis_suffix(plot_family_context)}'
        ax_integral.set_ylabel(
            _apply_norm_vol_suffix(
                integral_base_label,
                normalized=normalized,
                normalize_by_volume=normalize_by_volume,
                label_mode=label_mode,
            ),
            fontproperties=font,
        )
        ax_integral_aux = None
        if yb2_cumulative is not None and len(yb2_cumulative) > 0:
            ax_integral_aux = ax_integral.twinx()
            ax_integral_aux.plot(
                xb2,
                yb2_cumulative,
                '-',
                linewidth=max(1.2, line2),
                color=color_measured,
                alpha=0.45,
                label='Cumulative Magnetic Energy',
                zorder=1,
            )
            ax_integral_aux.fill_between(
                xb2,
                0.0,
                yb2_cumulative,
                color=color_measured,
                alpha=0.06,
                label='_nolegend_',
                zorder=1,
            )
            bcum_label = _apply_norm_vol_suffix(
                _axis_label_pd_y('cumulative_b', label_mode=label_mode),
                normalized=normalized,
                normalize_by_volume=False,
                label_mode=label_mode,
            )
            ax_integral_aux.set_ylabel(bcum_label, fontproperties=font, color=color_measured)
            ax_integral_aux.tick_params(axis='y', colors=color_measured)
            align_cumulative_overlay_zero(
                ax_integral,
                ax_integral_aux,
                y_aux_max=np.nanmax(yb2_cumulative),
                headroom=cumulative_headroom,
            )
        ax_integral.grid(alpha=0.3)
        _smart_legend(ax_integral, fig_integral, plot_params=plot_params, font_legend=font_legend)
        ax_integral.set_title(f'{title} (Cumulative Integrals)', y=y_title, fontproperties=font_title)
        if cancel_limits and x_axis == 'zeta':
            ax_integral.invert_xaxis()
        fig_integral.tight_layout()
        figures.append(fig_integral)
    
    if verbose:
        log_message(f'{plot_type.capitalize()} integrated magnetic energy and induction prediction plot created', tag='evolution', level=1)
        log_message(f'Components plotted: {", ".join(components_plotted)}', tag='evolution', level=1)
        if family_components_plotted:
            log_message(f'Family-specific components plotted: {", ".join(family_components_plotted)}', tag='evolution', level=1)
        if fig_integral is not None:
            log_message(f'Cumulative integral plot created with: {", ".join(integral_components_plotted)}', tag='evolution', level=1)
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
                print(f'Plotting... Volume plot saved as: {filename2}')

        if fig_integral is not None:
            filename3 = f'{folder}/{run}_{file_title}_integrals_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}_{plotid}{plot_suffix}.png'
            filename3 = safe_filename(filename3, verbose=verbose)
            fig_integral.savefig(filename3, dpi=dpi)
            if verbose:
                log_message(f'Cumulative integral plot saved as: {filename3}', tag='evolution', level=1)
                print(f'Plotting... Cumulative integral plot saved as: {filename3}')
                
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

    label_mode = _get_label_mode(plot_params)
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
    plot_integrals = bool(plot_params.get('plot_integrals', False))
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
    label_mode = _get_label_mode(plot_params)
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
        xlabel = _axis_label_x('years', x_scale='lin', label_mode=label_mode)
    else:
        x = np.array([grid_zeta[i] for i in range(len(grid_zeta))], dtype=float)
        if x.size and x[-1] < 0:
            x[-1] = abs(x[-1])
        xlabel = _axis_label_x('zeta', x_scale='lin', label_mode=label_mode)

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

    # Component palette used in plot_induction_radial_profiles
    component_map = [
        ('MIE_compres_B2', 'Compression', component_colors.get('compression', DEFAULT_PLOT_PALETTE['component_colors']['compression']), 'comp'),
        ('MIE_stretch_B2', 'Stretching', component_colors.get('stretching', DEFAULT_PLOT_PALETTE['component_colors']['stretching']), 'str'),
        ('MIE_advec_B2', 'Advection', component_colors.get('advection', DEFAULT_PLOT_PALETTE['component_colors']['advection']), 'adv'),
        ('MIE_diver_B2', 'Divergence', component_colors.get('divergence', DEFAULT_PLOT_PALETTE['component_colors']['divergence']), 'div'),
        ('MIE_drag_B2', 'Cosmic Drag', component_colors.get('drag', DEFAULT_PLOT_PALETTE['component_colors']['drag']), 'drag')
    ]

    plot_family_context = plot_params.get('plot_family_context', None)
    split_family_mode = plot_family_context is not None
    velocity_family_order = ('total', 'solenoidal', 'compressive')
    velocity_family_suffix = {
        'total': '',
        'solenoidal': '_solenoidal',
        'compressive': '_compressive',
    }
    velocity_family_styles = plot_params.get(
        'velocity_family_styles',
        {
            'total': '-',
            'solenoidal': '--',
            'compressive': ':',
        }
    )

    def _normalize_velocity_families(raw_families):
        if raw_families is None:
            active_fams = []
            vel_cfg = induction_params.get('velocity_field', {})
            for fam in velocity_family_order:
                fam_suffix = velocity_family_suffix[fam]
                if vel_cfg.get(fam, False) or any(fam_suffix and fam_suffix in key for key in pd_data.keys()):
                    active_fams.append(fam)
            return active_fams or ['total']
        if isinstance(raw_families, str):
            raw_families = [raw_families]
        normalized_families = []
        for family in raw_families:
            if family in velocity_family_order and family not in normalized_families:
                normalized_families.append(family)
        return normalized_families or ['total']

    def _family_axis_suffix(family):
        if not family or family == 'total':
            return ''
        if label_mode == 'math':
            if family == 'solenoidal':
                return r' - $\mathrm{Solenoidal\ Velocity\ Field}$'
            if family == 'compressive':
                return r' - $\mathrm{Compressive\ Velocity\ Field}$'
        if family == 'solenoidal':
            return ' - Solenoidal Velocity Field'
        if family == 'compressive':
            return ' - Compressive Velocity Field'
        return f' - {family.title()} Velocity Field'

    def _family_series_key(base_key, family, metric, compact=False):
        suffix = velocity_family_suffix.get(family, '')
        if compact:
            return f'int_{base_key}{suffix}_{metric}_compact'
        return f'int_{base_key}{suffix}_{metric}'

    def _family_series(family, base_key, metric, compact=False):
        key = _family_series_key(base_key, family, metric, compact=compact)
        values = pd_data.get(key)
        if values is None:
            return None, key
        return units * np.asarray(values, dtype=float), key

    def _diss_color(base_color):
        if not family_summary_legend:
            return base_color
        rgb = np.array(to_rgb(base_color), dtype=float)
        return tuple(np.clip(0.55 * rgb + 0.45, 0.0, 1.0))

    def _compact_family_color(family):
        if family == 'total':
            return color_compact_net
        rgb = np.array(to_rgb(color_compact_net), dtype=float)
        return tuple(np.clip(0.55 * rgb + 0.45, 0.0, 1.0))

    def _total_compact_reference_net():
        total_prod, _ = _family_series('total', 'MIE_total_B2', 'prod', compact=True)
        total_diss, _ = _family_series('total', 'MIE_total_B2', 'diss', compact=True)
        if total_prod is None or total_diss is None:
            return None
        return np.asarray(total_prod, dtype=float) - np.asarray(total_diss, dtype=float)

    def _format_family_label(base_label, family):
        if split_family_mode:
            return base_label
        return base_label if family == 'total' else f'{base_label} ({family})'

    component_key_map = {
        'MIE_compres_B2': 'compression',
        'MIE_stretch_B2': 'stretching',
        'MIE_advec_B2': 'advection',
        'MIE_diver_B2': 'divergence',
    }

    def _has_family_series(family, base_key, metric, compact=False):
        values, _ = _family_series(family, base_key, metric, compact=compact)
        if values is None:
            return False
        return should_plot_component(values, threshold=epsilon)

    requested_velocity_families = plot_params.get('velocity_families', None)
    plot_split = bool(plot_params.get('plot_split', False))
    selected_velocity_families = _normalize_velocity_families(requested_velocity_families)
    family_summary_legend = plot_family_context is None and len(selected_velocity_families) > 1
    components_cfg = induction_params.get('components', {})
    itemized_enabled = bool(components_cfg.get('itemized', False))

    # Accept either itemized totals or compact totals as the plotting source.
    has_any_family_totals = any(
        (
            f'int_MIE_total_B2{velocity_family_suffix.get(family, "")}_prod' in pd_data and
            f'int_MIE_total_B2{velocity_family_suffix.get(family, "")}_diss' in pd_data
        ) or (
            f'int_MIE_total_B2{velocity_family_suffix.get(family, "")}_prod_compact' in pd_data and
            f'int_MIE_total_B2{velocity_family_suffix.get(family, "")}_diss_compact' in pd_data
        )
        for family in selected_velocity_families
    )
    if not has_any_family_totals:
        if verbose:
            print('Production/dissipation plot skipped: no valid integrated P/D data')
        return []

    if plot_split and len(selected_velocity_families) > 1 and plot_family_context is None:
        figures = []
        plot_volume_once = bool(plot_params.get('volume_evolution', False))
        for family_index, family in enumerate(selected_velocity_families):
            split_params = plot_params.copy()
            split_params['velocity_families'] = [family]
            split_params['plot_split'] = False
            split_params['plot_family_context'] = family
            split_params['_internal_mode'] = True
            split_params['volume_evolution'] = plot_volume_once and family_index == 0
            figures.extend(
                plot_production_dissipation_evolution(
                    pd_data, split_params, induction_params,
                    grid_t, grid_zeta, rad,
                    verbose=verbose, save=save, folder=folder
                )
            )
        return figures

    family_series_map = {}
    for family in selected_velocity_families:
        fam_itemized_prod = _family_series(family, 'MIE_total_B2', 'prod', compact=False)[0]
        fam_itemized_diss = _family_series(family, 'MIE_total_B2', 'diss', compact=False)[0]
        fam_itemized_net = None if fam_itemized_prod is None or fam_itemized_diss is None else (fam_itemized_prod - fam_itemized_diss)
        family_series_map[family] = {
            'total_prod_itemized': fam_itemized_prod if itemized_enabled else None,
            'total_diss_itemized': fam_itemized_diss if itemized_enabled else None,
            'total_net_itemized': fam_itemized_net if itemized_enabled else None,
            'total_prod_compact': _family_series(family, 'MIE_total_B2', 'prod', compact=True)[0],
            'total_diss_compact': _family_series(family, 'MIE_total_B2', 'diss', compact=True)[0],
            'drag_prod': _family_series(family, 'MIE_drag_B2', 'prod')[0],
            'drag_diss': _family_series(family, 'MIE_drag_B2', 'diss')[0],
            'components': {
                'compression': {
                    'prod': _family_series(family, 'MIE_compres_B2', 'prod')[0],
                    'diss': _family_series(family, 'MIE_compres_B2', 'diss')[0],
                },
                'stretching': {
                    'prod': _family_series(family, 'MIE_stretch_B2', 'prod')[0],
                    'diss': _family_series(family, 'MIE_stretch_B2', 'diss')[0],
                },
                'advection': {
                    'prod': _family_series(family, 'MIE_advec_B2', 'prod')[0],
                    'diss': _family_series(family, 'MIE_advec_B2', 'diss')[0],
                },
                'divergence': {
                    'prod': _family_series(family, 'MIE_diver_B2', 'prod')[0],
                    'diss': _family_series(family, 'MIE_diver_B2', 'diss')[0],
                },
            },
        }

    def _relative_max_diff(reference, candidate):
        ref = np.asarray(reference, dtype=float)
        cand = np.asarray(candidate, dtype=float)
        nxy = min(ref.size, cand.size)
        if nxy == 0:
            return None
        ref = ref[:nxy]
        cand = cand[:nxy]
        denom = np.maximum(np.abs(cand), epsilon)
        diff = np.abs(ref - cand) / denom
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            return None
        return float(np.nanmax(diff))

    if verbose and all(components_cfg.get(key, False) for key in ('compression', 'stretching', 'advection', 'drag')):
        for family in selected_velocity_families:
            sfx = velocity_family_suffix.get(family, '')
            itemized_prod_f, _ = _family_series(family, 'MIE_total_B2', 'prod', compact=False)
            itemized_diss_f, _ = _family_series(family, 'MIE_total_B2', 'diss', compact=False)
            compact_prod_f, _ = _family_series(family, 'MIE_total_B2', 'prod', compact=True)
            compact_diss_f, _ = _family_series(family, 'MIE_total_B2', 'diss', compact=True)

            if itemized_prod_f is None or itemized_diss_f is None or compact_prod_f is None or compact_diss_f is None:
                continue

            rel_prod_comp = _relative_max_diff(itemized_prod_f, compact_prod_f)
            rel_diss_comp = _relative_max_diff(itemized_diss_f, compact_diss_f)
            if rel_prod_comp is None or rel_diss_comp is None:
                continue

            itemized_net_f = itemized_prod_f - itemized_diss_f
            compact_net_f = compact_prod_f - compact_diss_f
            rel_net = _relative_max_diff(itemized_net_f, compact_net_f)

            print(
                f'[{family}] Production/dissipation totals: itemized vs compact '
                f'(max rel diff P={rel_prod_comp:.3e}, D={rel_diss_comp:.3e}, N={rel_net:.3e}). '
                'Expected near-zero differences when the compact and itemized decompositions are numerically consistent.'
            )

    def _family_plot_style(family):
        return velocity_family_styles.get(family, '-')

    def _family_efficiency_color(family):
        if family == 'total':
            return color_efficiency
        rgb = np.array(to_rgb(color_efficiency), dtype=float)
        return tuple(np.clip(0.62 * rgb + 0.38, 0.0, 1.0))

    def _add_compact_legend_proxies(ax, include_totals=False, include_components=False, include_fractional=False, include_net=False, include_integrated_net=False):
        if not family_summary_legend:
            return
        for family in selected_velocity_families:
            ax.plot(
                [], [],
                color='0.25',
                linestyle=_family_plot_style(family),
                linewidth=line_main,
                label=f'Velocity family: {family}',
            )
        if include_totals:
            if plot_total_prod_diss:
                ax.plot([], [], color=color_prod, linestyle='-', linewidth=line_main, label=r'Total Production $P_{\mathrm{tot}}$')
                ax.plot([], [], color=_diss_color(color_diss), linestyle='-', linewidth=line_main, label=r'Total Dissipation $D_{\mathrm{tot}}$')
            if itemized_enabled:
                ax.plot([], [], color=color_itemized_net, linestyle='-', linewidth=line_main, label=r'Net (itemized) $N_{\mathrm{tot}}$')
            ax.plot([], [], color=color_compact_net, linestyle='-', linewidth=line_main, label=r'Net (compact) $N_{\mathrm{tot}}$')
        if include_fractional:
            ax.plot([], [], color=color_efficiency, linestyle='-', linewidth=line_main, label=r'Net Efficiency $\iota$')
        if include_net:
            net_prefix = 'Integrated ' if include_integrated_net else ''
            if itemized_enabled:
                ax.plot([], [], color=color_itemized_net, linestyle='-', linewidth=line_main, label=f'{net_prefix}Net total (itemized)')
            ax.plot([], [], color=color_compact_net, linestyle='-', linewidth=line_main, label=f'{net_prefix}Net total (compact)')
        if include_components:
            frac_prefix = include_fractional
            comp_prefix = 'Integrated ' if include_integrated_net else ''
            for prefix, label, color, sym in component_map:
                has_any = any(
                    _has_family_series(family, prefix, 'prod') or _has_family_series(family, prefix, 'diss')
                    for family in selected_velocity_families
                )
                if has_any:
                    if include_net:
                        ax.plot([], [], color=color, linestyle='-', linewidth=line_comp, label=rf'{comp_prefix}{label} $N_{{\mathrm{{{sym}}}}}$')
                    else:
                        prod_symbol = 'p' if frac_prefix else 'P'
                        diss_symbol = 'd' if frac_prefix else 'D'
                        ax.plot([], [], color=color, linestyle='-', linewidth=line_comp, label=rf'{comp_prefix}{label} ${prod_symbol}_{{\mathrm{{{sym}}}}}$')
                        ax.plot([], [], color=_diss_color(color), linestyle='-', linewidth=line_comp, label=rf'{comp_prefix}{label} ${diss_symbol}_{{\mathrm{{{sym}}}}}$')

    figures = []

    def _overlay_cumulative_magnetic_energy(ax, enabled=True, use_normalized=True):
        if not enabled:
            return None
        
        # Prefer normalized version (int_B2) if it exists and is requested; fall back to int_b2
        b2_key = None
        if use_normalized and normalized and 'int_B2' in pd_data:
            b2_key = 'int_B2'
        elif 'int_b2' in pd_data:
            b2_key = 'int_b2'
        
        if b2_key is None:
            return None

        b2_snap = units * np.asarray(pd_data.get(b2_key, []), dtype=float)
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
        b_label = _apply_norm_vol_suffix(
            _axis_label_pd_y('cumulative_b', label_mode=label_mode),
            normalized=normalized,
            normalize_by_volume=False,
            label_mode=label_mode,
        )
        ax_aux.set_ylabel(
            b_label,
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
        _smart_legend(ax, ax.get_figure(), plot_params=None, font_legend=font_legend)

    # Absolute production/dissipation rates
    if plot_absolute:
        fig_abs, ax_abs = plt.subplots(figsize=figure_size, dpi=dpi)

        for family in selected_velocity_families:
            family_style = _family_plot_style(family) if family_summary_legend else None
            family_series = family_series_map.get(family, {})

            if split_family_mode and family != 'total':
                total_net_reference = _total_compact_reference_net()
                if total_net_reference is not None and should_plot_component(total_net_reference, threshold=epsilon):
                    ax_abs.plot(
                        x,
                        total_net_reference,
                        '-',
                        linewidth=line_main,
                        color=color_compact_net,
                        label=r'Net total (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$',
                    )

            if plot_total_prod_diss:
                total_prod_series = family_series.get('total_prod_itemized')
                total_diss_series = family_series.get('total_diss_itemized')
                total_prod_label = r'Total Production $P_{\mathrm{tot}}$'
                total_diss_label = r'Total Dissipation $D_{\mathrm{tot}}$'
                if total_prod_series is None or total_diss_series is None:
                    total_prod_series = family_series.get('total_prod_compact')
                    total_diss_series = family_series.get('total_diss_compact')
                    total_prod_label = r'Total Production (compact) $P_{\mathrm{tot}}$'
                    total_diss_label = r'Total Dissipation (compact) $D_{\mathrm{tot}}$'
                if total_prod_series is not None and should_plot_component(total_prod_series):
                    ax_abs.plot(
                        x, total_prod_series,
                        '-.' if not family_summary_legend else family_style,
                        linewidth=line_main,
                        color=color_prod,
                        label=_format_family_label(total_prod_label, family) if not family_summary_legend else '_nolegend_',
                    )
                if total_diss_series is not None and should_plot_component(total_diss_series):
                    ax_abs.plot(
                        x, total_diss_series,
                        '-.' if not family_summary_legend else family_style,
                        linewidth=line_main,
                        color=_diss_color(color_diss),
                        label=_format_family_label(total_diss_label, family) if not family_summary_legend else '_nolegend_',
                    )

            total_net_itemized = family_series.get('total_net_itemized')
            if itemized_enabled and total_net_itemized is not None and should_plot_component(total_net_itemized):
                ax_abs.plot(
                    x, total_net_itemized,
                    '--' if not family_summary_legend else family_style,
                    linewidth=line_main,
                    color=color_itemized_net,
                    label=_format_family_label(r'Net (itemized) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$', family) if not family_summary_legend else '_nolegend_',
                )

            total_net_compact = None if family_series.get('total_prod_compact') is None or family_series.get('total_diss_compact') is None else (family_series['total_prod_compact'] - family_series['total_diss_compact'])
            if total_net_compact is not None and should_plot_component(total_net_compact):
                ax_abs.plot(
                    x, total_net_compact,
                    '-' if not family_summary_legend else family_style,
                    linewidth=line_main,
                    color=_compact_family_color(family),
                    label=_format_family_label(r'Net (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$', family) if not family_summary_legend else '_nolegend_',
                )

            for prefix, label, color, sym in component_map:
                if prefix == 'MIE_drag_B2' and family != 'total':
                    continue
                component_key = component_key_map.get(prefix)
                prod_series = family_series.get('drag_prod') if prefix == 'MIE_drag_B2' else family_series.get('components', {}).get(component_key, {}).get('prod')
                diss_series = family_series.get('drag_diss') if prefix == 'MIE_drag_B2' else family_series.get('components', {}).get(component_key, {}).get('diss')
                if prod_series is not None and should_plot_component(prod_series):
                    ax_abs.plot(
                        x, prod_series,
                        '--' if not family_summary_legend else family_style,
                        linewidth=line_comp,
                        color=color,
                        label=_format_family_label(rf'{label} $P_{{\mathrm{{{sym}}}}}$', family) if not family_summary_legend else '_nolegend_',
                    )
                if diss_series is not None and should_plot_component(diss_series):
                    ax_abs.plot(
                        x, diss_series,
                        ':' if not family_summary_legend else family_style,
                        linewidth=line_comp,
                        color=_diss_color(color),
                        label=_format_family_label(rf'{label} $D_{{\mathrm{{{sym}}}}}$', family) if not family_summary_legend else '_nolegend_',
                    )

        _add_compact_legend_proxies(ax_abs, include_totals=True, include_components=True)

        ax_abs.set_xlabel(xlabel, fontproperties=font)
        abs_label = _apply_norm_vol_suffix(
            _axis_label_pd_y('absolute', y_scale='lin', label_mode=label_mode),
            normalized=normalized,
            normalize_by_volume=normalize_by_volume,
            label_mode=label_mode,
        )
        ax_abs.set_ylabel(f'{abs_label}{_family_axis_suffix(plot_family_context)}', fontproperties=font)
        if x_scale == 'log':
            ax_abs.set_xscale('log')
            ax_abs.set_xlabel(_axis_label_x(x_axis, x_scale='log', label_mode=label_mode), fontproperties=font)
        if y_scale == 'log':
            ax_abs.set_yscale('log')
            abs_log_label = _apply_norm_vol_suffix(
                _axis_label_pd_y('absolute', y_scale='log', label_mode=label_mode),
                normalized=normalized,
                normalize_by_volume=normalize_by_volume,
                label_mode=label_mode,
            )
            ax_abs.set_ylabel(f'{abs_log_label}{_family_axis_suffix(plot_family_context)}', fontproperties=font)
        if not cancel_limits and xlim:
            ax_abs.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax_abs.set_ylim(ylim[0], ylim[1])
        if cancel_limits and x_axis == 'zeta':
            ax_abs.invert_xaxis()

        ax_abs.grid(alpha=0.3)
        plot_title_short = title.split('-')[0].strip()
        if _get_label_mode(plot_params) == 'math':
            plot_title_short = r'$E_B$ Evolution'
        ax_abs.set_title(f'{plot_title_short} - {region_label}{_family_axis_suffix(plot_family_context) if not split_family_mode else ""}', y=y_title, fontproperties=font_title)
        legend_outside = _smart_legend(ax_abs, fig_abs, plot_params=plot_params, font_legend=font_legend)
        if legend_outside:
            fig_abs.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            fig_abs.tight_layout()
        figures.append(fig_abs)

    # Fractional contributions and net efficiency
    if plot_fractional:
        fig_frac, ax_frac = plt.subplots(figsize=figure_size, dpi=dpi)

        for family in selected_velocity_families:
            family_style = _family_plot_style(family) if family_summary_legend else None
            family_suffix = velocity_family_suffix.get(family, '')
            if split_family_mode and family != 'total' and 'int_PD_iota' in pd_data:
                total_iota = np.asarray(pd_data['int_PD_iota'], dtype=float)
                if should_plot_component(total_iota, threshold=epsilon):
                    ax_frac.plot(
                        x,
                        total_iota,
                        '-',
                        linewidth=line_main,
                        color=color_efficiency,
                        label=r'Net Efficiency $\iota$ (total)',
                    )
                fam_iota_key = f'int_PD_iota{family_suffix}'
                if fam_iota_key in pd_data:
                    fam_iota = np.asarray(pd_data[fam_iota_key], dtype=float)
                    if should_plot_component(fam_iota, threshold=epsilon):
                        ax_frac.plot(
                            x,
                            fam_iota,
                            '-',
                            linewidth=line_main,
                            color=_family_efficiency_color(family),
                            label=r'Net Efficiency $\iota$',
                        )
            for prefix, label, color, sym in component_map:
                if prefix == 'MIE_drag_B2' and family != 'total':
                    continue
                if prefix == 'MIE_drag_B2':
                    prod_series = family_series_map[family].get('drag_prod')
                    diss_series = family_series_map[family].get('drag_diss')
                else:
                    component_key = {
                        'MIE_compres_B2': 'compression',
                        'MIE_stretch_B2': 'stretching',
                        'MIE_advec_B2': 'advection',
                        'MIE_diver_B2': 'divergence',
                    }.get(prefix)
                    prod_series = family_series_map[family].get('components', {}).get(component_key, {}).get('prod')
                    diss_series = family_series_map[family].get('components', {}).get(component_key, {}).get('diss')
                if prod_series is not None:
                    frac_p = np.asarray(pd_data.get(f'int_PD_frac_{prefix}{family_suffix}_prod', []), dtype=float)
                    if len(frac_p) > 0 and should_plot_component(frac_p, threshold=epsilon):
                        ax_frac.plot(
                            x, frac_p,
                            '--' if not family_summary_legend else family_style,
                            linewidth=line_comp,
                            color=color,
                            label=rf'{label} $p_{{\mathrm{{{sym}}}}}$' if not family_summary_legend else '_nolegend_',
                        )
                if diss_series is not None:
                    frac_d = np.asarray(pd_data.get(f'int_PD_frac_{prefix}{family_suffix}_diss', []), dtype=float)
                    if len(frac_d) > 0 and should_plot_component(frac_d, threshold=epsilon):
                        ax_frac.plot(
                            x, -frac_d,
                            ':' if not family_summary_legend else family_style,
                            linewidth=line_comp,
                            color=_diss_color(color),
                            label=rf'{label} $d_{{\mathrm{{{sym}}}}}$' if not family_summary_legend else '_nolegend_',
                        )

            iota_key = f'int_PD_iota{family_suffix}'
            if iota_key in pd_data and not (split_family_mode and family != 'total'):
                iota = np.asarray(pd_data[iota_key], dtype=float)
                if should_plot_component(iota, threshold=epsilon):
                    ax_frac.plot(
                        x, iota,
                        '-' if not family_summary_legend else family_style,
                        linewidth=line_main,
                        color=_family_efficiency_color(family),
                        label=r'Net Efficiency $\iota$' if not family_summary_legend else '_nolegend_',
                    )

        _add_compact_legend_proxies(ax_frac, include_fractional=True, include_components=True)

        ax_frac.set_xlabel(xlabel, fontproperties=font)
        ax_frac.set_ylabel(f'{_axis_label_pd_y("fractional", label_mode=label_mode)}{_family_axis_suffix(plot_family_context)}', fontproperties=font)
        if x_scale == 'log':
            ax_frac.set_xscale('log')
            ax_frac.set_xlabel(_axis_label_x(x_axis, x_scale='log', label_mode=label_mode), fontproperties=font)
        if not cancel_limits and xlim:
            ax_frac.set_xlim(xlim[0], xlim[1])
        ax_frac.set_ylim(-1.05, 1.05)
        if cancel_limits and x_axis == 'zeta':
            ax_frac.invert_xaxis()

        ax_frac.grid(alpha=0.3)
        frac_title = f'{title} (Fractions)'
        if _get_label_mode(plot_params) == 'math':
            frac_title = r'Fractional Contributions'
        ax_frac.set_title(f'{frac_title} - {region_label}{_family_axis_suffix(plot_family_context) if not split_family_mode else ""}', y=y_title, fontproperties=font_title)
        legend_outside = _smart_legend(ax_frac, fig_frac, plot_params=plot_params, font_legend=font_legend)
        if legend_outside:
            fig_frac.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            fig_frac.tight_layout()
        figures.append(fig_frac)

    # Net contributions: per-component net and total net curves
    if plot_net:
        fig_net, ax_net = plt.subplots(figsize=figure_size, dpi=dpi)

        for family in selected_velocity_families:
            family_style = _family_plot_style(family) if family_summary_legend else None
            family_series = family_series_map.get(family, {})
            family_display = 'total' if family == 'total' else family.title()
            if split_family_mode and family != 'total':
                total_net_reference = _total_compact_reference_net()
                if total_net_reference is not None and should_plot_component(total_net_reference, threshold=epsilon):
                    ax_net.plot(
                        x,
                        total_net_reference,
                        '-',
                        linewidth=line_main,
                        color=color_compact_net,
                        label=r'Net total (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$',
                    )
            family_compact_net = None if family_series.get('total_prod_compact') is None or family_series.get('total_diss_compact') is None else (family_series['total_prod_compact'] - family_series['total_diss_compact'])
            if family_compact_net is not None and should_plot_component(family_compact_net, threshold=epsilon):
                ax_net.plot(
                    x,
                    family_compact_net,
                    '-',
                    linewidth=line_main,
                    color=_compact_family_color(family),
                    label=(r'Net (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$'
                           if family != 'total' else
                           r'Net total (compact) $P_{\mathrm{tot}}-D_{\mathrm{tot}}$'),
                )
            for prefix, label, color, sym in component_map:
                if prefix == 'MIE_drag_B2' and family != 'total':
                    continue
                if prefix == 'MIE_drag_B2':
                    arr_p = family_series.get('drag_prod')
                    arr_d = family_series.get('drag_diss')
                else:
                    component_key = {
                        'MIE_compres_B2': 'compression',
                        'MIE_stretch_B2': 'stretching',
                        'MIE_advec_B2': 'advection',
                        'MIE_diver_B2': 'divergence',
                    }.get(prefix)
                    arr_p = family_series.get('components', {}).get(component_key, {}).get('prod')
                    arr_d = family_series.get('components', {}).get(component_key, {}).get('diss')
                if arr_p is not None and arr_d is not None:
                    net_i = arr_p - arr_d
                    if should_plot_component(net_i, threshold=epsilon):
                        ax_net.plot(
                            x,
                            net_i,
                            '--' if not family_summary_legend else family_style,
                            linewidth=line_comp,
                            color=color,
                            label=rf'{label} $N_{{\mathrm{{{sym}}}}}$' if not family_summary_legend else '_nolegend_',
                        )

            family_display = 'total' if family == 'total' else family.title()

            family_itemized_net = family_series.get('total_net_itemized')
            if itemized_enabled and family_itemized_net is not None and should_plot_component(family_itemized_net, threshold=epsilon):
                ax_net.plot(
                    x,
                    family_itemized_net,
                    '--' if not family_summary_legend else family_style,
                    linewidth=line_main,
                    color=color_itemized_net,
                    label=f'Net {family_display} (itemized)' if not family_summary_legend else '_nolegend_',
                )

        _add_compact_legend_proxies(ax_net, include_net=True, include_components=True)

        ax_net.set_xlabel(xlabel, fontproperties=font)
        net_label = _apply_norm_vol_suffix(
            _axis_label_pd_y('net', label_mode=label_mode),
            normalized=normalized,
            normalize_by_volume=normalize_by_volume,
            label_mode=label_mode,
        )
        ax_net.set_ylabel(f'{net_label}{_family_axis_suffix(plot_family_context)}', fontproperties=font)
        if x_scale == 'log':
            ax_net.set_xscale('log')
            ax_net.set_xlabel(_axis_label_x(x_axis, x_scale='log', label_mode=label_mode), fontproperties=font)
        if not cancel_limits and xlim:
            ax_net.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax_net.set_ylim(ylim[0], ylim[1])
        if cancel_limits and x_axis == 'zeta':
            ax_net.invert_xaxis()

        ax_net.grid(alpha=0.3)
        net_title = f'{title} (Net)'
        if _get_label_mode(plot_params) == 'math':
            net_title = r'Net Contributions $N$'
        ax_net.set_title(f'{net_title} - {region_label}{_family_axis_suffix(plot_family_context) if not split_family_mode else ""}', y=y_title, fontproperties=font_title)
        ax_net_aux = _overlay_cumulative_magnetic_energy(ax_net, enabled=plot_cumulative_magnetic_energy, use_normalized=normalized)
        legend_outside = _smart_legend(ax_net, fig_net, plot_params=plot_params, font_legend=font_legend)
        if legend_outside:
            fig_net.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            fig_net.tight_layout()
        figures.append(fig_net)

    fig_net_integral = None
    if plot_net and plot_integrals:
        fig_net_integral, ax_net_integral = plt.subplots(figsize=figure_size, dpi=dpi)
        integral_components_plotted = []

        def _plot_integral_curve(source_x, source_y, label, color, linestyle='-', linewidth=None, threshold=epsilon):
            if not should_plot_component(source_y, threshold=threshold):
                return
            cumulative_y = _cumulative_integral_series(source_x, source_y)
            if len(cumulative_y) == 0:
                return
            ax_net_integral.plot(
                source_x[:len(cumulative_y)],
                cumulative_y,
                linestyle=linestyle,
                linewidth=linewidth if linewidth is not None else line_main,
                color=color,
                label=label,
            )
            integral_components_plotted.append(label)

        if split_family_mode and selected_velocity_families != ['total']:
            total_net_reference = _total_compact_reference_net()
            if total_net_reference is not None and should_plot_component(total_net_reference, threshold=epsilon):
                ax_net_integral.plot(
                    x,
                    _cumulative_integral_series(x, total_net_reference),
                    '-',
                    linewidth=line_main,
                    color=color_compact_net,
                    label=r'Integrated Net total (compact) $\int N\,dt$',
                )

        for family in selected_velocity_families:
            family_style = _family_plot_style(family) if family_summary_legend else None
            family_series = family_series_map.get(family, {})
            family_compact_net = None if family_series.get('total_prod_compact') is None or family_series.get('total_diss_compact') is None else (family_series['total_prod_compact'] - family_series['total_diss_compact'])
            if family_compact_net is not None:
                _plot_integral_curve(
                    x,
                    family_compact_net,
                    (r'Integrated Net (compact) $\int N\,dt$'
                     if family != 'total' else
                     r'Integrated Net total (compact) $\int N\,dt$'),
                    _compact_family_color(family),
                    linestyle='-',
                    linewidth=line_main,
                )
            for prefix, label, color, sym in component_map:
                if prefix == 'MIE_drag_B2' and family != 'total':
                    continue
                if prefix == 'MIE_drag_B2':
                    arr_p = family_series.get('drag_prod')
                    arr_d = family_series.get('drag_diss')
                else:
                    component_key = {
                        'MIE_compres_B2': 'compression',
                        'MIE_stretch_B2': 'stretching',
                        'MIE_advec_B2': 'advection',
                        'MIE_diver_B2': 'divergence',
                    }.get(prefix)
                    arr_p = family_series.get('components', {}).get(component_key, {}).get('prod')
                    arr_d = family_series.get('components', {}).get(component_key, {}).get('diss')
                if arr_p is not None and arr_d is not None:
                    net_i = arr_p - arr_d
                    _plot_integral_curve(
                        x,
                        net_i,
                        rf'Integrated {label} $N_{{\mathrm{{{sym}}}}}$' if not family_summary_legend else '_nolegend_',
                        color,
                        linestyle='--' if not family_summary_legend else family_style,
                        linewidth=line_comp,
                    )

            family_itemized_net = family_series.get('total_net_itemized')
            if itemized_enabled and family_itemized_net is not None:
                _plot_integral_curve(
                    x,
                    family_itemized_net,
                    f'Integrated Net {family_display} (itemized)' if not family_summary_legend else '_nolegend_',
                    color_itemized_net,
                    linestyle='--' if not family_summary_legend else family_style,
                    linewidth=line_main,
                )

        _add_compact_legend_proxies(ax_net_integral, include_net=True, include_components=True, include_integrated_net=True)

        ax_net_integral.set_xlabel(xlabel, fontproperties=font)
        net_int_label = _apply_norm_vol_suffix(
            _axis_label_pd_y('net_integral', label_mode=label_mode),
            normalized=normalized,
            normalize_by_volume=normalize_by_volume,
            label_mode=label_mode,
        )
        ax_net_integral.set_ylabel(f'{net_int_label}{_family_axis_suffix(plot_family_context)}', fontproperties=font)
        if x_scale == 'log':
            ax_net_integral.set_xscale('log')
            ax_net_integral.set_xlabel(_axis_label_x(x_axis, x_scale='log', label_mode=label_mode), fontproperties=font)
        if not cancel_limits and xlim:
            ax_net_integral.set_xlim(xlim[0], xlim[1])
        if not cancel_limits and ylim:
            ax_net_integral.set_ylim(ylim[0], ylim[1])
        if cancel_limits and x_axis == 'zeta':
            ax_net_integral.invert_xaxis()

        ax_net_integral.grid(alpha=0.3)
        net_int_title = f'{title} (Net Integrals)'
        if _get_label_mode(plot_params) == 'math':
            net_int_title = r'Integrated Net Contributions $\int N\,dt$'
        ax_net_integral.set_title(f'{net_int_title} - {region_label}{_family_axis_suffix(plot_family_context) if not split_family_mode else ""}', y=y_title, fontproperties=font_title)
        ax_net_integral_aux = _overlay_cumulative_magnetic_energy(ax_net_integral, enabled=plot_cumulative_magnetic_energy, use_normalized=normalized)
        legend_outside = _smart_legend(ax_net_integral, fig_net_integral, plot_params=plot_params, font_legend=font_legend)
        if legend_outside:
            fig_net_integral.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            fig_net_integral.tight_layout()
        figures.append(fig_net_integral)

    if save and figures:
        if folder is None:
            folder = os.getcwd()

        sim_info = f'{induction_params["up_to_level"]}_{induction_params["F"]}_{induction_params["vir_kind"]}vir_{induction_params["rad_kind"]}rad_{induction_params["region"]}Region'
        if plot_family_context:
            family_info = f'_family_{plot_family_context}'
        elif len(selected_velocity_families) > 1:
            family_info = '_vf_' + '-'.join(selected_velocity_families)
        else:
            family_info = ''
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
            fname_abs = f'{folder}/{run}_{base_title}{family_info}_prod_diss_abs_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
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
            fname_frac = f'{folder}/{run}_{base_title}{family_info}_prod_diss_frac_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_frac = safe_filename(fname_frac, verbose=verbose)
            figures[frac_idx].savefig(fname_frac, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation fractional plot saved as: {fname_frac}')

        if net_idx is not None and len(figures) > net_idx:
            fname_net = f'{folder}/{run}_{base_title}{family_info}_prod_diss_net_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_net = safe_filename(fname_net, verbose=verbose)
            figures[net_idx].savefig(fname_net, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation net plot saved as: {fname_net}')

        if fig_net_integral is not None:
            fname_integral = f'{folder}/{run}_{base_title}{family_info}_prod_diss_integrals_{units_info}_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil", "")}.png'
            fname_integral = safe_filename(fname_integral, verbose=verbose)
            fig_net_integral.savefig(fname_integral, dpi=dpi)
            if verbose:
                print(f'Plotting... Production/Dissipation cumulative integrals plot saved as: {fname_integral}')

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
    aux_tick_labelsize = plot_params.get('aux_tick_labelsize', 11)
    aux_density_offset = plot_params.get('aux_density_offset', 1.18)
    fixed_legend = bool(plot_params.get('fixed_legend', False))
    figure_size = plot_params.get('figure_size', [12, 8])
    line_widths = plot_params.get('line_widths', [3, 1.5])
    title = plot_params.get('title', 'Magnetic Field Radial Profiles')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    velocity_family_order = ('total', 'solenoidal', 'compressive')
    velocity_family_suffix = {
        'total': '',
        'solenoidal': '_solenoidal',
        'compressive': '_compressive',
    }
    requested_velocity_families = plot_params.get('velocity_families', None)
    if requested_velocity_families is None:
        requested_velocity_families = [
            family for family in velocity_family_order
            if induction_params.get('velocity_field', {}).get(family, family == 'total')
        ]
    elif isinstance(requested_velocity_families, str):
        requested_velocity_families = [requested_velocity_families]
    selected_velocity_families = [
        family for family in velocity_family_order
        if family in requested_velocity_families
    ] or ['total']
    family_context = plot_params.get('_family_context', None)
    plot_split = bool(plot_params.get('plot_split', False))
    if plot_split and len(selected_velocity_families) > 1 and family_context is None:
        figures = []
        for family in selected_velocity_families:
            split_params = plot_params.copy()
            split_params['velocity_families'] = [family]
            split_params['plot_split'] = False
            split_params['_family_context'] = family
            figures.extend(plot_induction_radial_profiles(
                profile_data, split_params, induction_params,
                grid_t, grid_zeta, rad, verbose=verbose,
                save=save, folder=folder,
            ))
        return figures
    family_styles = plot_params.get(
        'velocity_family_styles',
        {'total': '-', 'solenoidal': '--', 'compressive': ':'},
    )
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
    
    # Coerce missing/scalar/mismatched profiles to aligned radial arrays.
    def series_array(key, scale):
        raw = profile_data.get(key, None)
        out = []
        for i in range(len(it_indx)):
            try:
                value = raw[i] if raw is not None else 0.0
            except (IndexError, TypeError, KeyError):
                value = raw if raw is not None and np.isscalar(raw) else 0.0

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

            arr[~np.isfinite(arr)] = np.nan
            out.append(scale * arr)
        return out

    components_cfg = induction_params.get('components', {})
    plot_kinetic_energy = bool(components_cfg.get('kinetic_energy', True))

    base_profiles = {
        'total': 'MIE_total_B2', 'itemized': 'ind_b2',
        'compression': 'MIE_compres_B2', 'stretching': 'MIE_stretch_B2',
        'advection': 'MIE_advec_B2', 'divergence': 'MIE_diver_B2',
        'drag': 'MIE_drag_B2',
    }
    family_profiles = {}
    for family in selected_velocity_families:
        suffix = velocity_family_suffix[family]
        family_profiles[family] = {}
        for name, base_key in base_profiles.items():
            if name == 'itemized':
                key = 'ind_b2_profile' if family == 'total' else '__missing_itemized_profile__'
            elif name == 'drag':
                key = 'MIE_drag_B2_profile'
            else:
                key = f'{base_key}{suffix}_profile'
            family_profiles[family][name] = series_array(key, units_y_1)

    split_family_reference_profiles = None
    if family_context is not None and family_context != 'total':
        split_family_reference_profiles = {
            'total': series_array('MIE_total_B2_profile', units_y_1),
            'itemized': series_array('ind_b2_profile', units_y_1),
        }

    reference_profiles = {
        'kinetic': series_array('kinetic_energy_profile', units_y_2),
        'magnetic': series_array('clus_b2_profile', units_y_2),
        'density': series_array('clus_rho_rho_b_profile', units_y_3),
    }
    all_profiles = [reference_profiles] + list(family_profiles.values())
    if plot_type == 'smoothed':
        for profiles in all_profiles:
            for key in profiles:
                profiles[key] = [gaussian_filter1d(arr, sigma=smoothing_sigma) for arr in profiles[key]]
        r_pro = r
        plot_suffix = f'smoothed_sigma_{smoothing_sigma}'
    elif plot_type == 'interpolated':
        r_new = np.linspace(min(r), max(r), num=interpolation_points, endpoint=True)
        for profiles in all_profiles:
            for key in profiles:
                profiles[key] = [interp1d(r, arr, kind=interpolation_kind, bounds_error=False, fill_value=np.nan) for arr in profiles[key]]
        r_pro = r_new
        plot_suffix = f'{interpolation_kind}_interpolated_{interpolation_points}_points'
    else:
        r_pro = r
        plot_suffix = 'raw'

    figures = []
    components_configs = []
    mechanism_specs = [
        ('compression', r'Compression $\Gamma_{\mathrm{comp}}$', 'compression', line2, component_alpha),
        ('stretching', r'Stretching $\Gamma_{\mathrm{str}}$', 'stretching', line2, component_alpha),
        ('advection', r'Advection $\Gamma_{\mathrm{adv}}$', 'advection', line2, component_alpha),
        ('divergence', r'Divergence $\Gamma_{\mathrm{div}}$', 'divergence', line2, component_alpha),
        ('drag', r'Cosmic Drag $\Gamma_{\mathrm{drag}}$', 'drag', line2, component_alpha),
    ]
    family_overlay = len(selected_velocity_families) > 1 and family_context is None
    energy_label_prefix = 'Magnetic Energy '
    family_compact_color = palette.get('induction_compact', DEFAULT_PLOT_PALETTE['induction_compact'])
    if family_context is not None and family_context != 'total':
        family_compact_rgb = np.asarray(to_rgb(family_compact_color), dtype=float)
        family_compact_color = tuple(np.clip(0.65 * family_compact_rgb + 0.35, 0.0, 1.0))
    for family in selected_velocity_families:
        style = family_styles.get(family, '-')
        family_label = ''
        profiles = family_profiles[family]
        compact_style = '-' if len(selected_velocity_families) == 1 and family == 'total' else style
        itemized_style = '--' if len(selected_velocity_families) == 1 and family == 'total' else style
        mechanism_style = '--' if family_context is not None or (len(selected_velocity_families) == 1 and family == 'total') else style
        components_configs.extend([
            (profiles['total'], f'{energy_label_prefix}from Compact Induction{family_label}' if family_context is None or family_context == 'total' else f'...from Compact Induction{family_label}', family_compact_color if family_context is not None else palette.get('induction_compact', DEFAULT_PLOT_PALETTE['induction_compact']), line1, '-' if family_context is not None else compact_style, AXIS_MAIN, 1.0),
            (profiles['itemized'], f'...from Itemized Induction{family_label}', palette.get('induction_itemized', DEFAULT_PLOT_PALETTE['induction_itemized']), line1, '--' if family_context is not None else itemized_style, AXIS_MAIN, 1.0),
        ])
        if split_family_reference_profiles is not None:
            components_configs.insert(
                0,
                (split_family_reference_profiles['total'], 'Magnetic Energy from Compact Induction (total)', palette.get('induction_compact', DEFAULT_PLOT_PALETTE['induction_compact']), line1, '-', AXIS_MAIN, 1.0),
            )
        for key, label, color_key, linewidth, alpha_curve in mechanism_specs:
            if family != 'total' and key == 'drag':
                continue
            if not components_cfg.get(key, True):
                continue
            components_configs.append((
                profiles[key], f'{label}{family_label}',
                component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key]),
                linewidth, mechanism_style, AXIS_MAIN, alpha_curve,
            ))

    # These fields are not decomposed upstream. Plot them once, with total-family
    # styling in overlay/total plots, and as references in split family plots.
    if (
        'total' in selected_velocity_families
        or family_context is not None
        or plot_magnetic_energy
        or plot_density
        or plot_kinetic_energy
    ):
        reference_style = family_styles.get('total', '-')
        if plot_kinetic_energy:
            components_configs.insert(0, (reference_profiles['kinetic'], 'Kinetic Energy Density', palette.get('kinetic_energy', DEFAULT_PLOT_PALETTE['kinetic_energy']), line1, reference_style, AXIS_ENERGY, 1.0))
        if plot_magnetic_energy:
            components_configs.insert(0, (reference_profiles['magnetic'], 'Magnetic Energy Density', palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']), line1, reference_style, AXIS_ENERGY, 1.0))
        if plot_density:
            components_configs.insert(0, (reference_profiles['density'], 'Density', palette.get('density', DEFAULT_PLOT_PALETTE['density']), line1, reference_style, AXIS_DENSITY, 1.0))

    # Decide which components have data (per snapshot we check existence)
    def has_nonzero(arr_or_callable, snap_idx, threshold=induction_params.get('epsilon', 1e-30)):
        if callable(arr_or_callable):
            try:
                y = arr_or_callable(r_pro)
                return np.any(np.isfinite(y)) and np.any(np.abs(y[np.isfinite(y)]) > threshold)
            except Exception:
                return False
        try:
            a = arr_or_callable[snap_idx]
            a = np.asarray(a, dtype=float)
            finite = np.isfinite(a)
            return np.any(finite) and np.any(np.abs(a[finite]) > threshold)
        except Exception:
            return False
        
    # Helper to plot signed data: family line style for positive values and markers for negatives.
    def plot_signed(ax, x, y, lw, ls, color, label, alpha=1.0, eps=induction_params.get('epsilon', 1e-30)):
        """
        Plot a single continuous line where positive intervals use `ls` and
        negative samples are marked with dots.
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

        # Build positive segments; negative values are represented by markers.
        segments = []
        sign_styles = []
        
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

        # Overlay only positive segments with the family's line style.
        lc_pos = LineCollection(
            [seg for seg, is_pos in zip(segments, sign_styles) if is_pos],
            linewidths=lw, colors=color, linestyles=ls, label='_nolegend_', alpha=alpha
        )
        ax.add_collection(lc_pos)

        negative = yv < 0
        if np.any(negative):
            ax.plot(
                xv[negative], yabs[negative],
                linestyle='None', marker='.', markersize=max(3.0, lw * 1.8),
                color=color, alpha=alpha, label='_nolegend_',
            )

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
        label_mode = _get_label_mode(plot_params)
        fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
        ax_energy = None
        ax_density = None
        if plot_magnetic_energy or plot_kinetic_energy:
            ax_energy = ax1.twinx()
        if plot_density:
            ax_density = ax1.twinx()
            if ax_energy is not None:
                ax_density.spines["right"].set_position(("axes", aux_density_offset))
            ax_density.set_frame_on(True)
            ax_density.patch.set_visible(False)
            for sp in ax_density.spines.values():
                sp.set_visible(True)

        snap_index = it_indx[snap_i]
        snap_z = np.abs(np.round(grid_zeta[snap_index], 2))
        z_text = f"{snap_z:6.2f}"
        ax1.set_title(f'{title} - z = {z_text}, $R_{{Vir}}$ = {np.round(rad,1)} Mpc', y=y_title, fontproperties=font_title)

        unique_handles = []
        unique_labels = []
        y_main_vals = []
        y_energy_vals = []
        y_density_vals = []
        negative_interval_found = False

        for (data_list, label, color, lw, ls, axis_type, alpha_curve) in components_configs:
            if not has_nonzero(data_list, snap_i):
                continue

            if callable(data_list[snap_i]):
                y = data_list[snap_i](r_pro)
            else:
                y = np.asarray(data_list[snap_i])
                if plot_type == 'interpolated' and y.size == r.size and r_pro.size != r.size:
                    y = np.interp(r_pro, r, y)
            negative_interval_found |= bool(np.any(np.asarray(y)[np.isfinite(y)] < 0))

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
            if label_mode == 'math':
                ax1.set_xlabel(r'$\log_{10}\!\left(r/R_{\mathrm{Vir}}\right)$', fontproperties=font)
            else:
                ax1.set_xlabel('Radial Distance log[r/$R_{Vir}$]', fontproperties=font)
        else:
            if label_mode == 'math':
                ax1.set_xlabel(r'$r/R_{\mathrm{Vir}}$', fontproperties=font)
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
            if label_mode == 'math':
                ax1.set_ylabel(r'$\mathrm{Induction\ Density}\ (\mathrm{erg}\,\mathrm{Mpc}^{-3}\,\mathrm{s}^{-1})$', fontproperties=font)
            else:
                ax1.set_ylabel('Induction Density (erg/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                if label_mode == 'math':
                    ax_energy.set_ylabel(r'$\mathrm{Energy\ Density}\ (\mathrm{erg}\,\mathrm{Mpc}^{-3})$', fontproperties=font)
                else:
                    ax_energy.set_ylabel('Energy Density (erg/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                if label_mode == 'math':
                    ax_density.set_ylabel(r'$\mathrm{Density}\ (\mathrm{g\,cm}^{-3})$', fontproperties=font)
                else:
                    ax_density.set_ylabel('Density (g/cm³)', fontproperties=font)
        elif units == energy_to_J:
            if label_mode == 'math':
                ax1.set_ylabel(r'$\mathrm{Induction\ Density}\ (\mathrm{J}\,\mathrm{Mpc}^{-3}\,\mathrm{s}^{-1})$', fontproperties=font)
            else:
                ax1.set_ylabel('Induction Density (J/$Mpc^{3}$/s)', fontproperties=font)
            if ax_energy is not None:
                if label_mode == 'math':
                    ax_energy.set_ylabel(r'$\mathrm{Energy\ Density}\ (\mathrm{J}\,\mathrm{Mpc}^{-3})$', fontproperties=font)
                else:
                    ax_energy.set_ylabel('Energy Density (J/$Mpc^{3}$)', fontproperties=font)
            if ax_density is not None:
                if label_mode == 'math':
                    ax_density.set_ylabel(r'$M_{\odot}\,\mathrm{Mpc}^{-3}$', fontproperties=font)
                else:
                    ax_density.set_ylabel('Density (M$_{\odot}$/Mpc³)', fontproperties=font)
        else:
            ax1.set_ylabel('Induction Density (arb. units / s)', fontproperties=font)
            if ax_energy is not None:
                ax_energy.set_ylabel('Energy Density (arb. units)', fontproperties=font)
            if ax_density is not None:
                ax_density.set_ylabel('Density (arb. units)', fontproperties=font)

        # Keep auxiliary axes consistent with the P/D radial-profile plots.
        if ax_energy is not None:
            if label_mode == 'math':
                if units == energy_to_erg:
                    energy_label = r'$\rho_{B}\ (\mathrm{erg}\,\mathrm{Mpc}^{-3})$'
                elif units == energy_to_J:
                    energy_label = r'$\rho_{B}\ (\mathrm{J}\,\mathrm{Mpc}^{-3})$'
                else:
                    energy_label = r'$\rho_{B}\ (\mathrm{arb.\ units})$'
            else:
                energy_label = ('Magnetic Energy Density (erg/$Mpc^{3}$)' if units == energy_to_erg
                                else 'Magnetic Energy Density (J/$Mpc^{3}$)' if units == energy_to_J
                                else 'Magnetic Energy Density (arb. units)')
            ax_energy.set_ylabel(energy_label, fontproperties=font, color=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']))
            ax_energy.tick_params(axis='y', colors=palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy']))
        if ax_density is not None:
            if label_mode == 'math':
                if units == energy_to_erg:
                    density_label = r'$\rho\ (\mathrm{g}\,\mathrm{cm}^{-3})$'
                elif units == energy_to_J:
                    density_label = r'$\rho\ (M_{\odot}\,\mathrm{Mpc}^{-3})$'
                else:
                    density_label = r'$\rho\ (\mathrm{arb.\ units})$'
            else:
                density_label = ('Density (g/cm$^{3}$)' if units == energy_to_erg
                                 else 'Density (M$_{\odot}$/Mpc$^{3}$)' if units == energy_to_J
                                 else 'Density (arb. units)')
            ax_density.set_ylabel(density_label, fontproperties=font, color=palette.get('density', DEFAULT_PLOT_PALETTE['density']))
            ax_density.tick_params(axis='y', colors=palette.get('density', DEFAULT_PLOT_PALETTE['density']))

        if family_context is not None and family_context != 'total':
            ax1.set_ylabel(f'{ax1.get_ylabel()} - {family_context.title()} Velocity Field', fontproperties=font)

        ax1.grid(alpha=0.3)
        if ax_energy is not None:
            ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax_energy.tick_params(axis='y', labelsize=aux_tick_labelsize)
        if ax_density is not None:
            ax_density.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax_density.tick_params(axis='y', labelsize=aux_tick_labelsize)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax_energy.get_legend_handles_labels() if ax_energy is not None else ([], [])
        h3, l3 = ax_density.get_legend_handles_labels() if ax_density is not None else ([], [])
        all_handles = h1 + h2 + h3
        all_labels = l1 + l2 + l3
        from matplotlib.lines import Line2D
        if negative_interval_found:
            all_handles.insert(0, Line2D(
                [0], [0], color=color_negative_interval, linestyle='None',
                marker='.', markersize=max(4.0, line2 * 1.8),
                label='Negative Interval'))
            all_labels.insert(0, 'Negative Interval')
        seen = set()
        for hh, ll in zip(all_handles, all_labels):
            if ll and not ll.startswith('_') and ll not in seen:
                seen.add(ll)
                unique_handles.append(hh)
                unique_labels.append(ll)

        if unique_handles:
            if family_overlay:
                family_handles = [
                    Line2D([0], [0], color='0.25', linestyle=family_styles.get(family, '-'),
                           linewidth=line1, label=f'Velocity family: {family}')
                    for family in selected_velocity_families
                ]
                # Keep family identity in the legend without repeating it on every curve.
                unique_handles = family_handles + unique_handles
                unique_labels = [handle.get_label() for handle in family_handles] + unique_labels
            if fixed_legend:
                ax1.legend(unique_handles, unique_labels, prop=font_legend,
                           loc='lower left', bbox_to_anchor=(0.02, 0.02),
                           bbox_transform=ax1.transAxes, ncol=2, frameon=True)
                legend_outside = False
            elif family_overlay:
                ax1.legend(unique_handles, unique_labels, prop=font_legend,
                           ncol=2, frameon=True)
                legend_outside = False
            else:
                # place axis legend first and let _smart_legend decide if it must move below
                ax1.legend(unique_handles, unique_labels, prop=font_legend, ncol=2, frameon=True)
                legend_outside = _smart_legend(ax1, fig1, plot_params=plot_params, font_legend=font_legend)
            if legend_outside:
                fig1.tight_layout(rect=[0, 0.08, 1, 1])
            else:
                fig1.tight_layout()
        else:
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
        family_info = f'_family_{family_context}' if family_context is not None else ''
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
            file_name = f'{folder}/{run}_{file_title}{family_info}_induction_profile_{sim_info}_{axis_info}_{limit_info}_{buffer_info}_{diff_cfg.get("stencil","")}_{plot_suffix}_{i}_{timestamp}.png'
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
    interpolation_kind = plot_params.get('interpolation_kind', 'cubic')
    assert interpolation_kind in ['linear', 'cubic', 'nearest'], "interpolation_kind must be 'linear', 'cubic', or 'nearest'"
    it_indx = list(plot_params.get('it_indx', []))
    if not it_indx:
        raise ValueError('it_indx must be provided and contain at least one index')

    x_scale = plot_params.get('x_scale', 'lin')
    y_scale = plot_params.get('y_scale', 'log')
    xlim = plot_params.get('xlim')
    ylim = plot_params.get('ylim')
    rylim = plot_params.get('rylim')
    dylim = plot_params.get('dylim')
    aux_tick_labelsize = plot_params.get('aux_tick_labelsize', 11)
    aux_density_offset = plot_params.get('aux_density_offset', 1.18)
    fixed_legend = bool(plot_params.get('fixed_legend', False))
    figure_size = plot_params.get('figure_size', [12, 8])
    line_widths = plot_params.get('line_widths', [3, 1.5])
    line_main = line_widths[0]
    line_comp = line_widths[1] if len(line_widths) > 1 else line_main
    title = plot_params.get('title', 'Production and Dissipation Radial Profile')
    dpi = plot_params.get('dpi', 300)
    run = plot_params.get('run', '_')
    threshold = float(plot_params.get('component_threshold', induction_params.get('differentiation', {}).get('epsilon', 1e-30)))
    smoothing_sigma = float(plot_params.get('smoothing_sigma', 1.10))
    interpolation_points = int(plot_params.get('interpolation_points', 500))
    plot_absolute = bool(plot_params.get('plot_absolute', True))
    plot_net = bool(plot_params.get('plot_net', False))
    plot_reconstructed_net = bool(plot_params.get('plot_reconstructed_net', False))
    plot_composed = bool(plot_params.get('plot_composed', False))
    production_dissipation_cfg = induction_params.get('production_dissipation', {})
    plot_profiles = bool(production_dissipation_cfg.get('plot_profiles', False))
    plot_fractional = bool(production_dissipation_cfg.get('plot_fractional_profiles', False))
    if not plot_profiles:
        if verbose:
            print('Production/dissipation radial profiles disabled by production_dissipation.plot_profiles')
        return []
    plot_density = bool(plot_params.get('plot_density', False))
    plot_magnetic_energy = bool(plot_params.get('plot_magnetic_energy', False))
    label_mode = _get_label_mode(plot_params)

    family_order = ('total', 'solenoidal', 'compressive')
    family_suffix = {'total': '', 'solenoidal': '_solenoidal', 'compressive': '_compressive'}
    requested = plot_params.get('velocity_families')
    if requested is None:
        requested = [f for f in family_order if induction_params.get('velocity_field', {}).get(f, f == 'total')]
    if isinstance(requested, str):
        requested = [requested]
    families = [f for f in family_order if f in requested] or ['total']
    plot_split = bool(plot_params.get('plot_split', False))
    family_context = plot_params.get('_family_context')
    if plot_split and len(families) > 1 and family_context is None:
        figures = []
        for family in families:
            params = plot_params.copy()
            params['velocity_families'] = [family]
            params['plot_split'] = False
            params['_family_context'] = family
            figures.extend(plot_production_dissipation_radial_profiles(
                profile_data, params, induction_params, grid_t, grid_zeta, rad,
                verbose=verbose, save=save, folder=folder))
        return figures

    profile_bin_centers = profile_data.get('profile_bin_centers')
    if profile_bin_centers is None:
        raise KeyError('profile_bin_centers not found in profile_data')
    if isinstance(profile_bin_centers, (list, tuple)):
        profile_bin_centers = next((np.asarray(p, dtype=float).ravel() for p in profile_bin_centers
                                    if p is not None and np.asarray(p).size), None)
    else:
        profile_bin_centers = np.asarray(profile_bin_centers, dtype=float).ravel()
    if profile_bin_centers is None or profile_bin_centers.size == 0:
        raise ValueError('profile_bin_centers contains no usable bins')
    nbins = profile_bin_centers.size
    r = profile_bin_centers / float(rad)

    units = plot_params.get('units', induction_params.get('units'))
    if units == energy_to_erg:
        units_y, units_energy, units_density = (energy_to_erg / length_to_mpc**3 / time_to_s,
                                                 energy_to_erg / length_to_mpc**3, density_to_cgs)
    elif units == energy_to_J:
        units_y, units_energy, units_density = (energy_to_J / length_to_mpc**3 / time_to_s,
                                                 energy_to_J / length_to_mpc**3, density_to_sunMpc3)
    else:
        units_y = units_energy = units_density = 1.0

    palette = get_plot_palette(plot_params, induction_params)
    component_colors = palette.get('component_colors', {})
    color_prod = palette.get('production', DEFAULT_PLOT_PALETTE['production'])
    color_diss = palette.get('dissipation', DEFAULT_PLOT_PALETTE['dissipation'])
    color_itemized_net = palette.get('net_itemized', DEFAULT_PLOT_PALETTE['net_itemized'])
    color_compact_net = palette.get('net_compact', DEFAULT_PLOT_PALETTE['net_compact'])
    color_efficiency = palette.get('efficiency', DEFAULT_PLOT_PALETTE['efficiency'])
    color_measured = palette.get('measured_energy', DEFAULT_PLOT_PALETTE['measured_energy'])
    color_density = palette.get('density', DEFAULT_PLOT_PALETTE['density'])
    components_cfg = induction_params.get('components', {})
    family_styles = plot_params.get('velocity_family_styles', {'total': '-', 'solenoidal': '--', 'compressive': ':'})

    def _family_efficiency_color(family):
        if family == 'total':
            return color_efficiency
        rgb = np.asarray(to_rgb(color_efficiency), dtype=float)
        return tuple(np.clip(0.65 * rgb + 0.35, 0.0, 1.0))

    def _family_compact_color(family):
        if family == 'total':
            return color_compact_net
        rgb = np.asarray(to_rgb(color_compact_net), dtype=float)
        return tuple(np.clip(0.65 * rgb + 0.35, 0.0, 1.0))

    component_map = [
        ('MIE_compres_B2', 'Compression', 'compression', 'comp'),
        ('MIE_stretch_B2', 'Stretching', 'stretching', 'str'),
        ('MIE_advec_B2', 'Advection', 'advection', 'adv'),
        ('MIE_diver_B2', 'Divergence', 'divergence', 'div'),
        ('MIE_drag_B2', 'Cosmic Drag', 'drag', 'drag'),
    ]
    def _raw_snapshot(raw, local_i, global_i):
        if raw is None:
            return None
        if np.isscalar(raw):
            return raw
        if isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.size == nbins:
            return raw
        try:
            n = len(raw)
        except TypeError:
            return raw
        if n == 0:
            return None
        source_i = global_i if global_i < n else local_i
        return raw[source_i] if source_i < n else None

    def _align(value, keep_zero=False):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return None
        if arr.size == 1:
            arr = np.full(nbins, float(arr[0]))
        elif arr.size != nbins:
            arr = np.interp(np.linspace(0, 1, nbins), np.linspace(0, 1, arr.size), arr)
        arr[~np.isfinite(arr)] = np.nan
        if not np.any(np.isfinite(arr)):
            return None
        if not keep_zero and not np.any(np.abs(arr[np.isfinite(arr)]) > threshold):
            return None
        return arr

    def _series(key, scale=units_y, keep_zero=False):
        raw = profile_data.get(key)
        result = []
        for local_i, global_i in enumerate(it_indx):
            value = _raw_snapshot(raw, local_i, global_i)
            arr = _align(value, keep_zero=keep_zero)
            result.append(None if arr is None else scale * arr)
        return result

    def _family_key(base, family, metric):
        return f'{base}{family_suffix[family]}_{metric}_profile'

    def _family_series(base, family, metric, keep_zero=False):
        return _series(_family_key(base, family, metric), keep_zero=keep_zero)

    profile_sets = {}
    for family in families:
        profile_sets[family] = {
            'total_prod': _series('MIE_total_B2_prod_itemized_profile') if family == 'total' else [None] * len(it_indx),
            'total_diss': _series('MIE_total_B2_diss_itemized_profile') if family == 'total' else [None] * len(it_indx),
            'total_net': _series('MIE_total_B2_net_itemized_profile') if family == 'total' else [None] * len(it_indx),
            'compact_prod': _series('MIE_total_B2_prod_compact_profile') if family == 'total' else _family_series('MIE_total_B2', family, 'prod'),
            'compact_diss': _series('MIE_total_B2_diss_compact_profile') if family == 'total' else _family_series('MIE_total_B2', family, 'diss'),
            'compact_net': _series('MIE_total_B2_net_compact_profile') if family == 'total' else _family_series('MIE_total_B2', family, 'net'),
            'reconstructed_prod': _series('MIE_total_B2_prod_itemized_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'reconstructed_diss': _series('MIE_total_B2_diss_itemized_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'reconstructed_net': _series('MIE_total_B2_net_itemized_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'reconstructed_compact_prod': _series('MIE_total_B2_prod_compact_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'reconstructed_compact_diss': _series('MIE_total_B2_diss_compact_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'reconstructed_compact_net': _series('MIE_total_B2_net_compact_reconstructed_profile') if family == 'total' else [None] * len(it_indx),
            'components': {},
            'fractional': {},
        }
        for base, _, _, _ in component_map:
            if base == 'MIE_drag_B2':
                use_family = 'total'
            else:
                use_family = family
            profile_sets[family]['components'][base] = {
                'prod': _family_series(base, use_family, 'prod', keep_zero=True),
                'diss': _family_series(base, use_family, 'diss', keep_zero=True),
                'net': _family_series(base, use_family, 'net'),
            }
            profile_sets[family]['fractional'][base] = {
                'prod': _series(f'PD_frac_{base}{family_suffix[use_family]}_prod_profile', scale=1.0),
                'diss': _series(f'PD_frac_{base}{family_suffix[use_family]}_diss_profile', scale=1.0),
            }

    references = {
        'magnetic': _series('clus_b2_profile', scale=units_energy),
        'density': _series('clus_rho_rho_b_profile', scale=units_density),
    }
    split_total_reference = None
    if family_context is not None and family_context != 'total':
        split_total_reference = {
            'compact_prod': _series('MIE_total_B2_prod_compact_profile'),
            'compact_diss': _series('MIE_total_B2_diss_compact_profile'),
            'compact_net': _series('MIE_total_B2_net_compact_profile'),
        }

    if plot_type == 'smoothed':
        if smoothing_sigma <= 0:
            raise ValueError('smoothing_sigma must be positive')
        r_plot = r
        plot_suffix = f'smoothed_sigma_{smoothing_sigma}'
    elif plot_type == 'interpolated':
        if interpolation_points < 2:
            raise ValueError('interpolation_points must be at least 2')
        r_plot = np.linspace(np.min(r), np.max(r), interpolation_points)
        plot_suffix = f'{interpolation_kind}_interpolated_{interpolation_points}_points'
    else:
        r_plot = r
        plot_suffix = 'raw'

    def _transform(arr):
        if arr is None:
            return None
        if plot_type == 'smoothed':
            return gaussian_filter1d(arr, sigma=smoothing_sigma)
        if plot_type == 'interpolated':
            valid = np.isfinite(arr)
            if np.count_nonzero(valid) < 2:
                return None
            kind = interpolation_kind
            if kind == 'cubic' and np.count_nonzero(valid) < 4:
                kind = 'linear'
            return interp1d(r[valid], arr[valid], kind=kind,
                            bounds_error=False, fill_value=np.nan)(r_plot)
        return arr

    def _auto_limits(values, scale):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if scale == 'log':
            values = values[values > 0]
            if values.size == 0:
                return None
            lo, hi = np.log10(values.min()), np.log10(values.max())
            span = max(hi - lo, 1e-6)
            return 10 ** (lo - 0.1 * span), 10 ** (hi + 0.1 * span)
        if values.size == 0:
            return None
        span = max(values.max() - values.min(), 1e-12)
        return values.min() - 0.1 * span, values.max() + 0.1 * span

    def _blend_color(base_color, target, amount):
        base = np.asarray(to_rgb(base_color), dtype=float)
        target_rgb = np.ones(3) if target == 'white' else np.zeros(3)
        return tuple((1.0 - amount) * base + amount * target_rgb)

    def _plot_signed(ax, xvals, values, linestyle, color, label, linewidth, alpha=1.0):
        values = _transform(values)
        if values is None:
            return False, False
        valid = np.isfinite(values)
        if not np.any(valid):
            return False, False
        xvals = np.asarray(xvals)[valid]
        signed = values[valid]
        negative = signed < 0
        yvals = np.maximum(np.abs(signed), threshold)
        positive_values = np.where(~negative, yvals, np.nan)
        if np.any(~negative):
            ax.plot(xvals, yvals, linestyle='-', linewidth=max(linewidth * 0.35, 0.3),
                    color=color, alpha=0.15 * alpha, label='_nolegend_')
            ax.plot(xvals, positive_values, linestyle=linestyle,
                linewidth=linewidth, color=color, alpha=alpha, label='_nolegend_')
        if np.any(negative):
            ax.plot(xvals[negative], yvals[negative], linestyle='None', marker='.',
                    markersize=max(3.0, linewidth * 1.8), color=color, alpha=alpha,
                    label='_nolegend_')
        ax.plot([], [], linestyle=linestyle, linewidth=linewidth, color=color,
                alpha=alpha, label=label)
        return True, bool(np.any(negative))

    def _add_legend(ax, fig, extra_handles=None):
        handles, labels = ax.get_legend_handles_labels()
        if extra_handles:
            handles = list(extra_handles[0]) + handles
            labels = list(extra_handles[1]) + labels
        unique_h, unique_l = [], []
        for handle, label in zip(handles, labels):
            if label and not label.startswith('_') and label not in unique_l:
                unique_h.append(handle)
                unique_l.append(label)
        if not unique_h:
            return
        if fixed_legend:
            ax.legend(unique_h, unique_l, prop=font_legend, loc='lower left',
                      bbox_to_anchor=(0.02, 0.02), bbox_transform=ax.transAxes,
                      ncol=2, frameon=True)
            if not composed_plot:
                fig.tight_layout()
        elif extra_handles:
            ax.legend(unique_h, unique_l, prop=font_legend, ncol=2)
            if not composed_plot:
                fig.tight_layout()
        else:
            ax.legend(unique_h, unique_l, prop=font_legend, ncol=2)
            outside = _smart_legend(ax, fig, plot_params=plot_params, font_legend=font_legend)
            if not composed_plot:
                fig.tight_layout(rect=[0, 0.08, 1, 1] if outside else None)

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 10,
                         'figure.titlesize': 20})
    font = FontProperties(size=12)
    font_title = FontProperties(size=17, weight='bold')
    font_legend = FontProperties(size=12)
    figures = []
    for snap_i, snap_index in enumerate(it_indx):
        negative_found = False
        composed_plot = plot_composed and plot_fractional
        if composed_plot:
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(figsize=figure_size, dpi=dpi)
            grid_spec = GridSpec(2, 1, figure=fig, height_ratios=(3.0, 1.25), hspace=0.08)
            ax = fig.add_subplot(grid_spec[0])
            ax_frac = fig.add_subplot(grid_spec[1], sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
            ax_frac = None
        ax_energy = ax.twinx() if plot_magnetic_energy else None
        ax_density = ax.twinx() if plot_density else None
        if ax_density is not None:
            if ax_energy is not None:
                ax_density.spines['right'].set_position(('axes', aux_density_offset))
            ax_density.set_frame_on(True)
            ax_density.patch.set_visible(False)
        z_value = abs(round(grid_zeta[snap_index], 2)) if snap_index < len(grid_zeta) else float('nan')
        family_label = ''
        ax.set_title(f'{title} - z = {z_value:6.2f}, $R_{{Vir}}$ = {np.round(rad, 1)} Mpc',
                 y=1.0 if composed_plot else 1.05, fontproperties=font_title)
        plotted = []
        fill_regions = []
        energy_reference_values = []
        density_reference_values = []
        families_to_draw = families
        drawn_drag = False
        for family in families_to_draw:
            data = profile_sets[family]
            style = family_styles.get(family, '-')
            suffix_label = '' if len(families) == 1 or family_context is None else f' ({family})'
            if plot_absolute:
                for key, color, label in (
                    ('total_prod', color_prod, f'Total Production (itemized){suffix_label}'),
                    ('total_diss', color_diss, f'Total Dissipation (itemized){suffix_label}')):
                    arr = data[key][snap_i]
                    if arr is not None:
                        is_dissipation = key.endswith('_diss')
                        total_style = style if is_dissipation else '-.'
                        ax.plot(r_plot, _transform(arr), linestyle=total_style,
                            linewidth=line_main, color=color,
                            label=label if len(families) == 1 else '_nolegend_')
                        plotted.append(arr)
                for key, color, label in (
                    ('compact_prod', color_prod, f'Total Production (compact){suffix_label}'),
                    ('compact_diss', color_diss, f'Total Dissipation (compact){suffix_label}')):
                    arr = data[key][snap_i]
                    if arr is not None:
                        is_dissipation = key.endswith('_diss')
                        total_style = style if is_dissipation else '-'
                        ax.plot(r_plot, _transform(arr), linestyle=total_style,
                            linewidth=line_comp, color=color,
                            label=label if len(families) == 1 else '_nolegend_')
                        plotted.append(arr)
                for key, color, label in (
                    ('reconstructed_prod', color_prod, f'Total Production (itemized reconstructed){suffix_label}'),
                    ('reconstructed_diss', color_diss, f'Total Dissipation (itemized reconstructed){suffix_label}')):
                    arr = data[key][snap_i]
                    if arr is not None:
                        is_dissipation = key.endswith('_diss')
                        total_style = style if is_dissipation else '--'
                        ax.plot(r_plot, _transform(arr), linestyle=total_style,
                            linewidth=line_comp, color=color, alpha=0.7,
                            label=label if len(families) == 1 else '_nolegend_')
                        plotted.append(arr)
                for key, color, label in (
                    ('reconstructed_compact_prod', color_prod, f'Total Production (compact reconstructed){suffix_label}'),
                    ('reconstructed_compact_diss', color_diss, f'Total Dissipation (compact reconstructed){suffix_label}')):
                    arr = data[key][snap_i]
                    if arr is not None:
                        is_dissipation = key.endswith('_diss')
                        total_style = style if is_dissipation else '-'
                        ax.plot(r_plot, _transform(arr), linestyle=total_style,
                            linewidth=line_comp, color=color, alpha=0.7,
                            label=label if len(families) == 1 else '_nolegend_')
                        plotted.append(arr)
            for base, label, color_key, sym in component_map:
                if not components_cfg.get(color_key, True):
                    continue
                if base == 'MIE_drag_B2' and drawn_drag:
                    continue
                comp = data['components'][base]
                arr_p, arr_d = comp['prod'][snap_i], comp['diss'][snap_i]
                if arr_p is not None and arr_d is not None:
                    p_plot = _transform(arr_p)
                    d_plot = _transform(arr_d)
                    fill_color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    valid_fill = np.isfinite(p_plot) & np.isfinite(d_plot)
                    if np.any(valid_fill):
                        fill_regions.append((p_plot, d_plot, valid_fill, fill_color))
                if arr_p is not None and should_plot_component(arr_p, threshold=threshold):
                    component_label = ('_nolegend_' if composed_plot else f'{label} $P_{{\\mathrm{{{sym}}}}}${suffix_label}') if len(families) == 1 else '_nolegend_'
                    ax.plot(r_plot, _transform(arr_p), linestyle='--' if len(families) == 1 else style, linewidth=line_comp, color=component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key]), alpha=0.75, label=component_label)
                    plotted.append(arr_p)
                if arr_d is not None and should_plot_component(arr_d, threshold=threshold):
                    diss_style = ':' if family_context is not None else style
                    component_label = ('_nolegend_' if composed_plot else f'{label} $D_{{\\mathrm{{{sym}}}}}${suffix_label}') if len(families) == 1 else '_nolegend_'
                    ax.plot(r_plot, _transform(arr_d), linestyle=diss_style, linewidth=line_comp, color=component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key]), alpha=0.48, label=component_label)
                    plotted.append(arr_d)
                if base == 'MIE_drag_B2':
                    drawn_drag = True
            if plot_net:
                if split_total_reference is not None:
                    total_reference_net = split_total_reference['compact_net'][snap_i]
                    if total_reference_net is not None:
                        _plot_signed(
                            ax, r_plot, total_reference_net, '-', color_compact_net,
                            'Net total (compact)', line_main)
                for key, color, label in (
                    ('compact_net', color_compact_net, f'Net total (compact){suffix_label}'),
                    ('total_net', color_itemized_net, f'Net total (itemized){suffix_label}'),
                    ('reconstructed_net', color_itemized_net, f'Net total (itemized reconstructed){suffix_label}'),
                    ('reconstructed_compact_net', color_compact_net, f'Net total (compact reconstructed){suffix_label}')):
                    if key in ('reconstructed_net', 'reconstructed_compact_net') and not plot_reconstructed_net:
                        continue
                    arr = data[key][snap_i]
                    if arr is not None:
                        if key == 'compact_net' and family_context is not None and family_context != 'total':
                            net_label = 'Net (compact)'
                        else:
                            net_label = label if len(families) == 1 else '_nolegend_'
                        reconstructed = key in ('reconstructed_net', 'reconstructed_compact_net')
                        if len(families) > 1:
                            net_style = style
                        elif plot_reconstructed_net:
                            net_style = '--' if reconstructed else '-'
                        else:
                            net_style = '-' if key == 'compact_net' else '--'
                        net_color = _family_compact_color(family) if key in ('compact_net', 'reconstructed_compact_net') else color
                        _, has_negative = _plot_signed(ax, r_plot, arr, net_style, net_color, net_label, line_main)
                        negative_found |= has_negative
                        plotted.append(np.abs(arr))
        if plot_magnetic_energy and references['magnetic'][snap_i] is not None:
            magnetic_reference = _transform(references['magnetic'][snap_i])
            ax_energy.plot(r_plot, magnetic_reference, color=color_measured, linewidth=line_main, label='Magnetic Energy Density')
            energy_reference_values.append(magnetic_reference)
        if plot_density and references['density'][snap_i] is not None:
            density_reference = _transform(references['density'][snap_i])
            ax_density.plot(r_plot, density_reference, color=color_density, linewidth=line_main, label='Density')
            density_reference_values.append(density_reference)
        if x_scale == 'log':
            ax.set_xscale('log')
        radial_xlabel = (r'$\log_{10}(r/R_{\mathrm{Vir}})$' if x_scale == 'log' else r'$r/R_{\mathrm{Vir}}$') if label_mode == 'math' else ('Radial Distance log[r/$R_{Vir}$]' if x_scale == 'log' else 'Radial Distance [r/$R_{Vir}$]')
        ax.set_xlabel('' if composed_plot else radial_xlabel, fontproperties=font)
        if xlim is not None:
            ax.set_xlim(xlim[0], xlim[1])
        if y_scale == 'log':
            ax.set_yscale('log')
            if ax_energy is not None:
                ax_energy.set_yscale('log')
            if ax_density is not None:
                ax_density.set_yscale('log')
        if ax_energy is not None:
            if rylim is not None:
                ax_energy.set_ylim(*rylim)
            elif energy_reference_values:
                energy_limits = _auto_limits(np.concatenate(energy_reference_values), y_scale)
                if energy_limits is not None:
                    ax_energy.set_ylim(*energy_limits)
        if ax_density is not None:
            if dylim is not None:
                ax_density.set_ylim(*dylim)
            elif density_reference_values:
                density_limits = _auto_limits(np.concatenate(density_reference_values), y_scale)
                if density_limits is not None:
                    ax_density.set_ylim(*density_limits)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        elif plotted:
            limits = _auto_limits(np.concatenate([np.asarray(p).ravel() for p in plotted]), y_scale)
            if limits is not None:
                ax.set_ylim(*limits)
        if fill_regions:
            y_lower, _ = ax.get_ylim()
            area_alpha = float(np.clip(plot_params.get('area_alpha', 0.24), 0.0, 1.0))
            for p_plot, d_plot, valid_fill, fill_color in fill_regions:
                p_visible = np.asarray(p_plot, dtype=float).copy()
                d_visible = np.asarray(d_plot, dtype=float).copy()
                valid_visible = valid_fill & np.isfinite(p_visible) & np.isfinite(d_visible)
                if y_scale == 'log':
                    floor = max(y_lower, np.finfo(float).tiny)
                    p_visible[p_visible <= 0.0] = floor
                    d_visible[d_visible <= 0.0] = floor
                d_dominates = valid_visible & (d_visible > p_visible)
                ax.fill_between(
                    r_plot, p_visible, d_visible, where=valid_visible,
                    interpolate=True,
                    color=_blend_color(fill_color, 'white', 0.45),
                    alpha=area_alpha, linewidth=0.0, label='_nolegend_', zorder=1)
                ax.fill_between(
                    r_plot, p_visible, d_visible, where=d_dominates,
                    interpolate=True,
                    color=_blend_color(fill_color, 'black', 0.22),
                    alpha=area_alpha, linewidth=0.0, label='_nolegend_', zorder=1)
        if ax_energy is not None:
            if label_mode == 'math':
                if units == energy_to_erg:
                    energy_label = r'$\rho_{B}\ (\mathrm{erg}\,\mathrm{Mpc}^{-3})$'
                elif units == energy_to_J:
                    energy_label = r'$\rho_{B}\ (\mathrm{J}\,\mathrm{Mpc}^{-3})$'
                else:
                    energy_label = r'$\rho_{B}\ (\mathrm{arb.\ units})$'
            else:
                energy_label = ('Magnetic Energy Density (erg/$Mpc^{3}$)' if units == energy_to_erg
                                else 'Magnetic Energy Density (J/$Mpc^{3}$)' if units == energy_to_J
                                else 'Magnetic Energy Density (arb. units)')
            ax_energy.set_ylabel(energy_label, fontproperties=font, color=color_measured)
            ax_energy.tick_params(axis='y', colors=color_measured)
        if ax_density is not None:
            if label_mode == 'math':
                if units == energy_to_erg:
                    density_label = r'$\rho\ (\mathrm{g}\,\mathrm{cm}^{-3})$'
                elif units == energy_to_J:
                    density_label = r'$\rho\ (M_{\odot}\,\mathrm{Mpc}^{-3})$'
                else:
                    density_label = r'$\rho\ (\mathrm{arb.\ units})$'
            else:
                density_label = ('Density (g/cm$^{3}$)' if units == energy_to_erg
                                 else 'Density (M$_{\odot}$/Mpc$^{3}$)' if units == energy_to_J
                                 else 'Density (arb. units)')
            ax_density.set_ylabel(density_label, fontproperties=font, color=color_density)
            ax_density.tick_params(axis='y', colors=color_density)
        ylabel = 'Production / Dissipation (erg/$Mpc^{3}$/s)' if units == energy_to_erg else ('Production / Dissipation (J/$Mpc^{3}$/s)' if units == energy_to_J else 'Production / Dissipation (arb. units)')
        ax.set_ylabel(ylabel, fontproperties=font)
        if family_context is not None:
            if family_context != 'total':
                family_ylabel = f'{ylabel}\n- {family_context.title()} Velocity Field' if composed_plot else f'{ylabel} - {family_context.title()} Velocity Field'
                ax.set_ylabel(family_ylabel, fontproperties=font)
        ax.grid(alpha=0.3)
        if ax_energy is not None:
            ax_energy.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax_energy.tick_params(axis='y', labelsize=aux_tick_labelsize)
        if ax_density is not None:
            ax_density.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax_density.tick_params(axis='y', labelsize=aux_tick_labelsize)
        from matplotlib.lines import Line2D
        extras = None
        if negative_found:
            extras = ([Line2D([0], [0], color='0.25', marker='.', linestyle='None', label='Negative Interval')], ['Negative Interval'])
        if len(families) > 1:
            fam_handles = [Line2D([0], [0], color='0.25', linestyle=family_styles.get(f, '-'), label=f'Velocity family: {f}') for f in families]
            fam_labels = [h.get_label() for h in fam_handles]
            if extras:
                fam_handles.extend(extras[0]); fam_labels.extend(extras[1])
            extras = (fam_handles, fam_labels)
            if plot_absolute:
                absolute_handles = list(extras[0]) if extras else []
                absolute_labels = list(extras[1]) if extras else []
                for key, color, linestyle, label in (
                    ('compact_prod', color_prod, '-', 'Total Production (compact)'),
                    ('compact_diss', color_diss, '-', 'Total Dissipation (compact)'),
                ):
                    if any(profile_sets[f][key][snap_i] is not None for f in families):
                        absolute_handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                                                        linewidth=line_comp, label=label))
                        absolute_labels.append(label)
                for base, label, color_key, sym in component_map:
                    if not components_cfg.get(color_key, True):
                        continue
                    color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    has_prod = any(profile_sets[f]['components'][base]['prod'][snap_i] is not None for f in families)
                    has_diss = any(profile_sets[f]['components'][base]['diss'][snap_i] is not None for f in families)
                    if has_prod:
                        absolute_handles.append(Line2D([0], [0], color=color, linestyle='-',
                                                        linewidth=line_comp, label=f'{label} production'))
                        absolute_labels.append(f'{label} production')
                    if has_diss:
                        absolute_handles.append(Line2D([0], [0], color=color, linestyle='-',
                                                        linewidth=line_comp, alpha=0.48, label=f'{label} dissipation'))
                        absolute_labels.append(f'{label} dissipation')
                if plot_net:
                    for key, color, label in (
                        ('compact_net', color_compact_net, 'Net total (compact)'),
                        ('total_net', color_itemized_net, 'Net total (itemized)'),
                    ):
                        if any(profile_sets[f][key][snap_i] is not None for f in families):
                            absolute_handles.append(Line2D([0], [0], color=color, linestyle='-',
                                                            linewidth=line_main, label=label))
                            absolute_labels.append(label)
                extras = (absolute_handles, absolute_labels)
            else:
                profile_handles = list(extras[0]) if extras else []
                profile_labels = list(extras[1]) if extras else []
                for base, label, color_key, sym in component_map:
                    if not components_cfg.get(color_key, True):
                        continue
                    color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    has_prod = any(profile_sets[f]['components'][base]['prod'][snap_i] is not None for f in families)
                    has_diss = any(profile_sets[f]['components'][base]['diss'][snap_i] is not None for f in families)
                    if has_prod:
                        profile_handles.append(Line2D([0], [0], color=color, linestyle='-',
                                                       linewidth=line_comp, label=rf'{label} $P_{{\mathrm{{{sym}}}}}$'))
                        profile_labels.append(rf'{label} $P_{{\mathrm{{{sym}}}}}$')
                    if has_diss:
                        profile_handles.append(Line2D([0], [0], color=color, linestyle='-',
                                                       linewidth=line_comp, alpha=0.48,
                                                       label=rf'{label} $D_{{\mathrm{{{sym}}}}}$'))
                        profile_labels.append(rf'{label} $D_{{\mathrm{{{sym}}}}}$')
                extras = (profile_handles, profile_labels)
        _add_legend(ax, fig, extras)
        if not composed_plot:
            figures.append(fig)

        if plot_fractional:
            if not composed_plot:
                fig_frac, ax_frac = plt.subplots(figsize=figure_size, dpi=dpi)
            else:
                fig_frac = fig
                ax_frac.tick_params(axis='x', labelbottom=True)
                ax.tick_params(axis='x', labelbottom=False)
            if not composed_plot:
                ax_frac.set_title(
                    f'{title} (Fractions){family_label} - z = {z_value:6.2f}, '
                    f'$R_{{Vir}}$ = {np.round(rad, 1)} Mpc',
                    y=1.05, fontproperties=font_title)
            for family in families_to_draw:
                data = profile_sets[family]
                style = family_styles.get(family, '-')
                suffix_label = '' if len(families) == 1 or family_context is None else f' ({family})'
                for base, label, color_key, sym in component_map:
                    if not components_cfg.get(color_key, True) or (base == 'MIE_drag_B2' and family != 'total'):
                        continue
                    frac = data['fractional'][base]
                    color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    prod_frac = frac['prod'][snap_i]
                    diss_frac = frac['diss'][snap_i]
                    if prod_frac is not None:
                        fraction_label = ('_nolegend_' if composed_plot else f'{label} $p_{{\\mathrm{{{sym}}}}}${suffix_label}') if len(families) == 1 else '_nolegend_'
                        ax_frac.plot(r_plot, _transform(prod_frac), linestyle='--' if len(families) == 1 else style, linewidth=line_comp, color=color, label=fraction_label)
                    if diss_frac is not None:
                        fraction_label = ('_nolegend_' if composed_plot else f'{label} $d_{{\\mathrm{{{sym}}}}}${suffix_label}') if len(families) == 1 else '_nolegend_'
                        diss_style = ':' if len(families) == 1 else style
                        ax_frac.plot(r_plot, -_transform(diss_frac), linestyle=diss_style, linewidth=line_comp, alpha=0.48, color=color, label=fraction_label)
            efficiency_curves = []
            if split_total_reference is not None:
                total_prod_ref = split_total_reference['compact_prod'][snap_i]
                total_diss_ref = split_total_reference['compact_diss'][snap_i]
                if total_prod_ref is not None and total_diss_ref is not None:
                    total_iota = np.divide(
                        total_prod_ref - total_diss_ref,
                        total_prod_ref,
                        out=np.zeros_like(total_prod_ref),
                        where=total_prod_ref > 0,
                    )
                    ax_frac.plot(
                        r_plot, _transform(total_iota), color=color_efficiency,
                        linewidth=line_main, linestyle='-',
                        label=r'Net Efficiency $\iota$ (total)',
                    )
            for family in families_to_draw:
                data = profile_sets[family]
                total_prod = data['compact_prod'][snap_i]
                total_diss = data['compact_diss'][snap_i]
                if total_prod is None or total_diss is None:
                    continue
                iota = np.divide(total_prod - total_diss, total_prod, out=np.zeros_like(total_prod), where=total_prod > 0)
                efficiency_color = _family_efficiency_color(family)
                efficiency_style = family_styles.get(family, '-') if len(families) > 1 else '-'
                efficiency_label = r'Net Efficiency $\iota$' if len(families) == 1 else '_nolegend_'
                ax_frac.plot(r_plot, _transform(iota), color=efficiency_color, linewidth=line_main,
                             linestyle=efficiency_style, label=efficiency_label)
                efficiency_curves.append((family, efficiency_color, efficiency_style))
            ax_frac.set_xlabel(radial_xlabel, fontproperties=font)
            if x_scale == 'log':
                ax_frac.set_xscale('log')
            if xlim is not None:
                ax_frac.set_xlim(xlim[0], xlim[1])
            ax_frac.set_ylim(-1.05, 1.05)
            fractional_ylabel = 'Fractional Contribution (-diss / +prod)'
            if family_context is not None and family_context != 'total':
                fractional_ylabel = f'{fractional_ylabel} - {family_context.title()} Velocity Field'
            if composed_plot:
                fractional_ylabel = 'Fractional Contributions\n(-diss / +prod)'
            ax_frac.set_ylabel(fractional_ylabel, fontproperties=font)
            ax_frac.grid(alpha=0.3)
            fraction_extras = None
            if len(families) > 1:
                from matplotlib.lines import Line2D
                fraction_handles = [Line2D([0], [0], color='0.25', linestyle=family_styles.get(f, '-'), label=f'Velocity family: {f}') for f in families]
                fraction_labels = [f'Velocity family: {f}' for f in families]
                if efficiency_curves:
                    fraction_handles.append(Line2D([0], [0], color=color_efficiency, linestyle='-', linewidth=line_main, label=r'Net Efficiency $\iota$'))
                    fraction_labels.append(r'Net Efficiency $\iota$')
                for base, label, color_key, sym in component_map:
                    if not components_cfg.get(color_key, True):
                        continue
                    color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    if any(profile_sets[f]['fractional'][base]['prod'][snap_i] is not None for f in families):
                        fraction_handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=line_comp, label=rf'{label} $p_{{\mathrm{{{sym}}}}}$'))
                        fraction_labels.append(rf'{label} $p_{{\mathrm{{{sym}}}}}$')
                    if any(profile_sets[f]['fractional'][base]['diss'][snap_i] is not None for f in families):
                        fraction_handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=line_comp, alpha=0.48, label=f'{label} $d_{{\\mathrm{{{sym}}}}}$'))
                        fraction_labels.append(rf'{label} $d_{{\mathrm{{{sym}}}}}$')
                fraction_extras = (fraction_handles, fraction_labels)
            if composed_plot:
                from matplotlib.lines import Line2D
                fraction_handles = []
                fraction_labels = []
                if len(families) > 1:
                    fraction_handles.extend(
                        Line2D([0], [0], color='0.25', linestyle=family_styles.get(f, '-'),
                               label=f'Velocity family: {f}')
                        for f in families
                    )
                    fraction_labels.extend(f'Velocity family: {f}' for f in families)
                if negative_found:
                    fraction_handles.append(Line2D(
                        [0], [0], color='0.25', marker='.', linestyle='None',
                        label='Negative Interval'))
                    fraction_labels.append('Negative Interval')
                if plot_net:
                    net_handles = []
                    net_labels = []
                    if split_total_reference is not None and split_total_reference['compact_net'][snap_i] is not None:
                        net_handles.append(Line2D(
                            [0], [0], color=color_compact_net, linestyle='-',
                            linewidth=line_main, label='Net total (compact)'))
                        net_labels.append('Net total (compact)')
                    for key, color, label, linestyle in (
                        ('compact_net', color_compact_net, 'Net total (compact)', '-'),
                        ('total_net', color_itemized_net, 'Net total (itemized)', '-'),
                        ('reconstructed_compact_net', color_compact_net, 'Net total (compact reconstructed)', '--'),
                        ('reconstructed_net', color_itemized_net, 'Net total (itemized reconstructed)', '--'),
                    ):
                        if key.startswith('reconstructed') and not plot_reconstructed_net:
                            continue
                        if any(profile_sets[f][key][snap_i] is not None for f in families):
                            if key == 'compact_net' and family_context is not None and family_context != 'total':
                                label = 'Net (compact)'
                            net_handles.append(Line2D([0], [0], color=color, linestyle=linestyle,
                                                      linewidth=line_main, label=label))
                            net_labels.append(label)
                    insert_at = 1 if fraction_labels and fraction_labels[0] == 'Negative Interval' else 0
                    fraction_handles[insert_at:insert_at] = net_handles
                    fraction_labels[insert_at:insert_at] = net_labels
                if efficiency_curves:
                    fraction_handles.append(Line2D(
                        [0], [0], color=color_efficiency, linestyle='-',
                        linewidth=line_main, label=r'Net Efficiency $\iota$'))
                    fraction_labels.append(r'Net Efficiency $\iota$')
                for base, label, color_key, sym in component_map:
                    if not components_cfg.get(color_key, True):
                        continue
                    color = component_colors.get(color_key, DEFAULT_PLOT_PALETTE['component_colors'][color_key])
                    has_prod = any(profile_sets[f]['fractional'][base]['prod'][snap_i] is not None for f in families)
                    has_diss = any(profile_sets[f]['fractional'][base]['diss'][snap_i] is not None for f in families)
                    if has_prod:
                        fraction_handles.append(Line2D(
                            [0], [0], color=color, linestyle='-', linewidth=line_comp,
                            label=rf'{label} $P_{{\mathrm{{{sym}}}}},p_{{\mathrm{{{sym}}}}}$'))
                        fraction_labels.append(rf'{label} $P_{{\mathrm{{{sym}}}}},p_{{\mathrm{{{sym}}}}}$')
                    if has_diss:
                        fraction_handles.append(Line2D(
                            [0], [0], color=color, linestyle='-', linewidth=line_comp,
                            alpha=0.48, label=rf'{label} $D_{{\mathrm{{{sym}}}}},d_{{\mathrm{{{sym}}}}}$'))
                        fraction_labels.append(rf'{label} $D_{{\mathrm{{{sym}}}}},d_{{\mathrm{{{sym}}}}}$')
                _add_legend(ax, fig, (fraction_handles, fraction_labels))
                if ax_frac.legend_ is not None:
                    ax_frac.legend_.remove()
                fig_frac.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.92, hspace=0.08)
                figures.append(fig_frac)
            else:
                _add_legend(ax_frac, fig_frac, fraction_extras)
                figures.append(fig_frac)

    if verbose:
        print('Plotting... Production/dissipation radial profile plots created')

    if save and figures:
        if folder is None:
            folder = os.getcwd()
        family_info = f'_family_{family_context}' if family_context else (('_vf_' + '-'.join(families)) if len(families) > 1 else '')
        sim_info = f'{induction_params.get("up_to_level", "")}_{induction_params.get("F", 1.0)}_{induction_params.get("vir_kind", "")}vir_{induction_params.get("rad_kind", "")}rad_{induction_params.get("region", None)}Region'
        diff_cfg = induction_params.get('differentiation', {})
        buffer_info = f'Buffered_{diff_cfg.get("interpol", "")}_siblings_{diff_cfg.get("use_siblings", "")}' if diff_cfg.get('buffer', False) else 'NoBuffer'
        base = '_'.join(title.split()[:3])
        for index, fig in enumerate(figures):
            filename = f'{folder}/{run}_{base}{family_info}_pd_profile_{sim_info}_{x_scale}_{y_scale}_{buffer_info}_{diff_cfg.get("stencil", "")}_{plot_suffix}_{index}.png'
            filename = safe_filename(filename, verbose=verbose)
            fig.savefig(filename, dpi=dpi)
            if verbose:
                print(f'Saved figure {index + 1}/{len(figures)}: {filename}')
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