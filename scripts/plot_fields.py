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

def plot_seed_spectrum(alpha_index, Bx, By, Bz, dx, mode = 1, epsilon = 1e-30, ncores=1, verbose = True, Save = False, DPI = 300, run = '_', folder = None):
    '''
    Plots the power spectrum of the magnetic field seed and other interesting quantities to check the generation.
    
    Args:
        - alpha_index: spectral index of the magnetic field
        - Bx: magnetic field component in Fourier space in the x direction
        - By: magnetic field component in Fourier space in the y direction
        - Bz: magnetic field component in Fourier space in the z direction
        - dx: cell size in Mpc
        - mode: integer to choose the plot mode
        - epsilon: small number to avoid division by zero
        - ncores: number of cores to use for the FFT
        - verbose: boolean to print the progress of the function
        - Save: boolean to save the plot or not
        - DPI: dots per inch in the plot
        - run: string to identify the run
        - folder: folder to save the plot
        
    Returns:
        - Plot of the magnetic field seed power spectrum and other comparatives
        
    Author: Marco Molina
    '''
    nmax, nmay, nmaz = Bx[0].shape
    
    klim = 1 # Set axis limits in the power spectrum plot (klim=1 and k0=0 for showing all, ko defines the starting point of the trend line)
    k0 = 0
    
    # Compute the magnetic field magnitude
    Bmag = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
    
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Magnitude computed')
    
    # Compute the magnetic power spectrum
    k_bins, P_k = spectral.power_spectrum_vector_field(Bx[0], By[0], Bz[0], dx=dx)
    P_k = np.where(P_k == 0, epsilon, P_k)
    k_bins = k_bins * 2 * np.pi
    
    # Compute the integral of the power spectrum
    integral_Pk = np.trapz(P_k, k_bins)
    
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Power Spectrum computed')
    
    ## Calculate the standard deviation of the power spectrum for each bin ##
    
    # First we need to compute the FFT, its amplitude square, and normalise it
    # Bx_fourier_amplitudes = fft.fftn(Bx[0], s=Bx[0].shape, workers=ncores) / np.sqrt(nmax*nmay*nmaz)
    # By_fourier_amplitudes = fft.fftn(By[0], s=By[0].shape, workers=ncores) / np.sqrt(nmax*nmay*nmaz)
    # Bz_fourier_amplitudes = fft.fftn(Bz[0], s=Bz[0].shape, workers=ncores) / np.sqrt(nmax*nmay*nmaz)
    Bx_fourier_amplitudes = fft.fftn(Bx[0], s=Bx[0].shape, norm="ortho", workers=ncores)
    By_fourier_amplitudes = fft.fftn(By[0], s=By[0].shape, norm="ortho", workers=ncores)
    Bz_fourier_amplitudes = fft.fftn(Bz[0], s=Bz[0].shape, norm="ortho", workers=ncores)

    Bx_fourier_amplitudes = (np.abs(Bx_fourier_amplitudes)**2).flatten()
    By_fourier_amplitudes = (np.abs(By_fourier_amplitudes)**2).flatten()
    Bz_fourier_amplitudes = (np.abs(Bz_fourier_amplitudes)**2).flatten()

    # Next we obtain the frequencies
    kx = np.fft.fftfreq(nmax, dx) * 2 * np.pi # Es seguro el factor 2*pi?
    ky = np.fft.fftfreq(nmay, dx) * 2 * np.pi
    kz = np.fft.fftfreq(nmaz, dx) * 2 * np.pi
    
    k_magnitude = np.meshgrid(kx, ky, kz, indexing='ij')
    k_magnitude = k_magnitude[0]**2 + k_magnitude[1]**2 + k_magnitude[2]**2
    k_magnitude = np.sqrt(k_magnitude).flatten()

    # Assuming isotropy, we can obtain the P(k) (1-dimensional power spectrum) standard deviation
    k_stats_bins = np.arange(kx[1]/2, np.abs(kx).max(), kx[1])

    P_k_x_std, _, _ = stats.binned_statistic(k_magnitude, Bx_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    P_k_y_std, _, _ = stats.binned_statistic(k_magnitude, By_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    P_k_z_std, _, _ = stats.binned_statistic(k_magnitude, Bz_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    
    del Bx_fourier_amplitudes, By_fourier_amplitudes, Bz_fourier_amplitudes, k_magnitude
    gc.collect()
    
    P_k_std = P_k_x_std + P_k_y_std + P_k_z_std
    
    del P_k_x_std, P_k_y_std, P_k_z_std
    gc.collect()
    
    per = 100
    while np.any(P_k[k0:-klim] - P_k_std[k0:-klim] <= 0):
        P_k_std = P_k_std * per/100
        per = per - 5
        if per <= 0:
            print("Warning: Unable to resolve condition within 100 iterations, standard deviation set to zero.")
            P_k_std = np.zeros_like(P_k_std)
            break
        
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Power Spectrum Standard Deviation computed')
    
    # Computes the magnetic field seed divergence
    diver_B = diff.periodic_divergence(Bx, By, Bz, dx, npatch = np.array([0]), stencil=5, kept_patches=None)
    
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Divergence computed')

    # Calculate the expected trend lines starting from specifict values of P_k
    ko = 3
    trend_line = P_k[k0+ko] * (k_bins[k0+ko:-klim] / k_bins[k0+ko])**(alpha_index)
        
    # Plot the magnetic power spectrum
    if mode == 0:
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 6), dpi =DPI)
        ax = [ax]
    
    # Spectrum with the Cumulative Relative Divergence of the Magnetic Field
    elif mode == 1:
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(16, 6), dpi=DPI)
        ax = ax.flatten()

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        nmax, nmay, nmaz = Bx[0].shape
        
        xxx = np.linspace(0, 100, 1001)
        if nmax >= 256 or nmay >= 256 or nmaz >= 256:
            sub_sample = int(0.1 * 256**3)
        else:
            sub_sample = int(0.2 * nmax * nmay * nmaz)
            
        # Consider only the central part of the box
        # random_inndex = np.random.choice((nmax-(nmax//2))*(nmay-(nmay//2))*(nmaz-(nmaz//2)), size=int(sub_sample), replace=False)
        # random_cells = np.abs(diver_B[0][nmax//4:-nmax//4,nmay//4:-nmay//4,nmaz//4:-nmaz//4].flatten()[random_inndex])
        
        # Consider the whole box
        random_inndex = np.random.choice(nmax*nmay*nmaz, size=int(sub_sample), replace=False)
        random_cells = np.abs(diver_B[0].flatten()[random_inndex])
        
        yyy = [np.percentile(random_cells, p) for p in xxx]

        ax[1].clear()
        ax[1].set_title(f'Cumulative Relative Divergence of the Magnetic Field, $N_c$ = {Bx[0].shape[0]}')
        ax[1].set_xlabel('Percentage of cells')
        ax[1].set_ylabel('|∇·B|/B')
        ax[1].plot(xxx, yyy / np.mean(Bx[0]**2 + By[0]**2 + Bz[0]**2)**0.5 * dx)
        ax[1].set_xscale('linear')
        ax[1].set_yscale('log')
        ax[1].grid()
        
        if verbose == True:
            print(f'Plotting... Magnetic Field Seed Cumulative Relative Divergence plotted')
    
    # Spectrum with the Cumulative Absolute Divergence of the Magnetic Field
    elif mode == 2:
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(16, 6), dpi=DPI)
        ax = ax.flatten()

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        nmax, nmay, nmaz = Bx[0].shape
        
        xxx = np.linspace(0, 100, 1001)
        if nmax >= 256 or nmay >= 256 or nmaz >= 256:
            sub_sample = int(0.1 * 256**3)
        else:
            sub_sample = int(0.2 * nmax * nmay * nmaz)
            
        # Consider only the central part of the box
        # random_inndex = np.random.choice((nmax-(nmax//2))*(nmay-(nmay//2))*(nmaz-(nmaz//2)), size=int(sub_sample), replace=False)
        # random_cells = np.abs(diver_B[0][nmax//4:-nmax//4,nmay//4:-nmay//4,nmaz//4:-nmaz//4].flatten()[random_inndex])
        
        # Consider the whole box
        random_inndex = np.random.choice(nmax*nmay*nmaz, size=int(sub_sample), replace=False)
        random_cells = np.abs(diver_B[0].flatten()[random_inndex])
        
        yyy = [np.percentile(random_cells, p) for p in xxx]

        ax[1].clear()
        ax[1].set_title(f'Cumulative Absolute Divergence of the Magnetic Field, $N_c$ = {Bx[0].shape[0]}')
        ax[1].set_xlabel('Percentage of cells')
        ax[1].set_ylabel('|∇·B|')
        ax[1].plot(xxx, yyy)
        ax[1].set_xscale('linear')
        ax[1].set_yscale('log')
        ax[1].grid()
        
        if verbose == True:
            print(f'Plotting... Magnetic Field Seed Cumulative Absolute Divergence plotted')
        
    # Spectrum with the Histograms of the Divergence and the Magnetic Field
    elif mode == 3:
        
        fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize=(16, 6), dpi=DPI)
        ax = ax.flatten()

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Histogram of the values of the divergence
        data = np.abs(diver_B[0].flatten())
        data[data == 0] = epsilon
        min_val = data[data > 0].min()
        max_val = data.max()
        log_bins = np.logspace(np.log10(min_val).real, np.log10(max_val).real, num=5000)
        
        ax[2].clear()  
        ax[2].set_xlabel('|∇·B|')
        ax[2].set_ylabel('Frequency')
        ax[2].set_title(f'Histogram of Divergence, $N_c$ = {Bx[0].shape[0]}')
        ax[2].hist(data, bins=log_bins, alpha=0.75, label='Periodic Divergence', log=True)
        ax[2].legend()
        ax[2].set_xscale('log')
        # ax[2].set_xlim(1e-40, 5e-1)
        # ax[2].set_ylim(1e-2, 1e6)
        ax[2].legend(loc='upper right')

        # Histogram of the values of the magnetic field
        data = Bmag[0].flatten()
        min_val = data[data > 0].min()
        max_val = data.max()
        log_bins = np.logspace(np.log10(min_val).real, np.log10(max_val).real, num=5000)
        
        ax[1].clear() 
        ax[1].set_xlabel('Gauss (G)')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title(f'Histogram of Magentic Field Intensity, $N_c$ = {Bx[0].shape[0]}')
        ax[1].hist(data, bins=log_bins, alpha=1, label='Magentic Field', log=True)
        ax[1].legend()
        ax[1].set_xscale('log')
        # ax[1].set_xlim(1e-40, 5e-1)
        # ax[1].set_ylim(1e-2, 1e6)
        ax[1].legend(loc='upper right')
        
        if verbose == True:
            print(f'Plotting... Magnetic Field Seed Histograms plotted')

    ax[0].clear()  
    ax[0].set_title(f'Random Transverse Projected Magnetic Field Power Spectrum')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('P(k)')
    ax[0].loglog(
        k_bins[k0:-klim],
        P_k[k0:-klim],
        label=f'Magnetic Power Spectrum\n$\int P(k) dk$ = {integral_Pk:.2e}'
    )
    ax[0].fill_between(k_bins[k0:-klim], P_k[k0:-klim] - P_k_std[k0:-klim], P_k[k0:-klim] + P_k_std[k0:-klim], color='gray', alpha=0.25)
    ax[0].loglog(k_bins[k0+ko:-klim], trend_line, linestyle='dotted', label=f'$k^{{{np.round(alpha_index,2)}}}$')
    ax[0].legend()
    # ax[0].set_xlim(0, k_bins[-klim] + 1)
    # ax[0].set_ylim(1e4, 1e12)
    ax[0].legend(loc='lower center')

    # Annotate the confidence percentage near the error area
    ax[0].text(
        k_bins[k0:-klim][0] + ((k_bins[k0:-klim][1] - k_bins[k0:-klim][0]) / 4),  # x-coordinate
        P_k[k0:-klim][1] - P_k_std[k0:-klim][1],  # y-coordinate (just above the error area)
        f"{per}% $\\sigma$",  # Annotation text
        fontsize=10,  # Font size
        color="gray",  # Text color
        ha="center",  # Horizontal alignment
        va="bottom",  # Vertical alignment
        # bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")  # Background box for better visibility
    )
    
    if verbose == True:
        print(f'Plotting... Magnetic Field Seed Power Spectrum plotted')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
            
        fig.savefig(folder + f'/Mag_Field_Power_Spectrum_{run}.png')
        
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
        ani.save(folder + f'/{file_title}_{run}_zoom.gif', writer='pillow', dpi = DPI)
        
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
        ani.save(folder + f'/{file_title}_{run}_scan.gif', writer='pillow', dpi = DPI)
        
    return ani
        
        
def plot_integral_evolution(evolution_data, plot_params, induction_params,
                            grid_t, grid_zeta, rad,
                            verbose=True, save=False, folder=None):
    """
    Plot the evolution of the integrated magnetic energy and its induction components.
    
    Args:
        - evolution_data: dictionary containing the evolution data from induction_energy_integral_evolution()
        - plot_params: dictionary containing plotting parameters:
            - evolution_type: 'total' or 'differential'
            - derivative: 'central' or 'implicit' 
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
            - title: title for the plots (default: 'Magnetic Field Evolution')
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
        if derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_total'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_total'
    else:  # differential
        index_O, index_F = 1, len(grid_t)
        if derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_derivative'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_derivative'
    
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
    
    def setup_axis(ax, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis):
        """Helper function to set up axis properties"""
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
        """Check if component has any non-zero values worth plotting"""
        return any(abs(val) > threshold for val in data)
    
    # Create figures list
    figures = []
    
    # Main evolution plot
    fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Main energy line (always plot)
    if evolution_type == 'total':
        label = 'Magnetic Energy'
    else:
        label = 'Magnetic Energy Induction'
    ax1.plot(x_data[index_O_plot:index_F_plot], n1_data[index_O_plot:index_F_plot], 
            linewidth=line1, label=label)
    
    # Induction prediction (plot if has data)
    if should_plot_component(n0_data):
        if evolution_type == 'total':
            label = 'Magnetic Energy \nfrom Induction'
        else:
            label = 'Predicted Induction'
        ax1.plot(x_data[index_o_plot:index_f_plot], n0_data, 
                linewidth=line1, label=label)
    
    # Track which components were plotted
    components_plotted = []
    
    # Total work (compacted)
    if should_plot_component(total_work_data):
        ax1.plot(x_data[index_o_plot:index_f_plot], total_work_data, '--', 
                linewidth=line1, label='Magnetic Energy \nfrom Induction (Compacted)')
        components_plotted.append('total')
    
    # Individual components with their colors
    component_configs = [
        (diver_work_data, 'Divergence', 'pink'),
        (compres_work_data, 'Compression', 'purple'),
        (stretch_work_data, 'Stretching', 'orange'),
        (advec_work_data, 'Advection', 'red'),
        (drag_work_data, 'Cosmic Drag', 'gray')
    ]
    
    for data, label, color in component_configs:
        if should_plot_component(data):
            ax1.plot(x_data[index_o_plot:index_f_plot], data, '--', 
                    linewidth=line2, label=label, color=color)
            components_plotted.append(label.lower())
    
    # Kinetic energy
    if should_plot_component(kinetic_work_data):
        ax1.plot(x_data[index_O_plot:index_F_plot], kinetic_work_data[index_O_plot:index_F_plot], 
                linewidth=line1, label='Kinetic Energy', color='#8FBC8F')
        components_plotted.append('kinetic')
    
    setup_axis(ax1, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis)
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
        
        # Save main plot
        file_title = '_'.join(title.split()[:3])
        filename1 = f'{folder}/{file_title}_integrated_energy_{sim_info}_{axis_info}_{limit_info}_{plotid}{plot_suffix}_{run}.png'
        fig1.savefig(filename1, dpi=dpi)
        
        if verbose:
            print(f'Plotting... Main plot saved as: {filename1}')
        
        # Save volume plot if created
        if volume_evolution:
            filename2 = f'{folder}/{file_title}_volume_{sim_info}_{axis_info}_{limit_info}_{plotid}_{run}.png'
            fig2.savefig(filename2, dpi=dpi)
            
            if verbose:
                print(f'Plotting... Volume plot saved as: {filename2}')
                
    if verbose:       
        for i, sim in enumerate(induction_params.get('sims', ['default'])):
            n_iter = len(induction_params.get('it', [0])) - 1 if derivative == 'central' else len(induction_params.get('it', [0])) - 2
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

def plot_rate_evolution(evolution_data, plot_params, induction_params,
                            grid_t, grid_zeta, rad,
                            verbose=True, save=False, folder=None):
    """
    Plot the evolution of the integrated magnetic energy and its induction components.
    
    Args:
        - evolution_data: dictionary containing the evolution data from induction_energy_integral_evolution()
        - plot_params: dictionary containing plotting parameters:
            - evolution_type: 'total' or 'differential'
            - derivative: 'central' or 'implicit' 
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
            - title: title for the plots (default: 'Magnetic Field Evolution')
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
        if derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_total'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_total'
    else:  # differential
        index_O, index_F = 1, len(grid_t)
        if derivative == 'central':
            index_o, index_f = 1, len(grid_t)
            plotid = 'central_derivative'
        else:  # implicit
            index_o, index_f = 2, len(grid_t)
            plotid = 'implicit_derivative'
    
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
    
    def setup_axis(ax, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis):
        """Helper function to set up axis properties"""
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
        """Check if component has any non-zero values worth plotting"""
        return any(abs(val) > threshold for val in data)
    
    # Create figures list
    figures = []
    
    # Main evolution plot
    fig1, ax1 = plt.subplots(figsize=figure_size, dpi=dpi)
    
    # Main energy line (always plot)
    if evolution_type == 'total':
        label = 'Magnetic Energy'
    else:
        label = 'Magnetic Energy Induction'
    ax1.plot(x_data[index_O_plot:index_F_plot], n1_data[index_O_plot:index_F_plot], 
            linewidth=line1, label=label)
    
    # Induction prediction (plot if has data)
    if should_plot_component(n0_data):
        if evolution_type == 'total':
            label = 'Magnetic Energy \nfrom Induction'
        else:
            label = 'Predicted Induction'
        ax1.plot(x_data[index_o_plot:index_f_plot], n0_data, 
                linewidth=line1, label=label)
    
    # Track which components were plotted
    components_plotted = []
    
    # Total work (compacted)
    if should_plot_component(total_work_data):
        ax1.plot(x_data[index_o_plot:index_f_plot], total_work_data, '--', 
                linewidth=line1, label='Magnetic Energy \nfrom Induction (Compacted)')
        components_plotted.append('total')
    
    # Individual components with their colors
    component_configs = [
        (diver_work_data, 'Divergence', 'pink'),
        (compres_work_data, 'Compression', 'purple'),
        (stretch_work_data, 'Stretching', 'orange'),
        (advec_work_data, 'Advection', 'red'),
        (drag_work_data, 'Cosmic Drag', 'gray')
    ]
    
    for data, label, color in component_configs:
        if should_plot_component(data):
            ax1.plot(x_data[index_o_plot:index_f_plot], data, '--', 
                    linewidth=line2, label=label, color=color)
            components_plotted.append(label.lower())
    
    # Kinetic energy
    if should_plot_component(kinetic_work_data):
        ax1.plot(x_data[index_O_plot:index_F_plot], kinetic_work_data[index_O_plot:index_F_plot], 
                linewidth=line1, label='Kinetic Energy', color='#8FBC8F')
        components_plotted.append('kinetic')
    
    setup_axis(ax1, x_scale, y_scale, xlim, ylim, cancel_limits, x_axis)
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
        
        # Save main plot
        file_title = '_'.join(title.split()[:3])
        filename1 = f'{folder}/{file_title}_integrated_energy_{sim_info}_{axis_info}_{limit_info}_{plotid}{plot_suffix}_{run}.png'
        fig1.savefig(filename1, dpi=dpi)
        
        if verbose:
            print(f'Plotting... Main plot saved as: {filename1}')
        
        # Save volume plot if created
        if volume_evolution:
            filename2 = f'{folder}/{file_title}_volume_{sim_info}_{axis_info}_{limit_info}_{plotid}_{run}.png'
            fig2.savefig(filename2, dpi=dpi)
            
            if verbose:
                print(f'Plotting... Volume plot saved as: {filename2}')
                
    if verbose:       
        for i, sim in enumerate(induction_params.get('sims', ['default'])):
            n_iter = len(induction_params.get('it', [0])) - 1 if derivative == 'central' else len(induction_params.get('it', [0])) - 2
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
        fig.write_image(folder + f'/{file_title}_{run}.png', width=900, height=900, scale=2, dpi = DPI)