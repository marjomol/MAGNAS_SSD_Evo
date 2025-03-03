"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

plot_fields module
Contains functions to plot the magnetic field seed or any other interesting quantities.

Created by Marco Molina Pradillo
"""

import numpy as np
import scripts.diff as diff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from scipy import stats
import plotly.graph_objects as go
import os
from . import spectral

def plot_seed_spectrum(alpha_index, Bx, By, Bz, dx, mode = 1, epsilon = 1e-30, Save = False, DPI = 300, run = '_', folder = None):
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
        - Save: boolean to save the plot or not
        - DPI: dots per inch in the plot
        - run: string to identify the run
        - folder: folder to save the plot
        
    Returns:
        - Plot of the magnetic field seed power spectrum and other comparatives
        
    Author: Marco Molina
    '''
    nmax, nmay, nmaz = Bx[0].shape
    
    # Compute the magnetic field magnitude
    Bmag = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
    
    # Compute the magnetic power spectrum
    k_bins, P_k = spectral.power_spectrum_vector_field(Bx[0], By[0], Bz[0], dx=dx)
    P_k = np.where(P_k == 0, epsilon, P_k)
    k_bins = k_bins * 2 * np.pi
    
    ## Calculate the standard deviation of the power spectrum for each bin ##
    
    # First we need to compute the FFT, its amplitude square, and normalise it
    Bx_fft = np.fft.fftn(Bx[0]) # Done with the scipy fft in spec, maybe the normalization is different
    By_fft = np.fft.fftn(By[0])
    Bz_fft = np.fft.fftn(Bz[0])
    Bx_fourier_amplitudes = (np.abs(Bx_fft)**2).flatten() / Bx[0].size**2 * (nmax*nmay*nmaz)*(dx)**3
    By_fourier_amplitudes = (np.abs(By_fft)**2).flatten() / By[0].size**2 * (nmax*nmay*nmaz)*(dx)**3
    Bz_fourier_amplitudes = (np.abs(Bz_fft)**2).flatten() / Bz[0].size**2 * (nmax*nmay*nmaz)*(dx)**3

    # Next we obtain the frequencies
    kx = np.fft.fftfreq(nmax, dx) * 2 * np.pi # Es seguro el factor 2*pi?
    ky = np.fft.fftfreq(nmay, dx) * 2 * np.pi
    kz = np.fft.fftfreq(nmaz, dx) * 2 * np.pi
    k_grid = np.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = k_grid[0]**2 + k_grid[1]**2 + k_grid[2]**2
    k_magnitude = np.sqrt(k_squared).flatten()

    # Assuming isotropy, we can obtain the P(k) (1-dimensional power spectrum) standard deviation
    k_stats_bins = np.arange(kx[1]/2, np.abs(kx).max(), kx[1])

    P_k_x_std, _, _ = stats.binned_statistic(k_magnitude, Bx_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    P_k_y_std, _, _ = stats.binned_statistic(k_magnitude, By_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    P_k_z_std, _, _ = stats.binned_statistic(k_magnitude, Bz_fourier_amplitudes, statistic='std', bins=k_stats_bins)
    
    P_k_std = P_k_x_std + P_k_y_std + P_k_z_std
    
    # Computes the magnetic field seed divergence
    diver_B = diff.periodic_divergence(Bx, By, Bz, dx, npatch = np.array([0]), stencil=5, kept_patches=None)

    # Calculate the expected trend lines starting from specifict values of P_k
    klim = 1 # Set axis limits in the power spectrum plot (klim=1 and k0=0 for showing all, ko defines the starting point of the trend line)
    k0 = 0
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
        sub_sample = (0.1*256/nmax) * nmax * nmay * nmaz
        random_inndex = np.random.choice((nmax-(nmax//2))*(nmay-(nmay//2))*(nmaz-(nmaz//2)), size=int(sub_sample), replace=False)
        random_cells = np.abs(diver_B[0][nmax//4:-nmax//4,nmay//4:-nmay//4,nmaz//4:-nmaz//4].flatten()[random_inndex])
        yyy = [np.percentile(random_cells, p) for p in xxx]

        ax[1].clear()
        ax[1].set_title(f'Cumulative Relative Divergence of the Magnetic Field, $N_c$ = {Bx[0].shape[0]}')
        ax[1].set_xlabel('Percentage of cells')
        ax[1].set_ylabel('|∇·B|/B')
        ax[1].plot(xxx, yyy / np.mean(Bx[0]**2 + By[0]**2 + Bz[0]**2)**0.5 * dx)
        ax[1].set_xscale('linear')
        ax[1].set_yscale('log')
        ax[1].grid()
    
    # Spectrum with the Cumulative Absolute Divergence of the Magnetic Field
    elif mode == 2:
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(16, 6), dpi=DPI)
        ax = ax.flatten()

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        nmax, nmay, nmaz = Bx[0].shape
        
        xxx = np.linspace(0, 100, 1001)
        sub_sample = (0.1*256/nmax) * nmax * nmay * nmaz
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

    ax[0].clear()  
    ax[0].set_title(f'Random Transverse Projected Magnetic Field Power Spectrum')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('P(k)')
    ax[0].loglog(k_bins[k0:-klim], P_k[k0:-klim], label='Magnetic Power Spectrum')
    ax[0].fill_between(k_bins[k0:-klim], P_k[k0:-klim] - P_k_std[k0:-klim], P_k[k0:-klim] + P_k_std[k0:-klim], color='gray', alpha=0.25)
    ax[0].loglog(k_bins[k0+ko:-klim], trend_line, linestyle='dotted', label=f'$k^{{{np.round(alpha_index,2)}}}$')
    ax[0].legend()
    ax[0].set_xlim(k_bins[k0] - 1, k_bins[-klim] + 1)
    ax[0].set_ylim(1e-28, 1e-12)
    ax[0].legend(loc='lower center')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
            
        fig.savefig(folder + f'/Mag_Field_Power_Spectrum_{run}.png')
        
def zoom_animation_3D(arr, dx, arrow_scale = 1, units = 'Mpc', title = 'Gaussian Filtered Magnetic Field Zoom', Save = False, DPI = 300, run = '_', folder = None):
    '''
    Generates an animation of the magnetic field seed in 3D with a zoom effect. Can be used for any other 3D spacial field.
    
    Args:
        - arr: 3D array to animate
        - dx: cell size in Mpc
        - arrow_scale: scale of the arrow in Mpc
        - units: units of the arrow scale
        - title: title of the animation
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
    
    nmax, nmay, nmaz = arr.shape
    
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
    
    def animate(frame):
        plt.clf()
        imdim = np.round((frame+arrow_scale)/dx, 0).astype(int)
        section = np.sum(arr[(nmax//2 - imdim):(nmax//2 + imdim), (nmay//2 - imdim):(nmay//2 + imdim), (nmaz//2 - depth//2):(nmaz//2 + depth//2)], axis=2)
        plt.imshow(section, cmap='viridis')
        plt.title(title)
        ctoMpc = arrow_scale/dx
        plt.arrow(imdim, imdim, ctoMpc-(ctoMpc/7), 0, head_width=(ctoMpc/14), head_length=(ctoMpc/7), fc=col, ec=col)
        plt.text(imdim, imdim-arrow_scale, f'{arrow_scale} {units}', color=col)

    ani = FuncAnimation(fig, animate, frames = range(1, max_frame), interval=inter)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ani.save(data_dir + '/animation.gif', writer='pillow')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
    
        file_title = ' '.join(title.split()[:4])
        ani.save(folder + f'/{file_title}_{run}_zoom.gif', writer='pillow', dpi = DPI)
        
def scan_animation_3D(arr, dx, study_box, arrow_scale = 1, units = 'Mpc', title = 'Gaussian Filtered Magnetic Field Scan', Save = False, DPI = 300, run = '_', folder = None):
    '''
    Generates an animation of the magnetic field seed in 3D with a scan effect. Can be used for any other 3D spacial field.
    
    Args:
        - arr: 3D array to animate
        - dx: cell size in Mpc
        - study_box: percentage of the box to scan centered in the middle of the scanning plane. Must be a float in (0, 1]
        - arrow_scale: scale of the arrow in Mpc
        - units: units of the arrow scale
        - title: title of the animation
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
    assert 0 < study_box <= 1, "Study box must be a float in (0, 1]"
    
    nmax, nmay, nmaz = arr.shape
    
    inter = 100
    depth = 2
    x_lsize = round(nmax//2 - nmax*study_box//2)
    x_dsize = round(nmax//2 + nmax*study_box//2)
    y_lsize = round(nmay//2 - nmay*study_box//2)
    y_dsize = round(nmay//2 + nmay*study_box//2)
    new_nmax = x_dsize - x_lsize
    col = 'red'
    
    fig = plt.figure(figsize=(5, 5))
    
    # Find the minimum and maximum values of the magnetic field among all the studied volume
    min_value = np.min([np.min(arr[x_lsize:x_dsize, y_lsize:y_dsize, i]) for i in range(len(arr[0]))])
    max_value = np.max([np.max(arr[x_lsize:x_dsize, y_lsize:y_dsize, i]) for i in range(len(arr[0]))])
    
    # # Check if the minimum and maximum values are valid
    # if min_value <= 0:
    #     min_value = 1e-8
    # if max_value <= 0:
    #     max_value = 1e-3
        
    # Create a logarithmic normalization for the color intensity and regulate the intensity of the color bar
    norm = LogNorm(vmin=100*min_value, vmax=max_value)
    
    def animate(frame):
        plt.clf()
        section = np.sum(arr[x_lsize:x_dsize, y_lsize:y_dsize, (frame - depth//2):(frame + depth//2)], axis=2)
        plt.imshow(section, cmap='viridis', norm=norm)
        # plt.imshow(section, cmap='viridis')
        plt.title(title)
        ctoMpc = arrow_scale/dx
        plt.arrow((new_nmax - 4*new_nmax//5), (new_nmax - new_nmax//10), ctoMpc-(ctoMpc/7), 0, head_width=(ctoMpc/14), head_length=(ctoMpc/7), fc=col, ec=col)
        plt.text((new_nmax - 4*new_nmax//5), (new_nmax - new_nmax//10) - 1.25 * arrow_scale, f'{arrow_scale} {units}', color=col)

    ani = FuncAnimation(fig, animate, frames = range(depth, nmax), interval=inter)
    
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    ani.save(data_dir + '/animation.gif', writer='pillow')
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
    
        file_title = ' '.join(title.split()[:4])
        ani.save(folder + f'/{file_title}_{run}_scan.gif', writer='pillow', dpi = DPI)
        
def plot_3D_volume(arr, axis_values, log = False, subvolume_factor = 1, subsampling_step = 2, axis_step = 10, quantity = ' ', axis_title = ['x', 'y', 'z'], units = ' ', title = ' ', invert = False, Save = False, DPI = 300, run = '_', folder = None):
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
    
    # Save the plots
    if Save == True:
        
        if folder is None:
            folder = os.getcwd()
            
        file_title = ' '.join(title.split()[:4])
        fig.write_image(folder + f'/{file_title}_{run}.png', width=900, height=900, scale=2, dpi = DPI)