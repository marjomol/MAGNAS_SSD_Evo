"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

spectral module
Provides useful tools to compute power spectra, energy power spectra, etc.

Created by David Vallés for MASCLET framework
"""

import numpy as np
from scipy import fft, stats

def power_spectrum_scalar_field(data, dx=1., ncores=1, do_zero_pad=False,
                                zero_pad_factor=1.):
    '''
    This function computes the power spectrum, P(k), of a 3D cubic scalar field.

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain
        - zero_pad_factor: if do_zero_pad is True, this is the factor by which the domain is 
                increased. For example, if zero_pad_factor=2, the domain is doubled. Should
                be a power of 2.

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Pk: the power spectrum at the kvals spatial frequency points

    '''

    # Step 1. Compute the FFT, its amplitude square, and normalise it
    #fft_data = np.fft.fftn(data, s=data.shape)#.shape
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if not do_zero_pad:
        shape = data.shape
    else:
        zero_pad_factor = int(zero_pad_factor)
        shape = [zero_pad_factor*s for s in data.shape]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
        
    fft_data = fft.fftn(data, s=shape, workers=ncores)
    fourier_amplitudes = (np.abs(fft_data)**2).flatten() / data.size**2 * (data.shape[0]*dx)**3
    nx,ny,nz = data.shape

    # Step 2. Obtain the frequencies
    frequencies_x = np.fft.fftfreq(shape[0], d=dx) * 2 * np.pi # Es seguro el factor 2*pi?
    frequencies_y = np.fft.fftfreq(shape[1], d=dx) * 2 * np.pi 
    frequencies_z = np.fft.fftfreq(shape[2], d=dx) * 2 * np.pi 
    a,b,c=np.meshgrid(frequencies_x, frequencies_y, frequencies_z, indexing='ij')
    knrm=np.sqrt(a**2+b**2+c**2).flatten()

    # Step 3. Assuming isotropy, obtain the P(k) (1-dimensional power spectrum)
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if not do_zero_pad:
        delta_f = frequencies_x[1]
    else:
        delta_f = float(zero_pad_factor)*frequencies_x[1]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
    
    kbins = np.arange(frequencies_x[1]/2, np.abs(frequencies_x).max(), delta_f)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Pk, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                    statistic = "mean",
                                    bins = kbins)

    return kvals, Pk

def power_spectrum_vector_field(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False,
                                zero_pad_factor=1.):
    '''
    This function computes the power spectrum, P(k), of a 3D cubic vector field.

    Args:
        - data_x, data_y, data_z: the 3D arrays containing the 3 components of the vector field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain
        - zero_pad_factor: if do_zero_pad is True, this is the factor by which the domain is
                increased. For example, if zero_pad_factor=2, the domain is doubled. Should
                be a power of 2.

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Pk: the power spectrum at the kvals spatial frequency points

    '''

    kvals, Pk_x = power_spectrum_scalar_field(data_x, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad,
                                            zero_pad_factor=zero_pad_factor)
    kvals, Pk_y = power_spectrum_scalar_field(data_y, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad,
                                            zero_pad_factor=zero_pad_factor)
    kvals, Pk_z = power_spectrum_scalar_field(data_z, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad,
                                            zero_pad_factor=zero_pad_factor)

    Pk = Pk_x+Pk_y+Pk_z

    return kvals, Pk

def energy_spectrum_scalar_field(data, dx=1., ncores=1, do_zero_pad=False, zero_pad_factor=1.):
    '''
    This function computes the energy power spectrum, E(k), of a 3D cubic scalar field.

    This is defined from the P(k) as:
        E(k) = 2 \pi k^2 P(k)

    And satisfies:
        \int_0^\infty E(k) dk = 1/2 <data^2>

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain
        - zero_pad_factor: if do_zero_pad is True, this is the factor by which the domain is
                increased. For example, if zero_pad_factor=2, the domain is doubled. Should
                be a power of 2.

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Ek: the energy power spectrum at the kvals spatial frequency points

    '''

    kvals, pk = power_spectrum_scalar_field(data, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad,
                                            zero_pad_factor=zero_pad_factor)
    Ek = pk * (2*np.pi*kvals**2)

    return kvals, Ek

def energy_spectrum_vector_field(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False,
                                zero_pad_factor=1.):
    '''
    This function computes the energy power spectrum, E(k), of a 3D cubic vector field.

    This is defined from the P(k) as:
        E(k) = 2 \pi k^2 P(k)

    And satisfies:
        \int_0^\infty E(k) dk = 1/2 <\vec data^2>

    Args:
        - data_x, data_y, data_z: the 3D arrays containing the 3 components of the vector field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain
        - zero_pad_factor: if do_zero_pad is True, this is the factor by which the domain is
                increased. For example, if zero_pad_factor=2, the domain is doubled. Should
                be a power of 2.

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Ek: the energy power spectrum at the kvals spatial frequency points

    '''
    kvals, pk = power_spectrum_vector_field(data_x, data_y, data_z, dx=dx, ncores=ncores, 
                                            do_zero_pad=do_zero_pad, zero_pad_factor=zero_pad_factor)
    Ek = pk * (2*np.pi*kvals**2)
    return kvals, Ek