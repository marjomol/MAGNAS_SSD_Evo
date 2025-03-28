"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

seed_generator module
Provides a set of functions to generate a stochastic magnetic field following a given power spectrum
in a cosmological context.

Created by Marco Molina Pradillo
"""

import gc
import os
import numpy as np
import scripts.utils as utils
import scripts.diff as diff
from scripts.units import *
from scipy.special import gamma
from matplotlib import pyplot as plt
npatch = np.array([0]) # We only want the zero patch for the seed

import pdb
np.set_printoptions(linewidth=200)

def random_hermitian_vector(n):
    '''
    Generates a random Hermitian vector of size n to be used in the generation of Fourier space quantities.
    
    Args:
        - n: size of the Hermitian vector. Must be an even number
        
    Returns:
        - vector: Hermitian vector of size n
        
    Author: Marco Molina
    '''
    
    # Ensure the size of the Hermitian vector is an even number
    assert n % 2 == 0, "The size of the Hermitian vector must be an even number"
    
    first_half = np.random.uniform(0, 1, size=(n//2 + 1))
    second_half = first_half[1:][::-1]
    
    vector = np.concatenate((first_half, second_half))
    
    return vector

def symmetrize(arr, mode = 0):
    '''
    Makes an array positive or negative symmetric to be used in the Fourier space.
    
    Args:
        - arr: input array to be made negative symmetric. Must be 3D and have the positive and negative Nyquist frequencies explicitly separated and centered.
        - mode: method to make the array positive or negative symmetric
        
    Returns:
        - arr: negative symmetric array
        
    Author: Marco Molina
    '''
    
    shape = arr.shape

    # Ensure the array is 3D
    assert arr.ndim == 3, "Input array must be 3D"
    
    # Ensure the size of the Hermitian vector is an even number
    assert (shape[0]-1) % 2 == 0, "The x-size of the array must be an even number"
    assert (shape[1]-1) % 2 == 0, "The y-size of the array must be an even number"
    assert (shape[2]-1) % 2 == 0, "The z-size of the array must be an even number"
    
    # Create a negative symmetric array in a fast vectorized way (only works for
    # the epectral signal pattern-resulting case of same phase for same k-magnitude value)
    if mode == 0:
        
        # Account for the zero and Nyquist planes and axis
        
        arr[(shape[0]//2+1):] = -arr[(shape[0]//2+1):]
        
        arr[0][0][(shape[1]//2+1):] = -arr[0][0][(shape[1]//2+1):]
        arr[0][(shape[2]//2+1):] = -arr[0][(shape[2]//2+1):]
    
    # Create an negative symmetric array in a slow but sure loop way (to be used on the array of
    # the imaginary part of the random phase)
    elif mode == 1:
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(int(arr.shape[2]/2)+1):
                    iconj = -i % arr.shape[0]
                    jconj = -j % arr.shape[1]
                    kconj = -k % arr.shape[2]                    
                    dif = arr[i, j, k] + arr[iconj, jconj, kconj]
                    if dif != 0:
                        arr[iconj, jconj, kconj] = -arr[i, j, k]

        arr[0][0][0] = -arr[0][0][0]
    
    # Create an positive symmetric array in a slow but sure loop way (to be used on the array of
    # the real part of the random phase)
    elif mode == 2:
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(int(arr.shape[2]/2)+1):
                    iconj = -i % arr.shape[0]
                    jconj = -j % arr.shape[1]
                    kconj = -k % arr.shape[2]
                    dif = arr[i, j, k] - arr[iconj, jconj, kconj]
                    if dif != 0:
                        arr[iconj, jconj, kconj] = arr[i, j, k]

        arr[0][0][0] = arr[0][0][0]
    
    return arr

def merge_nyquist(arr, axis='none', memmap = False, complex_bitformat = np.complex64):
    '''
    PRIMAL Seed Generator works with the Nyquist frequencies explicitly splited and centered in the Fourier
    space to ensure the correct signal transverse projection works. Given a 3D Fourier space array with the
    positive and negative Nyquist frequencies separated and centered, this function merges their signal making
    it apt for the numpy inverse Fourier transform where the Nyquist signal frequencies are implicitly merged.
    
    Args:
        - arr: imput array with the positive and negative Nyquist frequencies explicitly splited and centered
        - axis: axis of the random magnetic field seed amplitude needed to be generated. The axis must be x, y, or z
        - memmap: boolean to decide if the data is processed with memory cautions or not
        - complex_bitformat: format of the complex numbers in the array. Only used if memmap is True
        
    Returns:
        - arr: array with the positive and negative Nyquist frequencies merged
        
    Author: Marco Molina
    '''
    
    shape = arr.shape

    # Ensure the array is 3D
    assert arr.ndim == 3, "Input array must be 3D"
    
    # Ensure the size of the Hermitian vector is an even number
    assert (shape[0]-1) % 2 == 0, "The x-size of the array must be an even number"
    assert (shape[1]-1) % 2 == 0, "The y-size of the array must be an even number"
    assert (shape[2]-1) % 2 == 0, "The z-size of the array must be an even number"
    
    if memmap:
        assert axis != 'none', "The axis must be x, y, or z."
        
        # Join the signal of the positive and negative Nyquist frequencies while avoiding one of the two central planes
        # new_arr = np.memmap(f'data/_Imaginary_B_{axis}_temporary_file_.dat', dtype=complex_bitformat,
        #                     mode='w+', shape=(shape[0]-1, shape[1]-1, ((shape[2]-1)//2)+1))
        new_arr = np.memmap(f'data/_Imaginary_B_{axis}_temporary_file_.dat', dtype=complex_bitformat,
                    mode='w+', shape=(shape[0]-1, shape[1]-1, shape[2]-1))

        # Only half of the Fourier space is needed to use the Real numpy inverse Fourier transform
        # as the other half is the Hermitian conjugate of the first half
        new_arr[:shape[0]//2, :shape[0]//2, :shape[0]//2] = arr[:shape[0]//2, :shape[0]//2, :shape[0]//2]
        new_arr[:shape[0]//2, shape[0]//2:, :shape[0]//2] = arr[:shape[0]//2, shape[0]//2+1:, :shape[0]//2]
        new_arr[shape[0]//2:, :shape[0]//2, :shape[0]//2] = arr[shape[0]//2+1:, :shape[0]//2, :shape[0]//2]
        new_arr[shape[0]//2:, shape[0]//2:, :shape[0]//2] = arr[shape[0]//2+1:, shape[0]//2+1:, :shape[0]//2]
        
        new_arr[:shape[0]//2, :shape[0]//2, shape[0]//2:] = arr[:shape[0]//2, :shape[0]//2, shape[0]//2+1:]
        new_arr[:shape[0]//2, shape[0]//2:, shape[0]//2:] = arr[:shape[0]//2, shape[0]//2+1:, shape[0]//2+1:]
        new_arr[shape[0]//2:, :shape[0]//2, shape[0]//2:] = arr[shape[0]//2+1:, :shape[0]//2, shape[0]//2+1:]
        new_arr[shape[0]//2:, shape[0]//2:, shape[0]//2:] = arr[shape[0]//2+1:, shape[0]//2+1:, shape[0]//2+1:]
    
        del arr
        gc.collect()
        
        # Flush changes to disk if it's a memory-mapped array
        if isinstance(new_arr, np.memmap):
            new_arr.flush()
    
        return new_arr
    else:
        # Join the signal of the positive and negative Nyquist frequencies
        arr[shape[0]//2, :, :] = arr[shape[0]//2+1, :, :]
        arr[:, shape[1]//2, :] = arr[:, shape[1]//2+1, :]
        arr[:, :, shape[2]//2] = arr[:, :, shape[2]//2+1]
        
        # Erase one of the two Nyquist central planes in all three directions of the B_k 3D arrays
        arr = np.delete(arr, shape[0]//2+1, axis=0)
        arr = np.delete(arr, shape[1]//2+1, axis=1)
        arr = np.delete(arr, shape[2]//2+1, axis=2)
    
        return arr

def k_null(arr, k_val, k_mag):
    '''
    Makes the elemensts of the array corresponding to null and Nyquist frequency zero. The positive and negative
    Nyquist frequencies are assumed to be explicitly splited and centered.
    
    Args:
        - arr: input array to be made zero
        - k_val: wave vector unique values array in Mpc^{-1}
        - k_mag: wave number in Mpc^{-1}
    
    Returns:
        - arr: array with the null and Nyquist frequency zero coresponding elements.
        
    Author: Marco Molina
    '''
    
    # Ensure the array is 3D
    assert k_mag.ndim == 3, "Input array must be 3D"
    
    shape = k_mag.shape
    
    # Ensure the size of the Hermitian vector is an even number
    assert (shape[0]-1) % 2 == 0, "The x-size of the array must be an even number"
    assert (shape[1]-1) % 2 == 0, "The y-size of the array must be an even number"
    assert (shape[2]-1) % 2 == 0, "The z-size of the array must be an even number"
    
    
    arr[k_val == k_mag[0, 0, 0]] = 0.
    arr[k_val == k_mag[shape[0]//2, 0, 0]] = 0.
    arr[k_val == k_mag[0, shape[1]//2, 0]] = 0.
    arr[k_val == k_mag[0, 0, shape[2]//2]] = 0.
    arr[k_val == k_mag[shape[0]//2 + 1, 0, 0]] = 0.
    arr[k_val == k_mag[0, shape[1]//2 + 1, 0]] = 0.
    arr[k_val == k_mag[0, 0, shape[2]//2 + 1]] = 0.
    arr[k_val == k_mag[shape[0]//2, shape[1]//2, 0]] = 0.
    arr[k_val == k_mag[shape[0]//2, 0, shape[2]//2]] = 0.
    arr[k_val == k_mag[0, shape[1]//2, shape[2]//2]] = 0.
    arr[k_val == k_mag[shape[0]//2 + 1, shape[1]//2 + 1, 0]] = 0.
    arr[k_val == k_mag[shape[0]//2 + 1, 0, shape[2]//2 + 1]] = 0.
    arr[k_val == k_mag[0, shape[1]//2 + 1, shape[2]//2 + 1]] = 0.
    arr[k_val == k_mag[shape[0]//2, shape[1]//2, shape[2]//2]] = 0.
    arr[k_val == k_mag[shape[0]//2 + 1, shape[1]//2 + 1, shape[2]//2 + 1]] = 0.
    
    return arr

def random_phase(axis, k_mag, N, epsilon = 1e-30, mode = 0):
    '''
    Generates two random number 3D arrays for each three dimensions in order to be used in the generation of the
    magnetic field seed in Fourier space, transmiting the neccesary Fourier space information so as to ensure that
    it will be real after inversion to real space and with a random amplitude distribution around a constant value.
    
    The positive and negative Nyquist frequencies are assumed to be explicitly separated and centered.
    
    Different methods are implemented to generate the random numbers, each one with its own advantages and
    disadvantages depending on the aim. From modes 0 to 2 the methods are faster but generate pattern-like seeds;
    method 3 is the slowest but generates truly random seeds.
    
    Args:
        - axis: axis of the random number 3D array to be generated. The axis must be x, y, or z
        - k_mag: wave number magnitudes
        - N: tuple with the number of cells in each direction
        - epsilon: small number to avoid division by zero
        - mode: method to generate the random numbers
        
    Returns:
        - iota: random number 3D array for the asked dimension
        - beta: random number 3D array for the asked dimension
        
    Source: Developed for one dimenssion an any two quantities in H. Press, W.,
            T. Vetterling, W., A. Teukolsky, S., & P. Flannery, B. (1992).
            Numerical Recipes in Fortran 77 (Second, Vol. 1). Cambridge University Press.
            https://websites.pmc.ucsc.edu/~fnimmo/eart290c_17/NumericalRecipesinF77.pdf
            
            Adapted to work with the real and imaginary parts of any complex quantity by Vicent Quilis.

    Author: Marco Molina
    '''
    
    nmax, nmay, nmaz = N

    if axis == 'x':
        nma = nmax
    elif axis == 'y':
        nma = nmay
    elif axis == 'z':
        nma = nmaz
    else:
        raise ValueError('The axis must be x, y, or z.')

    # Method to create two 3D arrays of random numbers in a vectorized and fast way.
    # This method is correlating the random numbers and is restricting the range of the phase space, resulting in
    # Cartesian-like patterns in the final results, but it is faster and could be used in some test cases.
    if mode == 0:
        # Create two random number arrays for each three dimensions
        iota = [random_hermitian_vector(nma) for _ in range(sum(npatch)+1)]
        beta = [random_hermitian_vector(nma) for _ in range(sum(npatch)+1)]

        # Make elements corresponding to the null and Nyquist frequency zero in the iota arrays 
        for p in range(sum(npatch)+1):
            iota[p][0] = 0.
            iota[p][nma//2] = 0.
            iota[p][nma//2+1] = 0.

        # Create the grid of the random numbers
        iota_grid = [np.meshgrid(iota[p], iota[p], iota[p], indexing='ij') for p in range(sum(npatch)+1)]
        beta_grid = [np.meshgrid(beta[p], beta[p], beta[p], indexing='ij') for p in range(sum(npatch)+1)]

        # Assigning the same random number to the same k-magnitude value
        iota = [np.sqrt(iota_grid[p][0]**2 + iota_grid[p][1]**2 + iota_grid[p][2]**2) for p in range(sum(npatch)+1)]
        beta = [np.sqrt(beta_grid[p][0]**2 + beta_grid[p][1]**2 + beta_grid[p][2]**2) for p in range(sum(npatch)+1)]

        # Normalizing the random numbers between 0 and 1
        iota = [((iota[p] - np.min(iota[p])) / (np.max(iota[p]) - np.min(iota[p]))) for p in range(sum(npatch)+1)]
        beta = [((beta[p] - np.min(beta[p])) / (np.max(beta[p]) - np.min(beta[p]))) for p in range(sum(npatch)+1)]

        for p in range(sum(npatch)+1):
            iota[p][iota[p] == 0] = epsilon
            beta[p][beta[p] == 0] = epsilon

        # Making the random iota arrays negative symmetric so they transmit the Fourier space properties when used in the phase
        iota = [symmetrize(iota[p], mode = 0) for p in range(sum(npatch)+1)]

    # Method to asign the random numbers to each of the posible k-magnitude values specifically.
    # This method is correlating the random numbers and is restricting the range of the phase space, resulting in
    # Cartesian-like patterns in the final results, but it is faster and could be used in some test cases.
    elif mode == 1:
        # Get all the possible k-magnitude values
        k_values = [np.unique(k_mag[p]) for p in range(sum(npatch)+1)]

        # Generate random numbers for each k-magnitude value, two arrays in each direction
        iota_values = [np.random.uniform(0, 1, size=len(k_values[p])) for p in range(sum(npatch)+1)]
        beta_values = [np.random.uniform(0, 1, size=len(k_values[p])) for p in range(sum(npatch)+1)]

        # Assing zero to the elements corresponding to the null and Nyquist frequency values
        iota_values = [k_null(iota_values[p], k_values[p], k_mag[p]) for p in range(sum(npatch)+1)]
        beta_values[0][beta_values[0] == 0] = epsilon

        # Populate the grids of random numbers so as to have the same random numbers for the same k-magnitude values
        iota = [np.zeros_like(k_mag[p]) for p in range(sum(npatch)+1)]
        beta = [np.zeros_like(k_mag[p]) for p in range(sum(npatch)+1)]

        for i, k in enumerate(k_values[0]):
            idx = k_mag[0] == k
            iota[0][idx] = iota_values[0][i]
            beta[0][idx] = beta_values[0][i]
        
    # Method to asign random numbers to each of the Fourier space values in a vectorized and fast way.
    # This method is correlating the random numbers and is restricting the range of the phase space, resulting in
    # Cartesian-like patterns in the final results, but it is faster and could be used in some test cases.
    elif mode == 2:
        # Create two random number arrays for each three dimensions
        variables = {
            'iota_1': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)],
            'iota_2': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)],
            'iota_3': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)],
            'beta_1': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)],
            'beta_2': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)],
            'beta_3': [random_hermitian_vector(nmax) for _ in range(sum(npatch)+1)]
        }

        # Make elements corresponding to the null and Nyquist frequency zero in the iota arrays 
        for p in range(sum(npatch)+1):
            for key in variables.keys():
                variables[key][p][0] = 0.
                variables[key][p][nmax//2] = 0.
                variables[key][p][nmax//2+1] = 0.
                
        iota_1 = variables['iota_1']
        beta_1 = variables['beta_1']
        iota_2 = variables['iota_2']
        beta_2 = variables['beta_2']
        iota_3 = variables['iota_3']
        beta_3 = variables['beta_3']

        # Create the grid of the random numbers
        iota_grid = [np.meshgrid(iota_1[p], iota_2[p], iota_3[p], indexing='ij') for p in range(sum(npatch)+1)]
        beta_grid = [np.meshgrid(beta_1[p], beta_2[p], beta_3[p], indexing='ij') for p in range(sum(npatch)+1)]

        # Assigning the same random number to the same k-magnitude value
        iota = [np.sqrt(iota_grid[p][0]**2 + iota_grid[p][1]**2 + iota_grid[p][2]**2) for p in range(sum(npatch)+1)]
        beta = [np.sqrt(beta_grid[p][0]**2 + beta_grid[p][1]**2 + beta_grid[p][2]**2) for p in range(sum(npatch)+1)]

        # Normalizing the random numbers between 0 and 1
        iota = [((iota[p] - np.min(iota[p])) / (np.max(iota[p]) - np.min(iota[p]))) for p in range(sum(npatch)+1)]
        beta = [((beta[p] - np.min(beta[p])) / (np.max(beta[p]) - np.min(beta[p]))) for p in range(sum(npatch)+1)]

        for p in range(sum(npatch)+1):
            iota[p][iota[p] == 0] = epsilon
            beta[p][beta[p] == 0] = epsilon

        # Making the random iota arrays negative symmetric so they transmit the Fourier space properties when used in the phase
        iota = [symmetrize(iota[p], mode = 0) for p in range(sum(npatch)+1)]
        
    # Method to create two truly hermitic random number 3D arrays in a slow but sure way.
    # This is the only method that is not correlating the random numbers and is not restricting the range of the phase space,
    # resulting in a truly stochastic distribution of the random numbers in the final results.
    if mode == 3:
        
        # Randomize the arrays completely taking the array structure from k_mag        
        iota = [np.random.uniform(0, 1, size=k_mag[p].shape) for p in range(sum(npatch)+1)]
        beta = [np.random.uniform(0, 1, size=k_mag[p].shape) for p in range(sum(npatch)+1)]

        for p in range(sum(npatch)+1):
            iota[p][iota[p] == 0] = epsilon
            beta[p][beta[p] == 0] = epsilon

        # Making the random iota arrays negative symmetric so they transmit the Fourier space properties when used in the phase
        iota = [symmetrize(iota[p], mode = 1) for p in range(sum(npatch)+1)]
        
        # Making the beta arrays positive symmetric
        beta = [symmetrize(beta[p], mode = 2) for p in range(sum(npatch)+1)]
    
    return iota, beta

def vector_kd(alpha_index, lambda_scale, B_lambda, h_cosmo):
    '''
    Computes the damping scale of the cosmological magnetic field seed due to Alfven wave damping dissipation in the vector modes.
    
    Args:
        - alpha_index: spectral index of the magnetic field
        - lambda_scale: comoving smoothing length in Mpc
        - B_lambda: magnetic field amplitude at the comoving smoothing length in Gauss
        - h_cosmo: Hubble constant in units of 100 km/s/Mpc
        
    Returns:
        - fd: damping scale frequency in Mpc^{-1}
        - kd: damping scale wave number in Mpc^{-1}
    
    Source: Paoletti, D., & Finelli, F. (2019). Constraints on primordial magnetic fields from magnetically-induced
            perturbations: Current status and future perspectives with LiteBIRD and future ground based experiments.
            Journal of Cosmology and Astroparticle Physics, 2019(11), 028. https://doi.org/10.1088/1475-7516/2019/11/028
            
    Author: Marco Molina
    '''
    
    # Damping scale frequency in Mpc^{-1}
    kd = ((5.5*10**4)**(1/(alpha_index+5))) * ((B_lambda)**(-2/(alpha_index+5))) * ((2*np.pi/lambda_scale)**((alpha_index+3)/(alpha_index+5))) * (h_cosmo)**(1/(alpha_index+5)) * (0.0478*(h_cosmo**2)/0.022)**(1/(alpha_index+5))
    
    return kd

def power_spectrum_amplitude(k_mag, alpha_index, lambda_scale, B_lambda, h_cosmo, size, N, gauss_rad_factor = 1, filtering = True, verbose = False):
    '''
    Computes the amplitude of the power spectrum of the cosmological magnetic field seed with a chosen spectral index.
    
    Args:
        - k_mag: wave number in Mpc^{-1}
        - alpha_index: spectral index of the magnetic field
        - lambda_scale: comoving smoothing length in Mpc
        - B_lambda: magnetic field amplitude at the comoving smoothing length in Gauss
        - h_cosmo: Hubble constant in units of 100 km/s/Mpc
        - size: size of the box in Mpc
        - N: number of cells in each direction
        - gauss_rad_factor: factor to multiply the Gaussian filtering radius
        - filtering: boolean to apply the filtering and damping or not
        - verbose: boolean to print the parameters or not
    
    Returns:
        - P_B: amplitude of the power spectrum of the magnetic field seed
        
    Source: Vazza, F., Paoletti, D., Banfi, S., Finelli, F., Gheller, C., O’Sullivan, S. P., & Brüggen, M. (2021).
            Simulations and observational tests of primordial magnetic fields from Cosmic Microwave Background constraints.
            Monthly Notices of the Royal Astronomical Society, 500(4), 5350–5368. https://doi.org/10.1093/mnras/staa3532
            
    Author: Marco Molina
    '''
    
    # Power spectrum amplitude #
    
    # This are two different but equivalent ways to compute the power spectrum amplitude appearing in the bibliography.
    P_B = [(((2 * np.pi)**(alpha_index + 5) * B_lambda**2 * k_mag[p]**alpha_index) / (2 * gamma((alpha_index + 3)/2) * ((2 * np.pi) / lambda_scale)**(alpha_index + 3))) for p in range(sum(npatch)+1)]
    # P_B = [((2 * (np.pi)**2 * lambda_scale**3 * B_lambda**2) / gamma((alpha_index + 3)/2)) * (lambda_scale * k_mag[p])**alpha_index for p in range(sum(npatch)+1)]
    
    # Filtering the power spectrum #
    
    if filtering:
        
        n = max(N) # Maximum number of cells from each of the directions
    
        # Gaussian filtering
        R_fil = size / n 
        # R_fil = gauss_rad_factor * 2.0 * R_fil / lambda_scale # Gaussian filtering radius
        R_fil = gauss_rad_factor / lambda_scale # Gaussian filtering radius
        A_fil = k_mag[0]**2 * R_fil**2 / 2.0 # Amplitude of the Gaussian filter
        
        # Damping scale in (Mpc)^{-1}
        k_d = vector_kd(alpha_index, lambda_scale, B_lambda, h_cosmo)
        
        KD = k_mag[0] <= k_d # Damping scale mask
        
        P_B[0][~KD] = 0
        
        P_B = [P_B[p] * np.exp(-A_fil) for p in range(sum(npatch)+1)]
    
        if verbose:
            print('============================================================')
            print('Power Spectrum Amplitude completed')
            print('------------------------------------------------------------')
            print(f'Spectral index: {alpha_index}')
            print(f'Filtering Scale: {lambda_scale}')
            print(f'B0: {B_lambda} | h: {h_cosmo}')
            print('------------------------------------------------------------')
            print(f'Damping scale:            {k_d}')
            if np.any(KD):
                print(f'Maximum frequency number: {np.max(k_mag[0][KD])}')
            else:
                print(f'Maximum frequency number: none')
            print('------------------------------------------------------------')
            print(f'Filtering radius:         {R_fil}')
            if np.any(KD):
                print(f'Minimum frequency number: {np.min(k_mag[0][KD])}')
            else:
                print(f'Minimum frequency number: none')
            print('============================================================')
    
    return P_B

def generate_fourier_space(K, N, epsilon = 1e-30, verbose = False, debug = False):
    '''
    Generates the Fourier space quantities needed to compute the magnetic field seed.
    
    Args:
        - K: wave vector unique values array. Can be a chunk of the wave vector values in each direction.
        - N: number of cells in each direction
        - epsilon: small number to avoid division by zero
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
        
    Returns:
        - k_grid: wave vector k coordinates for each combination of kx, ky, and kz
        - k_magnitude: magnitude of the wave vector k for each combination of kx, ky, and kz
            
    Author: Marco Molina
    '''

    nmax, nmay, nmaz = N
    
    kx = K[0]
    ky = K[1]
    kz = K[2]
    
    # Components of the wave vector k for each combination of kx, ky, and kz.
    k_grid = [np.meshgrid(kx, ky, kz, indexing='ij') for _ in range(sum(npatch)+1)]
    
    # Squared magnitudes of the wave vector k for each combination of kx, ky, and kz.
    k_magnitude = [k_grid[p][0]**2 + k_grid[p][1]**2 + k_grid[p][2]**2 for p in range(sum(npatch)+1)]

    # Magnitude of the wave vector k for each combination of kx, ky, and kz.
    k_magnitude = [np.sqrt(k_magnitude[p]).astype(np.float64) for p in range(sum(npatch)+1)]

    # Avoid division by zero by setting zero magnitudes to a small number.
    for p in range(sum(npatch)+1):
        k_magnitude[p][k_magnitude[p] == 0] = epsilon
        
    if verbose:
        print('============================================================')
        print('Fourier Grid completed')
        print('------------------------------------------------------------')
        print(f'kx shape: {kx.shape}')
        print(f'kx grid shape: {k_grid[0][0].shape}')
        print('============================================================')
    
    # Debugging tests if needed, saving some k-arrays to check the values
    if debug:
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array('data/kx', k_grid[0][0])
            utils.save_3d_array('data/k_magnitude', k_magnitude[0])
        else:
            print('The debugging information is not saved because the number of cells is too large. Try with a smaller and manageable number of cells, like 6.')
        
        print('============================================================')
        print('Fourier Grid Information')
        print('------------------------------------------------------------')
        print(f'kx shape: {kx.shape}')
        print(f'ky shape: {ky.shape}')
        print(f'kz shape: {kz.shape}')
        print(f'kx grid shape: {k_grid[0][0].shape}')
        print(f'ky grid shape: {k_grid[0][1].shape}')
        print(f'kz grid shape: {k_grid[0][2].shape}')
        print(f'Frequency numbers: {kx}')
        print(f'k magnitude shape: {k_magnitude[0].shape}')
        print(f'Some k magnitude values: {k_magnitude[0][0:3, 0:3, 0:3]}')
        print('============================================================')
        assert np.allclose(k_grid[0][0][1:, 1:, 1:], -np.conj(np.flip(k_grid[0][0][1:, 1:, 1:]))), f"Debugging Test I Failed: kx grid is not simmetric."
        print(f"Debugging Test I Passed: kx grid is simmetric.")
        assert np.allclose(k_magnitude[0][1:, 1:, 1:], np.conj(np.flip(k_magnitude[0][1:, 1:, 1:]))), f"Debugging Test II Failed: k magnitude is not Hermitian."
        print(f"Debugging Test II Passed: k magnitude is Hermitian.")
        
        # Unitary vector in the direction of k
        k_hat = [[k_grid[p][0]/(k_magnitude[p]), k_grid[p][1]/(k_magnitude[p]), k_grid[p][2]/(k_magnitude[p])] for p in range(sum(npatch)+1)]
        # Verify that the magnitude of k_hat is 1 in each cell
        k_hat_magnitude = [np.sqrt(k_hat[p][0]**2 + k_hat[p][1]**2 + k_hat[p][2]**2) for p in range(sum(npatch)+1)]
        
        k_hat_magnitude[0][0,0,0] = 1.
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array('data/k_hat_magnitude', k_hat_magnitude[0])
        else:
            print('The debugging information is not saved because the number of cells is too large. Try with a smaller and manageable number of cells, like 6.')
            
        for p in range(sum(npatch)+1):
            
            assert np.allclose(k_hat_magnitude[p], 1), f"Debugging Test III Failed: Magnitude of k unitary vectors is not 1 for some elements."
            print(f"Debugging Test III Passed: Magnitude of k unitary vectors is 1 for all elements.")
            print('============================================================')
            
        del k_hat, k_hat_magnitude
        gc.collect()
        
    return k_grid, k_magnitude

def generate_seed_phase(axis, iota, beta, N, verbose = False, debug = False):
    '''
    Generates the random magnetic field seed phase in Fourier space.
    
    Args:
        - axis: axis of the random magnetic field seed phase needed to be generated. The axis must be x, y, or z
        - iota: random number 3D array for the asked dimension
            - If the hole array is processed, this must be a negative symmetric array resulting from @random_phase
        - beta: random number 3D array for the asked dimension
            - If the hole array is processed, this must be a positive symmetric array resulting from @random_phase
        - N: number of cells in each direction
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
        
    Returns:
        - B_phase_k: random magnetic field seed phase in Fourier space in the given direction
        
    Source: Phase theoretical treatment based on Vicent Quilis' procedure.
            
    Author: Marco Molina
    '''
    
    if axis not in ['x', 'y', 'z']:
        raise ValueError('The axis must be x, y, or z.')
    
    nmax, nmay, nmaz = N
    
    B_phase_k = [np.cos(2 * np.pi * iota[p]) * np.sqrt(-2 * np.log(beta[p]))
         + 1j * (np.sin(2 * np.pi * iota[p]) * np.sqrt(-2 * np.log(beta[p]))) for p in range(sum(npatch)+1)]
    
    if verbose:
        print('============================================================')
        print(f'Random {axis} Phase completed')
        print('------------------------------------------------------------')
        print(f'Random {axis} phase shape: {B_phase_k[0].shape}')
        print('============================================================')
    
    if debug:
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array(f'data/R(B_phase_k{axis})', np.real(B_phase_k[0]))
            utils.save_3d_array(f'data/I(B_phase_k{axis})', np.imag(B_phase_k[0]))
            utils.save_3d_array(f'data/iota_{axis}', iota[0])
            # utils.save_3d_array('data/DEBUG', (iota[0][1:, 1:, 1:] - (-np.conj(np.flip(iota[0][1:, 1:, 1:])))))
        
        print('============================================================')        
        print(f'Random {axis} Phase Information')
        print('------------------------------------------------------------')
        print(f'Random iota {axis} shape: {iota[0].shape}')
        print(f'Random beta {axis} shape: {beta[0].shape}')
        print(f'Random {axis} phase shape: {B_phase_k[0].shape}')
        print(f'Some random iota {axis} values: {iota[0][0:3, 0:3, 0:3]}')
        print(f'Some random beta {axis} values: {beta[0][0:3, 0:3, 0:3]}')
        print(f'Some random {axis} phase values: {B_phase_k[0][0:3, 0:3, 0:3]}')
        print('------------------------------------------------------------')

        # In case one needs cut the code to check the Hermitian properties of the random numbers when debugging
        assert np.allclose(iota[0][1:, 1:, 1:], -np.conj(np.flip(iota[0][1:, 1:, 1:]))), f"Debugging Test IV Failed: iota grid {axis} is not Hermitian."
        print(f"Debugging Test IV Passed: iota grid {axis} is Hermitian.")
        assert np.allclose(beta[0][1:, 1:, 1:], np.conj(np.flip(beta[0][1:, 1:, 1:]))), f"Debugging Test V Failed: beta grid {axis} is not Hermitian."
        print(f"Debugging Test V Passed: beta grid {axis} is Hermitian.")
        assert np.allclose(B_phase_k[0][1:, 1:, 1:], np.conj(np.flip(B_phase_k[0][1:, 1:, 1:]))), f"Debugging Test X Failed: k{axis} B phase is not Hermitian."
        print(f"Debugging Test VI Passed: k{axis} B phase is Hermitian.")
        print('============================================================')
        
    return B_phase_k

def generate_random_seed_amplitudes(axis, k_grid_axis, k_magnitude, P_B, B_phase_axis, size, N, verbose = False, debug = False):
    '''
    Generates a random magnetic field seed in Fourier space with a chosen spectral index and filtered amplitude that can be
    used to generate a cosmological magnetic field seed in real space using the inverse numpy Fast Fourier Transform.
    
    Args:
        - axis: axis of the random magnetic field seed amplitude needed to be generated. The axis must be x, y, or z
        - k_grid_axis: wave vector k coordinates axis component for each combination of kx, ky, and kz
        - k_magnitude: magnitude of the wave vector k for each combination of kx, ky, and kz
        - P_B: amplitude of the power spectrum of the magnetic field seed
        - B_phase_axis: random magnetic field seed phase in Fourier space in the given direction
        - size: size of the box in Mpc
        - N: number of cells in each direction
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
        
    Returns:
        - B_random_k: random magnetic field components in Fourier space in the given direction
        
    Source: Vazza, F., Paoletti, D., Banfi, S., Finelli, F., Gheller, C., O’Sullivan, S. P., & Brüggen, M. (2021).
            Simulations and observational tests of primordial magnetic fields from Cosmic Microwave Background constraints.
            Monthly Notices of the Royal Astronomical Society, 500(4), 5350–5368. https://doi.org/10.1093/mnras/staa3532
            
    Author: Marco Molina
    '''
        
    if axis not in ['x', 'y', 'z']:
        raise ValueError('The axis must be x, y, or z.')
    
    nmax, nmay, nmaz = N
    
    # Generate the random magnetic field components in Fourier space by agregating the phase and amplitude
    B_random_k = [np.sqrt(((size**3) * ((2 * np.pi)**3) * P_B[p] * (1 - (k_grid_axis/k_magnitude[p])**2))/2) * B_phase_axis[p] for p in range(sum(npatch)+1)]
    
    ##Revisar## Proyección transversal en la magnitud
    # B_random_k = [np.sqrt(((size**3) * ((2 * np.pi)**3) * P_B[p])/2) * B_phase_axis[p] for p in range(sum(npatch)+1)]
    
    # We need to habdle the null frequency after applying the magnitude proyection or the signal would tend to infity.
    if k_magnitude[0][0,0,0] == 0:
        B_random_k[0][0,0,0] = 0.
    
    if verbose:
        print('============================================================')
        print(f'Random Magnetic Field Seed {axis} Amplitude completed')
        print('------------------------------------------------------------')
        print(f'Random magnetic field {axis} component shape: {B_random_k[0].shape}')
        print('============================================================')
    
    # Save some arrays to check the values if needed
    if debug:
        
        # Generate the random magnetic field components amplitude in Fourier space    
        B_k_mod_squared = [(size**3) * ((2 * np.pi)**3) * P_B[p] * (1 - (k_grid_axis/k_magnitude[p])**2) for p in range(sum(npatch)+1)]
        # We need to habdle the null frequency after applying the magnitude proyection or the signal would tend to infity.
        B_k_mod_squared[0][0,0,0] = 0.
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:            
            utils.save_3d_array('data/P_B', P_B[0])
            utils.save_3d_array(f'data/1-(k_hat k_hat div k_squared)_{axis}', (1 - (k_grid_axis/k_magnitude[0])**2))
            utils.save_3d_array(f'data/B_k{axis}_mod_squared', B_k_mod_squared[0])
            utils.save_3d_array(f'data/B_random_k{axis}', B_random_k[0])
        
        print('============================================================')
        print(f'Random Magnetic Field Seed {axis} Amplitude Information')
        print('------------------------------------------------------------')
        print(f'Random magnetic field {axis} component shape: {B_random_k[0].shape}')
        print(f'Power Spectrum shape: {P_B[0].shape}')
        print(f'Some power spectrum values: {P_B[0][0:3, 0:3, 0:3]}')
        print('------------------------------------------------------------')
        assert np.allclose(P_B[0][1:, 1:, 1:], np.conj(np.flip(P_B[0][1:, 1:, 1:]))), f"Debugging Test VII Failed: P_B is not Hermitian."
        print(f"Debugging Test VII Passed: P_B is Hermitian.")
        assert np.allclose((1 - (k_grid_axis/k_magnitude[0])**2)[1:, 1:, 1:], np.conj(np.flip((1 - (k_grid_axis/k_magnitude[0])**2)[1:, 1:, 1:]))), f"Debugging Test VIII Failed: 1 - (k_hat k_hat div k_squared)_{axis} is not Hermitian."
        print(f"Debugging Test VIII Passed: 1 - (k_hat k_hat div k_squared)_{axis} is Hermitian.")
        assert np.allclose(B_k_mod_squared[0][1:, 1:, 1:], np.conj(np.flip(B_k_mod_squared[0][1:, 1:, 1:]))), f"Debugging Test IX Failed: B k{axis} module squared is not Hermitian."
        print(f"Debugging Test IX Passed: B k{axis} module squared is Hermitian.")
        assert np.allclose(B_random_k[0][1:, 1:, 1:], np.conj(np.flip(B_random_k[0][1:, 1:, 1:]))), f"Debugging Test X Failed: B random k{axis} is not Hermitian."
        print(f"Debugging Test X Passed: B random k{axis} is Hermitian.")
        print('============================================================')
    
    return B_random_k

def seed_transverse_projector(k_grid, B_random_K, N, verbose = False, debug = False):
    '''
    Computes the transverse projection of the magnetic field to ensure the null divergence.
    
    Args:
        - k_grid: wave vector k coordinates for each combination of kx, ky, and kz
        - B_random_K: random magnetic field component in Fourier space in the three directions
        - N: number of cells in each direction
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
        
    Returns:
        - k_dot_B: transverse projectior of the magnetic field to ensure the null divergence
            
    Author: Marco Molina
    '''
    
    nmax, nmay, nmaz = N
    
    B_random_kx = B_random_K[0]
    B_random_ky = B_random_K[1]
    B_random_kz = B_random_K[2]
    
    # Compute the transverse projection of the magnetic field to ensure the null divergence
    k_dot_B = [k_grid[p][0]**2 + k_grid[p][1]**2 + k_grid[p][2]**2 for p in range(sum(npatch)+1)] 
    
    k_dot_B = [(k_grid[p][0] * B_random_kx[p] + k_grid[p][1] * B_random_ky[p] + k_grid[p][2] * B_random_kz[p])/(k_dot_B[p]) for p in range(sum(npatch)+1)]

    # We need to habdle the null frequency after applying the magnitude proyection or the signal would tend to infity.
    if (k_grid[0][0]**2 + k_grid[0][1]**2 + k_grid[0][2]**2)[0,0,0] == 0:
        k_dot_B[0][0,0,0] = 0.

    if verbose:
        print('============================================================')
        print('Transverse Projection completed')
        print('------------------------------------------------------------')
        print(f'k_dot_B amplitude shape: {k_dot_B[0].shape}')
        print('============================================================')
        
    if debug:
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array('data/k_dot_B', k_dot_B[0])
        
        print('============================================================')
        print('Transverse Projection Information')
        print('------------------------------------------------------------')
        print(f'k_dot_B shape: {k_dot_B[0].shape}')
        print(f'Some k_dot_B values: {k_dot_B[0][0:3, 0:3, 0:3]}')
        print('============================================================')

    return k_dot_B

def generate_seed_magnetic_field(axis, k_grid_axis, B_random_k, k_dot_B, N, verbose = False, debug = False):
    '''
    Generates a random magnetic field seed in Fourier space with a chosen spectral index and filtered amplitude that can be
    used to generate a cosmological magnetic field seed in real space using the inverse numpy Fast Fourier Transform.
    
    Args:
        - axis: axis of the random magnetic field seed amplitude needed to be generated. The axis must be x, y, or z
        - k_grid_axis: wave vector k coordinates axis component for each combination of kx, ky, and kz
        - B_random_k: random magnetic field component in Fourier space in the three directions for the given axis
        - k_dot_B: transverse projectior of the magnetic field to ensure the null divergence
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not

    Returns:
        - B: random magnetic field component in Fourier space in the given direction
        
    Source: Vazza, F., Paoletti, D., Banfi, S., Finelli, F., Gheller, C., O’Sullivan, S. P., & Brüggen, M. (2021).
            Simulations and observational tests of primordial magnetic fields from Cosmic Microwave Background constraints.
            Monthly Notices of the Royal Astronomical Society, 500(4), 5350–5368. https://doi.org/10.1093/mnras/staa3532
            Phase theoretical treatment based on Vicent Quilis' procedure.
            
    Author: Marco Molina
    '''

    if axis not in ['x', 'y', 'z']:
        raise ValueError('The axis must be x, y, or z.')
    
    nmax, nmay, nmaz = N
    
    # Generate the random magnetic field seed phase in Fourier space
    Bk = [B_random_k[p] - (k_grid_axis * k_dot_B[p]) for p in range(sum(npatch)+1)]
    
    if verbose:
        print('============================================================')
        print(f'Magnetic Field Fourier {axis} Amplitude completed')
        print('------------------------------------------------------------')
        print(f'B_k{axis} amplitude shape: {Bk[0].shape}')
        print('============================================================')
    
    if debug:
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array(f'data/B_k{axis}', Bk[0])
    
        print('============================================================')
        print(f'Magnetic Field Fourier {axis} Amplitude Information')
        print('------------------------------------------------------------')
        print(f'B_k{axis} amplitude shape: {Bk[0].shape}')
        print(f'Some magnetic field {axis} Fourier amplitude values: {Bk[0][0:3, 0:3, 0:3]}')
        print('------------------------------------------------------------')
        for i in range(nmax):
            for j in range(nmay):
                for k in range(nmaz//2 + 1):

                    iconj = -i % Bk[0].shape[0]
                    jconj = -j % Bk[0].shape[1]
                    kconj = -k % Bk[0].shape[2]
                    
                    if Bk[0][i, j, k] != np.conj(Bk[0][iconj, jconj, kconj]):
                        print(f"Indices: ({i}, {j}, {k}) -> ({iconj}, {jconj}, {kconj})")
                        print(f"Value at ({i}, {j}, {k}): {Bk[0][i, j, k]}")
                        print(f"Value at ({iconj}, {jconj}, {kconj}): {Bk[0][iconj, jconj, kconj]}")
                    
                    assert Bk[0][i, j, k] == np.conj(Bk[0][iconj, jconj, kconj]), f"Debugging Test XI Failed: B_k{axis} is not Hermitian."
        print(f"Debugging Test XI Passed: B_k{axis} is Hermitian.")
        print('============================================================')
        
    return Bk

def transform_seed_magnetic_field(axis, B, alpha_index, size, N, gauss_rad_factor = 1,
                                memmap = False, verbose = False, debug = False, run = 'no_name'):
    '''
    Generates a random magnetic field seed in Fourier space with a chosen spectral index and filtered amplitude that can be
    used to generate a cosmological magnetic field seed in real space using the inverse numpy Fast Fourier Transform.
    
    Args:
        - axis: axis of the random magnetic field seed amplitude needed to be generated. The axis must be x, y, or z
        - B: random magnetic field component in Fourier space in the given direction
        - alpha_index: spectral index of the magnetic field
        - size: size of the box in Mpc
        - N: number of cells in each direction
        - gauss_rad_factor: factor to multiply the Gaussian filtering radius
        - memmap: boolean to decide if the data is processed with memory cautions or not
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
        - run: name of the run
        
    Returns:
        - B: random magnetic field component in real space in the given direction
        
    Source: Vazza, F., Paoletti, D., Banfi, S., Finelli, F., Gheller, C., O’Sullivan, S. P., & Brüggen, M. (2021).
            Simulations and observational tests of primordial magnetic fields from Cosmic Microwave Background constraints.
            Monthly Notices of the Royal Astronomical Society, 500(4), 5350–5368. https://doi.org/10.1093/mnras/staa3532
            Phase theoretical treatment based on Vicent Quilis' procedure.
            
    Author: Marco Molina
    '''

    if axis not in ['x', 'y', 'z']:
        raise ValueError('The axis must be x, y, or z.')
    
    nmax, nmay, nmaz = N
    dx = size/nmax

    # Get the magnetic field components in real space
    if memmap:
        B = [np.fft.ifftn(B[p]) for p in range(sum(npatch)+1)]
        B = [np.real(B[p] / B[p].size) for p in range(sum(npatch)+1)]
    else:
        B = [np.fft.ifftn(B[p]) for p in range(sum(npatch)+1)]
        
        if debug == False:
            B = [np.real(B[p] / B[p].size) for p in range(sum(npatch)+1)]
    
    if verbose:
        
        print('============================================================')
        print(f'Magnetic {axis} Field Real Space Information')
        print('------------------------------------------------------------')
        print(f'B_{axis} component shape: {B[0].shape}')
        print(f'Some magnetic {axis} field values: {B[0][0:3, 0:3, 0:3]}')
        print('============================================================')
        
        depth = 5
        fact = 5
        col = 'red'
        
        imdim = np.round((fact+gauss_rad_factor)/dx, 0).astype(int)
        section = np.sum(B[0][(B[0].shape[0]//2 - imdim):(B[0].shape[0]//2 + imdim), (B[0].shape[1]//2 - imdim):(B[0].shape[1]//2 + imdim), (B[0].shape[2]//2 - depth//2):(B[0].shape[2] + depth//2)], axis=2)
        plt.imshow(section, cmap='viridis')
        plt.title(f'Gaussian Filtered Magnetic {axis}-field, $\\alpha$ = {alpha_index}')
        ctoMpc = gauss_rad_factor/dx
        plt.arrow(imdim, imdim, ctoMpc-(ctoMpc/7), 0, head_width=(ctoMpc/14), head_length=(ctoMpc/7), fc=col, ec=col)
        plt.text(imdim, imdim-gauss_rad_factor, f'{gauss_rad_factor} Mpc', color=col)
        
        name = f'B{axis}_{run}_{nmax}_{size}_{alpha_index}'
        plt.savefig(os.path.join('data/', f'{name}.png'))

    if debug:
        
        B = [np.real(B[p] / (B[p].size)) for p in range(sum(npatch)+1)]
        
        if nmax <= 32 and nmay <= 32 and nmaz <= 32:
            utils.save_3d_array(f'data/B_k{axis}', B[0])
        
        assert np.isreal(B[0]).all(), f"Debugging Test XII Failed: B_{axis} is not real."
        print(f"Debugging Test XII Passed: B_{axis} is real.")
        print('============================================================')
        
    return B

def process_seed_chunk(chunk_size, i, j, k, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS):
    '''
    Process a chunk of the magnetic field seed generation.
    
    Args:
        - chunk_size: size of the chunk in each direction
        - i: index of the chunk in the x direction
        - j: index of the chunk in the y direction
        - k: index of the chunk in the z direction
        - Kx: wave vector k coordinates for the x direction
        - Ky: wave vector k coordinates for the y direction
        - Kz: wave vector k coordinates for the z direction
        - SEED_PARAMS: dictionary with the parameters for the seed generation
        - OUT_PARAMS: dictionary with the parameters for the output
        
    Returns:
        - B_chunk: magnetic field chunk in Fourier space in the given direction
        
    Author: Marco Molina
    '''
    
    chunk_size_x, chunk_size_y, chunk_size_z = chunk_size
    
    i_final = i + chunk_size_x
    j_final = j + chunk_size_y
    k_final = k + chunk_size_z
    
    kx = Kx[i:i_final]
    ky = Ky[j:j_final]
    kz = Kz[k:k_final] 
    
    k_grid, k_magnitude = generate_fourier_space(
        [kx, ky, kz], [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        SEED_PARAMS["epsilon"], verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    PB = power_spectrum_amplitude(
        k_magnitude, SEED_PARAMS["alpha"], SEED_PARAMS["lambda_comoving"], SEED_PARAMS["B0"], h, 
        SEED_PARAMS["size"], [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]], 
        gauss_rad_factor=SEED_PARAMS["smothing"], filtering=SEED_PARAMS["filtering"], verbose=OUT_PARAMS["verbose"]
        )
    
    iota = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    beta = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    
    B_random_phase = generate_seed_phase(
        'x', iota, beta, [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    B_random_kx = generate_random_seed_amplitudes(
        'x', k_grid[0][0], k_magnitude, PB, B_random_phase,
        SEED_PARAMS["size"], [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    iota = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    beta = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    
    B_random_phase = generate_seed_phase(
        'y', iota, beta, [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    B_random_ky = generate_random_seed_amplitudes(
        'y', k_grid[0][1], k_magnitude, PB, B_random_phase,
        SEED_PARAMS["size"], [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    iota = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    beta = [np.random.uniform(0, 1, (chunk_size_x, chunk_size_y, chunk_size_z)) for _ in range(sum(npatch)+1)]
    
    B_random_phase = generate_seed_phase(
        'z', iota, beta, [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    B_random_kz = generate_random_seed_amplitudes(
        'z', k_grid[0][2], k_magnitude, PB, B_random_phase,
        SEED_PARAMS["size"], [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    k_dot_B = seed_transverse_projector(
        k_grid, [B_random_kx, B_random_ky, B_random_kz],
        [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
    
    B_chunk_x = generate_seed_magnetic_field(
        'x', k_grid[0][0], B_random_kx, k_dot_B,
        [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )

    B_chunk_y = generate_seed_magnetic_field(
        'y', k_grid[0][1], B_random_ky, k_dot_B,
        [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )

    B_chunk_z = generate_seed_magnetic_field(
        'z', k_grid[0][2], B_random_kz, k_dot_B,
        [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
        verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"]
        )
        
    return B_chunk_x, B_chunk_y, B_chunk_z

def generate_seed(chunk_factor, SEED_PARAMS, OUT_PARAMS):
    '''
    Generates the magnetic field seed in the given direction.

    Args:
        - chunk_factor: factor to divide the array in chunks
        - SEED_PARAMS: dictionary with the parameters for the seed generation
        - OUT_PARAMS: dictionary with the parameters for the output

    Returns:
        - B: magnetic field component in the given direction
        
    Author: Marco Molina
    '''
    
    nmax, nmay, nmaz = SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]
    size = SEED_PARAMS["size"]
    dx = size/nmax
    
    chunk_size = min(nmax, nmay, nmaz) // chunk_factor

    # Ensure chunk_size is a multiple of every size and a divisor of 2
    while any(cell_size % chunk_size != 0 for cell_size in (nmax, nmay, nmaz)) or (chunk_size & (chunk_size - 1)) != 0:
        chunk_size -= 1
    if chunk_size == 1 and OUT_PARAMS["memmap"]:
        print('Chunks are only 1 cell in size. Caution, the chunk factor is too high.')
    
    debugChunk = False
    if (nmax != nmay or nmax != nmaz or nmay != nmaz or nmax != chunk_size) and OUT_PARAMS["debug"] == True:
        OUT_PARAMS["debug"] = False
        debugChunk = True
        print('Internal debugging will not take place if the array is not cubic or if it is chunked.')

    chunk_size = (chunk_size, chunk_size, chunk_size)
    
    if OUT_PARAMS["memmap"]:
        # Create memory-mapped file for B
        Bx = [np.memmap(f'data/_B_kx_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='w+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]
        Bx[0][0, 0, 0] = 0.
        By = [np.memmap(f'data/_B_ky_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='w+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]
        By[0][0, 0, 0] = 0.
        Bz = [np.memmap(f'data/_B_kz_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='w+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]
        Bz[0][0, 0, 0] = 0.
    else:
        Bx = [np.zeros((nmax+1, nmay+1, nmaz+1), dtype=OUT_PARAMS["complex_bitformat"]) for _ in range(sum(npatch)+1)]
        By = [np.zeros((nmax+1, nmay+1, nmaz+1), dtype=OUT_PARAMS["complex_bitformat"]) for _ in range(sum(npatch)+1)]
        Bz = [np.zeros((nmax+1, nmay+1, nmaz+1), dtype=OUT_PARAMS["complex_bitformat"]) for _ in range(sum(npatch)+1)]

    # Calculate the wave numbers for a Fourier transform
    Kx = np.fft.fftfreq(nmax, dx) * 2 * np.pi
    Ky = np.fft.fftfreq(nmay, dx) * 2 * np.pi
    Kz = np.fft.fftfreq(nmaz, dx) * 2 * np.pi
    
    ##Revisar## Factor 2*pi
    # Kx = np.fft.fftfreq(nmax, dx)
    # Ky = np.fft.fftfreq(nmay, dx)
    # Kz = np.fft.fftfreq(nmaz, dx)
    
    # Here we add the positive Nyquist frequency just before the negative Nyquist frequency already present
    Kx = np.insert(Kx, nmax//2, -Kx[nmax//2])
    Ky = np.insert(Ky, nmay//2, -Ky[nmay//2])
    Kz = np.insert(Kz, nmaz//2, -Kz[nmaz//2])

    # In case the memory is not enough to generate the magnetic field in one go, we generate it in chunks
    nchunk = 1
    if OUT_PARAMS["memmap"]:
        for i in range(1, nmax+1, chunk_size[0]):
            for j in range(1, nmay+1, chunk_size[1]):
                for k in range(1, nmaz//2 + 1, chunk_size[2]):
                    
                    if OUT_PARAMS["verbose"]:
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        print(f'Processing Chunk {nchunk} of {(nmax//chunk_size[0])*(nmay//chunk_size[1])*((nmaz//chunk_size[2])//2)}')
                        print(f'Chunk Size: {chunk_size}')
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                
                    B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, i, j, k, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
                    
                    i_final = i + chunk_size[0] - 1
                    j_final = j + chunk_size[1] - 1
                    k_final = k + chunk_size[2] - 1
                
                    Bx[0][i:i_final+1, j:j_final+1, k:k_final+1] = B_chunk_x[0]
                    By[0][i:i_final+1, j:j_final+1, k:k_final+1] = B_chunk_y[0]
                    Bz[0][i:i_final+1, j:j_final+1, k:k_final+1] = B_chunk_z[0]

                    iconj = -i % Bx[0].shape[0]
                    jconj = -j % Bx[0].shape[1]
                    kconj = -k % Bx[0].shape[2]
                    
                    iconj_final = -i_final % Bx[0].shape[0]
                    jconj_final = -j_final % Bx[0].shape[1]
                    kconj_final = -k_final % Bx[0].shape[2]
                    
                    Bx[0][iconj_final:iconj+1, jconj_final:jconj+1, kconj_final:kconj+1] = np.flip(np.flip(np.flip(np.conj(B_chunk_x[0]), axis=0), axis=1), axis=2)
                    By[0][iconj_final:iconj+1, jconj_final:jconj+1, kconj_final:kconj+1] = np.flip(np.flip(np.flip(np.conj(B_chunk_y[0]), axis=0), axis=1), axis=2)
                    Bz[0][iconj_final:iconj+1, jconj_final:jconj+1, kconj_final:kconj+1] = np.flip(np.flip(np.flip(np.conj(B_chunk_z[0]), axis=0), axis=1), axis=2)
                    
                    if OUT_PARAMS["debug"] or debugChunk:
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        print(f'Chunk {nchunk} of {(nmax//chunk_size[0])*(nmay//chunk_size[1])*((nmaz//chunk_size[2])//2)} completed')
                        print(f'Chunk Inicial Indices:           {i}, {j}, {k}')
                        print(f'Chunk Final Indices:             {i_final}, {j_final}, {k_final}')
                        print(f'Chunk 3D Range:           ({i}...{i_final+1}, {j}...{j_final+1}, {k}...{k_final+1})')
                        print(f'Chunk Final Conjugate Indices:   {iconj_final}, {jconj_final}, {kconj_final}')
                        print(f'Chunk Inicial Conjugate Indices: {iconj}, {jconj}, {kconj}')
                        print(f'Chunk Conjugate 3D Range: ({iconj_final}...{iconj+1}, {jconj_final}...{jconj+1}, {kconj_final}...{kconj+1})')
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    
                    nchunk += 1
        
        if OUT_PARAMS["verbose"]:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'Processing Null Axes')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        chunk_size = (nmax//2, 1, 1)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 1, 0, 0, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][1:(nmax//2 + 1), 0, 0] = B_chunk_x[0][:, 0, 0]
        Bx[0][(nmax//2 + 1):, 0, 0] = np.flip(np.conj(B_chunk_x[0][:, 0, 0]), axis=0)
        By[0][1:(nmax//2 + 1), 0, 0] = B_chunk_y[0][:, 0, 0]
        By[0][(nmax//2 + 1):, 0, 0] = np.flip(np.conj(B_chunk_y[0][:, 0, 0]), axis=0)
        Bz[0][1:(nmax//2 + 1), 0, 0] = B_chunk_z[0][:, 0, 0]
        Bz[0][(nmax//2 + 1):, 0, 0] = np.flip(np.conj(B_chunk_z[0][:, 0, 0]), axis=0)
        
        if OUT_PARAMS["verbose"]:
            print('X Axis completed')
        
        chunk_size = (1, nmax//2, 1)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 0, 1, 0, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][0, 1:(nmay//2 + 1), 0] = B_chunk_x[0][0, :, 0]
        Bx[0][0, (nmay//2 + 1):, 0] = np.flip(np.conj(B_chunk_x[0][0, :, 0]), axis=0)
        By[0][0, 1:(nmay//2 + 1), 0] = B_chunk_y[0][0, :, 0]
        By[0][0, (nmay//2 + 1):, 0] = np.flip(np.conj(B_chunk_y[0][0, :, 0]), axis=0)
        Bz[0][0, 1:(nmay//2 + 1), 0] = B_chunk_z[0][0, :, 0]
        Bz[0][0, (nmay//2 + 1):, 0] = np.flip(np.conj(B_chunk_z[0][0, :, 0]), axis=0)
        
        if OUT_PARAMS["verbose"]:
            print('Y Axis completed')
        
        chunk_size = (1, 1, nmax//2)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 0, 0, 1, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][0, 0, 1:(nmaz//2 + 1)] = B_chunk_x[0][0, 0, :]
        Bx[0][0, 0, (nmaz//2 + 1):] = np.flip(np.conj(B_chunk_x[0][0, 0, :]), axis=0)
        By[0][0, 0, 1:(nmaz//2 + 1)] = B_chunk_y[0][0, 0, :]
        By[0][0, 0, (nmaz//2 + 1):] = np.flip(np.conj(B_chunk_y[0][0, 0, :]), axis=0)
        Bz[0][0, 0, 1:(nmaz//2 + 1)] = B_chunk_z[0][0, 0, :]
        Bz[0][0, 0, (nmaz//2 + 1):] = np.flip(np.conj(B_chunk_z[0][0, 0, :]), axis=0)
        
        if OUT_PARAMS["debug"] or debugChunk:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'DEBUG Bxx: {Bx[0][:, 0, 0]}')
            print(f'DEBUG Bxy: {Bx[0][0, :, 0]}')
            print(f'DEBUG Bxz: {Bx[0][0, 0, :]}')
            print(f'DEBUG Byx: {By[0][:, 0, 0]}')
            print(f'DEBUG Byy: {By[0][0, :, 0]}')
            print(f'DEBUG Byz: {By[0][0, 0, :]}')
            print(f'DEBUG Bzx: {Bz[0][:, 0, 0]}')
            print(f'DEBUG Bzy: {Bz[0][0, :, 0]}')
            print(f'DEBUG Bzz: {Bz[0][0, 0, :]}')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            
        if OUT_PARAMS["verbose"]:
            print('Z Axis completed')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'Processing Null Planes')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        chunk_size = (nmax, nmay//2, 1)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 1, 1, 0, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][1:, 1:(nmay//2 + 1), 0] = B_chunk_x[0][:, :, 0]
        Bx[0][1:, (nmay//2 + 1):, 0] = np.flip(np.flip(np.conj(B_chunk_x[0][:, :, 0]), axis=0), axis=1)
        By[0][1:, 1:(nmay//2 + 1), 0] = B_chunk_y[0][:, :, 0]
        By[0][1:, (nmay//2 + 1):, 0] = np.flip(np.flip(np.conj(B_chunk_y[0][:, :, 0]), axis=0), axis=1)
        Bz[0][1:, 1:(nmay//2 + 1), 0] = B_chunk_z[0][:, :, 0]
        Bz[0][1:, (nmay//2 + 1):, 0] = np.flip(np.flip(np.conj(B_chunk_z[0][:, :, 0]), axis=0), axis=1)
        
        if OUT_PARAMS["verbose"]:
            print('XY Null Plane completed')
        
        chunk_size = (nmax, 1, nmaz//2)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 1, 0, 1, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][1:, 0, 1:(nmaz//2 + 1)] = B_chunk_x[0][:, 0, :]
        Bx[0][1:, 0, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_x[0][:, 0, :]), axis=0), axis=1)
        By[0][1:, 0, 1:(nmaz//2 + 1)] = B_chunk_y[0][:, 0, :]
        By[0][1:, 0, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_y[0][:, 0, :]), axis=0), axis=1)
        Bz[0][1:, 0, 1:(nmaz//2 + 1)] = B_chunk_z[0][:, 0, :]
        Bz[0][1:, 0, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_z[0][:, 0, :]), axis=0), axis=1)
        
        if OUT_PARAMS["verbose"]:
            print('XZ Null Plane completed')
        
        chunk_size = (1, nmay, nmaz//2)
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk(chunk_size, 0, 1, 1, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0][0, 1:, 1:(nmaz//2 + 1)] = B_chunk_x[0][0, :, :]
        Bx[0][0, 1:, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_x[0][0, :, :]), axis=0), axis=1)
        By[0][0, 1:, 1:(nmaz//2 + 1)] = B_chunk_y[0][0, :, :]
        By[0][0, 1:, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_y[0][0, :, :]), axis=0), axis=1)
        Bz[0][0, 1:, 1:(nmaz//2 + 1)] = B_chunk_z[0][0, :, :]
        Bz[0][0, 1:, (nmaz//2 + 1):] = np.flip(np.flip(np.conj(B_chunk_z[0][0, :, :]), axis=0), axis=1)
        
        if OUT_PARAMS["debug"] or debugChunk:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'DEBUG BxXY:')
            print(f'{Bx[0][:, :, 0]}')
            print(f'DEBUG ByXY:')
            print(f'{By[0][:, :, 0]}')
            print(f'DEBUG BzXY:')
            print(f'{Bz[0][:, :, 0]}')
            print(f'DEBUG BxXZ:')
            print(f'{Bx[0][:, 0, :]}')
            print(f'DEBUG ByXZ:')
            print(f'{By[0][:, 0, :]}')
            print(f'DEBUG BzXZ:')
            print(f'{Bz[0][:, 0, :]}')
            print(f'DEBUG BxYZ:')
            print(f'{Bx[0][0, :, :]}')
            print(f'DEBUG ByYZ:')
            print(f'{By[0][0, :, :]}')
            print(f'DEBUG BzYZ:')
            print(f'{Bz[0][0, :, :]}')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        if OUT_PARAMS["verbose"]:
            print('YZ Null Plane completed')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'Chunking Completed')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
    else:
        B_chunk_x, B_chunk_y, B_chunk_z = process_seed_chunk((nmax+1, nmay+1, nmaz+1), 0, 0, 0, Kx, Ky, Kz, SEED_PARAMS, OUT_PARAMS)
        Bx[0] = B_chunk_x[0]
        Bx[0][0, 0, 0] = 0.
        By[0] = B_chunk_y[0]
        By[0][0, 0, 0] = 0.
        Bz[0] = B_chunk_z[0]
        Bz[0][0, 0, 0] = 0.
        
    #DEBUG# B = [np.memmap(f'data/_B_kx_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]
    #DEBUG# B = [np.memmap(f'data/_B_ky_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]
    #DEBUG# B = [np.memmap(f'data/_B_kz_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax+1, nmay+1, nmaz+1)) for _ in range(sum(npatch)+1)]

    # After handling the transverse proyection and building the entire magnetic field in Fourier space, 
    # we can finally merge the explicitly separated positive and negative Nyquist frequencies
    Bx = [merge_nyquist(Bx[p], axis='x', memmap=OUT_PARAMS["memmap"], complex_bitformat = OUT_PARAMS["complex_bitformat"]) for p in range(sum(npatch)+1)]
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_B_kx_temporary_file_.dat')
    By = [merge_nyquist(By[p], axis='y', memmap=OUT_PARAMS["memmap"], complex_bitformat = OUT_PARAMS["complex_bitformat"]) for p in range(sum(npatch)+1)]
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_B_ky_temporary_file_.dat')
    Bz = [merge_nyquist(Bz[p], axis='z', memmap=OUT_PARAMS["memmap"], complex_bitformat = OUT_PARAMS["complex_bitformat"]) for p in range(sum(npatch)+1)]
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_B_kz_temporary_file_.dat')
    
    if OUT_PARAMS["verbose"]:
        print('============================================================')
        print(f'Nyquist Frequencies Merging Completed')
        print('============================================================')
    
    #DEBUG# B = [np.memmap(f'data/_Imaginary_B_x_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax, nmay, nmaz//2 + 1)) for _ in range(sum(npatch)+1)]
    #DEBUG# B = [np.memmap(f'data/_Imaginary_B_y_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax, nmay, nmaz//2 + 1)) for _ in range(sum(npatch)+1)]
    #DEBUG# B = [np.memmap(f'data/_Imaginary_B_z_temporary_file_.dat', dtype=OUT_PARAMS["complex_bitformat"], mode='r+', shape=(nmax, nmay, nmaz//2 + 1)) for _ in range(sum(npatch)+1)]

    if OUT_PARAMS["transform"]:
        Bx = transform_seed_magnetic_field(
            'x', Bx, SEED_PARAMS["alpha"], SEED_PARAMS["size"],
            [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
            gauss_rad_factor=SEED_PARAMS["smothing"], memmap=OUT_PARAMS["memmap"],
            verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"], run=OUT_PARAMS["run"]
            )
    if OUT_PARAMS["save"]:
        utils.save_magnetic_field_seed(Bx, 'x', OUT_PARAMS["transform"], OUT_PARAMS["format"], OUT_PARAMS["run"])
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_Imaginary_B_x_temporary_file_.dat')
        
    if OUT_PARAMS["transform"]:
        By = transform_seed_magnetic_field(
            'y', By, SEED_PARAMS["alpha"], SEED_PARAMS["size"],
            [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
            gauss_rad_factor=SEED_PARAMS["smothing"], memmap=OUT_PARAMS["memmap"],
            verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"], run=OUT_PARAMS["run"]
            )
    if OUT_PARAMS["save"]:
        utils.save_magnetic_field_seed(By, 'y', OUT_PARAMS["transform"], OUT_PARAMS["format"], OUT_PARAMS["run"])
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_Imaginary_B_y_temporary_file_.dat')
        
    if OUT_PARAMS["transform"]:
        Bz = transform_seed_magnetic_field(
            'z', Bz, SEED_PARAMS["alpha"], SEED_PARAMS["size"],
            [SEED_PARAMS["nmax"], SEED_PARAMS["nmay"], SEED_PARAMS["nmaz"]],
            gauss_rad_factor=SEED_PARAMS["smothing"], memmap=OUT_PARAMS["memmap"],
            verbose=OUT_PARAMS["verbose"], debug=OUT_PARAMS["debug"], run=OUT_PARAMS["run"]
            )
    if OUT_PARAMS["save"]:
        utils.save_magnetic_field_seed(Bz, 'z', OUT_PARAMS["transform"], OUT_PARAMS["format"], OUT_PARAMS["run"])
    if OUT_PARAMS["memmap"]:
        os.remove(f'data/_Imaginary_B_z_temporary_file_.dat')
    
    return Bx, By, Bz


def generate_seed_properties(Bx, By, Bz, alpha_index, size, N, verbose = False):
    '''
    Computes the magnitude and the divergence of the magnetic field.
    
    Args:
        - Bx: magnetic field component in the x direction
        - By: magnetic field component in the y direction
        - Bz: magnetic field component in the z direction
        - alpha_index: spectral index of the magnetic field
        - size: size of the box in Mpc
        - N: number of cells in each direction
        - verbose: boolean to print the parameters or not
        - debug: boolean to print debugging information or not
    
    Returns:
        - Bmag: magnitude of the magnetic field
        - diver_B: magnetic field divergence
        
    Author: Marco Molina
    '''
    
    nmax, nmay, nmaz = N
    dx = size/nmax
    
    # Compute the magnetic field magnitude
    Bmag = np.sqrt(Bx[0]**2 + By[0]**2 + Bz[0]**2)
    
    # Compute the magnetic field seed divergence
    diver_B = diff.periodic_divergence(Bx, By, Bz, dx, npatch = np.array([0]), stencil=5, kept_patches=None)
    
    if verbose:
        print('============================================================')
        print(f'Spectral Index: {alpha_index}')
        print('------------------------------------------------------------')
        print(f'Max field: {np.max(Bmag)}')
        print(f'Min field: {np.min(Bmag)}')
        print(f'Average field: {np.mean(Bmag)}')
        print('============================================================')
        print('Magnetic Field Divergence Information')
        print('------------------------------------------------------------')
        print(f'Magnetic field divergence shape: {diver_B[0].shape}')
        print(f'Some magnetic field divergence values: {diver_B[0][0:3, 0:3, 0:3]}')
        print('------------------------------------------------------------')
        print(f"Max absolute divergence: {np.max(np.abs(diver_B[0]))}")
        print(f"Min absolute divergence: {np.min(np.abs(diver_B[0]))}")
        print(f"Average absolute divergence: {np.mean(np.abs(diver_B[0]))}")
        print(f"Average divergence: {np.mean(diver_B[0])}")
        print(f"Dispersion divergence: {np.std(diver_B[0])}")
        print('============================================================')
    
    return Bmag, diver_B