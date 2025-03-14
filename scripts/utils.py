"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

utils module
Contains utility tool functions used by the PRIMAL Seed Generator modules.

Created by Marco Molina Pradillo
"""

import numpy as np
import sys
import os
from scipy.io import FortranFile
import h5py
import psutil
from config import SEED_PARAMS as seed_params
from config import OUTPUT_PARAMS as out_params

def check_memory():
    '''
    Checks if the arrays can fit in memory or if an alternative memory handeling method is needed.
    
    Returns:
        - True if the arrays can fit in memory, False otherwise
        
    Author: Marco Molina
    '''

    # Get the total RAM capacity in bytes
    ram_capacity = psutil.virtual_memory().total

    # Convert the RAM capacity to gigabytes
    ram_capacity_gb = ram_capacity / (1024 ** 3)

    print(f"Total RAM capacity: {ram_capacity_gb:.2f} GB")

    # Calculate the size of the arrays in bytes
    array_size = seed_params["nmax"] * seed_params["nmay"] * seed_params["nmaz"]
    array_size_bytes =  array_size * np.dtype(out_params["bitformat"]).itemsize

    print(f"Size of the arrays: {array_size_bytes / (1024 ** 3):.2f} GB")

    # Check if the arrays will use more than 1/4 of the total RAM
    if array_size_bytes > (ram_capacity / 4):
        memmap = True
        print("The arrays are too large to fit in memory: np.memmap will be used.")
        # Implement alternative method here
    else:
        memmap = False
        print("The arrays can fit in memory: np.memmap will not be used.")
        # Proceed with the current method
    
    return memmap

def save_3d_array(filename, array):
    '''
    Saves a 3D array to a text file.
    
    Args:
        - filename: name of the file to save the array
        - array: 3D array to save
        
    Author: Marco Molina
    '''
    with open(filename, 'w') as f:
        for i in range(array.shape[0]):
            np.savetxt(f, array[i], header=f'Slice {i}', comments='')
            f.write('\n')
            
def delete_temp_files(temp_files):
    '''
    Deletes temporary files used with np.memmap.
    
    Args:
        - temp_files: list of temporary files to delete
    
    Author: Marco Molina
    '''
    for file_path in temp_files:
        try:
            os.remove(file_path)
            print(f"Deleted memmap file: {file_path}")
        except OSError as e:
            print(f"Error deleting memmap file {file_path}: {e}")
            
def save_magnetic_field_seed(B, axis, format, run):
    '''
    Saves the magnetic field component to a file.
    
    Args:
        - B: magnetic field component to save
        - axis: axis of the magnetic field
        - run: run title
        - format: format of the file (txt, npy, hdf5, fortran)
        
    Author: Marco Molina
    '''
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    os.makedirs(data_dir, exist_ok=True)

    if format == 'txt':
        np.savetxt(os.path.join(data_dir, f'{name}.txt'), B[0].flatten())
    elif format == 'npy':
        np.save(os.path.join(data_dir, f'{name}.npy'), B[0])
    elif format == 'hdf5':
        with h5py.File(os.path.join(data_dir, f'{name}.h5'), 'w') as f:
            f.create_dataset(f'{name}', data=B[0])
    elif format == 'fortran':
        with FortranFile(os.path.join(data_dir, f'{name}.bin'), 'w') as f:
            f.write_record(B[0].astype(out_params["bitformat"]))
            
def load_magnetic_field(axis, run, format='fortran'):
    '''
    Loads the magnetic field component from a file.
    
    Args:
        - axis: axis of the magnetic field component
        - run: run title
        - format: format of the file (txt, npy, hdf5, fortran)
        
    Returns:
        - Magnetic field component
        
    Author: Marco Molina
    '''
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    rshape = (seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"])

    if format == 'txt':
        B = np.loadtxt(os.path.join(data_dir, f'{name}.txt')).reshape(rshape)
    elif format == 'npy':
        B = np.load(os.path.join(data_dir, f'{name}.npy'))
    elif format == 'hdf5':
        with h5py.File(os.path.join(data_dir, f'{name}.h5'), 'r') as f:
            B = f[f'{name}'][:]
    elif format == 'fortran':
        with FortranFile(os.path.join(data_dir, f'{name}.bin'), 'r') as f:
            B = f.read_record(dtype=out_params["bitformat"])
        B = np.frombuffer(B, dtype=out_params["bitformat"]).reshape(rshape)
    
    return B

def get_fortran_file_size(axis, run, dtype=np.float64):
    """
    Gets the size of a Fortran binary file seed component and calculates the array dimensions.

    Args:
        - axis: axis of the magnetic field
        - run: run number
        - dtype: data type of the array elements

    Returns:
        - Number of elements in the array.
        
    Author: Marco Molina
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    filepath = os.path.join(data_dir, f'{name}.bin')
    file_size = os.path.getsize(filepath)
    element_size = np.dtype(dtype).itemsize
    num_elements = file_size // element_size
    return num_elements
            
def is_ordered(vector):
    '''
    Checks if a vector is ordered.
    
    Args:
        - vector: vector to check
        
    Returns:
        - True if the vector is ordered, False otherwise
        
    Author: Marco Molina
    '''
    for i in range(len(vector) - 1):
        if vector[i] > vector[i+1]:
            return False
    return True

def update_loading_bar(progress):
    '''
    Updates and prints a loading bar in the console.
    
    Args:
        - progress: progress percentage
    
    Author: Marco Molina
    '''
    sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress // 1), progress))
    sys.stdout.flush()
    
def create_vector_levels(npatch):
    """
    Creates a vector containing the level for each patch. Nothing really important, just for ease

    Args:
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)

    Returns:
        numpy array containing the level for each patch

    Author: David Vallés
    """
    vector = [[0]] + [[i + 1] * x for (i, x) in enumerate(npatch[1:])]
    vector = [item for sublist in vector for item in sublist]
    return np.array(vector)