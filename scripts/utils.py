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
        mem = True
        print("The arrays are too large to fit in memory: chunking will be used.")
        # Implement alternative method here
    else:
        mem = False
        print("The arrays can fit in memory: chunking will not be used.")
        # Proceed with the current method
    
    return mem

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
            
def save_magnetic_field_seed(B, axis, real, dir, format, run):
    '''
    Saves the magnetic field component to a file.
    For Fortran files, the data is saved in binary unformatted format, writing the volume
    slice by slice and transposing it to Fortran order (column-major order). Metadata is
    written at the beginning of the file, including the dimensions of the array, the precision
    (float or double), and the complex flag (0 for real, 1 for complex).
    
    Args:
        - B: magnetic field component to save
        - axis: axis of the magnetic field
        - real: whether the space of the magnetic field component is real or not (fourier)
        - dir: directory to save the seed
        - format: format of the file (txt, npy, hdf5, fortran)
        - run: run title
        
    Author: Marco Molina
    '''
    
    assert format in ['txt', 'npy', 'hdf5', 'fortran'], 'Invalid format.'
    assert axis in ['x', 'y', 'z'], 'Invalid axis.'
    assert real in [True, False], 'Invalid space value.'
    
    if real:
        if out_params["bitformat"] == np.float32:
            precision_flag = 1
            complex_flag = 0
        elif out_params["bitformat"] == np.float64:
            precision_flag = 2
            complex_flag = 0
    else:
        if out_params["complex_bitformat"] == np.complex64:
            precision_flag = 1
            complex_flag = 1
        elif out_params["complex_bitformat"] == np.complex128:
            precision_flag = 2
            complex_flag = 1
    
    if complex_flag == 0:
        name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    
    os.makedirs(dir, exist_ok=True)
    
    Nx, Ny, Nz = B[0].shape

    if format == 'txt':
        with open(os.path.join(dir, f'{name}.txt'), 'w') as f:
            for slice in B[0]:
                np.savetxt(f, slice)
    elif format == 'npy':
        if complex_flag == 0:
            np.save(os.path.join(dir, f'{name}.npy'), B[0].astype(out_params["bitformat"]))
        else:
            np.save(os.path.join(dir, f'{name}.npy'), B[0].astype(out_params["complex_bitformat"]))
    elif format == 'hdf5':
        with h5py.File(os.path.join(dir, f'{name}.h5'), 'w') as f:
            f.create_dataset(f'{name}', data=B[0], chunks=True)
    elif format == 'fortran':
        with FortranFile(os.path.join(dir, f'{name}.bin'), 'w') as f:
            metadata = np.array([Nx, Ny, Nz, precision_flag, complex_flag], dtype=np.int32)
            f.write_record(metadata)
            for k in range(B[0].shape[2]):
                if complex_flag == 0:
                    slice = B[0][:, :, k].astype(out_params["bitformat"]).T
                else:
                    slice = B[0][:, :, k].astype(out_params["complex_bitformat"]).T
                f.write_record(slice)

    print(f'B{axis} Seed saved as {os.path.join(dir, name)}.({format})')
            
def load_magnetic_field(axis, real, rshape, dir, format, run):
    '''
    Loads the magnetic field component from a file.
    
    Args:
        - axis: axis of the magnetic field component
        - real: whether the space of the magnetic field component is real or not (fourier)
        - rshape: shape of the array
        - dir: directory to load the seed from
        - format: format of the file (txt, npy, hdf5, fortran)
        - run: run title
        
    Returns:
        - B: Magnetic field component
        
    Author: Marco Molina
    '''
    
    assert format in ['txt', 'npy', 'hdf5', 'fortran'], 'Invalid format.'
    assert axis in ['x', 'y', 'z'], 'Invalid axis.'
    assert real in [True, False], 'Invalid space value.'
    
    if real:
        name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'

    print(f"Loading {name} from {dir}...")
    
    if format == 'txt':
        # Load slice by slice if the file was saved in chunks
        B = []
        with open(os.path.join(dir, f'{name}.txt'), 'r') as f:
            for _ in range(rshape[0]):  # Iterate over slices
                slice = np.loadtxt(f, max_rows=rshape[1])  # Load one slice at a time
                B.append(slice)
        B = np.array(B)  # Convert list of slices to a NumPy array
    elif format == 'npy':
        B = np.load(os.path.join(dir, f'{name}.npy'))
    elif format == 'hdf5':
        with h5py.File(os.path.join(dir, f'{name}.h5'), 'r') as f:
            B = f[f'{name}'][:]  # Load the entire dataset
    elif format == 'fortran':
        with FortranFile(os.path.join(dir, f'{name}.bin'), 'r') as f:
                # Read metadata
                metadata = f.read_record(np.int32)
                if metadata.size != 5:
                    raise ValueError("Invalid metadata size. Expected 5 integers.")
                Nx, Ny, Nz, precision_flag, complex_flag = metadata

                if precision_flag == 1 and complex_flag == 0:
                    dtype = np.float32
                elif precision_flag == 2 and complex_flag == 0:
                    dtype = np.float64
                elif precision_flag == 1 and complex_flag == 1:
                    dtype = np.complex64
                elif precision_flag == 2 and complex_flag == 1:
                    dtype = np.complex128
                else:
                    raise ValueError(f"Unsupported combination: precision_flag={precision_flag}, complex_flag={complex_flag}")

                # Prepare empty array
                B = np.empty((Nx, Ny, Nz), dtype=dtype)
                
                slice = np.empty((Ny, Nx), dtype=dtype)
                
                for k in range(Nz):
                    slice[:, :] = f.read_record(dtype).reshape(Ny, Nx)
                    B[:, :, k] = slice.T
        # if real:
        #     elements = get_fortran_file_size(axis, True, out_params["data_folder"], out_params["run"])
        #     print(f'File elements: {elements-2}')
        #     print(f'Expected elements: {rshape[0] * rshape[1] * rshape[2]}')
        #     print(f'Expected size: {np.prod(rshape) * np.dtype(out_params["bitformat"]).itemsize}')
        #     inspect_fortran_file(axis, real, dir, run)
            
        #     B = np.fromfile(os.path.join(dir, f'{name}.bin'), dtype=out_params["bitformat"], offset=4)[:-1].reshape(rshape) 
        #     # with open(os.path.join(dir, f'{name}.bin'), 'r') as f:
        #     #     B = read_record(f, dtype=out_params["bitformat"]).reshape(rshape)
        #     # # B = np.frombuffer(B, dtype=out_params["bitformat"]).reshape(rshape)  # Reshape to the original shape
        #     B = B.T
        # else:
        #     elements = get_fortran_file_size(axis, False, out_params["data_folder"], out_params["run"])
        #     print(f'File elements: {elements-2}')
        #     print(f'Expected elements: {rshape[0] * rshape[1] * rshape[2]}')
        #     print(f'Expected size: {np.prod(rshape) * np.dtype(out_params["complex_bitformat"]).itemsize}')
        #     inspect_fortran_file(axis, real, dir, run)
            
        #     B = np.fromfile(os.path.join(dir, f'{name}.bin'), dtype=out_params["complex_bitformat"], offset=4)[:-1].reshape(rshape) 
        #     # with open(os.path.join(dir, f'{name}.bin'), 'r') as f:
        #     #     B = read_record(f, dtype=out_params["complex_bitformat"]).reshape(rshape)
        #     # # B = np.frombuffer(B, dtype=out_params["omplex_bitformat"]).reshape(rshape)  # Reshape to the original shape
        #     B = B.T
    
    return B

def inspect_fortran_file(axis, real, dir, run):
    '''
    Inspects the Fortran binary file to check the header, footer and content.
    
    Args:
        - axis: axis of the magnetic field component
        - real: whether the space of the magnetic field component is real or not (fourier)
        - dir: directory to load the seed from
        - run: run tag
        
    Author: Marco Molina
    '''
    
    if real:
        name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'

    filepath = os.path.join(dir, f'{name}.bin')
    
    with open(filepath, 'rb') as f:
        header = f.read(4)
        data = f.read()
        f.seek(-4, os.SEEK_END)
        footer = f.read(4)
        
        print(f"Header (raw): {header}")
        print(f"Footer (raw): {footer}")
        print(f"Header (int): {int.from_bytes(header, byteorder='little')}")
        print(f"Footer (int): {int.from_bytes(footer, byteorder='little')}")
        print(f"Data size:    {len(data)-4}")

def get_fortran_file_size(axis, real, dir, run):
    """
    Gets the size of a Fortran binary file seed component and calculates the array dimensions.

    Args:
        - axis: axis of the magnetic field
        - real: whether the space of the magnetic field component is real or not (fourier)
        - dir: directory to load the seed from
        - run: run number

    Returns:
        - Number of elements in the array.
        
    Author: Marco Molina
    """
    
    assert axis in ['x', 'y', 'z'], 'Invalid axis.'
    assert real in [True, False], 'Invalid space value.'
    
    if real:
        name = f'B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
        element_size = np.dtype(out_params["bitformat"]).itemsize
    else:
        name = f'B{axis}_fourier_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}'
        element_size = np.dtype(out_params["complex_bitformat"]).itemsize
    
    file_size = os.path.getsize(dir + f'/{name}.bin')
    num_elements = file_size // element_size
    
    return num_elements

def compare_arrays(axis, real, dir, run):
    '''
    Compares two arrays to check if they are equal.
    
    Args:
        - axis: axis of the magnetic field
        - real: whether the space of the magnetic field component is real or not (fourier)
        - dir: directory to load the seed from
        - run: run number
    
    Author: Marco Molina
    '''
    
    # Your original data
    if real:
        dtype = out_params["bitformat"]
        Nx, Ny, Nz = seed_params["nmax"], seed_params["nmay"], seed_params["nmaz"]
        original_data = np.load(f'{dir}/B{axis}_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}.npy')
    else:
        dtype = out_params["complex_bitformat"]
        Nx, Ny, Nz = seed_params["nmax"]+1, seed_params["nmay"]+1, seed_params["nmaz"]+1
        original_data = np.load(f'{dir}/B{axis}_fourier_{run}_{seed_params["nmax"]}_{seed_params["size"]}_{seed_params["alpha"]}.npy')
    
    # Fortran-loaded data
    fortran_data = np.fromfile(f'fortran_output_raw_{axis}.bin', dtype=dtype)
    fortran_data = fortran_data.reshape((Nx, Ny, Nz)).T

    # Compare them
    are_equal = np.allclose(original_data, fortran_data, atol=1e-6)

    print(f"Arrays are {'equal' if are_equal else 'different'}!")

    if not are_equal:
        diff = np.abs(original_data - fortran_data)
        print(f"Max difference: {np.max(diff)}")
            
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