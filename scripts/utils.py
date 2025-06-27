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


def patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax):
    """
    Returns, for a given patch, the comoving coordinates of its 8 vertices.

    Args:
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        List containing 8 tuples, each one containing the x, y, z coordinates of the vertex.

    """

    cellsize = size / nmax / 2 ** level

    leftmost_x = rx - cellsize
    leftmost_y = ry - cellsize
    leftmost_z = rz - cellsize

    vertices = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = leftmost_x + i * nx * cellsize
                y = leftmost_y + j * ny * cellsize
                z = leftmost_z + k * nz * cellsize

                vertices.append((x, y, z))

    return vertices


def patch_is_inside_sphere(R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax):
    """

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Returns True if the patch should contain cells within a sphere of radius r of the (clusrx, clusry, clusrz)
        point; False otherwise.

    """
    isinside = False

    vertices = patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax)
    xmin = vertices[0][0]
    ymin = vertices[0][1]
    zmin = vertices[0][2]
    xmax = vertices[-1][0]
    ymax = vertices[-1][1]
    zmax = vertices[-1][2]

    if xmin < clusrx < xmax and ymin < clusry < ymax and zmin < clusrz < zmax:
        return True

    cell_l0_size = size / nmax
    max_side = max([nx, ny, nz]) * cell_l0_size / 2 ** level
    upper_bound_squared = R ** 2 + max_side ** 2 / 2 # half the face diagoonal, (max_side * sqrt2 / 2)^2
    
    def dista_periodic_1d(d, size):
        return min(abs(d), size-abs(d))

    for vertex in vertices:
        distance_squared = dista_periodic_1d((vertex[0] - clusrx), size) ** 2 +\
                           dista_periodic_1d((vertex[1] - clusry), size) ** 2 +\
                           dista_periodic_1d((vertex[2] - clusrz), size) ** 2
        if distance_squared <= upper_bound_squared:
            isinside = True
            break

    return isinside


def which_patches_inside_sphere(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch,
                                size, nmax, kept_patches=None):
    """
    Finds which of the patches will contain cells within a radius r of a certain point (clusrx, clusry, clusrz) being
    its comoving coordinates.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        List containing the ipatch of the patches which should contain cells inside the considered radius.

    """
    levels = create_vector_levels(npatch)
    which_ipatch = [0]

    if kept_patches is None:
        kept_patches = np.ones(patchnx.size, dtype=bool)

    for ipatch in range(1, len(patchnx)):
        if not kept_patches[ipatch]:
            continue
        if patch_is_inside_sphere(R, clusrx, clusry, clusrz, levels[ipatch], patchnx[ipatch], patchny[ipatch],
                                patchnz[ipatch],
                                patchrx[ipatch], patchry[ipatch], patchrz[ipatch], size, nmax):
            which_ipatch.append(ipatch)
    return which_ipatch


def magnitude(field_x, field_y, field_z, kept_patches=None):
    '''
    Calculates the magnitude of a vector field.

    Args:
        field_x: a list of numpy arrays, each one containing the x component of a vector field
                defined on the corresponding grid of the AMR hierarchy

        field_y: idem for the y component of the vector field.
        
        field_z: idem for the z component of the vector field.

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting magnitude of the given vector field with the same structure as the input field

    
    Author: Marco José Molina Pradillo
    '''

    total_npatch = len(field_x)
    assert total_npatch == len(field_y) == len(field_z), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            mag = np.sqrt(field_x[ipatch]**2 + field_y[ipatch]**2 + field_z[ipatch]**2)
            field.append(mag)
        else:
            field.append(0)

    return field


def magnitude2(field_x, field_y, field_z, kept_patches=None):
    '''
    Calculates the magnitude squared of a vector field.

    Args:
        field_x: a list of numpy arrays, each one containing the x component of a vector field
                defined on the corresponding grid of the AMR hierarchy

        field_y: idem for the y component of the vector field.
        
        field_z: idem for the z component of the vector field.

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting magnitude squared of the given vector field with the same structure as the input field

    
    Author: Marco José Molina Pradillo
    '''

    total_npatch = len(field_x)
    assert total_npatch == len(field_y) == len(field_z), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            mag_squared = field_x[ipatch]**2 + field_y[ipatch]**2 + field_z[ipatch]**2
            field.append(mag_squared)
        else:
            field.append(0)
            
    return field


def vol_integral(field, units, zeta, cr0amr, solapst, npatch, patchrx, patchry, patchrz, patchnx, patchny, patchnz, size, nmax, coords, rad, kept_patches=None):
    """
    Given a scalar field and a sphere defined with a center (x,y,z) and a radious together with the patch structure, returns the volumetric integral of the field along the sphere.

    Args:
        - field: scalar field to be integrated
        - units: change of units factor to be multiplied by the final integral if one wants physical units
        - zeta: redshift of the simulation snap to calculate the scale factor
        - cr0amr: AMR maximum refinement factor (only the maximally resolved cells are considered)
        - solapst: AMR overlap factor (only the maximally resolved cells are considered)
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: number of cells in the coarsest resolution level
        - coords: center of the sphere in a numpy array [x,y,z]
        - rad: radius of the sphere
        - kept_patches: boolean array to select the patches to be considered in the integration. True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        - integral: volumetric integral of the field along the sphere
    """
    if kept_patches is None:
        total_npatch = len(field)
        kept_patches = np.ones((total_npatch,), dtype=bool)
        
    vector_levels = create_vector_levels(npatch)
    
    dx = size/nmax
    
    a = 1 / (1 + zeta) # We compute the scale factor
    
    integral = 0
    
    for p in range(len(kept_patches)): # We run across all the patches
        
        patch_res = dx/(2**vector_levels[p])
        
        x0 = patchrx[p] - patch_res/2 #Center of the left-bottom-front cell
        y0 = patchry[p] - patch_res/2
        z0 = patchrz[p] - patch_res/2
        
        x_grid = np.linspace(x0, x0 + patch_res*patchnx[p], patchnx[p])
        y_grid = np.linspace(y0, y0 + patch_res*patchny[p], patchny[p])
        z_grid = np.linspace(z0, z0 + patch_res*patchnz[p], patchnz[p])
        
        X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Create a boolean mask where the condition is True
        mask = ((coords[0] - X_grid)**2 + (coords[1] - Y_grid)**2 + (coords[2] - Z_grid)**2) <= rad**2
        
        # Calculate the physical volume of the cell in this simulation patch
        dr3 = (a*patch_res)**3
        
        # Calculate the integral of the scalar quantity over the volume
        
        masked = np.where(mask, field[p], 0)
        
        integral += np.sum(masked*cr0amr[p]*solapst[p])*dr3
    
    integral = units * integral
    
    print('Total integrated field: ' + str(integral))
    
    return integral