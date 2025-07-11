"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

utils module
Contains utility tool functions used by the MAGNAS SSD Evolution framework.

Created by Marco Molina Pradillo
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from scipy.io import FortranFile
import h5py
import psutil
import amr2uniform as a2u
from multiprocessing import Pool
from config import IND_PARAMS as ind_params
from config import OUTPUT_PARAMS as out_params
from numba import njit, prange
import warnings
from numba.typed import List

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
    array_size = ind_params["nmax"] * ind_params["nmay"] * ind_params["nmaz"]
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
        name = f'B{axis}_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
    
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
        name = f'B{axis}_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'

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
        name = f'B{axis}_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
    else:
        name = f'B{axis}_fourier_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'

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
        name = f'B{axis}_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
        element_size = np.dtype(out_params["bitformat"]).itemsize
    else:
        name = f'B{axis}_fourier_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}'
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
        Nx, Ny, Nz = ind_params["nmax"], ind_params["nmay"], ind_params["nmaz"]
        original_data = np.load(f'{dir}/B{axis}_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}.npy')
    else:
        dtype = out_params["complex_bitformat"]
        Nx, Ny, Nz = ind_params["nmax"]+1, ind_params["nmay"]+1, ind_params["nmaz"]+1
        original_data = np.load(f'{dir}/B{axis}_fourier_{run}_{ind_params["nmax"]}_{ind_params["size"]}_{ind_params["alpha"]}.npy')
    
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


def clean_field(field, cr0amr, solapst, npatch, up_to_level=1000):
    """
    Receives a field (with its refinement patches) and, using the cr0amr and solapst variables, returns the field
    having "cleaned" for refinements and overlaps. The user can specify the level of refinement required. This last
    level will be cleaned of overlaps, but not of refinements!

    Args:
        field: field to be cleaned
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        up_to_level: specify if only cleaning until certain level is desired

    Returns:
        "Clean" version of the field, with the same structure.

    Author: David Vallés
    """
    levels = create_vector_levels(npatch)
    up_to_level = min(up_to_level, levels.max())

    cleanfield = []
    if up_to_level == 0:
        cleanfield.append(field[0])
    else:
        cleanfield.append(field[0] * cr0amr[0])  # not overlap in l=0

    for level in range(1, up_to_level):
        for ipatch in range(sum(npatch[0:level]) + 1, sum(npatch[0:level + 1]) + 1):
            cleanfield.append(field[ipatch] * cr0amr[ipatch] * solapst[ipatch])

    # last level: no refinements
    for ipatch in range(sum(npatch[0:up_to_level]) + 1, sum(npatch[0:up_to_level + 1]) + 1):
        cleanfield.append(field[ipatch] * solapst[ipatch])

    return cleanfield


def mask_sphere(R, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz, kept_patches=None):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        cellsrx, cellsry, cellsrz: position fields
        kept_patches: 1d boolean array, True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        Field containing the mask as described.
        
    Author: David Vallés
    """
    if kept_patches is None:
        kept_patches = np.ones(len(cellsrx), dtype=bool)

    mask = [(cx - clusrx) ** 2 + (cy - clusry) ** 2 + (cz - clusrz) ** 2 < R ** 2 if ki else False
            for cx, cy, cz, ki in zip(cellsrx, cellsry, cellsrz, kept_patches)]

    return mask


def radial_profile_vw(field, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, cellsrx, cellsry,
                    cellsrz, cr0amr, solapst, npatch, size, nmax, up_to_level=1000, verbose=False,
                    kept_patches=None):
    """
    Computes a (volume-weighted) radial profile of the quantity given in the "field" argument, taking center in
    (clusrx, clusry, clusrz).

    Args:
        field: variable (already cleaned) whose profile wants to be got
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                logbins = True, rmin cannot be 0.
        cellsrx, cellsry, cellsrz: position fields
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        up_to_level: maximum AMR level to be considered for the profile
        verbose: if True, prints the patch being opened at a time
        kept_patches: 1d boolean array, True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        Two lists. One of them contains the center of each radial cell. The other contains the value of the field
        averaged across all the cells of the shell.
        
    Author: David Vallés
    """
    if kept_patches is None:
        kept_patches = np.ones(len(cellsrx), dtype=bool)

    # getting the bins
    try:
        assert (rmax > rmin)
    except AssertionError:
        print('You would like to input rmax > rmin...')
        return

    if logbins:
        try:
            assert (rmin > 0)
        except AssertionError:
            print('Cannot use rmin=0 with logarithmic binning...')
            return
        bin_bounds = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)

    else:
        bin_bounds = np.linspace(rmin, rmax, nbins + 1)

    bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2
    # profile = np.zeros(bin_centers.shape)
    profile = []

    # finding the volume-weighted mean
    levels = create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3

    field_vw = [f * cv for f, cv in zip(field, cell_volume)]

    if rmin > 0:
        cells_outer = mask_sphere(rmin, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz, kept_patches=kept_patches)
    else:
        cells_outer = [np.zeros(patch.shape, dtype='bool') for patch in field]

    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        cells_inner = cells_outer
        cells_outer = mask_sphere(r_out, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz, kept_patches=kept_patches)
        shell_mask = [inner ^ outer if ki else 0 for inner, outer, ki in zip(cells_inner, cells_outer, kept_patches)]
        shell_mask = clean_field(shell_mask, cr0amr, solapst, npatch, up_to_level=up_to_level)

        sum_field_vw = sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)])
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile.append(sum_field_vw / sum_vw)

    return bin_centers, np.asarray(profile)


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


def patch_is_inside_box(box_limits, level, nx, ny, nz, rx, ry, rz, size, nmax):
    """
    See "Returns:"

    Args:
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax)
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Returns True if the patch should contain cells within a sphere of radius r of the (clusrx, clusry, clusrz)
        point; False otherwise.
        
    Author: David Vallés
    """
    vertices = patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax)
    pxmin = vertices[0][0]
    pymin = vertices[0][1]
    pzmin = vertices[0][2]
    pxmax = vertices[-1][0]
    pymax = vertices[-1][1]
    pzmax = vertices[-1][2]

    bxmin = box_limits[0]
    bxmax = box_limits[1]
    bymin = box_limits[2]
    bymax = box_limits[3]
    bzmin = box_limits[4]
    bzmax = box_limits[5]

    overlap_x = (pxmin <= bxmax) and (bxmin <= pxmax)
    overlap_y = (pymin <= bymax) and (bymin <= pymax)
    overlap_z = (pzmin <= bzmax) and (bzmin <= pzmax)

    return overlap_x and overlap_y and overlap_z


def which_patches_inside_box(box_limits, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax,
                            kept_patches=None):
    """
    Finds which of the patches will contain cells within a box of defined vertices.

    Args:
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax)
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

    Author: David Vallés
    """
    levels = create_vector_levels(npatch)
    which_ipatch = [0]

    if kept_patches is None:
        kept_patches = np.ones(patchnx.size, dtype=bool)

    for ipatch in range(1, len(patchnx)):
        if not kept_patches[ipatch]:
            continue
        if patch_is_inside_box(box_limits, levels[ipatch], patchnx[ipatch], patchny[ipatch], patchnz[ipatch],
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


def vol_integral(field, units, zeta, cr0amr, solapst, npatch, patchrx, patchry, patchrz, patchnx, patchny, patchnz, size, nmax, coords, rad, kept_patches=None, vol=False):
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
        - vol: if True, returns the volume of the region instead of the integral. Default is False.

    Returns:
        - integral: volumetric integral of the field along the sphere
        
    Author: Marco Molina
    """
    if kept_patches is None:
        total_npatch = len(field)
        kept_patches = np.ones((total_npatch,), dtype=bool)
        
    vector_levels = create_vector_levels(npatch)
    
    dx = size/nmax
    
    a = 1 / (1 + zeta) # We compute the scale factor
    
    integral = 0
    
    if vol:
        field = [np.ones_like(field[p]) for p in range(1 + np.sum(npatch))] # If vol is True, we just want the volume, so we set the field to 1
    
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
    
    # print('Total integrated field: ' + str(integral))
    
    return integral


def compute_position_field_onepatch(args):
    """
    Returns a 3 matrices with the dimensions of the patch, containing the position for every cell centre

    Args: tuple containing, in this order:
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        level: refinement level of the given patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Matrices as defined
        
    Author: David Vallés
    """
    nx, ny, nz, rx, ry, rz, level, size, nmax, keep = args

    if not keep:
        return 0,0,0

    cellsize = size / nmax / 2 ** level
    first_x = rx - cellsize / 2
    first_y = ry - cellsize / 2
    first_z = rz - cellsize / 2
    patch_x = np.zeros((nx, ny, nz), dtype='f4')
    patch_y = np.zeros((nx, ny, nz), dtype='f4')
    patch_z = np.zeros((nx, ny, nz), dtype='f4')

    for i in range(nx):
        patch_x[i, :, :] = first_x + i * cellsize
    for j in range(ny):
        patch_y[:, j, :] = first_y + j * cellsize
    for k in range(nz):
        patch_z[:, :, k] = first_z + k * cellsize

    return patch_x, patch_y, patch_z


def compute_position_fields(patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax, ncores=1,
                            kept_patches=None):
    """
    Returns 3 fields (as usually defined) containing the x, y and z position for each of our cells centres.
    Args:
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        ncores: number of cores to be used in the computation
        kept_patches: 1d boolean array, True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        3 fields as described above
        
    Author: David Vallés
    """
    levels = create_vector_levels(npatch)
    if kept_patches is None:
        kept_patches = np.ones(patchnx.size, dtype=bool)

    if ncores == 1 or ncores == 0 or ncores is None:
        cellsrx = []
        cellsry = []
        cellsrz = []
        for ipatch in range(npatch.sum()+1):
            patches = compute_position_field_onepatch((patchnx[ipatch], patchny[ipatch], patchnz[ipatch],
                                                        patchrx[ipatch], patchry[ipatch], patchrz[ipatch],
                                                        levels[ipatch], size, nmax, kept_patches[ipatch]))
            cellsrx.append(patches[0])
            cellsry.append(patches[1])
            cellsrz.append(patches[2])
    else:
        with Pool(ncores) as p:
            positions = p.map(compute_position_field_onepatch,
                            [(patchnx[ipatch], patchny[ipatch], patchnz[ipatch], patchrx[ipatch], patchry[ipatch],
                            patchrz[ipatch], levels[ipatch], size, nmax, kept_patches[ipatch]) 
                            for ipatch in range(len(patchnx))])

        cellsrx = [p[0] for p in positions]
        cellsry = [p[1] for p in positions]
        cellsrz = [p[2] for p in positions]

    return cellsrx, cellsry, cellsrz


def uniform_field(field, clus_cr0amr, clus_solapst, grid_npatch,
                grid_patchnx, grid_patchny, grid_patchnz, 
                grid_patchrx, grid_patchry, grid_patchrz,
                nmax, size, Box,
                up_to_level=4, ncores=1, clus_kp=None, verbose=False):
    '''
    Cleans and computes the uniform version of the given field for the given AMR grid for its further projection.
    
    Args:
        - field: field to be cleaned and set uniform
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - grid_npatch: number of patches in the grid
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - nmax: maximum number of patches in the grid
        - size: size of the grid
        - Box: box coordinates
        - up_to_level: level of refinement in the AMR grid (default is 4)
        - ncores: number of cores to use for the computation (default is 1)
        - clus_kp: boolean array to select the patches to be considered in the computation. True if the patch is kept, False if not.
        - verbose: boolean to print the progress of the computation (default is False)
        
    Returns:
        - uniform_field: cleaned and projected field on a uniform grid
        
    Author: Marco Molina
    '''
    
    if up_to_level > 4:
        print("Warning: The resolution level is larger than 4. The code will take a long time to run.")
        
    uniform_field = uniform_grid_zoom_interpolate(
        field=field, box_limits=Box[1:], up_to_level=up_to_level,
        npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
        patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
        patchrz=grid_patchrz, size=size, nmax=nmax,
        interpolate=True, verbose=verbose, kept_patches=clus_kp, return_coords=False
    )
        
    # cleaned_field = clean_field(field, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=up_to_level)
    
    # uniform_field = a2u.main(box = Box[1:], up_to_level = up_to_level, nmax = nmax, size = size, npatch = grid_npatch, patchnx = grid_patchnx, patchny = grid_patchny,
    #                         patchnz = grid_patchnz, patchrx = grid_patchrx, patchry = grid_patchry, patchrz = grid_patchrz,
    #                         field = cleaned_field, ncores = ncores, verbose = verbose)
        
    return uniform_field

################################### THIS IS OUR MAIN UNIGRID FUNCTION ###############################################
def uniform_grid_zoom_interpolate(field, box_limits, up_to_level, npatch, patchnx, patchny, patchnz, patchrx, patchry,
                                  patchrz, size, nmax, interpolate=True, verbose=False, kept_patches=None, return_coords=False):
    """
    Builds a uniform grid, zooming on a box specified by box_limits, at level up_to_level, containing the most refined
    data at each region, and interpolating from coarser patches
    Args:
        field: field to be computed at the uniform grid. MUST BE UNCLEANED!!
        box_limits: a tuple in the form (imin, imax, jmin, jmax, kmin, kmax). Limits should correspond to l=0 cell
                    boundaries
        up_to_level: level up to which the fine grid wants to be obtained
        cr0amr: XXXXXXXXXXX
        solapst: XXXXXXXXXXX
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        interpolate (bool): if True (default) uses linear iterpolation for low-resolution regions. If False, uses NGP.
        verbose: if True, prints the patch being opened at a time
        kept_patches: list of patches that are read. If None, all patches are assumed to be present
        return_coords: if True, returns the coordinates of the uniform grid

    Returns:
        Uniform grid as described, and optionally the coordinates of the grid

    Author: David Vallés
    
    KNOWN ISSUES: if interpolate=True and box_limits reaches the edge of the simulation box, the interpolation will fail,
                  crashing the interpreter. This is a known issue and will be fixed in future versions.              
    """
    
    if up_to_level > 4:
        print("Warning: The resolution level is larger than 4. The code will take a long time to run.")

    def intt(x):
        if x >= 0:
            return int(x)
        else:
            return int(x) - 1

    if kept_patches is None:
        kept_patches = np.ones(patchnx.size, dtype='bool')

    coarse_cellsize = size / nmax
    uniform_cellsize = size / nmax / 2 ** up_to_level

    if type(box_limits[0]) in [float, np.float32, np.float64, np.float128]:
        box_limits = [intt((box_limits[0] + size / 2) * nmax / size),
                      intt((box_limits[1] + size / 2) * nmax / size)+1,
                      intt((box_limits[2] + size / 2) * nmax / size),
                      intt((box_limits[3] + size / 2) * nmax / size)+1,
                      intt((box_limits[4] + size / 2) * nmax / size),
                      intt((box_limits[5] + size / 2) * nmax / size)+1]
    bimin = box_limits[0]
    bimax = box_limits[1]
    bjmin = box_limits[2]
    bjmax = box_limits[3]
    bkmin = box_limits[4]
    bkmax = box_limits[5]

    # Warning for the issue mentioned above
    if interpolate and (bimin == 0 or bimax == nmax-1 or bjmin == 0 or bjmax == nmax-1 or bkmin == 0 or bkmax == nmax-1):
        warnings.warn('Interpolation may fail at the edges of the simulation box. Please, use NGP interpolation or '+\
                        'avoid reaching the edges of the box. This issue will be fixed in future versions.')

    bxmin = -size / 2 + bimin * coarse_cellsize
    bxmax = -size / 2 + (bimax + 1) * coarse_cellsize
    bymin = -size / 2 + bjmin * coarse_cellsize
    bymax = -size / 2 + (bjmax + 1) * coarse_cellsize
    bzmin = -size / 2 + bkmin * coarse_cellsize
    bzmax = -size / 2 + (bkmax + 1) * coarse_cellsize

    # BASE GRID
    reduction = 2 ** up_to_level
    uniform_size_x = (bimax - bimin + 1) * reduction
    uniform_size_y = (bjmax - bjmin + 1) * reduction
    uniform_size_z = (bkmax - bkmin + 1) * reduction

    fine_coordinates = List([np.linspace(bxmin + uniform_cellsize / 2, bxmax - uniform_cellsize / 2, uniform_size_x),
                        np.linspace(bymin + uniform_cellsize / 2, bymax - uniform_cellsize / 2, uniform_size_y),
                        np.linspace(bzmin + uniform_cellsize / 2, bzmax - uniform_cellsize / 2, uniform_size_z)])

    vertices_patches = np.zeros((npatch[0:up_to_level + 1].sum() + 1, 6))
    vertices_patches[0, :] = [-size / 2, size / 2, -size / 2, size / 2, -size / 2, size / 2]
    for l in range(1, up_to_level + 1):
        dx = coarse_cellsize / 2 ** l
        for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
            vertices_patches[ipatch, 0] = patchrx[ipatch] - dx
            vertices_patches[ipatch, 2] = patchry[ipatch] - dx
            vertices_patches[ipatch, 4] = patchrz[ipatch] - dx
            vertices_patches[ipatch, 1] = vertices_patches[ipatch, 0] + patchnx[ipatch] * dx
            vertices_patches[ipatch, 3] = vertices_patches[ipatch, 2] + patchny[ipatch] * dx
            vertices_patches[ipatch, 5] = vertices_patches[ipatch, 4] + patchnz[ipatch] * dx

    levels = create_vector_levels(npatch)

    cell_patch = np.zeros((uniform_size_x,uniform_size_y,uniform_size_z),dtype='i4')
    for l in range(1, up_to_level + 1, 1):
        reduction = 2 ** (up_to_level - l)
        dx = uniform_cellsize * reduction
        for ipatch in range(npatch[0:l + 1].sum(), npatch[0:l].sum(), -1):
            if not kept_patches[ipatch]:
                continue

            i1 = intt(((vertices_patches[ipatch, 0]+0.5*dx) - bxmin) / uniform_cellsize + 0.5)
            j1 = intt(((vertices_patches[ipatch, 2]+0.5*dx) - bymin) / uniform_cellsize + 0.5)
            k1 = intt(((vertices_patches[ipatch, 4]+0.5*dx) - bzmin) / uniform_cellsize + 0.5)
            i2 = i1 + reduction * (patchnx[ipatch]-1) - 1
            j2 = j1 + reduction * (patchny[ipatch]-1) - 1
            k2 = k1 + reduction * (patchnz[ipatch]-1) - 1
            if i2 > -1 and i1 < uniform_size_x and \
                    j2 > -1 and j1 < uniform_size_y and \
                    k2 > -1 and k1 < uniform_size_z:
                i1 = max([i1, 0])
                j1 = max([j1, 0])
                k1 = max([k1, 0])
                i2 = min([i2, uniform_size_x - 1])
                j2 = min([j2, uniform_size_y - 1])
                k2 = min([k2, uniform_size_z - 1])
                cell_patch[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1] = ipatch

    # 2. try to increase level using boundaries
    for l in range(1, up_to_level + 1, 1):
        reduction = 2 ** (up_to_level - l)
        dx = uniform_cellsize * reduction
        for ipatch in range(npatch[0:l + 1].sum(), npatch[0:l].sum(), -1):
            if not kept_patches[ipatch]:
                continue

            i1 = intt(((vertices_patches[ipatch, 0]) - bxmin) / uniform_cellsize + 0.5)
            j1 = intt(((vertices_patches[ipatch, 2]) - bymin) / uniform_cellsize + 0.5)
            k1 = intt(((vertices_patches[ipatch, 4]) - bzmin) / uniform_cellsize + 0.5)
            i2 = i1 + reduction * (patchnx[ipatch]) - 1
            j2 = j1 + reduction * (patchny[ipatch]) - 1
            k2 = k1 + reduction * (patchnz[ipatch]) - 1
            if i2 > -1 and i1 < uniform_size_x and \
                    j2 > -1 and j1 < uniform_size_y and \
                    k2 > -1 and k1 < uniform_size_z:
                i1 = max([i1, 0])
                j1 = max([j1, 0])
                k1 = max([k1, 0])
                i2 = min([i2, uniform_size_x - 1])
                j2 = min([j2, uniform_size_y - 1])
                k2 = min([k2, uniform_size_z - 1])
                cell_patch[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1] = np.where(l > levels[np.abs(cell_patch[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1])], -ipatch, cell_patch[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1])
                # if we can map it to a higher level, even though we cannot interpolate, we do it.

    @njit(parallel=True)
    def parallelize(uniform_size_x, uniform_size_y, uniform_size_z, fine_coordinates, cell_patch, vertices_patches,
                    field,interpolate,verbose):
        uniform = np.zeros((uniform_size_x, uniform_size_y, uniform_size_z), dtype=np.float32)
        for i in prange(uniform_size_x):
            if verbose:
                print('ix=',i,uniform_size_x)
            for j in range(uniform_size_y):
                for k in range(uniform_size_z):
                    x, y, z = fine_coordinates[0][i], fine_coordinates[1][j], fine_coordinates[2][k]
                    ipatch = cell_patch[i, j, k]
                    
                    # sign = 1 --> interpolate; sign = -1 --> copy
                    sign = 1
                    if ipatch < 0:
                        ipatch = -ipatch
                        sign = -1

                    n1, n2, n3 = patchnx[ipatch], patchny[ipatch], patchnz[ipatch]
                    l = levels[ipatch]
                    dx = coarse_cellsize / 2 ** l

                    if l == up_to_level:
                        xx, yy, zz = vertices_patches[ipatch, 0], vertices_patches[ipatch, 2], vertices_patches[ipatch, 4]
                        ii, jj, kk = int((x - xx) / dx), int((y - yy) / dx), int((z - zz) / dx)
                        uniform[i, j, k] = field[ipatch][ii, jj, kk]
                    else:
                        xx, yy, zz = vertices_patches[ipatch, 0], vertices_patches[ipatch, 2], vertices_patches[ipatch, 4]
                        ii, jj, kk = int((x - xx) / dx), int((y - yy) / dx), int((z - zz) / dx )
                        xbas, ybas, zbas = xx + (ii + 0.5) * dx, yy + (jj + 0.5) * dx, zz + (kk + 0.5) * dx
                        xbas, ybas, zbas = (x - xbas) / dx, (y - ybas) / dx, (z - zbas) / dx

                        # ii, jj, kk are the indices of the cell containing xx, yy, zz 
                        # if we want to interpolate and if the cell is not at the edge of the patch, 
                        # we will need to displace or not by one cell --> ii2, jj2, kk2 indices
                        # if we just copy (ngp), these are the correct indices (ii, jj, kk)
                        if interpolate and sign == 1:
                            ii2 = ii
                            if xbas < 0.:
                                xbas += 1.
                                ii2 -= 1

                            jj2 = jj
                            if ybas < 0.:
                                ybas += 1.
                                jj2 -= 1

                            kk2 = kk
                            if zbas < 0.:
                                zbas += 1.
                                kk2 -= 1

                            #if ii2 != 0 and ii2 != n1 - 1 and \
                            #   jj2 != 0 and jj2 != n2 - 1 and \
                            #   kk2 != 0 and kk2 != n3 - 1:   
                            if ii2 >= 0 and ii2 != n1 - 1 and \
                               jj2 >= 0 and jj2 != n2 - 1 and \
                               kk2 >= 0 and kk2 != n3 - 1 and \
                               abs(xbas) <= 1. and abs(ybas) <= 1. and abs(zbas) <= 1.:   
                                ubas = field[ipatch][ii2:ii2 + 2, jj2:jj2 + 2, kk2:kk2 + 2]

                                c00 = ubas[0, 0, 0] * (1 - xbas) + ubas[1, 0, 0] * xbas
                                c01 = ubas[0, 0, 1] * (1 - xbas) + ubas[1, 0, 1] * xbas
                                c10 = ubas[0, 1, 0] * (1 - xbas) + ubas[1, 1, 0] * xbas
                                c11 = ubas[0, 1, 1] * (1 - xbas) + ubas[1, 1, 1] * xbas

                                c0 = c00 * (1 - ybas) + c10 * ybas
                                c1 = c01 * (1 - ybas) + c11 * ybas

                                uniform[i, j, k] = c0 * (1 - zbas) + c1 * zbas
                            else:
                                raise ValueError('Interpolation failed, where it shouldnt. Please check or call your closest interpolator on-call...')
                        else:
                            uniform[i, j, k] = field[ipatch][ii, jj, kk]
                        
        return uniform

    uniform=parallelize(uniform_size_x, uniform_size_y, uniform_size_z, fine_coordinates, cell_patch, vertices_patches,
                        List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(field, kept_patches)]),
                        interpolate,verbose)
    

    



    if return_coords:
        return uniform, fine_coordinates
    else:
        return uniform
    
################################### THIS IS OUR MAIN UNIGRID FUNCTION ###############################################
unigrid = uniform_grid_zoom_interpolate
################################### THIS IS OUR MAIN UNIGRID FUNCTION ###############################################