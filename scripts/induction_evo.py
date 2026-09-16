"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

induction_evo module
Provides a set of functions to evolve the magnetic field in a cosmological context and studying the SSD
amplification mechanism.

Created by Marco Molina Pradillo
"""

import gc
import os
import sys
import time
from time import strftime
from time import gmtime
import numpy as np
import scripts.utils as utils
import scripts.diff as diff
import scripts.debug as debug_module
import buffer as buff
import scripts.readers as reader
from scripts.units import *
from scripts.test import analytic_test_fields, numeric_test_fields
from scipy.special import gamma
# from scipy import fft
from matplotlib import pyplot as plt
import pdb
import multiprocessing as mp
import gc
np.set_printoptions(linewidth=200)


# Use log_message from utils
log_message = utils.log_message


def find_most_massive_halo(sim_name, it, a0, dir_halos, dir_grids, data_folder, vir_kind=1, rad_kind=1, verbose=False):
    '''
    Finds the coordinates and radius of the most massive halo in each snapshot of the simulations. In case
    we are looking for the most massive halo to center our analysis, we need to build the python halo catalogue
    (by now we exclude subhalos)
    
    Args:
        - sim_name: simulation name
        - it: list of snapshots
        - a0: scale factor of the simulation (typically 1.0 for the last snapshot)
        - dir_halos: directory where the halo catalogues are stored
        - dir_grids: directory where the grids are stored
        - data_folder: directory where the data is stored
        - vir_kind: kind of virial radius to use (1: Reference virial radius at the last snap, 2: Reference virial radius at each epoch)
        - rad_kind: kind of radius to use (1: Comoving, 2: Physical)
        - verbose: boolean to print the coordinates and radius or not
        
    Returns:
        - coords: list of coordinates of the most massive halo in each snapshot
        - rad: list of radii of the most massive halo in each snapshot
        
    Author: Marco Molina
    '''

    # Find the most massive halo in each snapshot

    coords = []
    rad = []
    max_halo_mass = None

    # Read halos and zeta for each snapshot in reverse order so that we can track the same most massive halo
    for j in reversed(range(len(it))):
        halos = reader.read_families(it[j], path=dir_halos + sim_name, output_format='dictionaries', output_redshift=False,
                    min_mass=None, exclude_subhaloes=True, read_region=None, keep_boundary_contributions=False)
        
        _,_,_,_,zeta = reader.read_grids(it=it[j], path=dir_grids + sim_name, parameters_path=data_folder + '/' + sim_name + '/', digits=5,
            read_general=True, read_patchnum=False, read_dmpartnum=False, read_patchcellextension=False,
            read_patchcellposition=False, read_patchposition=False, read_patchparent=False, nparray=False)
        
        if j == len(it) - 1:
            # Find the index of the most massive halo
            max_mass_index = np.argmax([halo['M'] for halo in halos])
            id_max_mass = halos[max_mass_index]['id']
            max_halo_mass = halos[max_mass_index]['M']
            if vir_kind == 1:
                R_max_mass = halos[max_mass_index]['R']
        
        index = next((i for i, halo in enumerate(halos) if halo['id'] == id_max_mass), None)
        
        if index is not None:
            coords.append((halos[index]['x'], halos[index]['y'], halos[index]['z']))
            
            if vir_kind == 1 and rad_kind == 1:
                rad.append(R_max_mass) # Taking the Virial radius of the most massive halo at the last snap
            elif vir_kind == 1 and rad_kind == 2:
                rad.append(R_max_mass * (a0/(1 + zeta)))
            elif vir_kind == 2 and rad_kind == 1:
                rad.append(halos[index]['R']) # Changing the virial radius at each snap
            elif vir_kind == 2 and rad_kind == 2:
                rad.append(halos[index]['R'] * (a0/(1 + zeta)))
        else:
            coords.append(coords[-1])
            rad.append(rad[-1])
            if verbose:
                log_message("No halo found in snap " + str(it[j]) + ", using the previous one.", tag="halo", level=1)
    
    # Reverse the lists to match the original order of snapshots
    coords = coords[::-1]
    rad = rad[::-1]
    
    if verbose and coords:
        log_message(
            "Coordinates of the most massive halo in the last snap " + str(it[-1]) + ":",
            tag="halo",
            level=1
        )
        log_message("x: " + str(coords[-1][0]), tag="halo", level=2)
        log_message("y: " + str(coords[-1][1]), tag="halo", level=2)
        log_message("z: " + str(coords[-1][2]), tag="halo", level=2)
        log_message("Radius: " + str(rad[-1]) + " Mpc", tag="halo", level=2)
        if max_halo_mass is not None:
            log_message("Mass: " + str(max_halo_mass) + " Msun/h", tag="halo", level=2)
    
    return coords, rad


def create_region(sim_name, it, coords, rad, size, F=1.0, reg='BOX', verbose=False):
    '''
    Creates the boxes or spheres centered at the coordinates of the most massive halo or any other point in each snapshot.
    Automatically clips regions to simulation box boundaries and disables region reading if the clipped region
    equals the entire box.
    
    Args:
        - sim_name: simulation name
        - it: list of snapshots
        - coords: list of coordinates of the most massive halo in each snapshot
        - rad: list of radii of the most massive halo in each snapshot
        - size: size of the simulation box in Mpc (single value or list)
        - F: factor to scale the radius (default is 1.0)
        - red: region type to create ('BOX' or 'SPH', default is 'BOX')
            - BOX: creates a box
            - SPH: creates a sphere
            - None: all the domain is considered
        - verbose: boolean to print the coordinates and radius or not
        
    Returns:
        - region: list of boxes or spheres centered at the coordinates (or None if region equals entire box)
        - region_size: list of sizes of the boxes or spheres in Mpc
        
    Author: Marco Molina
    '''
    
    # Box boundaries (simulation box is centered at origin)
    box_min = -size / 2.0
    box_max = size / 2.0

    Rad = []
    region_size = []
    region = []

    for j in range(len(it)):
        Rad.append(F * rad[j])
        region_size.append(2 * Rad[-1])  # Size of the box in Mpc
        
        if reg == 'BOX':
            # Calculate region boundaries
            x1 = coords[j][0] - Rad[-1]
            x2 = coords[j][0] + Rad[-1]
            y1 = coords[j][1] - Rad[-1]
            y2 = coords[j][1] + Rad[-1]
            z1 = coords[j][2] - Rad[-1]
            z2 = coords[j][2] + Rad[-1]
            
            # Clip to box boundaries
            x1_clipped = max(x1, box_min)
            x2_clipped = min(x2, box_max)
            y1_clipped = max(y1, box_min)
            y2_clipped = min(y2, box_max)
            z1_clipped = max(z1, box_min)
            z2_clipped = min(z2, box_max)
            
            # Check if clipping occurred
            if (x1 < box_min or x2 > box_max or 
                y1 < box_min or y2 > box_max or 
                z1 < box_min or z2 > box_max):
                if verbose and j == 0:
                    log_message(f"Warning: Region for snapshot {it[j]} extends beyond simulation box.", tag="region", level=1)
                    log_message(f"Original: x=[{x1:.2f}, {x2:.2f}], y=[{y1:.2f}, {y2:.2f}], z=[{z1:.2f}, {z2:.2f}]", tag="region", level=2)
                    log_message(f"Clipped:  x=[{x1_clipped:.2f}, {x2_clipped:.2f}], y=[{y1_clipped:.2f}, {y2_clipped:.2f}], z=[{z1_clipped:.2f}, {z2_clipped:.2f}]", tag="region", level=2)
            
            # Check if clipped region equals entire box (with small tolerance)
            tolerance = 1e-6
            region_equals_box = (
                abs(x1_clipped - box_min) < tolerance and abs(x2_clipped - box_max) < tolerance and
                abs(y1_clipped - box_min) < tolerance and abs(y2_clipped - box_max) < tolerance and
                abs(z1_clipped - box_min) < tolerance and abs(z2_clipped - box_max) < tolerance
            )
            
            if region_equals_box:
                if verbose and j == 0:
                    log_message("Region equals entire box -> disabling region filter (reading all patches)", tag="region", level=2)
                region.append([None])
            else:
                region.append(["box", x1_clipped, x2_clipped, y1_clipped, y2_clipped, z1_clipped, z2_clipped])
                
        elif reg == 'SPH':
            # For spheres, check if radius extends beyond box
            effective_radius = Rad[-1]
            max_extent = max(
                abs(coords[j][0]) + effective_radius,
                abs(coords[j][1]) + effective_radius,
                abs(coords[j][2]) + effective_radius
            )
            
            if max_extent > box_max:
                if verbose and j == 0:
                    log_message(f"Warning: Spherical region for snapshot {it[j]} extends beyond simulation box.", tag="region", level=1)
                    log_message(f"Center: ({coords[j][0]:.2f}, {coords[j][1]:.2f}, {coords[j][2]:.2f})", tag="region", level=2)
                    log_message(f"Radius: {effective_radius:.2f} Mpc", tag="region", level=2)
                    log_message("Region equals entire box -> disabling region filter (reading all patches)", tag="region", level=2)
                region.append([None])
            else:
                region.append(["sphere", coords[j][0], coords[j][1], coords[j][2], Rad[-1]])
        else:
            region.append([None])
                
    if verbose:      
        # Print the coordinates
        if region[-1][0] is None:
            log_message("Region: None (using entire simulation box)", tag="region", level=1)
        else:
            log_message(str(region[-1][0]) + " region: " + str(region[-1]), tag="region", level=1)

    return region, region_size


def load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, dir_vortex, velocity_field, level, test, bitformat=np.float32, region=None, sim_characteristics=None, verbose=False, debug=False):
    '''
    Loads the data from the simulations for the given snapshots and prepares it for further analysis.
    This are the parameters we will need for each cell together with the magnetic field and the velocity,
    we read the information for each snap and divide it in the different fields.
    
    Args:
        - sims: list of simulation names
        - it: list of snapshots
        - a0: scale factor  at the present time (typically 1.0)
        - H0: Hubble constant at the present time
        - dir_grids: directory where the grids are stored
        - dir_gas: directory where the gas data is stored
        - dir_params: directory where the parameters are stored
        - dir_vortex: directory where the vortex data is stored
        - velocity_field: dictionary containing the target velocity field components to be processed (total, solenoidal, compressive)
        - level: level of the AMR grid to be used
        - test: Dictionary containing the parameters for the test fields:
            - test: boolean to use test fields or not
            - x_test, y_test, z_test: 3D grid coordinates.
            - k: Wave number for the sinusoidal test fields.
            - ω: Angular frequency for the sinusoidal test fields.
            - B0: Amplitude of the magnetic field.
        - bitformat: data type for the loaded fields (default is np.float32)
        - region: region coordinates to be used (default is None)
        - sim_characteristics: Dictionary with simulation characteristics (is_cooling, is_mascletB, etc.)
        - verbose: boolean to print the data type loaded or not (default is False)
        - debug: dictionary containing the parameters for the debug mode (if False, debug mode is disabled):
        
    Returns:
        - results: dictionary containing the loaded data:
            - grid_irr: index of the snapshot
            - grid_time: time of the snapshot
            - grid_zeta: redshift of the snapshot
            - grid_npatch: number of patches in the grid
            - grid_patchnx, grid_patchny, grid_patchnz: number of cells in each patch
            - grid_patchx, grid_patchy, grid_patchz: size of each patch
            - grid_patchrx, grid_patchry, grid_patchrz: position of each patch
            - grid_pare: parent patch of each patch
            - vector_levels: levels of refinement for each patch
            - clus_rho_rho_b: density contrast in the cluster
            - clus_vx, clus_vy, clus_vz: total velocity field components in the cluster
            - clus_vsolx, clus_vsoly, clus_vsolz: solenoidal velocity field components in the cluster
            - clus_vcompx, clus_vcompy, clus_vcompz: compressive velocity field components in the cluster
            - clus_cr0amr: cosmic ray energy density in the cluster (or refinement flag)
            - clus_solapst: solenoidal fraction in the cluster (or mask flag)
            - clus_kp: mask for valid patches
            - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
            - clus_B: normalized magnetic field magnitude in the cluster
            - clus_B2: normalized magnetic field squared in the cluster
            - clus_b2: magnetic field squared in the cluster
            - clus_v2: total velocity field squared in the cluster
            - clus_vsol2: solenoidal velocity field squared in the cluster
            - clus_vcomp2: compressible velocity field squared in the cluster
            - clus_pres: pressure (optional, None if not requested)
            - clus_pot: gravitational potential (optional, None if not requested)
            - clus_opot: old gravitational potential (optional, None if not requested)
            - clus_temp: temperature (optional, None if not requested)
            - clus_metalicity: metalicity (optional, None if not requested)
            - a: scale factor at the redshift zeta
            - E: E(z) function at the redshift zeta
            - H: Hubble parameter at the redshift zeta
            - rho_b: background density at the redshift zeta
        
    Author: Marco Molina
    '''
    # Load Simulation Data
    
    ## This are the parameters we will need for each cell together with the magnetic field and the velocity
    ## We read the information for each snap and divide it in the different fields
    
    if region[0] == None:
        region = None

    # Get simulation characteristics (default if not provided)
    if sim_characteristics is None:
        sim_characteristics = {
            "is_mascletB": True,
            "is_cooling": False,
            "has_cr0amr": True,
            "has_solapst": True,
            "output_pres": False,
            "output_pot": False,
            "output_opot": False,
            "output_temp": False,
            "output_metalicity": False
        }

    use_total_velocity = bool(velocity_field.get("total", True))
    use_solenoidal_velocity = bool(velocity_field.get("solenoidal", False))
    use_compressive_velocity = bool(velocity_field.get("compressive", False))
    vortex_requested = bool(use_solenoidal_velocity or use_compressive_velocity)
    
    # Log simulation characteristics if verbose
    if verbose:
        log_message(f"Loading data with characteristics: is_cooling={sim_characteristics.get('is_cooling', False)}, "
                    f"is_mascletB={sim_characteristics.get('is_mascletB', True)}")

    if test['test'] == False:
        # Read grid data using the reader
        grid = reader.read_grids(
            it=it,
            path=dir_grids + sims,
            parameters_path=dir_params,
            digits=5,
            read_general=True,
            read_patchnum=True,
            read_dmpartnum=False,
            read_patchcellextension=True,
            read_patchcellposition=True,
            read_patchposition=True,
            read_patchparent=True,
            nparray=True
        )

        # Unpack grid data with explicit variable names for clarity
        (
            grid_irr,
            grid_time,
            _,  # grid_nl (unused)
            _,  # grid_mass_dmpart (unused)
            grid_zeta,
            grid_npatch,
            grid_patchnx,
            grid_patchny,
            grid_patchnz,
            grid_patchx,
            grid_patchy,
            grid_patchz,
            grid_patchrx,
            grid_patchry,
            grid_patchrz,
            pare,
            *_
        ) = grid

        if verbose:
            log_message(
                f'Load order check: requested_it={it}, loaded_grid_irr={grid_irr}, '
                f'grid_time={grid_time:.6g}, grid_zeta={grid_zeta:.6g}',
                tag="order",
                level=1
            )
        
        # Only keep patches up to the desired AMR level and slice patch arrays accordingly
        grid_npatch[level+1:] = 0
        keep_count = int(1 + np.sum(grid_npatch))
        grid_patchnx = grid_patchnx[:keep_count]
        grid_patchny = grid_patchny[:keep_count]
        grid_patchnz = grid_patchnz[:keep_count]
        grid_patchx = grid_patchx[:keep_count]
        grid_patchy = grid_patchy[:keep_count]
        grid_patchz = grid_patchz[:keep_count]
        grid_patchrx = grid_patchrx[:keep_count]
        grid_patchry = grid_patchry[:keep_count]
        grid_patchrz = grid_patchrz[:keep_count]
        pare = pare[:keep_count]
        
        # DIAGNOSTIC: Check for patches in suspicious regions (z > 15 Mpc)
        if debug.get("patch_analysis", {}).get("enabled", False) == True if isinstance(debug, dict) else False:
            debug_module.analyze_patch_positions(
                grid_patchrx, grid_patchry, grid_patchrz,
                grid_patchnx, grid_patchny, grid_patchnz,
                pare, grid_npatch, dir_params=dir_params,
                suspicious_threshold=debug.get("patch_analysis", {}).get("suspicious_threshold", 15.0),
                verbose=True
            )

        # Read cluster data using simulation characteristics from config
        # Config defines both what exists (is_X, has_X) and what to read (read_X)
        clus_kwargs = reader.get_read_clus_kwargs(sim_characteristics, level, region)
        clus = reader.read_clus(
            it=it,
            path=dir_gas + sims,
            parameters_path=dir_params,
            digits=5,
            **clus_kwargs
        )

        # Unpack cluster data based on what was actually read
        clus_data = reader.unpack_clus_data(clus, clus_kwargs, region)

        if vortex_requested:
            vsolx, vsoly, vsolz, vcompx, vcompy, vcompz = reader.read_vortex_velocity_fields(
                it,
                path=dir_vortex + sims,
                parameters_path=dir_params,
                digits=5,
                grids_path=dir_grids + sims,
                grids_filename='grids',
                max_refined_level=level,
                read_solenoidal=use_solenoidal_velocity,
                read_compressive=use_compressive_velocity
            )
            if use_solenoidal_velocity:
                clus_vsolx, clus_vsoly, clus_vsolz = vsolx, vsoly, vsolz
            if use_compressive_velocity:
                clus_vcompx, clus_vcompy, clus_vcompz = vcompx, vcompy, vcompz
        else:
            clus_vsolx = clus_vsoly = clus_vsolz = None
            clus_vcompx = clus_vcompy = clus_vcompz = None

        # Extract the variables we need
        clus_rho_rho_b = clus_data['delta']
        if velocity_field.get("total", True):
            clus_vx = clus_data['vx']
            clus_vy = clus_data['vy']
            clus_vz = clus_data['vz']
        else:
            clus_vx = clus_vy = clus_vz = None
        clus_cr0amr = clus_data['cr0amr']
        clus_solapst = clus_data['solapst']
        clus_mbx = clus_data['Bx']
        clus_mby = clus_data['By']
        clus_mbz = clus_data['Bz']
        
        # Optional variables (may be None if not requested)
        clus_pres = clus_data['pres']
        clus_pot = clus_data['pot']
        clus_opot = clus_data['opot']
        clus_temp = clus_data['temp']
        clus_metalicity = clus_data['metalicity']
        clus_keep_patches = clus_data['keep_patches']
        
        # IMPORTANT: Verify that clus data matches the expected number of patches
        # This ensures configuration parameters are correct
        actual_num_patches = len(clus_rho_rho_b)
        expected_num_patches = keep_count
        
        if actual_num_patches != expected_num_patches:
            # Calculate which levels are present
            cumsum = 1
            max_level_in_data = 0
            for lev in range(1, len(grid_npatch)):
                if cumsum >= actual_num_patches:
                    break
                cumsum += grid_npatch[lev]
                if cumsum <= actual_num_patches:
                    max_level_in_data = lev
            
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Mismatch between grid and clus data!\n"
                f"{'='*80}\n"
                f"Expected patches (from grid file): {expected_num_patches}\n"
                f"Actual patches (from clus file):   {actual_num_patches}\n"
                f"\n"
                f"Grid npatch per level: {grid_npatch}\n"
                f"Maximum level in clus data: {max_level_in_data}\n"
                f"Requested level in config: {level}\n"
                f"\n"
                f"SOLUTION:\n"
                f"The 'nlevels' parameter in config.py should match the maximum refinement\n"
                f"level available in your simulation files.\n"
                f"\n"
                f"Please check your simulation files and update config.py:\n"
                f"  - If clus file only has {max_level_in_data} levels, set:\n"
                f"      IND_PARAMS['nlevels'] = [{max_level_in_data}]\n"
                f"      IND_PARAMS['level'] = [{max_level_in_data}]\n"
                f"      IND_PARAMS['up_to_level'] = [{max_level_in_data}]\n"
                f"\n"
                f"  - Or verify that your clus files contain all {level} levels\n"
                f"{'='*80}\n"
            )
            raise ValueError(error_msg)
        
        # Slice grid arrays to match clus data
        grid_patchnx = grid_patchnx[:keep_count]
        grid_patchny = grid_patchny[:keep_count]
        grid_patchnz = grid_patchnz[:keep_count]
        grid_patchx = grid_patchx[:keep_count]
        grid_patchy = grid_patchy[:keep_count]
        grid_patchz = grid_patchz[:keep_count]
        grid_patchrx = grid_patchrx[:keep_count]
        grid_patchry = grid_patchry[:keep_count]
        grid_patchrz = grid_patchrz[:keep_count]
        pare = pare[:keep_count]

    else:
        # Read grid data using the reader
        grid = reader.read_grids(
            it=it,
            path=dir_grids + sims,
            parameters_path=dir_params,
            digits=5,
            read_general=True,
            read_patchnum=False,
            read_dmpartnum=False,
            read_patchcellextension=False,
            read_patchcellposition=False,
            read_patchposition=False,
            read_patchparent=False,
            nparray=False
        )

        # Unpack grid data with explicit variable names for clarity
        (
            grid_irr,
            grid_time,
            _,  # grid_nl (unused)
            _,  # grid_mass_dmpart (unused)
            grid_zeta,
            *_
        ) = grid
        
        grid_patchrx = test['grid_patchrx_test']
        grid_patchry = test['grid_patchry_test']
        grid_patchrz = test['grid_patchrz_test']
        grid_patchnx = test['grid_patchnx_test']
        grid_patchny = test['grid_patchny_test']
        grid_patchnz = test['grid_patchnz_test']
        grid_npatch = test['grid_npatch_test']
        
    # Create vector_levels using the tools module
    vector_levels = utils.create_vector_levels(grid_npatch)

    a = a0 / (1 + grid_zeta)  # Scale factor at the redshift zeta
    E = utils.E(grid_zeta, omega_m, omega_lambda)
    H = H0*E
    rho_b = 3 * (H0)**2 * omega_m * (1 + grid_zeta)**3 # We compute the background density at this redshift
    # rho_b = 1
    
    if test['test'] == True:
        # TEST MODE: Only read delta (density), skip all other fields
        # Note: We still need to tell read_clus about file structure (is_mascletB, is_cooling)
        clus = reader.read_clus(
            it=it,
            path=dir_gas + sims,
            parameters_path=dir_params,
            digits=5,
            max_refined_level=level,
            output_delta=True,              # READ: density
            output_v=False,                 # SKIP: velocity
            output_pres=False,              # SKIP: pressure
            output_pot=False,               # SKIP: potential
            output_opot=False,              # SKIP: old potential
            output_temp=False,              # SKIP: temperature
            output_metalicity=False,        # SKIP: metalicity
            output_cr0amr=False,            # SKIP: refinement flag
            output_solapst=False,           # SKIP: solapst
            is_mascletB=sim_characteristics.get('is_mascletB', True),  # File structure info
            output_B=False,                 # SKIP: magnetic field
            is_cooling=sim_characteristics.get('is_cooling', False),   # File structure info
            verbose=False,
            read_region=region
        )

        # Unpack cluster data (only delta was read)
        (
            clus_rho_rho_b,
            *rest
        ) = clus
        
        rest = None
        clus_cr0amr = test['clus_cr0amr_test']
        clus_solapst = test['clus_solapst_test']
        
        clus_bv = numeric_test_fields(
            grid_time=grid_time,
            grid_npatch=grid_npatch,
            a=a,
            H=H,
            test_params=test
        )
        
        (
            clus_mbx,
            clus_mby,
            clus_mbz,
            clus_vx,
            clus_vy,
            clus_vz
        ) = clus_bv
    
    # Calculate number of patches based on ACTUAL data returned
    # Use the length of the clus arrays, not grid_npatch, because they may differ
    n = len(clus_rho_rho_b)
    
    # Determine mask for valid patches
    if region is not None and clus_keep_patches is not None:
        clus_kp = clus_keep_patches
    else:
        clus_kp = np.ones(n, dtype=bool)
    
    # Handle None values for optional variables
    # If cr0amr was not loaded, create default array
    if clus_cr0amr is None:
        clus_cr0amr = [np.ones_like(clus_rho_rho_b[p], dtype=bool) if bool(clus_kp[p]) else 0 for p in range(n)]
    
    # If solapst was not loaded, create default array  
    if clus_solapst is None:
        clus_solapst = [1 if p == 0 else (np.ones_like(clus_rho_rho_b[p], dtype=bool) if bool(clus_kp[p]) else 0) for p in range(n)]

    # Normalize magnetic field components
    # clus_Bx = [clus_mbx[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    # clus_By = [clus_mby[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    # clus_Bz = [clus_mbz[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    
    # B field was already normalized in the files, so we can directly assign it to Bx, By, Bz
    clus_Bx = [clus_mbx[p] if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_By = [clus_mby[p] if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_Bz = [clus_mbz[p] if bool(clus_kp[p]) else 0 for p in range(n)]
    
    if debug.get("divergence", {}).get("enabled", False) == True if isinstance(debug, dict) else False:
        debug_verbose = debug.get("divergence", {}).get("verbose", True) if isinstance(debug, dict) else True
        debug_module.compare_divergence_methods(
            clus_Bx,
            clus_By,
            clus_Bz,
            grid_npatch,
            clus_kp,
            grid_irr,
            dir_params,
            verbose=debug_verbose
        )
    
    # We denormalize the magnetic field components to get the physical magnetic field in code units
    clus_bx = [clus_mbx[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_by = [clus_mby[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_bz = [clus_mbz[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]

    if clus_vx is not None and clus_vy is not None and clus_vz is not None:
        clus_vx = [clus_vx[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vy = [clus_vy[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vz = [clus_vz[p] if bool(clus_kp[p]) else 0 for p in range(n)]
    if clus_vsolx is not None and clus_vsoly is not None and clus_vsolz is not None:
        clus_vsolx = [clus_vsolx[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vsoly = [clus_vsoly[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vsolz = [clus_vsolz[p] if bool(clus_kp[p]) else 0 for p in range(n)]
    if clus_vcompx is not None and clus_vcompy is not None and clus_vcompz is not None:
        clus_vcompx = [clus_vcompx[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vcompy = [clus_vcompy[p] if bool(clus_kp[p]) else 0 for p in range(n)]
        clus_vcompz = [clus_vcompz[p] if bool(clus_kp[p]) else 0 for p in range(n)]

    if verbose == True:
        log_message('Data type loaded for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype), tag="data", level=1)

    # Convert to float64 if not transforming to uniform grid
    if bitformat == np.float64:
        clus_rho_rho_b = [(1+clus_rho_rho_b[p]).astype(np.float64) if bool(clus_kp[p]) else (1+clus_rho_rho_b[p]) for p in range(n)] # Delta is (rho/rho_b) - 1
        clus_bx = [clus_bx[p].astype(np.float64) if bool(clus_kp[p]) else clus_bx[p] for p in range(n)]
        clus_by = [clus_by[p].astype(np.float64) if bool(clus_kp[p]) else clus_by[p] for p in range(n)]
        clus_bz = [clus_bz[p].astype(np.float64) if bool(clus_kp[p]) else clus_bz[p] for p in range(n)]
        clus_Bx = [clus_Bx[p].astype(np.float64) if bool(clus_kp[p]) else clus_Bx[p] for p in range(n)]
        clus_By = [clus_By[p].astype(np.float64) if bool(clus_kp[p]) else clus_By[p] for p in range(n)]
        clus_Bz = [clus_Bz[p].astype(np.float64) if bool(clus_kp[p]) else clus_Bz[p] for p in range(n)]
        if clus_vx is not None and clus_vy is not None and clus_vz is not None:
            clus_vx = [clus_vx[p].astype(np.float64) if bool(clus_kp[p]) else clus_vx[p] for p in range(n)]
            clus_vy = [clus_vy[p].astype(np.float64) if bool(clus_kp[p]) else clus_vy[p] for p in range(n)]
            clus_vz = [clus_vz[p].astype(np.float64) if bool(clus_kp[p]) else clus_vz[p] for p in range(n)]
        if clus_vsolx is not None and clus_vsoly is not None and clus_vsolz is not None:
            clus_vsolx = [clus_vsolx[p].astype(np.float64) if bool(clus_kp[p]) else clus_vsolx[p] for p in range(n)]
            clus_vsoly = [clus_vsoly[p].astype(np.float64) if bool(clus_kp[p]) else clus_vsoly[p] for p in range(n)]
            clus_vsolz = [clus_vsolz[p].astype(np.float64) if bool(clus_kp[p]) else clus_vsolz[p] for p in range(n)]
        if clus_vcompx is not None and clus_vcompy is not None and clus_vcompz is not None:
            clus_vcompx = [clus_vcompx[p].astype(np.float64) if bool(clus_kp[p]) else clus_vcompx[p] for p in range(n)]
            clus_vcompy = [clus_vcompy[p].astype(np.float64) if bool(clus_kp[p]) else clus_vcompy[p] for p in range(n)]
            clus_vcompz = [clus_vcompz[p].astype(np.float64) if bool(clus_kp[p]) else clus_vcompz[p] for p in range(n)]

    clus_b2 = [clus_bx[p]**2 + clus_by[p]**2 + clus_bz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_B2 = [clus_Bx[p]**2 + clus_By[p]**2 + clus_Bz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_B = [np.sqrt(clus_B2[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    if clus_vx is not None and clus_vy is not None and clus_vz is not None:
        clus_v2 = [clus_vx[p]**2 + clus_vy[p]**2 + clus_vz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        clus_v2 = None
    if clus_vsolx is not None and clus_vsoly is not None and clus_vsolz is not None:
        clus_vsol2 = [clus_vsolx[p]**2 + clus_vsoly[p]**2 + clus_vsolz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        clus_vsol2 = None
    if clus_vcompx is not None and clus_vcompy is not None and clus_vcompz is not None:
        clus_vcomp2 = [clus_vcompx[p]**2 + clus_vcompy[p]**2 + clus_vcompz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        clus_vcomp2 = None

    if verbose == True:
        log_message('Working data type for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype), tag="data", level=1)
        
    results = {
        'grid_irr': grid_irr,
        'grid_time': grid_time,
        'grid_zeta': grid_zeta,
        'grid_npatch': grid_npatch,
        'grid_patchnx': grid_patchnx,
        'grid_patchny': grid_patchny,
        'grid_patchnz': grid_patchnz,
        'grid_patchx': grid_patchx,
        'grid_patchy': grid_patchy,
        'grid_patchz': grid_patchz,
        'grid_patchrx': grid_patchrx,
        'grid_patchry': grid_patchry,
        'grid_patchrz': grid_patchrz,
        'grid_pare': pare,
        'vector_levels': vector_levels,
        'clus_rho_rho_b': clus_rho_rho_b,
        'clus_vx': clus_vx,
        'clus_vy': clus_vy,
        'clus_vz': clus_vz,
        'clus_vsolx': clus_vsolx,
        'clus_vsoly': clus_vsoly,
        'clus_vsolz': clus_vsolz,
        'clus_vcompx': clus_vcompx,
        'clus_vcompy': clus_vcompy,
        'clus_vcompz': clus_vcompz,
        'clus_cr0amr': clus_cr0amr,
        'clus_solapst': clus_solapst,
        'clus_kp': clus_kp,
        'clus_Bx': clus_Bx,
        'clus_By': clus_By,
        'clus_Bz': clus_Bz,
        'clus_bx': clus_bx,
        'clus_by': clus_by,
        'clus_bz': clus_bz,
        'clus_B': clus_B,
        'clus_b2': clus_b2,
        'clus_B2': clus_B2,
        'clus_v2': clus_v2,
        'clus_vsol2': clus_vsol2,
        'clus_vcomp2': clus_vcomp2,
        # Optional variables (may be None if not requested)
        'clus_pres': clus_pres,
        'clus_pot': clus_pot,
        'clus_opot': clus_opot,
        'clus_temp': clus_temp,
        'clus_metalicity': clus_metalicity,
        'a': a,
        'E': E,
        'H': H,
        'rho_b': rho_b
    }
    
    return results


def vectorial_quantities(components, clus_Bx, clus_By, clus_Bz,
                        active_velocities,
                        clus_kp, grid_npatch, grid_irr,
                        dx, stencil=3, verbose=False):
    '''
    Computes the vectorial calculus quantities of interest for the magnetic field and velocity field.
    Only the the necessary quantities are computed based on the components specified in the config "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
        - active_velocities: dictionary containing the active velocity fields
        - clus_kp: mask for valid patches
        - grid_npatch: number of patches in the grid
        - grid_irr: index of the snapshot
        - dx: size of the cells in Mpc
        - stencil: stencil to be used for the calculations
        - buffer_active: boolean to use buffer zones in the differential calculations (default is False)
        - nghost: number of ghost cells to be used in the differential calculations (default is 1)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed vectorial quantities (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - diver_B: divergence of the magnetic field
            - diver_v: divergence of the velocity field
            - v_nabla_B_x, v_nabla_B_y, v_nabla_B_z: directional derivative of the magnetic field along the velocity field
            - B_nabla_v_x, B_nabla_v_y, B_nabla_v_z: directional derivative of the velocity field along the magnetic field
            - v_X_B_x, v_X_B_y, v_X_B_z: cross product of the velocity and magnetic field
            - curl_v_X_B_x, curl_v_X_B_y, curl_v_X_B_z: total induction as the curl of the cross product of the velocity and magnetic field with drag term
        
    Author: Marco Molina
    '''
    
    # Vectorial calculus

    ## Here we calculate the different vectorial calculus quantities of our interest using the diff module.
    
    start_time_vector = time.time() # Record the start time
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    if clus_kp is None:
        clus_kp = np.ones(n, dtype=bool)
    
    results = {}
    
    if components.get('divergence', False):
        ### We compute the divergence of the magnetic field
        results['diver_B'] = diff.divergence(clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp, stencil)
    else:
        results['diver_B'] = zero

    ### Velocity dependent quantities, iterate over the active velocity fields (total, solenoidal, compressive)
    for vel_type, v_field in active_velocities.items():
        sfx = f"_{vel_type}" if vel_type != "total" else ""
        
        vx, vy, vz = v_field['x'], v_field['y'], v_field['z']
        
        if components.get('compression', False):
            ### We compute the divergence of the velocity field
            results[f'diver_v{sfx}'] = diff.divergence(vx, vy, vz, dx, grid_npatch, clus_kp, stencil)
        else:
            results[f'diver_v{sfx}'] = zero
            
        if components.get('stretching', False):
            ### We compute the directional derivative of the velocity field along the magnetic field
            results[f'B_nabla_v_x{sfx}'], results[f'B_nabla_v_y{sfx}'], results[f'B_nabla_v_z{sfx}'] = \
                diff.directional_derivative_vector_field(vx, vy, vz, clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp, stencil)
        else:
            results[f'B_nabla_v_x{sfx}'] = results[f'B_nabla_v_y{sfx}'] = results[f'B_nabla_v_z{sfx}'] = zero
            
        if components.get('advection', False):
            ### We compute the directional derivative of the magnetic field along the velocity field
            results[f'v_nabla_B_x{sfx}'], results[f'v_nabla_B_y{sfx}'], results[f'v_nabla_B_z{sfx}'] = \
                diff.directional_derivative_vector_field(clus_Bx, clus_By, clus_Bz, vx, vy, vz, dx, grid_npatch, clus_kp, stencil)
        else:
            results[f'v_nabla_B_x{sfx}'] = results[f'v_nabla_B_y{sfx}'] = results[f'v_nabla_B_z{sfx}'] = zero
            
        if components.get('total', False):
            ### We compute the cross product of the velocity and magnetic field
            v_X_B_x = [vy[p] * clus_Bz[p] - vz[p] * clus_By[p] if clus_kp[p] else 0 for p in range(n)] # We run across all the patches with the levels we are interested in
            v_X_B_y = [vz[p] * clus_Bx[p] - vx[p] * clus_Bz[p] if clus_kp[p] else 0 for p in range(n)] # We only want the patches inside the region of interest
            v_X_B_z = [vx[p] * clus_By[p] - vy[p] * clus_Bx[p] if clus_kp[p] else 0 for p in range(n)]
            
            ### The total induction as the curl of the cross product of the velocity and magnetic field with drag term.
            results[f'curl_v_X_B_x{sfx}'], results[f'curl_v_X_B_y{sfx}'], results[f'curl_v_X_B_z{sfx}'] = \
                diff.curl(v_X_B_x, v_X_B_y, v_X_B_z, dx, grid_npatch, clus_kp, stencil)
        else:
            results[f'curl_v_X_B_x{sfx}'] = results[f'curl_v_X_B_y{sfx}'] = results[f'curl_v_X_B_z{sfx}'] = zero

    end_time_vector = time.time()
    if verbose == True:
        total_time_vector = end_time_vector - start_time_vector
        log_message('Time for vector calculations in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_vector))), tag="vector", level=1)
        
    return results


def _debug_enabled(debug_params):
    """Check if any debug module is enabled (not counting percentile_params or other config)."""
    if not isinstance(debug_params, dict):
        return False
    # Only check known debug modules, NOT config fields like percentile_params
    known_debug_modules = {"buffer", "divergence", "field_analysis", "scan_animation"}
    for key, value in debug_params.items():
        if key in known_debug_modules and isinstance(value, dict) and value.get("enabled", False):
            return True
    return False


def _collect_divergence_values(diver_B, kept_patches=None, use_abs=True, exclude_zeros=True):
    vals_list = []
    if diver_B is None:
        return np.array([])
    for i, patch in enumerate(diver_B):
        if kept_patches is not None and not kept_patches[i]:
            continue
        if patch is None:
            continue
        if np.isscalar(patch):
            val = float(patch)
            if use_abs:
                val = abs(val)
            if exclude_zeros and val == 0.0:
                continue
            if np.isfinite(val):
                vals_list.append(np.array([val], dtype=float))
            continue
        arr = np.asarray(patch, dtype=float)
        if use_abs:
            arr = np.abs(arr)
        arr = arr[np.isfinite(arr)]
        if exclude_zeros:
            arr = arr[arr != 0.0]
        if arr.size:
            vals_list.append(arr.ravel())
    if vals_list:
        return np.concatenate(vals_list)
    return np.array([])


def filter_divergence_outliers(diver_B, kept_patches=None, method="mask",
                                percentile=99, use_abs=True, exclude_zeros=True,
                                verbose=False):
    method = method.lower()
    if method not in ("mask", "clip"):
        raise ValueError("divergence_filter method must be 'mask' or 'clip'.")

    vals = _collect_divergence_values(
        diver_B,
        kept_patches=kept_patches,
        use_abs=use_abs,
        exclude_zeros=exclude_zeros
    )
    if vals.size == 0:
        if verbose:
            log_message(
                f"Divergence filter: NO values to filter (empty array after collection)",
                tag="divergence_filter",
                level=1
            )
        return diver_B, None

    threshold = float(np.percentile(vals, percentile))
    if verbose:
        frac = float(np.mean(vals > threshold)) if vals.size else 0.0
        log_message(
            f"Divergence filter: method={method}, percentile={percentile}, threshold={threshold:.3e}, outlier_frac={frac:.3f}",
            tag="divergence_filter",
            level=1
        )

    filtered = []
    for i, patch in enumerate(diver_B):
        if patch is None:
            filtered.append(patch)
            continue
        if kept_patches is not None and not kept_patches[i]:
            filtered.append(0 if np.isscalar(patch) else np.zeros_like(patch))
            continue
        if np.isscalar(patch):
            val = float(patch)
            if method == "clip":
                if use_abs:
                    val = max(min(val, threshold), -threshold)
                else:
                    val = min(val, threshold)
            else:
                keep = abs(val) <= threshold if use_abs else val <= threshold
                val = val if keep else 0.0
            filtered.append(val)
            continue
        arr = np.asarray(patch)
        if method == "clip":
            if use_abs:
                arr = np.clip(arr, -threshold, threshold)
            else:
                arr = np.minimum(arr, threshold)
        else:
            if use_abs:
                mask = np.abs(arr) <= threshold
            else:
                mask = arr <= threshold
            arr = np.where(mask, arr, 0)
        filtered.append(arr)

    return filtered, threshold
    

def induction_equation(components, vectorial_quantities,
                        clus_Bx, clus_By, clus_Bz,
                        active_velocities,
                        clus_kp, grid_npatch, grid_irr,
                        H, a, mag=False, verbose=False):
    '''
    Computes the components of the cosmological magnetic induction equation and their magnitudes.
    Only computes the components that are set to True in "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - vectorial_quantities: dictionary containing the vectorial quantities computed in the previous step
            - diver_B: divergence of the magnetic field
            - diver_v: divergence of the velocity field
            - B_nabla_v_x, B_nabla_v_y, B_nabla_v_z: directional derivative of the velocity field along the magnetic field
            - v_nabla_B_x, v_nabla_B_y, v_nabla_B_z: directional derivative of the magnetic field along the velocity field
            - curl_v_X_B_x, curl_v_X_B_y, curl_v_X_B_z: total induction as the curl of the cross product of the velocity and magnetic field with drag term
        - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
        - active_velocities: dictionary containing the active velocity fields
            - e.g. total: total velocity field (clus_vx, clus_vy, clus_vz)
        - clus_kp: mask for valid patches
        - grid_npatch: number of patches in the grid
        - grid_irr: index of the snapshot
        - H: Hubble parameter
        - a: scale factor of the universe
        - mag: boolean to compute the magnitudes of the components (default is False)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed components of the magnetic induction equation (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - MIE_diver_x, MIE_diver_y, MIE_diver_z: null divergence of the magnetic field
            - MIE_compres_x, MIE_compres_y, MIE_compres_z: compressive component of the magnetic field induction
            - MIE_stretch_x, MIE_stretch_y, MIE_stretch_z: stretching component of the magnetic field induction
            - MIE_advec_x, MIE_advec_y, MIE_advec_z: advection component of the magnetic field induction
            - MIE_drag_x, MIE_drag_y, MIE_drag_z: cosmic drag component of the magnetic field induction
            - MIE_total_x, MIE_total_y, MIE_total_z: total magnetic induction energy in the compact way
        - magnitudes: dictionary containing the magnitudes of the components if mag is True:
            - MIE_diver_mag, MIE_drag_mag, MIE_compres_mag, MIE_stretch_mag, MIE_advec_mag, MIE_total_mag: magnitudes
        
    Author: Marco Molina
    '''
    # Magnetic Induction Equation
    
    ## In this section we are going to compute the cosmological induction equation and its components, calculating them with the results obtained before.
    ## This will be usefull to plot fluyd maps as the quantities involved are vectors.

    ### We compute here each contribution to the magnetic field induction.
    
    start_time_induction_terms = time.time() # Record the start time
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    
    if clus_kp is None:
        clus_kp = np.ones(n, dtype=bool)
    
    results = {}
    magnitudes = {} if mag else None
    
    ### Cosmic drag does not depend on velocity, we calculate it once
    if components.get('drag', False) or components.get('total', False):
        factor_drag = -0.5 * H
        results['MIE_drag_x'] = [(factor_drag * clus_Bx[p] if clus_kp[p] else 0) for p in range(n)]
        results['MIE_drag_y'] = [(factor_drag * clus_By[p] if clus_kp[p] else 0) for p in range(n)]
        results['MIE_drag_z'] = [(factor_drag * clus_Bz[p] if clus_kp[p] else 0) for p in range(n)]
        if mag:
            magnitudes['MIE_drag_mag'] = utils.magnitude(results['MIE_drag_x'], results['MIE_drag_y'], results['MIE_drag_z'], clus_kp)
    else:
        results['MIE_drag_x'] = results['MIE_drag_y'] = results['MIE_drag_z'] = zero
        if mag:
            magnitudes['MIE_drag_mag'] = zero

    ### The rest of the components depend on the velocity field, we iterate over the active velocity fields (total, solenoidal, compressive)
    inv_a = 1.0 / a
    
    for vel_type, v_field in active_velocities.items():
        sfx = f"_{vel_type}" if vel_type != "total" else ""
        vx, vy, vz = v_field['x'], v_field['y'], v_field['z']
        
        if components.get('divergence', False):
            ### The null divergence of the magnetic field for numerical error purposes.
            div_B = vectorial_quantities['diver_B']
            results[f'MIE_diver_x{sfx}'] = [(inv_a * vx[p] * div_B[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_diver_y{sfx}'] = [(inv_a * vy[p] * div_B[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_diver_z{sfx}'] = [(inv_a * vz[p] * div_B[p] if clus_kp[p] else 0) for p in range(n)]
        else:
            results[f'MIE_diver_x{sfx}'] = results[f'MIE_diver_y{sfx}'] = results[f'MIE_diver_z{sfx}'] = zero

        if components.get('compression', False):
            ### The compressive component.
            div_v = vectorial_quantities[f'diver_v{sfx}']
            results[f'MIE_compres_x{sfx}'] = [(-inv_a * clus_Bx[p] * div_v[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_compres_y{sfx}'] = [(-inv_a * clus_By[p] * div_v[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_compres_z{sfx}'] = [(-inv_a * clus_Bz[p] * div_v[p] if clus_kp[p] else 0) for p in range(n)]
        else:
            results[f'MIE_compres_x{sfx}'] = results[f'MIE_compres_y{sfx}'] = results[f'MIE_compres_z{sfx}'] = zero

        if components.get('stretching', False):
            ### The stretching component.
            results[f'MIE_stretch_x{sfx}'] = [(inv_a * vectorial_quantities[f'B_nabla_v_x{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_stretch_y{sfx}'] = [(inv_a * vectorial_quantities[f'B_nabla_v_y{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_stretch_z{sfx}'] = [(inv_a * vectorial_quantities[f'B_nabla_v_z{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
        else:
            results[f'MIE_stretch_x{sfx}'] = results[f'MIE_stretch_y{sfx}'] = results[f'MIE_stretch_z{sfx}'] = zero

        if components.get('advection', False):
            ### The advection component.
            results[f'MIE_advec_x{sfx}'] = [(-inv_a * vectorial_quantities[f'v_nabla_B_x{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_advec_y{sfx}'] = [(-inv_a * vectorial_quantities[f'v_nabla_B_y{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_advec_z{sfx}'] = [(-inv_a * vectorial_quantities[f'v_nabla_B_z{sfx}'][p] if clus_kp[p] else 0) for p in range(n)]
        else:
            results[f'MIE_advec_x{sfx}'] = results[f'MIE_advec_y{sfx}'] = results[f'MIE_advec_z{sfx}'] = zero

        if components.get('total', False):
            ### The total magnetic induction energy in the compact way.
            curl_v_B_x = vectorial_quantities[f'curl_v_X_B_x{sfx}']
            curl_v_B_y = vectorial_quantities[f'curl_v_X_B_y{sfx}']
            curl_v_B_z = vectorial_quantities[f'curl_v_X_B_z{sfx}']
            
            drag_x, drag_y, drag_z = results['MIE_drag_x'], results['MIE_drag_y'], results['MIE_drag_z']
            
            results[f'MIE_total_x{sfx}'] = [(inv_a * curl_v_B_x[p] + drag_x[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_total_y{sfx}'] = [(inv_a * curl_v_B_y[p] + drag_y[p] if clus_kp[p] else 0) for p in range(n)]
            results[f'MIE_total_z{sfx}'] = [(inv_a * curl_v_B_z[p] + drag_z[p] if clus_kp[p] else 0) for p in range(n)]
        else:
            results[f'MIE_total_x{sfx}'] = results[f'MIE_total_y{sfx}'] = results[f'MIE_total_z{sfx}'] = zero

        # Magnitudes específicas por tipo de velocidad
        if mag:
            prefixes = [('divergence', 'MIE_diver'), ('compression', 'MIE_compres'), 
                        ('stretching', 'MIE_stretch'), ('advection', 'MIE_advec'), ('total', 'MIE_total')]
            for comp_key, pref in prefixes:
                if components.get(comp_key, False):
                    magnitudes[f'{pref}_mag{sfx}'] = utils.magnitude(
                        results[f'{pref}_x{sfx}'], results[f'{pref}_y{sfx}'], results[f'{pref}_z{sfx}'], clus_kp
                    )
                else:
                    magnitudes[f'{pref}_mag{sfx}'] = zero
    
    if verbose:
        total_time = time.time() - start_time_induction_terms
        log_message(f"Time for calculating induction terms in snap {grid_irr}: {strftime('%H:%M:%S', gmtime(total_time))}", tag="induction", level=1)

    return results, magnitudes


def induction_equation_energy(components, velocity_field, induction_equation,
                            clus_Bx, clus_By, clus_Bz,
                            clus_rho_rho_b, clus_v2,
                            clus_kp, grid_npatch, grid_irr,
                            verbose=False):
    '''
    Computes the components of the cosmological magnetic induction equation in terms of the magnetic energy and its components.
    This will be useful to calculate volumetric integrals and energy budgets as the quantities involved are scalars.
    Only computes the components that are set to True in "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - velocity_field: string indicating the type of velocity fields to be used (e.g. "total", "solenoidal", "compressive")
        - induction_equation: dictionary containing the components of the magnetic induction equation computed in the previous step
            - MIE_diver_x, MIE_diver_y, MIE_diver_z: null divergence of the magnetic field
            - MIE_compres_x, MIE_compres_y, MIE_compres_z: compressive component of the magnetic field induction
            - MIE_stretch_x, MIE_stretch_y, MIE_stretch_z: stretching component of the magnetic field induction
            - MIE_advec_x, MIE_advec_y, MIE_advec_z: advection component of the magnetic field induction
            - MIE_drag_x, MIE_drag_y, MIE_drag_z: cosmic drag component of the magnetic field induction
            - MIE_total_x, MIE_total_y, MIE_total_z: total magnetic induction energy in the compact way
        - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
        - clus_rho_rho_b: density contrast of the cluster
        - clus_v2: squared velocity field
        - grid_npatch: number of patches in the grid
        - clus_kp: mask for valid patches
        - grid_irr: index of the snapshot
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed components of the magnetic induction equation in terms of the magnetic energy (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - MIE_diver_B2: null divergence of the magnetic field energy
            - MIE_compres_B2: compressive component of the magnetic field induction energy
            - MIE_stretch_B2: stretching component of the magnetic field induction energy
            - MIE_advec_B2: advection component of the magnetic field induction energy
            - MIE_drag_B2: cosmic drag component of the magnetic field induction energy
            - MIE_total_B2: total magnetic induction energy in the compact way
            - kinetic_energy_density: kinetic energy density of the cluster
        
    Author: Marco Molina
    '''
    # Magnetic Induction Equation in Terms of the Magnetic Energy
    
    ## In this section we are going to compute the cosmological induction equation in terms of the magnetic energy and its components, calculating them with the results obtained before.
    ## This will be usefull to calculate volumetric integrals and energy budgets as the quantities involved are scalars.
    
    ### We compute here each contribution to the magnetic fiel induction.

    start_time_induction_energy_terms = time.time() # Record the start time
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    
    if clus_kp is None:
        clus_kp = np.ones(n, dtype=bool)
    
    results = {}
    
    ## The cosmic drag term is independent of the velocity field, so we can compute it directly.    
    if components.get('drag', False):
        results['MIE_drag_B2'] = [
            (clus_Bx[p] * induction_equation['MIE_drag_x'][p] +
             clus_By[p] * induction_equation['MIE_drag_y'][p] +
             clus_Bz[p] * induction_equation['MIE_drag_z'][p] if clus_kp[p] else 0)
            for p in range(n)
        ]
    else:
        results['MIE_drag_B2'] = zero
        
    velocity_terms = [
        ('divergence', 'MIE_diver'),
        ('compression', 'MIE_compres'),
        ('stretching', 'MIE_stretch'),
        ('advection', 'MIE_advec'),
        ('total', 'MIE_total')
    ]
    
    velocity_mappings = [
        ("total", ""),
        ("solenoidal", "_solenoidal"),
        ("compressive", "_compressive")
    ]
    
    for key, sfx in velocity_mappings:
        # Evaluamos el trigger booleano del archivo de configuración
        if velocity_field.get(key, False):
            for comp_key, pref in velocity_terms:
                if components.get(comp_key, False):
                    # Producto escalar seguro basado estrictamente en el trigger activo
                    results[f'{pref}_B2{sfx}'] = [
                        (clus_Bx[p] * induction_equation[f'{pref}_x{sfx}'][p] +
                         clus_By[p] * induction_equation[f'{pref}_y{sfx}'][p] +
                         clus_Bz[p] * induction_equation[f'{pref}_z{sfx}'][p] if clus_kp[p] else 0)
                        for p in range(n)
                    ]
                else:
                    results[f'{pref}_B2{sfx}'] = zero

    ## The kinetic energy.

    if components.get('kinetic_energy', True) and clus_rho_rho_b is not None:
        results['kinetic_energy_density'] = [
            (0.5 * clus_rho_rho_b[p] * clus_v2[p] if clus_kp[p] else 0)
            for p in range(n)
        ]
    else:
        results['kinetic_energy_density'] = zero
    
    if verbose:
        total_time = time.time() - start_time_induction_energy_terms
        log_message(f"Time for calculating energy induction terms in snap {grid_irr}: {strftime('%H:%M:%S', gmtime(total_time))}", tag="induction_energy", level=1)
        
    return results


def production_dissipation_fields(components, velocity_field, induction_energy,
                                clus_kp, grid_npatch, grid_irr,
                                verbose=False):
    '''
    Splits each scalar induction-energy component into production (positive) and
    dissipation (negative) parts cell-wise.

    For each component Gamma_i this builds:
        - Gamma_i_plus  = max(Gamma_i, 0)
        - Gamma_i_minus = max(-Gamma_i, 0)

    Args:
        - components: dictionary of enabled induction components
        - velocity_field: dictionary of enabled velocity fields
        - induction_energy: dictionary produced by induction_equation_energy
        - clus_kp: mask for valid patches
        - grid_npatch: number of patches in the grid
        - grid_irr: snapshot index
        - verbose: whether to print timing information

    Returns:
        - results: dictionary with patch-wise arrays for production/dissipation terms

    Author: Marco Molina
    '''

    start_time_pd_terms = time.time()

    n = 1 + np.sum(grid_npatch)
    zero = [0] * n

    if clus_kp is None:
        clus_kp = np.ones(n, dtype=bool)

    results = {}
    
    velocity_terms = [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('total', 'MIE_total_B2')
    ]

    velocity_mappings = [
        ("total", ""),
        ("solenoidal", "_solenoidal"),
        ("compressive", "_compressive")
    ]
    
    ## Handle cosmic drag separately since it doesn't depend on the velocity field
    if components.get('drag', False):
        results['MIE_drag_B2_prod'] = [np.maximum(induction_energy['MIE_drag_B2'][p], 0.0) if clus_kp[p] else 0 for p in range(n)]
        results['MIE_drag_B2_diss'] = [np.maximum(-induction_energy['MIE_drag_B2'][p], 0.0) if clus_kp[p] else 0 for p in range(n)]
    else:
        results['MIE_drag_B2_prod'] = results['MIE_drag_B2_diss'] = zero

    for key, sfx in velocity_mappings:
        if velocity_field.get(key, False):
            for comp_key, pref in velocity_terms:
                full_prefix = f"{pref}{sfx}"
                
                if components.get(comp_key, False):
                    # Separación Celda a Celda (Cell-wise split)
                    results[f'{full_prefix}_prod'] = [
                        np.maximum(induction_energy[full_prefix][p], 0.0) if clus_kp[p] else 0
                        for p in range(n)
                    ]
                    results[f'{full_prefix}_diss'] = [
                        np.maximum(-induction_energy[full_prefix][p], 0.0) if clus_kp[p] else 0
                        for p in range(n)
                    ]
                else:
                    results[f'{full_prefix}_prod'] = results[f'{full_prefix}_diss'] = zero

    if verbose:
        total_time = time.time() - start_time_pd_terms
        log_message(f"Time for production/dissipation split in snap {grid_irr}: {strftime('%H:%M:%S', gmtime(total_time))}", tag="prod_diss", level=1)

    return results
    

def induction_vol_integral(components, velocity_field, induction_energy, clus_b2,
                            clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_zeta, grid_npatch, up_to_level,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, region_coords, rad,
                            units =1, production_dissipation=None,
                            rho_b=None,
                            volume_coordinates='physical', normalize_by_volume=False,
                            compute_induction_integrals=True,
                            compute_fractional_integrals=False,
                            integration_label='induction',
                            verbose=False):
    '''
    Computes the volume integral of the magnetic energy density and its components, as well as the induced magnetic energy.
    This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied volume. The kinetic energy
    density is also computed.
    Only computes the components that are set to True in "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - velocity_field: string indicating the type of velocity fields to be used (e.g. "total", "solenoidal", "compressive")
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy computed in the previous step
            - MIE_diver_B2: null divergence of the magnetic field energy
            - MIE_compres_B2: compressive component of the magnetic field induction energy
            - MIE_stretch_B2: stretching component of the magnetic field induction energy
            - MIE_advec_B2: advection component of the magnetic field induction energy
            - MIE_drag_B2: cosmic drag component of the magnetic field induction energy
            - MIE_total_B2: total magnetic induction energy in the compact way
            - kinetic_energy_density: kinetic energy density of the cluster
        - clus_b2: magnetic energy density in the cluster
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - clus_kp: mask for valid patches
        - grid_irr: index of the snapshot
        - grid_zeta: redshift of the snapshot
        - grid_npatch: number of patches in the grid
        - up_to_level: maximum refinement level to be considered
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - it: index of the snapshot in the simulation
        - sims: name of the simulation
        - nmax: maximum number of patches
        - size: size of the grid
        - coords: coordinates of the center of the integration grid
        - region_coords: coordinates defining the region of interest
        - rad: radius of the grid
        - units: factor to convert the units multiplied by the final result (default is 1)
        - production_dissipation: dict or bool to enable production/dissipation integral outputs
        - rho_b: background density at the snapshot. Used when production_dissipation['normalized'] is False
        - volume_coordinates: 'physical' or 'comoving' integration differential for dV
        - normalize_by_volume: if True, divide each field integral by total integration volume
        - compute_induction_integrals: if True, compute base induction integrals (int_MIE_*, int_b2, int_kinetic_energy, volume)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed volume integrals (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - int_MIE_diver_B2: volume integral of the null divergence of the magnetic field energy
            - int_MIE_compres_B2: volume integral of the compressive component
            - int_MIE_stretch_B2: volume integral of the stretching component
            - int_MIE_advec_B2: volume integral of the advection component
            - int_MIE_drag_B2: volume integral of the cosmic drag component
            - int_MIE_total_B2: volume integral of the total magnetic induction energy
            - int_kinetic_energy: volume integral of the kinetic energy density
            - int_b2: volume integral of the magnetic energy density
            - volume: volume of the studied region
            - production/dissipation integrals if production_dissipation is enabled:
                - int_MIE_diver_B2_prod, int_MIE_diver_B2_diss
                - int_MIE_compres_B2_prod, int_MIE_compres_B2_diss
                - int_MIE_stretch_B2_prod, int_MIE_stretch_B2_diss
                - int_MIE_advec_B2_prod, int_MIE_advec_B2_diss
                - int_MIE_drag_B2_prod, int_MIE_drag_B2_diss
                - int_MIE_total_B2_prod_compact, int_MIE_total_B2_diss_compact
                - int_MIE_total_B2_prod_itemized, int_MIE_total_B2_diss_itemized
                - int_PD_iota: fractional imbalance between total production and dissipation (only if compute_fractional_integrals is True)
        
    Author: Marco Molina
    '''
    ## Here we compute the volume integral of the magnetic energy density and its components
    
    start_time_induction = time.time() # Record the start time

    results = {}
    log_prefix = f'[{integration_label}] '

    velocity_field = velocity_field or {"total": True}
    velocity_mappings = [
        ("total", ""),
        ("solenoidal", "_solenoidal"),
        ("compressive", "_compressive")
    ]

    velocity_terms = [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('total', 'MIE_total_B2')
    ]

    def _integrate_field(field, vol=False):
        return utils.vol_integral(
            field, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
            size, nmax, coords, region_coords, rad, a0_masclet, units,
            kept_patches=clus_kp, vol=vol,
            volume_coordinates=volume_coordinates,
            normalize_by_volume=normalize_by_volume
        )
    
    ### We go first with the cosmic drag term, which is independent of the velocity field.
    if compute_induction_integrals and components.get('drag', False):
        results['int_MIE_drag_B2'] = _integrate_field(induction_energy['MIE_drag_B2'])
    else:
        results['int_MIE_drag_B2'] = 0.0
    
    for vel_key, sfx in velocity_mappings:
        if velocity_field.get(vel_key, False):
            for comp_key, pref in velocity_terms:
                full_key = f"{pref}{sfx}"
                if compute_induction_integrals and components.get(comp_key, False):
                    results[f'int_{full_key}'] = _integrate_field(induction_energy[full_key])
                    if verbose == True:
                        log_message(
                            f'{log_prefix}Snap {it} in {sims}: {comp_key} ({vel_key}) energy density volume integral done',
                            tag="integral",
                            level=1
                        )
                else:
                    results[f'int_{full_key}'] = 0.0

    pd_enabled = False
    if isinstance(production_dissipation, dict):
        pd_enabled = bool(production_dissipation.get('enabled', False))
    elif production_dissipation is not None:
        pd_enabled = bool(production_dissipation)

    if pd_enabled:
            rho_factor = 1.0
            if isinstance(production_dissipation, dict) and not production_dissipation.get('normalized', True):
                try:
                    rho_factor = float(rho_b)
                except (TypeError, ValueError):
                    rho_factor = 1.0
                    if verbose:
                        log_message(
                            f"Snap {it} in {sims}: invalid rho_b for physical P/D scaling; falling back to normalized scaling.",
                            tag="integral", level=1
                        )

            if components.get('drag', False):
                p_drag = rho_factor * _integrate_field(induction_energy['MIE_drag_B2_prod'])
                d_drag = rho_factor * _integrate_field(induction_energy['MIE_drag_B2_diss'])
                results['int_MIE_drag_B2_prod'] = p_drag
                results['int_MIE_drag_B2_diss'] = d_drag
            else:
                results['int_MIE_drag_B2_prod'] = results['int_MIE_drag_B2_diss'] = 0.0

            if verbose:
                log_message(f'{log_prefix}Snap {it} in {sims}: drag P/D volume integral done', tag="integral", level=1)

            itemized_enabled = bool(components.get('itemized', False))
            velocity_terms = [
                ('divergence', 'MIE_diver_B2', 'itemized'),
                ('compression', 'MIE_compres_B2', 'itemized'),
                ('stretching', 'MIE_stretch_B2', 'itemized'),
                ('advection', 'MIE_advec_B2', 'itemized'),
                ('total', 'MIE_total_B2', 'compact')
            ]

            active_families = [(vel_key, sfx) for vel_key, sfx in velocity_mappings if velocity_field.get(vel_key, False)]

            for vel_key, sfx in velocity_mappings:
                if velocity_field.get(vel_key, False):
                    family_itemized_prod = 0.0
                    family_itemized_diss = 0.0

                    # Drag is velocity-independent so it is included in every active family itemized total if present.
                    if itemized_enabled and components.get('drag', False):
                        family_itemized_prod += float(results.get('int_MIE_drag_B2_prod', 0.0))
                        family_itemized_diss += float(results.get('int_MIE_drag_B2_diss', 0.0))

                    for comp_key, pref, pd_mode in velocity_terms:
                        full_key = f"{pref}{sfx}"
                        
                        if components.get(comp_key, False):
                            p_i = rho_factor * _integrate_field(induction_energy[f'{full_key}_prod'])
                            d_i = rho_factor * _integrate_field(induction_energy[f'{full_key}_diss'])

                            if pd_mode == 'compact':
                                results[f'int_{full_key}_prod_compact'] = p_i
                                results[f'int_{full_key}_diss_compact'] = d_i
                            else:
                                # Per-component P/D curves are always stored because the plots depend on them.
                                results[f'int_{full_key}_prod'] = p_i
                                results[f'int_{full_key}_diss'] = d_i
                                family_itemized_prod += float(p_i)
                                family_itemized_diss += float(d_i)
                        else:
                            if pd_mode == 'compact':
                                results[f'int_{full_key}_prod_compact'] = results[f'int_{full_key}_diss_compact'] = 0.0
                            else:
                                results[f'int_{full_key}_prod'] = results[f'int_{full_key}_diss'] = 0.0

                        if verbose:
                            if pd_mode == 'compact':
                                log_message(f'{log_prefix}Snap {it} in {sims}: compact {vel_key} total P/D split volume integral done', tag="integral", level=1)
                            else:
                                log_message(f'{log_prefix}Snap {it} in {sims}: {comp_key} ({vel_key}) P/D volume integral done', tag="integral", level=1)

                    if itemized_enabled:
                        # Family-wise itemized totals
                        results[f'int_MIE_total_B2{sfx}_prod'] = family_itemized_prod
                        results[f'int_MIE_total_B2{sfx}_diss'] = family_itemized_diss

            if compute_fractional_integrals:
                p_drag = float(results.get('int_MIE_drag_B2_prod', 0.0))
                d_drag = float(results.get('int_MIE_drag_B2_diss', 0.0))

                for vel_key, sfx in active_families:
                    fam_prod = float(results.get(f'int_MIE_total_B2{sfx}_prod', results.get(f'int_MIE_total_B2{sfx}_prod_compact', 0.0)))
                    fam_diss = float(results.get(f'int_MIE_total_B2{sfx}_diss', results.get(f'int_MIE_total_B2{sfx}_diss_compact', 0.0)))

                    results[f'int_PD_iota{sfx}'] = 0.0 if fam_prod <= 0.0 else (fam_prod - fam_diss) / fam_prod

                    if components.get('drag', False):
                        results[f'int_PD_frac_MIE_drag_B2{sfx}_prod'] = 0.0 if fam_prod <= 0.0 else p_drag / fam_prod
                        results[f'int_PD_frac_MIE_drag_B2{sfx}_diss'] = 0.0 if fam_diss <= 0.0 else d_drag / fam_diss

                    for comp_key, pref, pd_mode in velocity_terms:
                        if pd_mode != 'itemized':
                            continue
                        full_key = f"{pref}{sfx}"
                        p_i = float(results.get(f'int_{full_key}_prod', 0.0))
                        d_i = float(results.get(f'int_{full_key}_diss', 0.0))
                        results[f'int_PD_frac_{full_key}_prod'] = 0.0 if fam_prod <= 0.0 else p_i / fam_prod
                        results[f'int_PD_frac_{full_key}_diss'] = 0.0 if fam_diss <= 0.0 else d_i / fam_diss

                        if verbose:
                            log_message(f'{log_prefix}Snap {it} in {sims}: {comp_key} ({vel_key}) fractional P/D volume integral computed', tag="integral", level=1)
    
    if components.get('kinetic_energy', True) and induction_energy.get('kinetic_energy_density'):
        results['int_kinetic_energy'] = _integrate_field(induction_energy['kinetic_energy_density'])
        if verbose:
            log_message(f'{log_prefix}Snap {it} in {sims}: Kinetic energy density volume integral done', tag="integral", level=1)
    else:
        results['int_kinetic_energy'] = 0.0

    if components.get('magnetic_energy', True) and clus_b2:
        results['int_b2'] = _integrate_field(clus_b2)
        if verbose:
            log_message(f'{log_prefix}Snap {it} in {sims}: Magnetic energy density volume integral done', tag="integral", level=1)
        
        results['int_B2'] = results['int_b2']
        
        if pd_enabled and isinstance(production_dissipation, dict):
            if production_dissipation.get('normalized', True):
                try:
                    rho_b_val = float(rho_b)
                    if rho_b_val > 0:
                        results['int_B2'] = results['int_b2'] / rho_b_val
                except (TypeError, ValueError):
                    pass  # Mantiene el valor base de results['int_b2'] si falla
    else:
        results['int_b2'] = 0.0
        results['int_B2'] = 0.0

    results['volume'] = _integrate_field(clus_b2, vol=True)

    if verbose:
        total_time_induction = time.time() - start_time_induction
        time_str = strftime("%H:%M:%S", gmtime(total_time_induction))
        
        if compute_induction_integrals and pd_enabled:
            log_message(f'{log_prefix}Time for induction and P/D integration in snap {grid_irr}: {time_str}', tag="integral", level=1)
        elif compute_induction_integrals:
            log_message(f'{log_prefix}Time for induction integration in snap {grid_irr}: {time_str}', tag="integral", level=1)
        elif pd_enabled:
            log_message(f'{log_prefix}Time for P/D integration in snap {grid_irr}: {time_str}', tag="integral", level=1)

    return results


def induction_energy_integral_evolution(components, induction_energy_integral,
                                        derivative, rho_b,
                                        grid_time, grid_zeta,
                                        normalized=False, verbose=False):
    '''
    Given the volume integrals of the magnetic energy density and its components at different redshifts,
    computes both total (integrated) and differential (rate of change) evolution of the magnetic integrated energy 
    and that of its components for their further representation attending to the time derivative prediction from 
    the induction equation.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - induction_energy_integral: dictionary containing the volume integrals of the magnetic induction equation in terms of the
                                    magnetic energy computed in the previous step for each simulation and iteration:
            - int_MIE_diver_B2: volume integral of the null divergence of the magnetic field energy
            - int_MIE_compres_B2: volume integral of the compressive component
            - int_MIE_stretch_B2: volume integral of the stretching component
            - int_MIE_advec_B2: volume integral of the advection component
            - int_MIE_drag_B2: volume integral of the cosmic drag component
            - int_MIE_total_B2: volume integral of the total magnetic induction energy
            - int_kinetic_energy: volume integral of the kinetic energy density
            - int_b2: volume integral of the magnetic energy density
            - volume: volume of the studied region
        - derivative: type of derivative to compute ('RK' for Runge-Kutta,
                'implicit_forward' for forward-implicit predictor,
            'central' for central differences,
            'alpha_fit' for least-squares calibrated explicit predictor,
            'rate' for rate of change).
        - rho_b: density contrast of the simulation
        - grid_time: time grid for the simulation
        - grid_zeta: redshift grid for the simulation
        - normalized: True to remove rho_b from magnetic evolution outputs, False to keep the current rho_b-scaled evolution
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the evolution of BOTH total and differential magnetic energy density and its components (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - evo_MIE_diver_B2: total evolution of the null divergence of the magnetic field energy
            - evo_MIE_diver_B2_diff: differential evolution of the null divergence of the magnetic field energy
            - evo_MIE_compres_B2: total evolution of the compressive component
            - evo_MIE_compres_B2_diff: differential evolution of the compressive component
            - evo_MIE_stretch_B2: total evolution of the stretching component
            - evo_MIE_stretch_B2_diff: differential evolution of the stretching component
            - evo_MIE_advec_B2: total evolution of the advection component
            - evo_MIE_advec_B2_diff: differential evolution of the advection component
            - evo_MIE_drag_B2: total evolution of the cosmic drag component
            - evo_MIE_drag_B2_diff: differential evolution of the cosmic drag component
            - evo_MIE_total_B2: total evolution of the total magnetic induction energy
            - evo_MIE_total_B2_diff: differential evolution of the total magnetic induction energy
            - evo_kinetic_energy: total evolution of the kinetic energy density
            - evo_kinetic_energy_diff: differential evolution of the kinetic energy density
            - evo_b2: total evolution of the magnetic energy density
            - evo_b2_diff: differential evolution of the magnetic energy density
            - evo_ind_b2: total evolution of the integrated magnetic energy from the induction equation
            - evo_ind_b2_diff: differential evolution of the integrated magnetic energy from the induction equation
            - evo_volume_phi: physical volume evolution
            - evo_volume_co: comoving volume evolution
            
    Author: Marco Molina
    '''
    
    assert derivative in ['RK', 'implicit_forward', 'central', 'alpha_fit', 'rate'], \
        "derivative must be 'RK', 'implicit_forward', 'central', 'alpha_fit', or 'rate'"
    assert isinstance(normalized, bool), "normalized must be a boolean (True or False)"
    
    ## Here we compute the evolution of the magnetic energy density and its components
    
    start_time_evolution = time.time() # Record the start time
    
    n = len(grid_time)-1
    zero = [0] * (n+1)
    
    results = {}

    rho_b_arr = np.asarray(rho_b if rho_b is not None else np.ones(n + 1, dtype=float), dtype=float)
    if rho_b_arr.ndim == 0:
        rho_b_arr = np.full(n + 1, float(rho_b_arr), dtype=float)
    rho_factor = rho_b_arr if not normalized else np.ones_like(rho_b_arr, dtype=float)
    rho_denominator = rho_b_arr if normalized else np.ones_like(rho_b_arr, dtype=float)
    
    # Calculate scale factor (a_{i+1}/a_i)^3 = ((1+z_i)/(1+z_{i+1}))^3
    # This accounts for the expansion factor in comoving volume integrals
    gz = np.asarray(grid_zeta, dtype=float)
    scale_factor = np.ones(n, dtype=float)
    # scale_factor = (a0_masclet / (1 + gz)) ** (1/3) if len(gz) > 1 else np.ones(n, dtype=float)
    # scale_factor = ((1 + gz[1:]) / (1 + gz[:-1])) ** 3 if len(gz) > 1 else np.ones(n, dtype=float)
    # scale_factor = ((1 + gz[1:]) / (1 + gz[:-1])) ** 4 if len(gz) > 1 else np.ones(n, dtype=float)
    # scale_factor = ((1 + gz[1:]) / (1 + gz[:-1])) if len(gz) > 1 else np.ones(n, dtype=float)
    print(f"Evolution scale factor: {scale_factor}")

    if verbose:
        gt = np.asarray(grid_time, dtype=float)
        gz = np.asarray(grid_zeta, dtype=float)
        rb = rho_b_arr
        dt = np.diff(gt) if len(gt) > 1 else np.array([], dtype=float)
        log_message(
            f'Evolution input check: derivative={derivative}, '
            f'n_snap={len(gt)}, n_steps={n}',
            tag="evolution",
            level=1
        )
        if len(gt) > 0:
            log_message(
                f'grid_time endpoints: [{gt[0]:.6g}, {gt[-1]:.6g}] | '
                f'grid_zeta endpoints: [{gz[0]:.6g}, {gz[-1]:.6g}]',
                tag="evolution",
                level=2
            )
        if len(dt) > 0:
            log_message(
                f'dt range: min={np.min(dt):.6g}, max={np.max(dt):.6g}',
                tag="evolution",
                level=2
            )
            if len(dt) > 6:
                dt_head = ", ".join(f"{v:.4g}" for v in dt[:3])
                dt_tail = ", ".join(f"{v:.4g}" for v in dt[-3:])
                log_message(
                    f'dt preview: head=[{dt_head}] tail=[{dt_tail}]',
                    tag="evolution",
                    level=2
                )
        int_b2 = np.asarray(induction_energy_integral.get('int_b2', []), dtype=float)
        if len(int_b2) > 0 and len(rb) > 0:
            log_message(
                f'int_b2 endpoints: [{int_b2[0]:.6g}, {int_b2[-1]:.6g}] | '
                f'rho_b endpoints: [{rb[0]:.6g}, {rb[-1]:.6g}]',
                tag="evolution",
                level=2
            )
    
    # Reconstruct induction totals using exact suffixes from induction_vol_integral: '', '_solenoidal', '_compressive'
    
    base_mechanisms = ["divergence", "compression", "stretching", "advection", "drag"]
    for sfx in ['', '_solenoidal', '_compressive']:
        ind_key = f"induction{sfx}"
        int_ind_key = f"int_ind_b2{sfx}" if sfx else "int_ind_b2"

        # Check if all constituent terms exist in induction_energy_integral
        prefix_map = {
            'divergence': f'MIE_diver_B2{sfx}',
            'compression': f'MIE_compres_B2{sfx}',
            'stretching': f'MIE_stretch_B2{sfx}',
            'advection': f'MIE_advec_B2{sfx}',
            'drag': 'MIE_drag_B2'
        }
        
        all_present = all(f'int_{prefix_map[k]}' in induction_energy_integral for k in base_mechanisms)

        if all_present and components.get("itemized", False):
            components[ind_key] = True
            induction_energy_integral[int_ind_key] = [
                sum(induction_energy_integral[f'int_{prefix_map[k]}'][i] for k in base_mechanisms)
                for i in range(n + 1)
            ]
        else:
            components[ind_key] = False

    # Alpha fit logic using the primary total induction component
    alpha_fit_value = None
    if derivative == 'alpha_fit' and components.get("induction", False):
        int_b2_arr = np.asarray(induction_energy_integral.get('int_b2', []), dtype=float)
        int_ind_arr = np.asarray(induction_energy_integral.get('int_ind_b2', []), dtype=float)
        gt = np.asarray(grid_time, dtype=float)
        rb = rho_b_arr
        sf = scale_factor

        if len(int_b2_arr) == n + 1 and len(int_ind_arr) >= n and len(gt) == n + 1 and len(rb) == n + 1:
            dt = gt[1:] - gt[:-1]
            x = rb[1:] * dt * int_ind_arr[:n]
            y = int_b2_arr[1:] - sf * (rb[1:] / rb[:-1]) * int_b2_arr[:-1]

            valid = np.isfinite(x) & np.isfinite(y)
            valid &= np.abs(x) > 0
            if np.any(valid):
                x_v = x[valid]
                y_v = y[valid]
                denom = np.dot(x_v, x_v)
                if denom > 0:
                    alpha_fit_value = float(np.dot(x_v, y_v) / denom)
                    results['alpha_fit'] = alpha_fit_value
                    if verbose:
                        y_pred = alpha_fit_value * x_v
                        mse = float(np.mean((y_pred - y_v) ** 2)) if len(y_v) > 0 else np.nan
                        log_message(
                            f"Alpha fit (least squares): alpha={alpha_fit_value:.6g}, samples={len(x_v)}, mse={mse:.6g}",
                            tag="evolution",
                            level=1
                        )
                        
    evolution_mappings = [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('drag', 'MIE_drag_B2'),
        ('total', 'MIE_total_B2'),
        ('induction', 'ind_b2'),
        
        # Solenoidal velocity field components
        ('divergence_solenoidal', 'MIE_diver_B2_solenoidal'),
        ('compression_solenoidal', 'MIE_compres_B2_solenoidal'),
        ('stretching_solenoidal', 'MIE_stretch_B2_solenoidal'),
        ('advection_solenoidal', 'MIE_advec_B2_solenoidal'),
        ('total_solenoidal', 'MIE_total_B2_solenoidal'),
        ('induction_solenoidal', 'ind_b2_solenoidal'),
        
        # Compressive velocity field components
        ('divergence_compressive', 'MIE_diver_B2_compressive'),
        ('compression_compressive', 'MIE_compres_B2_compressive'),
        ('stretching_compressive', 'MIE_stretch_B2_compressive'),
        ('advection_compressive', 'MIE_advec_B2_compressive'),
        ('total_compressive', 'MIE_total_B2_compressive'),
        ('induction_compressive', 'ind_b2_compressive'),
    ]

    for key, prefix in evolution_mappings:
        int_key = f'int_{prefix}'
        has_series = int_key in induction_energy_integral

        # Total Evolution
        if has_series:
            if derivative == 'RK':
                results[f'evo_{prefix}'] = diff.integrate_energy(grid_time, induction_energy_integral['int_b2'][0],
                                                            rho_factor, induction_energy_integral[int_key])
            elif derivative == 'central':
                results[f'evo_{prefix}'] = [(rho_factor[i+1] * ((1/rho_b[i]) * scale_factor[i] * (induction_energy_integral['int_b2'][i]) +
                2 * (grid_time[i+1] - grid_time[i]) * scale_factor[i] * (induction_energy_integral[int_key][i]))) for i in range(n)]
            elif derivative == 'implicit_forward':
                results[f'evo_{prefix}'] = [
                    (
                        (rho_factor[i+1] / rho_b[i]) * scale_factor[i] * induction_energy_integral['int_b2'][i]
                        + 2 * rho_b[i+1] * (grid_time[i+1] - grid_time[i]) * scale_factor[i] * (
                            induction_energy_integral[int_key][i]
                            + ((grid_time[i+1] - grid_time[i]) / (grid_time[i+1] - grid_time[i-1]))
                            * (induction_energy_integral[int_key][i+1] - induction_energy_integral[int_key][i-1])
                        )
                    )
                    for i in range(1, n)
                ]
            elif derivative == 'rate':
                results[f'evo_{prefix}'] = [(rho_factor[i+1] * ((1/rho_b[i]) * scale_factor[i] * (induction_energy_integral['int_b2'][i]) +
                (grid_time[i+1] - grid_time[i]) * scale_factor[i] * (induction_energy_integral[int_key][i+1] + induction_energy_integral[int_key][i]))) for i in range(n)]
            elif derivative == 'alpha_fit':
                alpha = alpha_fit_value if alpha_fit_value is not None else 2.0
                results[f'evo_{prefix}'] = [
                    (
                        rho_factor[i+1] * (
                            scale_factor[i] * (induction_energy_integral['int_b2'][i] / rho_b[i])
                            + alpha * (grid_time[i+1] - grid_time[i]) * scale_factor[i] * induction_energy_integral[int_key][i]
                        )
                    )
                    for i in range(n)
                ]
            if verbose:
                log_message(f'Energy evolution: total {key} volume energy integral evolution done', tag="evolution", level=1)
        else:
            results[f'evo_{prefix}'] = zero
        
        # Differential Evolution
        if has_series:
            if derivative == 'RK' or derivative == 'central':
                results[f'evo_{prefix}_diff'] = [2 * scale_factor[i] * rho_factor[i] * induction_energy_integral[int_key][i] for i in range(n)]
            elif derivative == 'implicit_forward':
                results[f'evo_{prefix}_diff'] = [
                    2 * scale_factor[i] * (
                        rho_factor[i] * induction_energy_integral[int_key][i]
                        + ((grid_time[i+1] - grid_time[i]) / (grid_time[i+1] - grid_time[i-1]))
                        * (rho_factor[i+1] * induction_energy_integral[int_key][i+1] - rho_factor[i-1] * induction_energy_integral[int_key][i-1])
                    )
                    for i in range(1, n)
                ]
            elif derivative == 'rate':
                results[f'evo_{prefix}_diff'] = [scale_factor[i] * (rho_factor[i+1] * induction_energy_integral[int_key][i+1] + rho_factor[i] * induction_energy_integral[int_key][i]) for i in range(n)]
            elif derivative == 'alpha_fit':
                alpha = alpha_fit_value if alpha_fit_value is not None else 2.0
                results[f'evo_{prefix}_diff'] = [alpha * scale_factor[i] * rho_factor[i] * induction_energy_integral[int_key][i] for i in range(n)]
            if verbose:
                log_message(f'Energy evolution: differential {key} energy integral evolution done', tag="evolution", level=1)
        else:
            results[f'evo_{prefix}_diff'] = [0.0 for _ in range(n)]
        
    # Kinetic Energy Evolution
    if components.get('kinetic_energy', True) and 'int_kinetic_energy' in induction_energy_integral:
        # Total: scaled by rho_factor
        results['evo_kinetic_energy'] = [rho_factor[i] * induction_energy_integral['int_kinetic_energy'][i] for i in range(n+1)]
        # Differential: time derivative with rho_factor scaling
        results['evo_kinetic_energy_diff'] = [
            (1/(grid_time[i+1] - grid_time[i])) * (
                rho_factor[i+1] * induction_energy_integral['int_kinetic_energy'][i+1] -
                rho_factor[i] * induction_energy_integral['int_kinetic_energy'][i])
            for i in range(n)
        ]
    else:
        results['evo_kinetic_energy'] = [0.0 for _ in range(n+1)]
        results['evo_kinetic_energy_diff'] = [0.0 for _ in range(n)]
    
    # Magnetic Energy Evolution
    if components.get('magnetic_energy', True) and 'int_b2' in induction_energy_integral:
            # Total: direct value normalized by rho_denominator
            results['evo_b2'] = [induction_energy_integral['int_b2'][i] / rho_denominator[i] for i in range(n+1)]
            # Differential: time derivative
            results['evo_b2_diff'] = [
            1/((grid_time[i+1] - grid_time[i])) * (
                (induction_energy_integral['int_b2'][i+1] / rho_denominator[i+1]) -
                (induction_energy_integral['int_b2'][i] / rho_denominator[i]))
            for i in range(n)
        ]
    else:
        results['evo_b2'] = [0.0 for _ in range(n+1)]
        results['evo_b2_diff'] = [0.0 for _ in range(n)]
    
    if verbose == True:
        log_message('Energy evolution: magnetic and kinetic energy integral evolution done', tag="evolution", level=1)

    if verbose and derivative == 'central' and 'evo_ind_b2' in results:
        pred = np.asarray(results['evo_ind_b2'], dtype=float)
        meas = np.asarray(results.get('evo_b2', []), dtype=float)
        src = np.asarray(induction_energy_integral.get('int_ind_b2', []), dtype=float)
        gt = np.asarray(grid_time, dtype=float)
        gz = np.asarray(grid_zeta, dtype=float)
        rb = np.asarray(rho_b, dtype=float)

        if len(pred) == n and len(meas) == n + 1 and len(src) >= n:
            sample_idx = sorted(set([0, min(1, n - 1), max(n - 1, 0)]))

            if n > 2:
                idx_meas = int(np.argmax(meas[1:])) + 1
                idx_pred = int(np.argmax(pred))
                sample_idx.extend([max(0, idx_meas - 1), max(0, idx_meas - 2), idx_pred])
                sample_idx = sorted(set(i for i in sample_idx if 0 <= i < n))

            for i in sample_idx:
                ratio = pred[i] / meas[i + 1] if meas[i + 1] != 0 else np.nan

                log_message(
                    "Central indexing check: "
                    f"i={i} -> pred_idx={i} maps to measured_idx={i+1}; "
                    f"t_i={gt[i]:.6g}, t_i1={gt[i+1]:.6g}, "
                    f"z_i={gz[i]:.6g}, z_i1={gz[i+1]:.6g}, "
                    f"int_ind_i={src[i]:.6g}, rho_i={rb[i]:.6g}, rho_i1={rb[i+1]:.6g}, "
                    f"pred={pred[i]:.6g}, meas_next={meas[i+1]:.6g}, pred/meas_next={ratio:.6g}",
                    tag="evolution",
                    level=2
                )

            ratio_all = np.divide(pred, meas[1:], out=np.full_like(pred, np.nan), where=meas[1:] != 0)
            finite_ratio = ratio_all[np.isfinite(ratio_all)]
            if len(finite_ratio) > 0:
                log_message(
                    f'Central ratio summary: min={np.min(finite_ratio):.6g}, '
                    f'median={np.median(finite_ratio):.6g}, max={np.max(finite_ratio):.6g}',
                    tag="evolution",
                    level=2
                )
            
    results['evo_volume_phi'] = [(induction_energy_integral['volume'][i]) for i in range(n+1)]
    results['evo_volume_co'] = [(induction_energy_integral['volume'][i] / ((1/(1+grid_zeta[i]))**3)) for i in range(n+1)]
    
    if verbose:
        log_message('Energy evolution: volume evolution done', tag="evolution", level=1)
    
    end_time_evolution = time.time()
    total_time_evolution = end_time_evolution - start_time_evolution
    
    if verbose:
        log_message('Time for evolution of the induction energy integral: '+str(strftime("%H:%M:%S", gmtime(total_time_evolution))), tag="evolution", level=1)

    return results

    
def induction_radial_profiles(components, velocity_field, induction_energy, clus_b2, clus_rho_rho_b, 
                            rho_b, clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_npatch, up_to_level,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, rmin, rad, 
                            nbins=25, logbins=True, units=1, debug=False, verbose=False):
    '''
    Computes the radial profiles of the magnetic energy density and its components, as well as the induced magnetic energy profile.
    This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied profile. The kinetic energy
    density profile is also computed.
    Only computes the components that are set to True in "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - velocity_field: list of velocity field components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["velocity_field"])
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy computed in the previous step
            - MIE_diver_B2: null divergence of the magnetic field energy
            - MIE_compres_B2: compressive component of the magnetic field induction energy
            - MIE_stretch_B2: stretching component of the magnetic field induction energy
            - MIE_advec_B2: advection component of the magnetic field induction energy
            - MIE_drag_B2: cosmic drag component of the magnetic field induction energy
            - MIE_total_B2: total magnetic induction energy in the compact way
            - kinetic_energy_density: kinetic energy density of the cluster
        - clus_b2: magnetic energy density in the cluster
        - clus_rho_rho_b: density contrast of the cluster
        - rho_b: density contrast of the simulation
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - clus_kp: mask for valid patches
        - grid_irr: index of the snapshot
        - grid_npatch: number of patches in the grid
        - up_tolevel: maximum refinement level to be considered
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - it: index of the snapshot
        - sims: name of the simulation
        - nmax: maximum number of patches
        - size: size of the grid
        - coords: coordinates of the region
        - rmin: minimum radius for the radial profile
        - rad: radius of the region
        - nbins: number of bins for the radial profile (default is 50)
        - logbins: boolean to use logarithmic bins (default is False)
        - units: factor to convert the units multiplied by the final result (default is 1)
        - debug: boolean to print the inner progress of the profile computation (default is False)
        - verbose: boolean to print the progress of the computation (default is False)
    
    Returns:
        - results: dictionary containing the computed radial profiles (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - MIE_diver_B2_profile: radial profile of the null divergence of the magnetic field energy
            - MIE_compres_B2_profile: radial profile of the compressive component
            - MIE_stretch_B2_profile: radial profile of the stretching component
            - MIE_advec_B2_profile: radial profile of the advection component
            - MIE_drag_B2_profile: radial profile of the cosmic drag component
            - MIE_total_B2_profile: radial profile of the total magnetic induction energy
            - kinetic_energy_profile: radial profile of the kinetic energy density
            - clus_B2_profile: radial profile of the magnetic energy density
            - clus_rho_rho_b_profile: radial profile of the density contrast
            - profile_bin_centers: centers of the bins for the radial profile
        
    Author: Marco Molina
    '''
    
    ## We can calculate the radial profiles of the magnetic energy density in the volume we have considered (usually the virial volume)
    
    start_time_profile = time.time() # Record the start time

    X, Y, Z = utils.compute_position_fields(grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz, grid_npatch, 
                                            size, nmax, ncores=1, kept_patches=clus_kp)
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    results = {}
    
    velocity_field = velocity_field or {"total": True}
    velocity_mappings = [
        ("total", ""),
        ("solenoidal", "_solenoidal"),
        ("compressive", "_compressive")
    ]
    velocity_terms = [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('total', 'MIE_total_B2')
    ]
    
    ### We first compute the drag component profile, as it is the only one that does not depend on the velocity field decomposition.
    if components.get('drag', False):
        profile_bin_centers, profile = utils.radial_profile_vw(
            field=induction_energy['MIE_drag_B2'], cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['MIE_drag_B2_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: drag profile done', tag="profiles", level=1)
    else:
        results['MIE_drag_B2_profile'] = 0.0
    
    ### Now we compute the induction energy profiles, which require all main components to be present.
    has_all_main_components = all(components.get(k, False) for k in ["divergence", "compression", "stretching", "advection", "drag"])
    
    #### Direct Induction (Sum of All Components)
    if has_all_main_components and velocity_field.get("total", False):
        induction_energy['ind_b2'] = [
            induction_energy['MIE_compres_B2'][p] + 
            induction_energy['MIE_diver_B2'][p] + 
            induction_energy['MIE_stretch_B2'][p] + 
            induction_energy['MIE_advec_B2'][p] + 
            induction_energy['MIE_drag_B2'][p] for p in range(n)
        ]
        
        profile_bin_centers, profile = utils.radial_profile_vw(
            field=induction_energy['ind_b2'], cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['ind_b2_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: direct total induction (ind_b2) profile done', tag="profiles", level=1)
    else:
        results['ind_b2_profile'] = 0.0

    #### Reconstructed Induction (Sum of Solenoidal + Compressive)
    # Only calculated if both velocity components are present to enable coupling
    if has_all_main_components and velocity_field.get("solenoidal", False) and velocity_field.get("compressive", False):
        induction_energy['ind_b2_vortex'] = [
            (induction_energy['MIE_compres_B2_solenoidal'][p] + induction_energy['MIE_compres_B2_compressive'][p]) +
            (induction_energy['MIE_diver_B2_solenoidal'][p]   + induction_energy['MIE_diver_B2_compressive'][p]) +
            (induction_energy['MIE_stretch_B2_solenoidal'][p] + induction_energy['MIE_stretch_B2_compressive'][p]) +
            (induction_energy['MIE_advec_B2_solenoidal'][p]   + induction_energy['MIE_advec_B2_compressive'][p]) +
            induction_energy['MIE_drag_B2'][p] for p in range(n)
        ]
        
        _, profile_rec = utils.radial_profile_vw(
            field=induction_energy['ind_b2_vortex'], cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['ind_b2_vortex_profile'] = rho_b * profile_rec
        if verbose:
            log_message(f'Snap {it} in {sims}: reconstructed vortex induction (solenoidal + compressive) profile done', tag="profiles", level=1)
    else:
        results['ind_b2_vortex_profile'] = 0.0
    
    ### Now we compute the profiles for each component of the induction energy equation, considering the velocity field decomposition if applicable.
    for vel_key, sfx in velocity_mappings:
        if velocity_field.get(vel_key, False):
            for comp_key, pref in velocity_terms:
                full_key = f"{pref}{sfx}"
                
                if components.get(comp_key, False):
                    profile_bin_centers, profile = utils.radial_profile_vw(
                        field=induction_energy[full_key], cr0amr=clus_cr0amr, solapst=clus_solapst, 
                        npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
                        rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                        size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
                    )
                    results[f'{full_key}_profile'] = rho_b * profile
                    if verbose:
                        log_message(f'Snap {it} in {sims}: {comp_key} ({vel_key}) profile done', tag="profiles", level=1)
                else:
                    results[f'{full_key}_profile'] = 0.0
        else:
            for comp_key, pref in velocity_terms:
                results[f'{pref}{sfx}_profile'] = 0.0
    
    if components.get('kinetic_energy', True) and induction_energy.get('kinetic_energy_density'):
        profile_bin_centers, profile = utils.radial_profile_vw(
            field=induction_energy['kinetic_energy_density'], cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['kinetic_energy_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: Kinetic profile done', tag="profiles", level=1)
    else:
        results['kinetic_energy_profile'] = 0.0
        
    if components.get('magnetic_energy', True) and clus_b2:
        profile_bin_centers, profile = utils.radial_profile_vw(
            field=clus_b2, cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['clus_b2_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: b2 profile done', tag="profiles", level=1)
    else:
        results['clus_b2_profile'] = 0.0
    
    if clus_rho_rho_b:
        profile_bin_centers, profile = utils.radial_profile_vw(
            field=clus_rho_rho_b, cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
        )
        results['clus_rho_rho_b_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: Density profile done', tag="profiles", level=1)
    else:
        results['clus_rho_rho_b_profile'] = 0.0
        # Extracción exclusiva de centros de bins geométricos si el campo de densidad no está disponible
        profile_bin_centers, _ = utils.radial_profile_vw(
            field=clus_b2, cr0amr=clus_cr0amr, solapst=clus_solapst, 
            npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
            rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=False
        )
    
    results['profile_bin_centers'] = profile_bin_centers
    
    if verbose:
        total_time_profile = time.time() - start_time_profile
        log_message(f'Time for profile calculation in snap {grid_irr}: {strftime("%H:%M:%S", gmtime(total_time_profile))}', tag="profiles", level=1)
        
    return results


def production_dissipation_radial_profiles(components, velocity_field, induction_energy,
                                        rho_b, clus_cr0amr, clus_solapst, clus_kp,
                                        grid_irr, grid_npatch, up_to_level,
                                        grid_patchrx, grid_patchry, grid_patchrz,
                                        grid_patchnx, grid_patchny, grid_patchnz,
                                        it, sims, nmax, size, coords, rmin, rad,
                                        nbins=25, logbins=True, units=1,
                                        compute_fractional_profiles=False,
                                        debug=False, verbose=False):
    '''
    Computes radial profiles for production/dissipation terms from induction-energy fields.

    Returns profiles for production, dissipation and net (production - dissipation)
    for each enabled component and total terms.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - velocity_field: list of velocity field components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["velocity_field"])
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy computed in the previous step
            - MIE_diver_B2_prod: production term for the null divergence of the magnetic field energy
            - MIE_diver_B2_diss: dissipation term for the null divergence of the magnetic field energy
            - MIE_compres_B2_prod: production term for the compressive component of the magnetic field induction energy
            - MIE_compres_B2_diss: dissipation term for the compressive component of the magnetic field induction energy
            - MIE_stretch_B2_prod: production term for the stretching component of the magnetic field induction energy
            - MIE_stretch_B2_diss: dissipation term for the stretching component of the magnetic field induction energy
            - MIE_advec_B2_prod: production term for the advection component of the magnetic field induction energy
            - MIE_advec_B2_diss: dissipation term for the advection component of the magnetic field induction energy
            - MIE_drag_B2_prod: production term for the cosmic drag component of the magnetic field induction energy
            - MIE_drag_B2_diss: dissipation term for the cosmic drag component of the magnetic field induction energy
            - MIE_total_B2_prod: total production term for the magnetic induction energy
            - MIE_total_B2_diss: total dissipation term for the magnetic induction energy
        - rho_b: density contrast of the simulation
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - clus_kp: mask for valid patches
        - grid_irr: index of the snapshot
        - grid_npatch: number of patches in the grid
        - up_tolevel: maximum refinement level to be considered
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - it: index of the snapshot
        - sims: name of the simulation
        - nmax: maximum number of patches
        - size: size of the grid
        - coords: coordinates of the region
        - rmin: minimum radius for the radial profile
        - rad: radius of the region
        - nbins: number of bins for the radial profile (default is 50)
        - logbins: boolean to use logarithmic bins (default is False)
        - units: factor to convert the units multiplied by the final result (default is 1)
        - compute_fractional_profiles: boolean to compute fractional profiles (default is False)
        - debug: boolean to print the inner progress of the profile computation (default is False)
        - verbose: boolean to print the progress of the computation (default is False)
    
    Returns:
        - results: dictionary containing the computed radial profiles for production, dissipation, and net terms, toguether with fractional profiles for the enabled velocity fields subdivisions (sufix indicates the velocity field type, e.g. _solenoidal, _compressive, no sufix for total velocity field):
            - MIE_diver_B2_prod_profile: radial profile of the production term for the null divergence of the magnetic field energy
            - MIE_diver_B2_diss_profile: radial profile of the dissipation term for the null divergence of the magnetic field energy
            - MIE_diver_B2_net_profile: radial profile of the net term (production - dissipation) for the null divergence of the magnetic field energy
            - MIE_compres_B2_prod_profile: radial profile of the production term for the compressive component of the magnetic field induction energy
            - MIE_compres_B2_diss_profile: radial profile of the dissipation term for the compressive component of the magnetic field induction energy
            - MIE_compres_B2_net_profile: radial profile of the net term (production - dissipation) for the compressive component of the magnetic field induction energy
            - MIE_stretch_B2_prod_profile: radial profile of the production term for the stretching component of the magnetic field induction energy
            - MIE_stretch_B2_diss_profile: radial profile of the dissipation term for the stretching component of the magnetic field induction energy
            - MIE_advec_B2_prod_profile: radial profile of the production term for the advection component of the magnetic field induction energy
            - MIE_advec_B2_diss_profile: radial profile of the dissipation term for the advection component of the magnetic field induction energy
            - MIE_drag_B2_prod_profile: radial profile of the production term for the cosmic drag component of the magnetic induction energy
            - MIE_drag_B2_diss_profile: radial profile of the dissipation term for the cosmic drag component of the magnetic induction energy
            - MIE_total_B2_prod_profile: radial profile of the total production term for the magnetic induction energy
            - MIE_total_B2_diss_profile: radial profile of the total dissipation term for the magnetic induction energy
            
    Author: Marco Molina
    '''

    start_time_profile = time.time()

    X, Y, Z = utils.compute_position_fields(
        grid_patchnx, grid_patchny, grid_patchnz,
        grid_patchrx, grid_patchry, grid_patchrz,
        grid_npatch, size, nmax, ncores=1, kept_patches=clus_kp
    )

    n = 1 + np.sum(grid_npatch)
    zero = 0.0
    results = {}

    velocity_field = velocity_field or {"total": True}
    velocity_mappings = [
        ("total", ""),
        ("solenoidal", "_solenoidal"),
        ("compressive", "_compressive")
    ]
    velocity_terms = [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('total', 'MIE_total_B2')
    ]

    ### Keep running totals for itemized curves (exclude compact total by construction).
    itemized_prod_acc = np.zeros(nbins, dtype=float)
    itemized_diss_acc = np.zeros(nbins, dtype=float)
    itemized_rec_prod_acc = np.zeros(nbins, dtype=float)
    itemized_rec_diss_acc = np.zeros(nbins, dtype=float)
    
    ### We start by computing the drag component, as it is the only one that does not depend on the velocity field decomposition.
    if components.get('drag', False):
        p_key, d_key = 'MIE_drag_B2_prod', 'MIE_drag_B2_diss'
        if p_key in induction_energy and d_key in induction_energy:
            _, prod_profile = utils.radial_profile_vw(
                field=induction_energy[p_key], cr0amr=clus_cr0amr, solapst=clus_solapst, 
                npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
                rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
            )
            _, diss_profile = utils.radial_profile_vw(
                field=induction_energy[d_key], cr0amr=clus_cr0amr, solapst=clus_solapst, 
                npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
                rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
            )
            prod_p = rho_b * np.asarray(prod_profile)
            diss_p = rho_b * np.asarray(diss_profile)
            
            results['MIE_drag_B2_prod_profile'] = prod_p
            results['MIE_drag_B2_diss_profile'] = diss_p
            results['MIE_drag_B2_net_profile']  = prod_p - diss_p
            
            # Sumamos al acumulador itemizado del sistema
            itemized_prod_acc += prod_p
            itemized_diss_acc += diss_p
            itemized_rec_prod_acc += prod_p
            itemized_rec_diss_acc += diss_p

            if verbose:
                log_message(f'Snap {it} in {sims}: drag production/dissipation profiles done', tag="profiles", level=1)
        else:
            results['MIE_drag_B2_prod_profile'] = results['MIE_drag_B2_diss_profile'] = results['MIE_drag_B2_net_profile'] = zero
    else:
        results['MIE_drag_B2_prod_profile'] = results['MIE_drag_B2_diss_profile'] = results['MIE_drag_B2_net_profile'] = zero

    for vel_key, sfx in velocity_mappings:
        if velocity_field.get(vel_key, False):
            for comp_key, prefix in velocity_terms:
                enabled = bool(components.get(comp_key, False))
                full_prefix = f"{prefix}{sfx}"
                prod_key = f"{full_prefix}_prod"
                diss_key = f"{full_prefix}_diss"

                if (not enabled) or (prod_key not in induction_energy) or (diss_key not in induction_energy):
                    results[f'{full_prefix}_prod_profile'] = zero
                    results[f'{full_prefix}_diss_profile'] = zero
                    results[f'{full_prefix}_net_profile'] = zero
                    continue

                _, prod_profile = utils.radial_profile_vw(
                    field=induction_energy[prod_key], cr0amr=clus_cr0amr, solapst=clus_solapst, 
                    npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
                    rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                    size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
                )
                _, diss_profile = utils.radial_profile_vw(
                    field=induction_energy[diss_key], cr0amr=clus_cr0amr, solapst=clus_solapst, 
                    npatch=grid_npatch, up_to_level=up_to_level, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], 
                    rmin=rmin, rmax=rad, nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                    size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug
                )

                prod_p = rho_b * np.asarray(prod_profile)
                diss_p = rho_b * np.asarray(diss_profile)

                results[f'{full_prefix}_prod_profile'] = prod_p
                results[f'{full_prefix}_diss_profile'] = diss_p
                results[f'{full_prefix}_net_profile']  = prod_p - diss_p

                # We only accumulate itemized terms if they belong to the total velocity or if the total velocity is not used, to the linear sum of solenoidal + compressive.
                if comp_key != 'total':
                    if vel_key == "total":
                        itemized_prod_acc += prod_p
                        itemized_diss_acc += diss_p
                    elif vel_key in ["solenoidal", "compressive"]:
                        itemized_rec_prod_acc += prod_p
                        itemized_rec_diss_acc += diss_p

                if verbose:
                    log_message(f'Snap {it} in {sims}: {comp_key} ({vel_key}) production/dissipation profiles done', tag="profiles", level=1)
        else:
            for comp_key, prefix in velocity_terms:
                full_prefix = f"{prefix}{sfx}"
                results[f'{full_prefix}_prod_profile'] = results[f'{full_prefix}_diss_profile'] = results[f'{full_prefix}_net_profile'] = zero

    ### Now we handle the itemized total velocity record (MIE_total_B2 direct or reconstructed)
    if velocity_field.get("total", False):
        results['MIE_total_B2_prod_itemized_profile'] = itemized_prod_acc
        results['MIE_total_B2_diss_itemized_profile'] = itemized_diss_acc
        results['MIE_total_B2_net_itemized_profile']  = itemized_prod_acc - itemized_diss_acc
    else:
        results['MIE_total_B2_prod_itemized_profile'] = results['MIE_total_B2_diss_itemized_profile'] = results['MIE_total_B2_net_itemized_profile'] = zero

    ### If both solenoidal and compressive components are present, we can reconstruct the total profile by summing them.
    if velocity_field.get("solenoidal", False) and velocity_field.get("compressive", False):
        results['MIE_total_B2_prod_itemized_reconstructed_profile'] = itemized_rec_prod_acc
        results['MIE_total_B2_diss_itemized_reconstructed_profile'] = itemized_rec_diss_acc
        results['MIE_total_B2_net_itemized_reconstructed_profile']  = itemized_rec_prod_acc - itemized_rec_diss_acc
    else:
        results['MIE_total_B2_prod_itemized_reconstructed_profile'] = results['MIE_total_B2_diss_itemized_reconstructed_profile'] = results['MIE_total_B2_net_itemized_reconstructed_profile'] = zero
    
    ### Now we handle the compact total velocity record (MIE_total_B2 direct or reconstructed)
    if velocity_field.get("total", False) and isinstance(results.get('MIE_total_B2_prod_profile', 0.0), np.ndarray):
        results['MIE_total_B2_prod_compact_profile'] = results['MIE_total_B2_prod_profile']
        results['MIE_total_B2_diss_compact_profile'] = results['MIE_total_B2_diss_profile']
        results['MIE_total_B2_net_compact_profile']  = results['MIE_total_B2_net_profile']
    else:
        results['MIE_total_B2_prod_compact_profile'] = results['MIE_total_B2_diss_compact_profile'] = results['MIE_total_B2_net_compact_profile'] = zero

    ### If both solenoidal and compressive components are present, we can reconstruct the total profile by summing them.
    if velocity_field.get("solenoidal", False) and velocity_field.get("compressive", False):
        p_rec = results['MIE_total_B2_solenoidal_prod_profile'] + results['MIE_total_B2_compressive_prod_profile']
        d_rec = results['MIE_total_B2_solenoidal_diss_profile'] + results['MIE_total_B2_compressive_diss_profile']
        results['MIE_total_B2_prod_compact_reconstructed_profile'] = p_rec
        results['MIE_total_B2_diss_compact_reconstructed_profile'] = d_rec
        results['MIE_total_B2_net_compact_reconstructed_profile']  = p_rec - d_rec
    else:
        results['MIE_total_B2_prod_compact_reconstructed_profile'] = results['MIE_total_B2_diss_compact_reconstructed_profile'] = results['MIE_total_B2_net_compact_reconstructed_profile'] = zero

    if compute_fractional_profiles:
        use_rec = not velocity_field.get("total", False)
        total_prod = np.maximum(itemized_rec_prod_acc if use_rec else itemized_prod_acc, 0.0)
        total_diss = np.maximum(itemized_rec_diss_acc if use_rec else itemized_diss_acc, 0.0)
        
        if isinstance(results.get('MIE_drag_B2_prod_profile', 0.0), np.ndarray):
            results['PD_frac_MIE_drag_B2_prod_profile'] = np.divide(results['MIE_drag_B2_prod_profile'], total_prod, out=np.zeros(nbins), where=total_prod > 0)
            results['PD_frac_MIE_drag_B2_diss_profile'] = np.divide(results['MIE_drag_B2_diss_profile'], total_diss, out=np.zeros(nbins), where=total_diss > 0)
        else:
            results['PD_frac_MIE_drag_B2_prod_profile'] = results['PD_frac_MIE_drag_B2_diss_profile'] = np.zeros(nbins, dtype=float)

        for vel_key, sfx in velocity_mappings:
            if velocity_field.get(vel_key, False):
                for comp_key, prefix in velocity_terms:
                    p_key = f'{prefix}{sfx}_prod_profile'
                    d_key = f'{prefix}{sfx}_diss_profile'
                    
                    if isinstance(results.get(p_key, 0.0), np.ndarray):
                        p_i = results[p_key]
                        d_i = results[d_key]
                        results[f'PD_frac_{prefix}{sfx}_prod_profile'] = np.divide(p_i, total_prod, out=np.zeros_like(p_i), where=total_prod > 0)
                        results[f'PD_frac_{prefix}{sfx}_diss_profile'] = np.divide(d_i, total_diss, out=np.zeros_like(d_i), where=total_diss > 0)
                    else:
                        results[f'PD_frac_{prefix}{sfx}_prod_profile'] = np.zeros(nbins, dtype=float)
                        results[f'PD_frac_{prefix}{sfx}_diss_profile'] = np.zeros(nbins, dtype=float)
            else:
                for comp_key, prefix in velocity_terms:
                    results[f'PD_frac_{prefix}{sfx}_prod_profile'] = results[f'PD_frac_{prefix}{sfx}_diss_profile'] = np.zeros(nbins, dtype=float)

    ref_field = induction_energy.get('MIE_total_B2', [0 for _ in range(n)])
    profile_bin_centers, _ = utils.radial_profile_vw(
        field=ref_field, cr0amr=clus_cr0amr, solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
        clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
        nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
        size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=False
    )
    results['profile_bin_centers'] = profile_bin_centers

    end_time_profile = time.time()
    if verbose:
        total_time_profile = end_time_profile - start_time_profile
        log_message(
            f'Time for production/dissipation profile calculation in snap {grid_irr}: {strftime("%H:%M:%S", gmtime(total_time_profile))}',
            tag="profiles", level=1
        )

    return results


def compute_percentile_thresholds(field_numerator, field_denominator, scale_factor,
                                cr0amr, solapst, npatch, up_to_level,
                                percentiles=(100, 90, 75, 50, 25),
                                use_abs=True, denom_eps=0.0, kept_patches=None,
                                exclude_boundaries=False, boundary_width=1,
                                exclude_zeros=True, verbose=False):
    '''
    Compute percentile thresholds of a ratio field in a single snapshot with safeguards and band edges.
    Applies clean_field to ensure only cells at maximum available resolution are considered.

    Args:
        - field_numerator: numerator field for the ratio (list of 3D arrays, one per patch)
        - field_denominator: denominator field for the ratio (list of 3D arrays, one per patch)
        - scale_factor: factor to multiply the ratio (unit conversion or scaling).
                    Can be a scalar or an array with one value per patch.
                    If array, must have length equal to number of patches.
        - cr0amr: refinement field (1: not refined; 0: refined)
        - solapst: overlap field (1: keep; 0: discard)
        - npatch: number of patches per level
        - up_to_level: maximum refinement level to consider
        - percentiles: tuple/list of percentiles to compute
        - use_abs: take absolute value of the ratio before percentiles
        - denom_eps: minimum absolute value allowed in denominator; smaller values are masked
        - kept_patches: 1d boolean array indicating which patches are inside the region (None to keep all)
        - exclude_boundaries: if True, exclude boundary cells from percentile calculation
        - boundary_width: number of boundary cells to exclude from each side (default 1)
        - exclude_zeros: if True, exclude zero values from percentile calculation
        - verbose: whether to print timing information

    Returns:
        - dict with keys:
            'percentiles': ndarray with the percentile thresholds (same order as input)
            'levels': ndarray of the requested percentiles
            'percentiles_plus': ndarray with percentile+1% thresholds (for error band upper limit)
            'percentiles_minus': ndarray with percentile-1% thresholds (for error band lower limit)
            'global_min': minimum finite value of the ratio (after scaling)
            'global_max': maximum finite value of the ratio (after scaling)
            'bands': list of (low, high) tuples for each percentile band using sorted levels
                (e.g., for shading between successive percentiles; includes the 0–min band)
            Returns None values if no finite data.

    Author: Marco Molina
    '''
    start_time_percentiles = time.time()

    # Apply clean_field to ensure only cells at maximum resolution are considered
    clean_numerator = utils.clean_field(field_numerator, cr0amr, solapst, npatch, up_to_level)
    clean_denominator = utils.clean_field(field_denominator, cr0amr, solapst, npatch, up_to_level)

    # Build a validity mask from AMR refinement/overlap flags (1 = valid, 0 = invalid)
    mask_template = []
    for patch in clean_numerator:
        if patch is None:
            mask_template.append(None)
        elif np.isscalar(patch):
            mask_template.append(np.array(patch, dtype=float))
        else:
            mask_template.append(np.ones_like(patch, dtype=float))
    valid_mask = utils.clean_field(mask_template, cr0amr, solapst, npatch, up_to_level)

    # Always print boundary exclusion status for diagnostics
    if exclude_boundaries and boundary_width > 0:
        log_message(f"Excluding {boundary_width} boundary cells from each patch side", tag="percentiles", level=1)
    elif verbose:
        log_message("Including all cells (boundaries NOT excluded)", tag="percentiles", level=1)

    # Handle scale_factor: scalar or array with one value per patch
    if np.isscalar(scale_factor):
        scale_arr = np.full(len(clean_numerator), scale_factor, dtype=float)
    else:
        scale_arr = np.asarray(scale_factor, dtype=float)
        if not isinstance(field_numerator, (list, tuple)):
            raise ValueError("field_numerator must be a list/tuple of arrays (one per patch)")
        if scale_arr.size != len(clean_numerator):
            raise ValueError(
                f"scale_factor array size ({scale_arr.size}) must match number of patches ({len(clean_numerator)})"
            )

    vals_list = []
    for i, (num_patch, denom_patch) in enumerate(zip(clean_numerator, clean_denominator)):
        if kept_patches is not None and not kept_patches[i]:
            continue

        if num_patch is None or denom_patch is None:
            continue

        if np.isscalar(num_patch) or np.isscalar(denom_patch):
            ratio_patch = np.divide(
                num_patch,
                denom_patch,
                out=np.array(np.nan, dtype=float),
                where=np.abs(denom_patch) > denom_eps,
            )
            if use_abs:
                ratio_patch = np.abs(ratio_patch)
            ratio_patch = ratio_patch * scale_arr[i]
            patch_vals = np.atleast_1d(ratio_patch)
            patch_mask = np.atleast_1d(valid_mask[i]).astype(bool)
        else:
            ratio_patch = np.divide(
                num_patch,
                denom_patch,
                out=np.full_like(num_patch, np.nan, dtype=float),
                where=np.abs(denom_patch) > denom_eps,
            )
            if use_abs:
                ratio_patch = np.abs(ratio_patch)
            ratio_patch = ratio_patch * scale_arr[i]

            patch_mask = valid_mask[i].astype(bool)

            if exclude_boundaries and boundary_width > 0:
                nx, ny, nz = ratio_patch.shape
                if nx > 2 * boundary_width and ny > 2 * boundary_width and nz > 2 * boundary_width:
                    interior_mask = np.ones((nx, ny, nz), dtype=bool)
                    interior_mask[:boundary_width, :, :] = False
                    interior_mask[-boundary_width:, :, :] = False
                    interior_mask[:, :boundary_width, :] = False
                    interior_mask[:, -boundary_width:, :] = False
                    interior_mask[:, :, :boundary_width] = False
                    interior_mask[:, :, -boundary_width:] = False
                    patch_mask = patch_mask & interior_mask

            patch_vals = ratio_patch[patch_mask]

        if patch_vals.size == 0:
            continue

        patch_vals = patch_vals[np.isfinite(patch_vals)]
        if exclude_zeros:
            patch_vals = patch_vals[patch_vals != 0.0]

        if patch_vals.size > 0:
            vals_list.append(np.asarray(patch_vals).ravel())

    if vals_list:
        vals = np.concatenate(vals_list)
    else:
        vals = np.array([])

    # Diagnostic output for data statistics
    total_patches = len(clean_numerator)
    patches_processed = len(vals_list)
    if verbose or (exclude_boundaries and boundary_width > 0):
        log_message(f"Patches processed: {patches_processed}/{total_patches}", tag="percentiles", level=2)
        log_message(f"Total valid values: {vals.size:,}", tag="percentiles", level=2)
        if vals.size > 0:
            log_message(f"Value range: [{np.min(vals):.3e}, {np.max(vals):.3e}]", tag="percentiles", level=2)

    if vals.size == 0:
        return {
            "percentiles": None,
            "levels": np.asarray(percentiles),
            "percentiles_plus": None,
            "percentiles_minus": None,
            "global_min": None,
            "global_max": None,
            "bands": None,
        }

    levels_arr = np.asarray(percentiles, dtype=float)
    thresholds = np.percentile(vals, levels_arr)
    
    # Compute ±1% error bands for each percentile
    percentiles_plus = np.percentile(vals, np.clip(levels_arr + 1, 0, 100))
    percentiles_minus = np.percentile(vals, np.clip(levels_arr - 1, 0, 100))

    gmin = float(np.min(vals))
    gmax = float(np.max(vals))

    sorted_idx = np.argsort(levels_arr)
    sorted_levels = levels_arr[sorted_idx]
    sorted_thresh = thresholds[sorted_idx]

    bands = []
    prev_edge = gmin
    for th in sorted_thresh:
        bands.append((prev_edge, float(th)))
        prev_edge = float(th)
    if sorted_levels.size == 0 or sorted_levels[-1] < 100:
        bands.append((prev_edge, gmax))

    end_time_percentiles = time.time()
    
    total_time_percentiles = end_time_percentiles - start_time_percentiles
    
    if verbose:
        log_message('Time for percentile thresholds computation: ' + str(strftime("%H:%M:%S", gmtime(total_time_percentiles))), tag="percentiles", level=1)

    return {
        "percentiles": thresholds,
        "levels": levels_arr,
        "percentiles_plus": percentiles_plus,
        "percentiles_minus": percentiles_minus,
        "global_min": gmin,
        "global_max": gmax,
        "bands": bands,
    }


def uniform_induction(components, induction_equation,
                    clus_cr0amr, clus_solapst, grid_npatch,
                    grid_patchnx, grid_patchny, grid_patchnz, 
                    grid_patchrx, grid_patchry, grid_patchrz,
                    it, sims, nmax, size, region_coords,
                    up_to_level=4, ncores=1, clus_kp=None, verbose=False):
    '''
    Cleans and computes the uniform section of the magnetic induction energy and its components for the given AMR grid for its further projection.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - induction_equation: dictionary containing the components of the magnetic induction equation computed in the previous step
            - MIE_diver_B_x, MIE_diver_B_y, MIE_diver_B_z: null divergence of the magnetic field
            - MIE_compres_x, MIE_compres_y, MIE_compres_z: compressive component of the magnetic field induction
            - MIE_stretch_x, MIE_stretch_y, MIE_stretch_z: stretching component of the magnetic field induction
            - MIE_advec_x, MIE_advec_y, MIE_advec_z: advection component of the magnetic field induction
            - MIE_drag_x, MIE_drag_y, MIE_drag_z: cosmic drag component of the magnetic field induction
            - MIE_total_x, MIE_total_y, MIE_total_z: total magnetic induction energy in the compact way
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - grid_npatch: number of patches in the grid
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - it: index of the snapshot in the simulation
        - sims: name of the simulation
        - nmax: maximum number of patches in the grid
        - size: size of the grid
        - region_coords: coordinates defining the region of interest
        - up_to_level: level of refinement in the AMR grid (default is 4)
        - ncores: number of cores to use for the computation (default is 1)
        - clus_kp: mask for valid patches
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - uniform_field: cleaned and projected field on a uniform grid
        
    Author: Marco Molina
    '''

    if region_coords[0] != "box":
        raise NotImplementedError("Only 'box' region_coords are implemented for uniform_induction.")
    
    start_time_uniform = time.time() # Record the start time
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    
    results = {}
    
    for key, prefix in [
        ('divergence', 'MIE_diver'),
        ('compression', 'MIE_compres'),
        ('stretching', 'MIE_stretch'),
        ('advection', 'MIE_advec'),
        ('drag', 'MIE_drag'),
        ('total', 'MIE_total')
    ]:
        if components.get(key, False):
            # results[f'uniform_{prefix}_x'], _, _, _ = utils.uniform_field(induction_equation[f'{prefix}_x'], clus_cr0amr, clus_solapst, grid_npatch,
            #                                                 grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
            #                                                 nmax, size, Box, up_to_level=up_to_level, ncores=ncores, clus_kp=clus_kp, verbose=verbose)
            # results[f'uniform_{prefix}_y'], _, _, _ = utils.uniform_field(induction_equation[f'{prefix}_y'], clus_cr0amr, clus_solapst, grid_npatch,
            #                                                 grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
            #                                                 nmax, size, Box, up_to_level=up_to_level, ncores=ncores, clus_kp=clus_kp, verbose=verbose)
            # results[f'uniform_{prefix}_z'], _, _, _ = utils.uniform_field(induction_equation[f'{prefix}_z'], clus_cr0amr, clus_solapst, grid_npatch,
            #                                                 grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
            #                                                 nmax, size, Box, up_to_level=up_to_level, ncores=ncores, clus_kp=clus_kp, verbose=verbose)
            results[f'uniform_{prefix}_x'] = utils.unigrid(
                                                field=induction_equation[f'{prefix}_x'], box_limits=region_coords[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            results[f'uniform_{prefix}_y'] = utils.unigrid(
                                                field=induction_equation[f'{prefix}_y'], box_limits=region_coords[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            results[f'uniform_{prefix}_z'] = utils.unigrid(
                                                field=induction_equation[f'{prefix}_z'], box_limits=region_coords[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            if verbose == True:
                log_message(f'Snap {it} in {sims}: {key} uniform field done', tag="projection", level=1)
                log_message(str(results[f'uniform_{prefix}_x'].shape), tag="projection", level=2)
                log_message(str(results[f'uniform_{prefix}_y'].shape), tag="projection", level=2)
                log_message(str(results[f'uniform_{prefix}_z'].shape), tag="projection", level=2)
        else:
            results[f'uniform_{prefix}_x'] = zero
            results[f'uniform_{prefix}_y'] = zero
            results[f'uniform_{prefix}_z'] = zero
            
    end_time_uniform = time.time()
    
    total_time_uniform = end_time_uniform - start_time_uniform
    
    if verbose == True:
        log_message('Time for uniform field calculation in snap '+ str(grid_npatch) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_uniform))), tag="projection", level=1)
    
    return results

        
def process_iteration(components, velocity_field, dir_grids, dir_gas, dir_params, dir_vortex,
                    sims, it, coords, region_coords, rad, rmin, level, up_to_level,
                    nmax, size, H0, a0, test, units=1, nbins=25, logbins=True,
                    stencil=3, buffer=True, use_siblings=True, interpol='TSC', nghost=1, blend=False,
                    parent=False, parent_interpol=None,
                    bitformat=np.float32, mag=False, sim_characteristics=None,
                    energy_evolution_config=None,
                    energy_evolution=True, profiles=True, induction_profiles=None, pd_profiles=False,
                    projection=False, percentiles=True, 
                    percentile_levels=(95, 90, 75, 50, 25), divergence_filter=None, debug_params=None,
                    production_dissipation=None,
                    return_options=None,
                    gc_worker_end=False, verbose=False):
    '''
    Processes a single iteration of the cosmological magnetic induction equation calculations.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - velocity_field: dictionary containing the velocity field components to be processed (total, solenoidal, compressive)
        - dir_grids: directory containing the grids
        - dir_gas: directory containing the gas data
        - dir_params: directory containing the parameters
        - dir_vortex: directory containing the vortex data
        - sims: name of the simulation
        - it: index of the snapshot in the simulation
        - coords: coordinates of the center of the integration grid
        - region_coords: integration region coordinates
        - rad: radii of the integration area
        - rmin: minimum radius for the radial profile
        - level: level of refinement in the AMR grid
        - up_to_level: level up to which to clean and uniform the fields (default is 4)
        - nmax: maximum number of patches in the grid
        - size: size of the grid
        - H0: Hubble parameter at the present time
        - a0: scale factor of the universe at the present time
        - test: Dictionary containing the parameters for the test fields:
            - test: boolean to use test fields or not
            - x_test, y_test, z_test: 3D grid coordinates.
            - k: Wave number for the sinusoidal test fields.
            - ω: Angular frequency for the sinusoidal test fields.
            - B0: Amplitude of the magnetic field.
        - units: factor to convert the units multiplied by the final result (default is 1
        - nbins: number of bins for the radial profile (default is 25)
        - logbins: boolean to use logarithmic bins (default is True)
        - stencil: stencil size for the magnetic induction equation (default is 3)
        - buffer: boolean to add ghost buffer cells before derivatives (default is True)
        - interpol: interpolation method for the ghost buffer (default is 'TSC')
        - nghost: number of ghost cells to add for the derivatives (default is 1)
        - bitformat: data type for the fields (default is np.float32)
        - mag: boolean to compute magnitudes (default is False)
        - sim_characteristics: Dictionary with simulation characteristics (is_cooling, is_mascletB, etc.)
        - energy_evolution_config: dictionary with energy evolution options
            (evolution_type, derivative, volume_coordinates, normalize_by_volume)
        - energy_evolution: boolean to compute energy evolution (default is True)
        - profiles: legacy boolean to compute induction radial profiles (default is True)
        - induction_profiles: boolean to compute induction radial profiles (default is None -> uses legacy 'profiles')
        - pd_profiles: boolean to compute production/dissipation radial profiles (default is False)
        - projection: boolean to compute uniform projection (default is True)
        - percentiles: boolean to compute percentile thresholds (default is True)
        - percentile_levels: tuple of percentile thresholds to compute (default is (100, 90, 75, 50, 25))
        - divergence_filter: dict with divergence filtering settings (default is None)
        - debug_params: dictionary with debug configuration (default is None, uses empty dict)
        - production_dissipation: dict with production/dissipation options
        - return_options: dictionary with export options from IND_PARAMS["return"] (default is None)
        - verbose: boolean to print progress information (default is False)
        
    Returns:
        - data: dictionary containing the loaded data from the simulation
        - vectorial: dictionary containing the computed vectorial quantities
        - induction: dictionary containing the components of the magnetic induction equation
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_integral: dictionary containing the volume integrals of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_profiles: dictionary containing radial profiles of induction-energy terms
        - production_dissipation_profiles: dictionary containing radial profiles of production/dissipation terms
        - induction_uniform: dictionary containing the uniform projection of the magnetic induction equation in terms of the magnetic energy
        - diver_B_percentiles: dictionary containing percentile thresholds of the magnetic field divergence
        - debug_fields: dictionary containing debug fields if requested

    Author: Marco Molina
    '''

    start_time_Total = time.time() # Record the start time

    if return_options is None:
        return_options = {}
    
    # Extract return configuration flags (vectorial / induction / energy)
    return_vectorial = return_options.get("fields", {}).get("vectorial", False)
    return_induction = return_options.get("fields", {}).get("induction", False)
    return_induction_energy = return_options.get("fields", {}).get("induction_energy", False)

    if energy_evolution_config is None:
        energy_evolution_config = {
            "volume_coordinates": "physical",
            "normalize_by_volume": False,
        }

    if induction_profiles is None:
        induction_profiles = profiles

    # Initialize debug parameters if not provided
    if debug_params is None:
        debug_params = {
            "buffer": {"enabled": False, "verbose": False},
            "divergence": {"enabled": False, "verbose": False},
            "field_analysis": {"enabled": False}
        }

    # Load Simulation Data
    
    ## This are the parameters we will need for each cell together with the magnetic field and the velocity
    ## We read the information for each snap and divide it in the different fields
    
    data = load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, dir_vortex,
                    velocity_field, level, test=test, 
                    bitformat=bitformat, region=region_coords, sim_characteristics=sim_characteristics,
                    verbose=verbose, debug=debug_params.get("divergence", {}) and debug_params.get("patch_analysis", {}))
    levels = utils.create_vector_levels(data['grid_npatch'])
    dx = size/nmax
    resolution = dx / (2 ** levels)
    
    # Run debug tests if enabled
    debug_fields = None
    pipeline_debug_results = None
    scan_pack = None
    if debug_params.get("buffer", {}).get("enabled", False) and velocity_field.get("total", False):
        if verbose:
            log_message(f"\n{'*'*80}", tag="debug", level=1)
            log_message("BUFFER DEBUG MODE ENABLED - Running buffer pipeline validation tests...", tag="debug", level=1)
            log_message(f"{'*'*80}", tag="debug", level=1)
        pipeline_debug_results = debug_module.run_debug_buffer_pipeline(data, size, nmax, nghost=nghost, 
                                                interpol=interpol, use_siblings=use_siblings, 
                                                bitformat=bitformat, 
                                                verbose=debug_params.get("buffer", {}).get("verbose", True))
    elif debug_params.get("buffer", {}).get("enabled", False) and velocity_field.get("total", True):
        if verbose:
            log_message(f"\n{'*'*80}", tag="debug", level=1)
            log_message("BUFFER DEBUG MODE DISABLED - No total velocity field detected. Skipping buffer debug tests.", tag="debug", level=1)
            log_message(f"{'*'*80}", tag="debug", level=1)
            
    # Buffering settings
    
    ## Buffer cells are added to the grid to ensure that the derivatives can be computed correctly at the boundaries of the patches. The number of ghost cells is determined by the stencil size and the buffer settings.
    
    if parent_interpol is None:
        parent_interpol = interpol

    parent_mode = bool(parent)

    blend_active = bool(blend) and bool(buffer)
    boundary_width = 1 if stencil == 3 else 2
    buffer_nghost = nghost
    if parent_mode and not blend_active:
        buffer_nghost = 0
    
    velocity_mappings = [
        ("total", "v"),          # Generates 'vx', 'vy', 'vz' keys
        ("solenoidal", "vsol"),  # Generates 'vsolx', 'vsoly', 'vsolz' keys
        ("compressive", "vcomp") # Generates 'vcompx', 'vcompy', 'vcompz' keys
    ]
    
    active_velocities = {}
    original_fields = None
    
    # Adjust buffer_nghost if blend is active and nghost was not specified
    if blend_active:
        if buffer_nghost == 0:
            buffer_nghost = boundary_width
        if verbose:
            log_message(
                f'Blend active: using buffer nghost={buffer_nghost} (stencil={stencil}) in addition to parent fill',
                tag="buffer",
                level=1
            )
        original_fields = {
                    f'{b}': data[f'clus_{b}'] for b in ['Bx', 'By', 'Bz']
                }
        original_fields['velocities'] = {}
        
    for key, suffix in velocity_mappings:
        if velocity_field.get(key, False):
            active_velocities[key] = {
                'x': data[f'clus_{suffix}x'],
                'y': data[f'clus_{suffix}y'],
                'z': data[f'clus_{suffix}z']
            }
            if blend_active:
                original_fields['velocities'][key] = {
                    'x': data[f'clus_{suffix}x'],
                    'y': data[f'clus_{suffix}y'],
                    'z': data[f'clus_{suffix}z']
                }
            
    # Add ghost buffer cells before derivatives
    run_buffer = buffer and buffer_nghost > 0
    if run_buffer:
        fields_to_buffer = [data['clus_Bx'], data['clus_By'], data['clus_Bz']]
        field_names = ['Bx', 'By', 'Bz']
        
    for key, suffix in velocity_mappings:
        if velocity_field.get(key, False):
            active_velocities[key] = {
                'x': data[f'clus_{suffix}x'],
                'y': data[f'clus_{suffix}y'],
                'z': data[f'clus_{suffix}z']
            }
            if run_buffer:
                fields_to_buffer.extend([data[f"clus_{suffix}x"], data[f"clus_{suffix}y"], data[f"clus_{suffix}z"]])
                field_names.extend([f"{suffix}x", f"{suffix}y", f"{suffix}z"])
                
    if run_buffer:
        buffered_field = buff.add_ghost_buffer(
            fields_to_buffer,
            data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
            size=size, nmax=nmax, nghost=buffer_nghost, interpol=interpol, use_siblings=use_siblings,
            kept_patches=data['clus_kp']
        )
        
        for name, array in zip(field_names, buffered_field):
            data[f'clus_{name}'] = array
            
        for key, suffix in velocity_mappings:
            if key in active_velocities:
                active_velocities[key] = {
                    'x': data[f'clus_{suffix}x'],
                    'y': data[f'clus_{suffix}y'],
                    'z': data[f'clus_{suffix}z']
                }
            
        if verbose:
            log_message('Ghost buffer added to magnetic and velocity fields', tag="buffer", level=1)

    # Vectorial calculus
    ## Here we calculate the different vectorial calculus quantities of our interest using the diff module.
    vectorial = vectorial_quantities(
        components, 
        data['clus_Bx'], data['clus_By'], data['clus_Bz'],
        active_velocities,
        data['clus_kp'], data['grid_npatch'], data['grid_irr'],
        dx, stencil=stencil, verbose=verbose
    )
            
    # Remove ghost buffer cells after derivatives (skip if parent mode uses frontier fill)
    if buffer and buffer_nghost > 0:
        if verbose:
            log_message('Removing ghost buffer from computed vectorial fields', tag="buffer", level=1)
        for key in vectorial.keys():
            vectorial[key] = buff.ghost_buffer_buster(
                vectorial[key], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                buffer_nghost, kept_patches=data['clus_kp']
            )
            
        if verbose:
            log_message('Removing ghost buffer from magnetic and velocity fields', tag="buffer", level=1)
        
        fields_to_clean = ['Bx', 'By', 'Bz']
        suffix_map = {"total": "v", "solenoidal": "vsol", "compressive": "vcomp"}
        for vel_type in active_velocities.keys():
            suffix = suffix_map[vel_type]
            fields_to_clean.extend([f"{suffix}x", f"{suffix}y", f"{suffix}z"])
                
        for key in fields_to_clean:
            data[f'clus_{key}'] = buff.ghost_buffer_buster(
                data[f'clus_{key}'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                buffer_nghost, kept_patches=data['clus_kp']
            )

        for vel_type, suffix in suffix_map.items():
            if vel_type in active_velocities:
                active_velocities[vel_type] = {
                    'x': data[f'clus_{suffix}x'],
                    'y': data[f'clus_{suffix}y'],
                    'z': data[f'clus_{suffix}z']
                }

    elif buffer and parent_mode and not blend_active:
        buffered_field = buff.add_ghost_buffer(
            [vectorial[key] for key in vectorial.keys()],
            data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
            size=size, nmax=nmax, nghost=0, interpol=parent_interpol, use_siblings=False,
            kept_patches=data['clus_kp']
        )
        
        for i, key in enumerate(vectorial.keys()):
            vectorial[key] = buffered_field[i]
            
        if verbose:
            log_message(
                f'Parent frontier filling applied to vectorial fields (parent_interpol={parent_interpol})',
                tag="buffer",
                level=1
            )

    if blend_active:
        vectorial_no_buffer = vectorial_quantities(
            components,
            original_fields['Bx'], original_fields['By'], original_fields['Bz'],
            original_fields['velocities'],
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            dx, stencil=stencil, verbose=False
        )

        parent_use_siblings = False
        parent_field = buff.add_ghost_buffer(
            [vectorial_no_buffer[key] for key in vectorial_no_buffer.keys()],
            data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
            size=size, nmax=nmax, nghost=0, interpol=parent_interpol, use_siblings=parent_use_siblings,
            kept_patches=data['clus_kp']
        )

        for idx, key in enumerate(vectorial.keys()):
            vectorial[key] = buff.blend_patch_boundaries(
                vectorial[key], parent_field[idx],
                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                boundary_width=boundary_width, kept_patches=data['clus_kp']
            )

        if verbose:
            log_message(
                f'Blend applied: boundary cells are averaged between buffer and parent fill '
                f'(parent_interpol={parent_interpol})',
                tag="buffer",
                level=1
            )

    diver_B_raw = vectorial.get('diver_B')
    div_filter_config = divergence_filter or {}
    filter_enabled = div_filter_config.get("enabled", False)

    if filter_enabled and not _debug_enabled(debug_params):
        method = divergence_filter.get("method", "mask")
        percentile = divergence_filter.get("percentile", 99)
        use_abs = divergence_filter.get("use_abs", True)
        exclude_zeros = divergence_filter.get("exclude_zeros", True)
        if verbose:
            log_message(
                f"Divergence filter: Attempting to filter with method={method}, percentile={percentile}",
                tag="divergence_filter",
                level=1
            )
        vectorial['diver_B'], _ = filter_divergence_outliers(
            diver_B_raw,
            kept_patches=data['clus_kp'],
            method=method,
            percentile=percentile,
            use_abs=use_abs,
            exclude_zeros=exclude_zeros,
            verbose=verbose
        )
    elif filter_enabled and _debug_enabled(debug_params) and verbose:
        log_message(
            "Divergence filter: SKIPPED (debug mode active)",
            tag="divergence_filter",
            level=1
        )

    # Magnetic Induction Equation
    
    ## In this section we are going to compute the cosmological induction equation and its components, calculating them with the results obtained before.
    ## This will be usefull to plot fluyd maps as the quantities involved are vectors.

    pd_enabled = False
    pd_fractional_any_enabled = False
    if isinstance(production_dissipation, dict):
        pd_enabled = bool(production_dissipation.get('enabled', False))
        pd_fractional_any_enabled = bool(
            production_dissipation.get('plot_fractional', False)
            or production_dissipation.get('plot_fractional_profiles', False)
        )
    elif production_dissipation is not None:
        pd_enabled = bool(production_dissipation)

    pd_integrals_enabled = False
    if isinstance(production_dissipation, dict):
        pd_integrals_enabled = bool(
            pd_enabled and (
                production_dissipation.get('plot_absolute', False)
                or production_dissipation.get('plot_fractional', False)
                or production_dissipation.get('plot_net', False)
            )
        )
    elif production_dissipation is not None:
        pd_integrals_enabled = bool(pd_enabled)
    
    # Determine if induction needs to be calculated based on downstream dependencies
    # Production/dissipation requires induction -> induction_energy -> integrals.
    compute_induction = (
        return_induction or return_induction_energy or
        energy_evolution or induction_profiles or pd_profiles or projection or mag or percentiles or pd_enabled
    )
    
    if compute_induction:
        induction, magnitudes = induction_equation(
            components, vectorial,
            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
            active_velocities,
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            data['H'], data['a'], mag=mag, verbose=verbose
        )
    else:
        induction, magnitudes = None, None
        if verbose:
            log_message('Induction equation skipped (not required by any enabled output).', tag="pipeline", level=1)
    
    # Magnetic Induction Equation in Terms of the Magnetic Energy
    
    ## In this section we are going to compute the cosmological induction equation in terms of the magnetic energy and its components, calculating them with the results obtained before.
    ## This will be usefull to calculate volumetric integrals and energy budgets as the quantities involved are scalars.
    
    # Determine if induction_energy needs to be calculated
    compute_induction_energy = return_induction_energy or energy_evolution or induction_profiles or pd_profiles or pd_enabled
    
    if compute_induction_energy and induction is not None:
        induction_energy = induction_equation_energy(
            components, velocity_field, induction,
            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
            data['clus_rho_rho_b'], data['clus_v2'],
            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
            verbose=verbose
        )
    else:
        induction_energy = None
        if verbose:
            if compute_induction_energy and induction is None:
                log_message('Induction energy skipped (induction not available).', tag="pipeline", level=1)
            elif not compute_induction_energy:
                log_message('Induction energy skipped (not required by any enabled output).', tag="pipeline", level=1)
    
    # Compute production/dissipation fields
    
    ## Here we compute the production and dissipation fields based on the induction energy and other relevant quantities.
    if pd_enabled and induction_energy is not None:
        
        pd_fields = production_dissipation_fields(
            components,
            velocity_field,
            induction_energy,
            data['clus_kp'],
            data['grid_npatch'],
            data['grid_irr'],
            verbose=verbose
        )
        induction_energy.update(pd_fields)
    else:
        if verbose:
            if not pd_enabled:
                log_message('Production/disipation set to False, skipping production/dissipation fields calculation.', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('Production/disipation fields skipped (induction energy not available).', tag="pipeline", level=1)

    # Volume Integral of the Magnetic Induction Equation

    ## Here we compute the volume integral of the magnetic energy density and its components, as well as the induced magnetic energy.
    ## This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied volume. The kinetic energy density is also computed.
    if (energy_evolution or pd_integrals_enabled) and induction_energy is not None:
        
        induction_energy_integral = {}

        if energy_evolution:
            evo_integral = induction_vol_integral(
                components, velocity_field, induction_energy, data['clus_b2'],
                data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                it, sims, nmax, size, coords, region_coords, rad,
                units=1, production_dissipation=False,
                rho_b=data.get('rho_b', None),
                volume_coordinates=energy_evolution_config.get('volume_coordinates', 'physical'),
                normalize_by_volume=energy_evolution_config.get('normalize_by_volume', False),
                integration_label='evolution',
                verbose=verbose
            )
            induction_energy_integral.update(evo_integral)

        if pd_enabled:
            pd_cfg = production_dissipation if isinstance(production_dissipation, dict) else {}
            pd_integral = induction_vol_integral(
                components, velocity_field, induction_energy, data['clus_b2'],
                data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                it, sims, nmax, size, coords, region_coords, rad,
                units=1, production_dissipation=production_dissipation,
                rho_b=data.get('rho_b', None),
                volume_coordinates=pd_cfg.get('volume_coordinates', 'physical'),
                normalize_by_volume=pd_cfg.get('normalize_by_volume', False),
                compute_induction_integrals=energy_evolution,
                compute_fractional_integrals=pd_fractional_any_enabled,
                integration_label='pd',
                verbose=verbose
            )

            if energy_evolution:
                pd_only = {
                    key: value for key, value in pd_integral.items()
                    if key in ('int_b2', 'int_B2') or key.startswith('int_PD_') or ('_prod' in key) or ('_diss' in key)
                }
                induction_energy_integral.update(pd_only)
            else:
                induction_energy_integral.update(pd_integral)
        
        if test['test'] == True:
            
            induction_test_energy = analytic_test_fields(data['grid_time'], data['grid_npatch'], data['a'], data['H'], data['clus_Bx'], test)
            
            induction_test_energy_integral = induction_vol_integral(components, induction_test_energy, data['clus_b2'],
                        data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                        data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                        data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                        it, sims, nmax, size, coords, region_coords, rad,
                        units=1, production_dissipation=False,
                        rho_b=data.get('rho_b', None),
                        volume_coordinates=energy_evolution_config.get('volume_coordinates', 'physical'),
                        normalize_by_volume=energy_evolution_config.get('normalize_by_volume', False),
                        integration_label='test',
                        verbose=verbose)
        else:
            induction_test_energy_integral = None
    else:
        induction_energy_integral = None
        induction_test_energy_integral = None
        if verbose == True:
            if not energy_evolution and not pd_enabled:
                log_message('Energy evolution and production/dissipation are disabled, skipping volume integral of the magnetic induction equation.', tag="pipeline", level=1)
            elif not energy_evolution and pd_enabled:
                log_message('Energy evolution is set to False, skipping volume integral evolution of the magnetic energy induction equation.', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('Energy evolution skipped (induction_energy not available).', tag="pipeline", level=1)
            
    # Radial Profiles of the Magnetic Induction Equation

    ## We can calculate the radial profiles of the magnetic energy density in the volume we have considered (usually the virial volume)
    if induction_profiles and induction_energy is not None:
        induction_energy_profiles = induction_radial_profiles(
            components, velocity_field, induction_energy, data['clus_b2'],
            data['clus_rho_rho_b'], data['rho_b'],
            data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
            data['grid_irr'], data['grid_npatch'], up_to_level,
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            it, sims, nmax, size, coords, rmin, rad,
            nbins=nbins, logbins=logbins, units=1, verbose=verbose
        )
    else:
        induction_energy_profiles = None
        if verbose:
            if not induction_profiles:
                log_message('Induction profiles are set to False, skipping radial profiles of the magnetic induction equation.', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('Induction radial profiles skipped (induction_energy not available).', tag="pipeline", level=1)

    # Radial Profiles of the Production/Dissipation Terms
    
    ## Here we can calculate the radial profiles of the production and dissipation terms in the volume we have considered (usually the virial volume)
    if pd_profiles and pd_enabled and induction_energy is not None:
        production_dissipation_profiles = production_dissipation_radial_profiles(
            components, velocity_field, induction_energy,
            data['rho_b'], data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
            data['grid_irr'], data['grid_npatch'], up_to_level,
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            it, sims, nmax, size, coords, rmin, rad,
            nbins=nbins, logbins=logbins, units=1,
            compute_fractional_profiles=pd_fractional_any_enabled,
            verbose=verbose
        )
    else:
        production_dissipation_profiles = None
        if verbose:
            if not pd_profiles:
                log_message('P/D radial profiles are set to False, skipping P/D profiles.', tag="pipeline", level=1)
            elif not pd_enabled:
                log_message('P/D radial profiles skipped (production_dissipation disabled).', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('P/D radial profiles skipped (induction_energy not available).', tag="pipeline", level=1)
    
    # Uniform Projection of the Magnetic Induction Equation

    ## We clean and compute the uniform section of the magnetic induction energy and its components for the given AMR grid for its further projection.
    if projection and induction is not None:
        
        induction_uniform = uniform_induction(components, induction,
                            data['clus_cr0amr'], data['clus_solapst'], data['grid_npatch'],
                            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                            it, sims, nmax, size, region_coords,
                            up_to_level=up_to_level, ncores=1, clus_kp=data['clus_kp'],
                            verbose=verbose)
    else:
        induction_uniform = None
        if verbose == True:
            if not projection:
                log_message('Projection is set to False, skipping uniform projection of the magnetic induction equation.', tag="pipeline", level=1)
            elif induction is None:
                log_message('Uniform projection skipped (induction not available).', tag="pipeline", level=1)
            
    # Percentile Thresholds of the Magnetic Field Divergence
    
    # Scale divergence by resolution to make it comparable to field magnitude
    # Divergence has units [field/length], multiplying by dx gives [field]
    
    # Extract percentile calculation options from debug_params if available
    if percentiles:
        percentile_params = debug_params.get("percentile_params", {}) if debug_params else {}
        exclude_boundaries = percentile_params.get("exclude_boundaries", False)
        boundary_width = percentile_params.get("boundary_width", 1)
        exclude_zeros = percentile_params.get("exclude_zeros", True)
        
        diver_B_percentiles = compute_percentile_thresholds(
            field_numerator=diver_B_raw,
            field_denominator=data['clus_B'],
            scale_factor=resolution,
            cr0amr=data['clus_cr0amr'],
            solapst=data['clus_solapst'],
            npatch=data['grid_npatch'],
            up_to_level=up_to_level,
            percentiles=percentile_levels,
            use_abs=True,
            denom_eps=0.0,
            kept_patches=data['clus_kp'],
            exclude_boundaries=exclude_boundaries,
            boundary_width=boundary_width,
            exclude_zeros=exclude_zeros,
            verbose=verbose
        )
    elif not percentiles:
        diver_B_percentiles = None
        if verbose == True:
            log_message('Percentiles is set to False, skipping percentile thresholds of the magnetic field divergence.', tag="pipeline", level=1)

    # Build scan visualization volume if requested (pure-debug data)
    if debug_params.get("scan_animation", {}).get("enabled", False):
        scan_pack = debug_module.build_scan_animation_data(
            data=data,
            size=size,
            nmax=nmax,
            region_coords=region_coords,
            nghost=nghost,
            use_siblings=use_siblings,
            up_to_level=up_to_level,
            bitformat=bitformat,
            verbose=debug_params.get("scan_animation", {}).get("verbose", False),
            clean_output=debug_params.get("clean_output", False)
        )
            
    if debug_params.get("field_analysis", {}).get("enabled", False):
        field_sources = [data, vectorial, induction, induction_energy]
        field_list = debug_params.get("field_analysis", {}).get("field_list", None)
        debug_fields = debug_module.analyze_debug_fields(
            field_sources=field_sources,
            region_coords=region_coords,
            data=data,
            debug_params=debug_params,
            up_to_level=up_to_level,
            size=size,
            nmax=nmax,
            pipeline_debug_results=pipeline_debug_results,
            scan_pack=scan_pack,
            verbose=verbose,
            it=it,
            sims=sims,
            field_list=field_list
        )
                
    else:
        # If no field analysis, still return any available debug artifacts
        if pipeline_debug_results is not None or scan_pack is not None:
            debug_fields = {}
            if pipeline_debug_results is not None:
                debug_fields['_pipeline_validation'] = pipeline_debug_results
            if scan_pack is not None:
                debug_fields.update(scan_pack)
        else:
            debug_fields = None

    if isinstance(return_options, dict) and return_options.get('enabled', False):
        start_time_export = time.time()
        utils.export_snapshot_fields(
            data=data,
            vectorial=vectorial,
            induction=induction,
            induction_energy=induction_energy,
            induction_energy_integral=induction_energy_integral,
            induction_test_energy_integral=induction_test_energy_integral,
            induction_energy_profiles=induction_energy_profiles,
            production_dissipation_profiles=production_dissipation_profiles,
            diver_B_percentiles=diver_B_percentiles,
            induction_uniform=induction_uniform,
            debug_fields=debug_fields,
            rad=rad,
            export_cfg=return_options,
            sim_name=sims,
            iteration=it,
            level=level,
            up_to_level=up_to_level,
            nmax=nmax,
            size=size,
            region_coords=region_coords,
            bitformat=bitformat,
            verbose=verbose,
        )
        total_time_export = time.time() - start_time_export
        if verbose == True:
            log_message(
                f'Time for writing return data in snap {it} in simulation {sims}: '
                f'{strftime("%H:%M:%S", gmtime(total_time_export))}',
                tag="export",
                level=1
            )
        
    data = {
        'grid_irr': data['grid_irr'],
        'grid_time': data['grid_time'],
        'grid_zeta': data['grid_zeta'],
        'rho_b': data['rho_b'],
        'resolution': resolution
    }
    
    # Apply return flags to filter outputs
    if not return_vectorial:
        vectorial = None
    if not return_induction:
        induction = None
    if not return_induction_energy:
        induction_energy = None
            
    end_time_Total = time.time()
    
    total_time_Total = end_time_Total - start_time_Total
    
    if verbose == True:
        log_message(f'Time for processing iteration {it} in simulation {sims}: {strftime("%H:%M:%S", gmtime(total_time_Total))}', tag="pipeline", level=1)

    if gc_worker_end:
        gc.collect()

    return data, induction_energy_integral, induction_test_energy_integral, induction_energy_profiles, production_dissipation_profiles, induction_uniform, diver_B_percentiles, debug_fields