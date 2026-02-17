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
np.set_printoptions(linewidth=200)


# Use log_message from utils
log_message = utils.log_message


def find_most_massive_halo(sims, it, a0, dir_halos, dir_grids, data_folder, vir_kind=1, rad_kind=1, verbose=False):
    '''
    Finds the coordinates and radius of the most massive halo in each snapshot of the simulations. In case
    we are looking for the most massive halo to center our analysis, we need to build the python halo catalogue
    (by now we exclude subhalos)
    
    Args:
        - sims: list of simulation names
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

    for i in range(len(sims)):
        # Read halos and zeta for each snapshot in reverse order so that we can track the same most massive halo
        for j in reversed(range(len(it))):
            
            halos = reader.read_families(it[j], path=dir_halos, output_format='dictionaries', output_redshift=False,
                        min_mass=None, exclude_subhaloes=True, read_region=None, keep_boundary_contributions=False)
            
            _,_,_,_,zeta = reader.read_grids(it = it[j], path=dir_grids+sims[i], parameters_path=data_folder+'/'+sims[i]+'/', digits=5, read_general=True, read_patchnum=False, read_dmpartnum=False,
            read_patchcellextension=False, read_patchcellposition=False, read_patchposition=False, read_patchparent=False, nparray=False)
            
            if j == len(it) - 1:
                # Find the index of the most massive halo
                max_mass_index = np.argmax([halo['M'] for halo in halos])
                id_max_mass = halos[max_mass_index]['id']
                max_halo_mass = halos[max_mass_index]['M']
                if vir_kind == 1:
                    R_max_mass = halos[max_mass_index]['R']
            
            index = next((i for i, halo in enumerate(halos) if halo['id'] == id_max_mass), None)
            
            if index != None:
                
                coords.append((halos[index]['x'], halos[index]['y'], halos[index]['z']))
                
                if vir_kind == 1 and rad_kind == 1:
                    rad.append(R_max_mass) # Taking the Virial radius of the most massive halo at the last snap
                elif vir_kind == 1 and rad_kind == 2:
                    rad.append(R_max_mass * (a0/(1 + zeta)))
                elif vir_kind == 2 and rad_kind == 1:
                    rad.append(halos[index]['R']) # Changing the virial radius at each snap
                elif vir_kind == 2 and rad_kind == 2:
                    rad.append(halos[index]['R'] * (a0/(1 + zeta)))
                            
            elif index == None:
                
                coords.append(coords[-1])
                rad.append(rad[-1])
                
            if verbose and index == None:
                log_message("No halo found in snap " + str(it[j]) + ", using the previous one.", tag="halo", level=1)
                
    if verbose:
        
        # Print the coordinates
        log_message("Coordinates of the most massive halo in the last snap " + str(it[-1]) + ":", tag="halo", level=1)
        log_message("x: " + str(coords[-1][0]), tag="halo", level=2)
        log_message("y: " + str(coords[-1][1]), tag="halo", level=2)
        log_message("z: " + str(coords[-1][2]), tag="halo", level=2)
        log_message("Radius: " + str(rad[-1]) + " Mpc", tag="halo", level=2)
        if max_halo_mass is not None:
            log_message("Mass: " + str(max_halo_mass) + " Msun/h", tag="halo", level=2)

    # Reverse the lists to match the original order of snapshots            
    coords = coords[::-1]
    rad = rad[::-1]
    
    return coords, rad


def create_region(sims, it, coords, rad, size, F=1.0, reg='BOX', verbose=False):
    '''
    Creates the boxes or spheres centered at the coordinates of the most massive halo or any other point in each snapshot.
    Automatically clips regions to simulation box boundaries and disables region reading if the clipped region
    equals the entire box.
    
    Args:
        - sims: list of simulation names
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
    
    # Handle size as list or single value
    if isinstance(size, list):
        box_size = size[0]  # Assume first simulation
    else:
        box_size = size
        
    # Box boundaries (simulation box is centered at origin)
    box_min = -box_size / 2.0
    box_max = box_size / 2.0
    
    Rad = []
    region_size = []
    region = []

    for i in range(len(sims)):
        for j in range(len(it)):
            Rad.append(F*rad[i+j])
            region_size.append(2 * Rad[-1])  # Size of the box in Mpc
            
            if reg == 'BOX':
                # Calculate region boundaries
                x1 = coords[i+j][0] - Rad[-1]
                x2 = coords[i+j][0] + Rad[-1]
                y1 = coords[i+j][1] - Rad[-1]
                y2 = coords[i+j][1] + Rad[-1]
                z1 = coords[i+j][2] - Rad[-1]
                z2 = coords[i+j][2] + Rad[-1]
                
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
                    abs(coords[i+j][0]) + effective_radius,
                    abs(coords[i+j][1]) + effective_radius,
                    abs(coords[i+j][2]) + effective_radius
                )
                
                if max_extent > box_max:
                    if verbose and j == 0:
                        log_message(f"Warning: Spherical region for snapshot {it[j]} extends beyond simulation box.", tag="region", level=1)
                        log_message(f"Center: ({coords[i+j][0]:.2f}, {coords[i+j][1]:.2f}, {coords[i+j][2]:.2f})", tag="region", level=2)
                        log_message(f"Radius: {effective_radius:.2f} Mpc", tag="region", level=2)
                        log_message("Region equals entire box -> disabling region filter (reading all patches)", tag="region", level=2)
                    region.append([None])
                else:
                    region.append(["sphere", coords[i+j][0], coords[i+j][1], coords[i+j][2], Rad[-1]])
            else:
                region.append([None])
                
    if verbose:      
        # Print the coordinates
        if region[-1][0] is None:
            log_message("Region: None (using entire simulation box)", tag="region", level=1)
        else:
            log_message(str(region[-1][0]) + " region: " + str(region[-1]), tag="region", level=1)

    return region, region_size


def load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, level, test, bitformat=np.float32, region=None, verbose=False, debug=False):
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
        - level: level of the AMR grid to be used
        - test: Dictionary containing the parameters for the test fields:
            - test: boolean to use test fields or not
            - x_test, y_test, z_test: 3D grid coordinates.
            - k: Wave number for the sinusoidal test fields.
            - ω: Angular frequency for the sinusoidal test fields.
            - B0: Amplitude of the magnetic field.
        - bitformat: data type for the loaded fields (default is np.float32)
        - region: region coordinates to be used (default is None)
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
            - clus_vx, clus_vy, clus_vz: velocity field components in the cluster
            - clus_cr0amr: cosmic ray energy density in the cluster
            - clus_solapst: solenoidal fraction in the cluster
            - clus_kp: mask for valid patches
            - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
            - clus_B: magnetic field magnitude in the cluster
            - clus_b2: normalized magnetic field squared in the cluster
            - clus_B2: magnetic field squared in the cluster
            - clus_v2: velocity field squared in the cluster
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

        # Read cluster data
        clus = reader.read_clus(
            it=it,
            path=dir_gas + sims,
            parameters_path=dir_params,
            digits=5,
            max_refined_level=level,
            output_delta=True,
            output_v=True,
            output_pres=False,
            output_pot=False,
            output_opot=False,
            output_temp=False,
            output_metalicity=False,
            output_cr0amr=True,
            output_solapst=True,
            is_mascletB=True,
            output_B=True,
            is_cooling=False,
            verbose=False,
            read_region=region
        )

        # Unpack cluster data
        (
            clus_rho_rho_b,
            clus_vx,
            clus_vy,
            clus_vz,
            clus_cr0amr,
            clus_solapst,
            clus_mbx,
            clus_mby,
            clus_mbz,
            *rest
        ) = clus
        # Slice cluster fields to kept patch count to stay aligned with grid_npatch
        clus_rho_rho_b = clus_rho_rho_b[:keep_count]
        clus_vx = clus_vx[:keep_count]
        clus_vy = clus_vy[:keep_count]
        clus_vz = clus_vz[:keep_count]
        clus_cr0amr = clus_cr0amr[:keep_count]
        clus_solapst = clus_solapst[:keep_count]
        clus_mbx = clus_mbx[:keep_count]
        clus_mby = clus_mby[:keep_count]
        clus_mbz = clus_mbz[:keep_count]
        if rest:
            rest = [r[:keep_count] if hasattr(r, '__len__') else r for r in rest]

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
        # Read cluster data
        clus = reader.read_clus(
            it=it,
            path=dir_gas + sims,
            parameters_path=dir_params,
            digits=5,
            max_refined_level=level,
            output_delta=True,
            output_v=False,
            output_pres=False,
            output_pot=False,
            output_opot=False,
            output_temp=False,
            output_metalicity=False,
            output_cr0amr= False,
            output_solapst= False,
            is_mascletB=True,
            output_B=False,
            is_cooling=False,
            verbose=False,
            read_region=region
        )

        # Unpack cluster data
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
    
    n = 1 + np.sum(grid_npatch)
    
    # Determine mask for valid patches
    if region is not None and rest:
        clus_kp = rest[0]
    else:
        clus_kp = np.ones(n, dtype=bool)

    # Normalize magnetic field components
    # clus_Bx = [clus_mbx[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    # clus_By = [clus_mby[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    # clus_Bz = [clus_mbz[p] / np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    
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
    
    clus_bx = [clus_mbx[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_by = [clus_mby[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_bz = [clus_mbz[p] * np.sqrt(rho_b) if bool(clus_kp[p]) else 0 for p in range(n)]

    if verbose == True:
        log_message('Data type loaded for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype), tag="data", level=1)

    # Convert to float64 if not transforming to uniform grid
    if bitformat == np.float64:
        clus_rho_rho_b = [(1+clus_rho_rho_b[p]).astype(np.float64) if bool(clus_kp[p]) else (1+clus_rho_rho_b[p]) for p in range(n)] # Delta is (rho/rho_b) - 1
        clus_vx = [clus_vx[p].astype(np.float64) if bool(clus_kp[p]) else clus_vx[p] for p in range(n)]
        clus_vy = [clus_vy[p].astype(np.float64) if bool(clus_kp[p]) else clus_vy[p] for p in range(n)]
        clus_vz = [clus_vz[p].astype(np.float64) if bool(clus_kp[p]) else clus_vz[p] for p in range(n)]
        clus_bx = [clus_bx[p].astype(np.float64) if bool(clus_kp[p]) else clus_bx[p] for p in range(n)]
        clus_by = [clus_by[p].astype(np.float64) if bool(clus_kp[p]) else clus_by[p] for p in range(n)]
        clus_bz = [clus_bz[p].astype(np.float64) if bool(clus_kp[p]) else clus_bz[p] for p in range(n)]
        clus_Bx = [clus_Bx[p].astype(np.float64) if bool(clus_kp[p]) else clus_Bx[p] for p in range(n)]
        clus_By = [clus_By[p].astype(np.float64) if bool(clus_kp[p]) else clus_By[p] for p in range(n)]
        clus_Bz = [clus_Bz[p].astype(np.float64) if bool(clus_kp[p]) else clus_Bz[p] for p in range(n)]

    clus_b2 = [clus_bx[p]**2 + clus_by[p]**2 + clus_bz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_B2 = [clus_Bx[p]**2 + clus_By[p]**2 + clus_Bz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_B = [np.sqrt(clus_B2[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    clus_v2 = [clus_vx[p]**2 + clus_vy[p]**2 + clus_vz[p]**2 if bool(clus_kp[p]) else 0 for p in range(n)]

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
        'clus_cr0amr': clus_cr0amr,
        'clus_solapst': clus_solapst,
        'clus_kp': clus_kp,
        'clus_Bx': clus_Bx,
        'clus_By': clus_By,
        'clus_Bz': clus_Bz,
        'clus_B': clus_B,
        'clus_b2': clus_b2,
        'clus_B2': clus_B2,
        'clus_v2': clus_v2,
        'a': a,
        'E': E,
        'H': H,
        'rho_b': rho_b
    }
    
    return results


def vectorial_quantities(components, clus_Bx, clus_By, clus_Bz,
                        clus_vx, clus_vy, clus_vz,
                        clus_kp, grid_npatch, grid_irr,
                        dx, stencil=3, verbose=False):
    '''
    Computes the vectorial calculus quantities of interest for the magnetic field and velocity field.
    Only the the necessary quantities are computed based on the components specified in the config "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - clus_Bx, clus_By, clus_Bz: magnetic field components in the cluster
        - clus_vx, clus_vy, clus_vz: velocity field components in the cluster
        - clus_kp: mask for valid patches
        - grid_npatch: number of patches in the grid
        - grid_irr: index of the snapshot
        - dx: size of the cells in Mpc
        - stencil: stencil to be used for the calculations
        - buffer_active: boolean to use buffer zones in the differential calculations (default is False)
        - nghost: number of ghost cells to be used in the differential calculations (default is 1)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed vectorial quantities:
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
        
    if components.get('compression', False):
        ### We compute the divergence of the velocity field
        results['diver_v'] = diff.divergence(clus_vx, clus_vy, clus_vz, dx, grid_npatch, clus_kp, stencil)
    else:
        results['diver_v'] = zero
        
    if components.get('stretching', False):
        ### We compute the directional derivative of the velocity field along the magnetic field
        
        results['B_nabla_v_x'], results['B_nabla_v_y'], results['B_nabla_v_z'] = diff.directional_derivative_vector_field(clus_vx, clus_vy, clus_vz, clus_Bx, clus_By, clus_Bz, dx, grid_npatch, clus_kp, stencil)
    else:
        results['B_nabla_v_x'] = zero
        results['B_nabla_v_y'] = zero
        results['B_nabla_v_z'] = zero
        
    if components.get('advection', False):
        ### We compute the directional derivative of the magnetic field along the velocity field
    
        results['v_nabla_B_x'], results['v_nabla_B_y'], results['v_nabla_B_z'] = diff.directional_derivative_vector_field(clus_Bx, clus_By, clus_Bz, clus_vx, clus_vy, clus_vz, dx, grid_npatch, clus_kp, stencil)
    else:
        results['v_nabla_B_x'] = zero
        results['v_nabla_B_y'] = zero
        results['v_nabla_B_z'] = zero
        
    if components.get('total', False):
        ### We compute the cross product of the velocity and magnetic field

        v_X_B_x = [clus_vy[p] * clus_Bz[p] - clus_vz[p] * clus_By[p] if bool(clus_kp[p]) else 0 for p in range(n)] # We run across all the patches with the levels we are interested in
        v_X_B_y = [clus_vz[p] * clus_Bx[p] - clus_vx[p] * clus_Bz[p] if bool(clus_kp[p]) else 0 for p in range(n)] # We only want the patches inside the region of interest
        v_X_B_z = [clus_vx[p] * clus_By[p] - clus_vy[p] * clus_Bx[p] if bool(clus_kp[p]) else 0 for p in range(n)]

        ### The total induction as the curl of the cross product of the velocity and magnetic field with drag term.
        
        results['curl_v_X_B_x'], results['curl_v_X_B_y'], results['curl_v_X_B_z'] = diff.curl(v_X_B_x, v_X_B_y, v_X_B_z, dx, grid_npatch, clus_kp, stencil)
    else:
        results['curl_v_X_B_x'] = zero
        results['curl_v_X_B_y'] = zero
        results['curl_v_X_B_z'] = zero

    end_time_vector = time.time()

    total_time_vector = end_time_vector - start_time_vector
    
    if verbose == True:
        log_message('Time for vector calculations in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_vector))), tag="vector", level=1)
        
    return results
    

def induction_equation(components, vectorial_quantities,
                        clus_Bx, clus_By, clus_Bz,
                        clus_vx, clus_vy, clus_vz,
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
        - clus_vx, clus_vy, clus_vz: velocity field components in the cluster
        - clus_kp: mask for valid patches
        - grid_npatch: number of patches in the grid
        - grid_irr: index of the snapshot
        - H: Hubble parameter
        - a: scale factor of the universe
        - mag: boolean to compute the magnitudes of the components (default is False)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed components of the magnetic induction equation:
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
    
    if components.get('divergence', False):
        ### The null divergence of the magnetic field for numerical error purposes.
        
        results['MIE_diver_x'] = [((1/a) * clus_vx[p] * vectorial_quantities['diver_B'][p]) if bool(clus_kp[p]) else 0 for p in range(n)] # We have to run across all the patches.
        results['MIE_diver_y'] = [((1/a) * clus_vy[p] * vectorial_quantities['diver_B'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_diver_z'] = [((1/a) * clus_vz[p] * vectorial_quantities['diver_B'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['MIE_diver_x'] = zero
        results['MIE_diver_y'] = zero
        results['MIE_diver_z'] = zero

    if components.get('compression', False):
        ### The compressive component.

        results['MIE_compres_x'] = [(-(1/a) * clus_Bx[p] * vectorial_quantities['diver_v'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_compres_y'] = [(-(1/a) * clus_By[p] * vectorial_quantities['diver_v'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_compres_z'] = [(-(1/a) * clus_Bz[p] * vectorial_quantities['diver_v'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['MIE_compres_x'] = zero
        results['MIE_compres_y'] = zero
        results['MIE_compres_z'] = zero

    if components.get('stretching', False):
        ### The stretching component.

        results['MIE_stretch_x'] = [((1/a) * vectorial_quantities['B_nabla_v_x'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_stretch_y'] = [((1/a) * vectorial_quantities['B_nabla_v_y'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_stretch_z'] = [((1/a) * vectorial_quantities['B_nabla_v_z'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['MIE_stretch_x'] = zero
        results['MIE_stretch_y'] = zero
        results['MIE_stretch_z'] = zero

    if components.get('advection', False):
        ### The advection component.

        results['MIE_advec_x'] = [(-(1/a) * vectorial_quantities['v_nabla_B_x'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_advec_y'] = [(-(1/a) * vectorial_quantities['v_nabla_B_y'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_advec_z'] = [(-(1/a) * vectorial_quantities['v_nabla_B_z'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['MIE_advec_x'] = zero
        results['MIE_advec_y'] = zero
        results['MIE_advec_z'] = zero

    if components.get('drag', False):
        ### The cosmic drag component.

        results['MIE_drag_x'] = [(-(1/2) * H * clus_Bx[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_drag_y'] = [(-(1/2) * H * clus_By[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        results['MIE_drag_z'] = [(-(1/2) * H * clus_Bz[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['MIE_drag_x'] = zero
        results['MIE_drag_y'] = zero
        results['MIE_drag_z'] = zero

    if components.get('total', False):
        ### The total magnetic induction energy in the compact way.
        
        if components.get('drag', False):
            results['MIE_total_x'] = [((1/a) * vectorial_quantities['curl_v_X_B_x'][p] + results['MIE_drag_x'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
            results['MIE_total_y'] = [((1/a) * vectorial_quantities['curl_v_X_B_y'][p] + results['MIE_drag_y'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
            results['MIE_total_z'] = [((1/a) * vectorial_quantities['curl_v_X_B_z'][p] + results['MIE_drag_z'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        else:
            results['MIE_total_x'] = [((1/a) * vectorial_quantities['curl_v_X_B_x'][p] + (-(1/2) * H * clus_Bx[p])) if bool(clus_kp[p]) else 0 for p in range(n)]
            results['MIE_total_y'] = [((1/a) * vectorial_quantities['curl_v_X_B_y'][p] + (-(1/2) * H * clus_By[p])) if bool(clus_kp[p]) else 0 for p in range(n)]
            results['MIE_total_z'] = [((1/a) * vectorial_quantities['curl_v_X_B_z'][p] + (-(1/2) * H * clus_Bz[p])) if bool(clus_kp[p]) else 0 for p in range(n)]

    else:
        results['MIE_total_x'] = zero
        results['MIE_total_y'] = zero
        results['MIE_total_z'] = zero

    # Compute magnitudes if requested
    if mag == True:
        magnitudes = {}
        for key, prefix in [
            ('divergence', 'MIE_diver'),
            ('compression', 'MIE_compres'),
            ('stretching', 'MIE_stretch'),
            ('advection', 'MIE_advec'),
            ('drag', 'MIE_drag'),
            ('total', 'MIE_total')
        ]:
            if components.get(key, False):
                magnitudes[f'{prefix}_mag'] = utils.magnitude(results[f'{prefix}_x'], results[f'{prefix}_y'], results[f'{prefix}_z'], clus_kp)
            else:
                magnitudes[f'{prefix}_mag'] = zero
    else:
        magnitudes = None

    end_time_induction_terms = time.time()

    total_time_induction_terms = end_time_induction_terms - start_time_induction_terms
    
    if verbose == True:
        log_message('Time for calculating the induction eq. terms in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction_terms))), tag="induction", level=1)

    return results, magnitudes


def induction_equation_energy(components, induction_equation,
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
        - results: dictionary containing the computed components of the magnetic induction equation in terms of the magnetic energy:
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
    
    for key, prefix in [
        ('divergence', 'MIE_diver'),
        ('compression', 'MIE_compres'),
        ('stretching', 'MIE_stretch'),
        ('advection', 'MIE_advec'),
        ('drag', 'MIE_drag'),
        ('total', 'MIE_total')
    ]:
        if components.get(key, False):
            results[f'{prefix}_B2'] = [(clus_Bx[p] * induction_equation[f'{prefix}_x'][p] + clus_By[p] * induction_equation[f'{prefix}_y'][p]
                                        + clus_Bz[p] * induction_equation[f'{prefix}_z'][p]) if bool(clus_kp[p]) else 0 for p in range(n)]
        else:
            results[f'{prefix}_B2'] = zero

    ## The kinetic energy.

    if clus_rho_rho_b:
        results['kinetic_energy_density'] = [((1/2) * clus_rho_rho_b[p] * clus_v2[p]) if bool(clus_kp[p]) else 0 for p in range(n)]
    else:
        results['kinetic_energy_density'] = zero
    
    end_time_induction_energy_terms = time.time()

    total_time_induction_energy_terms = end_time_induction_energy_terms - start_time_induction_energy_terms
    
    if verbose == True:
        log_message('Time for calculating the energy induction eq. terms in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction_energy_terms))), tag="induction_energy", level=1)
        
    return results
    

def induction_vol_integral(components, induction_energy, clus_b2,
                            clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_zeta, grid_npatch, up_to_level,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, region_coords, rad,
                            units =1, verbose=False):
    '''
    Computes the volume integral of the magnetic energy density and its components, as well as the induced magnetic energy.
    This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied volume. The kinetic energy
    density is also computed.
    Only computes the components that are set to True in "components" dictionary.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
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
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the computed volume integrals:
            - int_MIE_diver_B2: volume integral of the null divergence of the magnetic field energy
            - int_MIE_compres_B2: volume integral of the compressive component
            - int_MIE_stretch_B2: volume integral of the stretching component
            - int_MIE_advec_B2: volume integral of the advection component
            - int_MIE_drag_B2: volume integral of the cosmic drag component
            - int_MIE_total_B2: volume integral of the total magnetic induction energy
            - int_kinetic_energy: volume integral of the kinetic energy density
            - int_b2: volume integral of the magnetic energy density
            - volume: volume of the studied region
        
    Author: Marco Molina
    '''
    ## Here we compute the volume integral of the magnetic energy density and its components
    
    start_time_induction = time.time() # Record the start time
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    
    results = {}
    
    for key, prefix in [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('drag', 'MIE_drag_B2'),
        ('total', 'MIE_total_B2')
    ]:
        if components.get(key, False):
            results[f'int_{prefix}'] = utils.vol_integral(induction_energy[prefix], grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                            size, nmax, coords, region_coords, rad, a0_masclet, units, kept_patches=clus_kp)
    
            if verbose == True:
                log_message(f'Snap {it} in {sims}: {key} energy density volume integral done', tag="integral", level=1)
        else:
            results[f'int_{prefix}'] = zero
    
    if induction_energy['kinetic_energy_density']:
        results['int_kinetic_energy'] = utils.vol_integral(induction_energy['kinetic_energy_density'], grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                            size, nmax, coords, region_coords, rad, a0_masclet, units, kept_patches=clus_kp)
        if verbose == True:
            log_message(f'Snap {it} in {sims}: Kinetic energy density volume integral done', tag="integral", level=1)
    else:
        results['int_kinetic_energy'] = zero
        
    if clus_b2:
        results['int_b2'] = utils.vol_integral(clus_b2, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                                grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                size, nmax, coords, region_coords, rad, a0_masclet, units, kept_patches=clus_kp)
        if verbose == True:
            log_message(f'Snap {it} in {sims}: Magnetic energy density volume integral done', tag="integral", level=1)
    else:
        results['int_b2'] = zero
    
    results['volume'] = utils.vol_integral(induction_energy[prefix], grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                            size, nmax, coords, region_coords, rad, a0_masclet, units, kept_patches=clus_kp, vol=True)

    end_time_induction = time.time()

    total_time_induction = end_time_induction - start_time_induction

    if verbose == True:
        log_message('Time for induction integration in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction))), tag="integral", level=1)

    return results


def induction_energy_integral_evolution(components, induction_energy_integral,
                                        evolution_type, derivative, rho_b,
                                        grid_time, grid_zeta, verbose=False):
    '''
    Given the volume integrals of the magnetic energy density and its components at different redshifts,
    computes the evolution of the magnetic integrated energy and that of its components for their further representation
    attending to the time derivative prediction from the induction equation.
    
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
        - evolution_type: type of evolution to compute ('total' or 'differential')
        - derivative: type of derivative to compute ('RK' for Runge-Kutta, 'implicit' for implicit differences,
                    'central' for central differences, 'rate' for rate of change)
        - rho_b: density contrast of the simulation
        - grid_time: time grid for the simulation
        - grid_zeta: redshift grid for the simulation
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - results: dictionary containing the evolution of the magnetic energy density and its components:
            - evo_MIE_diver_B2: evolution of the null divergence of the magnetic field energy
            - evo_MIE_compres_B2: evolution of the compressive component
            - evo_MIE_stretch_B2: evolution of the stretching component
            - evo_MIE_advec_B2: evolution of the advection component
            - evo_MIE_drag_B2: evolution of the cosmic drag component
            - evo_MIE_total_B2: evolution of the total magnetic induction energy
            - evo_kinetic_energy: evolution of the kinetic energy density
            - evo_b2: evolution of the magnetic energy density
            - evo_ind_b2: evolution of the integrated magnetic energy from the induction equation
            - evo_volume: evolution of the volume of the studied region
            
    Author: Marco Molina
    '''
    
    assert evolution_type in ['total', 'differential'], "evolution_type must be 'total' or 'differential'"
    assert derivative in ['RK', 'implicit', 'central', 'rate'], "derivative must be 'RK', 'implicit', 'central' or 'rate'"
    
    ## Here we compute the evolution of the magnetic energy density and its components
    
    start_time_evolution = time.time() # Record the start time
    
    n = len(grid_time)-1
    zero = [0] * (n+1)
    
    results = {}
    
    main_keys = ["divergence", "compression", "stretching", "advection", "drag"]
    if all(components.get(k, False) for k in main_keys):
        components["induction"] = True
        induction_energy_integral['int_ind_b2'] = [(induction_energy_integral['int_MIE_diver_B2'][i] +
                                                    induction_energy_integral['int_MIE_compres_B2'][i] +
                                                    induction_energy_integral['int_MIE_stretch_B2'][i] +
                                                    induction_energy_integral['int_MIE_advec_B2'][i] +
                                                    induction_energy_integral['int_MIE_drag_B2'][i])
                                                    for i in range(n+1)]
    else:
        components["induction"] = False
    
    for key, prefix in [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('drag', 'MIE_drag_B2'),
        ('total', 'MIE_total_B2'),
        ('induction', 'ind_b2')
    ]:
        if evolution_type == 'total':
            if components.get(key, False):
                if derivative == 'RK':
                    results[f'evo_{prefix}'] = diff.integrate_energy(grid_time, induction_energy_integral[f'int_b2'][0],
                                                                rho_b, induction_energy_integral[f'int_{prefix}'])
                elif derivative == 'central':
                    results[f'evo_{prefix}'] = [(rho_b[i+1] * ((1/rho_b[i]) * (induction_energy_integral[f'int_b2'][i]) +
                    2 * (grid_time[i+1] - grid_time[i]) * (induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n)]
                elif derivative == 'implicit':
                    results[f'evo_{prefix}'] = [((rho_b[i+2]/rho_b[i+1]) * induction_energy_integral[f'int_b2'][i+1] +
                    2 * rho_b[i+2] * (grid_time[i+2] - grid_time[i+1]) * (induction_energy_integral[f'int_{prefix}'][i+1] +
                    ((grid_time[i+2] - grid_time[i+1])/(grid_time[i+2] - grid_time[i])) * (induction_energy_integral[f'int_{prefix}'][i+2] -
                    induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n-1)]
                elif derivative == 'rate':
                    results[f'evo_{prefix}'] = [(rho_b[i+1] * ((1/rho_b[i]) * (induction_energy_integral[f'int_b2'][i]) +
                    (grid_time[i+1] - grid_time[i]) * (induction_energy_integral[f'int_{prefix}'][i+1] + induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n)]
                if verbose == True:
                    log_message(f'Energy evolution: {key} volume energy integral evolution done', tag="evolution", level=1)
            else:
                results[f'evo_{prefix}'] = zero
            

        elif evolution_type == 'differential':
            if components.get(key, False):
                if derivative == 'RK':
                    results[f'evo_{prefix}'] = [2 * (induction_energy_integral[f'int_{prefix}'][i]) for i in range(n)]
                elif derivative == 'central':
                    results[f'evo_{prefix}'] = [2 * (induction_energy_integral[f'int_{prefix}'][i]) for i in range(n)]
                elif derivative == 'implicit':
                    results[f'evo_{prefix}'] = [2 * ((induction_energy_integral[f'int_{prefix}'][i+1] + ((grid_time[i+2] -
                    grid_time[i+1])/(grid_time[i+2] - grid_time[i])) * (induction_energy_integral[f'int_{prefix}'][i+2] -
                    induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n-1)]
                elif derivative == 'rate':
                    results[f'evo_{prefix}'] = [induction_energy_integral[f'int_{prefix}'][i+1] + induction_energy_integral[f'int_{prefix}'][i] for i in range(n)]
                if verbose == True:
                    log_message(f'Energy evolution: {key} energy integral evolution done', tag="evolution", level=1)
            else:
                results[f'evo_{prefix}'] = zero
    
    if evolution_type == 'total':
        results['evo_b2'] = [induction_energy_integral['int_b2'][i] for i in range(n+1)]
        results['evo_kinetic_energy'] = [rho_b[i] * induction_energy_integral['int_kinetic_energy'][i] for i in range(n+1)]
    elif evolution_type == 'differential':
        results['evo_b2'] = [(1/((grid_time[i+1] - grid_time[i]))) * (induction_energy_integral['int_b2'][i+1]/rho_b[i+1] - induction_energy_integral['int_b2'][i]/rho_b[i]) for i in range(n)]
        results['evo_kinetic_energy'] = [(1/(grid_time[i+1] - grid_time[i])) * ((rho_b[i+1] * induction_energy_integral['int_kinetic_energy'][i+1]) - (rho_b[i] * induction_energy_integral['int_kinetic_energy'][i])) for i in range(n)]
    
    if verbose == True:
        log_message('Energy evolution: magnetic and kinetic energy integral evolution done', tag="evolution", level=1)
            
    results['evo_volume_phi'] = [(induction_energy_integral['volume'][i]) for i in range(n+1)]
    results['evo_volume_co'] = [(induction_energy_integral['volume'][i] / ((1/(1+grid_zeta[i]))**3)) for i in range(n+1)]
    
    if verbose == True:
        log_message('Energy evolution: volume evolution done', tag="evolution", level=1)
    
    end_time_evolution = time.time()
    
    total_time_evolution = end_time_evolution - start_time_evolution
    
    if verbose == True:
        log_message('Time for evolution of the induction energy integral: '+str(strftime("%H:%M:%S", gmtime(total_time_evolution))), tag="evolution", level=1)

    return results

    
def induction_radial_profiles(components, induction_energy, clus_b2, clus_rho_rho_b, 
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
        - rho_b: desnity contrast of the simulation
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
        - results: dictionary containing the computed radial profiles:
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

    zero = 0.
    
    results = {}
    
    main_keys = ["divergence", "compression", "stretching", "advection", "drag"]
    if all(components.get(k, False) for k in main_keys):
        components["induction"] = True
        induction_energy['ind_b2'] = [induction_energy['MIE_compres_B2'][p] + induction_energy['MIE_diver_B2'][p] + induction_energy['MIE_stretch_B2'][p] + induction_energy['MIE_advec_B2'][p] + induction_energy['MIE_drag_B2'][p] for p in range(n)]
    else:
        components["induction"] = False
    
    for key, prefix in [
        ('divergence', 'MIE_diver_B2'),
        ('compression', 'MIE_compres_B2'),
        ('stretching', 'MIE_stretch_B2'),
        ('advection', 'MIE_advec_B2'),
        ('drag', 'MIE_drag_B2'),
        ('total', 'MIE_total_B2'),
        ('induction', 'ind_b2')
    ]:
        if components.get(key, False):
            _, profile = utils.radial_profile_vw(field=induction_energy[prefix], cr0amr=clus_cr0amr,
                                            solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
                                            clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                            nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                                            size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug)
            results[f'{prefix}_profile'] = rho_b * profile
            if verbose:
                log_message(f'Snap {it} in {sims}: {key} profile done', tag="profiles", level=1)
        else:
            results[f'{prefix}_profile'] = zero
    
    # if components.get('induction', False):
    #     results['post_ind_b2_profile'] = results['MIE_diver_B2_profile'] + results['MIE_compres_B2_profile'] + results['MIE_stretch_B2_profile'] + results['MIE_advec_B2_profile'] + results['MIE_drag_B2_profile']
    # else:
    #     results['post_ind_b2_profile'] = zero
    
    if induction_energy['kinetic_energy_density']:
        _, profile = utils.radial_profile_vw(field=induction_energy['kinetic_energy_density'], cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
                                                clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                                                size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug)
        results['kinetic_energy_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: Kinetic profile done', tag="profiles", level=1)
    else:
        results['kinetic_energy_profile'] = zero
        
    if clus_b2:
        _, profile = utils.radial_profile_vw(field=clus_b2, cr0amr=clus_cr0amr,
                                        solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
                                        clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                        nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                                        size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug)
        results['clus_b2_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: b2 profile done', tag="profiles", level=1)
    else:
        results['clus_b2_profile'] = zero
    
    if clus_rho_rho_b:
        profile_bin_centers, profile = utils.radial_profile_vw(field=clus_rho_rho_b, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
                                                clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                                                size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug)
        results['clus_rho_rho_b_profile'] = rho_b * profile
        if verbose:
            log_message(f'Snap {it} in {sims}: Density profile done', tag="profiles", level=1)
    else:
        results['clus_rho_rho_b_profile'] = zero
        profile_bin_centers, _ = utils.radial_profile_vw(field=clus_rho_rho_b, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, up_to_level=up_to_level,
                                                clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z,
                                                size=size, nmax=nmax, units=units, kept_patches=clus_kp, verbose=debug)
    
    results['profile_bin_centers'] = profile_bin_centers
    
    end_time_profile = time.time()

    total_time_profile = end_time_profile - start_time_profile
    
    if verbose == True:
        log_message('Time for profile calculation in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_profile))), tag="profiles", level=1)
        
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

        
def process_iteration(components, dir_grids, dir_gas, dir_params,
                    sims, it, coords, region_coords, rad, rmin, level, up_to_level,
                    nmax, size, H0, a0, test, units=1, nbins=25, logbins=True,
                    stencil=3, buffer=True, use_siblings=True, interpol='TSC', nghost=1, blend=False,
                    parent=False, parent_interpol=None,
                    bitformat=np.float32, mag=False,
                    energy_evolution=True, profiles=True, projection=True, percentiles=True, 
                    percentile_levels=(95, 90, 75, 50, 25), debug_params=None,
                    return_vectorial=False, return_induction=False, return_induction_energy=False,
                    verbose=False):
    '''
    Processes a single iteration of the cosmological magnetic induction equation calculations.
    
    Args:
        - components: list of components to be computed (set in the config file, accessed as a dictionary in IND_PARAMS["components"])
        - dir_grids: directory containing the grids
        - dir_gas: directory containing the gas data
        - dir_params: directory containing the parameters
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
        - energy_evolution: boolean to compute energy evolution (default is True)
        - profiles: boolean to compute radial profiles (default is True)
        - projection: boolean to compute uniform projection (default is True)
        - percentiles: boolean to compute percentile thresholds (default is True)
        - percentile_levels: tuple of percentile thresholds to compute (default is (100, 90, 75, 50, 25))
        - debug_params: dictionary with debug configuration (default is None, uses empty dict)
        - return_vectorial: boolean to return vectorial quantities dictionary (default is False)
        - return_induction: boolean to return induction equation dictionary (default is False)
        - return_induction_energy: boolean to return induction energy dictionary (default is False)
        - verbose: boolean to print progress information (default is False)
        
    Returns:
        - data: dictionary containing the loaded data from the simulation
        - vectorial: dictionary containing the computed vectorial quantities
        - induction: dictionary containing the components of the magnetic induction equation
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_integral: dictionary containing the volume integrals of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_profiles: dictionary containing the radial profiles of the magnetic induction equation in terms of the magnetic energy
        - induction_uniform: dictionary containing the uniform projection of the magnetic induction equation in terms of the magnetic energy
        - diver_B_percentiles: dictionary containing percentile thresholds of the magnetic field divergence
        - debug_fields: dictionary containing debug fields if requested

    Author: Marco Molina
    '''

    start_time_Total = time.time() # Record the start time

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
    
    data = load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, level, test=test, bitformat=bitformat, region=region_coords, verbose=verbose, debug=debug_params.get("divergence", {}) and debug_params.get("patch_analysis", {}))
    levels = utils.create_vector_levels(data['grid_npatch'])
    dx = size/nmax
    resolution = dx / (2 ** levels)
    
    # Run debug tests if enabled
    debug_fields = None
    pipeline_debug_results = None
    scan_pack = None
    if debug_params.get("buffer", {}).get("enabled", False):
        if verbose:
            log_message(f"\n{'*'*80}", tag="debug", level=1)
            log_message("BUFFER DEBUG MODE ENABLED - Running buffer pipeline validation tests...", tag="debug", level=1)
            log_message(f"{'*'*80}", tag="debug", level=1)
        pipeline_debug_results = debug_module.run_debug_buffer_pipeline(data, size, nmax, nghost=nghost, 
                                                interpol=interpol, use_siblings=use_siblings, 
                                                bitformat=bitformat, 
                                                verbose=debug_params.get("buffer", {}).get("verbose", True))
    
    if parent_interpol is None:
        parent_interpol = interpol

    parent_mode = bool(parent)

    blend_active = bool(blend) and bool(buffer)
    boundary_width = 1 if stencil == 3 else 2
    buffer_nghost = nghost
    if parent_mode and not blend_active:
        buffer_nghost = 0
    if blend_active and buffer_nghost == 0:
        buffer_nghost = boundary_width
        if verbose:
            log_message(
                f'Blend active: using buffer nghost={buffer_nghost} (stencil={stencil}) in addition to parent fill',
                tag="buffer",
                level=1
            )

    original_fields = None
    if blend_active:
        original_fields = {
            'Bx': data['clus_Bx'],
            'By': data['clus_By'],
            'Bz': data['clus_Bz'],
            'vx': data['clus_vx'],
            'vy': data['clus_vy'],
            'vz': data['clus_vz']
        }

    # Add ghost buffer cells before derivatives
    if buffer == True and buffer_nghost > 0:
        buffered_field = buff.add_ghost_buffer(
            [data['clus_Bx'], data['clus_By'], data['clus_Bz'], data['clus_vx'], data['clus_vy'], data['clus_vz']],
            data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
            size=size, nmax=nmax, nghost=buffer_nghost, interpol=interpol, use_siblings=use_siblings,
            kept_patches=data['clus_kp']
        )
        for i, key in enumerate(['Bx', 'By', 'Bz', 'vx', 'vy', 'vz']):
            data[f'clus_{key}'] = buffered_field[i]
        if verbose == True:
            log_message('Ghost buffer added to magnetic and velocity fields', tag="buffer", level=1)

    # Vectorial calculus
    ## Here we calculate the different vectorial calculus quantities of our interest using the diff module.
    vectorial = vectorial_quantities(components, data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                                data['clus_vx'], data['clus_vy'], data['clus_vz'],
                                data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                                dx, stencil=stencil, verbose=verbose)
            
    # Remove ghost buffer cells after derivatives (skip if parent mode uses frontier fill)
    if buffer == True and buffer_nghost > 0:
        if verbose == True:
            log_message('Removing ghost buffer from computed vectorial fields', tag="buffer", level=1)
        for key in vectorial.keys():
            vectorial[key] = buff.ghost_buffer_buster(
                vectorial[key], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                buffer_nghost, kept_patches=data['clus_kp']
            )
        if verbose == True:
            log_message('Removing ghost buffer from magnetic and velocity fields', tag="buffer", level=1)
        for key in ['Bx', 'By', 'Bz', 'vx', 'vy', 'vz']:
            data[f'clus_{key}'] = buff.ghost_buffer_buster(
                data[f'clus_{key}'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                buffer_nghost, kept_patches=data['clus_kp']
            )
    elif buffer == True and parent_mode and not blend_active:
        parent_use_siblings = False
        buffered_field = buff.add_ghost_buffer(
            [vectorial[key] for key in vectorial.keys()],
            data['grid_npatch'], data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
            data['grid_patchx'], data['grid_patchy'], data['grid_patchz'],
            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'], data['grid_pare'],
            size=size, nmax=nmax, nghost=0, interpol=parent_interpol, use_siblings=parent_use_siblings,
            kept_patches=data['clus_kp']
        )
        for i, key in enumerate(vectorial.keys()):
            vectorial[key] = buffered_field[i]
        if verbose == True:
            log_message(
                f'Parent frontier filling applied to vectorial fields (parent_interpol={parent_interpol})',
                tag="buffer",
                level=1
            )

    if blend_active:
        vectorial_no_buffer = vectorial_quantities(
            components,
            original_fields['Bx'], original_fields['By'], original_fields['Bz'],
            original_fields['vx'], original_fields['vy'], original_fields['vz'],
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

        if verbose == True:
            log_message(
                f'Blend applied: boundary cells are averaged between buffer and parent fill '
                f'(parent_interpol={parent_interpol})',
                tag="buffer",
                level=1
            )

    # Magnetic Induction Equation
    
    ## In this section we are going to compute the cosmological induction equation and its components, calculating them with the results obtained before.
    ## This will be usefull to plot fluyd maps as the quantities involved are vectors.
    
    # Determine if induction needs to be calculated based on downstream dependencies
    compute_induction = return_induction or energy_evolution or profiles or projection or mag or percentiles
    
    if compute_induction:
        induction, magnitudes = induction_equation(components, vectorial,
                            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                            data['clus_vx'], data['clus_vy'], data['clus_vz'],
                            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                            data['H'], data['a'], mag=mag, verbose=verbose)
    else:
        induction = None
        magnitudes = None
        if verbose:
            log_message('Induction equation skipped (not required by any enabled output).', tag="pipeline", level=1)
    
    # Magnetic Induction Equation in Terms of the Magnetic Energy
    
    ## In this section we are going to compute the cosmological induction equation in terms of the magnetic energy and its components, calculating them with the results obtained before.
    ## This will be usefull to calculate volumetric integrals and energy budgets as the quantities involved are scalars.
    
    # Determine if induction_energy needs to be calculated
    compute_induction_energy = return_induction_energy or energy_evolution or profiles
    
    if compute_induction_energy and induction is not None:
        induction_energy = induction_equation_energy(components, induction,
                                data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                                data['clus_rho_rho_b'], data['clus_v2'],
                                data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                                verbose=verbose)
    else:
        induction_energy = None
        if verbose and compute_induction_energy and induction is None:
            log_message('Induction energy skipped (induction not available).', tag="pipeline", level=1)
        elif verbose and not compute_induction_energy:
            log_message('Induction energy skipped (not required by any enabled output).', tag="pipeline", level=1)
    
    if energy_evolution and induction_energy is not None:
        # Volume Integral of the Magnetic Induction Equation
    
        ## Here we compute the volume integral of the magnetic energy density and its components, as well as the induced magnetic energy.
        ## This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied volume. The kinetic energy
        ## density is also computed.
        
        induction_energy_integral = induction_vol_integral(components, induction_energy, data['clus_b2'],
                                data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                                data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                                it, sims, nmax, size, coords, region_coords, rad,
                                units=1, verbose=verbose)
        
        if test['test'] == True:
            
            induction_test_energy = analytic_test_fields(data['grid_time'], data['grid_npatch'], data['a'], data['H'], data['clus_Bx'], test)
            
            induction_test_energy_integral = induction_vol_integral(components, induction_test_energy, data['clus_b2'],
                        data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                        data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                        data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                        data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                        it, sims, nmax, size, coords, region_coords, rad,
                        units=1, verbose=verbose)
        else:
            induction_test_energy_integral = None
        
    else:
        induction_energy_integral = None
        induction_test_energy_integral = None
        if verbose == True:
            if not energy_evolution:
                log_message('Energy evolution is set to False, skipping volume integral of the magnetic induction equation.', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('Energy evolution skipped (induction_energy not available).', tag="pipeline", level=1)
            
    if profiles and induction_energy is not None:
        # Radial Profiles of the Magnetic Induction Equation
    
        ## We can calculate the radial profiles of the magnetic energy density in the volume we have considered (usually the virial volume)
        
        induction_energy_profiles = induction_radial_profiles(components, induction_energy, data['clus_b2'],
                                    data['clus_rho_rho_b'], data['rho_b'],
                                    data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                                    data['grid_irr'], data['grid_npatch'], up_to_level,
                                    data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                                    data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                                    it, sims, nmax, size, coords, rmin, rad,
                                    nbins=nbins, logbins=logbins, units=1, verbose=verbose)
    else:
        induction_energy_profiles = None
        if verbose == True:
            if not profiles:
                log_message('Profiles are set to False, skipping radial profiles of the magnetic induction equation.', tag="pipeline", level=1)
            elif induction_energy is None:
                log_message('Radial profiles skipped (induction_energy not available).', tag="pipeline", level=1)
            
    if projection and induction is not None:
        # Uniform Projection of the Magnetic Induction Equation
    
        ## We clean and compute the uniform section of the magnetic induction energy and its components for the given AMR grid for its further projection.
        
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
            
    if percentiles:
        # Percentile Thresholds of the Magnetic Field Divergence
        
        # Scale divergence by resolution to make it comparable to field magnitude
        # Divergence has units [field/length], multiplying by dx gives [field]
        
        # Extract percentile calculation options from debug_params if available
        percentile_params = debug_params.get("percentile_params", {}) if debug_params else {}
        exclude_boundaries = percentile_params.get("exclude_boundaries", False)
        boundary_width = percentile_params.get("boundary_width", 1)
        exclude_zeros = percentile_params.get("exclude_zeros", True)
        
        diver_B_percentiles = compute_percentile_thresholds(
            field_numerator=vectorial['diver_B'],
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
        
    data = {
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

    return data, vectorial, induction, magnitudes, induction_energy, induction_energy_integral, induction_test_energy_integral, induction_energy_profiles, induction_uniform, diver_B_percentiles, debug_fields