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
import scripts.readers as reader
from scripts.units import *
from scripts.test import analytic_test_fields, numeric_test_fields
from scipy.special import gamma
# from scipy import fft
from matplotlib import pyplot as plt
import pdb
import multiprocessing as mp
np.set_printoptions(linewidth=200)


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

    # 

    coords = []
    rad = []

    for i in range(len(sims)):
        for j in reversed(range(len(it))):
            
            halos = reader.read_families(it[j], path=dir_halos, output_format='dictionaries', output_redshift=False,
                        min_mass=None, exclude_subhaloes=True, read_region=None, keep_boundary_contributions=False)
            
            _,_,_,_,zeta = reader.read_grids(it = it[j], path=dir_grids+sims[i], parameters_path=data_folder+'/'+sims[i]+'/', digits=5, read_general=True, read_patchnum=False, read_dmpartnum=False,
            read_patchcellextension=False, read_patchcellposition=False, read_patchposition=False, read_patchparent=False, nparray=False)
            
            if j == len(it) - 1:
                # Find the index of the most massive halo
                max_mass_index = np.argmax([halo['M'] for halo in halos])
                id_max_mass = halos[max_mass_index]['id']
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
                
            if verbose:
                
                # Print the coordinates
                print("Coordinates of the most massive halo in snap " + str(it[j]) + ":")
                print("x: " + str(coords[-1][0]))
                print("y: " + str(coords[-1][1]))
                print("z: " + str(coords[-1][2]))
                print("Radius: " + str(rad[-1]) + " Mpc")
                if index != None:
                    print("Mass: " + str(halos[max_mass_index]['M']*mass_to_sun) + " Msun")
                else:
                    print("No halo found in this snapshot, using the previous one.")
                
    coords = coords[::-1]
    rad = rad[::-1]
    
    return coords, rad


def create_region(sims, it, coords, rad, F=1.0, reg='BOX', verbose=False):
    '''
    Creates the boxes or spheres centered at the coordinates of the most massive halo or any other point in each snapshot.
    
    Args:
        - sims: list of simulation names
        - it: list of snapshots
        - coords: list of coordinates of the most massive halo in each snapshot
        - rad: list of radii of the most massive halo in each snapshot
        - F: factor to scale the radius (default is 1.0)
        - red: region type to create ('BOX' or 'SPH', default is 'BOX')
            - BOX: creates a box
            - SPH: creates a sphere
        - verbose: boolean to print the coordinates and radius or not
        
    Returns:
        - region: list of boxes or spheres centered at the coordinates
        - region_size: list of sizes of the boxes or spheres in Mpc
        
    Author: Marco Molina
    '''
    
    if reg == 'BOX':
        BOX = True
        SPH = False
    elif reg == 'SPH':
        BOX = False
        SPH = True
    else:
        BOX = False
        SPH = False
        
    Rad = []
    region_size = []
    Box = []
    Sph = []

    for i in range(len(sims)):
        for j in range(len(it)):

            Rad.append(F*rad[i+j])
            region_size.append(2 * Rad[-1])  # Size of the box in Mpc
            Box.append(["box", coords[i+j][0]-Rad[-1], coords[i+j][0]+Rad[-1], coords[i+j][1]-Rad[-1], coords[i+j][1]+Rad[-1], coords[i+j][2]-Rad[-1], coords[i+j][2]+Rad[-1]]) # Mpc
            # Box.append(["box",-size[i]/2,size[i]/2,-size[i]/2,size[i]/2,-size[i]/2,size[i]/2]) # Mpc
            Sph.append(["sphere", coords[i+j][0], coords[i+j][1], coords[i+j][2], Rad[-1]]) # Mpc
            
            if verbose:
                        
                # Print the coordinates
                print("Box: " + str(Box[-1]))
                print("Sphere: " + str(Sph[-1]))
    
    if BOX == True:
        region = Box
    else:
        region = Sph

    if BOX == False and SPH == False:
        region = [None for _ in range(len(sims)*len(it))]
        
    return region, region_size


def load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, level, test, A2U=False, region=None, verbose=False):
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
        - A2U: boolean to transform the AMR grid to a uniform grid (default is False)
        - region: region to be used (default is None)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        
    Author: Marco Molina
    '''
    # Load Simulation Data
    
    ## This are the parameters we will need for each cell together with the magnetic field and the velocity
    ## We read the information for each snap and divide it in the different fields

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
            read_patchcellposition=False,
            read_patchposition=True,
            read_patchparent=False,
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
            grid_patchrx,
            grid_patchry,
            grid_patchrz,
            *_
        ) = grid
        
        # Only keep patches up to the desired AMR level
        grid_npatch[level+1:] = 0
        
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
            clus_bx,
            clus_by,
            clus_bz,
            *rest
        ) = clus
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
            clus_bx,
            clus_by,
            clus_bz,
            clus_vx,
            clus_vy,
            clus_vz
        ) = clus_bv
        
    # Determine mask for valid patches
    if region is not None and rest:
        clus_kp = rest[0]
    else:
        clus_kp = np.ones((len(clus_bx),), dtype=bool)

    # Normalize magnetic field components
    clus_Bx = [clus_bx[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    clus_By = [clus_by[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    clus_Bz = [clus_bz[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    
    if verbose == True:
        print('Data type loaded for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype))

    # Convert to float64 if not transforming to uniform grid
    if A2U == False: 
        clus_rho_rho_b = [(1+clus_rho_rho_b[p]).astype(np.float64) if clus_kp[p] != 0 else (1+clus_rho_rho_b[p]) for p in range(1+np.sum(grid_npatch))] # Delta is (rho/rho_b) - 1
        clus_vx = [clus_vx[p].astype(np.float64) if clus_kp[p] != 0 else clus_vx[p] for p in range(1+np.sum(grid_npatch))]
        clus_vy = [clus_vy[p].astype(np.float64) if clus_kp[p] != 0 else clus_vy[p] for p in range(1+np.sum(grid_npatch))]
        clus_vz = [clus_vz[p].astype(np.float64) if clus_kp[p] != 0 else clus_vz[p] for p in range(1+np.sum(grid_npatch))]
        clus_bx = [clus_bx[p].astype(np.float64) if clus_kp[p] != 0 else clus_bx[p] for p in range(1+np.sum(grid_npatch))]
        clus_by = [clus_by[p].astype(np.float64) if clus_kp[p] != 0 else clus_by[p] for p in range(1+np.sum(grid_npatch))]
        clus_bz = [clus_bz[p].astype(np.float64) if clus_kp[p] != 0 else clus_bz[p] for p in range(1+np.sum(grid_npatch))]
        clus_Bx = [clus_Bx[p].astype(np.float64) if clus_kp[p] != 0 else clus_Bx[p] for p in range(1+np.sum(grid_npatch))]
        clus_By = [clus_By[p].astype(np.float64) if clus_kp[p] != 0 else clus_By[p] for p in range(1+np.sum(grid_npatch))]
        clus_Bz = [clus_Bz[p].astype(np.float64) if clus_kp[p] != 0 else clus_Bz[p] for p in range(1+np.sum(grid_npatch))]
    
    clus_b2 = [clus_bx[p]**2 + clus_by[p]**2 + clus_bz[p]**2 for p in range(1+np.sum(grid_npatch))]
    clus_B2 = [clus_Bx[p]**2 + clus_By[p]**2 + clus_Bz[p]**2 for p in range(1+np.sum(grid_npatch))]
    clus_v2 = [clus_vx[p]**2 + clus_vy[p]**2 + clus_vz[p]**2 for p in range(1+np.sum(grid_npatch))]
    
    if verbose == True:
        print('Working data type for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype))
        
    results = {
        'grid_irr': grid_irr,
        'grid_time': grid_time,
        'grid_zeta': grid_zeta,
        'grid_npatch': grid_npatch,
        'grid_patchnx': grid_patchnx,
        'grid_patchny': grid_patchny,
        'grid_patchnz': grid_patchnz,
        'grid_patchrx': grid_patchrx,
        'grid_patchry': grid_patchry,
        'grid_patchrz': grid_patchrz,
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
                        dx, stencil=5, verbose=False):
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
        
        v_X_B_x = [clus_vy[p] * clus_Bz[p] - clus_vz[p] * clus_By[p] if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))] # We run across all the patches with the levels we are interested in
        v_X_B_y = [clus_vz[p] * clus_Bx[p] - clus_vx[p] * clus_Bz[p] if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))] # We only want the patches inside the region of interest
        v_X_B_z = [clus_vx[p] * clus_By[p] - clus_vy[p] * clus_Bx[p] if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
            
        ### The total induction as the curl of the cross product of the velocity and magnetic field with drag term.
        
        results['curl_v_X_B_x'], results['curl_v_X_B_y'], results['curl_v_X_B_z'] = diff.curl(v_X_B_x, v_X_B_y, v_X_B_z, dx, grid_npatch, clus_kp, stencil)
    else:
        results['curl_v_X_B_x'] = zero
        results['curl_v_X_B_y'] = zero
        results['curl_v_X_B_z'] = zero

    end_time_vector = time.time()

    total_time_vector = end_time_vector - start_time_vector
    
    if verbose == True:
        print('Time for vector calculations in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_vector))))
        
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
    
    results = {}
    
    if components.get('divergence', False):
        ### The null divergence of the magnetic field for numerical error purposes.
        
        results['MIE_diver_x'] = [((1/a) * clus_vx[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)] # We have to run across all the patches.
        results['MIE_diver_y'] = [((1/a) * clus_vy[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_diver_z'] = [((1/a) * clus_vz[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_diver_x'] = zero
        results['MIE_diver_y'] = zero
        results['MIE_diver_z'] = zero

    if components.get('compression', False):
        ### The compressive component.
        
        results['MIE_compres_x'] = [(-(1/a) * clus_Bx[p] * vectorial_quantities['diver_v'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_compres_y'] = [(-(1/a) * clus_By[p] * vectorial_quantities['diver_v'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_compres_z'] = [(-(1/a) * clus_Bz[p] * vectorial_quantities['diver_v'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_compres_x'] = zero
        results['MIE_compres_y'] = zero
        results['MIE_compres_z'] = zero

    if components.get('stretching', False):
        ### The stretching component.
        
        results['MIE_stretch_x'] = [((1/a) * vectorial_quantities['B_nabla_v_x'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_stretch_y'] = [((1/a) * vectorial_quantities['B_nabla_v_y'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_stretch_z'] = [((1/a) * vectorial_quantities['B_nabla_v_z'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_stretch_x'] = zero
        results['MIE_stretch_y'] = zero
        results['MIE_stretch_z'] = zero

    if components.get('advection', False):
        ### The advection component.
        
        results['MIE_advec_x'] = [(-(1/a) * vectorial_quantities['v_nabla_B_x'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_advec_y'] = [(-(1/a) * vectorial_quantities['v_nabla_B_y'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_advec_z'] = [(-(1/a) * vectorial_quantities['v_nabla_B_z'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_advec_x'] = zero
        results['MIE_advec_y'] = zero
        results['MIE_advec_z'] = zero

    if components.get('drag', False):
        ### The cosmic drag component.
        
        results['MIE_drag_x'] = [(-(1/2) * H * clus_Bx[p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_drag_y'] = [(-(1/2) * H * clus_By[p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_drag_z'] = [(-(1/2) * H * clus_Bz[p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_drag_x'] = zero
        results['MIE_drag_y'] = zero
        results['MIE_drag_z'] = zero

    if components.get('total', False):
        ### The total magnetic induction energy in the compact way.
        
        if components.get('drag', False):
            results['MIE_total_x'] = [((1/a) * vectorial_quantities['curl_v_X_B_x'][p] + results['MIE_drag_x'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
            results['MIE_total_y'] = [((1/a) * vectorial_quantities['curl_v_X_B_y'][p] + results['MIE_drag_y'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
            results['MIE_total_z'] = [((1/a) * vectorial_quantities['curl_v_X_B_z'][p] + results['MIE_drag_z'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        else:
            results['MIE_total_x'] = [((1/a) * vectorial_quantities['curl_v_X_B_x'][p] + (-(1/2) * H * clus_Bx[p])) if clus_kp[p] != 0 else 0 for p in range(n)]
            results['MIE_total_y'] = [((1/a) * vectorial_quantities['curl_v_X_B_y'][p] + (-(1/2) * H * clus_By[p])) if clus_kp[p] != 0 else 0 for p in range(n)]
            results['MIE_total_z'] = [((1/a) * vectorial_quantities['curl_v_X_B_z'][p] + (-(1/2) * H * clus_Bz[p])) if clus_kp[p] != 0 else 0 for p in range(n)]

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
        print('Time for calculating the induction eq. terms in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction_terms))))

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
                                        + clus_Bz[p] * induction_equation[f'{prefix}_z'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        else:
            results[f'{prefix}_B2'] = zero

    ## The kinetic energy.

    if clus_rho_rho_b:
        results['kinetic_energy_density'] = [((1/2) * clus_rho_rho_b[p] * clus_v2[p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['kinetic_energy_density'] = zero
    
    end_time_induction_energy_terms = time.time()

    total_time_induction_energy_terms = end_time_induction_energy_terms - start_time_induction_energy_terms
    
    if verbose == True:
        print('Time for calculating the energy induction eq. terms in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction_energy_terms))))
        
    return results
    

def induction_vol_integral(components, induction_energy, clus_b2,
                            clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_zeta, grid_npatch, up_to_level,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, rad,
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
        - coords: coordinates of the grid
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
                                                            size, nmax, coords, rad, a0_masclet, units, kept_patches=clus_kp)
    
            if verbose == True:
                print(f'Snap {it} in {sims}: {key} energy density volume integral done')
        else:
            results[f'int_{prefix}'] = zero
    
    if induction_energy['kinetic_energy_density']:
        results['int_kinetic_energy'] = utils.vol_integral(induction_energy['kinetic_energy_density'], grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                            size, nmax, coords, rad, a0_masclet, units, kept_patches=clus_kp)
        if verbose == True:
            print(f'Snap {it} in {sims}: Kinetic energy density volume integral done')
    else:
        results['int_kinetic_energy'] = zero
        
    if clus_b2:
        results['int_b2'] = utils.vol_integral(clus_b2, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                                grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                size, nmax, coords, rad, a0_masclet, units, kept_patches=clus_kp)
        if verbose == True:
            print(f'Snap {it} in {sims}: Magnetic energy density volume integral done')
    else:
        results['int_b2'] = zero
    
    results['volume'] = utils.vol_integral(induction_energy[prefix], grid_zeta, clus_cr0amr, clus_solapst, grid_npatch, up_to_level,
                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                            size, nmax, coords, rad, a0_masclet, units, kept_patches=clus_kp, vol=True)

    end_time_induction = time.time()

    total_time_induction = end_time_induction - start_time_induction

    if verbose == True:
        print('Time for induction integration in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction))))

    return results


def induction_energy_integral_evolution(components, induction_energy_integral,
                                        evolution_type, derivative, rho_b,
                                        grid_time, grid_zeta, verbose=False):
    '''
    Given the volume integrals of the magnetic energy density and its components at different redshifts,
    computes the evolution of the magnetic integrated energy and that of its components for their further representation.
    
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
        - derivative: type of derivative to compute ('implicit' or 'central')
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
    assert derivative in ['implicit', 'central'], "derivative must be 'implicit' or 'central'"
    
    ## Here we compute the evolution of the magnetic energy density and its components
    
    start_time_evolution = time.time() # Record the start time
    
    n = len(grid_time)-1
    zero = [0] * n
    
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
                if derivative == 'central':
                    results[f'evo_{prefix}'] = [(rho_b[i+1] * ((1/rho_b[i]) * (induction_energy_integral[f'int_b2'][i]) +
                    2 * (grid_time[i+1] - grid_time[i]) * (induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n)]
                elif derivative == 'implicit':
                    results[f'evo_{prefix}'] = [((rho_b[i+2]/rho_b[i+1]) * induction_energy_integral[f'int_b2'][i+1] +
                    2 * rho_b[i+2] * (grid_time[i+2] - grid_time[i+1]) * (induction_energy_integral[f'int_{prefix}'][i+1] +
                    ((grid_time[i+2] - grid_time[i+1])/(grid_time[i+2] - grid_time[i])) * (induction_energy_integral[f'int_{prefix}'][i+2] -
                    induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n-1)]
                if verbose == True:
                    print(f'Energy evolution: {key} volume energy integral evolution done')
            else:
                results[f'evo_{prefix}'] = zero
            

        elif evolution_type == 'differential':
            if components.get(key, False):
                if derivative == 'central':
                    results[f'evo_{prefix}'] = [2 * (induction_energy_integral[f'int_{prefix}'][i]) for i in range(n)]
                elif derivative == 'implicit':
                    results[f'evo_{prefix}'] = [2 * ((induction_energy_integral[f'int_{prefix}'][i+1] + ((grid_time[i+2] -
                    grid_time[i+1])/(grid_time[i+2] - grid_time[i])) * (induction_energy_integral[f'int_{prefix}'][i+2] -
                    induction_energy_integral[f'int_{prefix}'][i]))) for i in range(n-1)]
                if verbose == True:
                    print(f'Energy evolution: {key} energy integral evolution done')
            else:
                results[f'evo_{prefix}'] = zero
    
    if evolution_type == 'total':
        results['evo_b2'] = [induction_energy_integral['int_b2'][i] for i in range(n+1)]
        results['evo_kinetic_energy'] = [rho_b[i] * induction_energy_integral['int_kinetic_energy'][i] for i in range(n+1)]
    elif evolution_type == 'differential':
        results['evo_b2'] = [(1/((grid_time[i+1] - grid_time[i]))) * (induction_energy_integral['int_b2'][i+1]/rho_b[i+1] - induction_energy_integral['int_b2'][i]/rho_b[i]) for i in range(n)]
        results['evo_kinetic_energy'] = [(1/(grid_time[i+1] - grid_time[i])) * ((rho_b[i+1] * induction_energy_integral['int_kinetic_energy'][i+1]) - (rho_b[i] * induction_energy_integral['int_kinetic_energy'][i])) for i in range(n)]
    
    if verbose == True:
        print(f'Energy evolution: magnetic and kinetic energy integral evolution done')
            
    results['evo_volume_phi'] = [(induction_energy_integral['volume'][i]) for i in range(n+1)]
    results['evo_volume_co'] = [(induction_energy_integral['volume'][i] / ((1/(1+grid_zeta[i]))**3)) for i in range(n+1)]
    
    if verbose == True:
        print('Energy evolution: volume evolution done')
    
    end_time_evolution = time.time()
    
    total_time_evolution = end_time_evolution - start_time_evolution
    
    if verbose == True:
        print('Time for evolution of the induction energy integral: '+str(strftime("%H:%M:%S", gmtime(total_time_evolution))))

    return results

    
def induction_radial_profiles(components, induction_energy, clus_b2, clus_rho_rho_b, 
                            clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_npatch,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, rmin, rad, 
                            nbins=25, logbins=True, level=100, verbose=False):
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
        - clus_cr0amr: AMR grid data
        - clus_solapst: overlap data
        - clus_kp: mask for valid patches
        - grid_irr: index of the snapshot
        - grid_npatch: number of patches in the grid
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
        - level: level of refinement in the AMR grid (default is 0)
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
            clean_field = utils.clean_field(induction_energy[prefix], clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
            _, results[f'{prefix}_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                            nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                            solapst=clus_solapst, npatch=grid_npatch, size=size, nmax=nmax, up_to_level=level)
            if verbose:
                print(f'Snap {it} in {sims}: {key} profile done')
        else:
            results[f'{prefix}_profile'] = zero
    
    if induction_energy['kinetic_energy_density']:
        clean_field = utils.clean_field(induction_energy['kinetic_energy_density'], clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        _, results['kinetic_energy_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size, nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: Kinetic profile done')
    else:
        results['kinetic_energy_profile'] = zero
        
    if clus_b2:
        clean_field = utils.clean_field(clus_b2, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        _, results['clus_B2_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size, nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: B2 profile done')
    else:
        results['clus_B2_profile'] = zero
    
    if clus_rho_rho_b:
        clean_field = utils.clean_field(clus_rho_rho_b, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        profile_bin_centers, results['clus_rho_rho_b_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size, nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: Density profile done')
    else:
        results['clus_rho_rho_b_profile'] = zero
        profile_bin_centers, _ = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size, nmax=nmax, up_to_level=level)
    
    results['profile_bin_centers'] = profile_bin_centers
    
    end_time_profile = time.time()

    total_time_profile = end_time_profile - start_time_profile
    
    if verbose == True:
        print('Time for profile calculation in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_profile))))
        
    return results


def uniform_induction(components, induction_equation,
                    clus_cr0amr, clus_solapst, grid_npatch,
                    grid_patchnx, grid_patchny, grid_patchnz, 
                    grid_patchrx, grid_patchry, grid_patchrz,
                    it, sims, nmax, size, Box,
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
        - Box: box coordinates
        - up_to_level: level of refinement in the AMR grid (default is 4)
        - ncores: number of cores to use for the computation (default is 1)
        - clus_kp: mask for valid patches
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - uniform_field: cleaned and projected field on a uniform grid
        
    Author: Marco Molina
    '''
    
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
                                                field=induction_equation[f'{prefix}_x'], box_limits=Box[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            results[f'uniform_{prefix}_y'] = utils.unigrid(
                                                field=induction_equation[f'{prefix}_y'], box_limits=Box[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            results[f'uniform_{prefix}_z'] = utils.unigrid(
                                                field=induction_equation[f'{prefix}_z'], box_limits=Box[1:], up_to_level=up_to_level,
                                                npatch=grid_npatch, patchnx=grid_patchnx, patchny=grid_patchny,
                                                patchnz=grid_patchnz, patchrx=grid_patchrx, patchry=grid_patchry,
                                                patchrz=grid_patchrz, size=size, nmax=nmax,
                                                interpolate=True, verbose=False, kept_patches=clus_kp, return_coords=False
                                            )
            if verbose == True:
                print(f'Snap {it} in {sims}: {key} uniform field done')
                print(results[f'uniform_{prefix}_x'].shape)
                print(results[f'uniform_{prefix}_y'].shape)
                print(results[f'uniform_{prefix}_z'].shape)
        else:
            results[f'uniform_{prefix}_x'] = zero
            results[f'uniform_{prefix}_y'] = zero
            results[f'uniform_{prefix}_z'] = zero
            
    end_time_uniform = time.time()
    
    total_time_uniform = end_time_uniform - start_time_uniform
    
    if verbose == True:
        print('Time for uniform field calculation in snap '+ str(grid_npatch) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_uniform))))
        
    return results


def process_iteration(components, dir_grids, dir_gas, dir_params,
                    sims, it, coords, Box, rad, rmin, level, up_to_level,
                    nmax, size, H0, a0, test, units =1, nbins=25, logbins=True,
                    stencil=3, A2U=False, mag=False,
                    energy_evolution=True, profiles=True, projection=True,
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
        - coords: coordinates of the grid
        - Box: box coordinates
        - rad: radii of the most massive halo in the snapshot
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
        - A2U: boolean to convert from A to U units (default is False)
        - mag: boolean to compute magnitudes (default is False)
        - energy_evolution: boolean to compute energy evolution (default is True)
        - profiles: boolean to compute radial profiles (default is True)
        - projection: boolean to compute uniform projection (default is True)
        - verbose: boolean to print progress information (default is False)
        
    Returns:
        - data: dictionary containing the loaded data from the simulation
        - vectorial: dictionary containing the computed vectorial quantities
        - induction: dictionary containing the components of the magnetic induction equation
        - induction_energy: dictionary containing the components of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_integral: dictionary containing the volume integrals of the magnetic induction equation in terms of the magnetic energy
        - induction_energy_profiles: dictionary containing the radial profiles of the magnetic induction equation in terms of the magnetic energy
        - induction_uniform: dictionary containing the uniform projection of the magnetic induction equation in terms of the magnetic energy

    Author: Marco Molina
    '''

    start_time_Total = time.time() # Record the start time
    
    if verbose == True:
        print(f'********************************************************************************')
        print(f"Processing iteration {it} in simulation {sims}")
        print(f'********************************************************************************')

    # Load Simulation Data
    
    ## This are the parameters we will need for each cell together with the magnetic field and the velocity
    ## We read the information for each snap and divide it in the different fields
    
    data = load_data(sims, it, a0, H0, dir_grids, dir_gas, dir_params, level, test=test, A2U=A2U, region=Box, verbose=verbose)
    
    # Vectorial calculus

    ## Here we calculate the different vectorial calculus quantities of our interest using the diff module.
    
    dx = size/nmax
    
    vectorial = vectorial_quantities(components, data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                                data['clus_vx'], data['clus_vy'], data['clus_vz'],
                                data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                                dx, stencil=stencil, verbose=verbose)
    
    # Magnetic Induction Equation
    
    ## In this section we are going to compute the cosmological induction equation and its components, calculating them with the results obtained before.
    ## This will be usefull to plot fluyd maps as the quantities involved are vectors.
    
    induction, magnitudes = induction_equation(components, vectorial,
                        data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                        data['clus_vx'], data['clus_vy'], data['clus_vz'],
                        data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                        data['H'], data['a'], mag=mag, verbose=verbose)
    
    # Magnetic Induction Equation in Terms of the Magnetic Energy
    
    ## In this section we are going to compute the cosmological induction equation in terms of the magnetic energy and its components, calculating them with the results obtained before.
    ## This will be usefull to calculate volumetric integrals and energy budgets as the quantities involved are scalars.
    
    induction_energy = induction_equation_energy(components, induction,
                            data['clus_Bx'], data['clus_By'], data['clus_Bz'],
                            data['clus_rho_rho_b'], data['clus_v2'],
                            data['clus_kp'], data['grid_npatch'], data['grid_irr'],
                            verbose=verbose)
    
    if energy_evolution:
        # Volume Integral of the Magnetic Induction Equation
    
        ## Here we compute the volume integral of the magnetic energy density and its components, as well as the induced magnetic energy.
        ## This is done according to the derived equation and compared to the actual magnetic energy integrated along the studied volume. The kinetic energy
        ## density is also computed.
        
        induction_energy_integral = induction_vol_integral(components, induction_energy, data['clus_b2'],
                                data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                                data['grid_irr'], data['grid_zeta'], data['grid_npatch'], up_to_level,
                                data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                                data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                                it, sims, nmax, size, coords, rad,
                                units=1, verbose=verbose)
        
    elif not energy_evolution:
        induction_energy_integral = None
        if verbose == True:
            print('Energy evolution is set to False, skipping volume integral of the magnetic induction equation.')
            
    if profiles:
        # Radial Profiles of the Magnetic Induction Equation
    
        ## We can calculate the radial profiles of the magnetic energy density in the volume we have considered (usually the virial volume)
        
        induction_energy_profiles = induction_radial_profiles(components, induction_energy, data['clus_b2'], data['clus_rho_rho_b'],
                                    data['clus_cr0amr'], data['clus_solapst'], data['clus_kp'],
                                    data['grid_irr'], data['grid_npatch'],
                                    data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                                    data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                                    it, sims, nmax, size, coords, rmin, rad,
                                    nbins=nbins, logbins=logbins, level=level, verbose=verbose)
    elif not profiles:
        induction_energy_profiles = None
        if verbose == True:
            print('Profiles are set to False, skipping radial profiles of the magnetic induction equation.')
            
    if projection:
        # Uniform Projection of the Magnetic Induction Equation
    
        ## We clean and compute the uniform section of the magnetic induction energy and its components for the given AMR grid for its further projection.
        
        induction_uniform = uniform_induction(components, induction,
                            data['clus_cr0amr'], data['clus_solapst'], data['grid_npatch'],
                            data['grid_patchnx'], data['grid_patchny'], data['grid_patchnz'],
                            data['grid_patchrx'], data['grid_patchry'], data['grid_patchrz'],
                            it, sims, nmax, size, Box,
                            up_to_level=up_to_level, ncores=1, clus_kp=data['clus_kp'],
                            verbose=verbose)
    elif not projection:
        induction_uniform = None
        if verbose == True:
            print('Projection is set to False, skipping uniform projection of the magnetic induction equation.')
    
    data = {
        'grid_time': data['grid_time'],
        'grid_zeta': data['grid_zeta'],
        'rho_b': data['rho_b']
    }
            
    end_time_Total = time.time()
    
    total_time_Total = end_time_Total - start_time_Total
    
    if verbose == True:
        print(f'Time for processing iteration {it} in simulation {sims}: {strftime("%H:%M:%S", gmtime(total_time_Total))}')

    return data, vectorial, induction, magnitudes, induction_energy, induction_energy_integral, induction_energy_profiles, induction_uniform