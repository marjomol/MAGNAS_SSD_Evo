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
import amr2uniform as a2u
from scripts.units import *
from scipy.special import gamma
from scipy import fft
from matplotlib import pyplot as plt
import pdb
import multiprocessing as mp
np.set_printoptions(linewidth=200)


def find_most_massive_halo(sims, it, a0, dir_halos, dir_grids, rawdir, vir_kind=1, rad_kind=1, verbose=False):
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
        - rawdir: directory where the raw data is stored
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
            
            _,_,_,_,zeta = reader.read_grids(it = it[j], path=dir_grids+sims[i], parameters_path=rawdir+sims[i]+'/', digits=5, read_general=True, read_patchnum=False, read_dmpartnum=False,
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
                print("Radius: " + str(rad[-1]))
                
    coords = coords[::-1]
    rad = rad[::-1]
    
    return coords, rad


def create_region(sims, it, coords, rad, F=1.0, BOX=False, SPH=False, verbose=False):
    '''
    Creates the boxes or spheres centered at the coordinates of the most massive halo in each snapshot.
    
    Args:
        - sims: list of simulation names
        - it: list of snapshots
        - coords: list of coordinates of the most massive halo in each snapshot
        - rad: list of radii of the most massive halo in each snapshot
        - F: factor to scale the radius (default is 1.0)
        - BOX: boolean to create boxes or not (default is False)
        - SPH: boolean to create spheres or not (default is False)
        - verbose: boolean to print the coordinates and radius or not
        
    Returns:
        - region: list of boxes or spheres centered at the coordinates
        
    Author: Marco Molina
    '''

    Rad = []
    Box = []
    Sph = []

    for i in range(len(sims)):
        for j in range(len(it)):

            Rad.append(F*rad[i+j])
            Box.append(["box", coords[i+j][0]-Rad[-1], coords[i+j][0]+Rad[-1], coords[i+j][1]-Rad[-1], coords[i+j][1]+Rad[-1], coords[i+j][2]-Rad[-1], coords[i+j][2]+Rad[-1]]) # Mpc
            # Box.append(["box",-size[i]/2,size[i]/2,-size[i]/2,size[i]/2,-size[i]/2,size[i]/2]) # Mpc
            Sph.append(["sphere", coords[i+j][0], coords[i+j][1], coords[i+j][2], Rad[-1]]) # Mpc
            
            if verbose:
                        
                # Print the coordinates
                print("Coordinates of the most massive halo in snap " + str(it[j]) + ":")
                print("x: " + str(coords[i+j][0]))
                print("y: " + str(coords[i+j][1]))
                print("z: " + str(coords[i+j][2]))
                print("Radius: " + str(Rad[-1]))
                print("Box: " + str(Box[-1]))
                print("Sphere: " + str(Sph[-1]))
                
    if BOX == True:
        region = Box
    else:
        region = Sph

    if BOX == False and SPH == False:
        region = [None for _ in range(len(sims)*len(it))]
        
    return region


def load_data(sims, it, rho_b, dir_grids, dir_params, dir_gas, level=3, A2U=True, region=None, verbose=False):
    '''
    Loads the data from the simulations for the given snapshots and prepares it for further analysis.
    This are the parameters we will need for each cell together with the magnetic field and the velocity,
    we read the information for each snap and divide it in the different fields.
    
    Args:
        - sims: list of simulation names
        - it: list of snapshots
        - rho_b: background density of the universe
        - dir_grids: directory where the grids are stored
        - rawdir: directory where the raw data is stored
        - dir_gas: directory where the gas data is stored
        - level: level of the AMR grid to be used (default is 3)
        - A2U: boolean to transform the AMR grid to a uniform grid (default is True)
        - region: region to be used (default is None)
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        
    Author: Marco Molina
    '''

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
        grid_t,
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

    # Create vector_levels using the tools module
    vector_levels = utils.create_vector_levels(grid_npatch)
    
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

    # Determine mask for valid patches
    if region is not None and rest:
        clus_kp = rest[0]
    else:
        clus_kp = np.ones((len(clus_bx[-1]),), dtype=bool)

    # Normalize magnetic field components
    clus_Bx = [clus_bx[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    clus_By = [clus_by[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    clus_Bz = [clus_bz[p] / np.sqrt(rho_b) if clus_kp[p] != 0 else 0 for p in range(1 + np.sum(grid_npatch))]
    
    if verbose == True:
        print('Data type loaded for snap '+ str(grid_irr) + ': ' + str(clus_vx[0].dtype))

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
    
    return (
        grid_irr,
        grid_t,
        grid_zeta,
        grid_npatch,
        grid_patchnx,
        grid_patchny,
        grid_patchnz,
        grid_patchrx,
        grid_patchry,
        grid_patchrz,
        vector_levels,
        clus_rho_rho_b,
        clus_vx,
        clus_vy,
        clus_vz,
        clus_cr0amr,
        clus_solapst,
        clus_kp,
        clus_Bx,
        clus_By,
        clus_Bz,
        clus_b2,
        clus_B2,
        clus_v2
    )


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
        
        results['diver_B'] = diff.divergence(clus_Bx, clus_By, clus_Bz, dx, grid_npatch, stencil, clus_kp)
    else:
        results['diver_B'] = zero
        
    if components.get('compression', False):
        ### We compute the divergence of the velocity field
        
        results['diver_v'] = diff.divergence(clus_vx, clus_vy, clus_vz, dx, grid_npatch, stencil, clus_kp)
    else:
        results['diver_v'] = zero
        
    if components.get('stretching', False):
        ### We compute the directional derivative of the velocity field along the magnetic field
        
        results['B_nabla_v_x'], results['B_nabla_v_y'], results['B_nabla_v_z'] = diff.directional_derivative_vector_field(clus_vx, clus_vy, clus_vz, clus_Bx, clus_By, clus_Bz, dx, grid_npatch, stencil, clus_kp)
    else:
        results['B_nabla_v_x'] = zero
        results['B_nabla_v_y'] = zero
        results['B_nabla_v_z'] = zero
        
    if components.get('advection', False):
        ### We compute the directional derivative of the magnetic field along the velocity field
    
        results['v_nabla_B_x'], results['v_nabla_B_y'], results['v_nabla_B_z'] = diff.directional_derivative_vector_field(clus_Bx, clus_By, clus_Bz, clus_vx, clus_vy, clus_vz, dx, grid_npatch, stencil, clus_kp)
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
        
        results['curl_v_X_B_x'], results['curl_v_X_B_y'], results['curl_v_X_B_z'] = diff.curl(v_X_B_x, v_X_B_y, v_X_B_z, dx, grid_npatch, stencil, clus_kp)
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
            - MIE_diver_B_x, MIE_diver_B_y, MIE_diver_B_z: null divergence of the magnetic field
            - MIE_compres_x, MIE_compres_y, MIE_compres_z: compressive component of the magnetic field induction
            - MIE_stretch_x, MIE_stretch_y, MIE_stretch_z: stretching component of the magnetic field induction
            - MIE_advec_x, MIE_advec_y, MIE_advec_z: advection component of the magnetic field induction
            - MIE_drag_x, MIE_drag_y, MIE_drag_z: cosmic drag component of the magnetic field induction
            - MIE_total_x, MIE_total_y, MIE_total_z: total magnetic induction energy in the compact way
        - magnitudes: dictionary containing the magnitudes of the components if mag is True:
            - MIE_diver_B_mag, MIE_drag_mag, MIE_compres_mag, MIE_stretch_mag, MIE_advec_mag, MIE_total_mag: magnitudes
        
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
        
        results['MIE_diver_B_x'] = [((1/a) * clus_vx[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)] # We have to run across all the patches.
        results['MIE_diver_B_y'] = [((1/a) * clus_vy[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
        results['MIE_diver_B_z'] = [((1/a) * clus_vz[p] * vectorial_quantities['diver_B'][p]) if clus_kp[p] != 0 else 0 for p in range(n)]
    else:
        results['MIE_diver_B_x'] = zero
        results['MIE_diver_B_y'] = zero
        results['MIE_diver_B_z'] = zero

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
    magnitudes = {}

    for key, prefix in [
        ('divergence', 'MIE_diver_B'),
        ('compression', 'MIE_compres'),
        ('stretching', 'MIE_stretch'),
        ('advection', 'MIE_advec'),
        ('drag', 'MIE_drag'),
        ('total', 'MIE_total')
    ]:
        if components.get(key, False) and mag:
            magnitudes[f'{prefix}_mag'] = utils.magnitude(results[f'{prefix}_x'], results[f'{prefix}_y'], results[f'{prefix}_z'], clus_kp)
        else:
            magnitudes[f'{prefix}_mag'] = zero

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
            - MIE_diver_B_x, MIE_diver_B_y, MIE_diver_B_z: null divergence of the magnetic field
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
    
    if components.get('divergence', False):
        ## The null divergence of the magnetic field energy for numerical error purposes.

        results['MIE_diver_B2'] = [(clus_Bx[p] * induction_equation['MIE_diver_B_x'][p] + clus_By[p] * induction_equation['MIE_diver_B_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_diver_B_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_diver_B2'] = zero
        
    if components.get('compression', False):
        ## The compressive component of the magnetic field induction energy.
        
        results['MIE_compres_B2'] = [(clus_Bx[p] * induction_equation['MIE_compres_x'][p] + clus_By[p] * induction_equation['MIE_compres_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_compres_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_compres_B2'] = zero
    
    if components.get('stretching', False):
        ## The stretching component.
        
        results['MIE_stretch_B2'] = [(clus_Bx[p] * induction_equation['MIE_stretch_x'][p] + clus_By[p] * induction_equation['MIE_stretch_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_stretch_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_stretch_B2'] = zero
        
    if components.get('advection', False):
        ## The advection component.
        
        results['MIE_advec_B2'] = [(clus_Bx[p] * induction_equation['MIE_advec_x'][p] + clus_By[p] * induction_equation['MIE_advec_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_advec_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_advec_B2'] = zero    
    
    if components.get('drag', False):
        ## The cosmic drag component.
        
        results['MIE_drag_B2'] = [(clus_Bx[p] * induction_equation['MIE_drag_x'][p] + clus_By[p] * induction_equation['MIE_drag_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_drag_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_drag_B2'] = zero
    
    if components.get('total', False):
        ## The total magnetic induction energy in the compact way.
        
        results['MIE_total_B2'] = [(clus_Bx[p] * induction_equation['MIE_total_x'][p] + clus_By[p] * induction_equation['MIE_total_y'][p]
                                + clus_Bz[p] * induction_equation['MIE_total_z'][p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['MIE_total_B2'] = zero

    ## The cinectic energy.

    if clus_rho_rho_b:
        results['kinetic_energy_density'] = [((1/2) * clus_rho_rho_b[p] * clus_v2[p]) if clus_kp[p] != 0 else 0 for p in range(1+np.sum(grid_npatch))]
    else:
        results['kinetic_energy_density'] = zero
    
    end_time_induction_energy_terms = time.time()

    total_time_induction_energy_terms = end_time_induction_energy_terms - start_time_induction_energy_terms
    
    if verbose == True:
        print('Time for calculating the energy induction eq. terms in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction_energy_terms))))
        
    return results
    

def induction_vol_integral(componets, induction_energy, clus_b2,
                            clus_cr0amr, clus_solapst, clus_kp,
                            grid_irr, grid_zeta, grid_npatch,
                            grid_patchrx, grid_patchry, grid_patchrz,
                            grid_patchnx, grid_patchny, grid_patchnz,
                            it, sims, nmax, size, coords, rad, a0, level,
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
        - grid_patchrx, grid_patchry, grid_patchrz: patch sizes in the x, y, and z directions
        - grid_patchnx, grid_patchny, grid_patchnz: number of patches in the x, y, and z directions
        - it: index of the snapshot in the simulation
        - sims: name of the simulation
        - nmax: maximum number of patches
        - size: size of the grid
        - coords: coordinates of the grid
        - rad: radius of the grid
        - a0: scale factor of the universe
        - level: level of refinement in the AMR grid
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
        if componets.get(key, False):
            results[f'int_{prefix}'] = utils.vol_integral(induction_energy[prefix], units, a0, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch,
                                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                            size[0], nmax, coords, rad, max_refined_level=level, kept_patches=clus_kp)
            if verbose == True:
                print(f'Snap {it} in {sims}: {key} energy density volume integral computed.')
        else:
            results[f'int_{prefix}'] = zero
    
    if induction_energy['kinetic_energy_density']:
        results['int_kinetic_energy'] = utils.vol_integral(induction_energy['kinetic_energy_density'], units, a0, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch,
                                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                            size[0], nmax, coords, rad, max_refined_level=level, kept_patches=clus_kp)
        if verbose == True:
            print(f'Snap {it} in {sims}: Kinetic energy density volume integral computed.')
    else:
        results['int_kinetic_energy'] = zero
        
    if clus_b2:
        results['int_b2'] = utils.vol_integral(clus_b2, units, a0, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch,
                                                grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                                size[0], nmax, coords, rad, max_refined_level=level, kept_patches=clus_kp)
        if verbose == True:
            print(f'Snap {it} in {sims}: Magnetic energy density volume integral computed.')
    else:
        results['int_b2'] = zero
    
    results['volume'] = utils.vol_integral(locals()[f'{prefix}_B2'], units, a0, grid_zeta, clus_cr0amr, clus_solapst, grid_npatch,
                                            grid_patchrx, grid_patchry, grid_patchrz, grid_patchnx, grid_patchny, grid_patchnz,
                                            size[0], nmax, coords, rad, max_refined_level=level, kept_patches=clus_kp, vol=True)

    end_time_induction = time.time()

    total_time_induction = end_time_induction - start_time_induction

    if verbose == True:
        print('Time for induction integration in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_induction))))

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
        - coords: coordinates of the grid
        - rmin: minimum radius for the radial profile
        - rad: radius of the grid
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
                                            size[0], nmax, ncores=1, kept_patches=clus_kp)
    
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
                                            solapst=clus_solapst, npatch=grid_npatch, size=size[0], nmax=nmax, up_to_level=level)
            if verbose:
                print(f'Snap {it} in {sims}: {key} profile done')
        else:
            results[f'{prefix}_profile'] = zero
    
    if induction_energy['kinetic_energy_density']:
        clean_field = utils.clean_field(induction_energy['kinetic_energy_density'], clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        _, results['kinetic_energy_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size[0], nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: Kinetic profile done')
    else:
        results['kinetic_energy_profile'] = zero
        
    if clus_b2:
        clean_field = utils.clean_field(clus_b2, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        _, results['clus_B2_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size[0], nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: B2 profile done')
    else:
        results['clus_B2_profile'] = zero
    
    if clus_rho_rho_b:
        clean_field = utils.clean_field(clus_rho_rho_b, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=level)
        profile_bin_centers, results['clus_rho_rho_b_profile'] = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size[0], nmax=nmax, up_to_level=level)
        if verbose:
            print(f'Snap {it} in {sims}: Density profile done')
    else:
        results['clus_rho_rho_b_profile'] = zero
        profile_bin_centers, _ = utils.radial_profile_vw(field=clean_field, clusrx=coords[0], clusry=coords[1], clusrz=coords[2], rmin=rmin, rmax=rad,
                                                nbins=nbins, logbins=logbins, cellsrx=X, cellsry=Y, cellsrz=Z, cr0amr=clus_cr0amr,
                                                solapst=clus_solapst, npatch=grid_npatch, size=size[0], nmax=nmax, up_to_level=level)
    
    results['profile_bin_centers'] = profile_bin_centers
    
    end_time_profile = time.time()

    total_time_profile = end_time_profile - start_time_profile
    
    if verbose == True:
        print('Time for profile calculation in snap '+ str(grid_irr) + ': '+str(strftime("%H:%M:%S", gmtime(total_time_profile))))
        
    return results
    

def uniform_field(field, clus_cr0amr, clus_solapst, grid_npatch,
                grid_patchnx, grid_patchny, grid_patchnz, 
                grid_patchrx, grid_patchry, grid_patchrz,
                nmax, size, Box,
                up_to_level=4, ncores=1, verbose=False):
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
        - verbose: boolean to print the progress of the computation (default is False)
        
    Returns:
        - uniform_field: cleaned and projected field on a uniform grid
        
    Author: Marco Molina
    '''
    
    if up_to_level > 4:
        print("Warning: The resolution level is larger than 4. The code will take a long time to run.")
        
    clean_field = utils.clean_field(field, clus_cr0amr, clus_solapst, grid_npatch, up_to_level=up_to_level)
    
    uniform_field = a2u.main(box = Box[1:], up_to_level = up_to_level, nmax = nmax, size = size, npatch = grid_npatch, patchnx = grid_patchnx, patchny = grid_patchny,
                            patchnz = grid_patchnz, patchrx = grid_patchrx, patchry = grid_patchry, patchrz = grid_patchrz,
                            field = clean_field, ncores = ncores, verbose = verbose)
        
    return uniform_field


def uniform_induction(components, induction_equation,
                    clus_cr0amr, clus_solapst, grid_npatch,
                    grid_patchnx, grid_patchny, grid_patchnz, 
                    grid_patchrx, grid_patchry, grid_patchrz,
                    it, sims, nmax, size, Box,
                    up_to_level=4, ncores=1, verbose=False):
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
        - verbose: boolean to print the data type loaded or not (default is False)
        
    Returns:
        - uniform_field: cleaned and projected field on a uniform grid
        
    Author: Marco Molina
    '''
    
    ### Preallocate all possible outputs as zeros
    
    n = 1 + np.sum(grid_npatch)
    zero = [0] * n
    
    results = {}
    
    for key, prefix in [
        ('divergence', 'MIE_diver_B'),
        ('compression', 'MIE_compres'),
        ('stretching', 'MIE_stretch'),
        ('advection', 'MIE_advec'),
        ('drag', 'MIE_drag'),
        ('total', 'MIE_total')
    ]:
        if components.get(key, False):
            results[f'uniform_{prefix}_x'] = uniform_field(induction_equation[f'{prefix}_x'], clus_cr0amr, clus_solapst, grid_npatch,
                                                            grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
                                                            nmax, size, Box, up_to_level=up_to_level, ncores=ncores, verbose=verbose)
            results[f'uniform_{prefix}_y'] = uniform_field(induction_equation[f'{prefix}_y'], clus_cr0amr, clus_solapst, grid_npatch,
                                                            grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
                                                            nmax, size, Box, up_to_level=up_to_level, ncores=ncores, verbose=verbose)
            results[f'uniform_{prefix}_z'] = uniform_field(induction_equation[f'{prefix}_z'], clus_cr0amr, clus_solapst, grid_npatch,
                                                            grid_patchnx, grid_patchny, grid_patchnz, grid_patchrx, grid_patchry, grid_patchrz,
                                                            nmax, size, Box, up_to_level=up_to_level, ncores=ncores, verbose=verbose)
            if verbose == True:
                print(f'Snap {it} in {sims}: {key} uniform field done')
        else:
            results[f'uniform_{prefix}_x'] = zero
            results[f'uniform_{prefix}_y'] = zero
            results[f'uniform_{prefix}_z'] = zero
        
    return results












    # if A2U == False:
    #     del MIE_diver_B_x, MIE_diver_B_y, MIE_diver_B_z, MIE_compres_x, MIE_compres_y, MIE_compres_z, MIE_stretch_x, MIE_stretch_y, MIE_stretch_z, MIE_advec_x, MIE_advec_y, MIE_advec_z, MIE_drag_x, MIE_drag_y, MIE_drag_z, MIE_total_x, MIE_total_y, MIE_total_z, clus_vx, clus_vy, clus_vz, clus_Bx, clus_By, clus_Bz, clus_v2
    #     gc.collect()