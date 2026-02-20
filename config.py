"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

config module
Defines configuration parameters for the simulation, including seed parameters and output parameters.

Created by Marco Molina Pradillo
"""

import numpy as np
import os
import psutil
import copy
from scripts.units import *
from scripts.readers import write_parameters
from scripts.test import test_limits

# ============================
# Only edit the section below
# ============================

# Induction Parameters #

IND_PARAMS = {
    "nmax": [128, 128],
    "nmay": [128, 128],
    "nmaz": [128, 128],
    "size": [40, 40], # Size of the box in Mpc
    "npalev": [40000, 40000],
    "nlevels": [7, 9],
    "namrx": [32, 32],
    "namry": [32, 32],
    "namrz": [32, 32],
    "nbins": [25, 25], # Number of bins for the profiles histograms
    "rmin": [0.01, 0.01], # Minimum radius to calculate the profiles
    "logbins": True, # Use logarithmic bins
    "F": 2, # Factor to multiply the viral radius to define the box size
    "vir_kind": 1, # 1: Reference virial radius at the last snap, 2: Reference virial radius at each epoch
    "rad_kind": 1, # 1: Comoving, 2: Physical
    # "units": energy_to_erg, # Factor to convert the units of the resulting volume integrals
    "units": 1, # Factor to convert the units of the resulting volume integrals
    # "level": [0,1,2,3,4,5,6,7], # Max. level of the AMR grid to be read
    # "up_to_level": [0,1,2,3,4,5,6,7], # AMR level up to which calculate
    # "level": [0,1,5], # Max. level of the AMR grid to be read
    # "up_to_level": [0,1,5], # AMR level up to which calculate
    "level": [4], # Max. level of the AMR grid to be read
    "up_to_level": [4], # AMR level up to which calculate
    # "level": [9], # Max. level of the AMR grid to be read
    # "up_to_level": [9], # AMR level up to which calculate
    "region": 'BOX', # Region of interest to calculate the induction components (BOX, SPH, or None)
    "a0": a0_masclet,
    # "a0": a0_isu,
    "H0": H0_masclet,
    # "H0": H0_isu,
    "epsilon": 1e-30,
    "stencil": 3, # Stencil to calculate the derivatives (either 3 or 5)
    "buffer": True, # Use buffer zones to avoid boundary effects BEFORE differentiating near patch boundaries (recommended but slower)
    "interpol": 'TRILINEAR', # Interpolation method ('TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST').
    "use_siblings": True, # Use sibling patches in buffer (if False, only parent interpolation is used)
    "parent": True, # Parent mode fills frontiers after differenciating using the interpolation in "parent_interpol" (no extra buffer cells)
    "parent_interpol": 'NEAREST', # Interpolation method for parent fill ('TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST')
    "blend": True, # Blend boundary values from buffer differentiation with parent-filled boundaries (only if parent is True)
    "divergence": True, # Process the divergence induction component
    "compression": True, # Process the compression induction component
    "stretching": True, # Process the stretching induction component
    "advection": True, # Process the advection induction component
    "drag": True, # Process the drag induction component
    "total": True, # Process the total induction component
    "mag": False, # Calculate magnetic induction components magnitudes
    "energy_evolution": False, # Calculate the evolution of the energy budget
    "evolution_type": 'total', # Type of evolution to calculate (total or differential)
    "derivative": 'central', # Derivative to use for the evolution (implicit, central, RK or rate)
    "profiles": True, # Calculate the profiles of the induction components
    "projection": False, # Calculate the projection of the induction components
    "A2U": False, # Transform the AMR grid to a uniform grid
    "percentiles": False, # Calculate percentile thresholds of the magnetic field divergence
    "percentile_levels": (95, 90, 75, 50, 25), # Percentile thresholds to compute
    "return_vectorial": False, # Return the vectorial components of the induction terms
    "return_induction": False, # Return the induction terms arrays
    "return_induction_energy": False, # Return the induction energy terms arrays
    "test_params": {
        "test": False,
        "B0": 2.3e-8
    }
}

# Directories and Results Parameters #

OUTPUT_PARAMS = {
    # "save": False,
    # "verbose": False,
    "save": True,
    "verbose": True,
    "save_terminal": True,  # Save terminal output to file
    "bitformat": np.float32,
    # "bitformat": np.float64,
    "format": "npy",
    "ncores": 1,
    "Save_Cores": 4, # Number of cores to save for the system (Increase this number if having troubles with the memory when multiprocessing)
    # "run": f'MAGNAS_SSD_Evo_profile_test_plots',
    "run": f'MAGNAS_SSD_Evo_divergence_test_step_by_step_3_refine_profile',
    # "run": f'MAGNAS_SSD_Evo_divergence_test_step_by_step_29_evo_fix_def',
    "sims": ["cluster_B_low_res_paper_2020"], # Simulation names, must match the name of the simulations folder in the data directory
    # "sims": ["cluster_L40_p32_agn_sim_7_bis"], # Simulation names, must match the name of the simulations folder in the data directory
    # "sims": ["cluster_B_low_res_paper_2020", "cluster_L40_p32_agn_sim_7_bis"], # Simulation names, must match the name of the simulations folder in the data directory
    # "it": [[1300], [500]], # For different redshift snap iterations analysis
    "it": [1800],
    # "it": list(range(1000, 2101, 50)) + [2119],
    # "it": list(range(250, 2101, 50)) + [2119], # For different redshift snap iterations analysis
    # "it": list(range(50, 2101, 50)) + [2119], # For different redshift snap iterations analysis
    "dir_DM": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_gas": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_grids": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_halos": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_ASOHF/",
    "dir_vortex": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_VORTEX/",
    # "outdir": "/scratch/marcomol/output_files_PRIMAL_",
    "outdir": "/home/marcomol/trabajo/data/out/",
    "plotdir": "plots/",
    "rawdir": "raw_data_out/",
    "terminaldir": "terminal_output/",
    "ID1": "dynamo/",
    # "ID2": "divergence_test_percentile_evo_def",
    "ID2": "divergence_deep_test",
    # "ID2": "profiles_test",
    "random_seed": 23 # Set the random seed for reproducibility
}


SIM_CHARACTERISTICS = {
# Simulation Characteristics #    
    # Default characteristics applied to all simulations unless overridden
    "default": {
        # EXISTENCE flags: What fields exist in the data files
        # Note: delta, vx/vy/vz always exist (not configurable)
        "is_mascletB": True,      # Files contain magnetic fields (Bx, By, Bz)
        "is_cooling": True,       # Files contain cooling data (temp, metalicity)
        "has_cr0amr": True,       # Files contain cosmic ray refinement flag
        "has_solapst": True,      # Files contain solapst mask flag
        "has_pres": True,         # Files contain pressure field
        "has_pot": True,          # Files contain gravitational potential
        "has_opot": False,        # Files contain old gravitational potential (rare)
        
        # READ flags: What fields to actually read (optional, defaults to True)
        # Set to False to skip reading a field that exists (saves memory/time)
        "read_velocity": True,    # Read velocity fields (vx, vy, vz)
        "read_B": True,           # Read magnetic fields (Bx, By, Bz)
        "read_pressure": False,    # Read pressure field
        "read_potential": False,   # Read gravitational potential
        "read_old_potential": False,  # Read old potential (rarely needed)
        "read_temperature": False, # Read temperature (if cooling exists)
        "read_metalicity": False,  # Read metalicity (if cooling exists)
        "read_cr0amr": True,      # Read refinement flag
        "read_solapst": True,     # Read solapst mask
    },
    # Simulation-specific overrides (use simulation name as key)
    "cluster_B_low_res_paper_2020": {
        "is_cooling": False,      # This simulation has NO cooling (no temp/metalicity in file)
        "is_mascletB": True,      # This simulation has magnetic fields (Bx, By, Bz in file)
    },
    "cluster_L40_p32_agn_sim_7_bis": {
        "is_cooling": True,       # This simulation HAS cooling (temp/metalicity in file)
        "is_mascletB": True,      # This simulation has magnetic fields (Bx, By, Bz in file)
    },
    # Add more simulations here as needed:
    # "your_simulation_name": {
    #     "is_cooling": False,
    #     "output_temp": False,
    #     etc...
    # }
}

EVO_PLOT_PARAMS = {
    'evolution_type': IND_PARAMS["evolution_type"],
    'derivative': IND_PARAMS["derivative"],
    'x_axis': 'zeta', # 'zeta' or 'years'
    'x_scale': 'lin', # 'lin' or 'log'
    'y_scale': 'log',
    'xlim': [2.5, 0], # None for auto
    # 'xlim': None, # None for auto
    # 'ylim': [1e57, 1e60], # For the test
    # 'ylim': [1e58, 1e63], # None for auto
    'ylim': None, # None for auto
    'cancel_limits': False, # bool to flip the x axis (useful for zeta)
    'figure_size': [12, 8], # [width, height]
    'line_widths': [5, 1.5], # [line1, line2] for main and component lines
    'plot_type': 'smoothed', # 'raw', 'smoothed', or 'interpolated' to choose plot style
    'smoothing_sigma': 1.1, # sigma for Gaussian smoothing (only for 'smoothed' type)
    'interpolation_points': 100, # number of points for interpolation (only for 'interpolated' type)
    'interpolation_kind': 'cubic', # 'linear', 'cubic', or 'nearest' for interpolation method (only for 'interpolated' type)
    'volume_evolution': True, # bool to plot volume evolution as additional figure
    'title': 'Magnetic Field Evolution Analysis',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

PROFILE_PLOT_PARAMS = {
    'it_indx': [0],
    # 'it_indx': [0,-1], # Index of the iteration to plot (default: first and last)
    # 'it_indx': list(range(len(OUTPUT_PARAMS['it']))), # Index to plot all iterations
    'x_scale': 'log', # 'lin' or 'log'
    'y_scale': 'log', # 'lin' or 'log'
    'xlim': None, # None for auto
    'ylim': None, # None for auto
    'rylim': None, # None for auto
    'dylim': None, # None for auto
    # 'xlim': [5e-3,1e0], # None for auto
    # 'ylim': [3e53,1e65], # None for auto
    # 'rylim': [3e39,3e47], # None for auto
    # 'dylim': [1e-28,1e-25], # None for auto
    'figure_size': [12, 8], # [width, height]
    'line_widths': [3, 2], # [line1, line2] for main and component lines
    'plot_type': 'smoothed', # 'raw', 'smoothed', or 'interpolated' to choose plot style
    'smoothing_sigma': 1.1, # sigma for Gaussian smoothing (only for 'smoothed' type)
    'interpolation_points': 100, # number of points for interpolation (only for 'interpolated' type)
    'interpolation_kind': 'cubic', # 'linear', 'cubic', or 'nearest' for interpolation method (only for 'interpolated' type)
    'title': 'Induction Radial Profile',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

PERCENTILE_PLOT_PARAMS = {
    'x_axis': 'years', # 'zeta' or 'years'
    'x_scale': 'lin', # 'lin' or 'log'
    'y_scale': 'log', # 'lin' or 'log'
    'xlim': None, # None for auto
    'ylim': None, # None for auto
    'figure_size': [6, 6], # [width, height]
    'line_widths': [2.0, 1.5], # [percentile_lines, max_line]
    'alpha_fill': 0.20, # transparency for shaded bands
    'title': 'Divergence Relative Error',
    # 'title': ' Percentile Threshold Evolution',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

SCAN_PLOT_PARAMS = {
    'study_box': 1.0,            # fraction of box side to scan (0,1]
    'depth': 2,                  # depth (slices) for the scan slab - small value to see short patches better
    'projection_mode': 'min',   # 'max', 'min' or 'sum' for the scan projection
    'arrow_scale': 1.0,          # scale for arrow annotation
    'units': 'Mpc',              # 'Mpc' or 'kpc'
    'cmap': 'magma',             # Matplotlib colormap for the scan
                                 # For discrete levels, use: 'tab10', 'Set1', 'Set2', 'Set3', 'Paired'
                                 # For continuous: 'viridis', 'plasma', 'inferno', 'magma'
                                 # Current options good for discrete: 'Accent', 'tab10', 'Set1'
    'interval': 100,             # ms between frames
    'title': 'Buffer Assignment Scan',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

DEBUG_PARAMS = {
    # ==== Iteration Selection for Debugging ====
    # Iteration indices to enable debugging on specific snapshots
    # Default: first and last snapshots
    # "it_indx": [0, -1],
    "it_indx": [-1],
    # ==== Unigrid Interpolation Mode ====
    "unigrid_interp_mode": 'DIRECT',  # 'DIRECT', 'NGP', or 'TRILINEAR'
                                       # 'DIRECT' (recommended): avoids spurious artifacts from zero-division in base grid
                                       # 'NGP': Nearest Grid Point with position calculations  
                                       # 'TRILINEAR': full trilinear interpolation (may cause artifacts at boundaries)
    # ==== Output Cleaning ====
    # If True, replace NaN/inf/outliers with zeros in debug visualizations ("sweep under the rug")
    # If False (default), preserve raw data to diagnose errors (recommended for debugging)
    "clean_output": False,
    # ==== Buffer Level Check ====
    "buffer_level_check": {
        "enabled": False, # Whether to include buffer level checks during the scan or buffer test
        "check_cell_patch": True,# Whether to check the cell_patch values during the buffer scan (only if buffer_level_check is True)
    },
    # ==== Patch Position Analysis ====
    "patch_analysis": {
        # "enabled": False,  # Enable patch position diagnostics
        "enabled": False,  # Enable patch position diagnostics
        "verbose": True,  # Print detailed results
        "suspicious_threshold": 18,  # Threshold for flagging suspicious patch positions (in Mpc/h)
    },
    # ==== Percentile Calculation Parameters ====
    "percentile_params": {
        "exclude_boundaries": False,  # Exclude patch boundary cells from percentile calculation
        "boundary_width": 2,          # Number of boundary cells to exclude (if exclude_boundaries=True)
        "exclude_zeros": False,        # Exclude zero values from percentile calculation
    },
    # ==== Buffer Pipeline Debug ====
    "buffer": {
        "enabled": False,  # Enable buffer pass-through validation test
        "verbose": True,  # Print detailed results
    },
    # ==== Divergence Method Debug ====
    "divergence": {
        "enabled": False,  # Enable divergence consistency validation test
        # "enabled": True,
        "verbose": True,  # Print detailed results
    },
    # ==== Ghost Cell Inspection (Buffer Diagnostics) ====
    # Uses interpol method from IND_PARAMS
    "ghost_cell_inspection": {
        "enabled": True,  # Enable ghost cell value inspection for buffer validation
        "verbose": True,   # Print detailed results
    },
    # ==== Buffer vs Extrapolation Comparison ====
    # Uses interpol, stencil, and use_siblings from IND_PARAMS
    "buffer_vs_extrapolation": {
        "enabled": True,  # Enable comparison of buffer vs. extrapolation divergence
        "verbose": True,   # Print detailed results
    },
    # ==== Divergence Spatial Distribution ====
    "divergence_spatial": {
        "enabled": True,  # Enable spatial distribution analysis of divergence
        "verbose": True,   # Print detailed results
    },
    # ==== Scan Animation Debug ====
    "scan_animation": {
        # "enabled": True,
        "enabled": False,
        "verbose": True,
        "save": True
    },
    # ==== Volume Analysis ====
    "volume_analysis": {
        "enabled": False,  # Enable volume analysis diagnostics
        "verbose": True  # Print detailed results
    },
    # ==== Field Analysis Debug (for field-specific outputs) ====
    "field_analysis": {
        "enabled": False,  # Enable field analysis debug outputs
        "bins": 100,  # Number of bins for histograms
        "log_scale": True,  # Use logarithmic scale
        "%points": 1001,  # Number of points for analysis
        "subsample_fraction": 0.2,  # Fraction of data to subsample
        "central_fraction": 1.0,  # Fraction of central region
        "uniform_grid": False,  # Convert to uniform grid (True) or keep as AMR (False)
        "clean_field": False,  # Whether to clean the field before plotting (removes masked regions)
        "title": "Divergence Induction",
        "quantities": ["Magnetic Field Energy", "Magnetic Field Divergence", 
                    "X-Divergence Induction", "Y-Divergence Induction", 
                    "Z-Divergence Induction", "Divergence Induction Energy"],
        "dpi": 300,
        "run": OUTPUT_PARAMS["run"]
    }
}

# ============================
# Only edit the section above
# ============================

def _expand_per_sim_param(value, sims_count, param_name):
    if isinstance(value, (list, tuple, np.ndarray)):
        value_list = list(value)
        if len(value_list) == 1 and sims_count > 1:
            return value_list * sims_count
        if len(value_list) == sims_count:
            return value_list
        if len(value_list) == 0:
            raise ValueError(f"IND_PARAMS['{param_name}'] cannot be empty")
        if sims_count == 1:
            return value_list
        raise ValueError(
            f"IND_PARAMS['{param_name}'] must have length 1 or {sims_count} (got {len(value_list)})"
        )
    return [value for _ in range(sims_count)]

def _normalize_it_list(it_value, sims_count):
    if isinstance(it_value, (list, tuple, np.ndarray)):
        it_list = list(it_value)
        if len(it_list) > 0 and all(isinstance(v, (list, tuple, np.ndarray)) for v in it_list):
            if len(it_list) == 1 and sims_count > 1:
                return [list(it_list[0]) for _ in range(sims_count)]
            if len(it_list) == sims_count:
                return [list(v) for v in it_list]
            raise ValueError(
                f"OUTPUT_PARAMS['it'] must have length 1 or {sims_count} when using per-sim lists"
            )
        return [list(it_list) for _ in range(sims_count)]
    return [[it_value] for _ in range(sims_count)]

sims_count = len(OUTPUT_PARAMS["sims"])

per_sim_keys = [
    "nmax",
    "nmay",
    "nmaz",
    "size",
    "npalev",
    "nlevels",
    "namrx",
    "namry",
    "namrz",
    "nbins",
    "rmin",
]

for key in per_sim_keys:
    IND_PARAMS[key] = _expand_per_sim_param(IND_PARAMS[key], sims_count, key)

OUTPUT_PARAMS["it"] = _normalize_it_list(OUTPUT_PARAMS["it"], sims_count)
OUTPUT_PARAMS["total_iterations"] = sum(len(it_list) for it_list in OUTPUT_PARAMS["it"])

## The output parameters are used to create the image directories and other formatting parameters

outdir = OUTPUT_PARAMS["outdir"]
plotdir = OUTPUT_PARAMS["plotdir"]
rawdir = OUTPUT_PARAMS["rawdir"]
terminaldir = OUTPUT_PARAMS["terminaldir"]
ID1 = OUTPUT_PARAMS["ID1"]
ID2 = OUTPUT_PARAMS["ID2"]

# We create the folder for the plots and data
image_folder = outdir + plotdir + ID1 + f'MAGNAS_SSD_{ID2}'
data_folder = outdir + rawdir + ID1 + f'MAGNAS_SSD_{ID2}'
terminal_folder = outdir + terminaldir + ID1 + f'MAGNAS_SSD_{ID2}'
parameters_folders = []

# List of folders to check
folders = [image_folder, data_folder, terminal_folder]
for i in range(len(OUTPUT_PARAMS['sims'])):
    folders.append(data_folder + '/' + OUTPUT_PARAMS['sims'][i] + '/')
    parameters_folders.append(data_folder + '/' + OUTPUT_PARAMS['sims'][i] + '/')

for folder in folders:
    # Check if the directory already exists
    if os.path.exists(folder):
        # If it exists, exit the loop
        pass
    else:
        # If it doesn't exist, create the directory
        os.makedirs(folder) 

OUTPUT_PARAMS["image_folder"] = image_folder
OUTPUT_PARAMS["data_folder"] = data_folder
OUTPUT_PARAMS["terminal_folder"] = terminal_folder
OUTPUT_PARAMS["parameters_folders"] = parameters_folders

# Determine the format of the output files
if OUTPUT_PARAMS["bitformat"] == np.float32:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex64
elif OUTPUT_PARAMS["bitformat"] == np.float64:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex128

if IND_PARAMS["stencil"] not in [3, 5]:
    raise ValueError("Invalid stencil value. It must be either 3 or 5.")
elif IND_PARAMS["stencil"] == 3:
    IND_PARAMS["nghost"] = 1
elif IND_PARAMS["stencil"] == 5:
    IND_PARAMS["nghost"] = 2

# Validate interpolation method and adjust parameters for parent mode
allowed_interpol = ['TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST']
if IND_PARAMS["interpol"] not in allowed_interpol:
    raise ValueError("Invalid interpolation method. Must be 'TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST'.")
if IND_PARAMS.get("parent_interpol", IND_PARAMS["interpol"]) not in allowed_interpol:
    raise ValueError("Invalid parent_interpol method. Must be 'TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST'.")
IND_PARAMS["parent_interpol"] = IND_PARAMS.get("parent_interpol", IND_PARAMS["interpol"])

# If parent mode is selected, enforce compatible parameters
if IND_PARAMS["parent"] is True:
    blend_enabled = IND_PARAMS.get("blend", False)
    method_name = 'PARENT' + ' + ' + IND_PARAMS["parent_interpol"]
    interp_desc = ''
    if IND_PARAMS["parent_interpol"] == 'TSC':
        interp_desc = 'TSC (smooth Triangular-Shaped Cloud)'
    elif IND_PARAMS["parent_interpol"] == 'TRILINEAR':
        interp_desc = 'Trilinear (linear interpolation)'
    elif IND_PARAMS["parent_interpol"] == 'NEAREST':
        interp_desc = 'Nearest Neighbor (discontinuous)'
    elif IND_PARAMS["parent_interpol"] == 'LINEAR':
        interp_desc = 'Linear (linear interpolation)'
    elif IND_PARAMS["parent_interpol"] == 'SPH':
        interp_desc = 'SPH (Smoothed Particle Hydrodynamics)'
    print(f"\n⚠️  Parent mode enabled ({method_name}):")
    if blend_enabled:
        print("    - blend: True (buffer and parent fill settings applied separately)")
    else:
        print("    - parent fill will use nghost=0 and use_siblings=False at call site")
    print(f"    - parent interpolation: {interp_desc}")

## Some seed parameters are calculated from the previous ones

size = IND_PARAMS["size"]
nmax = IND_PARAMS["nmax"]
a0 = IND_PARAMS["a0"]
H0 = IND_PARAMS["H0"]

dx = [size[i]/nmax[i] for i in range(len(size))]  # Cell size in Mpc/h
volume = [] # (Mpc)^3

for i in range(len(OUTPUT_PARAMS['sims'])):
    
    volume.append(size[i]**3) # (Mpc/h)^3
    write_parameters(IND_PARAMS['nmax'][i], IND_PARAMS['nmay'][i], IND_PARAMS['nmaz'][i],
                    IND_PARAMS['npalev'][i], IND_PARAMS['nlevels'][i], IND_PARAMS['namrx'][i],
                    IND_PARAMS['namry'][i], IND_PARAMS['namrz'][i], size[i], path=parameters_folders[i])
    
IND_PARAMS["dx"] = dx
IND_PARAMS["volume"] = volume
OUTPUT_PARAMS["dir_params"] = parameters_folders

# Helper function to get simulation characteristics
def get_sim_characteristics(sim_name):
    """
    Get the characteristics for a specific simulation by merging default 
    characteristics with simulation-specific overrides.
    
    Args:
        sim_name: Name of the simulation
        
    Returns:
        Dictionary with all simulation characteristics
    """
    # Start with default characteristics
    characteristics = SIM_CHARACTERISTICS["default"].copy()
    
    # Override with simulation-specific settings if they exist
    if sim_name in SIM_CHARACTERISTICS:
        characteristics.update(SIM_CHARACTERISTICS[sim_name])
    
    return characteristics

# Copy percentile params from DEBUG_PARAMS into PERCENTILE_PLOT_PARAMS
if "percentile_params" in DEBUG_PARAMS:
    PERCENTILE_PLOT_PARAMS.update(DEBUG_PARAMS["percentile_params"])

if IND_PARAMS["test_params"]["test"]:
    TEST = test_limits(a0, OUTPUT_PARAMS['dir_grids'], OUTPUT_PARAMS['dir_params'][0], 
                                            OUTPUT_PARAMS['sims'][0], OUTPUT_PARAMS['it'],
                                            nmax[0], size[0])
    
    (
        x_test,
        y_test,
        z_test,
        k,
        ω,
        clus_cr0amr_test,
        clus_solapst_test,
        grid_patchrx_test,
        grid_patchry_test,
        grid_patchrz_test,
        grid_patchnx_test,
        grid_patchny_test,
        grid_patchnz_test,
        grid_npatch_test
    ) = TEST
    
    print(f"Test fields parameters: k = {k}, ω = {ω}")
    
    IND_PARAMS["test_params"]["k"] = k
    IND_PARAMS["test_params"]["ω"] = ω
    IND_PARAMS["test_params"]["x_test"] = x_test
    IND_PARAMS["test_params"]["y_test"] = y_test
    IND_PARAMS["test_params"]["z_test"] = z_test
    IND_PARAMS["test_params"]["clus_cr0amr_test"] = clus_cr0amr_test
    IND_PARAMS["test_params"]["clus_solapst_test"] = clus_solapst_test
    IND_PARAMS["test_params"]["grid_patchrx_test"] = grid_patchrx_test
    IND_PARAMS["test_params"]["grid_patchry_test"] = grid_patchry_test
    IND_PARAMS["test_params"]["grid_patchrz_test"] = grid_patchrz_test
    IND_PARAMS["test_params"]["grid_patchnx_test"] = grid_patchnx_test
    IND_PARAMS["test_params"]["grid_patchny_test"] = grid_patchny_test
    IND_PARAMS["test_params"]["grid_patchnz_test"] = grid_patchnz_test
    IND_PARAMS["test_params"]["grid_npatch_test"] = grid_npatch_test
    IND_PARAMS["test_params"]["evo_plot_params"] = copy.deepcopy(EVO_PLOT_PARAMS)
    IND_PARAMS["test_params"]["evo_plot_params"]["run"] = IND_PARAMS["test_params"]["evo_plot_params"]["run"] + '_analytic_test_field'
    IND_PARAMS["test_params"]["evo_plot_params"]["title"] = 'Analytic Magnetic Field Evolution Analysis'

## Inducction components to be checked

IND_PARAMS["components"] = {
    "divergence": IND_PARAMS["divergence"],
    "compression": IND_PARAMS["compression"],
    "stretching": IND_PARAMS["stretching"],
    "advection": IND_PARAMS["advection"],
    "drag": IND_PARAMS["drag"],
    "total": IND_PARAMS["total"]
}

## Parallelization configuration based on available resources

# Get system information
ram_capacity = psutil.virtual_memory().total  # Total RAM in bytes
ram_capacity_gb = ram_capacity / (1024 ** 3)  # Convert to GB
cpu_count = psutil.cpu_count(logical=True)    # Total CPU cores (including hyperthreading)
cpu_physical = psutil.cpu_count(logical=False) # Physical cores only
max_level = max(IND_PARAMS.get("up_to_level", [0]))  # Max AMR level to process

print(f"\n{'='*60}")
print(f"SYSTEM RESOURCES")
print(f"{'='*60}")
print(f"Total RAM: {ram_capacity_gb:.2f} GB")
print(f"CPU cores: {cpu_physical} physical, {cpu_count} logical")
print(f"AMR max level: {max_level}")

# Calculate expected memory usage
max_nmax_index = int(np.argmax(IND_PARAMS["nmax"]))
max_nmay_index = int(np.argmax(IND_PARAMS["nmay"]))
max_nmaz_index = int(np.argmax(IND_PARAMS["nmaz"]))

# Base grid size
base_cells = IND_PARAMS["nmax"][max_nmax_index] * IND_PARAMS["nmay"][max_nmay_index] * IND_PARAMS["nmaz"][max_nmaz_index]
array_size_bytes = base_cells * np.dtype(OUTPUT_PARAMS["bitformat"]).itemsize

# Estimate total memory footprint (considering multiple fields: Bx, By, Bz, vx, vy, vz, derivatives, etc.)
# Typical run uses ~6 fields for input + ~10 for derivatives/outputs = ~16 arrays
estimated_arrays_count = 16

# AMR refinement increases peak array sizes locally; we apply a conservative multiplier
# Note: true AMR does not fill the whole domain at finest level, so we scale by a small coverage factor
amr_coverage_factor = 0.10  # 10% default coverage at finest levels (safety bias)
refinement_multiplier = (2 ** (3 * max_level)) * amr_coverage_factor if max_level > 0 else 1.0
refinement_multiplier = max(refinement_multiplier, 1.0)

estimated_memory_gb = ((array_size_bytes * refinement_multiplier) * estimated_arrays_count) / (1024 ** 3)

print(f"\n{'='*60}")
print(f"MEMORY ESTIMATION")
print(f"{'='*60}")
print(f"Base grid size: {IND_PARAMS['nmax'][max_nmax_index]} x {IND_PARAMS['nmay'][max_nmay_index]} x {IND_PARAMS['nmaz'][max_nmaz_index]}")
print(f"Single array size: {array_size_bytes / (1024 ** 3):.3f} GB")
print(f"Estimated total memory usage (AMR-aware): {estimated_memory_gb:.2f} GB")
mem_ratio = (estimated_memory_gb / ram_capacity_gb)
print(f"Memory usage ratio: {100 * mem_ratio:.1f}% of total RAM")

# Decide on parallelization strategy
print(f"\n{'='*60}")
print(f"PARALLELIZATION CONFIGURATION")
print(f"{'='*60}")

# Automatic recommendation based on memory and CPU availability
recommended_parallel = False
recommended_ncores = OUTPUT_PARAMS["ncores"]  # Default from config

# Determine core reservation based on risk
reserve_ratio = 0.0
if mem_ratio >= 0.60 or max_level >= 7:
    # Extreme risk: reserve ~70% cores for safety
    reserve_ratio = 0.70
elif mem_ratio >= 0.40 or max_level >= 6:
    # High risk: reserve ~50% cores
    reserve_ratio = 0.50
else:
    # Normal: reserve based on configured Save_Cores
    reserve_ratio = min(OUTPUT_PARAMS["Save_Cores"] / max(cpu_physical, 1), 0.50)

reserved_cores = max(OUTPUT_PARAMS["Save_Cores"], int(cpu_physical * reserve_ratio))
available_cores = max(1, cpu_physical - reserved_cores)

# High memory usage (>40% RAM): recommend parallelization to distribute load across iterations
if mem_ratio >= 0.60:
    print(f"⚠️  Extreme memory risk detected ({estimated_memory_gb:.2f} GB ≥ 60% of {ram_capacity_gb:.2f} GB)")
    print(f"   Recommendation: Prefer SERIAL or minimal parallelism to avoid OOM/segfaults")
    recommended_parallel = available_cores > 1 and len(OUTPUT_PARAMS["it"]) > 1
    recommended_ncores = min(available_cores, max(1, len(OUTPUT_PARAMS["it"]) // 2))
    print(f"   Reserved cores: {reserved_cores}; Recommended cores: {recommended_ncores}")
elif mem_ratio >= 0.40:
    print(f"⚠️  High memory usage detected ({estimated_memory_gb:.2f} GB > 40% of {ram_capacity_gb:.2f} GB)")
    print(f"   Recommendation: Use parallel processing to handle multiple iterations efficiently")
    recommended_parallel = True
    recommended_ncores = min(available_cores, len(OUTPUT_PARAMS["it"]))  # Don't exceed number of iterations
    print(f"   Reserved cores: {reserved_cores}; Recommended cores: {recommended_ncores}")
    
# Low memory, multiple iterations: can benefit from parallelization
elif len(OUTPUT_PARAMS["it"]) > 1:
    print(f"✓ Memory usage is manageable ({estimated_memory_gb:.2f} GB < 40% of {ram_capacity_gb:.2f} GB)")
    print(f"  {len(OUTPUT_PARAMS['it'])} iterations to process")
    recommended_ncores = min(available_cores, len(OUTPUT_PARAMS["it"]))
    if recommended_ncores > 1:
        print(f"  Recommendation: Parallel processing can speed up execution")
        print(f"  Recommended cores: {recommended_ncores}")
        recommended_parallel = True
    else:
        print(f"  Single core available after reserving {reserved_cores} for system")
        recommended_parallel = False
        
# Single iteration: parallelization doesn't help
else:
    print(f"✓ Single iteration mode - parallel processing not beneficial")
    recommended_parallel = False

# Interactive configuration
if recommended_parallel:
    ask = input(f"\nUse parallel processing with {recommended_ncores} cores? (y/n) [recommended]: ").strip().lower()
    if ask == 'y' or ask == '':  # Default to yes
        OUTPUT_PARAMS["parallel"] = True
        OUTPUT_PARAMS["ncores"] = recommended_ncores
        print(f"✓ Parallel processing ENABLED with {recommended_ncores} cores")
    else:
        OUTPUT_PARAMS["parallel"] = False
        OUTPUT_PARAMS["ncores"] = 1
        ask_custom = input("  Do you want to manually set number of cores? (y/n): ").strip().lower()
        if ask_custom == 'y':
            try:
                custom_ncores = int(input(f"  Enter number of cores (1-{cpu_count}): "))
                if 1 <= custom_ncores <= cpu_count:
                    OUTPUT_PARAMS["parallel"] = custom_ncores > 1
                    OUTPUT_PARAMS["ncores"] = custom_ncores
                    if custom_ncores > 1:
                        print(f"✓ Parallel processing ENABLED with {custom_ncores} core(s)")
                    else:
                        print(f"✓ Serial processing mode (1 core)")
                else:
                    print(f"⚠️  Invalid input, keeping default: 1 core")
                    print(f"✓ Serial processing mode (1 core)")
            except ValueError:
                print(f"⚠️  Invalid input, keeping default: 1 core")
                print(f"✓ Serial processing mode (1 core)")
        else:
            print(f"✓ Serial processing mode (1 core)")
else:
    # Serial mode by default
    ask = input(f"\nParallelization not recommended for this configuration. Proceed in serial mode? (y/n) [yes]: ").strip().lower()
    if ask == 'n':
        try:
            custom_ncores = int(input(f"Enter number of cores to use (1-{cpu_count}): "))
            if 1 <= custom_ncores <= cpu_count:
                OUTPUT_PARAMS["parallel"] = custom_ncores > 1
                OUTPUT_PARAMS["ncores"] = custom_ncores
                print(f"✓ Override: Parallel processing ENABLED with {custom_ncores} core(s)")
            else:
                print(f"⚠️  Invalid input, using serial mode")
                print(f"✓ Serial processing mode (1 core)")
                OUTPUT_PARAMS["parallel"] = False
                OUTPUT_PARAMS["ncores"] = 1
        except ValueError:
            print(f"⚠️  Invalid input, using serial mode")
            print(f"✓ Serial processing mode (1 core)")
            OUTPUT_PARAMS["parallel"] = False
            OUTPUT_PARAMS["ncores"] = 1
    else:
        OUTPUT_PARAMS["parallel"] = False
        OUTPUT_PARAMS["ncores"] = 1
        print(f"✓ Serial processing mode (1 core)")

print(f"{'='*60}\n")