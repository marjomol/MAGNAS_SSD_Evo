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
    # "nmax": [128, 128],
    # "nmay": [128, 128],
    # "nmaz": [128, 128],
    "nmax": [64, 64],
    "nmay": [64, 64],
    "nmaz": [64, 64],
    "size": [40, 40], # Size of the box in Mpc
    "npalev": [7000, 7000],
    "nlevels": [10],
    "namrx": [32, 32],
    "namry": [32, 32],
    "namrz": [32, 32],
    "nbins": [25, 25], # Number of bins for the profiles histograms
    "rmin": [0.01, 0.01], # Minimum radius to calculate the profiles
    # "level": [0,1,2,3,4,5,6,7], # Max. level of the AMR grid to be read
    # "up_to_level": [0,1,2,3,4,5,6,7], # AMR level up to which calculate
    # "level": [0,1,5], # Max. level of the AMR grid to be read
    # "up_to_level": [0,1,5], # AMR level up to which calculate
    "level": [10], # Max. level of the AMR grid to be read
    "up_to_level": [10], # AMR level up to which calculate
    # "units": energy_to_erg, # Factor to convert the units of the resulting volume integrals
    "units": 1.0, # Factor to convert the units of the resulting volume integrals
    "logbins": True, # Use logarithmic bins
    "F": 2, # Factor to multiply the viral radius to define the box size
    "vir_kind": 2, # 1: Reference virial radius at the last snap, 2: Reference virial radius at each epoch
    "rad_kind": 1, # 1: Comoving, 2: Physical
    "region": 'BOX', # Region of interest shape to calculate the induction components (BOX, SPH, or None)
    "a0": a0_masclet,
    # "a0": a0_isu,
    "H0": H0_masclet,
    # "H0": H0_isu,
    "differentiation": {
        "epsilon": 1e-30, # Small value to avoid division by zero in differentiation and other operations (if needed)
        "stencil": 3, # Stencil to calculate the derivatives (either 3 or 5)
        "buffer": True, # Use buffer zones to avoid boundary effects BEFORE differentiating near patch boundaries (recommended but slower)
        "interpol": 'TRILINEAR', # Interpolation method ('TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST').
        "use_siblings": True, # Use sibling patches in buffer (if False, only parent interpolation is used)
        "parent": True, # Parent mode fills frontiers after differenciating using the interpolation in "parent_interpol" (no extra buffer cells)
        "parent_interpol": 'NEAREST', # Interpolation method for parent fill ('TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST')
        "blend": True # Blend boundary values from buffer differentiation with parent-filled boundaries (only if parent is True)
    },
    "components": {
        "divergence": True, # Process the divergence induction component
        "compression": True, # Process the compression induction component
        "stretching": True, # Process the stretching induction component
        "advection": True, # Process the advection induction component
        "drag": True, # Process the drag induction component
        "total": True, # Process the total induction component
        "kinetic_energy": False, # Calculate/plot kinetic energy measured in the simulation
        "magnetic_energy": True # Calculate/plot magnetic energy measured in the simulation
    },
    "return": {
        "mag": False, # Return magnetic induction components magnitudes
        "return_vectorial": False, # Return the vectorial components of the induction terms
        "return_induction": False, # Return the induction terms arrays
        "return_induction_energy": False, # Return the induction energy terms arrays
        "projection": False, # Calculate the projection of the induction components
        "A2U": False # Transform the AMR grid to a uniform grid
    },
    "divergence_filter": {
        "enabled": True, # Whether to apply a filter to the divergence field based on percentile thresholds
        "method": "mask", # "mask" or "clip" for masking or clipping values above the threshold
        "percentile": 99, # Percentile threshold to apply for the filter (if enabled)
        "use_abs": True, # Whether to use absolute values when applying the percentile threshold (recommended)
        "exclude_zeros": True # Whether to exclude zero values when calculating the percentile threshold (recommended)
    },
    "energy_evolution": {
        "enabled": True, # Calculate the evolution of the energy budget
        "derivative": 'central', # Derivative to use for the evolution (implicit_forward, central, alpha_fit, RK or rate)
        "normalized": False, # True: keeps B/sqrt(rho_b) in magnetic evolution outputs; False: shows results respect to physical B
        "volume_coordinates": 'physical', # Integration volume differential: 'physical' (a^3 dV) or 'comoving' (dV)
        "normalize_by_volume": False, # If True, divide integrated quantities by total integration volume
        "plot_total": True, # Calculate and plot total (integrated) energy evolution
        "plot_differential": True, # Calculate and plot differential (rate of change) energy evolution
        "plot_profiles": True # Plot radial profiles for induction-energy terms
    },
    "production_dissipation": {
        "enabled": True, # Calculate production/dissipation decomposition from induction-energy terms
        "normalized": False, # True: keeps B/sqrt(rho_b); False: multiplies final P/D integrals by rho_b
        "volume_coordinates": 'physical', # Integration volume differential for P/D: 'physical' (a^3 dV) or 'comoving' (dV)
        "normalize_by_volume": False, # If True, divide P/D integrated quantities by total integration volume
        "plot_absolute": False, # Plot absolute production and dissipation rates
        "plot_fractional": False, # Plot fractional production/dissipation contributions and net efficiency
        "plot_net": False, # Plot net contributions (production - dissipation)
        "plot_profiles": True, # Plot radial profiles for production/dissipation terms
        "plot_fractional_profiles": True # Plot fractional contribution profiles for production/dissipation terms
    },
    "percentiles": {
        "enabled": True, # Calculate percentile thresholds of the magnetic field divergence
        "percentile_levels": (95, 90, 75, 50, 25) # Percentile thresholds to compute
    },
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
    # "bitformat": np.float32,
    "bitformat": np.float64,
    "format": "npy",
    "ncores": 1,
    "Save_Cores": 2, # Number of cores to save for the system (Increase this number if having troubles with the memory when multiprocessing)
    "memory_safety_factor": 1.88, # Optional empirical correction for memory estimator (1.0 = no correction).
        # Increase (e.g. 1.3-2.0) if observed peak RAM is consistently above estimated RAM.
    # Optional runtime RAM profiling in main.py
    "memory_profiling": {
        "enabled": False,
        "log_interval": 1,       # Log every N processed snapshots
        "include_children": True, # Include worker processes (parallel mode) in RAM accounting
        "sample_seconds": 0.5,    # Sampling period for peak RAM tracking
        "gc_main_each_iteration": True,  # Run gc.collect() in main process after each processed snapshot
        "gc_worker_end": True     # Run gc.collect() inside each worker at end of process_iteration
    },
    # Recycle worker processes every N tasks in parallel mode to mitigate RAM growth
    # (set to None to disable recycling).
    "max_tasks_per_child": 1,
    "run": f'MAGNAS_SSD_Evo_PD_42',
    # "run": f'MAGNAS_SSD_Evo_profile_test_plots',
    # "run": f'MAGNAS_SSD_Evo_RAM_test_1_serial',
    # "sims": ["cluster_B_low_res_paper_2020"], # Simulation names, must match the name of the simulations folder in the data directory
    # "sims": ["box_L40_p32_128_l9_2026_full_box_7_bis"], # Simulation names, must match the nameof the simulations folder in the data directory
    "sims": ["prova_ORPHEUS_l10_1e4cool"],
    # "sims": ["cluster_B_low_res_paper_2020", "cluster_L40_p32_agn_sim_7_bis"], # Simulation names, must match the name of the simulations folder in the data directory
    # "it": [[1300], [500]], # For different redshift snap iterations analysis
    # "it": [900],
    # "it": [2119],
    "it": list(range(100, 900, 50)), # For different redshift snap iterations analysis
    # "it": list(range(1900, 2101, 50)) + [2119],
    # "it": list(range(50, 2101, 50)) + [2119], # For different redshift snap iterations analysis
    # "it": list(range(350, 2101, 50)) + [2119], # For different redshift snap iterations analysis
    # "dir_DM": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    # "dir_gas": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    # "dir_grids": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_DM": "/mnt/perdiu/scratch/",
    "dir_gas": "/mnt/perdiu/scratch/",
    "dir_grids": "/mnt/perdiu/scratch/",
    "dir_halos": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_ASOHF/",
    "dir_vortex": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_VORTEX/",
    # "outdir": "/scratch/marcomol/output_files_PRIMAL_",
    "outdir": "/home/marcomol/trabajo/data/out/",
    "plotdir": "plots/",
    "rawdir": "raw_data_out/",
    "terminaldir": "terminal_output/",
    "ID1": "dynamo/",
    "ID2": "new_sim_induction_analysis",
    # "ID2": "RAM_test",
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
    "prova_ORPHEUS_l10_1e4cool": {
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

PLOT_PALETTES = {
    "active": "classic",
    "available": {
        "classic": {
            "measured_energy": "#1f77b4",
            "induction_itemized": "#ff7f0e",
            "induction_compact": "#800020",
            "kinetic_energy": "#17becf",
            "production": "#2ca02c",
            "dissipation": "#d62728",
            "net_itemized": "#ff7f0e",
            "net_compact": "#800020",
            "efficiency": "#1f77b4",
            "density": "#2ca02c",
            "max_curve": "#111111",
            "negative_interval": "#364243",
            "component_colors": {
                "compression": "#9467bd",
                "stretching": "#ff9896",
                "advection": "#e377c2",
                "divergence": "#c5b0d5",
                "drag": "#7f7f7f"
            },
            "percentile_cmap": "viridis"
        },
        "colorblind": {
            "measured_energy": "#0072B2",
            "induction_itemized": "#E69F00",
            "induction_compact": "#000000",
            "kinetic_energy": "#56B4E9",
            "production": "#009E73",
            "dissipation": "#D55E00",
            "net_itemized": "#E69F00",
            "net_compact": "#000000",
            "efficiency": "#CC79A7",
            "density": "#009E73",
            "max_curve": "#444444",
            "negative_interval": "#666666",
            "component_colors": {
                "compression": "#0072B2",
                "stretching": "#D55E00",
                "advection": "#009E73",
                "divergence": "#CC79A7",
                "drag": "#999999"
            },
            "percentile_cmap": "cividis"
        }
    }
}

EVO_PLOT_PARAMS = {
    'palette_name': PLOT_PALETTES["active"],
    'palettes': PLOT_PALETTES["available"],
    'units': IND_PARAMS["units"],
    'plot_total': IND_PARAMS["energy_evolution"]["plot_total"],
    'plot_differential': IND_PARAMS["energy_evolution"]["plot_differential"],
    'derivative': IND_PARAMS["energy_evolution"]["derivative"],
    'x_axis': 'zeta', # 'zeta' or 'years'
    'x_scale': 'lin', # 'lin' or 'log'
    'y_scale': 'log', # Only for total evolution; 'lin' or 'log'
    # 'xlim': [10, 0], # None for auto
    # 'xlim': [2.5, 0], # None for auto
    'xlim': None, # None for auto
    'ylim': None, # None for auto
    # 'ylim': [1e57, 1e60], # For the test
    # 'ylim': [1e58, 1e63], # None for auto
    # 'ylim': [1e-19, 1e-16], # None for auto
    # 'ylim': [1e-19, 1e-13], # None for auto
    # 'ylim': [-0.2e-28, 0.8e-28], # None for auto
    'cancel_limits': True, # If True, ignore manual xlim/ylim; for zeta plots, also invert x-axis automatically
    'figure_size': [12, 8], # [width, height]
    'line_widths': [5.0, 3.0], # [line1, line2] for main and component lines
    'plot_type': 'raw', # 'raw', 'smoothed', or 'interpolated' to choose plot style
    'smoothing_sigma': 1.1, # sigma for Gaussian smoothing (only for 'smoothed' type)
    'interpolation_points': 100, # number of points for interpolation (only for 'interpolated' type)
    'interpolation_kind': 'cubic', # 'linear', 'cubic', or 'nearest' for interpolation method (only for 'interpolated' type)
    'volume_evolution': True, # bool to plot volume evolution as additional figure
    'title': 'Magnetic Field Evolution Analysis',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

PROD_DISS_PLOT_PARAMS = {
    'palette_name': PLOT_PALETTES["active"],
    'palettes': PLOT_PALETTES["available"],
    'x_axis': 'zeta', # 'zeta' or 'years'
    'x_scale': 'lin', # 'lin' or 'log'
    'y_scale': 'log', # 'lin' or 'log'
    # 'xlim': [2.5, 0], # None for auto
    'xlim': None, # None for auto
    'ylim': None, # None for auto
    'cancel_limits': False, # If True, ignore manual xlim/ylim; for zeta plots, also invert x-axis automatically
    'figure_size': [12, 8], # [width, height]
    'line_widths': [5.0, 3.0], # [main, components]
    'title': 'Production and Dissipation Evolution',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"],
    'units': IND_PARAMS["units"],
    'plot_total_prod_diss': False, # If False, hide total production/dissipation curves (green/red). Net curves are still plotted when available.
    'plot_absolute': IND_PARAMS["production_dissipation"]["plot_absolute"],
    'plot_fractional': IND_PARAMS["production_dissipation"]["plot_fractional"],
    'plot_net': IND_PARAMS["production_dissipation"]["plot_net"]
}

INDUCTION_PROFILE_PLOT_PARAMS = {
    'palette_name': PLOT_PALETTES["active"],
    'palettes': PLOT_PALETTES["available"],
    'units': IND_PARAMS["units"],
    # 'it_indx': [-1],
    "it_indx": list(range(0, 900, 50)), # For different redshift snap iterations analysis
    # 'it_indx': [0,-1], # Index of the iteration to plot (default: first and last)
    # 'it_indx': list(range(len(OUTPUT_PARAMS['it']))), # Index to plot all iterations
    'x_scale': 'log', # 'lin' or 'log'
    'y_scale': 'log', # 'lin' or 'log'
    'xlim': None, # None for auto
    # 'ylim': None, # None for auto
    'rylim': None, # None for auto
    'dylim': None, # None for auto
    # 'xlim': [1e-17,1e-5], # None for auto
    'ylim': [1e-17,1e-5], # None for auto
    # 'rylim': [1e-21,1e-14], # None for auto
    # 'dylim': [1e-28,1e-25], # None for auto
    'figure_size': [12, 8], # [width, height]
    'line_widths': [5.0, 3.0], # [line1, line2] for main and component lines
    'plot_type': 'smoothed', # 'raw', 'smoothed', or 'interpolated' to choose plot style
    'smoothing_sigma': 1.1, # sigma for Gaussian smoothing (only for 'smoothed' type)
    'interpolation_points': 100, # number of points for interpolation (only for 'interpolated' type)
    'interpolation_kind': 'cubic', # 'linear', 'cubic', or 'nearest' for interpolation method (only for 'interpolated' type)
    'component_alpha': 0.65, # Opacity for individual induction component curves (totals remain fully opaque)
    'fixed_legend': True, # If True, place legend at a fixed position inside axes (best for animations)
    'title': 'Induction Radial Profile',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"],
    'plot_density': False, # Whether to plot the density profile as a reference
    'plot_magnetic_energy': False, # Whether to plot the magnetic energy profile as a reference
}

PROD_DISS_PROFILE_PLOT_PARAMS = {
    'palette_name': PLOT_PALETTES["active"],
    'palettes': PLOT_PALETTES["available"],
    'units': IND_PARAMS["units"],
    # 'it_indx': [-1],
    "it_indx": list(range(0, 900, 50)), # For different redshift snap iterations analysis
    'x_scale': 'log', # 'lin' or 'log'
    'y_scale': 'log', # 'lin' or 'log'
    'xlim': None, # None for auto
    # 'xlim': [1e-17,1e-5], # None for auto
    # 'ylim': None, # None for auto
    'ylim': [1e-17,1e-5], # None for auto
    'figure_size': [12, 8], # [width, height]
    'line_widths': [5.0, 3.0], # [line_main, line_components]
    'plot_type': 'smoothed', # 'raw', 'smoothed', or 'interpolated'
    'smoothing_sigma': 1.1, # sigma for Gaussian smoothing (only for 'smoothed')
    'interpolation_points': 100, # points for interpolation (only for 'interpolated')
    'interpolation_kind': 'cubic', # 'linear', 'cubic', or 'nearest'
    'component_alpha': 0.65, # Opacity for individual P/D component curves (totals remain fully opaque)
    'area_alpha': 0.18, # Opacity for shaded area between production and dissipation component curves
    'fixed_legend': True, # If True, place legend at a fixed position inside axes (best for animations)
    'plot_density': False, # Whether to plot the density profile as a reference
    'plot_magnetic_energy': False, # Whether to plot the magnetic energy profile as a reference
    'plot_net': False, # If True, also plots the net production/dissipation profiles for the different induction components
    'plot_absolute': False, # Plot production/dissipation component profiles
    'title': 'Production and Dissipation Radial Profile',
    'dpi': 300,
    'run': OUTPUT_PARAMS["run"]
}

PERCENTILE_PLOT_PARAMS = {
    'palette_name': PLOT_PALETTES["active"],
    'palettes': PLOT_PALETTES["available"],
    'units': IND_PARAMS["units"],
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
    # Iteration indexes where debug modules (except percentile thresholds) are executed.
    # Uses Python indexing over OUTPUT_PARAMS['it'] per simulation.
    "it_indx": [-1],
    # "it_indx": [0, -1],
    # "it_indx": list(range(len(OUTPUT_PARAMS['it']))),
    # ==== Unigrid Interpolation Mode ====
    "unigrid_interp_mode": 'DIRECT',  # 'DIRECT', 'NGP', or 'TRILINEAR'
                                       # 'DIRECT' (recommended): avoids spurious artifacts from zero-division in base grid
                                       # 'NGP': Nearest Grid Point with position calculations  
                                       # 'TRILINEAR': full trilinear interpolation (may cause artifacts at boundaries)
    # ==== Output Cleaning ====
    # If True, divergence-related debug diagnostics use clean_field(Bx,By,Bz)
    # to remove overlap/refinement duplicated cells before analysis.
    "clean_divergence_fields": True,
    # If True, replace NaN/inf/outliers with zeros in debug visualizations ("sweep under the rug")
    # If False (default), preserve raw data to diagnose errors (recommended for debugging)
    "clean_output": False,
    # ==== Percentile Threshold Debug Parameters ====
    # Parameters for percentile-threshold calculations during data processing
    # (these are copied into PERCENTILE_PLOT_PARAMS for plot labeling/metadata).
    "percentile_params": {
        "exclude_boundaries": False,  # Exclude patch boundary cells from percentile calculation
        "boundary_width": 2,          # Number of boundary cells to exclude (if exclude_boundaries=True)
        "exclude_zeros": False,       # Exclude zero values from percentile calculation
        "use_filtered_divergence": False,  # Reserved (currently raw divergence is used)
    },
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

diff_cfg = IND_PARAMS.get("differentiation", {})
ret_cfg = IND_PARAMS.get("return", {})
pct_cfg = IND_PARAMS.get("percentiles", {})

if diff_cfg.get("stencil") not in [3, 5]:
    raise ValueError("Invalid differentiation stencil value. It must be either 3 or 5.")
elif diff_cfg.get("stencil") == 3:
    IND_PARAMS["differentiation"]["nghost"] = 1
elif diff_cfg.get("stencil") == 5:
    IND_PARAMS["differentiation"]["nghost"] = 2

# Validate energy evolution configuration
energy_evo = IND_PARAMS.get("energy_evolution", {})
if not isinstance(energy_evo, dict):
    raise ValueError("IND_PARAMS['energy_evolution'] must be a dictionary")

allowed_derivative = ['RK', 'implicit_forward', 'central', 'alpha_fit', 'rate']
allowed_volume_coordinates = ['physical', 'comoving']

if energy_evo.get("derivative") not in allowed_derivative:
    raise ValueError("IND_PARAMS['energy_evolution']['derivative'] must be one of: RK, implicit_forward, central, alpha_fit, rate.")
if energy_evo.get("volume_coordinates") not in allowed_volume_coordinates:
    raise ValueError("IND_PARAMS['energy_evolution']['volume_coordinates'] must be 'physical' or 'comoving'.")
if not isinstance(energy_evo.get("normalize_by_volume"), bool):
    raise ValueError("IND_PARAMS['energy_evolution']['normalize_by_volume'] must be a boolean.")
if not isinstance(energy_evo.get("normalized", True), bool):
    raise ValueError("IND_PARAMS['energy_evolution']['normalized'] must be a boolean (True or False).")
if not isinstance(energy_evo.get("plot_profiles", False), bool):
    raise ValueError("IND_PARAMS['energy_evolution']['plot_profiles'] must be a boolean (True or False).")
if not isinstance(energy_evo.get("plot_total", True), bool):
    raise ValueError("IND_PARAMS['energy_evolution']['plot_total'] must be a boolean (True or False).")
if not isinstance(energy_evo.get("plot_differential", True), bool):
    raise ValueError("IND_PARAMS['energy_evolution']['plot_differential'] must be a boolean (True or False).")
if energy_evo.get("enabled", True) and not (
    energy_evo.get("plot_total", True)
    or energy_evo.get("plot_differential", True)
    or energy_evo.get("plot_profiles", False)
):
    raise ValueError(
        "IND_PARAMS['energy_evolution'] is enabled but no output is selected. "
        "Set at least one of 'plot_total', 'plot_differential', or 'plot_profiles' to True."
    )

# Validate production/dissipation plotting units mode
prod_diss_cfg = IND_PARAMS.get("production_dissipation", {})
if not isinstance(prod_diss_cfg.get("normalized", True), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['normalized'] must be a boolean (True or False).")
if prod_diss_cfg.get("volume_coordinates", "physical") not in allowed_volume_coordinates:
    raise ValueError("IND_PARAMS['production_dissipation']['volume_coordinates'] must be 'physical' or 'comoving'.")
if not isinstance(prod_diss_cfg.get("normalize_by_volume", False), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['normalize_by_volume'] must be a boolean (True or False).")
if not isinstance(prod_diss_cfg.get("plot_profiles", False), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['plot_profiles'] must be a boolean (True or False).")
if not isinstance(prod_diss_cfg.get("plot_absolute", True), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['plot_absolute'] must be a boolean (True or False).")
if not isinstance(prod_diss_cfg.get("plot_fractional", True), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['plot_fractional'] must be a boolean (True or False).")
if not isinstance(prod_diss_cfg.get("plot_net", True), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['plot_net'] must be a boolean (True or False).")
if not isinstance(prod_diss_cfg.get("plot_fractional_profiles", False), bool):
    raise ValueError("IND_PARAMS['production_dissipation']['plot_fractional_profiles'] must be a boolean (True or False).")

IND_PARAMS["energy_evolution"]["plot_profiles"] = bool(energy_evo.get("plot_profiles", False))
IND_PARAMS["production_dissipation"]["plot_profiles"] = bool(prod_diss_cfg.get("plot_profiles", False))
IND_PARAMS["production_dissipation"]["plot_fractional_profiles"] = bool(prod_diss_cfg.get("plot_fractional_profiles", False))

# Validate consistency: cannot request plots/returns for disabled subsystems
if not energy_evo.get("enabled", True) and energy_evo.get("plot_profiles", False):
    raise ValueError(
        "Error: energy_evolution['plot_profiles'] = True but energy_evolution['enabled'] = False. "
        "Cannot request profiles for disabled subsystem."
    )

if not prod_diss_cfg.get("enabled", True) and prod_diss_cfg.get("plot_profiles", False):
    raise ValueError(
        "Error: production_dissipation['plot_profiles'] = True but production_dissipation['enabled'] = False. "
        "Cannot request profiles for disabled subsystem."
    )
if not prod_diss_cfg.get("enabled", True) and prod_diss_cfg.get("plot_fractional_profiles", False):
    raise ValueError(
        "Error: production_dissipation['plot_fractional_profiles'] = True but production_dissipation['enabled'] = False. "
        "Cannot request fractional profiles for disabled subsystem."
    )

# Determine if there's actually something to do (coupling: nothing computed unless enabled + something to plot/return)
# Energy evolution is truly needed if enabled AND has any plot enabled
energy_evolution_truly_enabled = (
    energy_evo.get("enabled", True) and 
    (energy_evo.get("plot_total", True) or 
     energy_evo.get("plot_differential", True) or 
     energy_evo.get("plot_profiles", False))
)
IND_PARAMS["energy_evolution"]["_truly_enabled"] = energy_evolution_truly_enabled

# P/D is truly needed if enabled AND has any plot enabled
pd_truly_enabled = (
    prod_diss_cfg.get("enabled", True) and 
    (prod_diss_cfg.get("plot_absolute", True) or 
     prod_diss_cfg.get("plot_fractional", True) or 
     prod_diss_cfg.get("plot_net", True) or 
    prod_diss_cfg.get("plot_profiles", False) or
    prod_diss_cfg.get("plot_fractional_profiles", False))
)
IND_PARAMS["production_dissipation"]["_truly_enabled"] = pd_truly_enabled

# Validate interpolation method and adjust parameters for parent mode
allowed_interpol = ['TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST']
if diff_cfg.get("interpol") not in allowed_interpol:
    raise ValueError("Invalid interpolation method. Must be 'TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST'.")
if diff_cfg.get("parent_interpol", diff_cfg.get("interpol")) not in allowed_interpol:
    raise ValueError("Invalid parent_interpol method. Must be 'TSC', 'SPH', 'LINEAR', 'TRILINEAR', 'NEAREST'.")
IND_PARAMS["differentiation"]["parent_interpol"] = diff_cfg.get("parent_interpol", diff_cfg.get("interpol"))

# If parent mode is selected, enforce compatible parameters
if diff_cfg.get("parent", False) is True:
    blend_enabled = diff_cfg.get("blend", False)
    method_name = 'PARENT' + ' + ' + IND_PARAMS["differentiation"]["parent_interpol"]
    interp_desc = ''
    if IND_PARAMS["differentiation"]["parent_interpol"] == 'TSC':
        interp_desc = 'TSC (smooth Triangular-Shaped Cloud)'
    elif IND_PARAMS["differentiation"]["parent_interpol"] == 'TRILINEAR':
        interp_desc = 'Trilinear (linear interpolation)'
    elif IND_PARAMS["differentiation"]["parent_interpol"] == 'NEAREST':
        interp_desc = 'Nearest Neighbor (discontinuous)'
    elif IND_PARAMS["differentiation"]["parent_interpol"] == 'LINEAR':
        interp_desc = 'Linear (linear interpolation)'
    elif IND_PARAMS["differentiation"]["parent_interpol"] == 'SPH':
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

# AMR-aware multiplier using decreasing refined volume per level.
# This avoids unrealistic full-box 8^L scaling while keeping a conservative safety margin.
base_refined_fraction_l1 = 0.08  # Typical refined volume fraction at level 1
refined_fraction_decay = 0.55    # Refined fraction decreases with level
refinement_multiplier = 1.0
cum_refined_fraction = 1.0

for level in range(1, max_level + 1):
    level_refined_fraction = base_refined_fraction_l1 * (refined_fraction_decay ** (level - 1))
    cum_refined_fraction *= level_refined_fraction
    refinement_multiplier += (8 ** level) * cum_refined_fraction

# Additional AMR complexity correction from patch budget (npalev, patch size, max level).
max_npalev = max(IND_PARAMS["npalev"]) if isinstance(IND_PARAMS["npalev"], list) else IND_PARAMS["npalev"]
max_namrx = max(IND_PARAMS["namrx"]) if isinstance(IND_PARAMS["namrx"], list) else IND_PARAMS["namrx"]
max_namry = max(IND_PARAMS["namry"]) if isinstance(IND_PARAMS["namry"], list) else IND_PARAMS["namry"]
max_namrz = max(IND_PARAMS["namrz"]) if isinstance(IND_PARAMS["namrz"], list) else IND_PARAMS["namrz"]
patch_cell_ratio = (max_npalev * max_namrx * max_namry * max_namrz) / max(base_cells, 1)
amr_complexity_multiplier = 1.0 + 3.0 * np.log10(1.0 + patch_cell_ratio) * (1.0 + 0.15 * max_level)

# Keep estimate in a realistic but conservative range.
refinement_multiplier = min(max(refinement_multiplier, amr_complexity_multiplier, 1.0), 40.0)

single_array_gb = array_size_bytes / (1024 ** 3)
total_snapshots = OUTPUT_PARAMS.get("total_iterations", sum(len(it_list) for it_list in OUTPUT_PARAMS["it"]))

induction_component_keys = ["divergence", "compression", "stretching", "advection", "drag", "total"]
enabled_components = sum(bool(IND_PARAMS["components"].get(k, False)) for k in induction_component_keys)

# Worker memory model: arrays live concurrently during process_iteration in each subprocess.
worker_arrays = 10  # Core loaded fields + masks + geometry helpers
if diff_cfg.get("buffer", False):
    worker_arrays += 8
if diff_cfg.get("parent", False):
    worker_arrays += 3
if diff_cfg.get("blend", False):
    worker_arrays += 4
worker_arrays += max(4, 2 * enabled_components)

if IND_PARAMS.get("energy_evolution", {}).get("enabled", False):
    worker_arrays += 6
if IND_PARAMS.get("energy_evolution", {}).get("plot_profiles", False):
    worker_arrays += 4
if IND_PARAMS.get("production_dissipation", {}).get("plot_profiles", False):
    worker_arrays += 4
if IND_PARAMS.get("production_dissipation", {}).get("enabled", False):
    worker_arrays += 4
if pct_cfg.get("enabled", False):
    worker_arrays += 3
if ret_cfg.get("projection", False):
    worker_arrays += 8
if ret_cfg.get("mag", False):
    worker_arrays += 3

if DEBUG_PARAMS.get("field_analysis", {}).get("enabled", False):
    worker_arrays += 10
if DEBUG_PARAMS.get("scan_animation", {}).get("enabled", False):
    worker_arrays += 6
if DEBUG_PARAMS.get("buffer", {}).get("enabled", False) or DEBUG_PARAMS.get("divergence", {}).get("enabled", False):
    worker_arrays += 6

# Retained memory model in parent process (accumulated across snapshots).
retained_amr_arrays_per_snapshot = 0
if ret_cfg.get("return_vectorial", False):
    retained_amr_arrays_per_snapshot += max(6, enabled_components)
if ret_cfg.get("return_induction", False):
    retained_amr_arrays_per_snapshot += max(6, enabled_components)
if ret_cfg.get("return_induction_energy", False):
    retained_amr_arrays_per_snapshot += max(7, enabled_components + 1)
if ret_cfg.get("projection", False):
    retained_amr_arrays_per_snapshot += 3
if ret_cfg.get("mag", False):
    retained_amr_arrays_per_snapshot += 2
if DEBUG_PARAMS.get("field_analysis", {}).get("enabled", False):
    retained_amr_arrays_per_snapshot += 4
if DEBUG_PARAMS.get("scan_animation", {}).get("enabled", False):
    retained_amr_arrays_per_snapshot += 2

retained_meta_gb_per_snapshot = 0.01
if IND_PARAMS.get("energy_evolution", {}).get("enabled", False):
    retained_meta_gb_per_snapshot += 0.01
if IND_PARAMS.get("production_dissipation", {}).get("enabled", False):
    retained_meta_gb_per_snapshot += 0.01
if IND_PARAMS.get("energy_evolution", {}).get("plot_profiles", False):
    retained_meta_gb_per_snapshot += 0.02
if IND_PARAMS.get("production_dissipation", {}).get("plot_profiles", False):
    retained_meta_gb_per_snapshot += 0.02
if pct_cfg.get("enabled", False):
    retained_meta_gb_per_snapshot += 0.01

estimated_worker_memory_gb_raw = single_array_gb * refinement_multiplier * worker_arrays
estimated_retained_memory_gb_raw = total_snapshots * (
    single_array_gb * refinement_multiplier * retained_amr_arrays_per_snapshot + retained_meta_gb_per_snapshot
)

memory_safety_factor = max(1.0, float(OUTPUT_PARAMS.get("memory_safety_factor", 1.0)))
estimated_worker_memory_gb = estimated_worker_memory_gb_raw * memory_safety_factor
estimated_retained_memory_gb = estimated_retained_memory_gb_raw * memory_safety_factor

print(f"\n{'='*60}")
print(f"MEMORY ESTIMATION")
print(f"{'='*60}")
print(f"Base grid size: {IND_PARAMS['nmax'][max_nmax_index]} x {IND_PARAMS['nmay'][max_nmay_index]} x {IND_PARAMS['nmaz'][max_nmaz_index]}")
print(f"Single array size: {single_array_gb:.3f} GB")
print(f"Worker concurrent arrays (estimated): {worker_arrays}")
print(f"Retained AMR arrays per snapshot (estimated): {retained_amr_arrays_per_snapshot}")
print(f"AMR refinement multiplier: x{refinement_multiplier:.2f}")
print(
    "Workload profile (current config): "
    f"buffer={IND_PARAMS.get('buffer', False)}, "
    f"parent={IND_PARAMS.get('parent', False)}, "
    f"blend={IND_PARAMS.get('blend', False)}, "
    f"induction_profiles={IND_PARAMS.get('energy_evolution', {}).get('plot_profiles', False)}, "
    f"pd_profiles={IND_PARAMS.get('production_dissipation', {}).get('plot_profiles', False)}, "
    f"percentiles={IND_PARAMS.get('percentiles', False)}, "
    f"projection={IND_PARAMS.get('projection', False)}, "
    f"debug_field_analysis={DEBUG_PARAMS.get('field_analysis', {}).get('enabled', False)}, "
    f"debug_scan_animation={DEBUG_PARAMS.get('scan_animation', {}).get('enabled', False)}"
)
print("Note: RAM estimates are scenario-specific and valid for the current configuration profile.")
if memory_safety_factor > 1.0:
    print(f"Empirical safety factor: x{memory_safety_factor:.2f}")
    print(f"Estimated worker memory (raw): {estimated_worker_memory_gb_raw:.2f} GB")
    print(f"Estimated retained memory (raw): {estimated_retained_memory_gb_raw:.2f} GB")
print(f"Estimated worker memory per subprocess: {estimated_worker_memory_gb:.2f} GB")
print(f"Estimated retained memory across snapshots: {estimated_retained_memory_gb:.2f} GB")

# Decide on parallelization strategy
print(f"\n{'='*60}")
print(f"PARALLELIZATION CONFIGURATION")
print(f"{'='*60}")

# Automatic recommendation based on memory and CPU availability
recommended_parallel = False
recommended_ncores = OUTPUT_PARAMS["ncores"]  # Default from config
mem_ratio = estimated_worker_memory_gb / max(ram_capacity_gb, 1e-12)

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
max_parallel_workers = min(available_cores, total_snapshots)
estimated_peak_serial_gb = estimated_retained_memory_gb + estimated_worker_memory_gb
estimated_peak_parallel_gb = estimated_retained_memory_gb + estimated_worker_memory_gb * max_parallel_workers
peak_mem_ratio = (estimated_peak_parallel_gb / ram_capacity_gb)

# Expose estimator internals for runtime calibration/reporting in main.py
OUTPUT_PARAMS["estimated_worker_memory_gb"] = estimated_worker_memory_gb
OUTPUT_PARAMS["estimated_retained_memory_gb"] = estimated_retained_memory_gb
OUTPUT_PARAMS["estimated_peak_serial_gb"] = estimated_peak_serial_gb
OUTPUT_PARAMS["estimated_peak_parallel_gb"] = estimated_peak_parallel_gb
OUTPUT_PARAMS["estimated_parallel_workers"] = max_parallel_workers
OUTPUT_PARAMS["estimated_peak_mem_ratio"] = peak_mem_ratio

print(f"Estimated peak RAM (serial): {estimated_peak_serial_gb:.2f} GB")
print(f"Estimated peak RAM (parallel x{max_parallel_workers} workers): {estimated_peak_parallel_gb:.2f} GB")
print(f"Estimated peak memory ratio (parallel): {100 * peak_mem_ratio:.1f}% of total RAM")

# High memory usage (>40% RAM): recommend parallelization to distribute load across iterations
if peak_mem_ratio >= 0.60:
    print(f"⚠️  Extreme memory risk detected ({estimated_peak_parallel_gb:.2f} GB ≥ 60% of {ram_capacity_gb:.2f} GB)")
    print(f"   Recommendation: Prefer SERIAL or minimal parallelism to avoid OOM/segfaults")
    recommended_parallel = available_cores > 1 and total_snapshots > 1
    recommended_ncores = min(available_cores, max(1, total_snapshots // 2))
    print(f"   Reserved cores: {reserved_cores}; Recommended cores: {recommended_ncores}")
elif peak_mem_ratio >= 0.40:
    print(f"⚠️  High memory usage detected ({estimated_peak_parallel_gb:.2f} GB > 40% of {ram_capacity_gb:.2f} GB)")
    print(f"   Recommendation: Use parallel processing to handle multiple iterations efficiently")
    recommended_parallel = True
    recommended_ncores = min(available_cores, total_snapshots)  # Don't exceed number of snapshots
    print(f"   Reserved cores: {reserved_cores}; Recommended cores: {recommended_ncores}")
    
# Low memory, multiple iterations: can benefit from parallelization
elif total_snapshots > 1:
    print(f"✓ Memory usage is manageable ({estimated_peak_parallel_gb:.2f} GB < 40% of {ram_capacity_gb:.2f} GB)")
    print(f"  {total_snapshots} snapshots to process")
    recommended_ncores = min(available_cores, total_snapshots)
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
# If only 1 core is recommended, automatically use serial mode
if recommended_parallel and recommended_ncores == 1:
    recommended_parallel = False

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