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
from scripts.units import a0_masclet, H0_masclet, omega_lambda, omega_k, omega_m
from scripts.readers import write_parameters

# ============================
# Only edit the section below
# ============================

# Induction Parameters #

IND_PARAMS = {
    "nmax": [128],
    "nmay": [128],
    "nmaz": [128],
    "size": [40], # Size of the box in Mpc
    "units": [1], # Factor to convert the units of the resulting volume integrals
    "npalev": [13000],
    "nlevels": [7],
    "namrx": [32],
    "namry": [32],
    "namrz": [32],
    "nbins": [25], # Number of bins for the profiles histograms
    "rmin": [0.01], # Minimum radius to calculate the profiles
    "logbins": True, # Use logarithmic bins
    "F": 1.0, # Factor to multiply the viral radius to define the box size
    "vir_kind": 1, # 1: Reference virial radius at the last snap, 2: Reference virial radius at each epoch
    "rad_kind": 1, # 1: Comoving, 2: Physical
    "level": 100, # Max. level of the AMR grid to be used
    "up_to_level": 3, # AMR level up to which calculate
    "region": 'BOX', # Region of interest to calculate the induction components (BOX, SPH, or None)
    "a0": a0_masclet,
    "H0": H0_masclet,
    "zeta": 100,
    "epsilon": 1e-30,
    "divergence": False, # Process the divergence induction component
    "compression": True, # Process the compression induction component
    "stretching": False, # Process the stretching induction component
    "advection": False, # Process the advection induction component
    "drag": False, # Process the drag induction component
    "total": False, # Process the total induction component
    "mag": False, # Calculate magnetic induction components magnitudes
    "energy_evolution": True, # Calculate the evolution of the energy budget
    "evolution_type": 'total', # Type of evolution to calculate (total or differential)
    "derivative": 'central', # Derivative to use for the evolution (implicit or central)
    "profiles": False, # Calculate the profiles of the induction components
    "projection": False, # Calculate the projection of the induction components
    "A2U": True # Transform the AMR grid to a uniform grid
}

# Directories and Results Parameters #

OUTPUT_PARAMS = {
    "save": False,
    "verbose": True,
    "debug": False,
    "chunk_factor": 2,
    "bitformat": np.float32,
    "format": "npy",
    "ncores": 1,
    "Save_Cores": 8, # Number of cores to save for the system (Increase this number if having troubles with the memory when multiprocessing)
    "stencil": 5, # Stencil to calculate the derivatives (either 3 or 5)
    "dpi": 300,
    "run": f'MAGNAS_SSD_Evo_test',
    "sims": ["cluster_B_low_res_paper_2020"], # Simulation names, must match the name of the simulations folder in the data directory
    "it": [1050], # For different redshift snap iterations analysis
    # "it": list(range(1000, 2001, 50)) + [2119], 
    "dir_DM": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_gas": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_grids": "/home/marcomol/trabajo/data/in/scratch/quilis/",
    "dir_halos": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_ASOHF",
    "dir_vortex": "/home/marcomol/trabajo/data/in/scratch/marcomol/output_files_VORTEX",
    # "outdir": "/scratch/marcomol/output_files_PRIMAL_",
    "outdir": "/home/marcomol/trabajo/data/out/",
    "plotdir": "plots/",
    "rawdir": "raw_data_out/",
    "ID1": "dynamo/",
    "ID2": "test",
    "random_seed": 23 # Set the random seed for reproducibility
}

# ============================
# Only edit the section above
# ============================


## The output parameters are used to create the image directories and other formatting parameters

outdir = OUTPUT_PARAMS["outdir"]
plotdir = OUTPUT_PARAMS["plotdir"]
rawdir = OUTPUT_PARAMS["rawdir"]
ID1 = OUTPUT_PARAMS["ID1"]
ID2 = OUTPUT_PARAMS["ID2"]

# We create the folder for the plots and data
image_folder = outdir + plotdir + ID1 + f'MAGNAS_SSD_{ID2}'
data_folder = outdir + rawdir + ID1 + f'MAGNAS_SSD_{ID2}'
parameters_folders = []

# List of folders to check
folders = [image_folder, data_folder]
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
OUTPUT_PARAMS["parameters_folders"] = parameters_folders

# Determine the format of the output files
if OUTPUT_PARAMS["bitformat"] == np.float32:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex64
elif OUTPUT_PARAMS["bitformat"] == np.float64:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex128
    
    
## Some seed parameters are calculated from the previous ones

size = IND_PARAMS["size"]
nmax = IND_PARAMS["nmax"]
a0 = IND_PARAMS["a0"]
H0 = IND_PARAMS["H0"]
zeta = IND_PARAMS["zeta"]

dx = [size[i]/nmax[i] for i in range(len(size))]  # Cell size in Mpc/h

a = a0 / (1 + zeta)
E = (omega_lambda + omega_k/a**2 + omega_m/a**3)**(1/2)
H = H0*E
rho_b = 3 * (H0)**2 * omega_m * (1 + zeta)**3 # We compute the background density at this redshift

volume = [] # (Mpc)^3

for i in range(len(OUTPUT_PARAMS['sims'])):
    
    volume.append(size[i]**3) # (Mpc/h)^3
    write_parameters(IND_PARAMS['nmax'][i], IND_PARAMS['nmay'][i], IND_PARAMS['nmaz'][i],
                    IND_PARAMS['npalev'][i], IND_PARAMS['nlevels'][i], IND_PARAMS['namrx'][i],
                    IND_PARAMS['namry'][i], IND_PARAMS['namrz'][i], size[i], path=parameters_folders[i])
    
IND_PARAMS["dx"] = dx
IND_PARAMS["a"] = a
IND_PARAMS["E"] = E
IND_PARAMS["H"] = H
IND_PARAMS["rho_b"] = rho_b
IND_PARAMS["volume"] = volume
OUTPUT_PARAMS["dir_params"] = parameters_folders

## Inducction components to be checked

IND_PARAMS["components"] = {
    "divergence": IND_PARAMS["divergence"],
    "compression": IND_PARAMS["compression"],
    "stretching": IND_PARAMS["stretching"],
    "advection": IND_PARAMS["advection"],
    "drag": IND_PARAMS["drag"],
    "total": IND_PARAMS["total"]
}

## Check if the arrays can fit in memory or if an alternative memory handeling method is needed

# Get the total RAM capacity in bytes
ram_capacity = psutil.virtual_memory().total

# Convert the RAM capacity to gigabytes
ram_capacity_gb = ram_capacity / (1024 ** 3)

print(f"Total RAM capacity: {ram_capacity_gb:.2f} GB")

max_nmax_index = int(np.argmax(IND_PARAMS["nmax"]))
max_nmay_index = int(np.argmax(IND_PARAMS["nmay"]))
max_nmaz_index = int(np.argmax(IND_PARAMS["nmaz"]))

# Calculate the size of the arrays in bytes
array_size = IND_PARAMS["nmax"][max_nmax_index] * IND_PARAMS["nmay"][max_nmay_index] * IND_PARAMS["nmaz"][max_nmaz_index]
array_size_bytes =  array_size * np.dtype(OUTPUT_PARAMS["bitformat"]).itemsize

print(f"Maximum size of the arrays involved: {array_size_bytes / (1024 ** 3):.2f} GB")

# Check if the arrays will use more than 1/4 of the total RAM
if array_size_bytes > (ram_capacity / 4):
    mem = True
    para = True
    print("The arrays are too large to fit in memory:")
    print(" - Parallel chunking will be used")
    print(" - Arrays will be saved in the raw data folder, NOT as variables")
else:
    print("The arrays can fit in memory:")
    ask = input("Do you want to use chunking? (y/n): ")
    if ask.lower() == 'y':
        mem = True
        print(" - Chunking will be used")
        ask2 = input("Do you want to use parallel chunking? (y/n): ")
        if ask2.lower() == 'y':
            para = True
            print(" - Parallel chunking will be used")
    else:
        mem = False
        para = False
        print(" - Chunking will NOT be used")
            
    if OUTPUT_PARAMS["save"]:
        print(" - Arrays will be saved in the raw data folder, NOT as variables")
    else:
        print(" - Arrays will be saved as variables")

OUTPUT_PARAMS["memory"] = mem
OUTPUT_PARAMS["parallel"] = para