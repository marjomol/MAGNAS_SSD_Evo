"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

config module
Defines configuration parameters for the simulation, including seed parameters and output parameters.

Created by Marco Molina Pradillo
"""

import numpy as np
import os
import psutil
from scripts.units import a0_masclet, H0_masclet, omega_lambda, omega_k, omega_m

# ============================
# Only edit the section below
# ============================

# Seed Parameters #

# Some tabulated values for the magnetic field amplitude and corresponding spectral index are:
# B0 =    [   2,    1.87, 0.35, 0.042, 0.003]
# alpha = [-2.9,      -1,    0,     1,     2]

SEED_PARAMS = {
    "nmax": 256,
    "nmay": 256,
    "nmaz": 256,
    "size": 40, # Size of the box in Mpc
    "B0": 2, # Initial magnetic field amplitude in Gauss
    "alpha": -2.9, # Spectral index
    "lambda_comoving": 1.0, # Comoving smoothing length
    "smothing": 1,
    "filtering": True,
    "epsilon": 1e-30,
    "npalev": 13000,
    "nlevels": 7,
    "namrx": 32,
    "namry": 32,
    "namrz": 32,
    "a0": a0_masclet,
    "H0": H0_masclet,
    "zeta": 100,
}

# Directories and Results Parameters #

OUTPUT_PARAMS = {
    "save": True,
    "chunk_factor": 4,
    "bitformat": np.float32,
    "format": "fortran",
    "dpi": 300,
    "verbose": True,
    "debug": False,
    "run": f'PRIMAL_Seed_Gen_norm',
    "outdir": "/scratch/molina/output_files_PRIMAL_",
    "plotdir": "plots/",
    "rawdir": "raw_data/",
    "ID1": "seed/",
    "ID2": "norm",
    "random_seed": 23 # Set the random seed for reproducibility
}

# ============================
# Only edit the section above
# ============================

## Some seed parameters are calculated from the previous ones

size = SEED_PARAMS["size"]
nmax = SEED_PARAMS["nmax"]
a0 = SEED_PARAMS["a0"]
H0 = SEED_PARAMS["H0"]
zeta = SEED_PARAMS["zeta"]

dx = size/nmax # Size of the cells in Mpc
volume = size**3 # (Mpc)^3

a = a0 / (1 + zeta)
E = (omega_lambda + omega_k/a**2 + omega_m/a**3)**(1/2)
H = H0*E

SEED_PARAMS["dx"] = dx
SEED_PARAMS["volume"] = volume
SEED_PARAMS["a"] = a
SEED_PARAMS["E"] = E
SEED_PARAMS["H"] = H

## The output parameters are used to create the image directories and other formatting parameters

outdir = OUTPUT_PARAMS["outdir"]
plotdir = OUTPUT_PARAMS["plotdir"]
rawdir = OUTPUT_PARAMS["rawdir"]
ID1 = OUTPUT_PARAMS["ID1"]
ID2 = OUTPUT_PARAMS["ID2"]

# We create the folder for the plots and data
image_folder = outdir + plotdir + ID1 + f'PRIMAL_Seed_{ID2}'
data_folder = outdir + rawdir + ID1 + f'PRIMAL_Seed_{ID2}'

# List of folders to check
folders = [image_folder, data_folder]

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

# Determine the format of the output files
if OUTPUT_PARAMS["bitformat"] == np.float32:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex64
elif OUTPUT_PARAMS["bitformat"] == np.float64:
    OUTPUT_PARAMS["complex_bitformat"] = np.complex128

## Check if the arrays can fit in memory or if an alternative memory handeling method is needed

# Get the total RAM capacity in bytes
ram_capacity = psutil.virtual_memory().total

# Convert the RAM capacity to gigabytes
ram_capacity_gb = ram_capacity / (1024 ** 3)

print(f"Total RAM capacity: {ram_capacity_gb:.2f} GB")

# Calculate the size of the arrays in bytes
array_size = SEED_PARAMS["nmax"] * SEED_PARAMS["nmay"] * SEED_PARAMS["nmaz"]
array_size_bytes =  array_size * np.dtype(OUTPUT_PARAMS["complex_bitformat"]).itemsize

print(f"Maximum size of the arrays involved: {array_size_bytes / (1024 ** 3):.2f} GB")

# Check if the arrays will use more than 1/4 of the total RAM
if array_size_bytes > (2*ram_capacity / 5):
    memmap = True
    transform = False
    print("The arrays are too large to fit in memory: the seed will not be transformed")
    print(" - Chunking will be used")
    print(" - Memory mapping will be used")
    print(" - The seed will NOT be transformed to real space")
elif array_size_bytes > (ram_capacity / 4):
    memmap = True
    transform = True
    print("The arrays are too large to fit in memory:")
    print(" - Chunking will be used")
    print(" - Memory mapping will be used")
else:
    memmap = False
    # memmap = True # Uncomment this line to force the use of np.memmap
    transform = True
    print("The arrays can fit in memory: chunking will not be used.")
    print(" - Chunking will NOT be used")
    print(" - Memory mapping will NOT be used")

OUTPUT_PARAMS["memmap"] = memmap
OUTPUT_PARAMS["transform"] = transform