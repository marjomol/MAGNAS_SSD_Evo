"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

config module
Defines configuration parameters for the simulation, including seed parameters and output parameters.

Created by Marco Molina Pradillo
"""

import os
from scripts.units import a0_masclet, H0_masclet, omega_lambda, omega_k, omega_m

# ============================
# Only edit the section below
# ============================

# Seed Parameters #

# Some tabulated values for the magnetic field amplitude and corresponding spectral index are:
# B0 = [2, 1.87, 0.35, 0.042, 0.003]
# alpha = [-2.9, -1, 0, 1, 2]

SEED_PARAMS = {
    "nmax": 1024,
    "nmay": 1024,
    "nmaz": 1024,
    "size": 40, # Size of the box in Mpc
    "B0": [2], # Initial magnetic field amplitude in Gauss
    "alpha": [-2.9], # Spectral index
    "lambda_comoving": 1.0, # Comoving smoothing length
    "smothing": 1,
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
    "dpi": 300,
    "outdir": "/home/marcomol/trabajo/data/out/",
    "plotdir": "plots/",
    "rawdir": "raw_data_out/",
    "ID1": "seed/",
    "ID2": "optimized",
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

## The output parameters are used to create the image directories

outdir = OUTPUT_PARAMS["outdir"]
plotdir = OUTPUT_PARAMS["plotdir"]
rawdir = OUTPUT_PARAMS["rawdir"]
ID1 = OUTPUT_PARAMS["ID1"]
ID2 = OUTPUT_PARAMS["ID2"]

# We create the folder for the plots
image_folder = outdir + plotdir + ID1 + f'PRIMAL_Seed_{ID2}'

# List of folders to check
folders = [image_folder]

for folder in folders:
    # Check if the directory already exists
    if os.path.exists(folder):
        # If it exists, exit the loop
        pass
    else:
        # If it doesn't exist, create the directory
        os.makedirs(folder) 

OUTPUT_PARAMS["image_folder"] = image_folder