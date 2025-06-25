"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

readers module
Provides the necessary functions for reading MASCLET and ASOHF files and loading them in memory

Funtions created by David Vallés for MASCLET framework
"""

import os
import json
import numpy as np

def filename(it, filetype, digits=5):
    """
    Generates filenames for MASCLET output files

    Args:
        it: iteration number (int)
        filetype: 'g' for grids file; 'b' for gas file (baryonic); 'd' for dark matter (dm) (str)
        digits: number of digits the filename is written with (int)

    Returns: filename (str)

    """
    names = {'g': "grids", 'b': 'clus', 'd': 'cldm', 's': 'clst', 'v': 'velocity', 'm': 'MachNum_', 'f': 'filtlen_'}
    try:
        if it == 0:
            return names[filetype] + str(it).zfill(digits)
        elif np.floor(np.log10(it)) < digits:
            return names[filetype] + str(it).zfill(digits)
        else:
            raise ValueError("Digits should be greater to handle that iteration number")
    except KeyError:
        print('Insert a correct type: g, b, d, s, v, f or m')

def read_parameters_file(filename='masclet_parameters.json', path=''):
    """
    Returns dictionary containing the MASCLET parameters of the simulation, that have been previously written with the
    write_parameters() function in this same module.

    Args:
        filename: name of the MASCLET parameters file (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns:
        dictionary containing the parameters (and their names), namely:
        NMAX, NMAY, NMAZ: number of l=0 cells along each direction (int)
        NPALEV: maximum number of refinement cells per level (int)
        NLEVELS: maximum number of refinement level (int)
        NAMRX (NAMRY, NAMRZ): maximum size of refinement patches (in l-1 cell units) (int)
        SIZE: side of the simulation box in the chosen units (typically Mpc or kpc) (float)

    """
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def read_parameters(filename='masclet_parameters.json', path='', load_nma=True, load_npalev=True, load_nlevels=True,
                    load_namr=True, load_size=True):
    """
    Returns MASCLET parameters in the old-fashioned way (as a tuple).
    Legacy (can be used, but newer codes should try to switch to directly reading the dictionary with
    read_parameters_file() function).

    Args:
        filename: name of the MASCLET parameters file (str)
        path: path of the file (typically, the codename of the simulation) (str)
        load_nma: whether NMAX, NMAY, NMAZ are read (bool)
        load_npalev: whether NPALEV is read (bool)
        load_nlevels: whether NLEVELS is read (bool)
        load_namr: whether NAMRX, NAMRY, NAMRZ is read (bool)
        load_size: whether SIZE is read (bool)

    Returns:
        tuple containing, in this exact order, the chosen parameters from:
        NMAX, NMAY, NMAZ: number of l=0 cells along each direction (int)
        NPALEV: maximum number of refinement cells per level (int)
        NLEVELS: maximum number of refinement level (int)
        NAMRX (NAMRY, NAMRZ): maximum size of refinement patches (in l-1 cell units) (int)
        SIZE: side of the simulation box in the chosen units (typically Mpc or kpc) (float)

    """
    parameters = read_parameters_file(filename=filename, path=path)
    returnvariables = []
    if load_nma:
        returnvariables.extend([parameters[i] for i in ['NMAX', 'NMAY', 'NMAZ']])
    if load_npalev:
        returnvariables.append(parameters['NPALEV'])
    if load_nlevels:
        returnvariables.append(parameters['NLEVELS'])
    if load_namr:
        returnvariables.extend([parameters[i] for i in ['NAMRX', 'NAMRY', 'NAMRZ']])
    if load_size:
        returnvariables.append(parameters['SIZE'])
    return tuple(returnvariables)


def write_parameters(nmax, nmay, nmaz, npalev, nlevels, namrx, namry, namrz,
                    size, filename='masclet_parameters.json', path=''):
    """
    Creates a JSON file containing the parameters of a certain simulation

    Args:
        nmax: number of l=0 cells along the X-direction (int)
        nmay: number of l=0 cells along the Y-direction (int)
        nmaz: number of l=0 cells along the Z-direction (int)
        npalev: maximum number of refinement cells per level (int)
        nlevels: maximum number of refinement level (int)
        namrx: maximum X-size of refinement patches (in l-1 cell units) (int)
        namry: maximum Y-size of refinement patches (in l-1 cell units) (int)
        namrz: maximum Z-size of refinement patches (in l-1 cell units) (int)
        size: side of the simulation box in the chosen units (typically Mpc or kpc) (float)
        filename: name of the MASCLET parameters file to be saved (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns: nothing. A file is created in the specified path
    """
    parameters = {'NMAX': nmax, 'NMAY': nmay, 'NMAZ': nmaz,
                'NPALEV': npalev, 'NLEVELS': nlevels,
                'NAMRX': namrx, 'NAMRY': namry, 'NAMRZ': namrz,
                'SIZE': size}

    with open(os.path.join(path,filename), 'w') as json_file:
        json.dump(parameters, json_file)

def read_grids(it, path='', parameters_path='', digits=5, read_general=True, read_patchnum=True, read_dmpartnum=True,
                read_patchcellextension=True, read_patchcellposition=True, read_patchposition=True,
                read_patchparent=True, nparray=True):
    """
    reads grids files, containing the information needed for building the AMR structure

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation (str)
        digits: number of digits the filename is written with (int)
        read_general: whether irr, T, NL, MAP and ZETA are returned (bool)
        read_patchnum: whether NPATCH is returned (bool)
        read_dmpartnum: whether NPART is returned (bool)
        read_patchcellextension: whether PATCHNX, PATCHNY, PATCHNZ are returned (bool)
        read_patchcellposition: whether PATCHX, PATCHY, PATCHZ are returned (bool)
        read_patchposition: whether PATCHRX, PATCHRY, PATCHRZ are returned (bool)
        read_patchparent: whether PARENT is returned (bool)
        nparray: if True (default), all variables are returned as numpy arrays (bool)

    Returns: (in order)

        -only if readgeneral set to True
        irr: iteration number
        t: time
        nl: num of refinement levels
        mass_dmpart: mass of DM particles
        zeta: redshift

        -only if readpatchnum set to True
        npatch: number of patches in each level, starting in l=0

        -only if readdmpartnum set to True
        npart: number of dm particles in each leve, starting in l=0

        -only if readpatchcellextension set to True
        patchnx (...): x-extension of each patch (in level l cells) (and Y and Z)

        -only if readpatchcellposition set to True
        patchx (...): x-position of each patch (left-bottom-front corner; in level
        l-1 cells) (and Y and Z)
        CAUTION!!! IN THE OUTPUT, FIRST CELL IS 1. HERE, WE SET IT TO BE 0. THUS, PATCHNX's READ HERE WILL BE LOWER IN
        A UNIT FROM THE ONE WRITTEN IN THE FILE.

        -only if readpatchposition set to True
        patchrx (...): physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)

        -only if readpatchparent set to True
        pare: which (l-1)-cell is left-bottom-front corner of each patch in

    """
    nmax, nmay, nmaz, size = read_parameters(load_nma=True, load_npalev=False, load_nlevels=False,
                                                        load_namr=False, load_size=True, path=parameters_path)
    rx = - size / 2 + size / nmax

    grids = open(os.path.join(path, filename(it, 'g', digits)), 'r')

    # first, we load some general parameters
    irr, t, nl, mass_dmpart, _ = tuple(float(i) for i in grids.readline().split())
    irr = int(irr)
    # assert (it == irr)
    nl = int(nl)
    zeta = float(grids.readline().split()[0])
    # l=0
    _, ndxyz, _ = tuple(float(i) for i in grids.readline().split())[0:3]
    ndxyz = int(ndxyz)

    # vectors where the data will be stored
    npatch = [0]  # number of patches per level, starting with l=0
    npart = [ndxyz]  # number of dm particles per level, starting with l=0
    patchnx = [nmax]
    patchny = [nmay]
    patchnz = [nmaz]
    patchx = [0]
    patchy = [0]
    patchz = [0]
    patchrx = [rx]
    patchry = [rx]
    patchrz = [rx]
    pare = [0]

    for ir in range(1, nl + 1):
        level, npatchtemp, nparttemp = tuple(int(i) for i in grids.readline().split())[0:3]
        npatch.append(npatchtemp)
        npart.append(nparttemp)

        # ignoring a blank line
        grids.readline()

        # loading all values
        for i in range(sum(npatch[0:ir]) + 1, sum(npatch[0:ir + 1]) + 1):
            this_nx, this_ny, this_nz = tuple(int(i) for i in grids.readline().split())
            this_x, this_y, this_z = tuple(int(i) for i in grids.readline().split())
            this_rx, this_ry, this_rz = tuple(float(i) for i in grids.readline().split())
            this_pare = int(grids.readline())
            patchnx.append(this_nx)
            patchny.append(this_ny)
            patchnz.append(this_nz)
            patchx.append(this_x - 1)
            patchy.append(this_y - 1)
            patchz.append(this_z - 1)
            patchrx.append(this_rx)
            patchry.append(this_ry)
            patchrz.append(this_rz)
            pare.append(this_pare)

    # converts everything into numpy arrays if nparray set to True
    if nparray:
        npatch = np.array(npatch)
        npart = np.array(npart)
        patchnx = np.array(patchnx)
        patchny = np.array(patchny)
        patchnz = np.array(patchnz)
        patchx = np.array(patchx)
        patchy = np.array(patchy)
        patchz = np.array(patchz)
        patchrx = np.array(patchrx)
        patchry = np.array(patchry)
        patchrz = np.array(patchrz)
        pare = np.array(pare)

    grids.close()

    returnvariables = []

    if read_general:
        returnvariables.extend([irr, t, nl, mass_dmpart, zeta])
    if read_patchnum:
        returnvariables.append(npatch)
    if read_dmpartnum:
        returnvariables.append(npart)
    if read_patchcellextension:
        returnvariables.extend([patchnx, patchny, patchnz])
    if read_patchcellposition:
        returnvariables.extend([patchx, patchy, patchz])
    if read_patchposition:
        returnvariables.extend([patchrx, patchry, patchrz])
    if read_patchparent:
        returnvariables.append(pare)

    return tuple(returnvariables)