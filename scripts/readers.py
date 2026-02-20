"""
MAGNAS SSD Evolution
A tool to analyse simulated cosmological magnetic field induction and the Small Scale Dynamo amplification.

readers module
Provides the necessary functions for reading MASCLET and ASOHF files and loading them in memory

Funtions created by David Vallés for MASCLET framework
"""

import os
import json
import numpy as np
import utils
from cython_fortran_file import FortranFile as FF
import importlib.util
if (importlib.util.find_spec('tqdm') is None):
    def tqdm(x, desc=None): return x
else:
    from tqdm import tqdm

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
        npart: number of dm particles in each level, starting in l=0

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


def read_clus(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_delta=True, output_v=True,
            output_pres=True, output_pot=True, output_opot=False, output_temp=True, output_metalicity=True,
            output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True,
            verbose=False, use_tqdm=True, read_region=None):
    """
    Reads the gas (baryonic, clus) file

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_delta: whether delta (density contrast) is returned (bool)
        output_v: whether velocities (vx, vy, vz) are returned (bool)
        output_pres: whether pressure is returned (bool)
        output_pot: whether gravitational potential is returned (bool)
        output_opot: whether gravitational potential in the previous iteration is returned (bool)
        output_temp: whether temperature is returned (bool)
        output_metalicity: whether metalicity is returned (bool)
        output_cr0amr: whether "refined variable" (1 if not refined, 0 if refined) is returned (bool)
        output_solapst: whether "solapst variable" (1 if the cell is kept, 0 otherwise) is returned (bool)
        is_mascletB: whether the outputs correspond to masclet-B (contains magnetic fields) (bool)
        output_B: whether magnetic field is returned; only if is_mascletB = True (bool)
        is_cooling: whether there is cooling (an thus T and metalicity are written) or not (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        use_tqdm: whether to use tqdm to show progress bars (bool)
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
                    (None). If a region wants to be selected, there are the following possibilities:
                        - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
                        - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
                        - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"

    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.
        
        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
    """
    if output_B and (not is_mascletB):
        print('Error: cannot output magnetic field if the simulation has not.')
        print('Terminating')
        return

    nmax, nmay, nmaz, nlevels, size = read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                load_namr=False, load_size=True, path=parameters_path)
    npatch, patchnx, patchny, patchnz, \
            patchrx, patchry, patchrz = read_grids(it, path=path, parameters_path=parameters_path, read_general=False,
                                                    read_patchnum=True, read_dmpartnum=False,
                                                    read_patchcellextension=True, read_patchcellposition=False,
                                                    read_patchposition=True, read_patchparent=False)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = utils.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                        patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = utils.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                    patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    with FF(os.path.join(path, filename(it, 'b', digits))) as f:
        # read header
        it_clus = f.read_vector('i')[0]
        # assert(it == it_clus)
        f.seek(0)  # this is a little bit ugly but whatever
        time, z = tuple(f.read_vector('f')[1:3])

        # l=0
        if verbose:
            print('Reading base grid...')
        if output_delta:
            delta = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_v:
            vx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vy = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip(3)

        if output_pres:
            pres = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_pot:
            pot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_opot:
            opot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if is_cooling:
            if output_temp:
                temp = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip()

            if output_metalicity:
                metalicity = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip()

        if output_cr0amr:
            cr0amr = [np.reshape(f.read_vector('i'), (nmax, nmay, nmaz), 'F').astype('bool')]
        else:
            f.skip()

        if output_solapst:
            solapst = [1]

        if is_mascletB:
            if output_B:
                Bx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                By = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                Bz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip(3)


        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches.'.format(npatch[l]))
            for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l), disable=not use_tqdm):
                #if verbose:
                #    print('Reading patch {}'.format(ipatch))
                if output_delta and keep_patches[ipatch]:
                    delta.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                else:
                    f.skip()
                    if output_delta:
                        delta.append(0)

                if output_v and keep_patches[ipatch]:
                    vx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vy.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip(3)
                    if output_v:
                        vx.append(0)
                        vy.append(0)
                        vz.append(0)

                if output_pres and keep_patches[ipatch]:
                    pres.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                else:
                    f.skip()
                    if output_pres:
                        pres.append(0)

                if output_pot and keep_patches[ipatch]:
                    pot.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()
                    if output_pot:
                        pot.append(0)

                if output_opot and keep_patches[ipatch]:
                    opot.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()
                    if output_opot:
                        opot.append(0)

                if is_cooling:
                    if output_temp and keep_patches[ipatch]:
                        temp.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        f.skip()
                        if output_temp:
                            temp.append(0)

                    if output_metalicity and keep_patches[ipatch]:
                        metalicity.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        f.skip()
                        if output_metalicity:
                            metalicity.append(0)

                if output_cr0amr and keep_patches[ipatch]:
                    cr0amr.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F').astype('bool'))
                else:
                    f.skip()
                    if output_cr0amr:
                        cr0amr.append(0)

                if output_solapst and keep_patches[ipatch]:
                    solapst.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F').astype('bool'))
                else:
                    f.skip()
                    if output_solapst:
                        solapst.append(0)

                if is_mascletB:
                    if output_B and keep_patches[ipatch]:
                        Bx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                        By.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                        Bz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                    else:
                        f.skip(3)
                        if output_B:
                            Bx.append(0)
                            By.append(0)
                            Bz.append(0)

    returnvariables = []
    if output_delta:
        returnvariables.append(delta)
    if output_v:
        returnvariables.extend([vx, vy, vz])
    if output_pres:
        returnvariables.append(pres)
    if output_pot:
        returnvariables.append(pot)
    if output_opot:
        returnvariables.append(opot)
    if output_temp:
        returnvariables.append(temp)
    if output_metalicity:
        returnvariables.append(metalicity)
    if output_cr0amr:
        returnvariables.append(cr0amr)
    if output_solapst:
        returnvariables.append(solapst)
    if output_B:
        returnvariables.extend([Bx,By,Bz])

    if read_region is not None:
        returnvariables.append(keep_patches)

    return tuple(returnvariables)


def read_families(it, path='', output_format='dictionaries', output_redshift=False,
                min_mass=None, exclude_subhaloes=False, read_region=None, keep_boundary_contributions=False):
    '''
    Reads the families files, containing the halo catalogue.
    Can be outputted as a list of dictionaries, one per halo
    (output_format='dictionaries') or as a dictionary of 
    arrays (output_format='arrays').

    Args:
        - it: iteration number (int)
        - path: path of the families file (str)
        - output_format: 'dictionaries' or 'arrays'
        - output_redshift: whether to output the redshift of the snapshot, 
            after the halo catalogue (bool)
        - min_mass: minimum mass of the haloes to be output (float)
        - exclude_subhaloes: whether to exclude subhaloes (bool)
        - read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
            (None). If a region wants to be selected, there are the following possibilities:
            - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
            - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
            - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"
        - keep_boundary_contributions: only if read_region is used, whether to keep haloes that might be incomplete (True)
            or not (False).
    '''
    with open(os.path.join(path, 'families{:05d}'.format(it)), 'r') as f:
        _=f.readline()
        _,_,_,zeta = f.readline().split()
        zeta=float(zeta)
        for i in range(5):
            _=f.readline()
        haloes=[]
        for l in f:
            l=l.split()
            halo={}
            halo['id']=int(l[0])
            halo['substructureOf']=int(l[1])
            halo['x']=float(l[2])
            halo['y']=float(l[3])
            halo['z']=float(l[4])
            halo['Mvir']=float(l[5])
            halo['Rvir']=float(l[6])
            if halo['substructureOf']==-1:
                halo['M']=halo['Mvir']
                halo['R']=halo['Rvir']
            else:
                halo['M']=float(l[7])
                halo['R']=float(l[8])
            halo['partNum']=int(l[9])
            halo['mostBoundPart']=int(l[10])
            halo['xcm']=float(l[11])
            halo['ycm']=float(l[12])
            halo['zcm']=float(l[13])
            halo['majorSemiaxis']=float(l[14])
            halo['intermediateSemiaxis']=float(l[15])
            halo['minorSemiaxis']=float(l[16])
            halo['Ixx']=float(l[17])
            halo['Ixy']=float(l[18])
            halo['Ixz']=float(l[19])
            halo['Iyy']=float(l[20])
            halo['Iyz']=float(l[21])
            halo['Izz']=float(l[22])
            halo['Lx']=float(l[23])
            halo['Ly']=float(l[24])
            halo['Lz']=float(l[25])
            halo['sigma_v']=float(l[26])
            halo['vx']=float(l[27])
            halo['vy']=float(l[28])
            halo['vz']=float(l[29]) 
            halo['max_v']=float(l[30])
            halo['mean_vr']=float(l[31])
            halo['Ekin']=float(l[32])
            halo['Epot']=float(l[33])
            halo['Vcmax']=float(l[34])
            halo['Mcmax']=float(l[35])
            halo['Rcmax']=float(l[36])
            halo['R200m']=float(l[37])
            halo['M200m']=float(l[38])
            halo['R200c']=float(l[39])
            halo['M200c']=float(l[40])
            halo['R500m']=float(l[41])
            halo['M500m']=float(l[42])
            halo['R500c']=float(l[43])
            halo['M500c']=float(l[44])
            halo['R2500m']=float(l[45])
            halo['M2500m']=float(l[46])
            halo['R2500c']=float(l[47])
            halo['M2500c']=float(l[48])
            halo['fsub']=float(l[49])
            halo['Nsubs']=int(l[50])

            haloes.append(halo)
    
    if exclude_subhaloes:
        haloes=[halo for halo in haloes if halo['substructureOf']==-1]
    if min_mass is not None:
        haloes=[halo for halo in haloes if halo['M']>min_mass]

    if read_region is not None:
        kept_haloes = {}
        haloes_temp = [h for h in haloes]
        haloes = []

        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            
            for h in haloes_temp:
                R_reg = R + h['R'] if keep_boundary_contributions else R - h['R']
                if (h['x']-cx)**2 + (h['y']-cy)**2 + (h['z']-cz)**2 < R_reg**2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2

            for h in haloes_temp:
                xh1 = x1 - h['R'] if keep_boundary_contributions else x1 + h['R']
                xh2 = x2 + h['R'] if keep_boundary_contributions else x2 - h['R']
                yh1 = y1 - h['R'] if keep_boundary_contributions else y1 + h['R']
                yh2 = y2 + h['R'] if keep_boundary_contributions else y2 - h['R']
                zh1 = z1 - h['R'] if keep_boundary_contributions else z1 + h['R']
                zh2 = z2 + h['R'] if keep_boundary_contributions else z2 - h['R']

                if xh1 < h['x'] < xh2 and yh1 < h['y'] < yh2 and zh1 < h['z'] < zh2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    return_variables = []
    if output_format=='dictionaries':
        return_variables.append(haloes)
    elif output_format=='arrays':
        return_variables.append({k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()})
    else:
        raise ValueError('Error! output_format argument should be either dictionaries or arrays')

    if output_redshift:
        return_variables.append(zeta)

    if read_region is not None:
        return_variables.append(kept_haloes)

    if len(return_variables) == 1:
        return return_variables[0]
    else:
        return tuple(return_variables)


def unpack_clus_data(clus_tuple, read_clus_kwargs, region=None):
    """
    Dynamically unpacks the tuple returned by read_clus based on what was actually read.
    
    This function uses the kwargs that were passed to read_clus() to know which fields
    were actually read, and unpacks accordingly.
    
    The read_clus function returns variables in this order (only if output_X=True):
    1. delta (if output_delta=True)
    2. vx, vy, vz (if output_v=True)
    3. pres (if output_pres=True)
    4. pot (if output_pot=True)
    5. opot (if output_opot=True)
    6. temp (if output_temp=True)
    7. metalicity (if output_metalicity=True)
    8. cr0amr (if output_cr0amr=True)
    9. solapst (if output_solapst=True)
    10. Bx, By, Bz (if output_B=True)
    11. keep_patches (if read_region is not None)
    
    Args:
        clus_tuple: tuple returned by read_clus
        read_clus_kwargs: dict with kwargs that were passed to read_clus (from get_read_clus_kwargs)
        region: region specification (if None, no keep_patches expected)
        
    Returns:
        Dictionary with variables (None for fields that weren't read)
        
    Author: Marco Molina
    """
    
    # Initialize all variables to None
    result = {
        'delta': None,
        'vx': None,
        'vy': None,
        'vz': None,
        'pres': None,
        'pot': None,
        'opot': None,
        'temp': None,
        'metalicity': None,
        'cr0amr': None,
        'solapst': None,
        'Bx': None,
        'By': None,
        'Bz': None,
        'keep_patches': None
    }
    
    # Extract what was actually READ from kwargs
    output_delta = read_clus_kwargs.get('output_delta', True)
    output_v = read_clus_kwargs.get('output_v', True)
    output_pres = read_clus_kwargs.get('output_pres', False)
    output_pot = read_clus_kwargs.get('output_pot', False)
    output_opot = read_clus_kwargs.get('output_opot', False)
    output_temp = read_clus_kwargs.get('output_temp', False)
    output_metalicity = read_clus_kwargs.get('output_metalicity', False)
    output_cr0amr = read_clus_kwargs.get('output_cr0amr', False)
    output_solapst = read_clus_kwargs.get('output_solapst', False)
    output_B = read_clus_kwargs.get('output_B', False)
    
    # Convert tuple to list for easier indexing
    clus_list = list(clus_tuple)
    idx = 0
    
    # Unpack according to the order specified in read_clus
    if output_delta:
        result['delta'] = clus_list[idx]
        idx += 1
    
    if output_v:
        result['vx'] = clus_list[idx]
        result['vy'] = clus_list[idx + 1]
        result['vz'] = clus_list[idx + 2]
        idx += 3
    
    if output_pres:
        result['pres'] = clus_list[idx]
        idx += 1
    
    if output_pot:
        result['pot'] = clus_list[idx]
        idx += 1
    
    if output_opot:
        result['opot'] = clus_list[idx]
        idx += 1
    
    if output_temp:
        result['temp'] = clus_list[idx]
        idx += 1
    
    if output_metalicity:
        result['metalicity'] = clus_list[idx]
        idx += 1
    
    if output_cr0amr:
        result['cr0amr'] = clus_list[idx]
        idx += 1
    
    if output_solapst:
        result['solapst'] = clus_list[idx]
        idx += 1
    
    if output_B:
        result['Bx'] = clus_list[idx]
        result['By'] = clus_list[idx + 1]
        result['Bz'] = clus_list[idx + 2]
        idx += 3
    
    if region is not None and idx < len(clus_list):
        result['keep_patches'] = clus_list[idx]
        idx += 1
    
    # Validate that we consumed all elements
    if idx != len(clus_list):
        print(f"WARNING: unpack_clus_data consumed {idx} elements but tuple has {len(clus_list)}.")
        print(f"This may indicate a mismatch between read_clus flags and unpacking logic.")
        print(f"Read kwargs: {read_clus_kwargs}")
        print(f"Expected order: delta, vx, vy, vz, " + 
              (f"pres, " if output_pres else "") +
              (f"pot, " if output_pot else "") +
              (f"opot, " if output_opot else "") +
              (f"temp, " if output_temp else "") +
              (f"metalicity, " if output_metalicity else "") +
              (f"cr0amr, " if output_cr0amr else "") +
              (f"solapst, " if output_solapst else "") +
              (f"Bx, By, Bz" if output_B else ""))
    
    return result


def get_read_clus_kwargs(sim_characteristics, level, region=None, **override_flags):
    """
    Constructs keyword arguments for read_clus() based on config and optional overrides.
    
    TWO-LEVEL SYSTEM:
      1. EXISTENCE flags (is_cooling, is_mascletB, has_X): What EXISTS in files
      2. READ flags (read_X): What we WANT to read from existing fields
    
    Read flags are taken from config (SIM_CHARACTERISTICS) by default, but can be
    overridden via function parameters.
    
    Args:
        sim_characteristics: dict with existence and read preference flags
        level: maximum refinement level to read
        region: optional region specification for spatial filtering
        **override_flags: optional overrides for read_X flags (e.g., read_velocity=False)
        
    Returns:
        dict with kwargs for read_clus()
        
    Example:
        # Use config defaults:
        kwargs = get_read_clus_kwargs(sim_chars, level, region)
        
        # Override config to skip velocity:
        kwargs = get_read_clus_kwargs(sim_chars, level, region, read_velocity=False)
        
    Author: Marco Molina
    """
    # Extract EXISTENCE flags from config (what fields exist in files)
    is_cooling = sim_characteristics.get('is_cooling', False)
    is_mascletB = sim_characteristics.get('is_mascletB', True)
    has_pres = sim_characteristics.get('has_pres', True)
    has_pot = sim_characteristics.get('has_pot', True)
    has_opot = sim_characteristics.get('has_opot', False)
    has_cr0amr = sim_characteristics.get('has_cr0amr', True)
    has_solapst = sim_characteristics.get('has_solapst', True)
    
    # Extract READ preference flags from config (what we want to read)
    # Can be overridden by function parameters
    read_velocity = override_flags.get('read_velocity', 
                                       sim_characteristics.get('read_velocity', True))
    read_B = override_flags.get('read_B', 
                                sim_characteristics.get('read_B', True))
    read_pressure = override_flags.get('read_pressure', 
                                       sim_characteristics.get('read_pressure', True))
    read_potential = override_flags.get('read_potential', 
                                        sim_characteristics.get('read_potential', True))
    read_old_potential = override_flags.get('read_old_potential', 
                                           sim_characteristics.get('read_old_potential', False))
    read_temperature = override_flags.get('read_temperature', 
                                         sim_characteristics.get('read_temperature', True))
    read_metalicity = override_flags.get('read_metalicity', 
                                        sim_characteristics.get('read_metalicity', True))
    read_cr0amr = override_flags.get('read_cr0amr', 
                                     sim_characteristics.get('read_cr0amr', True))
    read_solapst = override_flags.get('read_solapst', 
                                      sim_characteristics.get('read_solapst', True))
    
    # Build kwargs: output_X = read_X AND has_X
    # (only request reading if we want it AND it exists)
    kwargs = {
        'max_refined_level': level,
        'output_delta': True,                                    # Always read density
        'output_v': read_velocity,                               # Velocity (always exists)
        'output_pres': read_pressure and has_pres,               # Pressure (if exists)
        'output_pot': read_potential and has_pot,                # Potential (if exists)
        'output_opot': read_old_potential and has_opot,          # Old potential (if exists)
        'output_temp': read_temperature and is_cooling,          # Temperature (if cooling)
        'output_metalicity': read_metalicity and is_cooling,     # Metalicity (if cooling)
        'output_cr0amr': read_cr0amr and has_cr0amr,             # CR flag (if exists)
        'output_solapst': read_solapst and has_solapst,          # Solapst (if exists)
        'output_B': read_B and is_mascletB,                      # B field (if mascletB)
        'is_mascletB': is_mascletB,                              # Tell reader: B exists
        'is_cooling': is_cooling,                                # Tell reader: cooling exists
        'verbose': False,
        'read_region': region
    }
    return kwargs