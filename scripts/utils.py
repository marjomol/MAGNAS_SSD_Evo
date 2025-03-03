"""
PRIMAL Seed Generator
A tool to generate initial conditions for cosmological simulations of primordial magnetic fields.

utils module
Contains utility tool functions used by the PRIMAL Seed Generator modules.

Created by Marco Molina Pradillo
"""

import numpy as np
import sys

def save_3d_array(filename, array):
    '''
    Saves a 3D array to a text file.
    
    Args:
        - filename: name of the file to save the array
        - array: 3D array to save
        
    Author: Marco Molina
    '''
    with open(filename, 'w') as f:
        for i in range(array.shape[0]):
            np.savetxt(f, array[i], header=f'Slice {i}', comments='')
            f.write('\n')
            
def is_ordered(vector):
    '''
    Checks if a vector is ordered.
    
    Args:
        - vector: vector to check
        
    Returns:
        - True if the vector is ordered, False otherwise
        
    Author: Marco Molina
    '''
    for i in range(len(vector) - 1):
        if vector[i] > vector[i+1]:
            return False
    return True

def update_loading_bar(progress):
    '''
    Updates and prints a loading bar in the console.
    
    Args:
        - progress: progress percentage
    
    Author: Marco Molina
    '''
    sys.stdout.write('\r[{0}] {1}%'.format('#' * (progress // 1), progress))
    sys.stdout.flush()
    
def create_vector_levels(npatch):
    """
    Creates a vector containing the level for each patch. Nothing really important, just for ease

    Args:
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)

    Returns:
        numpy array containing the level for each patch

    Author: David Vallés
    """
    vector = [[0]] + [[i + 1] * x for (i, x) in enumerate(npatch[1:])]
    vector = [item for sublist in vector for item in sublist]
    return np.array(vector)