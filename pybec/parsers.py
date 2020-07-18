"""
parsers.py
Module that mainly deals with extracting information from QuantumEspresso Output

Handles the primary functions
"""


import numpy as np
import os
import logging

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)


def get_final_positions(output_file, unit_multiplier=1.0):
    """
    Retrieves the final positions from the last step of the specified output file.
    
    Parameters
    ----------
    output_file : str
        File path to the output file for the step from which to pull converged forces.
    
    Returns
    -------
    dict
        Ionic coordinates in a dictionary where the keys are the element symbols and
        the values are the numpy coordinate array for all atoms of that element.
    """
    if not os.path.isfile(output_file):
        raise ValueError("Output File {} is not a valid file path".format(output_file))
    
    # strings to search for
    converged_str = 'convergence achieved for system relaxation'
    pos_start = 'ATOMIC_POSITIONS'
    
    # store whether system is converged
    converged = False
    
    # flag to store whether we have encountered a block of forces
    pos_block = False
    
    # store the forces
    pos = {}
    
    with open(output_file, 'r') as f:
        for line in f:    # iterate through the lines.  Go line by line to not over-use memory.
            # make sure system converged
            if converged_str in line:
                converged = True
            
            # find the force block
            elif pos_start in line:
                pos_block = True
                pos = {}    # erase earlier forces, keeping only most recent
                
            # store the forces, keeping only the most recent forces
            elif pos_block:
                line = line.split()
                if line == []:    # reached end of force block
                    pos_block = False
                else:
                    try:
                        pos[line[0]].append(line[1:])
                    except KeyError:
                        pos[line[0]] = [line[1:]]
    
    if converged:
        for key in pos:
            pos[key] = np.array(pos[key], dtype=np.float64) * unit_multiplier
        return pos
    else:
        raise Exception("Positions don't seem to have converged.  Check step and run again.")


def get_final_cell(output_file, unit_multiplier=1.0):
    """
    Retrieves the final cell vectors from the last step of the specified output file.
    
    Parameters
    ----------
    output_file : str
        File path to the output file for the step from which to pull converged forces.
    
    Returns
    -------
    np.ndarray
        A 3x3 array of the cell vectors, as 3 row vectors [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]
        
    """
    if not os.path.isfile(output_file):
        raise ValueError("Output File {} is not a valid file path".format(output_file))
    
    # strings to search for
    converged_str = 'convergence achieved for system relaxation'
    cell_start = 'CELL_PARAMETERS'
    
    # store whether system is converged
    converged = False
    
    # flag to store whether we have encountered a block of forces
    cell_block = False
    
    # store the forces
    cell = []
    
    with open(output_file, 'r') as f:
        for line in f:    # iterate through the lines.  Go line by line to not over-use memory.
            # make sure system converged
            if converged_str in line:
                converged = True
            
            # find the force block
            elif cell_start in line:
                cell_block = True
                cell = []    # erase earlier forces, keeping only most recent
                
            # store the forces, keeping only the most recent forces
            elif cell_block:
                line = line.split()
                if line == []:    # reached end of force block
                    cell_block = False
                else:
                    cell.append(line)
    
    if converged:
        cell = np.array(cell, dtype=np.float64) * unit_multiplier
        return cell
    else:
        raise Exception("Positions don't seem to have converged.  Check step and run again.")


def get_converged_forces(output_file):
    """
    Retrieves the converged forces from the last step of the specified output file.
    
    Parameters
    ----------
    output_file : str
        File path to the output file for the step from which to pull converged forces.
    
    Returns
    -------
    dict
        Ionic forces in a dictionary where the keys are the element symbols and
        the values are the numpy force array for all atoms of that element.
    """
    if not os.path.isfile(output_file):
        raise ValueError("Output File {} is not a valid file path".format(output_file))
    
    # strings to search for
    converged_str = 'convergence achieved for system relaxation'
    force_start = 'Forces acting on atoms (au):'
    
    # store whether system is converged
    converged = False
    
    # flag to store whether we have encountered a block of forces
    force_block = False
    
    # store the forces
    forces = {}
    
    with open(output_file, 'r') as f:
        for line in f:    # iterate through the lines.  Go line by line to not over-use memory.
            # make sure system converged
            if converged_str in line:
                converged = True
            
            # find the force block
            elif force_start in line:
                force_block = True
                forces = {}    # erase earlier forces, keeping only most recent
                
            # store the forces, keeping only the most recent forces
            elif force_block:
                line = line.split()
                if line == []:    # reached end of force block
                    force_block = False
                else:
                    try:
                        forces[line[0]].append(line[1:])
                    except KeyError:
                        forces[line[0]] = [line[1:]]
    
    if converged:
        for key in forces:
            forces[key] = np.array(forces[key], dtype=np.float64)
        return forces
    else:
        raise Exception("Forces don't seem to have converged.  Check step and run again.")
        
def get_coordinates(xyz_file, skip=2, d=None):
    """
    Loads the XYZ coordinates of the specified file.
    
    Parameters
    ----------
    xyz_file : str
        File path to the .xyz file containing the relaxed crystal structure
        
    Returns
    -------
    dict
        The coordinates in angstroms in a dictionary where the keys 
        are the element symbols and the values are the numpy
        arrays of the coordinates for all atoms of that element.
    """
    if not os.path.isfile(xyz_file):
        raise ValueError("XYZ File {} is not a valid file path".format(xyz_file))
        
    coords_arr = np.genfromtxt(xyz_file, skip_header=skip, delimiter=d, dtype=str)
    
    # store the coordinates
    coords = {}
    
    for element in np.unique(coords_arr[:,0]):    # make each unique element a key in the coords dict
        coords[element] = coords_arr[coords_arr[:,0] == element][:,1:].astype(np.float64)
    
    return coords


def get_lattice(xyz_file):
    """
    Loads the lattice vectors from the specified file.
    
    Parameters
    ----------
    xyz_file : str
        File path to the .xyz file containing the relaxed crystal structure
        
    Returns
    -------
    numpy.ndarray
        The lattice vectors in angstroms in a numpy array where the rows are 
        lattice vectors a, b, and c, and the columns are the unit directions
        x, y, z.
    """
    if not os.path.isfile(xyz_file):
        raise ValueError("XYZ File {} is not a valid file path".format(xyz_file))
    
    with open(xyz_file, 'r') as f:
        lat_list = f.readlines()[1].split()
        lattice = np.array([lat_list[0:3], lat_list[3:6], lat_list[6:9]], dtype=np.float64)
    
    return lattice
