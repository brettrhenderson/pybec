"""
parsers.py
Module that mainly deals with extracting information from QuantumEspresso Output

Handles the primary functions
"""


import numpy as np
import os
import logging
import glob
from collections import OrderedDict

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

def get_trajectory(output_file, unit_multiplier=1.0):
    """
    Retrieves the final positions from the last step of the specified output file.

    Parameters
    ----------
    output_file : str
        File path to the output file for the step from which to pull trajectory.

    Returns
    -------
    dict
        Ionic coordinates at each timestep in a dictionary where the keys are the element symbols and
        the values are the numpy coordinate array for all atoms of that element.
    """
    if not os.path.isfile(output_file):
        raise ValueError("Output File {} is not a valid file path".format(output_file))

    # strings to search for
    pos_start = 'ATOMIC_POSITIONS'
    stepline = 'Physical Quantities at step:'

    # flag to store whether we have encountered a block of positions
    pos_block = False

    # store the atomic positions
    stepnum = None
    traj = {}

    with open(output_file, 'r') as f:
        for line in f:  # iterate through the lines.  Go line by line to not over-use memory.
            # make sure system converged
            if stepline in line:
                stepnum = int(line.split(':')[1])

            # find the pos block
            elif pos_start in line:
                pos_block = True
                pos = {}  # erase earlier forces, keeping only most recent

            # store the positions
            elif pos_block:
                line = line.split()
                if not line:  # reached end of position block
                    pos_block = False
                    traj[stepnum] = pos  # store positions with step number
                else:
                    try:
                        pos[line[0]].append(line[1:])
                    except KeyError:
                        pos[line[0]] = [line[1:]]
    for key1 in traj:
        for key2 in traj[key1]:
            traj[key1][key2] = np.array(traj[key1][key2], dtype=np.float64) * unit_multiplier
    return traj

def get_dipole(output_file):
    """
    Retrieves the electric and ionic polarizations from output file.

    Parameters
    ----------
    output_file : str
        File path to the output file for the step from which to pull polarizations.

    Returns
    -------
    tuple
        header, pols. Header is a list of the header names associated with
        pols. pols is an N X 5 np.ndarray. For each of N steps, it contains The
        associated step number, timestep, electronic, ionic, and total cell dipole.
    """
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f1:
            line_1 = ''
            pols = []
            ts_found = False
            for line in f1:
                if not ts_found and "MD Simulation time step" in line:
                    ts = float(line.split('=')[1])
                elif line.startswith('Elct. dipole'):
                    step = float(line_1.split()[0])
                    e_dipole = float(line.split()[3])
                    i_dipole = float(line.split()[7])
                    pols.append([step, ts, e_dipole, i_dipole, e_dipole+i_dipole])
                line_1 = line
        return ['Step', 'Dt', 'E_Dipole', 'I_Dipole', 'Tot_Dipole'], np.array(pols)
    else:
        raise ValueError('Invalid input file name')

def get_full_dipole(directory, remove_dupes=True):
    """
    Gets the trajectory of the polarization.

    Parameters
    ----------

    directory : str
        The path to a directory containing all of the output files to parse for polarization data.

    remove_dupes : bool
        If true, keep only one of each step number, the one from the last file
        it occurs in.

    Returns
    -------

    numpy.ndarray
        Nx2 array with the first column containing the time and second column containing
        the overall polarization. Duplicate timesteps are removed during processing.

    """
    files = glob.glob(os.path.join(directory, '*'))
    files.sort()
    if remove_dupes:
        pols = {}
        for file in files:
            _, pol = get_dipole(file)
            if len(pol):
                for i, p in enumerate(pol):
                    pols[pol[i, 0]] = pol[i]
        pols = OrderedDict(sorted(pols.items()))
        pols = np.concatenate([pols[step].reshape(1,-1) for step in pols], axis=0)
    else:
        pols = []
        for file in files:
            _, pol = get_dipole(file)
            pols.append(pol)
        pols = np.concatenate(pols, axis=0)
    return pols

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

    for key in pos:
        pos[key] = np.array(pos[key], dtype=np.float64) * unit_multiplier
    if not converged:
        print("WARNING: Positions don't seem to have converged.")
    return pos


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

    cell = np.array(cell, dtype=np.float64) * unit_multiplier
    if not converged:
        print("WARNING: Cell doesn't seem to have converged.")
    return cell


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

def get_coordinates(xyz_file, skip=2, delimiter=None):
    """
    Loads the XYZ coordinates of the specified file.

    Parameters
    ----------
    xyz_file : str
        File path to the .xyz file containing the relaxed crystal structure

    delimiter : str
        delimiter separating x, y, z coordinates in the file, to be used
        by numpy's genfromtxt method

    Returns
    -------
    dict
        The coordinates in angstroms in a dictionary where the keys
        are the element symbols and the values are the numpy
        arrays of the coordinates for all atoms of that element.
    """
    if not os.path.isfile(xyz_file):
        raise ValueError("XYZ File {} is not a valid file path".format(xyz_file))

    coords_arr = np.genfromtxt(xyz_file, skip_header=skip, delimiter=delimiter, dtype=str)

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
