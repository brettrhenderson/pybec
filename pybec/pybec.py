"""
pybec.py
Python package for extracting and manipulating Born Effective Charges from QuantumEspresso Output

Handles the primary functions
"""


import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import re
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import glob
try:
    from ipywidgets import interact
except:
    pass
from pykrige.rk import Krige
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D
from matplotlib.collections import PolyCollection, LineCollection
from pykrige.compat import GridSearchCV
from scipy.interpolate import griddata
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared as ESS, RationalQuadratic as RQ, ConstantKernel as C, Product
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator as NND
from scipy.interpolate import LinearNDInterpolator as LND

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

A_TO_B = 0.529177249

# Default Markers to use when plotting different elements
MARKERS = 'o s h + x * p D v ^ < >'
# x = 0, y = 1, z = 2
DIRS = ['x', 'y', 'z']

#keep track of possible slice directions as a set
DIRS_SET = set(DIRS)

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
        raise ValueError("XYZ File {} is not a valid file path".format(output_file))
    
    with open(xyz_file, 'r') as f:
        lat_list = f.readlines()[1].split()
        lattice = np.array([lat_list[0:3], lat_list[3:6], lat_list[6:9]], dtype=np.float64)
    
    return lattice 


def get_centroid(atoms_dict, key='all'):
    """
    Calculate the centroid of the atoms of a certian element in a unit cell.
    
    Parameters
    ----------
    atoms_dict : dict
        dictionary of atomic coordinates in Angstroms,
        where the keys are the element symbols and the values are the
        numpy coordinate array for all atoms of that element.
    
    key : str
        the symbol of the element you would like to calculate the centroid
        for.  If 'all' is given, the centroid of the entire unit cell will
        be computed.
    
    Returns
    -------
    numpy.ndarray
        a numpy array of the x,y, and z coordinates (in Angstroms) of
        the centroid.
    """
    if key == 'all':  # get the centroid of all atoms in the coordinates dict
        requested_atoms = np.concatenate([atoms_dict[key] for key in atoms_dict], axis=0)
    else:
        requested_atoms = atoms_dict[key]
    return np.mean(requested_atoms, axis=0)

def get_spherical_coords_to_centroid(atoms_dict, centroid):
    """
    Calculate the centroid of the atoms of a certian element in a unit cell.
    
    Parameters
    ----------
    atoms_dict : dict
        Dictionary of atomic coordinates in Angstroms,
        where the keys are the element symbols and the values are the
        numpy coordinate array for all atoms of that element.
    
    centroid : str
        Numpy array of the x,y, and z coordinates (in Angstroms) of
        the centroid.
    
    Returns
    -------
    r : dict
        Dictionary of the distances of each ion in unit cell to the
        specified centroid.  The keys are the element symbols and the
        values are numpy.ndarrays of the distances for each ion.
        
    phi : dict
        Dictionary of the polar angle of each ion in unit cell with reference
        to an origin at the specified centroid.
        
    theta : dict
        Dictionary of the azimuthal angle of each ion in unit cell with reference
        to an origin at the specified centroid.
        
    """
    r = {}
    phi = {}
    theta = {}
    
    for key in atoms_dict:
        r[key] = np.array([np.linalg.norm(pos - centroid) for pos in atoms_dict[key]])
        phi[key] = np.array([np.arccos((pos - centroid)[2]/r_) for pos, r_ in zip(atoms_dict[key], r[key])])
        theta[key] = np.array([np.arctan2((pos - centroid)[1], (pos - centroid)[0]) for pos in atoms_dict[key]])
        theta[key][theta[key] < 0] += 2*np.pi
    return r, phi, theta
    

def get_dist_to_centroid(atoms_dict, centroid):
    """
    Calculate the centroid of the atoms of a certian element in a unit cell.
    
    Parameters
    ----------
    atoms_dict : dict
        Dictionary of atomic coordinates in Angstroms,
        where the keys are the element symbols and the values are the
        numpy coordinate array for all atoms of that element.
    
    centroid : str
        Numpy array of the x,y, and z coordinates (in Angstroms) of
        the centroid.
    
    Returns
    -------
    dict
        Dictionary of the distances of each ion in unit cell to the
        specified centroid.  The keys are the element symbols and the
        values are numpy.ndarrays of the distances for each ion.
        
    """
    dist_to_centroid = {}
    
    for key in atoms_dict:
        dist_to_centroid[key] = np.array([np.linalg.norm(pos - centroid) for pos in atoms_dict[key]])
    
    return dist_to_centroid


def get_BECs(for_0, for_1, e_field, e_field_direction):
    """
    Calculate the born effective charges for an array of ions.
    
    Parameters
    ----------
    for_0 : dict
        Ionic forces in zero field in a dictionary where the keys are the element 
        symbols and the values are the numpy force array for all atoms of that element.
    for_1 : dict
        Ionic forces in applied efield but with clamped ions in a dictionary formatted like for_0.
    e_field : float
        The magnitude of the applied electric field.
    e_field_direction : list
        The 3D vector direction of the efield.
        Ex: [0,0,1] is an electric field in the positive z-direction.
    
    """
    BEC = {}
    e_field_direction = np.array(e_field_direction) / np.linalg.norm(np.array(e_field_direction))
    
    # make sure the parsed forces have matching elements
    if set(for_0.keys()) != set(for_1.keys()):
        raise ValueError('Different elements present in the two provided files.')

    # get the Born Effective Charge using the finite difference between 0 field and clamped ion
    for key in for_0:
        if len(for_0[key]) != len(for_1[key]):
            raise ValueError('Provided files have different number of {} atoms'.format(key))
        BEC[key] = (for_1[key].dot(e_field_direction) - for_0[key].dot(e_field_direction)) / e_field
    return BEC


def infer_local_field(for_0, for_1, z_exp, e_ext=0.001, e_field_direction=[0,0,1]):
    """
    Calculate the born effective charges for an array of ions.
    
    Parameters
    ----------
    for_0 : dict
        Ionic forces in zero field in a dictionary where the keys are the element 
        symbols and the values are the numpy force array for all atoms of that element.
    for_1 : dict
        Ionic forces in applied efield but with clamped ions in a dictionary formatted like for_0.
    z_exp : dict
        Expected born effective charge for each element type from a matrix-only calculation.
        Keys are element symbols, and values are expected BECs.
    e_ext : float, optional, default: 0.001
        The magnitude of the applied electric field (au).
    e_field_direction : list, optional, default: [0,0,1]
        The 3D vector direction of the efield.
        Ex: [0,0,1] is an electric field in the positive z-direction.
    """
    e_loc = {}
    e_field_direction = np.array(e_field_direction) / np.linalg.norm(np.array(e_field_direction))
    
    # make sure the parsed forces have matching elements
    if set(for_0.keys()) != set(for_1.keys()):
        raise ValueError('Different elements present in the two provided files.')

    # get the Born Effective Charge using the finite difference between 0 field and clamped ion
    for key in for_0:
        if len(for_0[key]) != len(for_1[key]):
            raise ValueError('Provided files have different number of {} atoms'.format(key))
        e_loc[key] = (for_1[key].dot(e_field_direction) - for_0[key].dot(e_field_direction)) / z_exp[key] - e_ext
    return e_loc


def get_field_along_d(field_dict, sub_mean_field=False, e_field=0.25, e_field_direction=[0,0,1]):
    """
    Calculate the electric field along a specific direction from the FE results.
    
    Parameters
    ----------
    field_dict : dict
        Electric field at atomic locations in a dictionary where the keys are the element 
        symbols and the values are the numpy array of the electric field for all atoms of that element.
    sub_mean_field : bool, optional, default: False
        If set, the external applied field is subtracted from the calculated fields, meaning that only the local field
        disturbance caused by the inclusion will be plotted.
    e_field : float
        The magnitude of the applied electric field in V/m.
    e_field_direction : list
        The 3D vector direction of the efield.
        Ex: [0,0,1] is an electric field in the positive z-direction.
        
    Returns
    -------
    field : dict
        Electric field magnitude along the specified direction at atomic locations in a dictionary
        with same format as field_dict.
       
    """
    field = {}
    e_field_direction = np.array(e_field_direction) / np.linalg.norm(np.array(e_field_direction))

    for key in field_dict:
        if sub_mean_field:
            field_dict[key] = field_dict[key] - (e_field * e_field_direction)
        field[key] = field_dict[key].dot(e_field_direction.T)
    return field


def to_Bohr(coords):
    """Convert a coordinate dictionary from Angstroms to Bohr"""
    for key in coords:
        coords[key] = coords[key] * A_TO_B 
    return coords


def get_dipole_field(coords, dipole_loc=[0,0,0], p_vec=[0,0,1], p=1, is_angstrom=True):
    """
    Returns the electric field from a point dipole.
    
    Parameters
    ----------
    coords : dict
        Dictionary of atomic coordinates, where the keys are the element symbols,
        and the values are the numpy coordinate array for all atoms of that element.
    dipole_loc : list or numpy.ndarray
        The 3D coordinates of the dipole location.
        Ex: dipole_loc=get_centroid(coords, key='Ag')
    dipole_loc : list or numpy.ndarray
        The 3D coordinates of the dipole location.
        Ex: dipole_loc=get_centroid(coords, key='Ag')
    is_angstrom : bool, optional, default : True
        Indicates whether the input atomic coordinates are in Angstroms (if False, Bohr is assumed)
        
    Returns
    -------
    field : dict
        Electric field at atomic locations in a dictionary
        with same format as coords.
    """
    dipole_loc = np.array(dipole_loc)
    p_vec = np.array(p_vec)
    
    # verify that it is normalized first
    p_vec = p_vec / np.linalg.norm(p_vec)
    p_vec = p * p_vec
    
    # make sure everything is in atomic units
    if is_angstrom:
        coords = to_Bohr(coords)
        dipole_loc = A_TO_B * dipole_loc
    
    ions = as_dataframe(coords)
    field = {}
    for key in coords:
        r_vec = ions.loc[ions['element']==key][['X','Y','Z']].values - dipole_loc
        r_mag = np.linalg.norm(r_vec, axis=1).reshape(-1,1)
        r_unit = r_vec / r_mag
        field[key] = 1/r_mag**3*(np.dot(r_unit,  3*p_vec.T).reshape(-1, 1) * r_unit - p_vec)
    return field


def get_dipole_field_displaced(coords, dipole_loc=[0,0,0], p_vec=[0,0,1], q=1, d=0.1, is_angstrom=True):
    """
    Returns the electric field from a point dipole.
    
    Parameters
    ----------
    coords : dict
        Dictionary of atomic coordinates, where the keys are the element symbols,
        and the values are the numpy coordinate array for all atoms of that element.
    dipole_loc : list or numpy.ndarray
        The 3D coordinates of the dipole location.
        Ex: dipole_loc=get_centroid(coords, key='Ag')
    dipole_loc : list or numpy.ndarray
        The 3D coordinates of the dipole location.
        Ex: dipole_loc=get_centroid(coords, key='Ag')
    is_angstrom : bool, optional, default : True
        Indicates whether the input atomic coordinates are in Angstroms (if False, Bohr is assumed)    
    
    Returns
    -------
    field : dict
        Electric field at atomic locations in a dictionary
        with same format as coords.
    """
    dipole_loc = np.array(dipole_loc)
    p_vec = np.array(p_vec)
    
    # verify that it is normalized first
    p_vec = p_vec / np.linalg.norm(p_vec)
    
    # make sure everything is in atomic units
    if is_angstrom:
        coords = to_Bohr(coords)
        dipole_loc = A_TO_B * dipole_loc
    
    
    ions = as_dataframe(coords)
    field = {}
    for key in coords:
        ppos = dipole_loc + d/2 * p_vec
        pneg = dipole_loc - d/2 * p_vec
        r_pos = ions.loc[ions['element']==key][['X','Y','Z']].values - ppos
        r_pos_mag = np.linalg.norm(r_pos, axis=1).reshape(-1,1)
        r_neg = ions.loc[ions['element']==key][['X','Y','Z']].values - pneg
        r_neg_mag = np.linalg.norm(r_neg, axis=1).reshape(-1,1)
        r_pos_unit = r_pos / r_pos_mag
        r_neg_unit = r_neg / r_neg_mag
        pos_field = q / r_pos_mag**2 * r_pos_unit
        neg_field = -q / r_neg_mag**2 * r_neg_unit
        field[key] = pos_field + neg_field
    return field


def as_dataframe(atoms_dict, BEC=None, for0=None, for1=None):
    """make a pandas dataframe with the combined coordinates"""
    cols = ['element', 'X', 'Y', 'Z']
    if BEC is not None:
        cols.append('BEC')
    if for0 is not None:
        cols += ['Force 0x', 'Force 0y', 'Force 0z']
    if for1 is not None:
        cols += ['Force 1x', 'Force 1y', 'Force 1z']
    df = pd.DataFrame(columns=cols)
    for key in atoms_dict:
        d = {'element' : np.array([key for _ in atoms_dict[key]]),
             'X' : atoms_dict[key][:, 0],
             'Y' : atoms_dict[key][:, 1],
             'Z' : atoms_dict[key][:, 2]}
        if BEC is not None:
            d['BEC'] = BEC[key]
        if for0 is not None:
            d['Force 0x'] = for0[key][:,0]
            d['Force 0y'] = for0[key][:,1]
            d['Force 0z'] = for0[key][:,2]
        if for1 is not None:
            d['Force 1x'] = for1[key][:,0]
            d['Force 1y'] = for1[key][:,1]
            d['Force 1z'] = for1[key][:,2]
        df_temp = pd.DataFrame(data=d)
        df = df.append(df_temp, ignore_index = True)
    return df

def df_to_dicts(df):
    coords = {}
    BECs = {}
    for el in  df['element'].unique():
        coords[el] = df[df['element'] == el][['X','Y','Z']].values
        BECs[el] = df[df['element'] == el]['BEC'].values
    return coords, BECs


def gen_BEC_df(no_efield, clamped_ion, xyz, e_field=0.001, e_field_direction=[0,0,1], add_forces=False):
    
    # parse coordinates
    coords = get_coordinates(xyz)
    
    # parse forces
    for_0, for_1 = get_converged_forces(no_efield), get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = get_BECs(for_0, for_1, e_field, e_field_direction)
    if add_forces:
        return as_dataframe(coords, BEC, for_0, for_1)
    else:
        return as_dataframe(coords, BEC)


def ave_BEC_dict(elements, BECs):
    """
    Create a dictionary where the keys are the element symbols and the values are the average
    Born Effective Charges for that element. 
    """
    
    elements, BECs = (list(elements), list(BECs))
    BEC_dict = {el: [] for el in elements}
    
    for element, BEC in zip(elements, BECs):
        BEC_dict[element].append(BEC)
    for key in BEC_dict:
        BEC_dict[key] = np.mean(BEC_dict[key])
    return BEC_dict
        
        
def point_in_hull(point, hull, tolerance=1e-12, inc=False):
    """https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl/42165596#42165596"""
    if not isinstance(hull,ConvexHull):
        hull = ConvexHull(hull)
    if inc:
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)
    else:  # don't include points that lie within a tolerance of a facet of the hull
        for eq in hull.equations:
            # the point is within a certain tolerance of a facet of the hull
            if (np.dot(eq[:-1], point) + eq[-1] <= tolerance) & (np.dot(eq[:-1], point) + eq[-1] >= -tolerance):
                return False
            elif (np.dot(eq[:-1], point) + eq[-1] > tolerance):    # outside the hull
                return False
        return True
    
    
def points_in_hull(points, hull, tolerance=1e-12, inc=False):
    if not isinstance(hull,ConvexHull):
        hull = ConvexHull(hull)
    in_hull = []
    for point in points:
        in_hull.append(point_in_hull(point, hull, tolerance, inc))
    return np.array(in_hull)


def select_cell_segment(cell_df, lattice, frac_dir):
    """
    Select a portion of the unit cell along a given set of directions.
    
    The fractional direction is a vector of coefficients for the lattice vectors of the unit cell.
    For lattice vectors a, b, c, a direction vector of [0.5, 1, 0.5]
    means that we want the cell up to 0.5a and 0.5c, with all values of b.
    
    A negative value for direction means that we want values greater than (1 - fraction) times
    the lattice vector.  Thus, [-0.3, 1, 1] means that we want the cell segment
    greater than 0.7a. [-0.3,-0.3,-0.3] would mean we want all cell positions from that lie within
    [0.7a, a], [0.7b, b], and [0.7c, c].
    """
    if not np.array(frac_dir).all():  # there is a zero in the direction, which should not be
        raise ValueError('All fractional direction elements should be non-zero.')
    
    cell_seg = cell_df
    scaled_lattice_vecs = []
    shifts = []   # if the direction is negative, we need to add on a shift
    combos = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]).T
    
    for i, fraction in enumerate(frac_dir):
        if fraction < 0:
            
            # get the limits of the cell that you want to keep in this direction
            scaled_lattice_vecs.append(abs(fraction) * lattice[i])
            
            # get the cell greater than frac % 1
            shifts.append((fraction % 1.0) * lattice[i])
        else:
            scaled_lattice_vecs.append(fraction * lattice[i])
    
    # now add the combinations of lattice vectors to each other to get 8 points
    scaled_lattice_vecs = np.array(scaled_lattice_vecs).T
    vertices = np.dot(scaled_lattice_vecs, combos).T    # the vertices of a the selection box
    
    # if the origin is shifted, then add in the shifts
    for shift in np.array(shifts):
        vertices = vertices + shift
    
    # make the in_bounds inclusive if we are on the outer part of the unit cell, 
    # otherwise, leave it exclusive
#     inc=False
#     if len(shifts):
#         inc = True
    
    # now, find all points in the original cell that lie inside these vertices
    in_bounds = points_in_hull(cell_seg[['X', 'Y', 'Z']].values, vertices) # inc=inc)
    cell_seg = cell_seg[in_bounds]
    
    return cell_seg


def pad_cell(cell_df, lattice, pad=0.4):
    # Expand periodic boundaries by one lattice vector in every direction
    shifts = [-1, 0, 1]

    cell_expand = cell_df.copy(deep=True)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i, j, k) == (1,1,1):    # corresponds to the center unit cell, don't repeat it
                    continue
                else:
                    shift = np.array([shifts[i], shifts[j], shifts[k]]).T
                    log.debug('1. shift: {}'.format(shift))
                    lat_add = (lattice.T * shift).T
                    log.debug('2. lattice addition: \n{}'.format(lat_add))
                    
                    # vector specifying which chunk of the cell to multiply
                    # if we are moving in the direction [1,0,0], that is in the direction of lattice vector a
                    # and pad is 0.5, we select the first half of the cell along lattice vector a
                    # to multiply.
                    cell_selection = [el if el != 0. else 1. for el in shift * pad]
                    log.debug('3. cell selection: {}'.format(cell_selection))
                    
                    cell_temp = select_cell_segment(cell_df.copy(deep=True), lattice, cell_selection)
                    log.debug('4. selected cell df: \n{}'.format(cell_temp))
                    
                    # add each of the scaled lattice vectors to the position data
                    for vec in lat_add:
                        cell_temp[['X','Y','Z']] += vec
                    log.debug('5. shifted selected cell df: \n{}'.format(cell_temp))
                    cell_expand = pd.concat([cell_expand, cell_temp])
                    log.debug('6. Expanded cell df: \n{}'.format(cell_expand))
                    
    return cell_expand

def interp_3d(coord_arr, val_arr, resolution=100j, lattice=None, xlim=(0,1),
              ylim=(0,1), zlim=(0,1),method='linear', return_grids=False):
    
    if lattice is None:
        min_x, max_x = xlim
        min_y, max_y = ylim
        min_z, max_z = zlim
    else:
        # get the limits of the lattice along the slice direction
        x_lat = lattice[:, 0]
        y_lat = lattice[:, 1]
        z_lat = lattice[:, 2]

        # for plotting xlim and ylim
        min_x, max_x = min(x_lat), max(x_lat)
        min_y, max_y = min(y_lat), max(y_lat)
        min_z, max_z = min(z_lat), max(z_lat)
        
    x, y, z = coord_arr[:,0], coord_arr[:,1], coord_arr[:,2]
    points = np.array(list(zip(x, y, z)))
    
    # grid on which to evaluate interpolators
    grid_x, grid_y, grid_z = np.mgrid[min_x:max_x:resolution*1j, min_y:max_y:resolution*1j, min_z:max_z:resolution*1j]
    
    if method == 'kriging':
        uk3d = UniversalKriging3D(x, y, z, val_arr, variogram_model='spherical', nlags=10, enable_plotting=False, 
                          drift_terms=['regional_linear'], verbose=False)
        gridx = np.linspace(min_x, max_x, resolution)
        gridy = np.linspace(min_y, max_y, resolution)
        gridz = np.linspace(min_z, max_z, resolution)
        preds, uk_ss3d = uk3d.execute('grid', gridx, gridy, gridz)
        preds = preds.data
        
    elif method == 'gp':
        kernel = C(1.0, (1e-3, 1e3)) * RQ(2.0, 1.0, (1e-1, 2e1), (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                      alpha=1e-10)
        gp.fit(points, val_arr)
        preds = gp.predict(np.array(list(zip(grid_x.ravel(), grid_y.ravel(), 
                                            grid_z.ravel())))).reshape(resolution,resolution,resolution)
    elif method == 'rbf':
        rbfi = Rbf(x, y, z, val_arr)
        preds = rbfi(grid_x, grid_y, grid_z)
    elif method == 'linear':
        lndi = LND(points, val_arr)
        preds = lndi(grid_x, grid_y, grid_z)
    elif method == 'nearest':
        nndi = NND(points, val_arr)
        preds = nndi(grid_x, grid_y, grid_z)
        
    if return_grids:
        return preds, grid_x, grid_y, grid_z
    else:
        return preds
    
    
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume, fig):
    remove_keymap_conflicts({'j', 'k','l'})
    ax = fig.add_subplot(111)
    ax.volume = volume
    ax.dir=0
    ax.index = volume.shape[ax.dir] // 2  
    ax.set_title('Slices in the X-direction')
    ax.set_xlabel('Y / $\AA$')
    ax.set_ylabel('Z / $\AA$')
    plot = ax.imshow(volume[ax.index].T, origin='bottom',)
    fig.canvas.mpl_connect('key_press_event', process_key)
    return fig, ax, plot


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    elif event.key == 'l':
        next_direction(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[ax.dir]  # wrap around using %
    
    p = re.compile(r'\d+\s\/')
    title = ax.get_title()
    ax.set_title(p.sub('{} /'.format(ax.index), title))
    
    if ax.is_kriging:
        direction = (ax.dir + 1) % 3
    else:
        direction = ax.dir
    
    if direction == 0:
        img = volume[ax.index]
    elif direction == 1:
        img = volume[:,ax.index,:]
    else:
        img = volume[:,:,ax.index]
    
    if not ax.is_kriging:
        img = img.T
    ax.images[0].set_array(img) 
    
    if hasattr(ax, 'add_atoms') and ax.add_atoms:
        plot_atoms(ax)
    
def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[ax.dir]
    
    p = re.compile(r'\d+\s\/')
    title = ax.get_title()
    ax.set_title(p.sub('{} /'.format(ax.index), title))
    
    if ax.is_kriging:
        direction = (ax.dir + 1) % 3
    else:
        direction = ax.dir
    
    if direction == 0:
        img = volume[ax.index]
    elif direction == 1:
        img = volume[:,ax.index,:]
    else:
        img = volume[:,:,ax.index]
    
    if not ax.is_kriging:
        img = img.T
    ax.images[0].set_array(img) 
    
    if hasattr(ax, 'add_atoms') and ax.add_atoms:
        plot_atoms(ax)
    
def plot_atoms(ax):
    
    ax.collections = []  # clear existing atoms
    
    # Plot the atoms using the provided color map and marker dict if given    
    for i, el in enumerate(ax.atoms['element'].unique()):

        # get a slice corresponding to the current index
        df_slice = ax.atoms.loc[lambda df: np.array([ax.index in item for item in df['interval']])]

        # get all of the current element rows
        df_el_slice = df_slice.loc[df_slice['element'] == el]    

        # choose marker, using the provided ones if given (one for each unique element)
        if ax.marker_dict is not None:
            try:
                marker = ax.marker_dict[el]
            except KeyError:
                raise ValueError("Marker dict provided has no marker specified for {}".format(el))
        # use the default color list
        else:
            marker = 'o'

        if ax.cmap_atoms:
            color = df_el_slice['BEC']
            vmin, vmax = ax.min_ion_color, ax.max_ion_color
        elif ax.color_dict is None:
            color, vmin, vmax, ax.ion_cmap = None, None, None, None
        else:
            try:
                color = ax.color_dict[el]
                vmin, vmax, ax.ion_cmap = None, None, None
            except KeyError:
                raise ValueError("Color dict provided has no color specified for {}".format(el))

        # plot on 2D axes  
        plot_atoms = ax.scatter(df_el_slice[ax.other_dirs[0]], df_el_slice[ax.other_dirs[1]], c=color, 
                   cmap=ax.ion_cmap, alpha=1, edgecolors='k', label=el, marker=marker,
                   vmin=vmin, vmax=vmax)
    return plot_atoms

def next_direction(ax):
    """Go to the next slice direction."""
    p = re.compile(r'[XYZ](?=\-direction)')
    volume = ax.volume
    ax.dir = (ax.dir + 1) % 3
    slice_dir = DIRS[ax.dir].upper()
    ax.other_dirs = [direct.upper() for direct in list(DIRS_SET - set(slice_dir.lower()))]

    title = ax.get_title()
    ax.set_title(p.sub(DIRS[ax.dir].upper(), title))
    ax.set_xlabel('{} / $\AA$'.format(ax.other_dirs[0]))
    ax.set_ylabel('{} / $\AA$'.format(ax.other_dirs[1]))
    
    if ax.is_kriging:
        direction = (ax.dir + 1) % 3
    else:
        direction = ax.dir
    
    if direction == 0:
        img = volume[ax.index]
    elif direction == 1:
        img = volume[:,ax.index,:]
    else:
        img = volume[:,:,ax.index]
    
    if not ax.is_kriging:
        img = img.T
    ax.images[0].set_array(img) 
     
        
    if hasattr(ax, 'add_atoms') and ax.add_atoms:
        labels = range(1, ax.num_bins + 1)
        ax.atoms['bin'] = pd.cut(ax.atoms[slice_dir], ax.num_bins, labels=labels)
        ax.atoms = ax.atoms.sort_values(by='bin', ascending=True)
        intervals = pd.cut(pd.Series(np.arange(0,ax.res)), ax.num_bins).unique()
        ax.atoms['interval'] = intervals[ax.atoms['bin'].values.astype(int) - 1]
        plot_atoms(ax)


def multi_slice_viewer_BEC(fig, volume, lattice, add_atoms=True, cell=None, color_dict=None, marker_dict=None,
                           cmap_atoms=False, cmap='viridis', ion_cmap='plasma', cbar_pos='right', num_bins=6,
                           resolution=50, is_kriging=False):
    # configure matplotlib to use my defined keystroke callbacks for j,k,l instead of any defaults
    remove_keymap_conflicts({'j', 'k', 'l'})
    ax = fig.add_subplot(111)

    # always start in the x direction
    ax.dir = 0
    slice_dir = DIRS[ax.dir].upper()
    ax.other_dirs = [direct.upper() for direct in list(DIRS_SET - set(slice_dir.lower()))]
    ax.is_kriging = is_kriging

    # get the limits of the lattice along the slice direction
    x_lat = lattice[:, DIRS.index(ax.other_dirs[0].lower())]
    y_lat = lattice[:, DIRS.index(ax.other_dirs[1].lower())]

    # for plotting xlim and ylim
    min_x, max_x = min(x_lat), max(x_lat)
    min_y, max_y = min(y_lat), max(y_lat)

    # the interpolated BECs
    ax.volume = volume

    # start in the middle slice
    if ax.is_kriging:
        ax.index = volume.shape[(ax.dir + 1) % 3] // 2
    else:
        ax.index = volume.shape[ax.dir] // 2

    # add some runtime properties to the ax to store configuration
    ax.add_atoms = add_atoms
    if add_atoms:
        ax.atoms = cell
        ax.color_dict = color_dict
        ax.marker_dict = marker_dict
        ax.min_ion_color = min(cell['BEC'])
        ax.max_ion_color = max(cell['BEC'])
        ax.cmap_atoms = cmap_atoms
        ax.ion_cmap = ion_cmap
        ax.num_bins = num_bins
        ax.res = resolution

    cbar_pos = cbar_pos
    min_volume = np.nanmin(volume)
    max_volume = np.nanmax(volume)

    ax.set_xlabel('{} / $\AA$'.format(ax.other_dirs[0]))
    ax.set_ylabel('{} / $\AA$'.format(ax.other_dirs[1]))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    if ax.is_kriging:
        img = volume[:, ax.index, :]
    else:
        img = volume[ax.index].T

    enhance_plot = ax.imshow(img,
                             origin='lower',
                             extent=(min_x, max_x, min_y, max_y),
                             cmap=cmap,
                             vmin=min_volume,
                             vmax=max_volume)

    if ax.add_atoms:
        labels = range(1, ax.num_bins + 1)
        ax.atoms['bin'] = pd.cut(ax.atoms[slice_dir], ax.num_bins, labels=labels)
        ax.atoms = ax.atoms.sort_values(by='bin', ascending=True)
        intervals = pd.cut(pd.Series(np.arange(0, ax.res)), ax.num_bins).unique()
        ax.atoms['interval'] = intervals[ax.atoms['bin'].values.astype(int) - 1]
        atoms_plot = plot_atoms(ax)
    else:
        atoms_plot = None

    fig.canvas.mpl_connect('key_press_event', process_key)
    return ax, enhance_plot, atoms_plot


def add_colorbar(fig, plot, cbar_pos, shrink=0.5):
    # Now add the colorbar
    pad = 0.0    # helps tight layout
    if cbar_pos == 'top':
        pad = 3.0    # helps tight layout
        cbaxes = fig.add_axes([0.14, 1, 0.8, 0.03])
        cb = fig.colorbar(plot, cax = cbaxes, orientation='horizontal', shrink=shrink)
    elif cbar_pos == 'bottom':
        cb = fig.colorbar(plot, orientation='horizontal', shrink=shrink)
    else:
        cb = fig.colorbar(plot, shrink=shrink)
    return cb, pad


def plot_BEC_heatmap_slices(fig, no_efield, clamped_ion, xyz, matrix_name, np_name, matrix_no_efield,
                            matrix_clamped_ion, matrix_xyz, e_field=.001, e_field_direction=2,
                            interpolation='linear', cbar_pos='top', legend=True, marker_dict=None, pad=0.2,
                            color_dict=None, grid=False, cmap='magma', ion_cmap='plasma', num_bins=6,
                            track_slices=True, res=100, add_atoms=True, cmap_atoms=False, cbar_shrink=1.0):

    ######################################
    # 1. Interpolate for the nanocomposite
    ######################################
    # combine the atoms data into a pandas dataframe
    cell = gen_BEC_df(no_efield, clamped_ion, xyz, e_field=e_field, e_field_direction=e_field_direction)
    lattice = get_lattice(xyz)

    # pad the cell by a few layers according to its periodicity
    cell_ex = pad_cell(cell, lattice, pad=pad)

    # interpolate over a grid of points that spans at least the original unit cell
    NC_interp_BEC = interp_3d(cell_ex[['X', 'Y', 'Z']].values, abs(cell_ex['BEC'].values),
                              resolution=res, lattice=lattice, method=interpolation)

    ######################################
    # 2. Interpolate for the Matrix Only
    ######################################
    # combine the atoms data into a pandas dataframe
    matrix_cell = gen_BEC_df(matrix_no_efield, matrix_clamped_ion, matrix_xyz)

    # pad the cell by a few layers according to its periodicity
    matrix_cell_ex = pad_cell(matrix_cell, lattice, pad=pad)

    # interpolate over a grid of points that spans at least the original unit cell
    matrix_interp_BEC = interp_3d(matrix_cell_ex[['X', 'Y', 'Z']].values, abs(matrix_cell_ex['BEC'].values),
                                  resolution=res, lattice=lattice, method=interpolation)

    ###################################################
    # 3. Take the Difference to measure the enhancement
    ###################################################
    enhancement_grid = NC_interp_BEC - matrix_interp_BEC

    ax, enhance_plot, atoms_plot = multi_slice_viewer_BEC(fig, enhancement_grid, lattice,
                                                          add_atoms=add_atoms,
                                                          cell=cell,
                                                          color_dict=color_dict,
                                                          marker_dict=marker_dict,
                                                          cmap_atoms=cmap_atoms,
                                                          cmap=cmap,
                                                          ion_cmap=ion_cmap,
                                                          cbar_pos=cbar_pos,
                                                          num_bins=num_bins,
                                                          resolution=res,
                                                          is_kriging=(interpolation == 'kriging'))

    if track_slices:
        ax.set_title(
            'BECs for {} Matrix with {} NanoParticle ({} / {})'.format(matrix_name, np_name, res // 2, res))
    else:
        ax.set_title(
            'BECs for {} Matrix with {} NanoParticle ({} / {})'.format(matrix_name, np_name, res // 2, res))

    # Now add the colorbars
    if atoms_plot is not None and cmap_atoms:
        cb_atoms, pad = add_colorbar(fig, atoms_plot, cbar_pos, shrink=cbar_shrink)
        cb_atoms.set_label('Ionic BEC')
    else:
        cb_atoms = None

    cb_enhance, pad = add_colorbar(fig, enhance_plot, cbar_pos, shrink=cbar_shrink)
    cb_enhance.set_label('BEC Enhancement')

    if legend and add_atoms:
        ax.legend()

    if grid:
        ax.grid()

    return ax, cb_enhance, cb_atoms, pad


def krige_grid_search(BEC_df):
    # 3D Kring param opt
    param_dict3d = {"method": ["ordinary3d", "universal3d"],
                    "variogram_model": ["linear", "power", "gaussian", "spherical"],
                    # "nlags": [4, 6, 8],
                    # "weight": [True, False]
                    }

    estimator = GridSearchCV(Krige(), param_dict3d, verbose=False, cv=5, iid=False)

    # Data
    X3 = BEC_df[['X', 'Y', 'Z']].values
    y = BEC_df['BEC'].values

    # run the gridsearch
    estimator.fit(X=X3, y=y)

    if hasattr(estimator, 'best_score_'):
        print('best_score R2 = {:.3f}'.format(estimator.best_score_))
        print('best_params = ', estimator.best_params_)

    print('\nCV results::')
    if hasattr(estimator, 'cv_results_'):
        for key in ['mean_test_score', 'mean_train_score', 'param_method', 'param_variogram_model']:
            print(' - {} : {}'.format(key, estimator.cv_results_[key]))

    return estimator.best_params_


def dipole_field(R, p_vec, centroid):
    r_vec_int = np.array([R[0].ravel(), R[1].ravel(), R[2].ravel()]).T - centroid
    r_mag_int = np.linalg.norm(r_vec_int, axis=1).reshape(-1,1)
    r_unit_int = r_vec_int / r_mag_int
    dipole_field_int = 1/r_mag_int**3*(np.dot(r_unit_int,  3*p_vec.T).reshape(-1, 1) * r_unit_int - p_vec)
    return dipole_field_int.T.reshape(R.shape)

def find_np_atoms(df_matrix, df_np):
    np_pos = df_np[['X', 'Y', 'Z']].values
    matrix_pos = df_matrix[['X', 'Y', 'Z']].values
    
    indices = []
    for i in range(len(np_pos)):
        min_dist = 100
        idx = 0
        for j in range(len(matrix_pos)):
            dist = np.linalg.norm(matrix_pos[j] - np_pos[i])
            if dist < min_dist:
                min_dist = dist
                idx = j
        indices += [idx]
    return indices

def col_major_string(v):
    string = ''
    for k in range(v.shape[2]):
        for j in range(v.shape[1]):
            row = ''
            for i in range(v.shape[0]):
                row += str(v[i, j, k])
                row += ' '
            row = row[:-1]
            row += '\n'
            string += row
        string += '\n'
    return string[:-1]

def arr_2d_string(arr):
    string = ''
    for row in arr:
        for el in row:
            string += str(el)
            string += ' '
        string = string[:-1]
        string += '\n'
    return string[:-1]

def indent(string, num_spaces):
    #string = (' ' * num_spaces) + ('\n' + ' ' * num_spaces).join(string.split('\n'*after_x_newlines))
    string = (' ' * num_spaces) + re.sub(r'\n(?=\S)', '\n' + ' ' * num_spaces, string)
    return string
        

def add_datagrid_3d(input_filename, output_filename, data, span_vectors, name="my3d_data"):
    
    name = ''.join(name.split())
    with open(input_filename, 'r') as f:
        content = f.readlines()    

    if re.search("BEGIN_DATAGRID_3D_{}".format(name), ''.join(content)) is not None:
        print("Datagrid with the name {} already exists, skipping.".format(name))
        return

    # search for existing 3D DATAGRID
    found_existing = False
    for i, line in enumerate(content):
        # if datagrid of the same name already exists, pass
        if "Auto_inserted_data_dont_change_this_line" in line:
            found_existing = True
            string = ("   BEGIN_DATAGRID_3D_{}\n".format(name) +
            "     {} {} {}\n".format(data.shape[0], data.shape[1], data.shape[2]) + "     0.0 0.0 0.0\n" +
            indent(arr_2d_string(span_vectors), 5) + '\n' +
            indent(col_major_string(data), 7) +
            "   END_DATAGRID_3D_{}\n".format(name))
            content.insert(i+1, string)
            break
    if not found_existing:
        string = (" BEGIN_BLOCK_DATAGRID_3D\n   Auto_inserted_data_dont_change_this_line\n" +
        "   BEGIN_DATAGRID_3D_{}\n".format(name) +
        "     {} {} {}\n".format(data.shape[0], data.shape[1], data.shape[2]) +
        "     0.0 0.0 0.0\n" +
        indent(arr_2d_string(span_vectors), 5) + '\n' +
        indent(col_major_string(data), 7) +
        "   END_DATAGRID_3D_{}\n END_BLOCK_DATAGRID_3D".format(name))
        content.append(string)
    
    with open(output_filename, 'w+') as o:
        o.write(''.join(content))
        
def grid_krig_execute(krig_execute):
    def f(grid_x, grid_y, grid_z):
        return krig_execute('grid', grid_x, grid_y, grid_z)
    return f

def apply_chunks(interpolator, grid_x, grid_y, grid_z):
    n1 = grid_x.shape[1]
    ix = da.from_array(grid_x, chunks=(1, n1, n1))
    iy = da.from_array(grid_y, chunks=(1, n1, n1))
    iz = da.from_array(grid_z, chunks=(1, n1, n1))
    iv = da.map_blocks(interpolator, ix, iy, iz, dtype=float)
    return iv.compute()

def testfunc(g1, g2, g3):
    a1, a2, a3 = g1[:,0,0], g2[0,:,0], g3[0,0,:]
    gr1, gr2, gr3 = np.meshgrid(a1, a2, a3, indexing='ij')
    return gr1+gr2+gr3

def apply_kriging_chunks(krig_interp, grid_x, grid_y, grid_z, chunksize=None):
    
    def interp_chunk(interp, g1, g2, g3):
        a1, a2, a3 = g1[:,0,0], g2[0,:,0], g3[0,0,:]
        krig_out, sd_out = interp(a1, a2, a3)
        return krig_out.data.T
    
    if chunksize is None:
        n = grid_x.shape[0]
        chunksize = (1, n, n)
    a_array = da.from_array(grid_x, chunks=chunksize)
    b_array = da.from_array(grid_y, chunks=chunksize)
    c_array = da.from_array(grid_z, chunks=chunksize)
    d = da.map_blocks(interp_chunk, krig_interp, a_array, b_array, c_array, dtype=float)
    return d.compute()
