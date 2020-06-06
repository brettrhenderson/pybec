"""
analysis.py
Module for manipulating data from QuantumEspresso Output after Parsing

Contains all functions for calculating Born Effective Charges
"""


import numpy as np
import pandas as pd
from pybec import utils
import dask.array as da
from pybec import parsers
from scipy.spatial import ConvexHull
from pykrige.rk import Krige
from pykrige.uk3d import UniversalKriging3D
from pykrige.compat import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ, ConstantKernel as C
from scipy.interpolate import Rbf
from scipy.interpolate import NearestNDInterpolator as NND
from scipy.interpolate import LinearNDInterpolator as LND
import logging

log = logging.getLogger(__name__)
#log.setLevel(logging.DEBUG)

A_TO_B = 0.529177249

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
        phi[key] = np.array([np.arccos((pos - centroid)[2] / r_) for pos, r_ in zip(atoms_dict[key], r[key])])
        theta[key] = np.array([np.arctan2((pos - centroid)[1], (pos - centroid)[0]) for pos in atoms_dict[key]])
        theta[key][theta[key] < 0] += 2 * np.pi
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


def infer_local_field(for_0, for_1, z_exp, e_ext=0.001, e_field_direction=[0, 0, 1]):
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


def get_field_along_d(field_dict, sub_mean_field=False, e_field=0.25, e_field_direction=[0, 0, 1]):
    """
    Calculate the electric field along a specific direction from the FE results.

    Parameters
    ----------
    field_dict : dict
        Electric field at atomic locations in a dictionary where the keys are the element
        symbols and the values are the numpy array of the electric field for all atoms of that element.
    sub_mean_field : bool, optional
        If set, the external applied field is subtracted from the calculated fields, meaning that only the local field
        disturbance caused by the inclusion will be plotted. Defaults to False.
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


def get_dipole_field(coords, dipole_loc=[0, 0, 0], p_vec=[0, 0, 1], p=1, is_angstrom=True):
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

    ions = utils.as_dataframe(coords)
    field = {}
    for key in coords:
        r_vec = ions.loc[ions['element'] == key][['X', 'Y', 'Z']].values - dipole_loc
        r_mag = np.linalg.norm(r_vec, axis=1).reshape(-1, 1)
        r_unit = r_vec / r_mag
        field[key] = 1 / r_mag ** 3 * (np.dot(r_unit, 3 * p_vec.T).reshape(-1, 1) * r_unit - p_vec)
    return field


def get_dipole_field_displaced(coords, dipole_loc=[0, 0, 0], p_vec=[0, 0, 1], q=1, d=0.1, is_angstrom=True):
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

    ions = utils.as_dataframe(coords)
    field = {}
    for key in coords:
        ppos = dipole_loc + d / 2 * p_vec
        pneg = dipole_loc - d / 2 * p_vec
        r_pos = ions.loc[ions['element'] == key][['X', 'Y', 'Z']].values - ppos
        r_pos_mag = np.linalg.norm(r_pos, axis=1).reshape(-1, 1)
        r_neg = ions.loc[ions['element'] == key][['X', 'Y', 'Z']].values - pneg
        r_neg_mag = np.linalg.norm(r_neg, axis=1).reshape(-1, 1)
        r_pos_unit = r_pos / r_pos_mag
        r_neg_unit = r_neg / r_neg_mag
        pos_field = q / r_pos_mag ** 2 * r_pos_unit
        neg_field = -q / r_neg_mag ** 2 * r_neg_unit
        field[key] = pos_field + neg_field
    return field


def gen_BEC_df(no_efield, clamped_ion, xyz, e_field=0.001, e_field_direction=[0, 0, 1], add_forces=False):
    # parse coordinates
    coords = parsers.get_coordinates(xyz)

    # parse forces
    for_0, for_1 = parsers.get_converged_forces(no_efield), parsers.get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = get_BECs(for_0, for_1, e_field, e_field_direction)
    if add_forces:
        return utils.as_dataframe(coords, BEC, for_0, for_1)
    else:
        return utils.as_dataframe(coords, BEC)


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
    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull)
    if inc:
        return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)
    else:  # don't include points that lie within a tolerance of a facet of the hull
        for eq in hull.equations:
            # the point is within a certain tolerance of a facet of the hull
            if (np.dot(eq[:-1], point) + eq[-1] <= tolerance) & (np.dot(eq[:-1], point) + eq[-1] >= -tolerance):
                return False
            elif (np.dot(eq[:-1], point) + eq[-1] > tolerance):  # outside the hull
                return False
        return True


def points_in_hull(points, hull, tolerance=1e-12, inc=False):
    if not isinstance(hull, ConvexHull):
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
    shifts = []  # if the direction is negative, we need to add on a shift
    combos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T

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
    vertices = np.dot(scaled_lattice_vecs, combos).T  # the vertices of a the selection box

    # if the origin is shifted, then add in the shifts
    for shift in np.array(shifts):
        vertices = vertices + shift

    # make the in_bounds inclusive if we are on the outer part of the unit cell,
    # otherwise, leave it exclusive
    #     inc=False
    #     if len(shifts):
    #         inc = True

    # now, find all points in the original cell that lie inside these vertices
    in_bounds = points_in_hull(cell_seg[['X', 'Y', 'Z']].values, vertices)  # inc=inc)
    cell_seg = cell_seg[in_bounds]

    return cell_seg


def pad_cell(cell_df, lattice, pad=0.4):
    # Expand periodic boundaries by one lattice vector in every direction
    shifts = [-1, 0, 1]

    cell_expand = cell_df.copy(deep=True)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i, j, k) == (1, 1, 1):  # corresponds to the center unit cell, don't repeat it
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
                        cell_temp[['X', 'Y', 'Z']] += vec
                    log.debug('5. shifted selected cell df: \n{}'.format(cell_temp))
                    cell_expand = pd.concat([cell_expand, cell_temp])
                    log.debug('6. Expanded cell df: \n{}'.format(cell_expand))

    return cell_expand


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


def apply_kriging_chunks(krig_interp, grid_x, grid_y, grid_z, chunksize=None):
    def interp_chunk(interp, g1, g2, g3):
        a1, a2, a3 = g1[:, 0, 0], g2[0, :, 0], g3[0, 0, :]
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


def interp_3d(coord_arr, val_arr, resolution=100j, lattice=None, xlim=(0, 1),
              ylim=(0, 1), zlim=(0, 1), method='linear', return_grids=False):
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

    x, y, z = coord_arr[:, 0], coord_arr[:, 1], coord_arr[:, 2]
    points = np.array(list(zip(x, y, z)))

    # grid on which to evaluate interpolators
    grid_x, grid_y, grid_z = np.mgrid[min_x:max_x:resolution * 1j, min_y:max_y:resolution * 1j,
                             min_z:max_z:resolution * 1j]

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
                                             grid_z.ravel())))).reshape(resolution, resolution, resolution)
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
    r_mag_int = np.linalg.norm(r_vec_int, axis=1).reshape(-1, 1)
    r_unit_int = r_vec_int / r_mag_int
    dipole_field_int = 1 / r_mag_int ** 3 * (np.dot(r_unit_int, 3 * p_vec.T).reshape(-1, 1) * r_unit_int - p_vec)
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