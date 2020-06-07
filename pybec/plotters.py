"""
plotters.py
Several convenience functions for plotting Born Effective Charges in various ways.
"""

import numpy as np
import matplotlib.pyplot as plt
import pybec.parsers as parsers
import pybec.analysis as analysis
import os
import re
import pandas as pd

# Default Markers to use when plotting different elements
MARKERS = 'o s h + x * p D v ^ < >'

# x = 0, y = 1, z = 2
DIRS = ['x', 'y', 'z']

#keep track of possible slice directions as a set
DIRS_SET = set(DIRS)


def plot_BEC_v_spher(no_efield, clamped_ion, xyz, matrix_name, np_name, np_element, to_plot='r', centroid_of='all', 
                    e_field=.001, e_field_direction=[0,0,1], cmap=None, cbar_pos = 'top', legend=True, figsize=(8,4),
                    marker_dict=None, color_dict=None, grid=False, alpha=1.0, cbar_shrink=1.0):
    """
    Plots the Born Effective Charges for each ion against the distance of that ion from the nanoparticle centroid.
    
    Parameters
    ---------
    no_efield : str
        File path to the zero field step output file
    clamped_ion : str
        File path to the clamped ion step output file
    xyz : str
        File path to the optimized structure .xyz file
    matrix_name : str
        Whatever you want to call the matrix, e.g. 'MgO'
    np_name : str
        Whatever you want to call the mNP, e.g. 'Ag8_333'
    centroid_of : str, optional
        The element you want to calculate the centroid for when calculating and plotting against
        the distance to centroid.  If 'all', it will calculate the centroid of the whole unit cell.
        'Ag' will calculate the centroid of all silver ions in the unit cell in the relaxed structure.
        Default: 'all'
    e_field : float, optional
        The electric field in au. Default: 0.001
    e_field_direction : int, optional
        The direction the field is applied, where the x-axis is [1,0,0], y-axis is [0,1,0], and z-axis is [0,0,1].
        Default: [0,0,1]
    cmap : string, optional
        The matplotlib colormap style used to color the absolute value of their Born Effective Charge.
        If None, the data instead will be colored by element using the color_dict colors or default matplotlib ones.
        Default: None
    cbar_pos : str, optional
        Where to place the colorbar if cmap is used.  Can be 'top', 'right', or 'bottom'. Default: 'top'
    legend : bool, optional
        Whether to include a legend labelling the different elements. Default: True
    figsize : (int, int), optional
        Width, Height of the figure. Default: (8,4)
    marker_dict : dict, optional
        Custom dictionary specifying which markers to use for each element, where the keys
        are element symbols and the values are valid matplotlib marker symbols. Default: None
        Ex: {'O': 'o', 'Ag': 'x', 'Mg': '>'}
    color_dict : dict, optional
        Custom dictionary specifying which colors to use for each element, where the keys
        are element symbols and the values are valid matplotlib color symbols. Default: None
        Ex: {'O': 'r', 'Ag': 'k', 'Mg': 'y'}
    grid : bool, optional
        Whether to include grid lines in the plot. Default: False
    alpha : float
        The alpha channel value for the plot colors. Default: 1.0
    
    Returns
    -------
    None
    
    """

    # parse coordinates
    coords = parsers.get_coordinates(xyz)

    # calculate unit cell centroid:
    centroid = analysis.get_centroid(coords, centroid_of)

    # calculate the distances of all ions to centroid
    r, phi, theta = analysis.get_spherical_coords_to_centroid(coords, centroid)
    if to_plot == 'theta':
        to_plot_ = theta
    elif to_plot == 'phi':
        to_plot_ = phi
    else:
        to_plot_ = r   
    max_dist = max([max(to_plot_[key]) for key in to_plot_])  # for plotting
    min_dist = min([min(to_plot_[key]) for key in to_plot_])  # for plotting

    # parse forces
    for_0, for_1 = parsers.get_converged_forces(no_efield), parsers.get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = analysis.get_BECs(for_0, for_1, e_field, e_field_direction)
    # for colormap
    max_BEC = max([max(abs(BEC[key])) for key in BEC if key != np_element])  # for plotting
    min_BEC = min([min(abs(BEC[key])) for key in BEC if key != np_element])  # for plotting

    # plot the BECs against distance from centroid
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i, el in enumerate(BEC):
        # choose marker, using the provided ones if given
        if marker_dict is not None:
            try:
                marker = marker_dict[el]
            except KeyError:
                raise ValueError("Marker dict provided has no marker specified for {}".format(el))
        # use the default color list
        else:
            marker = MARKERS[i % len(MARKERS)]

        # if a colormap is desired
        if cmap is not None and el != np_element:
            plot = ax.scatter(to_plot_[el], BEC[el], c=np.abs(BEC[el]), cmap=cmap, 
                        alpha=alpha, edgecolors='k', label=el, marker=marker, vmin=min_BEC, vmax=max_BEC)
        # else just group elements by color
        else:
            color = None
            if el == np_element:
                color = 'k'
            # use custom color dictionary if provided
            if color_dict is not None:
                try:
                    color = color_dict[el]
                except KeyError:
                    raise ValueError("Color dict provided has no color specified for {}".format(el))
            ax.scatter(to_plot_[el], BEC[el], alpha=alpha, c=color, edgecolors='k',
                        label=el, marker=marker, vmin=min_BEC, vmax=max_BEC)

    ax.plot(np.linspace(0, max_dist + 1, 2), np.zeros(2), 'k:')
    if to_plot == 'r': 
        ax.set_xlabel('Distance from {} mNP Centroid / $\AA$'.format(np_name))
    if to_plot == 'theta': 
        ax.set_xlabel('Azimuthal Angle / $rad$'.format(np_name))
    if to_plot == 'phi': 
        ax.set_xlabel('Polar Angle / $rad$'.format(np_name))
    ax.set_ylabel('$Z^*$')
    #ax.set_ylabel('$Z^*_{{{}{}}}$'.format(DIRS[e_field_direction], DIRS[e_field_direction]))
    ax.set_title('Born Effective Charges for {} Matrix with {} NanoParticle'.format(matrix_name, np_name))
    buffer = (max_dist - min_dist) * 0.05
    ax.set_xlim([min_dist - buffer, max_dist + buffer])

    # Now add the colorbar if specified
    if cmap is not None:
        cb, pad = add_colorbar(fig, plot, cbar_pos, shrink=cbar_shrink)

    if legend:
        ax.legend(loc='right')
    
    if grid:
        ax.grid()

    return fig, ax, cb, pad


def plot_FE_E_v_spher(xyz, xyzE, matrix_name, np_name, np_element, to_plot='r', centroid_of='all', sub_mean_field=False,
                    e_field=.25, e_field_direction=[0,0,1], center_dict=None, ref_neg=True, cmap=None, cbar_pos = 'top', 
                    legend=True, figsize=(8,4), marker_dict=None, color_dict=None, grid=False, alpha=1.0, cbar_shrink=1.0,
                    units="au"):
    """
    Plots the electric field predicted on each ion by a continuum (FE) approach.
    
    Parameters
    ---------

    xyz : str
        File path to the optimized structure .xyz file
    xyzE : str
        File path to the file containing the FE efield for each atomic coordinate, or
        Dictionary of electric field values, where the keys are the element symbols,
        and the values are the Nx3 numpy array of electric field values for all atoms of that element.
        Ex: xyzE=get_dipole_field(get_coordinates(xyz), get_centroid(blah, key='Ag'), q=1)
    matrix_name : str
        Whatever you want to call the matrix, e.g. 'MgO'
    np_name : str
        Whatever you want to call the mNP, e.g. 'Ag8_333'
    centroid_of : str, optional
        The element you want to calculate the centroid for when calculating and plotting against
        the distance to centroid.  If 'all', it will calculate the centroid of the whole unit cell.
        'Ag' will calculate the centroid of all silver ions in the unit cell in the relaxed structure.
        defaults to 'all'
    sub_mean_field: bool, optional
        If set, the external applied field is subtracted from the calculated fields, meaning that only the local field
        disturbance caused by the inclusion will be plotted. defaults to True
    e_field : float, optional
        The applied electric field in V/m. Default: 0.25
    e_field_direction : int, optional
        The direction the field is applied, where the x-axis is [1,0,0], y-axis is [0,1,0], and z-axis is [0,0,1].
        Default: [0,0,1]
    center_dict : dict, optional
        Custom dictionary specifying where to center each element on the y-axis of the plot, where the keys are element
        symbols and the values are floats or integers. Default: None
        Ex: {'O': -2, 'Mg': 2, 'Ag': 0}
    cmap : string, optional
        The matplotlib colormap style used to color the absolute value of their Born Effective Charge.
        If None, the data instead will be colored by element using the color_dict colors or default matplotlib ones.
        Default: None
    cbar_pos : str, optional
        Where to place the colorbar if cmap is used.  Can be 'top', 'right', or 'bottom'. Default: 'top'
    legend : bool, optional
        Whether to include a legend labelling the different elements. Default: True
    figsize : (int, int), optional
        Width, Height of the figure. Default: (8,4)
    marker_dict : dict, optional
        Custom dictionary specifying which markers to use for each element, where the keys
        are element symbols and the values are valid matplotlib marker symbols. Default: None
        Ex: {'O': 'o', 'Ag': 'x', 'Mg': '>'}
    color_dict : dict, optional
        Custom dictionary specifying which colors to use for each element, where the keys
        are element symbols and the values are valid matplotlib color symbols. Default: None
        Ex: {'O': 'r', 'Ag': 'k', 'Mg': 'y'}
    grid : bool, optional
        Whether to include grid lines in the plot. Default: False
    alpha : float, optional
        The alpha channel value for the plot colors. Default: 1.0
    units : str, optional
        The units for electric field to be added to the y-axis. Default: "au"
    
    Returns
    -------
    None
    
    """

    # parse coordinates
    coords = parsers.get_coordinates(xyz)
    
    # parse Efield from the FE simulation
    if isinstance(xyzE, str):
        if not os.path.isfile(xyzE):
            raise ValueError('Invalid file path given for electric field at atomic coordinates.')
        FE_Efield = parsers.get_coordinates(xyzE, skip=1, d=',')
    elif isinstance(xyzE, dict):  # efield already read in or generated some other way 
        FE_Efield = xyzE
    else:
        raise ValueError('xyzE must either be file path for electric field in .xyz format or dictionary with'
                         'elements for keys and Nx3 ndarray of efield for values')

    # calculate unit cell centroid:
    centroid = analysis.get_centroid(coords, centroid_of)

    # calculate the distances of all ions to centroid
    r, phi, theta = analysis.get_spherical_coords_to_centroid(coords, centroid)
    if to_plot == 'theta':
        to_plot_ = theta
    elif to_plot == 'phi':
        to_plot_ = phi
    else:
        to_plot_ = r   
    max_dist = max([max(to_plot_[key]) for key in to_plot_])  # for plotting
    min_dist = min([min(to_plot_[key]) for key in to_plot_])  # for plotting

    # calculate field in desired direction
    if FE_Efield[list(FE_Efield)[0]].ndim > 1:
        field_to_plot = analysis.get_field_along_d(FE_Efield, sub_mean_field, e_field, e_field_direction)
    else:
        field_to_plot = FE_Efield    
    
    max_ftp = max([max(abs(field_to_plot[key])) for key in field_to_plot if key != np_element])  # for plotting
    min_ftp = min([min(abs(field_to_plot[key])) for key in field_to_plot if key != np_element])  # for plotting
    
    # center different elements at different points on y-axis if desired
    if center_dict is not None:
        for i, el in enumerate(field_to_plot):
            try:
                # also need to reflect data vertically to mimic greater magnitude of BEC with bigger EField
                if center_dict[el] < 0 and ref_neg:
                    field_to_plot[el] *= -1
                field_to_plot[el] += center_dict[el]
            except KeyError:
                raise ValueError("Center dict provided has no center specified for {}".format(el))
    else:
        center_dict = {el: 0 for el in field_to_plot}
    
    # for colormap
    # old way
    # max_ftp = max([max(abs(field_to_plot[key])) for key in field_to_plot if key != np_element])  # for plotting
    # min_ftp = min([min(abs(field_to_plot[key])) for key in field_to_plot if key != np_element])  # for plotting
    
    
    # plot the BECs against desired spherical coordinate
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for i, el in enumerate(field_to_plot):
        # choose marker, using the provided ones if given
        if marker_dict is not None:
            try:
                marker = marker_dict[el]
            except KeyError:
                raise ValueError("Marker dict provided has no marker specified for {}".format(el))
        # use the default color list
        else:
            marker = MARKERS[i % len(MARKERS)]

        # if a colormap is desired
        if cmap is not None and el != np_element:
            # old coloring option: np.abs(field_to_plot[el])
            plot = ax.scatter(to_plot_[el], field_to_plot[el], 
                              c=(field_to_plot[el] - center_dict[el])*(-1)**(center_dict[el]<0), cmap=cmap, 
                              alpha=alpha, edgecolors='k', label=el, marker=marker, vmin=min_ftp, vmax=max_ftp)
        # else just group elements by color
        else:
            color = None
            if el == np_element:
                color = 'k'
            # use custom color dictionary if provided
            if color_dict is not None:
                try:
                    color = color_dict[el]
                except KeyError:
                    raise ValueError("Color dict provided has no color specified for {}".format(el))
            ax.scatter(to_plot_[el], field_to_plot[el], alpha=alpha, c=color, edgecolors='k',
                        label=el, marker=marker, vmin=min_ftp, vmax=max_ftp)

    ax.plot(np.linspace(0, max_dist + 1, 2), np.zeros(2), 'k:')
    if to_plot == 'r': 
        ax.set_xlabel('Distance from {} mNP Centroid / $\AA$'.format(np_name))
    if to_plot == 'theta': 
        ax.set_xlabel('Azimuthal Angle / $rad$'.format(np_name))
    if to_plot == 'phi': 
        ax.set_xlabel('Polar Angle / $rad$'.format(np_name))
    ax.set_ylabel(f'$E$ / ${units}$')
    #ax.set_ylabel('$Z^*_{{{}{}}}$'.format(DIRS[e_field_direction], DIRS[e_field_direction]))
    ax.set_title('Electric Field for {} Matrix with {} NanoParticle'.format(matrix_name, np_name))
    buffer = (max_dist - min_dist) * 0.05
    ax.set_xlim([min_dist - buffer, max_dist + buffer])

    # Now add the colorbar if specified
    if cmap is not None:
        cb, pad = add_colorbar(fig, plot, cbar_pos, shrink=cbar_shrink)

    if legend:
        ax.legend(loc='right')
    
    if grid:
        ax.grid()

    return fig, ax, cb, pad
    
    
def plot_BEC_3D(no_efield, clamped_ion, xyz, matrix_name, np_name, e_field=.001,
             e_field_direction=2, cbar_pos = 'top', legend=True, figsize=(8,8),
             marker_dict=None, grid=False, cmap='magma', alpha=1.0, cbar_shrink=1.0):
    
    # parse coordinates
    coords = parsers.get_coordinates(xyz)
    
    # parse forces
    for_0, for_1 = parsers.get_converged_forces(no_efield), parsers.get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = analysis.get_BECs(for_0, for_1, e_field, e_field_direction)
    
    # for colormap
    max_BEC = max([max(BEC[key]) for key in BEC])  # for plotting
    min_BEC = min([min(BEC[key]) for key in BEC])  # for plotting
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    for i, el in enumerate(BEC):
        # choose marker, using the provided ones if given
        if marker_dict is not None:
            try:
                marker = marker_dict[el]
            except KeyError:
                raise ValueError("Marker dict provided has no marker specified for {}".format(el))
        # use the default color list
        else:
            marker = MARKERS[i % len(MARKERS)]
    
        # plot on 3D axes
        plot = ax.scatter(coords[el][:,0], coords[el][:,1], coords[el][:,2], c=BEC[el], cmap=cmap, 
                   alpha=alpha, edgecolors='k', label=el, marker=marker, vmin=min_BEC, vmax=max_BEC)

    ax.set_xlabel('X / $\AA$')
    ax.set_ylabel('Y / $\AA$')
    ax.set_zlabel('Z / $\AA$')
    ax.set_title('Born Effective Charges for {} Matrix with {} NanoParticle'.format(matrix_name, np_name))
    
    # Now add the colorbar if specified
    cb, pad = add_colorbar(fig, plot, cbar_pos, shrink=cbar_shrink)

    if legend:
        ax.legend()
    
    if grid:
        ax.grid()

    return fig, ax, cb, pad


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume, fig):
    remove_keymap_conflicts({'j', 'k', 'l'})
    ax = fig.add_subplot(111)
    ax.volume = volume
    ax.dir = 0
    ax.index = volume.shape[ax.dir] // 2
    ax.set_title('Slices in the X-direction')
    ax.set_xlabel('Y / $\AA$')
    ax.set_ylabel('Z / $\AA$')
    plot = ax.imshow(volume[ax.index].T, origin='bottom', )
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
        img = volume[:, ax.index, :]
    else:
        img = volume[:, :, ax.index]

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
        img = volume[:, ax.index, :]
    else:
        img = volume[:, :, ax.index]

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
        img = volume[:, ax.index, :]
    else:
        img = volume[:, :, ax.index]

    if not ax.is_kriging:
        img = img.T
    ax.images[0].set_array(img)

    if hasattr(ax, 'add_atoms') and ax.add_atoms:
        labels = range(1, ax.num_bins + 1)
        ax.atoms['bin'] = pd.cut(ax.atoms[slice_dir], ax.num_bins, labels=labels)
        ax.atoms = ax.atoms.sort_values(by='bin', ascending=True)
        intervals = pd.cut(pd.Series(np.arange(0, ax.res)), ax.num_bins).unique()
        ax.atoms['interval'] = intervals[ax.atoms['bin'].values.astype(int) - 1]
        plot_atoms(ax)


def multi_slice_viewer_BEC(fig, volume, lattice, add_atoms=True, cell=None, color_dict=None, marker_dict=None,
                           cmap_atoms=False, cmap='viridis', ion_cmap='plasma', num_bins=6,
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
    pad = 0.0  # helps tight layout
    if cbar_pos == 'top':
        pad = 3.0  # helps tight layout
        cbaxes = fig.add_axes([0.14, 1, 0.8, 0.03])
        cb = fig.colorbar(plot, cax=cbaxes, orientation='horizontal', shrink=shrink)
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
    cell = analysis.gen_BEC_df(no_efield, clamped_ion, xyz, e_field=e_field, e_field_direction=e_field_direction)
    lattice = parsers.get_lattice(xyz)

    # pad the cell by a few layers according to its periodicity
    cell_ex = analysis.pad_cell(cell, lattice, pad=pad)

    # interpolate over a grid of points that spans at least the original unit cell
    NC_interp_BEC = analysis.interp_3d(cell_ex[['X', 'Y', 'Z']].values, abs(cell_ex['BEC'].values),
                                       resolution=res, lattice=lattice, method=interpolation)

    ######################################
    # 2. Interpolate for the Matrix Only
    ######################################
    # combine the atoms data into a pandas dataframe
    matrix_cell = analysis.gen_BEC_df(matrix_no_efield, matrix_clamped_ion, matrix_xyz)

    # pad the cell by a few layers according to its periodicity
    matrix_cell_ex = analysis.pad_cell(matrix_cell, lattice, pad=pad)

    # interpolate over a grid of points that spans at least the original unit cell
    matrix_interp_BEC = analysis.interp_3d(matrix_cell_ex[['X', 'Y', 'Z']].values, abs(matrix_cell_ex['BEC'].values),
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