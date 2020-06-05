"""
plotters.py
Several convenience functions for plotting Born Effective Charges in various ways.
"""

import numpy as np
import matplotlib.pyplot as plt
from pybec.pybec import *


def plot_BEC_v_spher(no_efield, clamped_ion, xyz, matrix_name, np_name, np_element, to_plot='r', centroid_of='all', 
                    e_field=.001, e_field_direction=[0,0,1], cmap=None, cbar_pos = 'top', legend=True, figsize=(8,4),
                    marker_dict=None, color_dict=None, grid=False, alpha=1.0, cbar_shrink=1.0):
    """
    Plots the Born Effective Charges for each ion against the distance of that ion from the nanoparticle centroid.
    
    Parmeters
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
    centroid_of : str, optional, default: 'all'
        The element you want to calculate the centroid for when calculating and plotting against
        the distance to centroid.  If 'all', it will calculate the centroid of the whole unit cell.
        'Ag' will calculate the centroid of all silver ions in the unit cell in the relaxed structure.
    e_field : float, optional, default: 0.001
        The electric field in au.
    e_field_direction : int, optional, default: [0,0,1]
        The direction the field is applied, where the x-axis is [1,0,0], y-axis is [0,1,0], and z-axis is [0,0,1].
    cmap : string, optional, default: None
        The matplotlib colormap style used to color the absolute value of their Born Effective Charge.
        If None, the data instead will be colored by element using the color_dict colors or default matplotlib ones.
    cbar_pos : str, optional, default: 'top'
        Where to place the colorbar if cmap is used.  Can be 'top', 'right', or 'bottom'
    legend : bool, optional, default: True
        Whether to include a legend labelling the different elements.
    figsize : (int, int), optional, defalt: (8,4)
        Width, Height of the figure.
    marker_dict : dict, optional, default: None
        Custom dictionary specifying which markers to use for each element, where the keys
        are element symbols and the values are valid matplotlib marker symbols.
        Ex: {'O': 'o', 'Ag': 'x', 'Mg': '>'}
    color_dict : dict, optional, default: None
        Custom dictionary specifying which colors to use for each element, where the keys
        are element symbols and the values are valid matplotlib color symbols.
        Ex: {'O': 'r', 'Ag': 'k', 'Mg': 'y'}
    grid : bool, optional, default: False
        Whether to include grid lines in the plot
    alpha : float, default: 1.0
        The alpha channel value for the plot colors
    
    Returns
    -------
    None
    
    """

    # parse coordinates
    coords = get_coordinates(xyz)

    # calculate unit cell centroid:
    centroid = get_centroid(coords, centroid_of)

    # calculate the distances of all ions to centroid
    r, phi, theta = get_spherical_coords_to_centroid(coords, centroid)
    if to_plot == 'theta':
        to_plot_ = theta
    elif to_plot == 'phi':
        to_plot_ = phi
    else:
        to_plot_ = r   
    max_dist = max([max(to_plot_[key]) for key in to_plot_])  # for plotting
    min_dist = min([min(to_plot_[key]) for key in to_plot_])  # for plotting

    # parse forces
    for_0, for_1 = get_converged_forces(no_efield), get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = get_BECs(for_0, for_1, e_field, e_field_direction)
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
    centroid_of : str, optional, default: 'all'
        The element you want to calculate the centroid for when calculating and plotting against
        the distance to centroid.  If 'all', it will calculate the centroid of the whole unit cell.
        'Ag' will calculate the centroid of all silver ions in the unit cell in the relaxed structure.
    sub_mean_field: bool, optional, default: True
        If set, the external applied field is subtracted from the calculated fields, meaning that only the local field
        disturbance caused by the inclusion will be plotted.
    e_field : float, optional, default: 0.25
        The applied electric field in V/m.
    e_field_direction : int, optional, default: [0,0,1]
        The direction the field is applied, where the x-axis is [1,0,0], y-axis is [0,1,0], and z-axis is [0,0,1].
    center_dict : dict, optional, default: None
        Custom dictionary specifying where to center each element on the y-axis of the plot, where the keys are element
        symbols and the values are floats or integers.
        Ex: {'O': -2, 'Mg': 2, 'Ag': 0}
    cmap : string, optional, default: None
        The matplotlib colormap style used to color the absolute value of their Born Effective Charge.
        If None, the data instead will be colored by element using the color_dict colors or default matplotlib ones.
    cbar_pos : str, optional, default: 'top'
        Where to place the colorbar if cmap is used.  Can be 'top', 'right', or 'bottom'
    legend : bool, optional, default: True
        Whether to include a legend labelling the different elements.
    figsize : (int, int), optional, defalt: (8,4)
        Width, Height of the figure.
    marker_dict : dict, optional, default: None
        Custom dictionary specifying which markers to use for each element, where the keys
        are element symbols and the values are valid matplotlib marker symbols.
        Ex: {'O': 'o', 'Ag': 'x', 'Mg': '>'}
    color_dict : dict, optional, default: None
        Custom dictionary specifying which colors to use for each element, where the keys
        are element symbols and the values are valid matplotlib color symbols.
        Ex: {'O': 'r', 'Ag': 'k', 'Mg': 'y'}
    grid : bool, optional, default: False
        Whether to include grid lines in the plot
    alpha : float, default: 1.0
        The alpha channel value for the plot colors
    units : str, optional, default: "au"
        The units for electric field to be added to the y-axis.
    
    Returns
    -------
    None
    
    """

    # parse coordinates
    coords = get_coordinates(xyz)
    
    # parse Efield from the FE simulation
    if isinstance(xyzE, str):
        if not os.path.isfile(xyzE):
            raise ValueError('Invalid file path given for electric field at atomic coordinates.')
        FE_Efield = get_coordinates(xyzE, skip=1, d=',')
    elif isinstance(xyzE, dict):  # efield already read in or generated some other way 
        FE_Efield = xyzE
    else:
        raise ValueError('xyzE must either be file path for electric field in .xyz format or dictionary with'
                         'elements for keys and Nx3 ndarray of efield for values')

    # calculate unit cell centroid:
    centroid = get_centroid(coords, centroid_of)

    # calculate the distances of all ions to centroid
    r, phi, theta = get_spherical_coords_to_centroid(coords, centroid)
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
        field_to_plot = get_field_along_d(FE_Efield, sub_mean_field, e_field, e_field_direction)
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
    coords = get_coordinates(xyz)
    
    # parse forces
    for_0, for_1 = get_converged_forces(no_efield), get_converged_forces(clamped_ion)

    # calculate Born Effective Charges
    BEC = get_BECs(for_0, for_1, e_field, e_field_direction)
    
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
