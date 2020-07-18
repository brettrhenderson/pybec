"""
output.py
Module for outputting volumetric data to standard formats

Includes functions to output 3D grid data to the XCrysden file format, which can be read in by XCrysden, VESTA,
or other visualization software.
"""

import re


def print_coordinates(coords, xyz_file, comment='', format='block', unit='angstrom'):
    """
    Prints the XYZ coordinates to a specified file.

    Parameters
    ----------
    coords : collections.OrderedDict
        The coordinates in angstroms in a dictionary where the keys
        are the element symbols and the values are the numpy
        arrays of the coordinates for all atoms of that element.
    xyz_file : str
        File path to the output .xyz file
    comment : str
        Comment to include in line 2 of xyz file

    Returns
    -------
    None
    """

    with open(xyz_file, 'w+') as f:
        if format == 'xyz':
            f.write(f'{sum([len(coords[key]) for key in coords]):12}\n')
            f.write(comment.replace('\n', '') + '\n')
        else:
            f.write(f'ATOMIC_POSITIONS ({unit})\n')
        for key in coords:
            for atom in coords[key]:
                f.write(f'{key:2}      {atom[0]:12.9f}   {atom[1]:12.9f}   {atom[2]:12.9f}\n')


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
    # string = (' ' * num_spaces) + ('\n' + ' ' * num_spaces).join(string.split('\n'*after_x_newlines))
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
            content.insert(i + 1, string)
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
