"""
utils.py
Module with some helpful methods for converting between data structures.

The rest of the package often utilizes atomic data in the form of either pandas Dataframes or
dictionaries with element symbols for keys.  Functions in this module convert between these
two representations
"""
import pandas as pd
import numpy as np


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
        d = {'element': np.array([key for _ in atoms_dict[key]]),
             'X': atoms_dict[key][:, 0],
             'Y': atoms_dict[key][:, 1],
             'Z': atoms_dict[key][:, 2]}
        if BEC is not None:
            d['BEC'] = BEC[key]
        if for0 is not None:
            d['Force 0x'] = for0[key][:, 0]
            d['Force 0y'] = for0[key][:, 1]
            d['Force 0z'] = for0[key][:, 2]
        if for1 is not None:
            d['Force 1x'] = for1[key][:, 0]
            d['Force 1y'] = for1[key][:, 1]
            d['Force 1z'] = for1[key][:, 2]
        df_temp = pd.DataFrame(data=d)
        df = df.append(df_temp, ignore_index=True)
    return df


def df_to_dicts(df):
    coords = {}
    BECs = {}
    for el in df['element'].unique():
        coords[el] = df[df['element'] == el][['X', 'Y', 'Z']].values
        BECs[el] = df[df['element'] == el]['BEC'].values
    return coords, BECs