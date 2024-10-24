# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:12:28 2021

@author: albdag
"""

from collections import Counter
import os
from typing import Any, Iterable

import numpy as np
import PIL
import tifffile


def shareAxis(ax, target, share=True): # deprecated! Moved to _CanvasBase -> share_axis method
    shared_X = target._shared_axes['x']
    shared_Y = target._shared_axes['y']
# For older version of Matplotlib
    # shared_X = target._shared_x_axes
    # shared_Y = target._shared_y_axes
    if share:
        shared_X.join(target, ax)
        shared_Y.join(target, ax)
    else:
        shared_X.remove(ax)
        shared_Y.remove(ax)


def get_dictkey(dictionary: dict, value: Any, default: Any|None=None):
    '''
    Get a dictionary key from a given value. If multiple keys are linked to the
    same value, only the first in the list will be returned.

    Parameters
    ----------
    dictionary : dict
        The input dictionary.
    value : Any type
        The filter value to search the key.
    default : Any type | None, optional
        If no key is found, this value is returned. The default is None.

    Returns
    -------
    key or default : Any type
        The corresponding key or <default> if no key is found.

    '''
    keys = [k for k, v in dictionary.items() if v == value]
    if keys: return keys[0]
    return default


def sort_dict_by_list(dictionary: dict, ordered_list: list, mode='keys'):
    '''
    Sort a dictionary according to a desidered ordered list of its keys or 
    values.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be ordered.
    ordered_list : list
        The ordered list used to sort the dictionary. It cannot contain keys or
        values not included in the dictionary, but it can have just some of 
        them. In this case, the returned sorted dictionary will only contain 
        the listed keys/values.
    mode : str, optional
        Whether to interpret the ordered list as the dictionary 'keys' list
        or the dictionary 'values' list. The default is 'keys'.

    Raises
    ------
    ValueError
        The mode parameter must be one of ('keys', 'values').

    Returns
    -------
    dict
        The sorted dictionary.

    '''
    if mode == 'keys':
        return {k: dictionary[k] for k in ordered_list}

    elif mode == 'values':
        return {get_dictkey(dictionary, v): v for v in ordered_list}
    
    else:
        raise ValueError(f'{mode} is not a valid mode.')


# def get_mode(array, ordered=True, unique_indices=False, np_axis=None): # deprecated! Moved to MineralMap base class
#     '''
#      A function to calculate the modal amounts of each unique entry of an array.

#     Parameters
#     ----------
#     array : numpy.ndarray
#         Input array.
#     ordered : bool, optional
#         Wether or not the unique values must be sorted by mode. The default is True.
#     unique_indices : bool, optional
#         Wether or not the unique values must be returned as indices.
#         If False they are returned as the original unique values. The default is False.
#     np_axis : int or None, optional
#         Axis arg of the numpy.unique() function. The default is None.

#     Returns
#     -------
#     unique : numpy.ndarray
#         The unique values.
#     perc : numpy.ndarray
#         The percentage amount of each unique value.

#     '''
#     unique, counts = np.unique(array, return_counts=True, axis=np_axis)
#     perc = 100*counts/counts.sum()
#     if unique_indices:
#         unique = np.arange(len(unique))
#     if ordered:
#         freq_sort = np.argsort(-counts)
#         unique, perc = unique[freq_sort], perc[freq_sort]
#     return (unique, perc)


# def map2ASCII(map_path): # deprecated. Moved to image_analysis_function as 'image2array'
# # # If the image has a .tiff format we use the libtiff module
# #     if splitext(map_path)[1].upper() in ('.TIFF', '.TIF'):
# #         libtiff.libtiff_ctypes.suppress_warnings()
# #         img = libtiff.TIFFfile(map_path, use_memmap=False)
# #     # Adjust array shape by removing dimensions that are == 1 (squeeze)
# #     # and swapping dimensions(transpose) to get 1st -> row, 2nd -> col, 3rd -> channels
# #         arr = np.transpose(img.get_tiff_array(), (1,2,0)).squeeze()
# #         return arr

# # If the image has a .tiff format we use the TIFFFILE module
# # Adjust array shape by removing dimensions that are == 1 (squeeze)
#     if os.path.splitext(map_path)[1].upper() in ('.TIFF', '.TIF'):
#         arr = tifffile.imread(map_path).squeeze()
#         return arr
# # otherwise we use PIL
#     else:
#         img = PIL.Image.open(map_path)
#         mode = img.mode
#         if mode == 'CMYK':
#             img = img.convert('RGB')
#         dtype = 'uint8' if mode in ('1', 'L', 'P') else 'uint32'
#         arr = np.array(img, dtype=dtype)
#         return arr



# def MergeMaps(maps, mask=False): # !!! deprecated. moved to ML_tools
#     try:
#         if mask:
#             out = np.vstack([m.compressed() for m in maps]).T
#         else:
#             out = np.vstack([m.flatten() for m in maps]).T
#         return (out, True)
#     except ValueError: # maps are not of the same size
#         return (None, False)


def path2filename(path: str, ext=False):
    '''
    Extract the file name from a full filepath.

    Parameters
    ----------
    path : str
        Full file path.
    ext : bool, optional
        Also include the file extension. The default is False.

    Returns
    -------
    str
        File name.

    '''
    name, e = os.path.splitext(os.path.split(path)[-1])
    return name + e if ext else name


def extend_filename(fname: str, add: str, ext: str|None=None):
    '''
    Append text to an existent file name.

    Parameters
    ----------
    fname : str
        File name or full filepath.
    add : str
        Text to add.
    ext : str | None, optional
        New file extension. If None, the original extension is used. The 
        default is None.

    Returns
    -------
    str
        Extended file name.

    '''
    name, e = os.path.splitext(fname)
    if ext is not None: e = ext
    return name + add + e


# def RGB2float(RGB_list): #deprecated! Moved to plots-> rgb_to_float
#     floatRGB = [(r/255, g/255, b/255) for (r,g,b) in RGB_list]
#     if len(floatRGB) == 1:
#         return floatRGB[0]
#     else:
#         return floatRGB


def guessMap(mapName, targets, caseSens=False): # Find a more elegant solution.
    mapName = mapName.strip()
    for t in targets:
        if not caseSens:
            if t.lower() == mapName.lower():
                return t
        else:
            if t == mapName:
                return t
    else:
        return False


def most_frequent(iterable: Iterable):
    '''
    Get most frequent element in iterable.

    Parameters
    ----------
    iterable : Iterable
        List of elements.

    Returns
    -------
    most_freq : Any type
        Most frequent element in the iterable.

    '''
    most_freq = Counter(iterable).most_common(1)[0][0]
    return most_freq


# # !!! deprecated. if possible use it in ModelBasedClassifier() or via MineralMap encoder
# def encode_labels(array, transDict, dtype='int16'):
#     ''' From labels to IDs. TransDict is {'label': ID} '''
#     res = np.copy(array)
#     for k, v in transDict.items(): res[array==k] = v
#     return res.astype(dtype)


# # !!! deprecated. if possible use it in ModelBasedClassifier() or via MineralMap encoder
# def decode_labels(array, transDict, dtype='U8'):
#     ''' From IDs to labels. TransDict is {'label': ID} '''
#     res = np.copy(array).astype(dtype)
#     for k, v in transDict.items(): res[array==v] = k
#     return res

