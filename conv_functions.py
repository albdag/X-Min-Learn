# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:12:28 2021

@author: dagos
"""

from os.path import split, splitext
from collections import Counter
import numpy as np
from PIL import Image
# import libtiff
import tifffile


def shareAxis(ax, target, share=True):
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

def get_globalRangeBounds(data):
    glbMin = min([arr.min() for arr in data])
    glbMax = max([arr.max() for arr in data])
    return (glbMin, glbMax)

def get_dictkey(Dict, value, default=None):
    '''
    A function to get a key in a dictionary from a given value. If multiple keys are
    linked to the same value, only the first in the list will be returned.

    Parameters
    ----------
    Dict : dict
        The input dictionary.
    value : Any type
        The filter value to search the key.
    default : Any type, optional
        If no key is found, this value is returned instead. The default is None.

    Returns
    -------
    key or default : Any type
        The corresponding key or <default> if no key is found.

    '''
    keys = [k for k, v in Dict.items() if v == value]
    if keys: return keys[0]
    return default


def orderDictByList(Dict, ordered_list, mode='keys'):
    '''
    A function to reorder a dictionary according to a desidered ordered list
    of its keys or values.

    Parameters
    ----------
    Dict : DICT
        The dictionary to be ordered.
    ordered_list : LIST
        The ordered list to be used to reorder the dictionary. The list cannot have
        different keys/values of the dict, but can have only a portion of them.
        In this last case, the returned dict will only contain the list keys/values.
    mode : STR, optional
        Whether to interpret the ordered list as the dictionary 'keys' list
        or the dictionary 'values' list. The default is 'keys'.

    Raises
    ------
    ValueError
        If the <mode> parameter is not one of ('keys', 'values').

    Returns
    -------
    dict
        The ordered dictionary.

    '''
    if mode == 'keys':
        return {k: Dict[k] for k in ordered_list}

    elif mode == 'values':
        return {get_dictkey(Dict, v): v for v in ordered_list}
    else:
        raise ValueError(f'{mode} is not a valid mode.')

def get_mode(array, ordered=True, unique_indices=False, np_axis=None):
    '''
     A function to calculate the modal amounts of each unique entry of an array.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    ordered : bool, optional
        Wether or not the unique values must be sorted by mode. The default is True.
    unique_indices : bool, optional
        Wether or not the unique values must be returned as indices.
        If False they are returned as the original unique values. The default is False.
    np_axis : int or None, optional
        Axis arg of the numpy.unique() function. The default is None.

    Returns
    -------
    unique : numpy.ndarray
        The unique values.
    perc : numpy.ndarray
        The percentage amount of each unique value.

    '''
    unique, counts = np.unique(array, return_counts=True, axis=np_axis)
    perc = 100*counts/counts.sum()
    if unique_indices:
        unique = np.arange(len(unique))
    if ordered:
        freq_sort = np.argsort(-counts)
        unique, perc = unique[freq_sort], perc[freq_sort]
    return (unique, perc)

def map2ASCII(map_path):
# # If the image has a .tiff format we use the libtiff module
#     if splitext(map_path)[1].upper() in ('.TIFF', '.TIF'):
#         libtiff.libtiff_ctypes.suppress_warnings()
#         img = libtiff.TIFFfile(map_path, use_memmap=False)
#     # Adjust array shape by removing dimensions that are == 1 (squeeze)
#     # and swapping dimensions(transpose) to get 1st -> row, 2nd -> col, 3rd -> channels
#         arr = np.transpose(img.get_tiff_array(), (1,2,0)).squeeze()
#         return arr

# If the image has a .tiff format we use the TIFFFILE module
# Adjust array shape by removing dimensions that are == 1 (squeeze)
    if splitext(map_path)[1].upper() in ('.TIFF', '.TIF'):
        arr = tifffile.imread(map_path).squeeze()
        return arr
# otherwise we use PIL
    else:
        img = Image.open(map_path)
        mode = img.mode
        if mode == 'CMYK':
            img = img.convert('RGB')
        dtype = 'uint8' if mode in ('1', 'L', 'P') else 'uint32'
        arr = np.array(img, dtype=dtype)
        return arr

def RGBAtoRGB(rgba):
    row, col, _ = rgba.shape

    rgb = np.empty((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    rgb[:,:,0] = r * a + (1.0 - a) * 255
    rgb[:,:,1] = g * a + (1.0 - a) * 255
    rgb[:,:,2] = b * a + (1.0 - a) * 255

    return np.asarray(rgb, dtype=rgba.dtype)


def rescaleTo8bit(array):
    return np.round((array/array.max())*255).astype('uint8')


def invertArray(array):
    inv_array = array.max() - array
    return inv_array

def composeRGBA(arrays, shape):
    arrays = [np.round(a/a.max(), 2) for a in arrays]
    RGBA = np.ones((*shape, 4))
    for n, a in enumerate(arrays):
        RGBA[:, :, n] = a
    return RGBA


def compile_minMap(labels, probs):
    assert labels.shape == probs.shape
    return np.dstack((labels, probs)).astype('U8')
   # to save : np.save(fpath(.npy), minmap, allow_pickle=False)
   # to load : np.load(fpath)

   # gestire il saving/loading  di file .gz, .txt, .npy


def MergeMaps(maps, mask=False): # !!! deprecated. moved to ML_tools
    try:
        if mask:
            out = np.vstack([m.compressed() for m in maps]).T
        else:
            out = np.vstack([m.flatten() for m in maps]).T
        return (out, True)
    except ValueError: # maps are not of the same size
        return (None, False)

    # out_arr = np.array([])
    # try:
    #     for m in maps:
    #         m = np.reshape(m, (m.size, 1)) # Reshape maps as one-column vectors
    #         if len(out_arr) == 0:
    #             out_arr = m
    #         else:
    #             out_arr = np.c_[out_arr, m]
    #     return (out_arr, True)
    # except ValueError: # Raises when maps are not of the same size
    #     return (None, False)

def path2fileName(path, ext=False):
    name, e = splitext(split(path)[-1])
    return name + e if ext else name

def extendFileName(path, add, ext=None):
    name, e = splitext(path)
    if ext is not None: e = ext
    return name + add + e

def RGB_randList(n, tol=None):
    if tol is None: tol = 256//n

    _rgb = np.random.randint(256, size=3)
    RGB_list = _rgb.reshape(1,3)

    for x in range(n-1):
        while np.any(np.all(abs(_rgb - RGB_list) <= tol, axis=1)):
            _rgb = np.random.randint(256, size=3)
        RGB_list = np.r_[RGB_list, _rgb.reshape(1,3)]

    return RGB_list.tolist()

def RGB2float(RGB_list):
    floatRGB = [(r/255, g/255, b/255) for (r,g,b) in RGB_list]
    if len(floatRGB) == 1:
        return floatRGB[0]
    else:
        return floatRGB

def RGB_complementary(r, g, b):
    _r, _g, _b = r, g, b
# HiLo function
    if b < g: g, b = b, g
    if g < r: r, g = g, r
    if b < g: g, b = b, g
    k = r + b
    return tuple(k - u for u in (_r, _g, _b))

def guessMap(mapName, targets, caseSens=False):
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

def most_frequent(iterable):
    '''
    Get most frequent element in iterable.

    Parameters
    ----------
    iterable : iterable object
        List of elements.

    Returns
    -------
    most_freq : object
        Most frequent element.

    '''
# # Raise error if elements within iterable have different types
#     types = {type(i) for i in iterable}
#     if len(types) > 1:
#         raise TypeError('The elements in the iterable have different types.')
# # Find the most frequent element via numpy unique function
#     print(types)
#     arr = np.array(iterable, dtype=object)
#     unq, cnt = np.unique(arr, return_counts=True)
#     most_freq = unq[cnt.argmax()]
#     print(most_freq)
# # If all elements in the iterable are identical this function attempts to
# # return a sub-element. To avoid this, we impose to return the first element of
# # the iterable.
#     if not isinstance(most_freq, types.pop()):
#         most_freq = iterable[0]

#     return most_freq

    most_freq = Counter(iterable).most_common(1)[0][0]
    return most_freq

# !!! deprecated. if possible use it in ModelBasedClassifier() or via MineralMap encoder
def encode_labels(array, transDict, dtype='int16'):
    ''' From labels to IDs. TransDict is {'label': ID} '''
    res = np.copy(array)
    for k, v in transDict.items(): res[array==k] = v
    return res.astype(dtype)

# !!! deprecated. if possible use it in ModelBasedClassifier() or via MineralMap encoder
def decode_labels(array, transDict, dtype='U8'):
    ''' From IDs to labels. TransDict is {'label': ID} '''
    res = np.copy(array).astype(dtype)
    for k, v in transDict.items(): res[array==v] = k
    return res
