# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:32:14 2024

@author: albdag
"""
import os

import numpy as np
from PIL import Image
from scipy import ndimage
import tifffile




def construct_kernel_filter(shape: str, size: int):

    conn = 1 if shape == 'diamond' else 2
    kernel = ndimage.generate_binary_structure(rank=2, connectivity=conn)
    kernel = ndimage.iterate_structure(kernel, size // 2)

    if shape == 'circle':
        # Compute the Euclidean distance from center (x=size//2; y=size//2)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - size//2)**2 + (y - size//2)**2)
        # Obtain the circle structure [size/2 = radius (float)]
        kernel = dist <= size/2

    return kernel

        
def apply_binary_morph(arr: np.ndarray, operation: str, structure: np.ndarray,
                       mask: np.ndarray|None=None):
    '''
    Apply binary morphological image processing operation to given array.

    Parameters
    ----------
    arr : ndarray
        Input binary array.
    operation : str
        Morphological operation. Must be one of ('Erosion + Reconstruction', 
        'Opening', 'Closing', 'Erosion', 'Dilation', 'Fill Holes').
    structure : ndarray
        Binary structuring element used for the operation.
    mask : ndarray or None, optional
        If given, only element with a True mask value will be processed. It 
        must have the same shape of the input array. The default is None.

    Returns
    -------
    ndarray
        Processed array.

    Raises
    ------
    ValueError
        Operation must be a valid string.

    '''
    if operation == 'Erosion + Reconstruction':
        erosion = ndimage.binary_erosion(arr, structure, mask=mask)
        return ndimage.binary_propagation(erosion, structure, mask=arr)
    
    elif operation == 'Opening':
        return ndimage.binary_opening(arr, structure, mask=mask)
    
    elif operation == 'Closing':
        return ndimage.binary_closing(arr, structure, mask=mask)
    
    elif operation == 'Erosion':
        return ndimage.binary_erosion(arr, structure, mask=mask)
    
    elif operation == 'Dilation':
        return ndimage.binary_dilation(arr, structure, mask=mask)
    
    elif operation == 'Fill Holes':
        return ndimage.binary_fill_holes(arr, structure)
    
    else:
        raise ValueError('f"{operation}" is not a valid operation.')
    

def replace_with_nearest(array: np.ndarray, mask: np.ndarray):
    '''
    Replace elements in `array` where the corresponding positions in the `mask`
    are marked as `True` (or non-zero) with their nearest non-masked values. 
    The nearest values are determined using a Euclidean distance transform.

    Parameters
    ----------
    array : np.ndarray
        The original array.
    mask : np.ndarray
        Binary array indicating the values that must be replaced. It must have
        the same shape of 'array'.

    Returns
    -------
    np.ndarray
        A modified version of `array` where values at masked positions are 
        replaced by their nearest unmasked neighbors.

    '''
    idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    return array[tuple(idx)]


def nan_aware_mode(array: np.ndarray, nan_value=None, nan_threshold=0.5):
    '''
    Custom mode function that returns the most frequent value in the given
    array. If a 'nan_value' is provided, this function returns such value if 
    its ratio in the input array is bigger than the 'nan_threshold' value.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    nan_value : Any, optional
        The value to be treated as NaN. If None, no value will be treated as 
        NaN. The default is None.
    nan_threshold : float, optional
        The NaN ratio threshold, which should range between 0 and 1. Ignored if 
        'nan_value' is None. The default is 0.5.

    Returns
    -------
    Any
        The most frequent value in the input array or the NaN value.
    '''

# NaN threshold logic
    nan_count = np.count_nonzero(array == nan_value)
    if nan_count > array.size * nan_threshold:
        return nan_value

# Return most frequent value
    unique, counts = np.unique(array[array != nan_value], return_counts=True)
    return unique[np.argmax(counts)]


def apply_mode_filter(array: np.ndarray, structure: np.ndarray, nan_value=None,
                      nan_threshold=0.5):
    '''
    Apply a maximum frequency (mode) filter on input array, using a NaN aware
    approach (see 'nan_aware_mode' function for details).

    Parameters
    ----------
    array : np.ndarray
        Input array.
    structure : np.ndarray
        Binary structuring element used for the operation.
    nan_value : Any, optional
        The value to be treated as NaN, required by the 'nan_aware_function'. 
        The default is None.
    nan_threshold : float, optional
        The NaN ratio threshold, required by the 'nan_aware_function'. The 
        default is 0.5.

    Returns
    -------
    out : np.ndarray
        Filtered array.
        
    '''
    kwargs = {'nan_value': nan_value, 'nan_threshold': nan_threshold}
    out = ndimage.generic_filter(array, nan_aware_mode, footprint=structure, 
                                 mode='nearest', extra_keywords=kwargs)
    return out


def binary_merge(arrays: list[np.ndarray], mode: str):
    '''
    Merge multiple binary arrays applying a union (product) or an intersection 
    (sum) strategy.

    Parameters
    ----------
    arrays : list of ndarrays
        List of binary arrays.
    mode : str
        Merge strategy. Can be 'union' (or 'U') or 'intersection' (or 'I').

    Returns
    -------
    ndarray
        Merged binary array.

    Raises
    ------
    ValueError
        The mode argument must be one of 'union', 'U', 'intersection', or 'I'.
    ValueError
        The arrays list must contain at least two arrays.

    '''
# UNION (0 * 1 = 0). If used with masks, all holes are preserved.
    if mode in ('union', 'U'):
        func = np.prod
# NTERSECTION (0 + 1 = 1). If used with masks, only overlapping holes survive.
    elif mode in ('intersection', 'I'):
        func = np.sum
    else:
        raise ValueError(f'{mode} is not a valid argument for mode.')
    
    if len(arrays) < 2:
        raise ValueError('This function requires at least 2 arrays.')
    else:
        merged = func(np.array(arrays), axis=0)

    return merged


def hex2rgb(hex: str):
    '''
    Convert color HEX string to RGB tuple.

    Parameters
    ----------
    hex : str
        Color string.

    Returns
    -------
    rgb : tuple[int]
        RGB color triplet.

    '''
    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return rgb


def rgb2hex(rgb: tuple[int]):
    '''
    Convert RGB tuple to color HEX string.

    Parameters
    ----------
    rgb : tuple[int]
        RGB color triplet.

    Returns
    -------
    hex : str
        Color HEX string.
        
    '''
    hex = '#{:02x}{:02x}{:02x}'.format(*rgb)
    return hex


def rgb_complementary(rgb: tuple[int]): # Currently not used
    '''
    Return complemetary RGB color.

    Parameters
    ----------
    rgb : tuple of int
        RGB triplet.

    Returns
    -------
    tuple of int
        Complementary RGB triplet.

    '''
    _r, _g, _b = r, g, b = rgb
# HiLo function
    if b < g: g, b = b, g
    if g < r: r, g = g, r
    if b < g: g, b = b, g
    k = r + b
    return tuple(k - u for u in (_r, _g, _b))


def rescale_to_8bit(array: np.ndarray): # Currently not used
    '''
    Rescale an array to 8bit. Unsigned integer type only.

    Parameters
    ----------
    array : ndarray
        Input array.

    Returns
    -------
    ndarray
        The array rescaled to 8bit unsigned integer.

    '''
    return np.round((array/array.max())*255).astype('uint8')


def rgba2rgb(rgba: np.ndarray):
    '''
    Convert RGBA array to RGB array.

    Parameters
    ----------
    rgba : ndarray
        RGBA array.

    Returns
    -------
    rgb : ndarray
        Converted RGB array.

    '''
    row, col, _ = rgba.shape

    rgb = np.empty((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    rgb[:,:,0] = r * a + (1.0 - a) * 255
    rgb[:,:,1] = g * a + (1.0 - a) * 255
    rgb[:,:,2] = b * a + (1.0 - a) * 255

    return np.asarray(rgb, dtype=rgba.dtype)


def greyscale2rgb(greyscale: np.ndarray):
    '''
    Convert 1 channel "greyscale" array to 3 channels "RGB" array.

    Parameters
    ----------
    greyscale : np.ndarray
        Input greyscale array.

    Returns
    -------
    rgb : np.ndarray
        Output RGB array.

    '''
    rgb = np.stack((greyscale, ) * 3, axis=-1)
    return rgb


def binary2greyscale(binary: np.ndarray):
    '''
    Convert binary array to greyscale array.

    Parameters
    ----------
    binary : np.ndarray
        Input binary array.

    Returns
    -------
    greyscale : np.ndarray
        Output greyscale array.
        
    '''
    greyscale = binary.astype('uint8') * 255
    return greyscale


def image2array(path: str, dtype='int64'):
    '''
    Convert image data to array.

    Parameters
    ----------
    path : str
        Path to image.
    dtype : np.dtype or str, optional
        Output array dtype. The default is 'int64'.

    Returns
    -------
    array : np.ndarray
        Output array.

    '''
# If the image has a .tiff format we use the 'tiffile' module
    if os.path.splitext(path)[1].upper() in ('.TIFF', '.TIF'):
        # Adjust array shape by removing dimensions that are == 1 (squeeze)
        array = tifffile.imread(path).squeeze().astype(dtype)

# For all the other image formats we use PIL
    else:
        img = Image.open(path)
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        array = np.array(img, dtype=dtype)
    
    return array


def noisy_array(shape: float, scale: float, array_shape: tuple[int], 
                dtype='int32'):
    '''
    Generate random noisy integer array with a Gamma distribution function.

    Parameters
    ----------
    shape : float
        The shape of the Gamma distribution.
    scale : float
        The scale of the Gamma distribution.
    array_shape : tuple[int]
        The shape of the output array.
    dtype : np.dtype or str, optional
        Output array dtype. The default is 'int32'.

    Returns
    -------
    noisy : np.ndarray
        Noisy integer array.

    '''
    noisy = np.random.gamma(shape, scale, size=array_shape).round()
    return noisy.astype(dtype)