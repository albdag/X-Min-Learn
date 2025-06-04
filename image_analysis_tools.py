# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:32:14 2024

@author: albdag
"""
import os

import numpy as np
from numpy.typing import DTypeLike, NDArray
from PIL import Image
from scipy import ndimage
import tifffile


def construct_kernel_filter(shape: str, size: int) -> np.ndarray:
    '''
    Return a binary kernel structure of shape 'shape' and size 'size', required
    by image filtering algorithms.

    Parameters
    ----------
    shape : str
        Kernel shape. Can be one of 'square', 'circle' or 'diamond'.
    size : int
        Kernel size. 

    Returns
    -------
    np.ndarray
        Binary kernel structure.

    Raises
    ------
    ValueError
        Raised if 'shape' is not one of 'square', 'circle' or 'diamond'.

    Example
    -------
    construct_kernel_filter('square', 3)
    >>> [[1 1 1],
         [1 1 1],
         [1 1 1]]
    construct_kernel_filter('diamond', 3)
    >>> [[0 1 0],
         [1 1 1],
         [0 1 0]]
    construct_kernel_filter('circle', 5)
    >>> [[0 1 1 1 0],
         [1 1 1 1 1],
         [1 1 1 1 1],
         [1 1 1 1 1],
         [0 1 1 1 0]]

    '''
# Check that 'shape' is a valid argument
    valid_shapes = ('square', 'circle', 'diamond')
    if shape not in valid_shapes:
        raise ValueError(f'"shape" can only be: {valid_shapes}, not {shape}')

# Construct kernel
    conn = 1 if shape == 'diamond' else 2
    kernel = ndimage.generate_binary_structure(rank=2, connectivity=conn)
    kernel = ndimage.iterate_structure(kernel, size // 2)

# Apply Euclidean distance for a circle kernel
    if shape == 'circle':
        # Compute the Euclidean distance from center (x=size//2; y=size//2)
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - size//2)**2 + (y - size//2)**2)
        # Obtain the circle structure [size/2 = radius (float)]
        kernel = dist <= size/2

    return kernel

        
def apply_binary_morph(
    arr: np.ndarray,
    operation: str,
    structure: np.ndarray,
    mask: np.ndarray | None = None
) -> np.ndarray:
    '''
    Apply binary morphological image processing operation to the given array.

    Parameters
    ----------
    arr : numpy ndarray
        Input binary array.
    operation : str
        Morphological operation. Must be one of 'Erosion + Reconstruction', 
        'Opening', 'Closing', 'Erosion', 'Dilation' or 'Fill Holes'.
    structure : numpy ndarray
        Binary structuring element used for the operation. For more details see
        'construct_kernel_filter' function.
    mask : ndarray or None, optional
        If given, only element with a True mask value will be processed. It 
        must have the same shape of 'arr'. The default is None.

    Returns
    -------
    numpy ndarray
        Filtered array.

    Raises
    ------
    ValueError
        Raised if 'operation' is not a valid string.

    '''
    match operation:
        case 'Erosion + Reconstruction':
            erosion = ndimage.binary_erosion(arr, structure, mask=mask)
            return ndimage.binary_propagation(erosion, structure, mask=arr)
        case 'Opening':
            return ndimage.binary_opening(arr, structure, mask=mask)
        case 'Closing':
            return ndimage.binary_closing(arr, structure, mask=mask)
        case 'Erosion':
            return ndimage.binary_erosion(arr, structure, mask=mask)
        case 'Dilation':
            return ndimage.binary_dilation(arr, structure, mask=mask)
        case 'Fill Holes':
            return ndimage.binary_fill_holes(arr, structure)
        case _:
            raise ValueError('f"{operation}" is not a valid operation.')
    

def replace_with_nearest(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''
    Replace elements in 'array' with their nearest non-masked values only where
    the corresponding positions in the 'mask' are marked as 'True' (non-zero). 
    The nearest values are determined using a Euclidean distance transform.

    Parameters
    ----------
    array : numpy ndarray
        The original array.
    mask : numpy ndarray
        Binary array indicating the values that must be replaced. It must have
        the same shape of 'array'.

    Returns
    -------
    numpy ndarray
        A modified version of 'array' where values at masked positions are 
        replaced by their nearest unmasked neighbors.

    '''
    idx = ndimage.distance_transform_edt(mask, return_distances=False,
                                         return_indices=True)
    return array[tuple(idx)]


def nan_aware_mode(
    array: np.ndarray,
    nan_value: float | int | str | None = None,
    nan_threshold: float = 0.5
) -> float | int | str :
    '''
    Custom mode function that returns the most frequent value in the given
    array. If a 'nan_value' is provided, this function returns such value if 
    its ratio in the input array is bigger than the 'nan_threshold' value.

    Parameters
    ----------
    array : numpy ndarray
        Input array.
    nan_value : float or int or str or None, optional
        The value to be treated as NaN. If None, no value will be treated as 
        NaN. The default is None.
    nan_threshold : float, optional
        The NaN ratio threshold, which should range between 0 and 1. Ignored if 
        'nan_value' is None. The default is 0.5.

    Returns
    -------
    float or int or str
        The most frequent value in the input array or the NaN value.

    '''

# NaN threshold logic
    nan_count = np.count_nonzero(array == nan_value)
    if nan_count > array.size * nan_threshold:
        return nan_value

# Return most frequent value
    unique, counts = np.unique(array[array != nan_value], return_counts=True)
    return unique[np.argmax(counts)]


def apply_mode_filter(
    array: np.ndarray,
    structure: np.ndarray,
    nan_value: float | int | str | None = None,
    nan_threshold: float = 0.5
) -> np.ndarray:
    '''
    Apply a maximum frequency (mode) filter on input array, using a NaN aware
    approach. See 'nan_aware_mode' function for more details.

    Parameters
    ----------
    array : numpy ndarray
        Input array.
    structure : numpy ndarray
        Binary structuring element used for the operation. For more details see
        'construct_kernel_filter'.
    nan_value : float or int or str or None, optional
        The value to be treated as NaN. See 'nan_aware_mode' function. The 
        default is None.
    nan_threshold : float, optional
        NaN ratio threshold. See 'nan_aware_mode' function. The default is 0.5.

    Returns
    -------
    out : numpy ndarray
        Filtered array.
        
    '''
    kwargs = {'nan_value': nan_value, 'nan_threshold': nan_threshold}
    out = ndimage.generic_filter(array, nan_aware_mode, footprint=structure, 
                                 mode='nearest', extra_keywords=kwargs)
    return out


def binary_merge(arrays: list[np.ndarray], mode: str) -> np.ndarray:
    '''
    Merge multiple binary arrays applying a union (product) or an intersection 
    (sum) strategy.

    Parameters
    ----------
    arrays : list[numpy ndarray]
        List of binary arrays.
    mode : str
        Merge strategy. Can be 'union' (or 'U') or 'intersection' (or 'I').

    Returns
    -------
    numpy ndarray
        Merged binary array.

    Raises
    ------
    ValueError
        Raised if 'mode' argument is invalid.
    ValueError
        Raised if 'arrays' list contains less than two arrays.

    '''
# UNION (0 * 1 = 0). If used with masks, all holes are preserved.
# INTERSECTION (0 + 1 = 1). If used with masks, only overlapping holes survive.
    match mode:
        case 'union' | 'U':
            func = np.prod
        case 'intersection' | 'I':
            func = np.sum
        case _:
            raise ValueError(f'{mode} is not a valid argument for mode.')
    
    if len(arrays) < 2:
        raise ValueError('"arrays" list must contain at least 2 arrays.')
    else:
        merged = func(np.array(arrays), axis=0)

    return merged


def hex2rgb(hex: str) -> tuple[int, int, int]:
    '''
    Convert color HEX string to RGB tuple.

    Parameters
    ----------
    hex : str
        Color string.

    Returns
    -------
    rgb : tuple[int, int, int]
        RGB color triplet.

    '''
    hex = hex.lstrip('#')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return rgb


def rgb2hex(rgb: tuple[int, int, int]) -> str:
    '''
    Convert RGB tuple to color HEX string.

    Parameters
    ----------
    rgb : tuple[int, int, int]
        RGB color triplet.

    Returns
    -------
    hex : str
        Color HEX string.
        
    '''
    hex = '#{:02x}{:02x}{:02x}'.format(*rgb)
    return hex


def rgb_complementary(rgb: tuple[int, int, int]) -> tuple[int, int, int]: # Currently not used
    '''
    Return complemetary RGB color.

    Parameters
    ----------
    rgb : tuple[int, int, int]
        RGB triplet.

    Returns
    -------
    tuple[int, int, int]
        Complementary RGB triplet.

    '''
    _r, _g, _b = r, g, b = rgb

# HiLo function
    if b < g: g, b = b, g
    if g < r: r, g = g, r
    if b < g: g, b = b, g
    k = r + b

    return tuple(k - u for u in (_r, _g, _b))


def rescale_to_8bit(array: np.ndarray) -> NDArray[np.uint8]: # Currently not used
    '''
    Rescale an array to 8bit. Unsigned integer type only.

    Parameters
    ----------
    array : numpy ndarray
        Input array.

    Returns
    -------
    numpy ndarray
        The array rescaled to 8-bit unsigned integer.

    '''
    return np.round((array/array.max())*255).astype('uint8')


def rgba2rgb(rgba: np.ndarray) -> np.ndarray:
    '''
    Convert RGBA array to RGB array.

    Parameters
    ----------
    rgba : numpy ndarray
        RGBA array.

    Returns
    -------
    rgb : numpy ndarray
        Converted RGB array.

    '''
    row, col, _ = rgba.shape

    rgb = np.empty((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    rgb[:,:,0] = r * a + (1.0 - a) * 255
    rgb[:,:,1] = g * a + (1.0 - a) * 255
    rgb[:,:,2] = b * a + (1.0 - a) * 255

    return np.asarray(rgb, dtype=rgba.dtype)


def greyscale2rgb(greyscale: np.ndarray) -> np.ndarray:
    '''
    Convert 1 channel "greyscale" array to 3 channels "RGB" array.

    Parameters
    ----------
    greyscale : numpy ndarray
        Input greyscale array.

    Returns
    -------
    rgb : numpy ndarray
        Output RGB array.

    '''
    rgb = np.stack((greyscale, ) * 3, axis=-1)
    return rgb


def binary2greyscale(binary: np.ndarray) -> NDArray[np.uint8]:
    '''
    Convert binary array to greyscale array. Unsigned integer type only.

    Parameters
    ----------
    binary : numpy ndarray
        Input binary array.

    Returns
    -------
    greyscale : numpy ndarray
        Output 8-bit unsigned integer "greyscale" array.
        
    '''
    greyscale = binary.astype('uint8') * 255
    return greyscale


def image2array(path: str, dtype: DTypeLike = 'int64') -> np.ndarray:
    '''
    Convert image data to array.

    Parameters
    ----------
    path : str
        Path to image.
    dtype : numpy DTypeLike, optional
        Output array dtype. The default is 'int64'.

    Returns
    -------
    array : numpy ndarray
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


def noisy_array(
    shape: float,
    scale: float,
    array_shape: tuple[int, int], 
    dtype: DTypeLike = 'int32'
) -> np.ndarray:
    '''
    Generate random noisy integer array using a Gamma distribution function.

    Parameters
    ----------
    shape : float
        The shape of the Gamma distribution.
    scale : float
        The scale of the Gamma distribution.
    array_shape : tuple[int, int]
        The shape of the output array.
    dtype : numpy DTypeLike, optional
        Output array dtype. The default is 'int32'.

    Returns
    -------
    noisy : numpy ndarray
        Noisy integer array.

    '''
    noisy = np.random.gamma(shape, scale, size=array_shape).round()
    return noisy.astype(dtype)