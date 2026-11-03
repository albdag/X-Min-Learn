"""
Contains algorithms and utilities for image processing and transformation,
including image filtering techniques.

Created on Wed Oct 23 12:32:14 2024
@author: albdag
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy import ndimage

# Available binary morphological image processing algorithms
BinaryMorphologyAlgorithms = Literal[
    'Erosion + Reconstruction', 
    'Opening', 
    'Closing', 
    'Erosion', 
    'Dilation', 
    'Fill Holes'
]

# Kernel shape types
KernelShape = Literal['square', 'circle', 'diamond']

# Binary merge modes
type BinaryMergeModes = Literal['I', 'intersection', 'U', 'union']


def construct_kernel_filter(shape: KernelShape, size: int) -> npt.NDArray[Any]:
    '''
    Return a binary kernel structure of shape 'shape' and size 'size', required
    by image filtering algorithms.

    Parameters
    ----------
    shape : KernelShape
        Kernel shape. Can be one of 'square', 'circle' or 'diamond'.
    size : int
        Kernel size. 

    Returns
    -------
    NDArray[Any]
        Binary kernel structure.

    Raises
    ------
    ValueError
        Raised if 'shape' is not one of 'square', 'circle' or 'diamond'.

    Example
    -------
    >>> kern_square = construct_kernel_filter('square', 3)
    >>> kern_square
    >>> [[1 1 1],
         [1 1 1],
         [1 1 1]]
    >>> kern_diamond = construct_kernel_filter('diamond', 3)
    >>> kern_diamond
    >>> [[0 1 0],
         [1 1 1],
         [0 1 0]]
    >>> kern_circle = construct_kernel_filter('circle', 5)
    >>> kern_circle
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

    return np.asarray(kernel)


def apply_binary_morph(
    arr: npt.NDArray[Any],
    operation: BinaryMorphologyAlgorithms,
    structure: npt.NDArray[Any],
    mask: npt.NDArray[Any] | None = None
) -> npt.NDArray[Any]:
    '''
    Apply binary morphological image processing operation to the given array.

    Parameters
    ----------
    arr : NDArray[Any]
        Input binary array.
    operation : BinaryMorphologyAlgorithms
        Morphological operation. Must be one of 'Erosion + Reconstruction', 
        'Opening', 'Closing', 'Erosion', 'Dilation' or 'Fill Holes'.
    structure : NDArray[Any]
        Binary structuring element used for the operation. For more details see
        'construct_kernel_filter' function.
    mask : NDArray[Any] | None, optional
        If given, only element with a True mask value will be processed. It 
        must have the same shape of 'arr'. The default is None.

    Returns
    -------
    NDArray[Any]
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
            result = ndimage.binary_fill_holes(arr, structure)
            return arr if result is None else result
        case _:
            raise ValueError('f"{operation}" is not a valid operation.')
    

def replace_with_nearest(
        array: npt.NDArray[Any], mask: npt.NDArray[Any]) -> npt.NDArray[Any]:
    '''
    Replace elements in 'array' with their nearest non-masked values only where
    the corresponding positions in the 'mask' are marked as 'True' (non-zero). 
    The nearest values are determined using a Euclidean distance transform.

    Parameters
    ----------
    array : NDArray[Any]
        The original array.
    mask : NDArray[Any]
        Binary array indicating the values that must be replaced. It must have
        the same shape of 'array'.

    Returns
    -------
    NDArray[Any]
        A modified version of 'array' where values at masked positions are 
        replaced by their nearest unmasked neighbors.

    '''
    idx = ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True)
    
    if idx is None: # safety
        return array
    return array[tuple(idx)]


def nan_aware_mode(
    array: npt.NDArray[Any],
    nan_value: float | str | None = None,
    nan_threshold: float = 0.5
) -> float | int | str :
    '''
    Custom mode function that returns the most frequent value in the given
    array. If a 'nan_value' is provided, this function returns such value if 
    its ratio in the input array is bigger than the 'nan_threshold' value.

    Parameters
    ----------
    array : NDArray[Any]
        Input array.
    nan_value : float | str | None, optional
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
    if nan_value is not None:
        nan_count = np.count_nonzero(array == nan_value)
        if nan_count > array.size * nan_threshold:
            return nan_value
        array = array[array != nan_value]

    # Return most frequent value
    unique, counts = np.unique(array, return_counts=True)
    return unique[np.argmax(counts)]


def apply_mode_filter(
    array: npt.NDArray[Any],
    structure: npt.NDArray[Any],
    nan_value: float | str | None = None,
    nan_threshold: float = 0.5
) -> npt.NDArray[Any]:
    '''
    Apply a maximum frequency (mode) filter on input array, using a NaN aware
    approach. See 'nan_aware_mode' function for more details.

    Parameters
    ----------
    array : NDArray[Any]
        Input array.
    structure : NDArray[Any]
        Binary structuring element used for the operation. For more details see
        'construct_kernel_filter'.
    nan_value : float | str | None, optional
        The value to be treated as NaN. See 'nan_aware_mode' function. The 
        default is None.
    nan_threshold : float, optional
        NaN ratio threshold. See 'nan_aware_mode' function. The default is 0.5.

    Returns
    -------
    NDArray[Any]
        Filtered array.
        
    '''
    extra_kwargs = {'nan_value': nan_value, 'nan_threshold': nan_threshold}
    return ndimage.generic_filter(
        array,
        nan_aware_mode,
        footprint=structure, 
        mode='nearest',
        extra_keywords=extra_kwargs
    )


def binary_merge(
        arrays: list[npt.NDArray[Any]], mode: BinaryMergeModes) -> npt.NDArray[Any]:
    '''
    Merge multiple binary arrays applying an intersection (product) or a union 
    (sum) strategy.

    Parameters
    ----------
    arrays : list[NDArray[Any]]
        List of binary arrays.
    mode : BinaryMergeModes
        Merge strategy. Can be 'intersection' (or 'I') or 'union' (or 'U').

    Returns
    -------
    NDArray[Any]
        Merged binary array.

    Raises
    ------
    ValueError
        Raised if 'mode' argument is invalid.
    ValueError
        Raised if 'arrays' list contains less than two arrays.

    '''
# INTERSECTION (0 * 1 = 0). If used with masks, all holes are preserved.
# UNION (0 + 1 = 1). If used with masks, only overlapping holes survive.
    match mode:
        case 'intersection' | 'I':
            func = np.prod
        case 'union' | 'U':
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
    tuple[int, int, int]
        RGB color triplet.

    '''
    hex = hex.lstrip('#')
    return (int(hex[0:2], 16), int(hex[2:4], 16), int(hex[4:6], 16))


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

    return (k - _r, k - _g, k - _b)


def rescale_to_8bit(array: npt.NDArray[Any]) -> npt.NDArray[np.uint8]: # Currently not used
    '''
    Rescale an array to 8bit. Unsigned integer type only.

    Parameters
    ----------
    array : NDArray[Any]
        Input array.

    Returns
    -------
    NDArray[np.uint8]
        The array rescaled to 8-bit unsigned integer.

    '''
    return np.round((array / array.max()) * 255).astype('uint8', copy=False)


def rgba2rgb(rgba: npt.NDArray[Any]) -> npt.NDArray[Any]:
    '''
    Convert RGBA array to RGB array.

    Parameters
    ----------
    rgba : NDArray[Any]
        RGBA array.

    Returns
    -------
    rgb : NDArray[Any]
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


def greyscale2rgb(greyscale: npt.NDArray[Any]) -> npt.NDArray[Any]:
    '''
    Convert 1 channel "greyscale" array to 3 channels "RGB" array.

    Parameters
    ----------
    greyscale : NDArray[Any]
        Input greyscale array.

    Returns
    -------
    rgb : NDArray[Any]
        Output RGB array.

    '''
    rgb = np.stack((greyscale, ) * 3, axis=-1)
    return rgb


def binary2greyscale(binary: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    '''
    Convert binary array to greyscale array. Unsigned integer type only.

    Parameters
    ----------
    binary : NDArray[np.bool_]
        Input binary array.

    Returns
    -------
    greyscale : NDArray[np.uint8]
        Output 8-bit unsigned integer "greyscale" array.
        
    '''
    greyscale = (binary * 255).astype('uint8', copy=False)
    return greyscale


def image2array(path: str, dtype: npt.DTypeLike = 'int64') -> npt.NDArray[Any]:
    '''
    Convert image data to array.

    Parameters
    ----------
    path : str
        Path to image.
    dtype : DTypeLike, optional
        Output array dtype. The default is 'int64'.

    Returns
    -------
    NDArray[Any]
        Output array.

    '''
    img = Image.open(path)
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    return np.array(img, dtype=dtype)


def noisy_array(
    shape: float,
    scale: float,
    array_shape: tuple[int, int], 
    dtype: npt.DTypeLike = 'int32'
) -> npt.NDArray[Any]:
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
    dtype : DTypeLike, optional
        Output array dtype. The default is 'int32'.

    Returns
    -------
    noisy : NDArray[Any]
        Noisy integer array.

    '''
    noisy = np.random.gamma(shape, scale, size=array_shape).round()
    return noisy.astype(dtype, copy=False)