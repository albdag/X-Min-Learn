"""
A collection of utility functions that streamline common operations across 
different parts of the application.

Created on Wed Apr 28 12:12:28 2021
@author: albdag
"""

import os
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal


def get_dictkey[K, V](dictionary: Mapping[K, V], value: V) -> K:
    '''
    Get a dictionary key from a given value. If multiple keys are linked to the
    same value, only the first occurrence will be returned.

    Parameters
    ----------
    dictionary : Mapping[K, V]
        The input dictionary.
    value : V
        A valid value to search for.

    Returns
    -------
    K
        The corresponding key.

    Raises
    ------
    ValueError
        Raised if 'value' does not exist in the dictionary.

    '''
    for k, v in dictionary.items():
        if v == value:
            return k
    raise ValueError(f'{value} is not a valid dictionary value.')


def sort_dict[K, V](
    dictionary: Mapping[K, V],
    ordered: Iterable,
    mode: Literal['keys', 'values'] = 'keys'
) -> dict[K, V]:
    '''
    Sort a dictionary according to an ordered iterable of its keys or values.

    Parameters
    ----------
    dictionary : Mapping[K, V]
        The dictionary to be ordered.
    ordered : Iterable
        The ordered iterable of dictionary keys or values used to sort the 
        dictionary. Contained values that do not match any key or value are
        skipped. The iterable can include only some keys or values; in this
        case, the returned sorted dictionary will also be sliced accordingly.
    mode : Literal['keys', 'values'], optional
        Whether 'ordered' contains keys or values. The default is 'keys'.

    Returns
    -------
    dict[K, V]
        The sorted, and potentially sliced, dictionary.

    Raises
    ------
    ValueError
        Raised if 'mode' argument is not 'keys' or 'values'.

    '''
    match mode:
        case 'keys':
            return {k: dictionary[k] for k in ordered if k in dictionary}

        case 'values':
            result: dict[K, V] = {}
            for v in ordered:
                try:
                    result[get_dictkey(dictionary, v)] = v
                except ValueError:
                    continue  # Skip values not present in dictionary
            return result

        case _:
            raise ValueError(f'Invalid "mode" argument: {mode}.')


def path2filename(path: str, ext: bool = False) -> str:
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


def extend_filename(fname: str, add: str, ext: str | None = None) -> str:
    '''
    Append text of 'add' to an existent file name 'fname'.

    Parameters
    ----------
    fname : str
        File name or full filepath.
    add : str
        Text to add.
    ext : str or None, optional
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


def match_map( 
    map_name: str,
    targets: Iterable[str],
    match_case: bool = False
) -> str | None:
    '''
    Return the string in 'targets' that matches the provided 'map_name'. If 
    multiple strings in 'targets' match 'map_name', only the first occurrence
    will be returned.


    Parameters
    ----------
    map_name : str
        Reference map name to be matched.
    targets : Iterable[str]
        Iterable of map names from which extracting the matching string.
    match_case : bool, optional
        Whether the research should be case sensitive. The default is False.

    Returns
    -------
    str or None
        Matching string from 'targets' or None if no matching string is found.

    '''
    map_name = map_name.strip()
    for t in targets:
        _match = t == map_name if match_case else t.lower() == map_name.lower()
        if _match:
            return t
    return None


def most_frequent(iterable: Iterable[Any]) -> Any:
    '''
    Get most frequent element in iterable.

    Parameters
    ----------
    iterable : Iterable[Any]
        Iterable of elements.

    Returns
    -------
    most_freq : Any
        Most frequent element in the iterable.

    '''
    most_freq = Counter(iterable).most_common(1)[0][0]
    return most_freq