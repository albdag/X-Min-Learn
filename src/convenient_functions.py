# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:12:28 2021

@author: albdag
"""

from collections import Counter
from collections.abc import Iterable
import os


def get_dictkey(
    dictionary: dict,
    value: object,
    default: object | None = None
) -> object | None:
    '''
    Get a dictionary key from a given value. If multiple keys are linked to the
    same value, only the first occurrence will be returned.

    Parameters
    ----------
    dictionary : dict
        The input dictionary.
    value : object
        The filter value to search the key. Can be any type.
    default : object or None, optional
        If no key is found, this value is returned instead. Can be any type.
        The default is None.

    Returns
    -------
    k : object or None
        The corresponding key or 'default' if no matching key is found.

    '''
    keys = [k for k, v in dictionary.items() if v == value]
    k = default if not keys else keys[0]
    return k


def sort_dict_by_list(dictionary: dict, ordered_list: list, mode: str = 'keys') -> dict:
    '''
    Sort a dictionary according to a desidered ordered list of its keys or 
    values.

    Parameters
    ----------
    dictionary : dict
        The dictionary to be ordered.
    ordered_list : list
        The ordered list used to sort the dictionary. It cannot contain keys or
        values not included in the dictionary, but it can include just some of 
        them. In this case, the returned sorted dictionary will only contain 
        those keys with their associated values.
    mode : str, optional
        If 'keys', dictionary will be sorted by its keys; if 'values', it will
        be sorted by its values. The default is 'keys'.

    Returns
    -------
    dict
        The sorted dictionary.

    Raises
    ------
    ValueError
        Raised if 'mode' argument is not 'keys' or 'values'.

    '''
    match mode:
        case 'keys':
            return {k: dictionary[k] for k in ordered_list}
        case 'values':
            return {get_dictkey(dictionary, v): v for v in ordered_list}
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


def most_frequent(iterable: Iterable) -> object:
    '''
    Get most frequent element in iterable.

    Parameters
    ----------
    iterable : Iterable
        Iterable of elements.

    Returns
    -------
    most_freq : object
        Most frequent element in the iterable.

    '''
    most_freq = Counter(iterable).most_common(1)[0][0]
    return most_freq