# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:09:21 2021

@author: albdag
"""

from PyQt5.QtCore import QByteArray, QSettings


SETTINGS = QSettings('.//SETTINGS//X-MinLearn.ini', QSettings.IniFormat)
DEFAULT_SETTINGS = {
    'GUI/fontsize': (10, int),
    'GUI/high_dpi_scaling': (False, bool),
    'GUI/smooth_animation': (True, bool),
    'GUI/tools_tabbed': (True, bool),
    'GUI/window_state': (None, QByteArray),
    'plots/roi_color': ('#19232d', str),
    'plots/roi_selcolor': ('#55ffff', str),
    'plots/roi_filled': (False, bool),
    'data/extended_model_log': (False, bool),
    'data/decimal_precision': (2, int),
    'data/mask_merging_rule': ('union', str),
    'data/warn_phase_count': (25, int),
    'system/in_path': ('.\\', str),
    'system/out_path': ('.\\', str),
}


def get_setting(name: str) -> QByteArray | bool | int | str:
    '''
    Get current value of setting with name 'name'.

    Parameters
    ----------
    name : str
        A valid setting name.

    Returns
    -------
    QByteArray or bool or int or str
        Current value associated with the required setting.

    Raises
    ------
    ValueError
        Raised if 'name' is not a valid setting.

    '''
    if not name in DEFAULT_SETTINGS:
        raise ValueError(f'{name} is not a valid setting.')
    
    default, type_ = DEFAULT_SETTINGS[name]
    return SETTINGS.value(name, default, type_)


def edit_setting(name: str, value: QByteArray | bool | int | str) -> None:
    '''
    Change the current value of setting with name 'name'.

    Parameters
    ----------
    name : str
        A valid setting name. If it does not exist, it will be created.
    value : QByteArray or bool or int or str
        Value to be assigned to setting.

    '''
    SETTINGS.setValue(name, value)


def clear_settings() -> None:
    '''
    Clear all settings.

    '''
    SETTINGS.clear()


def get_dir(direction: str) -> str:
    '''
    Get last accessed input or output directory.  

    Parameters
    ----------
    direction : str
        Must be 'in' for input directory or 'out' for output directory.

    Returns
    -------
    str
        Last input or output directory.

    Raises
    ------
    ValueError
        Raised if 'direction' argument is not 'in' or 'out'.

    '''
    if direction not in ('in', 'out'):
        raise ValueError(f'{direction} must be "in" or "out".')
    return get_setting(f'system/{direction}_path')


def set_dir(direction: str, dirpath: str) -> None:
    '''
    Set 'dirpath' as the last accessed input or output directory.

    Parameters
    ----------
    direction : str
        Must be 'in' for input directory or 'out' for output directory.
    dirpath : str
        Path to directory.

    Raises
    ------
    ValueError
        Raised if 'direction' is not 'in' or 'out'.

    '''
    if direction not in ('in', 'out'):
        raise ValueError(f'{direction} must be "in" or "out".')
    edit_setting(f'system/{direction}_path', dirpath)