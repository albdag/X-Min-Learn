"""
Handles configuration settings and user preferences, allowing customization of
application behavior and appearance.

Created on Tue Dec 14 14:09:21 2021
@author: albdag
"""

from pathlib import Path
from typing import Any, ClassVar

from PyQt5.QtCore import QByteArray, QSettings, QStandardPaths

ICON_DIR = Path(':') / 'icons' # using QResources
DOC_DIR = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)

class SettingsManager:
    
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        'GUI/fontsize': (10, int),
        'GUI/high_dpi_scaling': (False, bool),
        'GUI/smooth_animation': (True, bool),
        'GUI/tools_tabbed': (True, bool),
        'plots/roi_color': ('#19232d', str),
        'plots/roi_selcolor': ('#55ffff', str),
        'plots/roi_filled': (False, bool),
        'data/extended_model_log': (False, bool),
        'data/decimal_precision': (2, int),
        'data/mask_merging_rule': ('union', str),
        'data/warn_phase_count': (25, int),
        'system/in_path': (DOC_DIR, str),
        'system/out_path': (DOC_DIR, str),
        'system/window_state': (None, QByteArray)
    }

    def __init__(self, root_dir: Path) -> None:
        '''
        A class for globally accessing and configuring user-specific settings.

        Parameters
        ----------
        root_dir : Path
            Path to directory where settings file is stored.

        '''
        self._path = (root_dir / 'settings.ini')
        self.settings = QSettings(self._path.as_posix(), QSettings.Format.IniFormat)


    def get(self, name: str) -> Any:
        '''
        Get current value of setting with name 'name'.

        Parameters
        ----------
        name : str
            A valid setting name.

        Returns
        -------
        Any
            Current value associated with the required setting.

        Raises
        ------
        ValueError
            Raised if 'name' is not a valid setting.

        '''
        if not name in self._DEFAULTS:
            raise ValueError(f'{name} is not a valid setting.')
        
        default, type_ = self._DEFAULTS[name]
        return self.settings.value(name, default, type_)


    def set(self, name: str, value: Any) -> None:
        '''
        Change the current value of setting with name 'name'.

        Parameters
        ----------
        name : str
            A valid setting name. If it does not exist, it will be created.
        value : Any
            Value to be assigned to setting.

        '''
        self.settings.setValue(name, value)


    def default(self) -> None:
        '''
        Reset non-system settings to their default values.

        '''
        for name, (default, _) in self._DEFAULTS.items():
            if not name.startswith('system/'):
                self.settings.setValue(name, default)


    def clear(self) -> None:
        '''
        Clear all settings. This will also indirectly reset them to _DEFAULTS.

        '''
        self.settings.clear()



manager = None
def setup_manager(root: Path) -> None:
    '''
    Setup the settings manager by providing the path to the 'root' directory
    that contains the settings file. This function is meant to be internally
    called only during application initialization (see 'main.py').

    Parameters
    ----------
    root : Path
        Path to directory where settings file is stored.

    '''
    global manager
    manager = SettingsManager(root)