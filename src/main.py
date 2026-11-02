# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:31:39 2024

@author: albdag
"""

from pathlib import Path
import resources
import sys

from PyQt5.QtCore import Qt, QStandardPaths
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

import settings
import style


# Global application information
APPNAME = 'XMinLearn'
APPVERSION = '1.0.0-beta.3'

# Windows shell options for distribution
try:
    from ctypes import windll  # Only exists on Windows.
    myappid = '.'.join((APPNAME, APPVERSION))
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

# Set directory for user configurations (settings)
config_loc = QStandardPaths.StandardLocation.AppConfigLocation
settings_dir = Path(QStandardPaths.writableLocation(config_loc)) / APPNAME
if not settings_dir.is_dir():
    settings_dir.mkdir(parents=True, exist_ok=True)
settings._setup_manager(settings_dir)

# Set high DPI management
do_high_dpi_scaling = settings.manager.get('GUI/high_dpi_scaling')
if (dpi_scaling := getattr(Qt, 'AA_EnableHighDpiScaling', None)) is not None:
    QApplication.setAttribute(dpi_scaling, do_high_dpi_scaling)
if (dpi_pixmap := getattr(Qt, 'AA_UseHighDpiPixmaps', None)) is not None:
    QApplication.setAttribute(dpi_pixmap, do_high_dpi_scaling)


# Main application execution
if __name__ == "__main__":

    app = QApplication(sys.argv)

# Show splash screen
    loader = QSplashScreen(QPixmap(str(style.ICONS.get('LOGO_SPLASH'))))
    loader.show()
    loader.showMessage('\n\nLoading app', style.A_HCENTER, QColor(style.IVORY))

# Set application properties
    app.setApplicationName(APPNAME)
    app.setApplicationDisplayName('X-Min Learn')
    app.setApplicationVersion(APPVERSION)

# Set application style
    app.setStyle('fusion')
    app.setPalette(style.getPalette('default'))
    app.setFont(style.getFont('Arial'))

# Show main window
    from main_window import MainWindow
    main_win = MainWindow()
    main_win.show()
    loader.finish(main_win)
    sys.exit(app.exec())