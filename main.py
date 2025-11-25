# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:31:39 2024

@author: albdag
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

import style
import preferences as pref


# WINDOWS SHELL OPTION FOR DISTRIBUTION
try:
    from ctypes import windll  # Only exists on Windows.
    myappid = 'XMinLearn.beta.1.0.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

# DPI MANAGEMENT
high_dpi_scaling = pref.get_setting('GUI/high_dpi_scaling')
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, high_dpi_scaling)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, high_dpi_scaling)


if __name__ == "__main__":

    app = QApplication(sys.argv)

# Show splash screen
    loader_bg = QPixmap(str(style.ICONS.get('LOGO_SPLASH')))
    flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
    loader = QSplashScreen(loader_bg, flags=flags)
    loader.show()
    loader.showMessage('\n\nLoading app', Qt.AlignHCenter, QColor(style.IVORY))

# Set application properties
    app.setApplicationName('X-Min Learn')
    app.setApplicationDisplayName('X-Min Learn')
    app.setApplicationVersion('beta.1.0.0')
    app.setStyle('fusion')
    app.setPalette(style.getPalette('default'))
    app.setFont(style.getFont('Arial'))

# Show main window
    from main_window import MainWindow
    main_win = MainWindow()
    main_win.show()
    loader.finish(main_win)
    sys.exit(app.exec())

