# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:31:39 2024

@author: albdag
"""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

import preferences as pref


# WINDOWS SHELL OPTION FOR DISTRIBUTION
try:
    from ctypes import windll  # Only exists on Windows.
    myappid = 'X-MinLearn.alpha.1.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

# DPI MANAGEMENT
if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    loader_bg = QPixmap('Icons/XML_logo.png').scaledToWidth(400)
    flags = Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
    loader = QSplashScreen(loader_bg, flags=flags)
    loader.show()
    loader.showMessage('\nLoading app...', Qt.AlignHCenter, QColor(pref.IVORY))

    app.setStyle('fusion')
    pref.setAppPalette(app)

    # pref.setAppFont(pref.get_setting('main/fontsize', 11))
    # font = app.font()
    # font.setPointSize(pref.get_setting('main/fontsize', 11))
    # app.setFont(font)
    pref.setAppFont(app)

    from main_window import MainWindow
    main_win = MainWindow()
    main_win.show()
    loader.finish(main_win)
    sys.exit(app.exec())

