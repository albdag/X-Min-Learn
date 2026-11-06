"""
Primary entry point of the application.

Created on Wed Mar 20 12:31:39 2024
@author: albdag
"""

import sys
from pathlib import Path

from PyQt5.QtCore import QStandardPaths, Qt
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

import crashlog
import resources  # noqa: F401
import settings
import style

# Setup a custom global excepthook early to also catch startup errors
sys.excepthook = crashlog.custom_excepthook

# Global application information
APPNAME = 'XMinLearn'
APPVERSION = '1.0.0-beta.3'

# Windows shell options for distribution
try:
    from ctypes import windll  # Only exists on Windows.
    app_id = f'{APPNAME}.{APPVERSION}'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
except ImportError:
    pass

# Set directory for user settings and application crash logging
config_loc = QStandardPaths.StandardLocation.AppConfigLocation
config_dir = Path(QStandardPaths.writableLocation(config_loc)) / APPNAME
if not config_dir.is_dir():
    config_dir.mkdir(parents=True, exist_ok=True)
settings.setup_manager(config_dir)
crashlog.setup_logger(config_dir)

# Set high DPI management
do_high_dpi_scaling = settings.manager.get('GUI/high_dpi_scaling')
if (dpi_scaling := getattr(Qt, 'AA_EnableHighDpiScaling', None)) is not None:
    QApplication.setAttribute(dpi_scaling, do_high_dpi_scaling)
if (dpi_pixmap := getattr(Qt, 'AA_UseHighDpiPixmaps', None)) is not None:
    QApplication.setAttribute(dpi_pixmap, do_high_dpi_scaling)


# Main application execution
if __name__ == "__main__":

# Ensure the crash logger is cleanly closed when application is closed
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(crashlog.logger.close)

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