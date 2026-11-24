"""
Defines the aesthetics of the graphical interface, including color
schemes, fonts, and widget styling for a consistent appearance.

Created on Tue Oct 29 10:41:40 2024
@author: albdag
"""

from typing import Literal

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette
from PyQt5.QtWidgets import QHeaderView, QSizePolicy

import settings

#  ------------------------------------------------------------------  #
#                     COMMON QT FLAGS AND ATTRIBUTES
#  ------------------------------------------------------------------  #
# Alignment
A_TOP = Qt.AlignmentFlag.AlignTop
A_RIGHT = Qt.AlignmentFlag.AlignRight
A_BOTTOM = Qt.AlignmentFlag.AlignBottom
A_LEFT = Qt.AlignmentFlag.AlignLeft
A_CENTER = Qt.AlignmentFlag.AlignCenter
A_HCENTER = Qt.AlignmentFlag.AlignHCenter
A_VCENTER = Qt.AlignmentFlag.AlignVCenter

# Aspect ratio mode
ARM_IGNORE = Qt.AspectRatioMode.IgnoreAspectRatio
ARM_KEEP = Qt.AspectRatioMode.KeepAspectRatio
ARM_KEEPBYEXPANDING = Qt.AspectRatioMode.KeepAspectRatioByExpanding

# Orientation
O_HORIZONTAL = Qt.Orientation.Horizontal
O_VERTICAL = Qt.Orientation.Vertical

# Header resize mode
HRM_INTERACTIVE = QHeaderView.ResizeMode.Interactive
HRM_FIXED = QHeaderView.ResizeMode.Fixed
HRM_STRETCH = QHeaderView.ResizeMode.Stretch
HRM_RESIZETOCONTENTS = QHeaderView.ResizeMode.ResizeToContents

# Size policy
SP_EXPANDING = QSizePolicy.Policy.Expanding
SP_FIXED = QSizePolicy.Policy.Fixed
SP_IGNORED = QSizePolicy.Policy.Ignored
SP_MAXIMUM = QSizePolicy.Policy.Maximum
SP_MINIMUM = QSizePolicy.Policy.Minimum
SP_MINIMUMEXPANDING = QSizePolicy.Policy.MinimumExpanding
SP_PREFERRED = QSizePolicy.Policy.Preferred

#  ------------------------------------------------------------------  #
#                            GUI COLORS
#  ------------------------------------------------------------------  #
# Interface colors
BLACK_PEARL = "#19232D"  # (25, 35, 45)
BLOSSOM = "#EEB5B6"  # (238, 181, 182)
BLOSSOM_LIGHT = "#F8DEDF"  # (248, 222, 223)
CASPER = "#C9C9C9"  # (201, 201, 201)
CASPER_DARK = "#B0B0B0"  # (176, 176, 176)
CASPER_LIGHT = "#E3E3E3"  # (227, 227, 227)
IVORY = "#F9F9F4"  # (249, 249, 244)
SAN_MARINO = "#497096"  # (73, 112, 150)
SAN_MARINO_LIGHT = "#DCE5EE"  # (220, 229, 238)
XML_PETROLEUM = "#005858"  # (0, 88, 88)

# Special status colors
OK_GREEN = "#4CAF50"  # (76, 175, 80)
BAD_RED = "#ED4337"  # (237, 67, 55)
WARN_YELLOW = "#FFCC00"  # (255, 204, 0)

# Histogram mask color
HIST_MASK = "#FFA500BF"  # (255, 165, 0, alpha=0.75)

#  ------------------------------------------------------------------  #
#                               ICONS
#  ------------------------------------------------------------------  #
ICONDIR = settings.ICON_DIR
ICONS = {
    # General icons ----------------------------------------------------
    "ACCURACY": ICONDIR / "accuracy.png",
    "ADD_MASK": ICONDIR / "add_mask.png",
    "ADD_ROI": ICONDIR / "add_roi.png",
    "ARROW_DOWN": ICONDIR / "arrow_down.png",
    "ARROW_UP": ICONDIR / "arrow_up.png",
    "BULLSEYE": ICONDIR / "bullseye.png",
    "CARET_DOUBLE_YELLOW": ICONDIR / "caret_double_yellow.png",
    "CARET_DOWN_RED": ICONDIR / "caret_down_red.png",
    "CARET_UP_GREEN": ICONDIR / "caret_up_green.png",
    "CHEVRON_DOWN": ICONDIR / "chevron_down.png",
    "CHEVRON_UP": ICONDIR / "chevron_up.png",
    "CIRCLE": ICONDIR / "circle.png",
    "CIRCLE_ADD": ICONDIR / "circle_add.png",
    "CIRCLE_ADD_GREEN": ICONDIR / "circle_add_green.png",
    "CIRCLE_DEL_RED": ICONDIR / "circle_del_red.png",
    "CLEAR": ICONDIR / "clear.png",
    "COPY": ICONDIR / "copy.png",
    "CUBE": ICONDIR / "cube.png",
    "DIAMOND": ICONDIR / "diamond.png",
    "DICE": ICONDIR / "dice.png",
    "EDIT": ICONDIR / "edit.png",
    "EXPORT": ICONDIR / "export.png",
    "FILE_BLANK": ICONDIR / "file_blank.png",
    "FILE_ERROR": ICONDIR / "file_error.png",
    "FIX": ICONDIR / "fix.png",
    "GEAR": ICONDIR / "gear.png",
    "HIDDEN": ICONDIR / "hidden.png",
    "HIGHLIGHT": ICONDIR / "highlight.png",
    "IMPORT": ICONDIR / "import.png",
    "INFO": ICONDIR / "info.png",
    "INVERT": ICONDIR / "invert.png",
    "LEGEND": ICONDIR / "legend.png",
    "LOG": ICONDIR / "log.png",
    "LOSS": ICONDIR / "loss.png",
    "MASK": ICONDIR / "mask.png",
    "MERGE": ICONDIR / "merge.png",
    "MINERAL": ICONDIR / "mineral.png",
    "OPEN": ICONDIR / "open.png",
    "PALETTE": ICONDIR / "palette.png",
    "PAN": ICONDIR / "pan.png",
    "PASTE": ICONDIR / "paste.png",
    "PERCENT": ICONDIR / "percent.png",
    "PLOT": ICONDIR / "plot.png",
    "RANDOMIZE_COLOR": ICONDIR / "randomize_color.png",
    "RANGE": ICONDIR / "range.png",
    "REFRESH": ICONDIR / "refresh.png",
    "REMOVE": ICONDIR / "remove.png",
    "RENAME": ICONDIR / "rename.png",
    "RGBA": ICONDIR / "rgba.png",
    "ROI": ICONDIR / "roi.png",
    "ROI_SEARCH": ICONDIR / "roi_search.png",
    "ROW_ADD": ICONDIR / "row_add.png",
    "ROW_DEL": ICONDIR / "row_del.png",
    "SAVE": ICONDIR / "save.png",
    "SAVE_AS": ICONDIR / "save_as.png",
    "SCORES": ICONDIR / "scores.png",
    "SQUARE": ICONDIR / "square.png",
    "STACK": ICONDIR / "stack.png",
    "TABLE": ICONDIR / "table.png",
    "TEST": ICONDIR / "test.png",
    "TEST_SET": ICONDIR / "test_set.png",
    "TICK": ICONDIR / "tick.png",
    "TRAIN_SET": ICONDIR / "train_set.png",
    "VALIDATION_SET": ICONDIR / "validation_set.png",
    "WARNING": ICONDIR / "warning.png",
    "WRENCH": ICONDIR / "wrench.png",
    "ZOOM": ICONDIR / "zoom.png",
    "ZOOM_DEFAULT": ICONDIR / "zoom_default.png",
    "ZOOM_IN": ICONDIR / "zoom_in.png",
    "ZOOM_OUT": ICONDIR / "zoom_out.png",
    # Logo folder ------------------------------------------------------
    "LOGO_32": ICONDIR / "logo" / "logo_32.png",
    "LOGO_SPLASH": ICONDIR / "logo" / "logo_splash.png",
    # Panes folder -----------------------------------------------------
    "DATA_MANAGER": ICONDIR / "panes" / "data_manager.png",
    "HISTOGRAM_VIEWER": ICONDIR / "panes" / "histogram_viewer.png",
    "MODE_VIEWER": ICONDIR / "panes" / "mode_viewer.png",
    "PROBABILITY_MAP_VIEWER": ICONDIR / "panes" / "probability_map_viewer.png",
    "RGBA_MAP_VIEWER": ICONDIR / "panes" / "rgba_map_viewer.png",
    "ROI_EDITOR": ICONDIR / "panes" / "roi_editor.png",
    # QSS folder -------------------------------------------------------
    "CARET_DOWN": ICONDIR / "qss" / "caret_down.png",
    "CARET_DOWN_GRAY": ICONDIR / "qss" / "caret_down_gray.png",
    "CARET_LEFT": ICONDIR / "qss" / "caret_left.png",
    "CARET_RIGHT": ICONDIR / "qss" / "caret_right.png",
    "CARET_UP": ICONDIR / "qss" / "caret_up.png",
    "CLOSE": ICONDIR / "qss" / "close.png",
    "HANDLE_H": ICONDIR / "qss" / "handle_h.png",
    "HANDLE_V": ICONDIR / "qss" / "handle_v.png",
    "TICK_BLACK": ICONDIR / "qss" / "tick_black.png",
    "UNDOCK": ICONDIR / "qss" / "undock.png",
    # Tools folder -----------------------------------------------------
    "DATA_VIEWER": ICONDIR / "tools" / "data_viewer.png",
    "DATASET_BUILDER": ICONDIR / "tools" / "dataset_builder.png",
    "MINERAL_CLASSIFIER": ICONDIR / "tools" / "mineral_classifier.png",
    "MODEL_LEARNER": ICONDIR / "tools" / "model_learner.png",
    "PHASE_REFINER": ICONDIR / "tools" / "phase_refiner.png",
}

#  ------------------------------------------------------------------  #
#                             STYLE SHEETS
#  ------------------------------------------------------------------  #
SS_BUTTON = f"""
    QPushButton {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
        border-radius: 4px;
        padding: 5px;
    }}
    QPushButton:!enabled {{
        background-color: {CASPER};
        border: 1px solid {CASPER_DARK};
    }}
    QPushButton:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
    }}
    QPushButton:pressed {{
        border: 2px solid {SAN_MARINO};
    }}
    QPushButton:checked {{
        background-color: {BLOSSOM};
        border: 2px solid {SAN_MARINO};
    }}
    QPushButton:flat {{
        border: none;
        background-color: transparent;
    }}
    """

SS_RADIOBUTTON = f"""
    QRadioButton::indicator {{
        border: 3px solid {BLACK_PEARL};
        border-radius: 6px; /* circle-like */
        height: 7px;
        width: 7px;
    }}
    QRadioButton::indicator:enabled:checked {{
        background-color: {BLOSSOM};
    }}
    QRadioButton::indicator:enabled:unchecked {{
        background-color: {BLACK_PEARL};
    }}
    QRadioButton::indicator:!enabled:checked {{
        border-color: {CASPER_DARK};
        background-color: {CASPER};
    }}
    QRadioButton::indicator:!enabled:unchecked {{
        border-color: {CASPER_DARK};
        background-color: {CASPER_DARK};
    }}
    """

SS_TOOLBUTTON = f"""
    QToolButton {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
        border-radius: 4px;
    }}
    QToolButton:!enabled {{
        background-color: {CASPER};
        border: 1px solid {CASPER_DARK};
    }}
    QToolButton:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
    }}
    QToolButton:pressed {{
        border: 2px solid {BLACK_PEARL};
    }}
    QToolButton:checked {{
        background-color: {BLOSSOM};
        border: 2px solid {BLACK_PEARL};
    }}
    """

SS_COMBOX = f"""
    QComboBox {{
        background-color: {IVORY};
        border: 2px solid {BLACK_PEARL};
        border-radius: 3px;
        color: {BLACK_PEARL};
        selection-background-color: {BLOSSOM};
    }}
    QComboBox:!enabled {{
        background-color: {CASPER};
        border: 1px solid {CASPER_DARK};
        color: gray;
    }}
    QComboBox:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
    }}
    QComboBox::drop-down {{
        background-color: {BLACK_PEARL};
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
    }}
    QComboBox::drop-down:!enabled {{
        background-color: {CASPER_DARK};
    }}
    QComboBox::down-arrow {{
        image: url({ICONS["CARET_DOWN"].as_posix()});
    }}
    QComboBox::down-arrow:!enabled {{
        image: url({ICONS["CARET_DOWN_GRAY"].as_posix()});
    }}
    QComboBox QAbstractItemView {{
        background-color: {IVORY};
        border: 1px solid {SAN_MARINO};
        color: {BLACK_PEARL};
    }}
    """

SS_MENU = f"""
    QMenu {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
    }}
    QMenu::item {{
        border: 1px solid transparent;
        margin: 2px 0px 2px 0px; 
        padding: 2px 40px 2px 10px;
    }}
    QMenu::item:!enabled {{
        color: gray;
    }}
    QMenu::item:selected {{
        border-color: {BLACK_PEARL};
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM_LIGHT}
        );
    }}
    QMenu::icon {{
        padding: 1px 1px 1px 2px;
        border: 2px solid transparent;
        border-radius: 4px;
    }}
    QMenu::icon:checked {{
        border-color: {SAN_MARINO};
    }}
    QMenu::indicator {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
        left: 4px;
        height: 13px;
        width: 13px;
    }}
    QMenu::indicator:checked {{
        image: url({ICONS["TICK_BLACK"].as_posix()});
    }}
    """

SS_MENUBAR = f"""
    QMenuBar {{
        background-color: {IVORY};
        border-top: 1px solid {BLACK_PEARL};
        padding: 3px;
    }}
    QMenuBar::item {{
        border: 1px solid transparent;
        border-radius: 2px;
        margin-bottom: 1px;
        padding: 5px;
    }}
    QMenuBar::item:selected {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1, 
            stop: 0 {IVORY}, stop: 1 {BLOSSOM_LIGHT}
        );
        border-color: {BLACK_PEARL};
    }}
    QToolButton {{
        border: 1px solid transparent;
        border-radius: 2px;
        padding: 1px;
    }}
    QToolButton:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1, 
            stop: 0 {IVORY}, stop: 1 {BLOSSOM_LIGHT}
        );
        border-color: {BLACK_PEARL};
    }} 
    QToolButton:pressed {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1, 
            stop: 0 {IVORY}, stop: 1 {BLOSSOM_LIGHT}
        );
        border-color: {BLACK_PEARL};
        padding: 2px;
    }} 
    """

SS_SCROLLBARH = (
    SS_MENU
    + f"""
    QScrollBar:horizontal {{
        background-color: {CASPER_DARK};
        border: 1px solid transparent;
        height: 23px;
        margin: 5px 20px 0px 20px;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
        min-width: 20px;
    }}
    QScrollBar::handle:horizontal:hover {{
        border: 1px solid {SAN_MARINO};
    }}
    QScrollBar::add-line:horizontal {{
        background-color: {BLACK_PEARL};
        border: 1px solid {SAN_MARINO};
        border-radius: 3px;
        margin-top: 5px;
        width: 20px;
        subcontrol-position: right;
        subcontrol-origin: margin;
    }}
    QScrollBar::sub-line:horizontal {{
        background-color: {BLACK_PEARL};
        border: 1px solid {SAN_MARINO};
        border-radius: 3px;
        margin-top: 5px;
        width: 20px;
        subcontrol-position: left;
        subcontrol-origin: margin;
    }}
    QScrollBar::add-line:horizontal:hover,
    QScrollBar::sub-line:horizontal:hover {{
        background-color: {SAN_MARINO};
        border: 1px solid {BLACK_PEARL};
    }}
    QScrollBar::left-arrow:horizontal {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS["CARET_LEFT"].as_posix()});
    }}
    QScrollBar::right-arrow:horizontal {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS["CARET_RIGHT"].as_posix()});
    }}
    QScrollBar::left-arrow:horizontal:pressed,
    QScrollBar::right-arrow:horizontal:pressed {{
        padding: 1px;
    }}
    """
)

SS_SCROLLBARV = (
    SS_MENU
    + f"""
    QScrollBar:vertical {{
        background-color: {CASPER_DARK};
        border: 1px solid transparent;
        width: 23px;
        margin: 20px 0px 20px 5px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {IVORY};
        border: 1px solid {BLACK_PEARL};
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        border: 1px solid {SAN_MARINO};
    }}
    QScrollBar::add-line:vertical {{
        background-color: {BLACK_PEARL};
        border: 1px solid {SAN_MARINO};
        border-radius: 3px;
        margin-left: 5px;
        height: 20px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
    }}
    QScrollBar::sub-line:vertical {{
        background-color: {BLACK_PEARL};
        border: 1px solid {SAN_MARINO};
        border-radius: 3px;
        margin-left: 5px;
        height: 20px;
        subcontrol-position: top;
        subcontrol-origin: margin;
    }}
    QScrollBar::add-line:vertical:hover,
    QScrollBar::sub-line:vertical:hover {{
        background-color: {SAN_MARINO};
        border: 1px solid {BLACK_PEARL};
    }}
    QScrollBar::up-arrow:vertical {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS["CARET_UP"].as_posix()});
    }}
    QScrollBar::down-arrow:vertical {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS["CARET_DOWN"].as_posix()});
    }}
    QScrollBar::up-arrow:vertical:pressed,
    QScrollBar::down-arrow:vertical:pressed {{
        padding: 1px;
    }}
    """
)

SS_TOOLBAR = (
    SS_MENU
    + f"""
    QToolBar {{
        spacing: 3px;
    }}
    QToolBar::separator {{
        background-color: {SAN_MARINO};
        height: 2px;
        margin: 7px;
        width: 2px;
    }}
    QToolButton {{
        border: 1px solid transparent;
        border-radius: 3px;
        padding: 1px;
    }}
    QToolButton[popupMode="1"] {{
        padding-right: 12px;
    }}
    QToolButton[popupMode="1"]:pressed{{
        padding-right: 16px;
    }}
    QToolButton::menu-button {{
        border: 1px solid transparent;
        border-radius: 3px;
    }}
    QToolButton::menu-button:hover {{
        border-color: {BLACK_PEARL};
    }}
    QToolButton:hover,
    QToolButton:checked:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 1 {BLOSSOM}
        );
        border-color: {BLACK_PEARL};
    }}
    QToolButton:pressed,
    QToolButton:checked:pressed {{
        padding: 4px;
    }}
    QToolButton:checked {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 0.5 {CASPER_LIGHT}, stop: 1 {CASPER_DARK}
        );
        border: 2px solid {SAN_MARINO};
        padding: 0px;
    }}
    QToolBarExtension {{
        border-radius: 2px;
        margin: 1px;
        padding: 1px;
    }}
    QToolBarExtension:pressed {{
        padding: 1px;
    }}
    """
)

SS_MAINTOOLBAR = f"""
    QToolBar {{
        background-color: {CASPER_LIGHT};
        border: 1px solid {BLACK_PEARL};
        border-radius: 4px;
        margin: 2px;
        padding: 4px;
        spacing: 8px;
    }}
    QToolBar::handle {{
        background-color: {SAN_MARINO};
        border: 1px solid {BLACK_PEARL};
        border-radius: 2px;
        margin: 2px;
        padding: 0px;
    }}
    QToolBar::handle:top,
    QToolBar::handle:bottom {{
        image: url({ICONS["HANDLE_H"].as_posix()});
        width: 12px;
    }}
    QToolBar::handle:left,
    QToolBar::handle:right {{
        image: url({ICONS["HANDLE_V"].as_posix()});
        height: 12px;
    }}
    QToolBar::separator {{
        background-color: {SAN_MARINO};
        height: 3px;
        margin: 7px;
        width: 3px;
    }}
    QToolButton {{
        border: 1px solid transparent;
        border-radius: 3px;
        icon-size: 32px;
        margin: 2px;
        padding: 10px;
    }}
    QToolButton:hover,
    QToolButton:checked:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 1 {BLOSSOM}
        );
        border-color: {BLACK_PEARL}
    }}
    QToolButton:pressed,
    QToolButton:checked:pressed {{
        padding: 14px;
    }}
    QToolButton:checked {{
        border: 2px solid {SAN_MARINO};
        padding: 9px;
    }}
    QToolBarExtension {{
        border-radius: 2px;
        margin: 1px;
        padding: 1px;
    }}
    QToolBarExtension:hover,
    QToolBarExtension:checked:hover {{
        border-color: {BLACK_PEARL};
    }} 
    QToolBarExtension:pressed,
    QToolBarExtension:checked:pressed {{
        padding: 2px;
    }}
    QToolBarExtension:checked {{
        border: 1px solid {SAN_MARINO};
        padding: 1px;
    }}
    """

SS_TABWIDGET = f"""
    QTabWidget::pane {{
        background-color: {CASPER_LIGHT};
        border: 2px solid {SAN_MARINO};
        border-top-right-radius: 3px;
        border-top-left-radius: 0px;
        border-bottom-right-radius: 3px;
        border-bottom-left-radius: 3px;
        padding: 1px;
    }}
    QTabBar {{
        background-color: transparent;
    }}
    QTabBar::tab {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 0.75 {IVORY}, stop: 1 {CASPER_LIGHT}
        );
        border-color: {BLACK_PEARL};
        border-style: solid;
        border-width: 1px 1px 0px 1px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 8ex;
        margin: 0px 1px 0px 1px;
        padding: 5px 10px 10px 10px;
    }}
    QTabBar::tab:first {{
        margin-left: 0px;
    }}
    QTabBar::tab:last {{
        margin-right: 0px;
    }}
    QTabBar::tab:hover {{
        border-width: 2px 2px 0px 2px;
        border-color: {SAN_MARINO};
        padding: 4px 9px 10px 9px;
    }}
    QTabBar::tab:selected {{
        border-width: 2px 2px 5px 2px;
        border-color: {SAN_MARINO};
        padding: 4px 9px 5px 9px;
    }}
    QTabBar::close-button {{
        image: url({ICONS["CLOSE"].as_posix()});
        subcontrol-position: right;
    }}
    QTabBar::close-button:pressed {{
        border: 1px solid {SAN_MARINO};
    }}
    QGroupBox#FramedWrapper {{
        background-color: transparent;
        border: 2px solid {SAN_MARINO};
        border-radius: 0px 0px 3px 3px;
        padding: 2px;
    }}
    """

SS_MAINTABWIDGET = (
    SS_TABWIDGET
    + f"""
    QTabWidget::pane {{
        background-color: {CASPER};
        border: 3px solid {SAN_MARINO};
        border-top: 4px solid {SAN_MARINO};
        border-top-right-radius: 4px;
        border-top-left-radius: 0px;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 4px;
        margin-bottom: 2px;
    }}
    QTabBar::tab {{
        margin: 2px 2px 0px 0px;
    }}
    """
)

SS_SPLITTER = f"""
    QSplitter::handle {{
        margin: 0px;
        width: 16px;
        height: 16px
    }}
    QSplitter::handle:horizontal {{
        image: url({ICONS["HANDLE_H"].as_posix()});
    }}
    QSplitter::handle:vertical {{
        image: url({ICONS["HANDLE_V"].as_posix()});
    }}
    QSplitter::handle:hover {{
        background-color: {BLOSSOM_LIGHT};
        border: 1px solid {BLACK_PEARL};
        border-radius: 3px;
        image: none;
        margin: 5px;
    }}
    """

SS_GROUPAREA_BASE = f"""
    QGroupBox {{
        border: 2px solid {SAN_MARINO};
        border-radius: 3px;
        padding-top: 2px;
    }}
    QGroupBox {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 {IVORY}, stop: 0.25 {CASPER_LIGHT},
            stop: 0.75 {CASPER_LIGHT}, stop: 1 {CASPER}
        );
    }}
    """

SS_GROUPAREA_NOFRAME = (
    SS_GROUPAREA_BASE
    + """
    QGroupBox {
        background-color: transparent;
        border-width: 0px;
    }
    """
)

SS_GROUPAREA_TITLE = (
    SS_GROUPAREA_BASE
    + f"""
    QGroupBox {{
        font: bold;
        padding-top: 8ex;
    }}
    QGroupBox::title {{
        color: {BLACK_PEARL};
        padding: 0px 9px 0px 9px;
        subcontrol-origin: padding;
        top: 3ex;
    }}
    QGroupBox::title:!enabled {{
        color: gray;
    }}
    """
)

SS_GROUPSCROLLAREA_FRAME = f"""
    QScrollArea {{
        background-color: transparent;
        border: 1px solid {SAN_MARINO};
        border-radius: 2px;
        padding: 10px;
    }}
    QGroupBox#GroupScrollAreaWrapper,
    QWidget#GroupScrollAreaWrapper {{
        background-color: transparent;
    }}
    """

SS_GROUPSCROLLAREA_NOFRAME = (
    SS_GROUPSCROLLAREA_FRAME
    + """
    QScrollArea {
        border: none;
        padding: 0px;
    }
    """
)

SS_PATHLABEL = f"""
    QLabel {{
        background-color: {BLACK_PEARL};
        border: 3px solid {SAN_MARINO};
        border-radius: 3px;
        color: {IVORY};
        padding: 2px;
    }}
    QLabel:!enabled {{
        background-color: {CASPER_DARK};
        border-color: {CASPER};
        color: gray;
    }}
    """

SS_MAINWINDOW = f"""
    QMainWindow::separator {{
        margin: 0px;
        width: 16px;
        height: 16px;
    }}
    QMainWindow::separator:horizontal {{
        image: url({ICONS["HANDLE_V"].as_posix()});
    }}
    QMainWindow::separator:vertical {{
        image: url({ICONS["HANDLE_H"].as_posix()});
    }}
    QMainWindow::separator:hover {{
        background-color: {BLOSSOM_LIGHT};
        border: 1px solid {BLACK_PEARL};
        border-radius: 3px;
        image: none;
        margin: 5px;
    }}
    QTabBar {{
        background-color: transparent;
    }}
    QTabBar::tab {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 0.75 {IVORY}, stop: 1 {CASPER_LIGHT}
        );
        border-color: {BLACK_PEARL};
        border-style: solid;
        border-width: 1px 1px 0px 1px; 
        border-top-left-radius: 2px;
        border-top-right-radius: 2px;
        margin: 5px 1px 0px 0px;
        padding: 5px 10px 5px 10px;
    }}
    QTabBar::tab:first {{
        margin-left: 4px;
    }}
    QTabBar::tab:last {{
        margin-right: 4px;
    }}
    QTabBar::tab:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {SAN_MARINO_LIGHT}, stop: 0.75 {SAN_MARINO_LIGHT}, 
            stop: 1 {CASPER_LIGHT}
        );
        margin-top: 3px;
    }}
    QTabBar::tab:selected {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {SAN_MARINO_LIGHT}, stop: 0.75 {SAN_MARINO_LIGHT}, 
            stop: 1 {CASPER_LIGHT}
        );
        border-bottom: 3px solid {SAN_MARINO};
        margin-top: 3px;
    }}
    QStatusBar {{
        background-color: {IVORY};
    }}
    """

SS_PANE = f"""
    QDockWidget {{
        font-weight: bold;
        titlebar-close-icon: url({ICONS["CLOSE"].as_posix()});
        titlebar-normal-icon: url({ICONS["UNDOCK"].as_posix()});
    }}
    QDockWidget::title {{
        background-color: {SAN_MARINO_LIGHT};
        border-color: {SAN_MARINO};
        border-style: solid;
        border-width: 1px 0px 1px 10px; 
        margin: 2px 0px 2px 0px;
        padding: 8px;
    }}
    QDockWidget::title:hover {{
        border-width: 2px 0px 2px 12px;
        padding: 7px 8px 7px 6px;
    }}
    QDockWidget::close-button,
    QDockWidget::float-button {{
        border-radius: 3px;
        subcontrol-position: right;
        subcontrol-origin: padding;
        width: 16px;
    }}
    QDockWidget::close-button {{
        right: 10px;
    }}
    QDockWidget::float-button {{
        right: 33px;
    }}
    QDockWidget::close-button:hover,
    QDockWidget::float-button:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
        border: 1px solid {BLOSSOM};
    }}
    QDockWidget::close-button:pressed,
    QDockWidget::float-button:pressed {{
        border: 1px solid {SAN_MARINO};
    }}
    QScrollArea#PaneScrollArea {{
        border: 1px solid transparent;
        margin: 0px; /* 0-margin (bottom & right) to avoid scrollbar issues */
        padding-bottom: 1px;
    }}
    """

SS_DATAMANAGER = (
    SS_MENU
    + f"""
    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {{
        border-image: none;
        image: url({ICONS["CARET_RIGHT"].as_posix()});
    }}
    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {{
        border-image: none;
        image: url({ICONS["CARET_DOWN"].as_posix()});
    }}
    """
)

#  ------------------------------------------------------------------  #
#                             FUNCTIONS
#  ------------------------------------------------------------------  #
def getIcon(name: str) -> QIcon:
    """
    Construct and return the QIcon associated with key 'name' (see
    'ICONS' dictionary for valid keys). If key is invalid, an empty
    QIcon is returned.

    Parameters
    ----------
    name : str
        A valid icon key.

    Returns
    -------
    QIcon
        The icon associated with 'name' or an empty icon, if 'name' is
        invalid.
    """
    path = ICONS.get(name)
    return QIcon(None) if path is None else QIcon(str(path))


def getFont(family: str = "default") -> QFont:
    """
    Return font of type 'family' with the pointsize specified in app
    settings.

    Parameters
    ----------
    family : str, optional
        A valid font family name. If "default", the default PyQt font
        will be returned. The default is "default".

    Returns
    -------
    font : QFont
        Requested font.
    """
    pointsize = settings.manager.get("GUI/fontsize")

    match family:
        case "default":
            font = QFont()
            font.setPointSize(pointsize)
        case _:
            font = QFont(family, pointsize)

    return font


def getPalette(kind: Literal["default", "darkmode"] = "default") -> QPalette:
    """
    Return custom application palette of type 'kind'.

    Parameters
    ----------
    kind : Literal["default", "darkmode"], optional
        Custom palette. At the moment, only "default" is fully
        implemented. The default is "default".

    Returns
    -------
    palette : QPalette
        Custom application palette.
    """
    palette = QPalette()

    active = QPalette.ColorGroup.Active
    inactive = QPalette.ColorGroup.Inactive
    disabled = QPalette.ColorGroup.Disabled

    base_role = QPalette.ColorRole.Base
    button_role = QPalette.ColorRole.Button
    button_text_role = QPalette.ColorRole.ButtonText
    highlight_role = QPalette.ColorRole.Highlight
    highlight_text_role = QPalette.ColorRole.HighlightedText
    placeholder_text_role = QPalette.ColorRole.PlaceholderText
    text_role = QPalette.ColorRole.Text
    window_role = QPalette.ColorRole.Window
    window_text_role = QPalette.ColorRole.WindowText

    if kind == "default":
        # Window (A general background color)
        palette.setColor(active, window_role, QColor(CASPER))
        palette.setColor(inactive, window_role, QColor(CASPER))

        # WindowText (A general foreground color)
        palette.setColor(window_text_role, QColor(BLACK_PEARL))

        # Base (Background of text entries, combobox, toolbar handles etc.)
        palette.setColor(active, base_role, QColor(BLACK_PEARL))
        palette.setColor(inactive, base_role, QColor(BLACK_PEARL))
        palette.setColor(disabled, base_role, QColor("darkGray"))

        # AlternateBase (Alt background in views with alternating rows)
        # palette.setColor(QPalette.ColorRole.AlternateBase, QColor('red'))

        # ToolTips background & text
        # palette.setColor(inactive, QPalette.ColorRole.ToolTipBase, QColor(SAN_MARINO))
        # palette.setColor(inactive, QPalette.ColorRole.ToolTipText, QColor(IVORY))

        # PlaceholderText (Placeholder color for various text input widgets)
        palette.setColor(placeholder_text_role, QColor(CASPER))

        # Text (The foreground color used with Base)
        palette.setColor(active, text_role, QColor(IVORY))
        palette.setColor(inactive, text_role, QColor(IVORY))
        palette.setColor(disabled, text_role, QColor(CASPER))

        # Button (The general button background color)
        palette.setColor(active, button_role, QColor(IVORY))
        palette.setColor(inactive, button_role, QColor(IVORY))
        palette.setColor(disabled, button_role, QColor(CASPER))

        # ButtonText (A foreground color used with the Button color)
        palette.setColor(active, button_text_role, QColor(BLACK_PEARL))
        palette.setColor(inactive, button_text_role, QColor(BLACK_PEARL))
        palette.setColor(disabled, button_text_role, QColor("darkGray"))

        # Highlight and highlighted text
        palette.setColor(highlight_role, QColor(BLOSSOM))
        palette.setColor(highlight_text_role, QColor(BLACK_PEARL))

        # palette.setColor(QPalette.ColorRole.BrightText, QColor('red'))
        # palette.setColor(QPalette.ColorRole.Link, QColor('red'))

    elif kind == "darkmode":  # not ready, stylesheets need reworking
        # Window (A general background color)
        palette.setColor(active, window_role, QColor(BLACK_PEARL))
        palette.setColor(inactive, window_role, QColor(BLACK_PEARL))

        # WindowText (A general foreground color)
        palette.setColor(window_text_role, QColor(IVORY))

        # Base (Background of text entries, combobox, toolbar handles etc.)
        palette.setColor(active, base_role, QColor(BLACK_PEARL))
        palette.setColor(inactive, base_role, QColor(BLACK_PEARL))
        palette.setColor(disabled, base_role, QColor("darkGray"))

        # AlternateBase (Alt background in views with alternating rows)
        # palette.setColor(QPalette.ColorRole.AlternateBase, QColor('red'))

        # ToolTips background & text
        # palette.setColor(inactive, QPalette.ColorRole.ToolTipBase, QColor(SAN_MARINO))
        # palette.setColor(inactive, QPalette.ColorRole.ToolTipText, QColor(IVORY))

        # PlaceholderText (Placeholder color for various text input widgets)
        palette.setColor(placeholder_text_role, QColor(CASPER))

        # Text (The foreground color used with Base)
        palette.setColor(active, text_role, QColor(IVORY))
        palette.setColor(inactive, text_role, QColor(IVORY))
        palette.setColor(disabled, text_role, QColor(CASPER))

        # Button (The general button background color)
        palette.setColor(active, button_role, QColor(BLACK_PEARL))
        palette.setColor(inactive, button_role, QColor(BLACK_PEARL))
        palette.setColor(disabled, button_role, QColor("darkGray"))

        # ButtonText (A foreground color used with the Button color)
        palette.setColor(active, button_text_role, QColor(IVORY))
        palette.setColor(inactive, button_text_role, QColor(IVORY))
        palette.setColor(disabled, button_text_role, QColor(CASPER))

        # Highlight and highlighted text
        palette.setColor(highlight_role, QColor(BLOSSOM))
        palette.setColor(highlight_text_role, QColor(BLACK_PEARL))

        # palette.setColor(QPalette.ColorRole.BrightText, QColor('red'))
        # palette.setColor(QPalette.ColorRole.Link, QColor('red'))

    return palette
