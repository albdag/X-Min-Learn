# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:41:40 2024

@author: albdag
"""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette

import preferences as pref


#  -------------------------------------------------------------------------  #
#                              GUI COLORS 
#  -------------------------------------------------------------------------  #

# Interface colors
BLACK_PEARL = '#19232D'         # (25, 35, 45)
BLOSSOM = '#EEB5B6'             # (238, 181, 182)
BLOSSOM_LIGHT = '#F8DEDF'       # (248, 222, 223)
CASPER = '#C9C9C9'              # (201, 201, 201)
CASPER_DARK = '#B0B0B0'         # (176, 176, 176)
CASPER_LIGHT = '#E3E3E3'        # (227, 227, 227)
IVORY = '#F9F9F4'               # (249, 249, 244)
SAN_MARINO = '#497096'          # (73, 112, 150)
SAN_MARINO_LIGHT = '#DCE5EE'    # (220, 229, 238)
XML_PETROLEUM = '#005858'       # (0, 88, 88)

# Special status colors
OK_GREEN = '#4CAF50'            # (76, 175, 80)
BAD_RED = '#ED4337'             # (237, 67, 55)
WARN_YELLOW = '#FFCC00'         # (255, 204, 0)

# Histogram mask color
HIST_MASK = '#FFA500BF'         # (255, 165, 0, alpha=0.75)


#  -------------------------------------------------------------------------  #
#                                  ICONS 
#  -------------------------------------------------------------------------  #

ICONS = {
    'ACCURACY': Path('Icons') / 'accuracy.png',
    'ADD_MASK': Path('Icons') / 'add_mask.png',
    'ADD_ROI': Path('Icons') / 'add_roi.png',
    'ARROW_DOWN': Path('Icons') / 'arrow_down.png',
    'ARROW_UP': Path('Icons') / 'arrow_up.png',
    'BULLSEYE': Path('Icons') / 'bullseye.png',
    'CARET_DOUBLE_YELLOW': Path('Icons') / 'caret_double_yellow.png',
    'CARET_DOWN_RED': Path('Icons') / 'caret_down_red.png',
    'CARET_UP_GREEN': Path('Icons') / 'caret_up_green.png',
    'CHEVRON_DOWN': Path('Icons') / 'chevron_down.png',
    'CHEVRON_UP': Path('Icons') / 'chevron_up.png',
    'CIRCLE': Path('Icons') / 'circle.png',
    'CIRCLE_ADD': Path('Icons') / 'circle_add.png',
    'CIRCLE_ADD_GREEN': Path('Icons') / 'circle_add_green.png',
    'CIRCLE_DEL_RED': Path('Icons') / 'circle_del_red.png',
    'CLEAR': Path('Icons') / 'clear.png',
    'COPY': Path('Icons') / 'copy.png', 
    'CUBE': Path('Icons') / 'cube.png',
    'DIAMOND': Path('Icons') / 'diamond.png',
    'DICE': Path('Icons') / 'dice.png',
    'EDIT': Path('Icons') / 'edit.png',
    'EXPORT': Path('Icons') / 'export.png',
    'FILE_BLANK': Path('Icons') / 'file_blank.png',
    'FILE_ERROR': Path('Icons') / 'file_error.png',
    'FIX': Path('Icons') / 'fix.png',
    'GEAR': Path('Icons') / 'gear.png',
    'HIDDEN': Path('Icons') / 'hidden.png',
    'HIGHLIGHT': Path('Icons') / 'highlight.png',
    'IMPORT': Path('Icons') / 'import.png',
    'INFO': Path('Icons') / 'info.png',
    'INVERT': Path('Icons') / 'invert.png',
    'LEGEND': Path('Icons') / 'legend.png',
    'LOG': Path('Icons') / 'log.png',
    'LOSS': Path('Icons') / 'loss.png',
    'MASK': Path('Icons') / 'mask.png',
    'MERGE': Path('Icons') / 'merge.png',
    'MINERAL': Path('Icons') / 'mineral.png',
    'OPEN': Path('Icons') / 'open.png',
    'PALETTE': Path('Icons') / 'palette.png',
    'PAN': Path('Icons') / 'pan.png',
    'PASTE': Path('Icons') / 'paste.png',
    'PERCENT': Path('Icons') / 'percent.png',
    'PLOT': Path('Icons') / 'plot.png',
    'RANDOMIZE_COLOR': Path('Icons') / 'randomize_color.png',
    'RANGE': Path('Icons') / 'range.png',
    'REFRESH': Path('Icons') / 'refresh.png',
    'REMOVE': Path('Icons') / 'remove.png',
    'RENAME': Path('Icons') / 'rename.png',
    'RGBA': Path('Icons') / 'rgba.png',
    'ROI': Path('Icons') / 'roi.png',
    'ROI_SEARCH': Path('Icons') / 'roi_search.png',
    'ROW_ADD': Path('Icons') / 'row_add.png',
    'ROW_DEL': Path('Icons') / 'row_del.png',
    'SAVE': Path('Icons') / 'save.png',
    'SAVE_AS': Path('Icons') / 'save_as.png',
    'SCORES': Path('Icons') / 'scores.png',
    'SQUARE': Path('Icons') / 'square.png',
    'STACK': Path('Icons') / 'stack.png',
    'TABLE': Path('Icons') / 'table.png',
    'TEST': Path('Icons') / 'test.png',
    'TEST_SET': Path('Icons') / 'test_set.png',
    'TICK': Path('Icons') / 'tick.png',
    'TRAIN_SET': Path('Icons') / 'train_set.png',
    'VALIDATION_SET': Path('Icons') / 'validation_set.png',
    'WARNING': Path('Icons') / 'warning.png',
    'WRENCH': Path('Icons') / 'wrench.png',
    'ZOOM': Path('Icons') / 'zoom.png',
    'ZOOM_DEFAULT': Path('Icons') / 'zoom_default.png',
    'ZOOM_IN': Path('Icons') / 'zoom_in.png',
    'ZOOM_OUT': Path('Icons') / 'zoom_out.png',
    
# logo folder
    'LOGO_32X32': Path('Icons') / 'logo' / 'logo_32x32.png',
    'LOGO_512X512': Path('Icons') / 'logo' / 'logo_512x512.png',
    'LOGO_HD': Path('Icons') / 'logo' / 'logo_hd.png',

# panes folder
    'DATA_MANAGER': Path('Icons') / 'panes' / 'data_manager.png',
    'HISTOGRAM_VIEWER': Path('Icons') / 'panes' / 'histogram_viewer.png',
    'MODE_VIEWER': Path('Icons') / 'panes' / 'mode_viewer.png',
    'PROBABILITY_MAP_VIEWER': Path('Icons') / 'panes' / 'probability_map_viewer.png',
    'RGBA_MAP_VIEWER': Path('Icons') / 'panes' / 'rgba_map_viewer.png',
    'ROI_EDITOR': Path('Icons') / 'panes' / 'roi_editor.png',

# qss folder
    'CARET_DOWN': Path('Icons') / 'qss' / 'caret_down.png',
    'CARET_DOWN_GRAY': Path('Icons') / 'qss' / 'caret_down_gray.png',
    'CARET_LEFT': Path('Icons') / 'qss' / 'caret_left.png',
    'CARET_RIGHT': Path('Icons') / 'qss' / 'caret_right.png',
    'CARET_UP': Path('Icons') / 'qss' / 'caret_up.png',
    'CLOSE': Path('Icons') / 'qss' / 'close.png',
    'HANDLE_H': Path('Icons') / 'qss' / 'handle_h.png',
    'HANDLE_V': Path('Icons') / 'qss' / 'handle_v.png',
    'RADIOBTN_CHECKED': Path('Icons') / 'qss' / 'radiobtn_checked.png',
    'TICK_BLACK': Path('Icons') / 'qss' / 'tick_black.png',
    'UNDOCK': Path('Icons') / 'qss' / 'undock.png',

# tools folder
    'DATA_VIEWER': Path('Icons') / 'tools' / 'data_viewer.png',
    'DATASET_BUILDER': Path('Icons') / 'tools' / 'dataset_builder.png',
    'MINERAL_CLASSIFIER': Path('Icons') / 'tools' / 'mineral_classifier.png',
    'MODEL_LEARNER': Path('Icons') / 'tools' / 'model_learner.png',
    'PHASE_REFINER': Path('Icons') / 'tools' / 'phase_refiner.png'
}


#  -------------------------------------------------------------------------  #
#                              STYLE SHEETS 
#  -------------------------------------------------------------------------  #

SS_BUTTON = (
    f'''
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
    '''
)


SS_RADIOBUTTON = (
    f'''
    QRadioButton::indicator::checked:enabled {{
        image: url({ICONS.get('RADIOBTN_CHECKED').as_posix()});
    }}
    '''
)


SS_TOOLBUTTON = (
    f'''
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
    '''
)


SS_COMBOX = (
    f'''
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
        image: url({ICONS.get('CARET_DOWN').as_posix()});
    }}

    QComboBox::down-arrow:!enabled {{
        image: url({ICONS.get('CARET_DOWN_GRAY').as_posix()});
    }}

    QComboBox QAbstractItemView {{
        background-color: {IVORY};
        border: 1px solid {SAN_MARINO};
        color: {BLACK_PEARL};
    }}
    '''
)


SS_MENU = (
    f'''
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
        image: url({ICONS.get('TICK_BLACK').as_posix()});
    }}
    '''
)


SS_MENUBAR = (
    f'''
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
    '''
)


SS_SCROLLBARH = (
    SS_MENU + 
    f'''
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
        image: url({ICONS.get('CARET_LEFT').as_posix()});
    }}

    QScrollBar::right-arrow:horizontal {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS.get('CARET_RIGHT').as_posix()});
    }}

    QScrollBar::left-arrow:horizontal:pressed,
    QScrollBar::right-arrow:horizontal:pressed {{
        padding: 1px;
    }}
    '''
)


SS_SCROLLBARV = (
    SS_MENU +
    f'''
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
        image: url({ICONS.get('CARET_UP').as_posix()});
    }}

    QScrollBar::down-arrow:vertical {{
        background-color: transparent;
        border-width: 0px;
        padding: 0px;
        image: url({ICONS.get('CARET_DOWN').as_posix()});
    }}

    QScrollBar::up-arrow:vertical:pressed,
    QScrollBar::down-arrow:vertical:pressed {{
        padding: 1px;
    }}
    '''
)


SS_TOOLBAR = (
    SS_MENU +
    f'''
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
    '''
)


SS_MAINTOOLBAR = (
    f'''
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
        image: url({ICONS.get('HANDLE_H').as_posix()});
        width: 12px;
    }}

    QToolBar::handle:left,
    QToolBar::handle:right {{
        image: url({ICONS.get('HANDLE_V').as_posix()});
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
    '''
)


SS_TABWIDGET = (
    f'''
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
        image: url({ICONS.get('CLOSE').as_posix()});
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
    '''
)


SS_MAINTABWIDGET = (
    SS_TABWIDGET +
    f'''
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
    '''
)


SS_SPLITTER = (
    f'''
    QSplitter::handle {{
        margin: 0px;
        width: 16px;
        height: 16px
    }}

    QSplitter::handle:horizontal {{
        image: url({ICONS.get('HANDLE_H').as_posix()});
    }}

    QSplitter::handle:vertical {{
        image: url({ICONS.get('HANDLE_V').as_posix()});
    }}

    QSplitter::handle:hover {{
        background-color: {BLOSSOM_LIGHT};
        border: 1px solid {BLACK_PEARL};
        border-radius: 3px;
        image: none;
        margin: 5px;
    }}
    '''
)


SS_GROUPAREA_BASE = (
    f'''
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
    '''
)


SS_GROUPAREA_NOFRAME = (
    SS_GROUPAREA_BASE +
    '''
    QGroupBox {
        background-color: transparent;
        border-width: 0px;
    }
    '''
)


SS_GROUPAREA_TITLE = (
    SS_GROUPAREA_BASE + 
    f'''
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
    '''
)


SS_GROUPSCROLLAREA_FRAME = (
    f'''
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
    '''
)


SS_GROUPSCROLLAREA_NOFRAME = (
    SS_GROUPSCROLLAREA_FRAME +
    '''
    QScrollArea {
        border: none;
        padding: 0px;
    }
    '''
)


SS_PATHLABEL = (
    f'''
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
    '''
)


SS_MAINWINDOW = (
    f'''
    QMainWindow::separator {{
        margin: 0px;
        width: 16px;
        height: 16px;
    }}

    QMainWindow::separator:horizontal {{
        image: url({ICONS.get('HANDLE_V').as_posix()});
    }}

    QMainWindow::separator:vertical {{
        image: url({ICONS.get('HANDLE_H').as_posix()});
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
    '''
)


SS_PANE = (
    f'''
    QDockWidget {{
        font-weight: bold;
        titlebar-close-icon: url({ICONS.get('CLOSE').as_posix()});
        titlebar-normal-icon: url({ICONS.get('UNDOCK').as_posix()});
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
        margin: 0px; /* margin bottom and right must be 0 to avoid scrollbar glitches */
        padding-bottom: 1px;
    }}
    '''
)


SS_DATAMANAGER = (
    SS_MENU +
    f'''
    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {{
        border-image: none;
        image: url({ICONS.get('CARET_RIGHT').as_posix()});
    }}

    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {{
        border-image: none;
        image: url({ICONS.get('CARET_DOWN').as_posix()});
    }}
    '''
)



def getIcon(name: str) -> QIcon:
    '''
    Construct and return the QIcon associated with key 'name' (see 'ICONS'
    dictionary for valid keys). If key is invalid, return an empty QIcon.

    Parameters
    ----------
    name : str
        A valid icon key.

    Returns
    -------
    QIcon
        The icon associated with 'name', or an empty icon if 'name' is invalid.

    '''
    path = ICONS.get(name)
    return QIcon(None) if path is None else QIcon(str(path))


def getFont(family: str = 'default') -> QFont:
    '''
    Return font of type 'family' with the pointsize specified in app settings.

    Parameters
    ----------
    family : str, optional
        A valid font family name. If 'default', the default PyQt font will be 
        returned. The default is 'default'.

    Returns
    -------
    QFont
        Requested font.

    '''
    pointsize = pref.get_setting('GUI/fontsize')
    
    match family:
        case 'default':
            font = QFont()
            font.setPointSize(pointsize)
        case _:
            font = QFont(family, pointsize)

    return font


def getPalette(kind: str = 'default') -> QPalette:
    '''
    Return custom application palette of type 'kind'.

    Parameters
    ----------
    kind : str, optional
        Custom palette. At the moment, only 'default' is fully implemented. The
        default is 'default'.

    Returns
    -------
    QPalette
        Custom application palette.
        
    '''
    palette = QPalette()
    
    if kind == 'default':
    # Window (A general background color)
        palette.setColor(QPalette.Active, QPalette.Window, QColor(CASPER)) 
        palette.setColor(QPalette.Inactive, QPalette.Window, QColor(CASPER)) 

    # WindowText (A general foreground color)
        palette.setColor(QPalette.WindowText, QColor(BLACK_PEARL))

    # Base (Background for text entry widgets, comboboxes, toolbar handle etc.)
        palette.setColor(QPalette.Active, QPalette.Base, QColor(BLACK_PEARL)) 
        palette.setColor(QPalette.Inactive, QPalette.Base, QColor(BLACK_PEARL)) 
        palette.setColor(QPalette.Disabled, QPalette.Base, Qt.darkGray)

    # AlternateBase (Alternate background in views with alternating row colors)
        # palette.setColor(QPalette.AlternateBase, Qt.red)

    # ToolTips background & text
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipBase, QColor(SAN_MARINO))
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipText, QColor(IVORY))

    # PlaceholderText (Placeholder color for various text input widgets)
        palette.setColor(QPalette.PlaceholderText, QColor(CASPER))

    # Text (The foreground color used with Base)
        palette.setColor(QPalette.Active, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(CASPER))

    # Button (The general button background color)
        palette.setColor(QPalette.Active, QPalette.Button, QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.Button, QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor(CASPER))

    # ButtonText (A foreground color used with the Button color)
        palette.setColor(QPalette.Active, QPalette.ButtonText, QColor(BLACK_PEARL))
        palette.setColor(QPalette.Inactive, QPalette.ButtonText, QColor(BLACK_PEARL))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)

    # Highlight and highlighted text
        palette.setColor(QPalette.Highlight, QColor(BLOSSOM))
        palette.setColor(QPalette.HighlightedText, QColor(BLACK_PEARL))

        # palette.setColor(QPalette.BrightText, Qt.red)
        # palette.setColor(QPalette.Link, Qt.red)


    elif kind == 'darkmode': # not ready to use yet. Stylesheets need reworking
    # Window (A general background color)
        palette.setColor(QPalette.Active, QPalette.Window, QColor(BLACK_PEARL)) 
        palette.setColor(QPalette.Inactive, QPalette.Window, QColor(BLACK_PEARL)) 

    # WindowText (A general foreground color)
        palette.setColor(QPalette.WindowText, QColor(IVORY))

    # Base (Background for text entry widgets, comboboxes, toolbar handle etc.)
        palette.setColor(QPalette.Active, QPalette.Base, QColor(BLACK_PEARL)) 
        palette.setColor(QPalette.Inactive, QPalette.Base, QColor(BLACK_PEARL)) 
        palette.setColor(QPalette.Disabled, QPalette.Base, Qt.darkGray)

    # AlternateBase (Alternate background in views with alternating row colors)
        # palette.setColor(QPalette.AlternateBase, Qt.red)

    # ToolTips background & text
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipBase, QColor(SAN_MARINO))
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipText, QColor(IVORY))

    # PlaceholderText (Placeholder color for various text input widgets)
        palette.setColor(QPalette.PlaceholderText, QColor(CASPER))

    # Text (The foreground color used with Base)
        palette.setColor(QPalette.Active, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(CASPER))

    # Button (The general button background color)
        palette.setColor(QPalette.Active, QPalette.Button, QColor(BLACK_PEARL))
        palette.setColor(QPalette.Inactive, QPalette.Button, QColor(BLACK_PEARL))
        palette.setColor(QPalette.Disabled, QPalette.Button, Qt.darkGray)

    # ButtonText (A foreground color used with the Button color)
        palette.setColor(QPalette.Active, QPalette.ButtonText, QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.ButtonText, QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(CASPER))

    # Highlight and highlighted text
        palette.setColor(QPalette.Highlight, QColor(BLOSSOM))
        palette.setColor(QPalette.HighlightedText, QColor(BLACK_PEARL))

        # palette.setColor(QPalette.BrightText, Qt.red)
        # palette.setColor(QPalette.Link, Qt.red)

    return palette