# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:09:21 2021

@author: albdag
"""

from PyQt5.QtCore import QSettings


# SETTINGS
settings = QSettings('.//settings//X-MinLearn.ini', QSettings.IniFormat)

settings_dict = {
    'GUI/fontsize': (10, int),
    'GUI/smooth_animation': (False, bool),
    'plots/roi_color': ('#19232d', str),
    'plots/roi_selcolor': ('#55ffff', str),
    'plots/roi_filled': (False, bool),
    'plots/mask_merging_rule': ('intersection', str),
    'data/extended_model_log': (False, bool),
    'data/decimal_precision': (3, int),
    'system/in_path': ('.\\', str),
    'system/out_path': ('.\\', str),
    }


def get_setting(name: str):
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
    if not name in settings_dict:
        raise ValueError(f'{name} is not a valid setting.')
    
    default, type_ = settings_dict[name]
    return settings.value(name, default, type_)


def edit_setting(name: str, value: object):
    '''
    Change the current value of setting with name 'name'.

    Parameters
    ----------
    name : str
        A valid setting name. If setting does not exist, it will be created.
    value : Any
        Value to be assigned to setting.

    '''
    settings.setValue(name, value)


def clear_settings():
    '''
    Clear all settings.

    '''
    settings.clear()


def get_dir(direction: str):
    '''
    Get last accessed input or output directory.  

    Parameters
    ----------
    direction : str
        Must be 'in' for input directory or 'out' for output directory.

    Returns
    -------
    str
        Last input/output directory.

    Raises
    ------
    ValueError
        Raised if 'direction' is not 'in' or 'out'.

    '''
    if direction not in ('in', 'out'):
        raise ValueError(f'{direction} must be "in" or "out".')
    return get_setting(f'system/{direction}_path')


def set_dir(direction: str, dirpath: str):
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


# # X-Min Learn GUI colors
# BLACK_PEARL = '#19232D'         # (25, 35, 45)
# # CASPER = '#A9B9BC'              # (169, 185, 188)
# CASPER = '#C9C9C9'              # (201, 201, 201)
# # CASPER_DARK = '#9AA8AB'         # (154, 168, 171)
# CASPER_DARK = '#B0B0B0'         # (188, 188, 188)
# CASPER_LIGHT = '#E3E3E3'
# SAN_MARINO = '#497096'          # (73, 112, 150)
# SAN_MARINO_LIGHT = '#DCE5EE'
# IVORY = '#F9F9F4'               # (249, 249, 244)
# BLOSSOM = '#EEB5B6'             # (238, 181, 182)
# BLOSSOM_LIGHT = '#F8DEDF'
# XML_PETROLEUM = '#005858'       # (0, 88, 88)

# # Special button colors
# BTN_GREEN = '#4CAF50'        
# BTN_RED = '#ED4337'             # (237, 67, 55)

# # Histogram mask color
# HIST_MASK = '#FFA500BF'         # (255, 165, 0, alpha=0.75)


# # All style-sheets should go into a new module "stylesheets"
# SS_menu = (
#     '''QMenu {
#             background-color: %s;
#             border: 1px solid %s;}'''
#     '''QMenu::item {
#             padding: 2px 40px 2px 10px;
#             border: 1px solid transparent;}'''
#     '''QMenu::item:!enabled {
#             color: gray;}'''
#     '''QMenu::item:selected {
#             border-color: %s;
#             background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                         stop: 0 white, stop: 1 %s);}'''
#     '''QMenu::icon {
#             padding: 1px 1px 1px 2px;
#             border: 2px solid transparent;
#             border-radius: 4px;}'''
#     '''QMenu::icon:checked {
#             border-color: %s;}'''
#     '''QMenu::indicator {
#             background-color: %s;
#             border: 1px solid %s;
#             left: 4px;
#             height: 13px;
#             width: 13px;}'''
#     '''QMenu::indicator:checked {
#             image: url(Icons/done.png);}'''
#     %(IVORY, BLACK_PEARL, SAN_MARINO, BLOSSOM, SAN_MARINO, IVORY, BLACK_PEARL))


# SS_menuBar = (
#     '''QMenuBar {
#             background-color: %s;
#             border: 1px outset transparent;
#             padding: 3px;}'''
#     '''QMenuBar::item:selected {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);
#             border: 1px outset %s;
#             border-radius: 3px;}'''
#     %(IVORY, BLOSSOM, BLACK_PEARL))


# SS_splitter = (
#     '''QSplitter::handle {
#             background-color: %s;
#             border: 1px solid %s;
#             border-radius: 2px;
#             margin: 10px;}'''
#     '''QSplitter::handle:hover {
#             background-color: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5,
#                                               fx: 0.5, fy: 0.5,
#                                               stop: 0 %s, stop: 1 %s);}'''
#     %(SAN_MARINO, BLACK_PEARL, BLOSSOM_LIGHT, BLOSSOM))


# SS_horizScrollBar = (
#     '''QScrollBar:horizontal {
#             border: 1px solid %s;
#             background: %s;
#             height: 18px;
#             margin: 0px 20px 0px 20px;}'''
#     '''QScrollBar::handle:horizontal {
#             border: 1px solid %s;
#             background: %s;
#             min-width: 20px;}'''
#     '''QScrollBar::handle:horizontal:hover {
#             border: 2px solid %s;}'''
#     '''QScrollBar::add-line:horizontal {
#             border: 2px solid %s;
#             border-radius: 4px;
#             background: %s;
#             width: 20px;
#             subcontrol-position: right;
#             subcontrol-origin: margin;}'''
#     '''QScrollBar::add-line:horizontal:hover {
#             border: 2px solid %s;
#             background: %s;}'''
#     '''QScrollBar::sub-line:horizontal {
#             border: 2px solid %s;
#             border-radius: 4px;
#             background: %s;
#             width: 20px;
#             subcontrol-position: left;
#             subcontrol-origin: margin;}'''
#     '''QScrollBar::sub-line:horizontal:hover {
#             border: 2px solid %s;
#             background: %s;}'''
#     '''QScrollBar::left-arrow:horizontal,
#        QScrollBar::right-arrow:horizontal {
#             border: 1px solid %s;
#             width: 3px;
#             height: 3px;
#             background: %s;}'''
#     '''QScrollBar::left-arrow:horizontal:pressed,
#        QScrollBar::right-arrow:horizontal:pressed {
#             background: %s;}'''
#     %(BLACK_PEARL, CASPER_DARK, BLACK_PEARL, IVORY, SAN_MARINO, SAN_MARINO,
#       BLACK_PEARL, BLACK_PEARL, SAN_MARINO, SAN_MARINO, BLACK_PEARL,
#       BLACK_PEARL, SAN_MARINO, SAN_MARINO, IVORY, BLACK_PEARL))



# SS_vertScrollBar = (
#     '''QScrollBar:vertical {
#             border: 1px solid %s;
#             background: %s;
#             width: 18px;
#             margin: 20px 0px 20px 0px;}'''
#     '''QScrollBar::handle:vertical {
#             border: 1px solid %s;
#             background: %s;
#             min-height: 20px;}'''
#     '''QScrollBar::handle:vertical:hover {
#             border: 2px solid %s;}'''
#     '''QScrollBar::add-line:vertical {
#             border: 2px solid %s;
#             border-radius: 4px;
#             background: %s;
#             height: 20px;
#             subcontrol-position: bottom;
#             subcontrol-origin: margin;}'''
#     '''QScrollBar::add-line:vertical:hover {
#             border: 2px solid %s;
#             background: %s;}'''
#     '''QScrollBar::sub-line:vertical {
#             border: 2px solid %s;
#             border-radius: 4px;
#             background: %s;
#             height: 20px;
#             subcontrol-position: top;
#             subcontrol-origin: margin;}'''
#     '''QScrollBar::sub-line:vertical:hover {
#             border: 2px solid %s;
#             background: %s;}'''
#     '''QScrollBar::up-arrow:vertical,
#        QScrollBar::down-arrow:vertical {
#             border: 1px solid %s;
#             width: 3px;
#             height: 3px;
#             background: %s;}'''
#     '''QScrollBar::up-arrow:vertical:pressed,
#        QScrollBar::down-arrow:vertical:pressed {
#             background: %s;}'''
#     %(BLACK_PEARL, CASPER_DARK, BLACK_PEARL, IVORY, SAN_MARINO, SAN_MARINO,
#       BLACK_PEARL, BLACK_PEARL, SAN_MARINO, SAN_MARINO, BLACK_PEARL,
#       BLACK_PEARL, SAN_MARINO, SAN_MARINO, IVORY, BLACK_PEARL))


# SS_combox = (
#     '''QComboBox {
#             background-color: %s;
#             border: 2px solid %s;
#             border-radius: 3px;
#             color: black;
#             selection-background-color: %s;}'''
#     '''QComboBox:!enabled {
#             background-color: lightgray;
#             border: 1px solid gray;
#             color: darkgray;}'''
#     '''QComboBox:hover {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);}'''
#     '''QComboBox::drop-down {
#             background-color: %s;
#             border-left: 1px solid %s;
#             subcontrol-origin: padding;
#             subcontrol-position: top right;
#             width: 20px;}'''
#     '''QComboBox::drop-down:!enabled {
#             background-color: gray;}'''
#     '''QComboBox::down-arrow {
#             image: url(Icons/arrowDown.png);}'''
#     '''QComboBox::down-arrow:!enabled {
#             image: url(Icons/arrowDown_dark.png);}'''
#     '''QComboBox QAbstractItemView {
#             background-color: %s;
#             border: 1px solid %s;
#             color: black;}'''
#     %(IVORY, BLACK_PEARL, BLOSSOM, BLOSSOM, BLACK_PEARL, BLACK_PEARL, IVORY,
#       SAN_MARINO))


# SS_button = (
#     '''QPushButton {
#             background-color: %s;
#             border: 1px solid %s;
#             border-radius: 4px;
#             padding: 5px;}'''
#     '''QPushButton:!enabled {
#             background-color: lightgray;
#             border: 1px solid gray;}'''
#     '''QPushButton:flat {
#             border: none;
#             background-color: transparent;}'''
#     '''QPushButton:!flat:hover {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);}'''
#     '''QPushButton:pressed {
#             border: 2px solid %s;}'''
#     '''QPushButton:checked {
#             background-color: %s;
#             border: 2px solid %s;}'''
#     %(IVORY, BLACK_PEARL, BLOSSOM, SAN_MARINO, BLOSSOM, SAN_MARINO))


# SS_toolbutton = (
#     '''QToolButton {
#             background-color: %s;
#             border: 1px solid %s;
#             border-radius: 4px;}'''
#     '''QToolButton:!enabled {
#             background-color: lightgray;
#             border: 1px solid gray;}'''
#     '''QToolButton:hover {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);}'''
#     '''QToolButton:pressed {
#             border: 2px solid %s;}'''
#     '''QToolButton:checked {
#             background-color: %s;
#             border: 2px solid %s;}'''
#     %(IVORY, BLACK_PEARL, BLOSSOM, BLACK_PEARL, BLOSSOM, BLACK_PEARL))


# SS_radioButton = (
#     '''QRadioButton::indicator::checked {
#             image: url(Icons/radiobtn_checked.png);}''')

# SS_dataManager = (SS_menu +
#     '''QTreeWidget::branch:has-children:!has-siblings:closed,
#        QTreeWidget::branch:closed:has-children:has-siblings {
#             border-image: none;
#             image: url(Icons/arrowRight.png);}'''
#     '''QTreeWidget::branch:open:has-children:!has-siblings,
#        QTreeWidget::branch:open:has-children:has-siblings {
#             border-image: none;
#             image: url(Icons/arrowDown.png);}''')


# SS_tabWidget = (
#     '''QTabWidget::pane {
#             background-color: %s;
#             border: 2px solid %s;
#             border-top-right-radius: 3px;
#             border-top-left-radius: 0px;
#             border-bottom-right-radius: 3px;
#             border-bottom-left-radius: 3px;
#             padding:1px;}'''
#     '''QTabBar::tab {
#             background: %s;
#             border: 1px solid darkgray;
#             border-top-left-radius: 4px;
#             border-top-right-radius: 4px;
#             min-width: 8ex;
#             padding-top: 2px;
#             padding-bottom: 2px;
#             padding-right: 6px;
#             padding-left: 6px;
#             margin-right: 4px;}'''
#     '''QTabBar::tab:!selected {
#             margin-top: 3px;}'''
#     '''QTabBar::tab:selected {
#             border-color: %s;
#             border-bottom: 5px solid %s;}'''
#     '''QTabBar::tab:hover {
#             background: %s;}'''
#     '''QTabBar::close-button {
#             image: url(Icons/close_pane.png);
#             subcontrol-position: right;}'''
#     '''QTabBar::close-button:pressed {
#             border: 1px solid %s;}'''
#     %(CASPER_LIGHT, SAN_MARINO, IVORY, SAN_MARINO, SAN_MARINO, SAN_MARINO,
#       SAN_MARINO))


# SS_mainTabWidget = (SS_tabWidget +
#     '''QTabWidget::pane {
#             background-color: %s;
#             border: 3px solid %s;
#             border-top: 4px solid %s;
#             border-top-right-radius: 4px;
#             border-top-left-radius: 0px;
#             border-bottom-right-radius: 4px;
#             border-bottom-left-radius: 4px;}'''
#     %(CASPER_DARK, SAN_MARINO, SAN_MARINO))


# # SS_table = (
# #     '''QTableWidget QTableCornerButton::section {
# #             background: %s;
# #             border: 1px outset %s;
# #             border-radius: 1px;}'''
# #     %(IVORY, BLACK_PEARL))


# SS_pathLabel = (
#     '''QLabel {
#             background-color: %s;
#             border: 3px inset %s;
#             border-radius: 3px;
#             color: %s;
#             padding: 2px;}'''
#     %(BLACK_PEARL, SAN_MARINO, IVORY))


# SS_grouparea_notitle = (   
#     '''QGroupBox {
#             border: 2px solid %s;
#             border-radius: 3px;
#             padding-top: 2px;}'''
# '''QGroupBox:enabled {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
#                                               stop: 0 %s, stop: 0.25 %s,
#                                               stop: 0.75 %s, stop: 1 %s);}'''
#     %(SAN_MARINO, IVORY, CASPER_LIGHT, CASPER_LIGHT, CASPER))


# SS_grouparea_title = (SS_grouparea_notitle + 
#     '''QGroupBox {
#             font: bold;
#             padding-top: 8ex;}'''
#     '''QGroupBox::title {
#             color: black;
#             padding: 0px;
#             subcontrol-origin: padding;
#             top: 3ex;}'''
#     '''QGroupBox::title:!enabled {
#             color: gray;}''')


# SS_groupScrollArea_frame = (
#     '''QScrollArea {
#             border: 1px solid %s;
#             border-radius: 2px;
#             padding: 1px;}'''
#     %(SAN_MARINO))


# SS_groupScrollArea_noframe = (SS_groupScrollArea_frame +
#     '''QScrollArea {
#             border-color: transparent;}''')          

# SS_toolbar = (SS_menu +
#     '''QToolBar {
#         spacing: 3px;}'''
#     '''QToolBar::separator {
#             background: %s;
#             height: 2px;
#             margin: 7px;
#             width: 2px;}'''
#     '''QToolButton {
#             border: 2px solid transparent;
#             border-radius: 3px;}'''
#     '''QToolButton[popupMode="1"] { 
#             padding-right: 12px;}'''
#     '''QToolButton::menu-button {
#             border: 1px solid transparent;
#             border-radius: 3px;}'''
#     '''QToolButton::menu-button:hover {
#             border-color: %s;}'''
#     '''QToolButton:pressed {
#             border-color: %s;}'''
#     '''QToolButton:checked {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 %s, stop: 0.5 %s, 
#                                               stop: 1 %s);
#             border-color: %s;}'''
#     '''QToolButton:hover {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);}'''


#     %(SAN_MARINO, BLACK_PEARL, SAN_MARINO, IVORY, CASPER_LIGHT, CASPER_DARK, 
#       SAN_MARINO, BLOSSOM))


# # SS_mainToolbar = (SS_toolbar + 
# #     '''QToolBar {
# #             background-color: %s;
# #             border: 3px groove %s;
# #             border-radius: 4px;
# #             margin: 4px;
# #             spacing: 16px;}'''
# #     '''QToolBar::handle {
# #             background-color: %s;
# #             border: 1px solid %s;
# #             border-radius: 2px;}'''
# #     '''QToolBar::handle:top, QToolbar::handle:bottom {
# #             image: url(Icons/toolbar_handle_h.png);
# #             width: 10px;}'''      
# #     '''QToolBar::handle:left, QToolbar::handle:right {
# #             image: url(Icons/toolbar_handle_v.png);
# #             height: 10px;}'''   
# #     '''QToolButton {
# #             padding: 10px;}'''
# #     %(CASPER_DARK, SAN_MARINO, SAN_MARINO, BLACK_PEARL))

# SS_mainToolbar = ( 
#     '''QToolBar {
#             background-color: %s;
#             border: 3px groove %s;
#             border-radius: 4px;
#             margin: 4px;
#             spacing: 16px;}'''
#     '''QToolBar::handle {
#             background-color: %s;
#             border: 1px solid %s;
#             border-radius: 2px;}'''
#     '''QToolBar::handle:top, QToolbar::handle:bottom {
#             image: url(Icons/toolbar_handle_h.png);
#             width: 10px;}'''      
#     '''QToolBar::handle:left, QToolbar::handle:right {
#             image: url(Icons/toolbar_handle_v.png);
#             height: 10px;}''' 
#     '''QToolBar::separator {
#             background: %s;
#             height: 3px;
#             margin: 7px;
#             width: 3px;}'''  

#     '''QToolButton {
#             border: 2px solid transparent;
#             border-radius: 3px;
#             icon-size: 32px;
#             padding: 10px;}'''
#     '''QToolButton:pressed {
#             border-color: %s;}'''
#     '''QToolButton:checked {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 %s, stop: 0.5 %s, 
#                                               stop: 1 %s);
#             border-color: %s;}'''
#     '''QToolButton:hover {
#             background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                               stop: 0 white, stop: 1 %s);}'''
#     '''QToolBarExtension {
#             padding: 0px;}'''

#     %(CASPER_DARK, SAN_MARINO, SAN_MARINO, BLACK_PEARL, SAN_MARINO, SAN_MARINO, 
#       IVORY, CASPER_LIGHT, CASPER_DARK, SAN_MARINO, BLOSSOM))


# SS_mainWindow = (
#     '''QMainWindow::separator {
#             width: 8px;
#             height: 8px;
#             border-radius: 2px;
#             margin: 8px;}'''
#     '''QMainWindow::separator:hover {
#             background: qradialgradient(cx: 0.5, cy: 0.5, radius: 0.5,
#                                         fx: 0.5, fy: 0.5,
#                                         stop: 0 %s, stop: 1 %s);
#             border: 1px solid %s;}'''
#     %(BLOSSOM_LIGHT, BLOSSOM, BLACK_PEARL))


# SS_pane = (
#     '''QDockWidget {
#         font-weight: bold;
#         titlebar-close-icon: url(Icons/close_pane.png);
#         titlebar-normal-icon: url(Icons/undock_pane.png);}'''
#     '''QDockWidget::title {
#         background: %s;
#         border: 2px outset %s;
#         border-bottom-width: 0px;
#         border-top-left-radius: 1px;
#         border-top-right-radius: 1px;
#         padding: 8px;}'''
#     '''QDockWidget::close-button, QDockWidget::float-button {
#         border-radius: 3px;
#         subcontrol-position: right;
#         subcontrol-origin: padding;
#         width: 16px;}'''
#     '''QDockWidget::close-button {
#         right: 10px;}'''
#     '''QDockWidget::float-button {
#         right: 33px;}'''
#     '''QDockWidget::close-button:hover, QDockWidget::float-button:hover {
#         background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
#                                     stop: 0 white, stop: 1 %s);
#         border: 1px solid %s;}'''
#     '''QDockWidget::close-button:pressed, QDockWidget::float-button:pressed {
#         border: 1px solid %s;}'''
#     '''QScrollArea#PaneScrollArea, QGroupBox#PaneGroupArea {
#         border: 2px solid %s;
#         border-bottom-left-radius: 2px;
#         border-bottom-right-radius: 2px;
#         margin-right: 1px;
#         margin-left: 1px;
#         padding: 8px;}'''
#     %(SAN_MARINO_LIGHT, BLACK_PEARL, BLOSSOM, BLOSSOM, SAN_MARINO, BLACK_PEARL))