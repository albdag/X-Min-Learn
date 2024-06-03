# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:09:21 2021

@author: albdag
"""

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtGui import QColor, QFont, QPalette


# SETTINGS
# QC.QCoreApplication.setOrganizationName('GeoDataLab')
# QC.QCoreApplication.setApplicationName('X-Min Learn')
settings = QSettings('.//settings//X-MinLearn.ini', QSettings.IniFormat)


# X-Min Learn GUI colors
BLACK_PEARL = '#19232D'         # (25, 35, 45)
# CASPER = '#A9B9BC'              # (169, 185, 188)
CASPER = '#C9C9C9'              # (201, 201, 201)
# CASPER_DARK = '#9AA8AB'         # (154, 168, 171)
CASPER_DARK = '#B0B0B0'         # (188, 188, 188)
SAN_MARINO = '#497096'          # (73, 112, 150)
IVORY = '#F9F9F4'               # (249, 249, 244)
BLOSSOM = '#EEB5B6'             # (238, 181, 182)
XML_PETROLEUM = '#005858'       # (0, 88, 88)

# Special button colors
BTN_GREEN = '#1CA916'           # (28, 169, 22)
BTN_RED = '#A21709'             # (162, 23, 9)

# Histogram mask color
HIST_MASK = '#FFA500BF'         # (255, 165, 0, alpha=0.75)


# All style-sheets should go into a new module "stylesheets"
SS_menu = (
    '''QMenu {
            background-color: %s;
            border: 1px solid %s;}'''
    '''QMenu::item {
            padding: 2px 40px 2px 10px;
            border: 1px solid transparent;}'''
    '''QMenu::item:!enabled {
            color: gray;}'''
    '''QMenu::item:selected {
            border-color: %s;
            background: %s;}'''
    '''QMenu::icon {
            padding: 1px 1px 1px 2px;
            border: 2px solid transparent;
            border-radius: 4px;}'''
    '''QMenu::icon:checked {
            border-color: %s;}'''
    '''QMenu::indicator {
            background-color: %s;
            border: 1px solid %s;
            left: 4px;
            height: 13px;
            width: 13px;}'''
    '''QMenu::indicator:checked {
            image: url(Icons/done.png);}'''
    %(IVORY, BLACK_PEARL, SAN_MARINO, BLOSSOM, SAN_MARINO, IVORY, BLACK_PEARL))


SS_menuBar = (
    '''QMenuBar {
            background-color: %s;}'''
    '''QMenuBar::item:selected {
            background-color: %s;
            margin-top: 2px;
            border: 1px solid %s;
            border-top-left-radius: 2px;
            border-top-right-radius: 2px;}'''
    %(IVORY, BLOSSOM, BLACK_PEARL))


SS_splitter = (
    '''QSplitter::handle {
            background-color: %s;
            border: 1px solid %s;
            border-radius: 2px;
            margin: 10px;}'''
    '''QSplitter::handle:hover {
            background-color: %s;}'''
    %(SAN_MARINO, BLACK_PEARL, BLOSSOM))


SS_horizScrollBar = (
    '''QScrollBar:horizontal {
            border: 1px solid %s;
            background: %s;
            height: 18px;
            margin: 0px 20px 0px 20px;}'''
    '''QScrollBar::handle:horizontal {
            border: 1px solid %s;
            background: %s;
            min-width: 20px;}'''
    '''QScrollBar::handle:horizontal:hover {
            border: 2px solid %s;}'''
    '''QScrollBar::add-line:horizontal {
            border: 2px solid %s;
            border-radius: 4px;
            background: %s;
            width: 20px;
            subcontrol-position: right;
            subcontrol-origin: margin;}'''
    '''QScrollBar::add-line:horizontal:hover {
            border: 2px solid %s;
            background: %s;}'''
    '''QScrollBar::sub-line:horizontal {
            border: 2px solid %s;
            border-radius: 4px;
            background: %s;
            width: 20px;
            subcontrol-position: left;
            subcontrol-origin: margin;}'''
    '''QScrollBar::sub-line:horizontal:hover {
            border: 2px solid %s;
            background: %s;}'''
    '''QScrollBar::left-arrow:horizontal,
       QScrollBar::right-arrow:horizontal {
            border: 1px solid %s;
            width: 3px;
            height: 3px;
            background: %s;}'''
    '''QScrollBar::left-arrow:horizontal:pressed,
       QScrollBar::right-arrow:horizontal:pressed {
            background: %s;}'''
    %(BLACK_PEARL, CASPER_DARK, BLACK_PEARL, IVORY, SAN_MARINO, SAN_MARINO,
      BLACK_PEARL, BLACK_PEARL, SAN_MARINO, SAN_MARINO, BLACK_PEARL,
      BLACK_PEARL, SAN_MARINO, SAN_MARINO, IVORY, BLACK_PEARL))



SS_vertScrollBar = (
    '''QScrollBar:vertical {
            border: 1px solid %s;
            background: %s;
            width: 18px;
            margin: 20px 0px 20px 0px;}'''
    '''QScrollBar::handle:vertical {
            border: 1px solid %s;
            background: %s;
            min-height: 20px;}'''
    '''QScrollBar::handle:vertical:hover {
            border: 2px solid %s;}'''
    '''QScrollBar::add-line:vertical {
            border: 2px solid %s;
            border-radius: 4px;
            background: %s;
            height: 20px;
            subcontrol-position: bottom;
            subcontrol-origin: margin;}'''
    '''QScrollBar::add-line:vertical:hover {
            border: 2px solid %s;
            background: %s;}'''
    '''QScrollBar::sub-line:vertical {
            border: 2px solid %s;
            border-radius: 4px;
            background: %s;
            height: 20px;
            subcontrol-position: top;
            subcontrol-origin: margin;}'''
    '''QScrollBar::sub-line:vertical:hover {
            border: 2px solid %s;
            background: %s;}'''
    '''QScrollBar::up-arrow:vertical,
       QScrollBar::down-arrow:vertical {
            border: 1px solid %s;
            width: 3px;
            height: 3px;
            background: %s;}'''
    '''QScrollBar::up-arrow:vertical:pressed,
       QScrollBar::down-arrow:vertical:pressed {
            background: %s;}'''
    %(BLACK_PEARL, CASPER_DARK, BLACK_PEARL, IVORY, SAN_MARINO, SAN_MARINO,
      BLACK_PEARL, BLACK_PEARL, SAN_MARINO, SAN_MARINO, BLACK_PEARL,
      BLACK_PEARL, SAN_MARINO, SAN_MARINO, IVORY, BLACK_PEARL))


SS_combox = (
    '''QComboBox {
            background-color: %s;
            border: 2px solid %s;
            border-radius: 3px;
            color: black;
            selection-background-color: %s;}'''
    '''QComboBox:!enabled {
            background-color: lightgray;
            border: 1px solid gray;
            color: darkgray;}'''
    '''QComboBox:hover {
            border: 2px solid %s;}'''
    '''QComboBox::drop-down {
            background-color: %s;
            border-left: 1px solid %s;
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 16px;}'''
    '''QComboBox::drop-down:!enabled {
            background-color: gray;}'''
    '''QComboBox::down-arrow {
            image: url(Icons/arrowDown.png);}'''
    '''QComboBox QAbstractItemView {
            background-color: %s;
            border: 1px solid %s;}'''
    %(IVORY, BLACK_PEARL, BLOSSOM, SAN_MARINO, BLACK_PEARL, BLACK_PEARL,
      IVORY, SAN_MARINO))


SS_button = (
    '''QPushButton {
            background-color: %s;
            border: 1px solid %s;
            border-radius: 4px;
            padding: 5px;}'''
    '''QPushButton:!enabled {
            background-color: lightgray;
            border: 1px solid gray;}'''
    '''QPushButton:flat {
            border: none;
            background-color: transparent;}'''
    '''QPushButton:hover {
            background-color: %s;}'''
    '''QPushButton:flat:hover {
            background-color: transparent;}'''
    '''QPushButton:pressed {
            border: 2px solid %s;}'''
    '''QPushButton:checked {
            background-color: %s;
            border: 2px solid %s;}'''
    %(IVORY, BLACK_PEARL, BLOSSOM, BLACK_PEARL, BLOSSOM, BLACK_PEARL))


SS_radioButton = (
    '''QRadioButton::indicator::checked {
            image: url(Icons/radiobtn_checked.png);}''')

SS_dataManager = (SS_menu +
    '''QTreeWidget::branch:has-children:!has-siblings:closed,
       QTreeWidget::branch:closed:has-children:has-siblings {
            border-image: none;
            image: url(Icons/arrowRight.png);}'''
    '''QTreeWidget::branch:open:has-children:!has-siblings,
       QTreeWidget::branch:open:has-children:has-siblings {
            border-image: none;
            image: url(Icons/arrowDown.png);}''')


SS_tabWidget = (
    '''QTabWidget::pane {
            background: %s;
            border-width: 0px;}'''
    '''QTabBar::tab {
            background: %s;
            border: 1px solid darkgray;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            min-width: 8ex;
            padding-top: 2px;
            padding-bottom: 2px;
            padding-right: 6px;
            padding-left: 6px;
            margin-left: 4px;
            margin-right: 4px;}'''
    '''QTabBar::tab:!selected {
            margin-top: 3px;}'''
    '''QTabBar::tab:selected {
            border-color: %s;
            border-bottom: 5px solid %s;}'''
    '''QTabBar::tab:hover {
            background: %s;}'''
    %(CASPER_DARK, IVORY, SAN_MARINO, SAN_MARINO, SAN_MARINO))


ss_mainTabWidget = (SS_tabWidget +
    '''QTabWidget::pane {
            border: 2px solid %s;
            border-top: 3px solid %s;
            border-top-right-radius: 4px;
            border-bottom-right-radius: 4px;
            border-bottom-left-radius: 4px;
            margin-right: 4px;
            margin-bottom: 4px;
            margin-left: 4px;}'''
    %(SAN_MARINO, SAN_MARINO))


SS_table = (
    '''QTableWidget QTableCornerButton::section {
            background: %s;
            border: 1px outset %s;
            border-radius: 1px;}'''
    %(IVORY, BLACK_PEARL))


SS_pathLabel = (
    '''QLabel {
            border: 2px solid %s;
            border-radius: 3px;
            padding: 2px;}'''
    %(SAN_MARINO))


SS_grouparea_notitle = (   
    '''QGroupBox {
            border: 1px solid %s;
            border-radius: 3px;
            padding-top: 2px;}'''
    %(BLACK_PEARL))


SS_grouparea_title = (SS_grouparea_notitle + 
    '''QGroupBox {
            font: bold;
            padding-top: 7ex;}'''
    '''QGroupBox::title {
            color: black;
            padding: 2ex;
            subcontrol-origin: padding;}''')


SS_groupScrollArea_frame = (
    '''QScrollArea {
            border: 1px solid %s;
            padding: 1px;}'''
    %(BLACK_PEARL))


SS_groupScrollArea_noframe = (SS_groupScrollArea_frame +
    '''QScrollArea {
            border-color: transparent;}''')          

SS_toolbar = (SS_menu +
    '''QToolButton:hover {
            background-color: %s;}'''
    '''QToolBar::separator {
            background: %s;
            height: 2px;
            margin: 7px;
            width: 2px;}'''
    %(BLOSSOM, SAN_MARINO))


SS_mainToolbar = (SS_toolbar + 
    '''QToolBar {
            background-color: %s;
            border: 3px groove %s;
            border-radius: 4px;
            margin: 4px;
            spacing: 16px;}'''
    '''QToolBar::handle {
            background-color: %s;
            border: 1px solid %s;
            border-radius: 2px;}'''
    '''QToolBar::handle:top, QToolbar::handle:bottom {
            image: url(Icons/toolbar_handle_h.png);
            width: 10px;}'''      
    '''QToolBar::handle:left, QToolbar::handle:right {
            image: url(Icons/toolbar_handle_v.png);
            height: 10px;}'''   
    '''QToolButton {
            padding: 10px;}'''
    %(CASPER_DARK, SAN_MARINO, SAN_MARINO, BLACK_PEARL))


SS_mainWindow = (
    '''QMainWindow::separator {
            width: 8px;
            height: 8px;
            border-radius: 2px;
            margin: 3px;}'''
    '''QMainWindow::separator:hover {
            background: %s;
            border: 1px solid %s;}'''
    %(BLOSSOM, BLACK_PEARL))


SS_pane = (
    '''QDockWidget {
        border: 1px solid %s;
        font-weight: bold;}'''
    '''QDockWidget::title {
        background: %s;}'''
    '''QDockWidget::close-button, QDockWidget::float-button {
        background: %s;
        border: 1px solid %s;
        padding: -1px;}'''
    '''QDockWidget::close-button:hover, QDockWidget::float-button:hover {
        background: %s;}'''
    '''QDockWidget::close-button:pressed, QDockWidget::float-button:pressed {
        border: 2px solid %s;
        padding: 1px;}'''
    %(BLACK_PEARL, CASPER_DARK, IVORY, SAN_MARINO, BLOSSOM, SAN_MARINO))




def get_setting(name, default=None, type=None):
    if type is None:
        return settings.value(name, default)
    else:
        return settings.value(name, default, type)

def edit_setting(name, value):
    settings.setValue(name, value)

def clear_settings():
    settings.clear()

# def setAppFont(fontsize, font=QFont()): # !!!
#     app = QApplication.instance()
#     # size = f'{font.pointSize()}pt'
#     size = f'{fontsize}pt'
#     name = font.family()
#     app.setStyleSheet('''QWidget{font: %s %s;}''' %(size, name))
#     # font.setPointSize(int(fontsize))
#     # app.setFont(font)

def setAppFont(app): # !!!
    font = app.font()
    font.setPointSize(get_setting('main/fontsize', 10, type=int))
    app.setFont(font)

def setAppPalette(app, kind='default'):
    if kind == 'default':
        palette = app.palette()

    # Window (A general background color)
        palette.setColor(QPalette.Active, QPalette.Window,
                         QColor(CASPER)) #QColor(169, 185, 188))
        palette.setColor(QPalette.Inactive, QPalette.Window,
                         QColor(CASPER)) # QG.QColor(46, 64, 82)

    # WindowText (A general foreground color)
        palette.setColor(QPalette.WindowText, Qt.black)

    # Base (Background for text entry widgets, comboboxes, toolbar handle etc.)
        palette.setColor(QPalette.Active, QPalette.Base,
                         QColor(BLACK_PEARL)) # QG.QColor(64, 89, 115)
        palette.setColor(QPalette.Inactive, QPalette.Base,
                         QColor(BLACK_PEARL)) #QG.QColor(87, 101, 115)
        palette.setColor(QPalette.Disabled, QPalette.Base,
                         Qt.darkGray)

    # AlternateBase (Alternate background in views with alternating row colors)
        # palette.setColor(QPalette.AlternateBase, Qt.red)

    # ToolTips background & text
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipBase,
        #                   QColor(*SAN_MARINO))
        # palette.setColor(QPalette.Inactive, QPalette.ToolTipText,
        #                   QColor(*IVORY))

    # PlaceholderText (Placeholder color for various text input widgets)
        palette.setColor(QPalette.PlaceholderText, Qt.lightGray)

    # Text (The foreground color used with Base)
        palette.setColor(QPalette.Active, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.Text, QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.Text, Qt.lightGray)

    # Button (The general button background color)
        palette.setColor(QPalette.Active, QPalette.Button,
                         QColor(IVORY))
        palette.setColor(QPalette.Inactive, QPalette.Button,
                         QColor(IVORY))
        palette.setColor(QPalette.Disabled, QPalette.Button,
                         Qt.lightGray)

    # ButtonText (A foreground color used with the Button color)
        palette.setColor(QPalette.Active, QPalette.ButtonText,
                         Qt.black)
        palette.setColor(QPalette.Inactive, QPalette.ButtonText,
                         Qt.black)
        palette.setColor(QPalette.Disabled, QPalette.ButtonText,
                         Qt.darkGray)

    # Highlight and highlighted text
        palette.setColor(QPalette.Highlight, QColor(BLOSSOM))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        # palette.setColor(QPalette.BrightText, Qt.red)
        # palette.setColor(QPalette.Link, Qt.red)

        app.setPalette(palette)

def get_dirPath(direction):
    assert direction in ('in', 'out')
    path = get_setting(f'system/{direction}dirPath', '.\\')
    return path

def set_dirPath(direction, path):
    assert direction in ('in', 'out')
    edit_setting(f'system/{direction}dirPath', path)


