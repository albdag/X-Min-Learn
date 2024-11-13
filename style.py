# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:41:40 2024

@author: albdag
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette

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
#                                  FONT 
#  -------------------------------------------------------------------------  #

def getFont(family: str): # !!!
    pointsize = pref.get_setting('GUI/fontsize')
    
    if family == 'default':
        font = QFont()
        font.setPointSize(pointsize)
    else:
        font = QFont(family, pointsize)

    return font


#  -------------------------------------------------------------------------  #
#                                PALETTE 
#  -------------------------------------------------------------------------  #

def getPalette(kind: str):
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

    QPushButton:flat {{
        border: none;
        background-color: transparent;
    }}

    QPushButton:!flat:hover {{
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
    '''
)


SS_RADIOBUTTON = (
    '''
    QRadioButton::indicator::checked {
        image: url(Icons/radiobtn_checked.png);
    }
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
        background-color: darkgray;
    }}

    QComboBox::down-arrow {{
        image: url(Icons/arrowDown.png);
    }}

    QComboBox::down-arrow:!enabled {{
        image: url(Icons/arrowDown_dark.png);
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
        border-color: {SAN_MARINO};
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
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
        image: url(Icons/done.png);
    }}
    '''
)


SS_MENUBAR = (
    f'''
    QMenuBar {{
        background-color: {IVORY};
        border: 1px outset transparent;
        padding: 3px;
    }}

    QMenuBar::item:selected {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1, 
            stop: 0 white, stop: 1 {BLOSSOM}
        );
        border: 1px outset {BLACK_PEARL};
        border-radius: 3px;
    }}
    '''
)


SS_SCROLLBARH = (
    f'''
    QScrollBar:horizontal {{
        border: 1px solid {BLACK_PEARL};
        background: {CASPER_DARK};
        height: 18px;
        margin: 0px 20px 0px 20px;
    }}

    QScrollBar::handle:horizontal {{
        border: 1px solid {BLACK_PEARL};
        background: {IVORY};
        min-width: 20px;
    }}

    QScrollBar::handle:horizontal:hover {{
        border: 2px solid {SAN_MARINO};
    }}

    QScrollBar::add-line:horizontal {{
        border: 2px solid {SAN_MARINO};
        border-radius: 4px;
        background: {BLACK_PEARL};
        width: 20px;
        subcontrol-position: right;
        subcontrol-origin: margin;
    }}

    QScrollBar::add-line:horizontal:hover {{
        border: 2px solid {BLACK_PEARL};
        background: {SAN_MARINO};
    }}

    QScrollBar::sub-line:horizontal {{
        border: 2px solid {SAN_MARINO};
        border-radius: 4px;
        background: {BLACK_PEARL};
        width: 20px;
        subcontrol-position: left;
        subcontrol-origin: margin;
    }}

    QScrollBar::sub-line:horizontal:hover {{
        border: 2px solid {BLACK_PEARL};
        background: {SAN_MARINO};
    }}

    QScrollBar::left-arrow:horizontal,
    QScrollBar::right-arrow:horizontal {{
        border: 1px solid {SAN_MARINO};
        width: 3px;
        height: 3px;
        background: {IVORY};
    }}

    QScrollBar::left-arrow:horizontal:pressed,
    QScrollBar::right-arrow:horizontal:pressed {{
        background: {BLACK_PEARL};
    }}
    '''
)


SS_SCROLLBARV = (
    f'''
    QScrollBar:vertical {{
        border: 1px solid {BLACK_PEARL};
        background: {CASPER_DARK};
        width: 18px;
        margin: 20px 0px 20px 0px;
    }}

    QScrollBar::handle:vertical {{
        border: 1px solid {BLACK_PEARL};
        background: {IVORY};
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        border: 2px solid {SAN_MARINO};
    }}

    QScrollBar::add-line:vertical {{
        border: 2px solid {SAN_MARINO};
        border-radius: 4px;
        background: {BLACK_PEARL};
        height: 20px;
        subcontrol-position: bottom;
        subcontrol-origin: margin;
    }}

    QScrollBar::add-line:vertical:hover {{
        border: 2px solid {BLACK_PEARL};
        background: {SAN_MARINO};
    }}

    QScrollBar::sub-line:vertical {{
        border: 2px solid {SAN_MARINO};
        border-radius: 4px;
        background: {BLACK_PEARL};
        height: 20px;
        subcontrol-position: top;
        subcontrol-origin: margin;
    }}

    QScrollBar::sub-line:vertical:hover {{
        border: 2px solid {BLACK_PEARL};
        background: {SAN_MARINO};
    }}

    QScrollBar::up-arrow:vertical,
    QScrollBar::down-arrow:vertical {{
        border: 1px solid {SAN_MARINO};
        width: 3px;
        height: 3px;
        background: {IVORY};
    }}

    QScrollBar::up-arrow:vertical:pressed,
    QScrollBar::down-arrow:vertical:pressed {{
        background: {BLACK_PEARL};
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
        background: {SAN_MARINO};
        height: 2px;
        margin: 7px;
        width: 2px;
    }}

    QToolButton {{
        border: 2px solid transparent;
        border-radius: 3px;
    }}

    QToolButton[popupMode="1"] {{
        padding-right: 12px;
    }}

    QToolButton::menu-button {{
        border: 1px solid transparent;
        border-radius: 3px;
    }}

    QToolButton::menu-button:hover {{
        border-color: {BLACK_PEARL};
    }}

    QToolButton:pressed {{
        border-color: {SAN_MARINO};
    }}

    QToolButton:checked {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 0.5 {CASPER_LIGHT}, stop: 1 {CASPER_DARK}
        );
        border-color: {SAN_MARINO};
    }}

    QToolButton:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
    }}
    '''
)


SS_MAINTOOLBAR = (
    f'''
    QToolBar {{
        background-color: {CASPER_DARK};
        border: 3px groove {SAN_MARINO};
        border-radius: 4px;
        margin: 4px;
        spacing: 16px;
    }}

    QToolBar::handle {{
        background-color: {SAN_MARINO};
        border: 1px solid {BLACK_PEARL};
        border-radius: 2px;
    }}

    QToolBar::handle:top,
    QToolBar::handle:bottom {{
        image: url(Icons/toolbar_handle_h.png);
        width: 10px;
    }}

    QToolBar::handle:left,
    QToolBar::handle:right {{
        image: url(Icons/toolbar_handle_v.png);
        height: 10px;
    }}

    QToolBar::separator {{
        background: {SAN_MARINO};
        height: 3px;
        margin: 7px;
        width: 3px;
    }}

    QToolButton {{
        border: 2px solid transparent;
        border-radius: 3px;
        icon-size: 32px;
        padding: 10px;
    }}

    QToolButton:pressed {{
        border-color: {SAN_MARINO};
    }}

    QToolButton:checked {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 {IVORY}, stop: 0.5 {CASPER_LIGHT}, stop: 1 {CASPER_DARK}
        );
        border-color: {SAN_MARINO};
    }}

    QToolButton:hover {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
    }}

    QToolBarExtension {{
        padding: 0px;
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

    QTabBar::tab {{
        background: {IVORY};
        border: 1px solid {CASPER_DARK};
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        min-width: 8ex;
        padding-top: 2px;
        padding-bottom: 2px;
        padding-right: 6px;
        padding-left: 6px;
        margin-right: 4px;
    }}

    QTabBar::tab:!selected {{
        margin-top: 3px;
    }}

    QTabBar::tab:selected {{
        border-color: {SAN_MARINO};
        border-bottom: 5px solid {SAN_MARINO};
    }}

    QTabBar::tab:hover {{
        background: {SAN_MARINO};
    }}

    QTabBar::close-button {{
        image: url(Icons/close_pane.png);
        subcontrol-position: right;
    }}

    QTabBar::close-button:pressed {{
        border: 1px solid {SAN_MARINO};
    }}
    '''
)


SS_MAINTABWIDGET = (
    SS_TABWIDGET +
    f'''
    QTabWidget::pane {{
        background-color: {CASPER_DARK};
        border: 3px solid {SAN_MARINO};
        border-top: 4px solid {SAN_MARINO};
        border-top-right-radius: 4px;
        border-top-left-radius: 0px;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 4px;
    }}
    '''
)


SS_SPLITTER = (
    f'''
    QSplitter::handle {{
        background-color: {SAN_MARINO};
        border: 1px solid {BLACK_PEARL};
        border-radius: 2px;
        margin: 10px;
    }}

    QSplitter::handle:hover {{
        background-color: qradialgradient(
            cx: 0.5, cy: 0.5, radius: 0.5,
            fx: 0.5, fy: 0.5,
            stop: 0 {BLOSSOM_LIGHT}, stop: 1 {BLOSSOM}
        );
    }}
    '''
)


SS_GROUPAREA_NOTITLE = (
    f'''
    QGroupBox {{
        border: 2px solid {SAN_MARINO};
        border-radius: 3px;
        padding-top: 2px;
    }}

    QGroupBox:enabled {{
        background-color: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 1,
            stop: 0 {IVORY}, stop: 0.25 {CASPER_LIGHT},
            stop: 0.75 {CASPER_LIGHT}, stop: 1 {CASPER}
        );
    }}
    '''
)


SS_GROUPAREA_TITLE = (
    SS_GROUPAREA_NOTITLE + 
    f'''
    QGroupBox {{
        font: bold;
        padding-top: 8ex;
    }}

    QGroupBox::title {{
        color: {BLACK_PEARL};
        padding: 0px;
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
        border: 1px solid {SAN_MARINO};
        border-radius: 2px;
        padding: 1px;
    }}
    '''
)


SS_GROUPSCROLLAREA_NOFRAME = (
    SS_GROUPSCROLLAREA_FRAME +
    '''
    QScrollArea {
        border-color: transparent;
    }
    '''
)


SS_PATHLABEL = (
    f'''
    QLabel {{
        background-color: {BLACK_PEARL};
        border: 3px inset {SAN_MARINO};
        border-radius: 3px;
        color: {IVORY};
        padding: 2px;
    }}
    '''
)


SS_MAINWINDOW = (
    f'''
    QMainWindow::separator {{
        width: 8px;
        height: 8px;
        border-radius: 2px;
        margin: 8px;
    }}

    QMainWindow::separator:hover {{
        background: qradialgradient(
            cx: 0.5, cy: 0.5, radius: 0.5,
            fx: 0.5, fy: 0.5,
            stop: 0 {BLOSSOM_LIGHT}, stop: 1 {BLOSSOM}
        );
        border: 1px solid {BLACK_PEARL};
    }}
    '''
)


SS_PANE = (
    f'''
    QDockWidget {{
        font-weight: bold;
        titlebar-close-icon: url(Icons/close_pane.png);
        titlebar-normal-icon: url(Icons/undock_pane.png);
    }}

    QDockWidget::title {{
        background: {SAN_MARINO_LIGHT};
        border: 2px outset {BLACK_PEARL};
        border-bottom-width: 0px;
        border-top-left-radius: 1px;
        border-top-right-radius: 1px;
        padding: 8px;
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
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 white, stop: 1 {BLOSSOM}
        );
        border: 1px solid {BLOSSOM};
    }}

    QDockWidget::close-button:pressed,
    QDockWidget::float-button:pressed {{
        border: 1px solid {SAN_MARINO};
    }}

    QScrollArea#PaneScrollArea,
    QGroupBox#PaneGroupArea {{
        border: 2px solid {BLACK_PEARL};
        border-bottom-left-radius: 2px;
        border-bottom-right-radius: 2px;
        margin-right: 1px;
        margin-left: 1px;
        padding: 8px;
    }}
    '''
)


SS_DATAMANAGER = (
    SS_MENU +
    '''
    QTreeWidget::branch:has-children:!has-siblings:closed,
    QTreeWidget::branch:closed:has-children:has-siblings {
        border-image: none;
        image: url(Icons/arrowRight.png);
    }

    QTreeWidget::branch:open:has-children:!has-siblings,
    QTreeWidget::branch:open:has-children:has-siblings {
        border-image: none;
        image: url(Icons/arrowDown.png);
    }
    '''
)