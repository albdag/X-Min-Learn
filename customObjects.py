# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:27:17 2021

@author: albdag
"""

from PyQt5 import QtWidgets as QW
from PyQt5.QtGui import (QColor, QCursor, QFont, QFontMetrics, QDrag, QRegion,
                        QIcon, QImage, QIntValidator, QPixmap, QTextDocument)
from PyQt5.QtCore import QLocale, QSize, Qt, pyqtSignal, QMimeData, QPoint

from ast import literal_eval
from weakref import proxy
from os.path import dirname, exists, splitext

import numpy as np
from scipy import ndimage as nd
from pandas import read_csv, concat

from matplotlib import (colors as mpl_colors, colormaps as mpl_cmaps, patheffects as mpe,
                        text as mpl_text, use as mpl_use)
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import (FigureCanvasQTAgg,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, PolygonSelector, SpanSelector, MultiCursor
from mpl_toolkits.axes_grid1 import make_axes_locatable

import preferences as pref
import conv_functions as CF
import plots
from _base import InputMap, MineralMap, RoiMap, Mask

# mpl_use('Qt5Agg')
















class DataGroup(QW.QTreeWidgetItem):
    '''
    A class for data groups. A data group can be considered as a rock sample,
    that holds both input maps (under the 'Input Maps' subgroup), mineral maps
    (under the 'Mineral Maps' subgroup) and [in future] point analysis data
    (under the 'Point Analysis' subgroup).
    '''

    def __init__(self, name):
        '''
        DataGroup class constructor.

        Parameters
        ----------
        name : str
            The name of the group.

        Returns
        -------
        None.

        '''
        super(DataGroup, self).__init__()

    # Set the flags. Data groups are selectable and their name is editable
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable |
                      Qt.ItemIsEnabled | Qt.ItemIsEditable)

    # Set the font as bold
        font = self.font(0)
        font.setBold(True)
        self.setFont(0, font)
        self.setText(0, name)

    # Add the subgroups
        self.inmaps = DataSubGroup('Input Maps')
        self.inmaps.setIcon(0, QIcon(r'Icons/inmap.png'))
        self.minmaps = DataSubGroup('Mineral Maps')
        self.minmaps.setIcon(0, QIcon(r'Icons/minmap.png'))
        self.masks = DataSubGroup('Masks')
        self.masks.setIcon(0, QIcon(r'Icons/mask.png'))
        # add self.points = DataSubGroup('Point Analysis')
        self.subgroups = (self.inmaps, self.minmaps, self.masks)
        self.addChildren(self.subgroups)


    def setShapeWarnings(self):
        '''
        Set a warning state to every loaded data object whose shape differs
        from the sample overall trending shape.

        Returns
        -------
        None.

        '''
    # Collect in a single list the objects from all subgroup
        # include points data (maybe?)
        objects = []
        for subgr in self.subgroups:
            objects.extend(subgr.getChildren())

    # Extract the shape from each object data and obtain the trending shape  
        if len(objects):
            shapes = [o.get('data').shape for o in objects]
            trend_shape = CF.most_frequent(shapes)

    # Set a warning state to each object whose data shape differs from trend
            for idx, shp in enumerate(shapes):
                objects[idx].setWarning(shp != trend_shape)


    def getCompositeMask(self, include='selected', mode='intersection',
                         ignore_single_mask=False):
        '''
        Get the mask array resulting from merging all the checked or selected
        masks loaded in this group.

        Parameters
        ----------
        include : str, optional
            Whether to merge the 'selected' or 'checked' maps. The default is
            'selected'.
        mode : str, optional
            How the composite mask should be constructed. If 'union' or 'U', it
            is the product of all the masks. If 'intersection' or 'I', it is
            the sum of all the masks. The default is 'union'.
        ignore_single_mask : bool, optional
            Whether the function should not return a composite mask if only one
            mask is selected/checked. The default is False.

        Raises
        ------
        TypeError
            The include argument is not one of 'selected' and 'checked'.
        TypeError
            The mode argument is not one of 'union' and 'intersection'.

        Returns
        -------
        comp_mask : Mask object or None
            The composite mask object or None if no mask is included.

        '''
        cld = self.masks.getChildren()

        if include == 'selected':
            masks = [c.get('data') for c in cld if c.isSelected()]
        elif include == 'checked':
            masks = [c.get('data') for c in cld if c.checkState(0)]
        else:
            raise TypeError('f{include} is not a valid argument for include.')

        if len(masks) == 0:
            comp_mask = None
        elif len(masks) == 1:
            comp_mask = None if ignore_single_mask else masks[0]
        else:
            comp_mask = Mask(CF.merge_masks([m.mask for m in masks], mode))

        return comp_mask


    def clear(self):
        for subgr in self.subgroups:
            subgr.takeChildren()



class DataSubGroup(QW.QTreeWidgetItem):
    '''
    A class for data subgroups. A data subgroup is a convenient object that
    separates the different data types within the same group. Their main
    function is to provide a better organization and visualization of the data
    in the manager. Therefore they offer less customization options to users.
    '''

    def __init__(self, name):
        '''
        DataSubGroup class constructor.

        Parameters
        ----------
        name : str
            The name of the subgroup.

        Returns
        -------
        None.

        '''
        super(DataSubGroup, self).__init__()

    # Set main attributes
        self.name = name

    # Set the flags. Data subgroups can be selected but cannot be edited
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable |
                      Qt.ItemIsEnabled)

    # Set the font as underlined
        font = self.font(0)
        # font.setItalic(True)
        font.setUnderline(True)
        self.setFont(0, font)
        self.setText(0, name)

    def isEmpty(self):
        '''
        Check if this DataSubGroup object is populated with data or not.

        Returns
        -------
        empty : bool
            Whether or not the object is empty.

        '''
        empty = self.childCount() == 0
        return empty


    def addData(self, data):
        '''
        Add data to the subgroup in the form of data objects (i.e., instances
        of DataObject).

        Parameters
        ----------
        data : InputMap, MineralMap, Mask or PointAnalysis [in future].
            The data to be added to the subgroup.

        Returns
        -------
        None.

        '''
        self.addChild(DataObject(data))
        self.parent().setShapeWarnings()


    def delChild(self, child):
        self.takeChild(self.indexOfChild(child))
        self.parent().setShapeWarnings()


    def clear(self):
        self.takeChildren()
        self.parent().setShapeWarnings()


    def getChildren(self):
        '''
        Get all the DataObject items owned by this subgroup.

        Returns
        -------
        children : List
            List of DataObject children.

        '''
        children = [self.child(idx) for idx in range(self.childCount())]
        return children



class DataObject(QW.QTreeWidgetItem):
    '''
    A class for data objects. Data objects are special containers of data, that
    permit a convenient access to all the data-related information. Such
    information (e.g., filepath, arrays, icons etc.) can be stored and
    organized within a data object using different columns and item roles. The
    data object class is a reimplementation of a QTreeWidgetItem whose type is
    set to 'UserType' in order to unlock more customization options for the
    developer.
    '''

    def __init__(self, data):
        '''
        DataObject class constructor.

        Parameters
        ----------
        data : InputMap, MineralMap, Mask or PointAnalysis [in future]
            The data linked to the data object.

        Returns
        -------
        None.

        '''
    # Set the data object type to 'User Type' for more customization options
        super(DataObject, self).__init__(type=QW.QTreeWidgetItem.UserType)

    # Set the flags. A data object is selectable and editable by the user.
        self.setFlags(Qt.ItemIsSelectable | Qt.ItemIsUserCheckable |
                      Qt.ItemIsEnabled | Qt.ItemIsEditable)

    # Set the data with custom role. It is a user-role (int in [0, 256]) that
    # does not overwrite any default Qt role. It is set arbitrarily to 100
        self.setData(0, 100, data)
    # Set the display name of data as its filename, using a DisplayRole (= 0)
        self.setData(0, 0, self.generateDisplayName())
    # Set the "edited" state as False, using a user-role (= 101).
        self.setData(0, 101, False)
    # A save icon is shown if item is edited --> DecorationRole (= 1)
        self.setData(0, 1, QIcon())
    # A tooltip indicates if item is edited --> ToolTipRole (= 3)
        self.setData(0, 3, '')
    # Set the "warning" state as False, using a user-role (= 102).
        self.setData(0, 102, False)
    # A warn icon is shown (2nd col) if item has warnings --> DecorationRole (= 1)
        self.setData(1, 1, QIcon())
    # A tooltip (2nd col) indicates if item has warnings --> ToolTipRole (= 3)
        self.setData(1, 3, '')

    # Set the "checked" state for togglable data (Masks and Points [in future])
        if isinstance(data, (Mask,)): # add PointAnalysis class
            self.setData(0, 10, Qt.Unchecked) # CheckedRole (= 10)


    def generateDisplayName(self):
        '''
        Generate a display name for the object based on its filepath. If the
        object has an invalid filepath, a generic name will be generated.

        Returns
        -------
        name : str
            Display name.

        '''
        data = self.get('data')
        filepath = data.filepath
        if filepath is None:
            if isinstance(data, (InputMap, MineralMap)):
                obj_type = 'map'
            else:
                obj_type = 'mask'
            # add Point Analysis here
            name = f'Unnamed {obj_type}'
        else:
            name = CF.path2fileName(filepath)

        return name

    def holdsInputMap(self):
        '''
        Check if the object holds input map data.

        Returns
        -------
        bool
            Wether or not the object holds input map data.

        '''
        return isinstance(self.get('data'), InputMap)


    def holdsMineralMap(self):
        '''
        Check if the object holds mineral map data.

        Returns
        -------
        bool
            Wether or not the object holds mineral map data.
        '''
        return isinstance(self.get('data'), MineralMap)


    def holdsMap(self):
        '''
        Check if the object holds generic map data.

        Returns
        -------
        bool
            Wether or not the object holds map data.

        '''
        return self.holdsInputMap() or self.holdsMineralMap()


    def holdsMask(self):
        '''
        Check if the object holds mask data.

        Returns
        -------
        bool
            Wether or not the object holds mask data.
        '''
        return isinstance(self.get('data'), Mask)


    # def holdsPointsData(self):





    def get(self, *args):
        '''
        Convenient function to get data from the object.

        Parameters
        ----------
        *args : str
            One or multiple <attributes> keys.

        Returns
        -------
        out : list
            Requested object data. It returns just the first element of the list
            if its lenght is 1.

        '''

    # Define a dictionary holding all the object's attributes
        attributes = {'data' : self.data(0, 100),
                      'name' : self.data(0, 0),
                      'is_edited' : self.data(0, 101),
                      'has_warning' : self.data(0, 102),
                      'save_icon' : self.data(0, 1),
                      'warn_icon' : self.data(1, 1),
                      'edit_tooltip' : self.data(0, 3),
                      'warn_tooltip' : self.data(1, 3),
                      'checked' : self.data(0, 10)
                      }

    # Raises a keyerror if invalid arg is passed
        out = [attributes[a] for a in args]
        if len(out) == 1: out = out[0]
        return out


    def setEdited(self, edited):
    # Set the 'isEdited' attribute
        self.setData(0, 101, edited)
    # Show/hide the save icon
        icon = QIcon('Icons/edit_white.png') if edited else QIcon()
        self.setData(0, 1, icon)
    # Show/hide the edited tooltip
        text = 'Edits not saved' if edited else ''
        self.setData(0, 3, text)


    def setWarning(self, warning):
    # Set the 'has_warning' attribute
        self.setData(1, 102, warning)
    # Show/hide the warn icon
        icon = QIcon('Icons/warnIcon.png') if warning else QIcon()
        self.setData(1, 1, icon)
    # Show/hide the warning tooltip
        text = 'Unfitting shapes' if warning else ''
        self.setData(1, 3, text)















# class Legend(QW.QTreeWidget):
#     '''
#     A legend object linked to a plotted classified image.
#     '''

#     instances = []
#     itemColorChanged = pyqtSignal(DataObject)
#     itemRenamed = pyqtSignal(DataObject)

#     def __init__(self, amounts=True, parent=None):
#         '''
#         Legend class constructor.

#         Parameters
#         ----------
#         amounts : bool, optional
#             Include classes amounts (percentage) in the legend. The default is
#             True.
#         parent : QWidget or None, optional
#             GUI parent widget of the legend. The default is None.

#         Returns
#         -------
#         None.

#         '''
#     # Weakly track all class instances
#         self.__class__.instances.append(proxy(self))

#     # Call the constructor of the parent class
#         super(Legend, self).__init__(parent)

#     # Define main attributes
#         self.amounts = amounts
#         self.precision = pref.get_setting('plots/legendDec', 3, type=int)

#     # Initialize the current mineral map and the current parent_item. The parent
#     # item is the currently selected mineral map item in the DataManager
#         self.current_parent_item = None
#         self.current_mineral_map = None

#     # Customize the legend appearence and properties (headers & selection mode)
#         self.setColumnCount(1 + self.amounts)
#         self.setHeaderLabels(['Class'] + ['Amount'] * self.amounts)
#         self.header().setSectionResizeMode(QW.QHeaderView.Interactive)
#         self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

#     # Set custom scrollbars
#         self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
#         self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))

#     # Enable custom context menu
#         self.setContextMenuPolicy(Qt.CustomContextMenu)

#     # Set events connections
#         # self.itemClicked.connect(self.selectItem) find a quick and responsive way
#         self.itemDoubleClicked.connect(self.changeItemColor)
#         self.customContextMenuRequested.connect(self.showContextMenu)



#     def showContextMenu(self, point):
#         '''
#         Shows a context menu with custom actions.

#         Parameters
#         ----------
#         point : QPoint
#             The position of the context menu event that the widget receives.

#         Returns
#         -------
#         None.

#         '''

#     # Get the item that is clicked from <point> and define a menu.
#         item = self.itemAt(point)
#         if item is None: return
#         menu = QW.QMenu()

#     # Rename class
#         rename = QW.QAction(QIcon('Icons/rename.png'), 'Rename')
#         rename.triggered.connect(lambda: self.rename_class(item))

#     # to do Merge classes

#     # Copy current color HEX string
#         copy_color = QW.QAction('Copy color string')
#         copy_color.triggered.connect(lambda: self.copyColorHexToClipboard(item))

#     # Change color
#         set_color = QW.QAction(QIcon('Icons/palette.png'), 'Set color')
#         set_color.triggered.connect(lambda: self.changeItemColor(item))

#     # Randomize color
#         random_color = QW.QAction(QIcon('Icons/randomize_color.png'),
#                                   'Randomize color')
#         random_color.triggered.connect(lambda: self.randomize_color(item))

#     # Randomize palette
#         random_palette = QW.QAction(QIcon('Icons/randomize_color.png'),
#                                    'Randomize full colormap')
#         random_palette.triggered.connect(self.randomize_palette)


#     # Add actions to menu
#         menu.addAction(rename)
#         menu.addSeparator()
#         menu.addActions([copy_color, set_color, random_color, random_palette])

#     # Set the menu style-sheet
#         menu.setStyleSheet(pref.SS_menu)

#     # Show the menu in the same spot where the user triggered the event
#         menu.exec(QCursor.pos())


#     def rename_class(self, item):
#         old_name = item.text(0)
#         name, ok = QW.QInputDialog.getText(self, 'X-Min Learn',
#                                            'Rename class (max. 8 ASCII '\
#                                            'characters):', text=f'{old_name}')

#     # Proceed only if the new name is an ASCII <= 8 characters string
#         if ok and 0 < len(name) < 9 and name.isascii():

#         # Send error if name is already taken
#             if name in self.current_mineral_map.get_phases():
#                 return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                                f'{name} class already exists.')

#         # Set the new name to the legend item
#             item.setText(0, name)

#         # Update the Mineral Map
#             self.current_mineral_map.rename_phase(old_name, name)

#         # Emit the pyqt signal
#             self.itemRenamed.emit(self.current_parent_item)



#     def copyColorHexToClipboard(self, item):
#         '''
#         Copy the selected phase color to the clipboard as a HEX string.

#         Parameters
#         ----------
#         item : QTreeWidgetItem
#             The selected phase item.

#         Returns
#         -------
#         None.

#         '''
#     # Get the hex string of phase color
#         phase_name = item.text(0)
#         rgb_color = self.current_mineral_map.get_phase_color(phase_name)
#         hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
#     # Copy the string to the clipboard
#         clipboard = QW.QApplication.clipboard()
#         clipboard.setText(hex_color)



#     def changeItemColor(self, item):
#         '''
#         Sets the color of a class. This function is called when double-clicking
#         on a legend's item.

#         Parameters
#         ----------
#         item : QTreeWidgetItem object
#             The double-clicked item in the legend.

#         Returns
#         -------
#         None.

#         '''
#     # Get the old color and the new color
#         phase = item.text(0)
#         old_col = self.current_mineral_map.get_phase_color(phase)
#         new_col = QW.QColorDialog.getColor(initial=QColor(*old_col))

#         if new_col.isValid():
#         # Transform the new color to an RGB tuple
#             rgb = tuple(new_col.getRgb()[:-1])

#         # Set the new color to the legend item
#             item.setIcon(0, RGBIcon(rgb))

#         # Update the Mineral Map palette
#             self.current_mineral_map.set_phase_color(phase, rgb)

#         # Emit the pyqt signal
#             self.itemColorChanged.emit(self.current_parent_item)



#     def randomize_palette(self):
#     # Randomize the Mineral Map palette
#         rand_palette = self.current_mineral_map.rand_colorlist()
#         self.current_mineral_map.set_palette(rand_palette)

#     # Update the legend
#         self.update(self.current_mineral_map)

#     # Emit the pyqt signal
#         self.itemColorChanged.emit(self.current_parent_item)



#     def randomize_color(self, item):
#     # Randomize selected phase color
#         phase_name = item.text(0)
#         rand_color = self.current_mineral_map.rand_colorlist(1)[0]
#         self.current_mineral_map.set_phase_color(phase_name, rand_color)

#     # Update the legend
#         self.update(self.current_mineral_map)

#     # Emit the pyqt signal
#         self.itemColorChanged.emit(self.current_parent_item)



#     def set_precision(self, value):
#         '''
#         Set the number of decimals of the class amounts.

#         Parameters
#         ----------
#         value : int
#             Number of decimals to be shown in the legend.

#         Returns
#         -------
#         None.

#         '''
#         self.precision = value
#         self.update()



#     def setParentItem(self, parent_item):
#     # Safety: check if parent item holds mineral map data
#         if parent_item.holdsMineralMap():
#             self.current_parent_item = parent_item


#     def update(self, mineral_map):
#         '''
#         Updates the legend.

#         Parameters
#         ----------
#         mineral_map : MineralMap
#             The Mineral Map object linked to the current view of the legend.

#         Returns
#         -------
#         None.

#         '''

#     # Clear the legend
#         self.clear()

#     # Save the mineral map as the current mineral map
#         self.current_mineral_map = mineral_map

#     # Set icon and text of each unique phase in the Mineral Map
#         phases = mineral_map.get_phases()
#         for p in phases:
#             color = mineral_map.get_phase_color(p)
#             text = [p]
#         # Add the amounts in the second column (if enabled)
#             if self.amounts:
#                 amount = round(mineral_map.get_phase_amount(p), self.precision)
#                 text.append(f'{amount}%')
#             i = QW.QTreeWidgetItem(self, text)
#             i.setIcon(0, RGBIcon(color))


class Legend(QW.QTreeWidget):
    '''
    An legend object that allows data editing if set as interactive. It sends
    various signals to notify each edit, which must be catched and handled by
    other widget(s).
    '''

    instances = []
    colorChangeRequested = pyqtSignal(QW.QTreeWidgetItem, tuple) # item, color
    randomPaletteRequested = pyqtSignal()
    itemRenameRequested = pyqtSignal(QW.QTreeWidgetItem, str) # item, name
    itemsMergeRequested = pyqtSignal(list, str) # list of classes, name
    itemHighlightRequested = pyqtSignal(bool, int) # highlight on/off, item index
    maskExtractionRequested = pyqtSignal(list) # list of classes



    def __init__(self, amounts=True, interactive=False, parent=None):
        '''
        Legend class constructor.

        Parameters
        ----------
        amounts : bool, optional
            Include classes amounts (percentage) in the legend. The default is
            True.
        interactive : bool, optional
            Whether the legend should allow editing of its items. The default
            is False.
        parent : QWidget or None, optional
            GUI parent widget of the legend. The default is None.

        Returns
        -------
        None.

        '''
    # Weakly track all class instances
        self.__class__.instances.append(proxy(self))

    # Call the constructor of the parent class
        super(Legend, self).__init__(parent)

    # Define main attributes
        self._highlighted_item = None
        self.amounts = amounts
        self.interactive = interactive
        self.precision = pref.get_setting('plots/legendDec', 3, type=int)
        
    # Customize the legend appearence and properties (headers & selection mode)
        self.setColumnCount(2 + self.amounts)
        self.setHeaderLabels(['Color', 'Class'] + ['Amount'] * self.amounts)
        self.header().setSectionResizeMode(QW.QHeaderView.Interactive)
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

    # Disable default editing. Item editing is forced via requestClassRename()
        self.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)

    # Set custom scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))

    # Enable custom context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    # Set stylesheet (right-click menu when editing items name)
        self.setStyleSheet(pref.SS_menu)

    # Set events connections if the legend is interactive
        if interactive:
            self._connect_slots()


    def _connect_slots(self):
        self.itemDoubleClicked.connect(self.onDoubleClick)
        self.customContextMenuRequested.connect(self.showContextMenu)


    def showContextMenu(self, point):
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        Returns
        -------
        None.

        '''

    # Get the item that is clicked from <point> and define a menu.
        i = self.itemAt(point)
        if i is None: return
        menu = StyledMenu()

    # Rename class
        rename = QW.QAction(QIcon('Icons/rename.png'), 'Rename')
        rename.triggered.connect(lambda: self.requestClassRename(i))

    # Merge classes
        merge = QW.QAction('Merge')
        merge.setEnabled(len(self.selectedItems()) > 1)
        merge.triggered.connect(self.requestClassMerge)

    # Copy current color HEX string
        copy_color = QW.QAction('Copy color string')
        copy_color.triggered.connect(lambda: self.copyColorHexToClipboard(i))

    # Change color
        set_color = QW.QAction(QIcon('Icons/palette.png'), 'Set color')
        set_color.triggered.connect(lambda: self.requestColorChange(i))

    # Randomize color
        rand_color = QW.QAction(QIcon('Icons/randomize_color.png'),
                                'Randomize color')
        rand_color.triggered.connect(lambda: self.requestRandomColorChange(i))

    # Randomize palette
        rand_palette = QW.QAction(QIcon('Icons/randomize_color.png'),
                                  'Randomize full colormap')
        rand_palette.triggered.connect(self.randomPaletteRequested.emit)

    # Higlight item
        highlight = QW.QAction(QIcon(r'Icons/highlight.png'), 'Highlight')
        highlight.setCheckable(True)
        highlight.setChecked(i == self._highlighted_item)
        highlight.toggled.connect(lambda t: self.requestItemHighlight(t, i))

    # Extract mask
        extract_mask = QW.QAction('Extract mask')
        extract_mask.triggered.connect(self.requestMaskFromClass)


    # Add actions to menu
        menu.addAction(rename)
        menu.addAction(merge)
        menu.addSeparator()
        menu.addActions([copy_color, set_color, rand_color, rand_palette])
        menu.addSeparator()
        menu.addAction(highlight)
        menu.addAction(extract_mask)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def onDoubleClick(self, item, column):
        if column == 0:
            self.requestColorChange(item)
        elif column == 1:
            self.requestClassRename(item)
        else:
            return

    def copyColorHexToClipboard(self, item):
        '''
        Copy the selected phase color to the clipboard as a HEX string.

        Parameters
        ----------
        item : QTreeWidgetItem
            The selected phase item.

        Returns
        -------
        None.

        '''
    # Get the hex string of phase color
        rgb_color = literal_eval(item.whatsThis(0)) # tuple parsing from string
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
    # Copy the string to the clipboard
        clipboard = QW.QApplication.clipboard()
        clipboard.setText(hex_color)


    def requestColorChange(self, item):
        '''
        Request to change the color of a class by sending a signal. The signal
        must be catched and handled by the widget that contains the legend.
        This function is also triggered when double-clicking on an item.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.

        Returns
        -------
        None.

        '''
    # Get the old color and the new color (as rgb tuple)
        old_col = literal_eval(item.whatsThis(0)) # tuple parsing from str
        new_col = QW.QColorDialog.getColor(initial=QColor(*old_col))
    # Emit the signal
        if new_col.isValid():
            rgb = tuple(new_col.getRgb()[:-1])
            self.colorChangeRequested.emit(item, rgb)


    def requestRandomColorChange(self, item):
        '''
        Request to randomize the color of a class by sending a signal. The
        signal must be catched and handled by the widget that contains the
        legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.

        Returns
        -------
        None.

        '''
        self.colorChangeRequested.emit(item, ())


    def changeItemColor(self, item, color):
        '''
        Change the color of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item whose color must be changed.
        color : tuple
            RGB color triplet.

        Returns
        -------
        None.

        '''
    # Set the new color to the legend item
        item.setIcon(0, RGBIcon(color))
    # Also set the new whatsThis string
        item.setWhatsThis(0, str(color))



    def requestClassRename(self, item):
        '''
        Request to change the name of a class by sending a signal. The signal
        must be catched and handled by the widget that contains the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests to be renamed.

        Returns
        -------
        None.

        '''
        old_name = item.text(1)
        name, ok = QW.QInputDialog.getText(self, 'X-Min Learn',
                                           'Rename class (max. 8 ASCII '\
                                           'characters):', text=f'{old_name}')
        if not ok:
            return
    # Proceed only if the new name is an ASCII <= 8 characters string
        elif 0 < len(name) <= 8 and name.isascii():
            self.itemRenameRequested.emit(item, name)
        else:
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'Invalid name.')


    def rename_class(self, item, name):
        '''
        Change the name of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item to be renamed.
        name : str
            New item name.

        Returns
        -------
        None.

        '''
    # Set the new name to the legend item
        item.setText(1, name)


    def requestClassMerge(self):
        '''
        Request to merge two or more mineral classes.

        Returns
        -------
        None.

        '''
        classes = [i.text(1) for i in self.selectedItems()]
        if len(classes) > 1:
            text = f'Merge {classes} in a new class (max. 8 ASCII characters):'
            name, ok = QW.QInputDialog.getText(self, 'X-Min Learn', text)
            if not ok:
                return
        # Proceed only if the new name is an ASCII <= 8 characters string
            elif 0 < len(name) <= 8 and name.isascii():
                self.itemsMergeRequested.emit(classes, name)
            else:
                return QW.QMessageBox.critical(self, 'X-Min Learn',
                                               'Invalid name.')


    def requestItemHighlight(self, toggled, item):
        self._highlighted_item = item if toggled else None
        item_idx = self.indexOfTopLevelItem(item)
        self.itemHighlightRequested.emit(toggled, item_idx)


    def requestMaskFromClass(self):
        '''
        Request to extract a mask from the selected mineral classes.

        Returns
        -------
        None.

        '''
        classes = [i.text(1) for i in self.selectedItems()]
        self.maskExtractionRequested.emit(classes)


    def set_precision(self, value):
        '''
        Set the number of decimals of the class amounts.

        Parameters
        ----------
        value : int
            Number of decimals to be shown in the legend.

        Returns
        -------
        None.

        '''
        self.precision = value
        # self.update()



    def update(self, mineral_map):
        '''
        Updates the legend.

        Parameters
        ----------
        mineral_map : MineralMap
            The Mineral Map object linked to the current view of the legend.

        Returns
        -------
        None.

        '''
    # Clear the legend
        self.clear()

    # Reset the highlighted item reference
        self._highlighted_item = None

    # Populate the legend with mineral classes
        phases = mineral_map.get_phases()
        for p in phases:
            color = mineral_map.get_phase_color(p)

            i = QW.QTreeWidgetItem(self)
            i.setIcon(0, RGBIcon(color))    # icon [column 0]
            i.setWhatsThis(0, str(color))   # RGB string ['virtual' column 0]
            i.setText(1, p)                 # phase name [column 1]
            if self.amounts:                # amounts (optional) [column 2]
                amount = round(mineral_map.get_phase_amount(p), self.precision)
                i.setText(2, f'{amount}%')

    # Resize columns
        self.header().resizeSections(QW.QHeaderView.ResizeToContents)









class StyledTabWidget(QW.QTabWidget):
    def __init__(self, parent=None):
        super(StyledTabWidget, self).__init__(parent)

        self.setStyleSheet(pref.SS_tabWidget)












# class MatrixCanvas(FigureCanvasQTAgg):

#     def __init__(self, parent=None, size=(9, 9), title='',
#                   xlab='', ylab='', cbar=True, tight=False):
#         self.fig = Figure(figsize=size, tight_layout=tight)
#         self.fig.patch.set(facecolor='w', edgecolor='#19232D', linewidth=2)
#         self.ax = self.fig.add_subplot(111)
#         super(MatrixCanvas, self).__init__(self.fig)

#         self.title = title
#         self.xlab = xlab
#         self.ylab = ylab

#         self.mtx = None
#         self.showCbar = cbar
#         self.cax = None

#         self.init_ax()

#     def init_ax(self):
#         self.ax.set_title(self.title)
#         self.ax.set_xlabel(self.xlab)
#         self.ax.set_ylabel(self.ylab)

#     def isEmpty(self):
#         return self.mtx == None

#     def set_ticks(self, labels, axis='both'):
#         assert axis in ('x', 'y', 'both')
#         if axis in ('x', 'both'):
#             self.ax.set(xticks=np.arange(len(labels)), xticklabels=labels)
#             self.ax.tick_params(labelbottom=True, labeltop=False)
#         if axis in ('y', 'both'):
#             self.ax.set(yticks=np.arange(len(labels)), yticklabels=labels)

#         self.draw()
#         self.flush_events()

#     def annotate(self, data):
#         n_rows, n_cols = data.shape
#         row_ind = np.arange(n_rows)
#         col_ind = np.arange(n_cols)
#         x_coord, y_coord = np.meshgrid(col_ind, row_ind)

#         for txt, x, y in zip(data.flatten(), x_coord.flatten(), y_coord.flatten()):
#             self.ax.annotate(txt, (x, y), va='center', ha='center', color='w',
#                               path_effects=[mpe.withStroke(linewidth=2,
#                                                           foreground='k')])

#     def remove_annotations(self):
#         for child in self.ax.get_children():
#             if isinstance(child, mpl_text.Annotation):
#                 child.remove()

#     def update_canvas(self, data):
#         if self.isEmpty():
#             self.mtx = self.ax.matshow(data, cmap='inferno', interpolation='none')
#             if self.showCbar:
#                 divider = make_axes_locatable(self.ax)
#                 self.cax = divider.append_axes("right", size="5%", pad=0.1)
#                 self.cbar = self.fig.colorbar(self.mtx, cax=self.cax)
#         else:
#             self.mtx.set_data(data)
#             self.mtx.set_clim(data.min(), data.max())
#             # set extent is to allow to plot a different shaped matrix without the need of call
#             # clear_canvas() before
#             self.mtx.set_extent((-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5))
#             self.remove_annotations()

#         self.annotate(data)
#         self.draw()
#         self.flush_events()

#     def clear_canvas(self):
#         self.mtx = None
#         self.ax.cla()
#         if self.showCbar and self.cax is not None:
#             self.cax.cla()
#         self.init_ax()
#         self.draw()
#         self.flush_events()






class CurvePlotCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, size=(6.4, 4.8), tight=False,
                 title='', xlab='', ylab='', grid=True):
        self.fig = Figure(figsize=size, tight_layout=tight)
        self.fig.patch.set(facecolor='w', edgecolor='#19232D', linewidth=2)
        self.ax = self.fig.add_subplot(111)
        super(CurvePlotCanvas, self).__init__(self.fig)

        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.gridOn = grid

        self.init_ax()

    def init_ax(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
        self.ax.grid(self.gridOn)

    def add_curve(self, xdata, ydata, label='', color='k'):
        self.ax.plot(xdata, ydata, label = label, color=color)

    def has_curves(self):
        return bool(len(self.ax.lines))

    def update_canvas(self, data=None):
    # data has to be a list of tuple, each one containing xdata and ydata of a single curve
    # example: data = [(x1, y1), (x2, y2)]
        if data is not None:
            curves = self.ax.lines # get all the plot instances (Line2D)
            assert len(data) == len(curves)
            for n in range(len(curves)):
                curves[n].set_data(data[n])
            self.ax.relim()
            self.ax.autoscale_view()

        self.ax.legend(loc='best')
        self.draw()
        self.flush_events()

    def homeAction(self):
        self.ax.relim()
        self.ax.autoscale()
        self.draw()
        self.flush_events()

    def clear_canvas(self):
        self.ax.cla()
        self.init_ax()
        self.draw()
        self.flush_events()


class PieCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, size=(5,5), tight=False, _3D=True, perc='%d%%'):
        self.fig = Figure(figsize=size, tight_layout=tight)
        self.fig.patch.set(facecolor=CF.RGB2float([(169, 185, 188)]),
                           edgecolor='#19232D', linewidth=2)
        self.ax = self.fig.add_subplot(111)
        super(PieCanvas, self).__init__(self.fig)

        self.setSizePolicy(QW.QSizePolicy.Fixed, QW.QSizePolicy.Fixed)
        self._3D = _3D
        self.perc_lbl = perc
        self.pie = None

        self.init_ax()

    def init_ax(self):
        self.ax.cla()
        self.ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        self.ax.axis('off')

    def update_canvas(self, data, labels, title=''):
        if self.pie is not None:
            self.init_ax()
        self.ax.set_title(title)
        self.pie = self.ax.pie(data, explode=[.1]*len(data) if self._3D else None,
                               labels=labels, autopct=self.perc_lbl, shadow=self._3D)
        self.draw()
        self.flush_events()

    def clear_canvas(self):
        self.init_ax()
        self.draw()
        self.flush_events()


class SilhouetteScoreCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, size=(6.4, 4.8), tight=False):
        self.fig = Figure(figsize=size, tight_layout=tight)
        self.fig.patch.set(facecolor='w', edgecolor='#19232D', linewidth=2)
        self.ax = self.fig.add_subplot(111)
        super(SilhouetteScoreCanvas, self).__init__(self.fig)

        self.title = 'Silhouette Plot'
        self.xlab = 'Silhouette Coefficient'
        self.ylab = 'Cluster'
        self.y_btm_init = 15

        self.init_ax()

    def init_ax(self):
        self.ax.cla()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)


    def alterColors(self, colors):
        _colors = dict(zip(colors.keys(), CF.RGB2float(colors.values())))
        idx = 0
        for artist in self.ax.get_children():
            if isinstance(artist, PolyCollection):
                lbl = artist.get_label()
                artist.set(fc = _colors[lbl])
                idx += 1
        self.draw()
        self.flush_events()

    def update_canvas(self, sil_values, sil_avg, colors):
        '''sil_values = dict(i_cluster_class : sil_values_for_class_i)'''
        self.clear_canvas()
        y_btm = self.y_btm_init
        _colors = dict(zip(colors.keys(), CF.RGB2float(colors.values())))


        for cluster, values in sil_values.items():
            values.sort()
            y_top = y_btm + len(values)

            self.ax.fill_betweenx(np.arange(y_btm, y_top), 0, values, label=cluster,
                                  fc=_colors[cluster], ec='black', lw=0.3)

            y_btm = y_top + self.y_btm_init

        # Draw the average value line and set the legend referring to it
        avg_ = self.ax.axvline(x=sil_avg, color='r', ls='--', lw=2,
                               path_effects=[mpe.withStroke(foreground='k')])
        self.ax.legend([avg_], [f'Avg score (excl. _ND_)\n{sil_avg}' ], loc='best')

        self.draw()
        self.flush_events()

    def clear_canvas(self):
        self.init_ax()
        self.draw()
        self.flush_events()




class RGBIcon(QIcon):
    def __init__(self, rgb_tuple, size=(64, 64)):
        self.rgb = rgb_tuple
        self.size = size

        pixmap = QPixmap(*self.size)
        pixmap.fill(QColor(*self.rgb))

        super(RGBIcon, self).__init__(pixmap)



class Crosshair(MultiCursor): # !!! Not used yet
    def __init__(self, canvas, *axes):
        super(Crosshair, self).__init__(canvas, axes=axes, useblit=True,
                                        horizOn=True, vertOn=True,
                                        color='r', lw=1)



class PolySel(PolygonSelector): # future improvement to ROIs

    def __init__(self, canvasAx, onselect, useblit=True):
        self.props = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        super(PolySel, self).__init__(canvasAx, onselect, useblit=useblit,
                                      props=self.props, grab_range=10)
        self.ax = canvasAx
        self.onselect = onselect
        self.canvas = self.ax.figure.canvas
        self.useblit = useblit

        self.set_active(False)
        self.canvas.mpl_connect('figure_enter_event', lambda evt: self.updateCursor())
        self.canvas.mpl_connect('scroll_event', lambda evt: self.updateCursor())
        # # Callback to update rectangle selector with useblit=True when zooming
        if self.useblit:
            self.canvas.mpl_connect('motion_notify_event', self.update_)
            self.canvas.mpl_connect('scroll_event', self.update_)


    def updateCursor(self):
        if self.active:
            self.canvas.setCursor(QCursor(Qt.PointingHandCursor))
        # else:
        #     self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def update_(self, event):
        if self.active:
            self.update()


class RectSel(RectangleSelector):
    '''
    A customized rectangle selector widget, tailored to ROI selection.

    '''
    def __init__(self, ax, onselect, interactive=True, btns=[1]):
        '''
        RectSel class constructor.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The ax where the selector must be drawn.
        onselect : function
            Callback function that is called after the selection is created.
        useblit : bool, optional
            Whether to use blitting for faster rendering. The default is True.
        btns : list or None, optional
            List of mouse buttons that can trigger the drawing event. Left = 1,
            Middle = 2 and Right = 3. If None all the buttons are included. The
            default is [1].

        Returns
        -------
        None.

        '''
    # Customize the appearence of the rectangle selector and of its handles
        rect_props = dict(fc=pref.BLACK_PEARL, ec=pref.BLOSSOM, alpha=0.8, lw=2,
                          fill=True)
        handle_props = dict(mfc=pref.BLOSSOM, mec=pref.BLACK_PEARL, alpha=1)

    # drag_from_anywhere=True causes a rendering glitch when a former selection
    # is active, you draw a new one and, without releasing the left button, you
    # start resizing it.
        kwargs = {'minspanx': 1,
                  'minspany': 1,
                  'useblit': True,
                  'props': rect_props,
                  'spancoords': 'data',
                  'button': btns,
                  'grab_range': 10,
                  'handle_props': handle_props,
                  'interactive': interactive,
                  'drag_from_anywhere': True}

        super(RectSel, self).__init__(ax, onselect, **kwargs)

    # By default the selector is turned off
        self.set_active(False)


    def fixed_extents(self, shape, fmt='matrix', mode='full'):
        '''
        Get integer extents of the selector after checking if it lies within
        the provided shape. Different output extents format can be selected.

        Parameters
        ----------
        shape : tuple
            Control shape. It must be provided as (rows, cols).
        fmt : STR, optional
            The output format of coords. If 'matrix' the coords format is
            (row_min, row_max, col_min, col_max). If 'xy' the coords format is
            (x_min, x_max, y_min, y_max). The default is 'matrix'.
        mode : str, optional
            How to treat pixels at the border of the selector region. At the
            moment the only supported mode is 'full', which means that only
            the pixels that are entirely covered by the selector region will be
            included. The default is 'full'.

        Returns
        -------
        extents : tuple or None
            Fixed extents as integer indices. If the selector region falls
            entirely outside the map area, extents will be None.
        '''
    # Get the default extents
        xmin, xmax, ymin, ymax = self.extents

    # Exclude pixels not entirely selected --> <mode> : 'full'
    # ymax and xmax should be decreased by one (-1) before being rounded
    # but this is skipped due to range and/or array slice mechanics,
    # where the second index is always excluded -> [xmin, xmax)
        if mode == 'full':
            xmin = round(xmin + 1)
            xmax = round(xmax)
            ymin = round(ymin + 1)
            ymax = round(ymax)
        else:
            raise NameError(f'No mode available for {mode}')

    # Exit function if the extents are completely outside the map
        if xmax < 0 or xmin > shape[1] or ymax < 0 or ymin > shape[0]:
            return None

    # Fix extents that are partially outside the map borders
        if xmin < 0: xmin = 0
        if xmax > shape[1]: xmax = shape[1]
        if ymin < 0: ymin = 0
        if ymax > shape[0]: ymax = shape[0]

    # Return the fixed extents formatted according to <fmt>
        if fmt == 'matrix':
            extents = (ymin, ymax, xmin, xmax)
        elif fmt == 'xy':
            extents = (xmin, xmax, ymin, ymax)
        else:
            raise NameError('No format available for {fmt}')

        return extents


    def fixed_rect_bbox(self, shape, mode='full'):
        '''
        Get the bounding box of the selector after checking if it lies within
        the provided shape. The bounding box values are expressed as integers.

        Parameters
        ----------
        shape : tuple
            Control shape. It must be provided as (rows, cols).
        mode : str, optional
            How to treat pixels at the border of the selector region. At the
            moment the only supported mode is 'full', which means that only
            the pixels that are entirely covered by the selector region will be
            included. The default is 'full'.

        Returns
        -------
        bbox : tuple or None
            Fixed bounding box with integer values. If the selector region
            falls entirely outside the map area, the bounding box will be None.

        '''
    # Get the default bounding box
        x0, y0, w, h = self._rect_bbox

    # Exit function if the bounding box is completely outside the map
        if x0+0.5 > shape[1] or x0+w < -0.5 or y0+0.5 > shape[0] or y0+h < -0.5:
            return None

    # Fix extents that are partially outside the map borders
        if x0 < -0.5:
            w += x0 + 0.5
            x0 = -0.5
        if x0 + w + 0.5 > shape[1]:
            w = shape[1] - 0.5 - x0
        if y0 < -0.5:
            h += y0 + 0.5
            y0 = -0.5
        if y0 + h + 0.5 > shape[0]:
            h = shape[0] - 0.5 - y0

    # Exclude pixels not entirely selected --> <mode> : 'full'
        if mode == 'full':
            _x0 = round(x0 + 1) - 0.5
            _y0 = round(y0 + 1) - 0.5
            w = int(w + x0 - _x0)
            h = int(h + y0 - _y0)

            if w < 1 or h < 1: # It should never happen
                return None
        else:
            raise NameError(f'No mode available for {mode}')

    # Return the fixed bounding box
        bbox = (_x0, _y0, w, h)
        return bbox


    def update(self):
        '''
        Reimplementation of the default update function. It also calls the
        updateCursor function after the default update operations are done.

        Returns
        -------
        None.

        '''
        super(RectSel, self).update()
        self.updateCursor()


    def updateCursor(self):
        '''
        Updates the cursor depending on the state of the selector. When it is
        active, the pointing hand cursor is set, otherwise the arrow cursor.

        Returns
        -------
        None.

        '''
        if self.active:
            self.canvas.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))






class HeatmapScaler(SpanSelector):
    '''
    A customized span selector widget, tailored to scale heatmaps histograms.

    '''

    def __init__(self, ax, onselect, interactive=True, btns=[1]):
        '''
        Constructor.

        Parameters
        ----------
        ax : Matplotlib Axes
            The ax where the span selection is performed.
        onselect : func
            The function that triggers after a selection event.
        interactive : bool, optional
            Wether the selector is interactive. The default is True.
        btns : list or None, optional
            List of buttons that can trigger the selection. Can be left mouse 
            button [1], wheel mouse button [2], right mouse button [3] or None,
            which means all the buttons. The default is [1].

        '''
    # Customize the appearence of the span selector and its handles
        span_props = dict(fc=pref.BLACK_PEARL, ec=pref.BLACK_PEARL, alpha=0.8,
                          fill=True)
        handle_props = dict(linewidth=2, color=pref.BLOSSOM)
    
    # Define the scaler properties
        kwargs = {'direction': 'horizontal',
                  'minspan': 0,
                  'useblit': True,
                  'props': span_props,
                  'interactive': interactive,
                  'button': btns,
                  'handle_props': handle_props,
                  'grab_range': 10,
                  'drag_from_anywhere': True}

        super(HeatmapScaler, self).__init__(ax, onselect, **kwargs)

    # By default the selector is turned off
        self.set_active(False)



class StyledComboBox(QW.QComboBox):
    def __init__(self, parent=None):
        super(StyledComboBox, self).__init__(parent)

        self.setStyleSheet(pref.SS_combox)



class AutoUpdateComboBox(StyledComboBox):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super(AutoUpdateComboBox, self).__init__(parent)

    def showPopup(self):
        self.clicked.emit()
        super(AutoUpdateComboBox, self).showPopup()


class StyledMenu(QW.QMenu):
    def __init__(self, parent=None):
        super(StyledMenu, self).__init__(parent)

    # Set stylesheet
        self.setStyleSheet(pref.SS_menu)





class StyledListWidget(QW.QListWidget):
    def __init__(self, parent=None, extendedSelection=True):
        super(StyledListWidget, self).__init__(parent)
    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))
    # Set extended selection mode if requested
        if extendedSelection:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

    def getItems(self):
        items = [self.item(row) for row in range(self.count())]
        return items


class StyledLineEdit(QW.QLineEdit):
    def __init__(self, text=None, parent=None):
        if text is None:
            super(StyledLineEdit, self).__init__(parent)
        else:
            super(StyledLineEdit, self).__init__(text, parent)
    # Set stylesheet for context menu
        self.setStyleSheet(pref.SS_menu)


class StyledTable(QW.QTableWidget):
    def __init__(self, rows, cols, parent=None, extendedSelection=True):
        super(StyledTable, self).__init__(rows, cols, parent)
    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))
    # Set extended selection mode if requested
        if extendedSelection:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
    # Set stylesheet (corner button and context menu)
        self.setStyleSheet(pref.SS_table + pref.SS_menu)




class StyledSpinBox(QW.QSpinBox):
    def __init__(self, min_value=0, max_value=1, step=1, parent=None):
        super(StyledSpinBox, self).__init__(parent)

    # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

    # Set stylesheet (context menu)
        self.setStyleSheet(pref.SS_menu)


class CBoxMapLayout(QW.QGridLayout):
    cboxPressed = pyqtSignal()

    def __init__(self, paths, parent=None):
        super(CBoxMapLayout, self).__init__()
        self.parent = parent

        self.setColumnStretch(0, 1)
        self.Cbox_list = []


        for idx, pth in enumerate(paths):
            cbox = QW.QCheckBox(CF.path2fileName(pth))
            cbox.setObjectName(str(idx))
            cbox.setChecked(True)
            cbox.pressed.connect(lambda: self.cboxPressed.emit())
            self.Cbox_list.append(cbox)
            rename_btn = IconButton('Icons/rename.png')
            rename_btn.setObjectName(str(idx))
            rename_btn.setFlat(True)
            rename_btn.clicked.connect(self.rename)
            self.addWidget(cbox, idx, 0)
            self.addWidget(rename_btn, idx, 1)

    def rename(self):
        name, ok = QW.QInputDialog.getText(self.parent, 'Edit Name', 'Type name:',
                                           flags=Qt.MSWindowsFixedSizeDialogHint)
        if ok and name != '':
            idx = self.sender().objectName()
            self.Cbox_list[int(idx)].setText(name)




class RadioBtnLayout(QW.QBoxLayout):
    selectionChanged = pyqtSignal(int) # return the id of button

    def __init__(self, btn_names, orient='vertical', parent=None):
        if orient == 'horizontal':
            direction = QW.QBoxLayout.LeftToRight
        elif orient == 'vertical':
            direction = QW.QBoxLayout.TopToBottom
        else:
            raise NameError(f'{orient} is not a valid orientation.')

        super(RadioBtnLayout, self).__init__(direction)
        self.btn_group = QW.QButtonGroup()

        for i, n in enumerate(btn_names):
            btn = QW.QRadioButton(n)
            self.btn_group.addButton(btn, id=i)
            if i == 0: btn.setChecked(True)
            btn.clicked.connect(lambda _, idx=i: self.selectionChanged.emit(idx))
            self.addWidget(btn)


        # self.get_button(0).setChecked(True)

    def get_button(self, id):
        return self.btn_group.button(id)

    def get_buttonList(self) :
        return self.btn_group.buttons()

    def get_checked(self, asID=False):
        if asID: return self.btn_group.checkedId()
        else:    return self.btn_group.checkedButton()


class StyledButton(QW.QPushButton):
    def __init__(self, icon=None, text=None, bg_color=None):
        super(StyledButton, self).__init__()

        if icon is not None:
            self.setIcon(icon)

        if text is not None:
            self.setText(text)

    # Overwrite default background color if requested
        ss = pref.SS_button
        if bg_color is not None:
            ss+= 'StyledButton {background-color: %s; font: bold;}' %(bg_color)
        self.setStyleSheet(ss)

        self.setSizePolicy(QW.QSizePolicy.Preferred, QW.QSizePolicy.Fixed)




class IconButton(StyledButton): # !!! deprecated, use StyledButton instead

    def __init__(self, iconPath, text=None):
        super(IconButton, self).__init__(QIcon(iconPath), text)





class GroupArea(QW.QGroupBox):

    def __init__(self, qObject, title='', checkable=False, parent=None):
        super(GroupArea, self).__init__(title, parent)
        self.setAlignment(Qt.AlignHCenter)
        self.setCheckable(checkable)
        self.setStyleSheet(pref.SS_groupArea)

        if isinstance(qObject, QW.QLayout):
            LayoutBox = qObject
        else:
            LayoutBox = QW.QBoxLayout(QW.QBoxLayout.TopToBottom)
            LayoutBox.addWidget(qObject)

        self.setLayout(LayoutBox)


class StyledScrollBar(QW.QScrollBar):

    def __init__(self, orientation):
        super(StyledScrollBar, self).__init__(orientation)
    # Set the stylesheet
        if orientation == Qt.Horizontal:
            self.setStyleSheet(pref.SS_horizScrollBar)
        else:
            self.setStyleSheet(pref.SS_vertScrollBar)


class GroupScrollArea(QW.QScrollArea):

    def __init__(self, qObject, title='', hscroll=True, vscroll=True,
                 parent=None):
        super(GroupScrollArea, self).__init__(parent)

        if isinstance(qObject, QW.QLayout):
            if title=='':
                wid = QW.QFrame()
                wid.setLayout(qObject)
            else:
                wid = GroupArea(qObject, title)
        else:
            wid = qObject

        self.setWidget(wid)
        self.setWidgetResizable(True)

        self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))

        if not hscroll:
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if not vscroll:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)



class SplitterLayout(QW.QBoxLayout):

    def __init__(self, orient=Qt.Horizontal, parent=None):

        if orient == Qt.Horizontal:
            direction = QW.QBoxLayout.LeftToRight
        else:
            direction = QW.QBoxLayout.TopToBottom

        super(SplitterLayout, self).__init__(direction, parent)
        self.splitter = QW.QSplitter(orient)
        self.splitter.setStyleSheet(pref.SS_splitter)
        super(SplitterLayout, self).addWidget(self.splitter)


    def insertLayout(self, layout, index, stretch=0):
        wid = QW.QFrame()
        wid.setLayout(layout)
        self.splitter.insertWidget(index, wid)
        self.splitter.setStretchFactor(index, stretch)

    def addLayout(self, layout, stretch=0):
        self.insertLayout(layout, -1, stretch)

    def addLayouts(self, layouts, stretches=None):
        if stretches is None:
            stretches = (0,)*len(layouts)
        else:
            assert len(layouts) == len(stretches)

        for l, s in zip(layouts, stretches):
            self.addLayout(l, s)


    def insertWidget(self, widget, index, stretch=0):
        self.splitter.insertWidget(index, widget)
        self.splitter.setStretchFactor(index, stretch)

    def addWidget(self, widget, stretch=0):
        self.insertWidget(widget, -1, stretch)

    def addWidgets(self, widgets, stretches=None):
        if stretches is None:
            stretches = (0,)*len(widgets)
        else:
            assert len(widgets) == len(stretches)

        for w, s in zip(widgets, stretches):
            self.addWidget(w, s)



class SplitterGroup(QW.QSplitter):

    def __init__(self, qObjects=None, stretch=None, orient=Qt.Horizontal):
        super(SplitterGroup, self).__init__(orient)

        self.orient = orient
        self.setStyleSheet(pref.SS_splitter)

        for obj in qObjects:
            if not obj.isWidgetType():
                w_obj = QW.QFrame()
                w_obj.setLayout(obj)
                self.addWidget(w_obj)
            else:
                self.addWidget(obj)

        if stretch is not None:
            assert len(stretch) == len(qObjects)
            for i, s in enumerate(stretch):
                self.setStretchFactor(i, s)


    def addStretchedWidget(self, widget, stretch, idx=-1):
        self.insertWidget(idx, widget)
        self.setStretchFactor(idx, stretch)



class PathLabel(QW.QLabel):

    def __init__(self, fullpath='', full_display=True, elide=True):
        super(PathLabel, self).__init__()
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setStyleSheet(pref.SS_pathLabel + pref.SS_menu)


        self.fullpath = fullpath
        self.full_display = full_display
        self.elide = elide

        self.display()

    @property
    def _displayedText(self):
        text = self.fullpath
        if not self.full_display:
            text = CF.path2fileName(text, ext=True)

        return text

    def display(self):
        text = self._displayedText
        self.setTextElided(text) if self.elide else self.setText(text)
        self.setToolTip(self.fullpath)


    def setTextElided(self, text):
        font = pref.get_setting('main/font', QFont())
        metrics = QFontMetrics(font)
        elided = metrics.elidedText(text, Qt.ElideRight, self.width()-2)
        self.setText(elided)

    def setPath(self, path, auto_display=True):
        if path is None: path = '*Path not found'
        self.fullpath = path
        if auto_display: self.display()

    def clearPath(self):
        self.clear()
        self.fullpath = ''
        self.setToolTip('')

    def resizeEvent(self, event):
        event.accept()
        if self.elide:
            self.setTextElided(self._displayedText)


class PopUpProgBar(QW.QProgressDialog):
# !!! Should the parent be None by default to delete the pbar after execution? (because it has no parent alive)
    def __init__(self, parent, n_iter, label='', cancel=True, forceShow=False):
        btn_text = 'Abort' if cancel else None
        flags = Qt.Tool | Qt.WindowTitleHint | Qt.WindowStaysOnTopHint
        super(PopUpProgBar, self).__init__(label, btn_text, 0, n_iter, parent,
                                           flags=flags)
        self.setWindowModality(Qt.ApplicationModal) # experimental --> original was Qt.WindowModal
        self.setWindowTitle('Please wait...')
        if forceShow:
            self.forceShow()
        else:
            self.setMinimumDuration(1000)
        # ONE OF THE FOLLOWING PRODUCES BUG OF DIALOGS NOT HIDDEN IN WINDOW PREVIEW (NOT REALLY TRUE!)
        # self.setAutoReset(True)
        # self.setAutoClose(True)
        # The real reason is because the window is modal and some instances of this class are called
        # inside a paintEvent (read the doc for QProgressDialog)
        # MAYBE SOLVED WITH THE FLAG <QT.TOOL>

    def increase(self):
        self.setValue(self.value() + 1)




# class PulsePopUpProgBar(QW.QDialog):
#     def __init__(self, text=''):
#         super(PulsePopUpProgBar, self).__init__(flags=Qt.WindowTitleHint)
#         self.setWindowModality(Qt.WindowModal)
#         self.setWindowTitle('Please wait...')

#         label = QW.QLabel(text)
#         label.setMaximumHeight(20)
#         self.progBar = QW.QProgressBar()

#         layout = QW.QVBoxLayout()
#         layout.addWidget(label, alignment=Qt.AlignHCenter)
#         layout.addWidget(self.progBar)
#         self.setLayout(layout)
#         self.show()

#     def start(self):
#         QW.QApplication.processEvents()
#         self.progBar.setRange(0, 0)

#     def stop(self):
#         self.progBar.setRange(0, 1)
#         self.progBar.setValue(1)
#         self.close()







class DecimalPointSelector(StyledComboBox):

    def __init__(self):
        super(DecimalPointSelector, self).__init__()
        local_decimalPoint = QLocale().decimalPoint()
        self.addItems(['.', ','])
        self.setCurrentText(local_decimalPoint)

class CSVSeparatorSelector(StyledComboBox):

    def __init__(self):
        super(CSVSeparatorSelector, self).__init__()
        local_CSVseparator = ',' if QLocale().decimalPoint() == '.' else ';'
        self.addItems([',', ';'])
        self.setCurrentText(local_CSVseparator)

class CsvChunkReader(object):

    def __init__(self, dec, sep=None, eng='python',
                 chunksize=2**20//8, pBar=False):
        self.dec = dec
        self.sep = sep
        self.eng = eng
        self.chunksize = chunksize
        self.pBar = pBar


    def read(self, filepath):

        if self.pBar:
            with open(filepath) as temp:
                nChunks = sum(1 for _ in temp) // self.chunksize
            progBar = PopUpProgBar(None, nChunks + 1, 'Loading Dataset', cancel=False)

        chunkList = []
        with read_csv(filepath, decimal=self.dec, sep=self.sep,
                      engine=self.eng, chunksize=self.chunksize) as reader:
            for n, chunk in enumerate(reader):
                chunkList.append(chunk)
                if self.pBar: progBar.setValue(n)

        dataframe = concat(chunkList)
        if self.pBar: progBar.setValue(nChunks + 1)
        return dataframe


class KernelSelector(QW.QWidget):

    structureChanged = pyqtSignal()

    def __init__(self, rank=2, parent=None):
        super(KernelSelector, self).__init__()
        self.rank=rank

        # Shapes -> Square: 0, Circle: 1, Rhombus: 2
        self.shape = 0
        # self.sq_connect = True
        self.iter = 1

        self.structure = None
        self.build_defaultStructure()

        self.init_ui()

    def init_ui(self):
    # Kernel structure representation
        self.grid = QW.QGridLayout()
        self.build_grid()
        grid_box = GroupArea(self.grid)
        grid_box.setStyleSheet('''border: 2 solid black;
                                  padding-top: 0px;''')

    # # Square connectivity checkbox
    #     self.connect_cbox = QW.QCheckBox('Square Connectivity')
    #     self.connect_cbox.setChecked(self.sq_connect)
    #     self.connect_cbox.stateChanged.connect(self.set_connectivity)

    # Default kernel shapes radio buttons group
        self.kernelShapes_btns = RadioBtnLayout(('Square', 'Circle', 'Rhombus'),
                                                orient='horizontal')
        self.kernelShapes_btns.selectionChanged.connect(self.set_shape)

    # Kernel size slider selector
        self.kernelSize_slider = QW.QSlider(Qt.Horizontal)
        self.kernelSize_slider.setMinimum(1)
        self.kernelSize_slider.setMaximum(5)
        self.kernelSize_slider.setSingleStep(1)
        self.kernelSize_slider.setSliderPosition(self.iter)
        self.kernelSize_slider.valueChanged.connect(self.set_iterations)
        kernelSize_form = QW.QFormLayout()
        kernelSize_form.addRow('Kernel size', self.kernelSize_slider)

    # Adjust main layout
        mainLayout = QW.QVBoxLayout()
        mainLayout.addWidget(grid_box, alignment=Qt.AlignCenter)
        # mainLayout.addWidget(self.connect_cbox)
        mainLayout.addLayout(self.kernelShapes_btns)
        mainLayout.addLayout(kernelSize_form)
        self.setLayout(mainLayout)

    def get_structure(self):
        return self.structure

    def updateKernel(self):
        self.build_defaultStructure()
        self.build_grid()
        self.structureChanged.emit()

    def build_defaultStructure(self):
        # Connectivity (conn) allows to build only rhombic (1) or squared (2) structures
        conn = 1 if self.shape == 2 else 2
        struct = nd.generate_binary_structure(self.rank, conn)
        struct = nd.iterate_structure(struct, self.iter)
        # If a circle (self.shape=1) was required, transform the squared structure into circular
        if self.shape == 1:
            struct = self._drawCircleStructure(struct)
        self.structure = struct

    def _drawCircleStructure(self, baseStructure):
        # J is the number of rows/cols of base (square) structure
        J = baseStructure.shape[0]
        # Compute the Euclidean distance from center (x=J//2; y=J//2)
        Y, X = np.ogrid[:J, :J]
        dist = np.sqrt((X - J//2)**2 + (Y - J//2)**2)
        # Return the circle mask
        return dist <= J/2 # J/2 = radius (in float type)

    def set_shape(self, idx):
        self.shape = idx
        self.updateKernel()

    # def set_connectivity(self, isSquared):
    #     self.sq_connect = isSquared
    #     self.updateKernel()

    def set_iterations(self, value):
        self.iter = value
        self.updateKernel()

    def _clear_grid(self, grid):
        for i in reversed(range(grid.count())):
            grid.itemAt(i).widget().setParent(None)

    def draw_node(self, active):
        img = QImage(8, 8, QImage.Format_RGB32)
        rgb = (255,0,0) if active else (0,0,0)
        img.fill(QColor(*rgb))
        node = QPixmap.fromImage(img)
        return node

    def build_grid(self):
        self._clear_grid(self.grid)
        radius = self.structure.shape[0]
        for row in range(radius):
            for col in range(radius):
                wid = QW.QLabel()
                wid.setPixmap(self.draw_node(self.structure[row, col]))
                self.grid.addWidget(wid, row, col)

    # def build_grid(self):
    #     self._clear_grid(self.grid)
    #     radius = self.structure.shape[0]
    #     for row in range(radius):
    #         for col in range(radius):
    #             node = QW.QPushButton()
    #             node.setMaximumSize(12,12)
    #             node.setCheckable(True)
    #             node.setStyleSheet('''QPushButton {background-color : black;}'''
    #                                '''QPushButton::checked {background-color : red;}''')
    #             node.setChecked(self.structure[row, col])
    #             node.toggled.connect(lambda tgl, r=row, c=col: self.activate_node(tgl, r, c))
    #             self.grid.addWidget(node, row, col)

    # def activate_node(self, active, row, col):
    #     # Avoid Erosion + recontruction freezing bug, by limiting the number of active nodes to 5
    #     if np.count_nonzero(self.structure) <= 5 and not active:
    #         self.sender().setChecked(True)
    #         return
    #     self.structure[row, col] = active
    #     self.structureChanged.emit()
    #     print('FUNCTION Finished')


class RichMsgBox(QW.QMessageBox):
    def __init__(self, parent, icon=QW.QMessageBox.NoIcon, title='', text='',
                 btns=QW.QMessageBox.Ok, default_btn=QW.QMessageBox.NoButton,
                 detailedText='', cbox=None):

        super(RichMsgBox, self).__init__(parent)
        self.setIcon(icon)
        self.setWindowTitle(title)
        self.setText(text)
        self.setStandardButtons(btns)
        self.setDefaultButton(default_btn)
        self.setDetailedText(detailedText)
        self.setCheckBox(cbox)

        self.exec()

class LineSeparator(QW.QFrame):
    def __init__(self, parent=None):
        super(LineSeparator, self).__init__(parent)

        self.setFrameShape(QW.QFrame.HLine)

class PixelFinder(QW.QFrame):
    def __init__(self, canvas, parent=None):
        super(PixelFinder, self).__init__()
        self.canvas = canvas

        self.setStyleSheet('''QFrame {border: 1 solid black;}''')

    # X and Y coords input (with integer validator)
        validator = QIntValidator(0, 10**8)

        self.X_input = StyledLineEdit()
        self.X_input.setAlignment(Qt.AlignHCenter)
        self.X_input.setPlaceholderText('X (C)')
        self.X_input.setValidator(validator)
        self.X_input.setToolTip('X or Column value')
        self.X_input.setMaximumWidth(50)

        self.Y_input = StyledLineEdit()
        self.Y_input.setAlignment(Qt.AlignHCenter)
        self.Y_input.setPlaceholderText('Y (R)')
        self.Y_input.setValidator(validator)
        self.Y_input.setToolTip('Y or Row value')
        self.Y_input.setMaximumWidth(50)

    # Go to pixel button
        self.go_btn = IconButton('Icons/bullseye.png')
        self.go_btn.setFlat(True)
        self.go_btn.setToolTip('Zoom to pixel')
        self.go_btn.clicked.connect(self.zoomToPixel)

    # Adjust widget layout
        mainLayout = QW.QHBoxLayout()
        mainLayout.setContentsMargins(5, 2, 5, 2) # l, t, r, b
        mainLayout.addWidget(self.go_btn, alignment=Qt.AlignHCenter)
        mainLayout.addWidget(self.X_input, alignment=Qt.AlignHCenter)
        mainLayout.addWidget(self.Y_input, alignment=Qt.AlignHCenter)
        self.setLayout(mainLayout)


    def zoomToPixel(self):
        x, y = None, None
        if self.X_input.hasAcceptableInput():
            x = int(self.X_input.text())
        if self.Y_input.hasAcceptableInput():
            y = int(self.Y_input.text())

        if x is not None and y is not None:
            self.canvas.zoom_to(x, y)

    def set_InputCellsMaxWidth(self, width):
        self.X_input.setMaximumWidth(width)
        self.Y_input.setMaximumWidth(width)






class DocumentBrowser(QW.QWidget):

    def __init__(self, readOnly=False, parent=None):
        super(DocumentBrowser, self).__init__(parent)
    # Set main attributes
        self.readOnly = readOnly
        self.font = QFont()
        self.placeHolderText = ''
    # Set GUI
        self._init_ui()

    def _init_ui(self):
    # Text browser space
        self.browser = QW.QTextEdit()
        self.browser.setStyleSheet(pref.SS_menu)
        self.browser.setReadOnly(self.readOnly)

    # Browser toolbar
        self.toolbar = QW.QToolBar(self.browser)
        self.toolbar.setStyleSheet(pref.SS_toolbar)

    # Edit document checkbox
        self.edit_cbox = QW.QCheckBox('Enable editing')
        self.edit_cbox.setChecked(not self.readOnly)
        self.edit_cbox.stateChanged.connect(
            lambda state: self.browser.setReadOnly(not state))
        self.toolbar.addWidget(self.edit_cbox)

    # Search box
        self.searchBox = StyledLineEdit()
        self.searchBox.setPlaceholderText('Search')
        self.searchBox.setClearButtonEnabled(True)
        self.searchBox.setMaximumWidth(100)
        self.searchBox.editingFinished.connect(self._findTextDown)
        self.toolbar.addWidget(self.searchBox)

    # Search Up Action
        self.searchUpAction = QW.QAction(QIcon('Icons/arrow_up.png'),
                                         'Search Up', self.toolbar)
        self.searchUpAction.triggered.connect(self._findTextUp)

    # Search Down Action
        self.searchDownAction = QW.QAction(QIcon('Icons/arrow_down.png'),
                                         'Search Down', self.toolbar)
        self.searchDownAction.triggered.connect(self._findTextDown)

    # Zoom in Action
        self.zoomInAction = QW.QAction(QIcon('Icons/zoom_in.png'),
                                       'Zoom in', self.toolbar)
        self.zoomInAction.triggered.connect(lambda: self._alterZoom(+1))

    # Zoom out Action
        self.zoomOutAction = QW.QAction(QIcon('Icons/zoom_out.png'),
                                        'Zoom out', self.toolbar)
        self.zoomOutAction.triggered.connect(lambda: self._alterZoom(-1))

    # Add Actions to toolbar
        self.toolbar.addActions((self.searchUpAction, self.searchDownAction,
                                 self.zoomInAction, self.zoomOutAction))
        self.toolbar.insertSeparator(self.zoomInAction)

    # Adjust Main Layout
        mainLayout = QW.QVBoxLayout()
        mainLayout.addWidget(self.toolbar)
        mainLayout.addWidget(self.browser)
        self.setLayout(mainLayout)


    def setDoc(self, doc_path):
        if exists(doc_path):
            with open(doc_path, 'r') as log:
                doc = QTextDocument(log.read(), self.browser)
                self.browser.setDocument(doc)
        else:
            self.browser.clear()
            self.browser.setPlaceholderText(self.placeHolderText)

    def setDefaultPlaceHolderText(self, text):
        self.placeHolderText = text


    def _alterZoom(self, value):
        newSize = self.font.pointSize() + value
        if 0 < newSize < 80:
            self.font.setPointSize(newSize)
            self.browser.document().setDefaultFont(self.font)

    def _findTextUp(self):
        self.browser.setFocus()
        self.browser.find(self.searchBox.text(), QTextDocument.FindBackward)

    def _findTextDown(self):
        self.browser.setFocus()
        self.browser.find(self.searchBox.text())


class RoiPatch(Rectangle):
    def __init__(self, bbox, color, filled):
        x0, y0, w, h = bbox
        lw = 2

        super(RoiPatch, self).__init__((x0, y0), w, h, linewidth=lw,
                                       color=color, fill=filled)

class RoiAnnotation(mpl_text.Annotation):
    def __init__(self, text, anchor_patch, xy=(0, 1)):
        bbox = dict(boxstyle='round', fc=pref.IVORY, ec=pref.BLACK_PEARL)

        super(RoiAnnotation, self).__init__(text, xy, xycoords=anchor_patch,
                                            bbox=bbox, annotation_clip=True)





