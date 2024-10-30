# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:27:17 2021

@author: albdag
"""

from ast import literal_eval
import os
from typing import Iterable, Callable
from weakref import proxy

from PyQt5 import QtWidgets as QW
from PyQt5 import QtGui as QG
from PyQt5 import QtCore as QC

import numpy as np

from _base import InputMap, MineralMap, Mask
import convenient_functions as cf
import image_analysis_tools as iatools
import preferences as pref
import style


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

        '''
        super(DataGroup, self).__init__()

    # Set the flags. Data groups are selectable and their name is editable
        self.setFlags(QC.Qt.ItemIsSelectable | QC.Qt.ItemIsUserCheckable |
                      QC.Qt.ItemIsEnabled | QC.Qt.ItemIsEditable)

    # Set the font as bold
        font = self.font(0)
        font.setBold(True)
        self.setFont(0, font)
        self.setText(0, name)

    # Add the subgroups
        self.inmaps = DataSubGroup('Input Maps')
        self.inmaps.setIcon(0, QG.QIcon(r'Icons/inmap.png'))
        self.minmaps = DataSubGroup('Mineral Maps')
        self.minmaps.setIcon(0, QG.QIcon(r'Icons/minmap.png'))
        self.masks = DataSubGroup('Masks')
        self.masks.setIcon(0, QG.QIcon(r'Icons/mask.png'))
        # add self.points = DataSubGroup('Point Analysis')
        self.subgroups = (self.inmaps, self.minmaps, self.masks)
        self.addChildren(self.subgroups)


    def setShapeWarnings(self):
        '''
        Set a warning state to every loaded data object whose shape differs
        from the sample overall trending shape.

        '''
    # Collect in a single list the objects from all subgroup
        # include points data (maybe?)
        objects = []
        for subgr in self.subgroups:
            objects.extend(subgr.getChildren())

    # Extract the shape from each object data and obtain the trending shape  
        if len(objects):
            shapes = [o.get('data').shape for o in objects]
            trend_shape = cf.most_frequent(shapes)

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
            comp_mask = Mask(iatools.binary_merge([m.mask for m in masks], mode))

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

        '''
        super(DataSubGroup, self).__init__()

    # Set main attributes
        self.name = name

    # Set the flags. Data subgroups can be selected but cannot be edited
        self.setFlags(QC.Qt.ItemIsSelectable | QC.Qt.ItemIsUserCheckable |
                      QC.Qt.ItemIsEnabled)

    # Set the font as underlined
        font = self.font(0)
        font.setItalic(True)
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

        '''
        self.addChild(DataObject(data))
        self.parent().setShapeWarnings()


    def delChild(self, child):
        '''
        Remove child DataObject from the subgroup.

        Parameters
        ----------
        child : DataObject
            The child to be removed
        '''
        self.takeChild(self.indexOfChild(child))
        self.parent().setShapeWarnings()


    def clear(self):
        '''
        Remove all children from the subgroup.

        '''
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
        data : InputMap, MineralMap, Mask, PointAnalysis [in future]
            The data linked to the data object.

        parent : QTreeWidget, QTreeWidgetItem, None
            This object's parent. The default is None.

        '''
    # Set the data object type to 'User Type' for more customization options
        super(DataObject, self).__init__(type = QW.QTreeWidgetItem.UserType)

    # Set the flags. A data object is selectable and editable by the user.
        self.setFlags(QC.Qt.ItemIsSelectable | QC.Qt.ItemIsUserCheckable |
                      QC.Qt.ItemIsEnabled | QC.Qt.ItemIsEditable)

    # Set the data with custom role. It is a user-role (int in [0, 256]) that
    # does not overwrite any default Qt role. It is set arbitrarily to 100
        self.setData(0, 100, data)
    # Set the display name of data as its filename, using a DisplayRole (= 0)
        self.setData(0, 0, self.generateDisplayName())
    # Set the "edited" state as False, using a user-role (= 101).
        self.setData(0, 101, False)
    # A save icon is shown if item is edited --> DecorationRole (= 1)
        self.setData(0, 1, QG.QIcon())
    # A tooltip indicates if item is edited --> ToolTipRole (= 3)
        self.setData(0, 3, '')
    # Set the "warning" state as False, using a user-role (= 102).
        self.setData(0, 102, False)
    # A warn icon is shown (2nd col) if item has warnings --> DecorationRole (= 1)
        self.setData(1, 1, QG.QIcon())
    # A tooltip (2nd col) indicates if item has warnings --> ToolTipRole (= 3)
        self.setData(1, 3, '')

    # Set the "checked" state for togglable data (Masks and Points [in future])
        if isinstance(data, (Mask,)): # add PointAnalysis class
            self.setData(0, 10, QC.Qt.Unchecked) # CheckedRole (= 10)


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
            name = cf.path2filename(filepath)

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
            Requested object data. It returns the first element of the list if
            its lenght is 1.

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


    def setEdited(self, edited:bool):
        '''
        Toggle on/off the edited state of this object both internally and 
        visually (icon and tooltip).

        Parameters
        ----------
        edited : bool
            Edited state.

        '''
    # Set the 'isEdited' attribute
        self.setData(0, 101, edited)
    # Show/hide the save icon
        icon = QG.QIcon('Icons/edit_white.png') if edited else QG.QIcon()
        self.setData(0, 1, icon)
    # Show/hide the edited tooltip
        text = 'Edits not saved' if edited else ''
        self.setData(0, 3, text)


    def setWarning(self, warning:bool):
        '''
        Toggle on/off the warning state of this object both internally and 
        visually (icon and tooltip).

        Parameters
        ----------
        warning : bool
            Warning state.

        '''
    # Set the 'has_warning' attribute
        self.setData(1, 102, warning)
    # Show/hide the warn icon
        icon = QG.QIcon('Icons/warnIcon.png') if warning else QG.QIcon()
        self.setData(1, 1, icon)
    # Show/hide the warning tooltip
        text = 'Unfitting shapes' if warning else ''
        self.setData(1, 3, text)


class Legend(QW.QTreeWidget):
    '''
    An interactive legend object for displaying and editing the names and the 
    palette colors of mineral phases within a mineral map. Optionally, the 
    mineral modal amounts are displayed as well. It can include a right-click 
    context menu action for advanced interactions. This object sends various 
    signals to notify each edit request, which must be catched and handled by 
    other widgets to be effective.
    '''
    instances = []
    colorChangeRequested = QC.pyqtSignal(QW.QTreeWidgetItem, tuple) # item, col
    randomPaletteRequested = QC.pyqtSignal()
    itemRenameRequested = QC.pyqtSignal(QW.QTreeWidgetItem, str) # item, name
    itemsMergeRequested = QC.pyqtSignal(list, str) # list of classes, name
    itemHighlightRequested = QC.pyqtSignal(bool, QW.QTreeWidgetItem) # on/off, item
    maskExtractionRequested = QC.pyqtSignal(list) # list of classes


    def __init__(self, amounts=True, context_menu=False, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        amounts : bool, optional
            Include classes amounts (percentage) in the legend. The default is
            True.
        context_menu : bool, optional
            Whether a context menu should popup when right-clicking on a legend
            item. The default is False.
        parent : QWidget or None, optional
            GUI parent widget of the legend. The default is None.

        '''
    # Weakly track all class instances
        self.__class__.instances.append(proxy(self))

    # Call the constructor of the parent class
        super(Legend, self).__init__(parent)

    # Define main attributes
        self._highlighted_item = None
        self.amounts = amounts
        self.has_context_menu = context_menu
        self.precision = pref.get_setting('plots/legendDec', 3, type=int)
        
    # Customize the legend appearence and properties (headers & selection mode)
        self.setColumnCount(2 + self.amounts)
        self.setHeaderLabels(['Color', 'Class'] + ['Amount'] * self.amounts)
        self.header().setSectionResizeMode(QW.QHeaderView.Interactive)
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

    # Disable default editing. Item editing is forced via requestClassRename()
        self.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)

    # Set custom scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))

    # Enable custom context menu
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)

    # Set stylesheet (right-click menu when editing items name)
        self.setStyleSheet(style.SS_MENU)

    # Connect signals with slots
        self._connect_slots()


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
        self.itemDoubleClicked.connect(self.onDoubleClick)
        if self.has_context_menu:
            self.customContextMenuRequested.connect(self.showContextMenu)


    def showContextMenu(self, point: QC.QPoint):
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''

    # Get the item that is clicked from <point> and define a menu.
        i = self.itemAt(point)
        if i is None: return
        menu = StyledMenu()

    # Rename class
        menu.addAction(QG.QIcon(r'Icons/rename.png'), 'Rename',
                       lambda: self.requestClassRename(i))

    # Merge classes
        merge = menu.addAction('Merge', self.requestClassMerge)
        merge.setEnabled(len(self.selectedItems()) > 1)

    # Separator
        menu.addSeparator()
 
    # Copy current color HEX string
        menu.addAction(
            'Copy color string', lambda: self.copyColorHexToClipboard(i))

    # Change color
        menu.addAction(QG.QIcon(r'Icons/palette.png'), 'Set color',
                       lambda: self.requestColorChange(i))

    # Randomize color
        menu.addAction(QG.QIcon(r'Icons/randomize_color.png'), 'Random color',
                       lambda: self.requestRandomColorChange(i))

    # Randomize palette
        menu.addAction(QG.QIcon(r'Icons/randomize_color.png'), 'Randomize all',
                       self.randomPaletteRequested.emit)
        
    # Separator
        menu.addSeparator()

    # Higlight item
        highlight = QW.QAction(QG.QIcon(r'Icons/highlight.png'), 'Highlight')
        highlight.setCheckable(True)
        highlight.setChecked(i == self._highlighted_item)
        highlight.toggled.connect(lambda t: self.requestItemHighlight(t, i))

    # Extract mask
        menu.addAction(QG.QIcon(r'Icons/mask.png'), 'Extract mask', 
                       self.requestMaskFromClass)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QG.QCursor.pos())


    def onDoubleClick(self, item: QW.QTreeWidgetItem, column: int):
        '''
        Wrapper function that triggers different actions depending on which 
        column was double-clicked.

        Parameters
        ----------
        item : QTreeWidgetItem
            The legend item that was double-clicked.
        column : int
            The column that was double-clicked.

        '''
        if column == 0:
            self.requestColorChange(item)
        elif column == 1:
            self.requestClassRename(item)
        else:
            self.requestItemHighlight(item != self._highlighted_item, item)

    def copyColorHexToClipboard(self, item: QW.QTreeWidgetItem):
        '''
        Copy the selected phase color to the clipboard as a HEX string.

        Parameters
        ----------
        item : QTreeWidgetItem
            The selected phase item.

        '''
    # Get the hex string of phase color
        rgb_color = literal_eval(item.whatsThis(0)) # tuple parsing from string
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_color)
    # Copy the string to the clipboard
        clipboard = QW.qApp.clipboard()
        clipboard.setText(hex_color)


    def requestColorChange(self, item: QW.QTreeWidgetItem):
        '''
        Request to change the color of a class by sending a signal. The signal
        must be catched and handled by the widget that contains the legend.
        This function is also triggered when double-clicking on the item icon.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.

        '''
    # Get the old color and the new color (as rgb tuple)
        old_col = literal_eval(item.whatsThis(0)) # tuple parsing from str
        new_col = QW.QColorDialog.getColor(initial=QG.QColor(*old_col))
    # Emit the signal
        if new_col.isValid():
            rgb = tuple(new_col.getRgb()[:-1])
            self.colorChangeRequested.emit(item, rgb)


    def requestRandomColorChange(self, item: QW.QTreeWidgetItem):
        '''
        Request to randomize the color of a class by sending a signal. The
        signal must be catched and handled by the widget that contains the
        legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.

        '''
        self.colorChangeRequested.emit(item, ())


    def changeItemColor(self, item: QW.QTreeWidgetItem, color: tuple[int]):
        '''
        Change the color of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item whose color must be changed.
        color : tuple
            RGB color triplet.

        '''
    # Set the new color to the legend item
        item.setIcon(0, RGBIcon(color))
    # Also set the new whatsThis string
        item.setWhatsThis(0, str(color))


    def requestClassRename(self, item: QW.QTreeWidgetItem):
        '''
        Request to change the name of a class by sending a signal. The signal
        must be catched and handled by the widget that contains the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests to be renamed.

        '''
    # Deny renaming protected '_ND'_ class
        old_name = item.text(1)
        if old_name == '_ND_':
            return MsgBox(self, 'Crit', 'This class cannot be renamed.')
    
    # Do nothing if the dialog is canceled or the class is not renamed
        name, ok = QW.QInputDialog.getText(self, 'X-Min Learn',
                                           'Rename class (max. 8 ASCII '\
                                           'characters):', text=f'{old_name}')
        if not ok or name == old_name:
            return
        
    # Deny renaming to protected '_ND_' class
        elif name == '_ND_':
            return MsgBox(self, 'Crit', '"_ND_" is a protected class name.')
        
    # Deny renaming if the name already exists
        elif self.hasClass(name):
            return MsgBox(self, 'Crit', f'{name} is already taken.')

    # Deny renaming if the new name is not an ASCII <= 8 characters string
        elif 0 < len(name) <= 8 and name.isascii():
            self.itemRenameRequested.emit(item, name)
        else:
            return MsgBox(self, 'Crit',  'Invalid name.')


    def renameClass(self, item: QW.QTreeWidgetItem, name: str):
        '''
        Change the name of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item to be renamed.
        name : str
            New item name.

        '''
    # Set the new name to the legend item
        item.setText(1, name)


    def requestClassMerge(self):
        '''
        Request to merge two or more mineral classes.

        '''
    # Do nothing if less than 2 classes are selected
        classes = [i.text(1) for i in self.selectedItems()]
        if len(classes) < 2:
            return
        
    # Deny renaming protected '_ND'_ class
        if '_ND_' in classes:
            return MsgBox(self, 'Crit', '"_ND_" class cannot be merged.')
        
    # Do nothing if the dialog is canceled
        text = f'Merge {classes} in a new class (max. 8 ASCII characters):'
        name, ok = QW.QInputDialog.getText(self, 'X-Min Learn', text)
        if not ok:
            return
        
    # Deny renaming to protected '_ND_' class
        elif name == '_ND_':
            return MsgBox(self, 'Crit', '"_ND_" is a protected class name.')
    
    # Deny renaming if the name already exists (excluding selected classes)
        elif name not in classes and self.hasClass(name):
            return MsgBox(self, 'Crit', f'{name} is already taken.')

    # Deny renaming if the new name is not an ASCII <= 8 characters string
        elif 0 < len(name) <= 8 and name.isascii():
            self.itemsMergeRequested.emit(classes, name)
        else:
            return MsgBox(self, 'Crit', 'Invalid name.')


    def requestItemHighlight(self, toggled: bool, item: QW.QTreeWidgetItem):
        '''
        Request to highlight the selected mineral class. Highlight means to
        show ONLY the selected class in map.

        Parameters
        ----------
        toggled : bool
            Highlight on / off
        item : QTreeWidgetItem
            The item to be highlighted

        '''
    # Change the current highlighted item attribute
        self._highlighted_item = item if toggled else None

    # Send the request signal
        self.itemHighlightRequested.emit(toggled, item)


    def requestMaskFromClass(self):
        '''
        Request to extract a mask from the selected mineral classes.

        '''
        classes = [i.text(1) for i in self.selectedItems()]
        self.maskExtractionRequested.emit(classes)


    def setPrecision(self, value: int):
        '''
        Set the number of decimals of the class amounts.

        Parameters
        ----------
        value : int
            Number of decimals to be shown in the legend.

        '''
        self.precision = value
        # self.update()


    def addClass(self, name: str, color: tuple[int], amount: float):
        '''
        Add a new mineral class to the legend.

        Parameters
        ----------
        name : str
            Class name.
        color : tuple[int]
            Class color as RGB triplet. 
        amount : float
            Class modal amount. This value is ignored if legend amounts are not
            enabled.

        '''
        item = QW.QTreeWidgetItem(self)
        item.setIcon(0, RGBIcon(color))    # icon [column 0]
        item.setWhatsThis(0, str(color))   # RGB string ['virtual' column 0]
        item.setText(1, name)              # class name [column 1]
        if self.amounts:                   # amounts (optional) [column 2]
            amount = round(amount, self.precision)
            item.setText(2, f'{amount}%')


    def hasClass(self, name: str):
        '''
        Check if the given class is already displayed in the legend. The search
        is done through the class name.

        Parameters
        ----------
        name : str
            Class name to check.

        Returns
        -------
        bool
            Whether the legend already contains a class with the given name.

        '''
        n_classes = self.topLevelItemCount()
        class_names = [self.topLevelItem(i).text(1) for i in range(n_classes)]
        return name in class_names


    def update(self, mineral_map: MineralMap):
        '''
        Updates the legend.

        Parameters
        ----------
        mineral_map : MineralMap
            The Mineral Map object linked to the current view of the legend.

        '''
    # Clear the legend
        self.clear()

    # Reset the highlighted item reference
        self._highlighted_item = None

    # Populate the legend with mineral classes
        for phase in mineral_map.get_phases():
            color = mineral_map.get_phase_color(phase)
            amount = mineral_map.get_phase_amount(phase)
            self.addClass(phase, color, amount)

    # Resize columns
        self.header().resizeSections(QW.QHeaderView.ResizeToContents)







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






# class CurvePlotCanvas(FigureCanvasQTAgg):

#     def __init__(self, parent=None, size=(6.4, 4.8), tight=False,
#                  title='', xlab='', ylab='', grid=True):
#         self.fig = Figure(figsize=size, tight_layout=tight)
#         self.fig.patch.set(facecolor='w', edgecolor='#19232D', linewidth=2)
#         self.ax = self.fig.add_subplot(111)
#         super(CurvePlotCanvas, self).__init__(self.fig)

#         self.title = title
#         self.xlab = xlab
#         self.ylab = ylab
#         self.gridOn = grid

#         self.init_ax()

#     def init_ax(self):
#         self.ax.set_title(self.title)
#         self.ax.set_xlabel(self.xlab)
#         self.ax.set_ylabel(self.ylab)
#         self.ax.grid(self.gridOn)

#     def add_curve(self, xdata, ydata, label='', color='k'):
#         self.ax.plot(xdata, ydata, label = label, color=color)

#     def has_curves(self):
#         return bool(len(self.ax.lines))

#     def update_canvas(self, data=None):
#     # data has to be a list of tuple, each one containing xdata and ydata of a single curve
#     # example: data = [(x1, y1), (x2, y2)]
#         if data is not None:
#             curves = self.ax.lines # get all the plot instances (Line2D)
#             assert len(data) == len(curves)
#             for n in range(len(curves)):
#                 curves[n].set_data(data[n])
#             self.ax.relim()
#             self.ax.autoscale_view()

#         self.ax.legend(loc='best')
#         self.draw()
#         self.flush_events()

#     def homeAction(self):
#         self.ax.relim()
#         self.ax.autoscale()
#         self.draw()
#         self.flush_events()

#     def clear_canvas(self):
#         self.ax.cla()
#         self.init_ax()
#         self.draw()
#         self.flush_events()



class RGBIcon(QG.QIcon):
    '''
    Convenient class to generate a colored icon useful in legends.
    '''
    def __init__(self, rgb:tuple, size=(64, 64), edgecolor=style.IVORY, lw=1):
        '''
        Constructor.

        Parameters
        ----------
        rgb : tuple
            RGB triplet.
        size : tuple, optional
            Icon size. The default is (64, 64).
        edgecolor : str, optional
            Color of the icon border, expressed as a HEX string. The default is
            #F9F9F4 (IVORY).
        lw : int, optional
            Border line width. The default is 1.

        '''
    # Set main attributes
        self.rgb = rgb
        self.height, self.width = size

    # Create a pixmap
        pixmap = QG.QPixmap(self.height, self.width)
        pixmap.fill(QG.QColor(*self.rgb))

    # Add border
        painter = QG.QPainter(pixmap)
        pen = QG.QPen(QG.QColor(edgecolor))
        pen.setWidth(lw)
        painter.setPen(pen)
        # -1 to make the border inside the pixmap
        painter.drawRect(0, 0, self.width - 1, self.height - 1)  
        painter.end()

    # Create the icon using the pixmap
        super(RGBIcon, self).__init__(pixmap)



class StyledButton(QW.QPushButton):
    '''
    QSS-styled reimplementation of a QPushButton.
    '''
    def __init__(self, icon=None, text=None, bg_color=None, text_padding=1):
        '''
        Constructor.

        Parameters
        ----------
        icon : QIcon | None, optional
            Button icon. The default is None.
        text : str | None, optional
            Button text. The default is None.
        bg_color : str | None, optional
            Background color of the button. If None the default color is used.
            The default is None.
        text_padding : int, optional
            Adds some space between the icon and the text. The default is 1.

        '''
        super(StyledButton, self).__init__()
        self.setSizePolicy(QW.QSizePolicy.Preferred, QW.QSizePolicy.Fixed)

    # Set icon
        if icon:
            self.setIcon(icon)
            if text: # add spacing between icon and text
                text = text.rjust(text_padding + len(text))
    # Set text
        if text:
            self.setText(text)

    # Overwrite default background color if requested
        ss = style.SS_BUTTON
        if bg_color:
            ss+= 'QPushButton {background-color: %s; font: bold;}' %(bg_color)
        self.setStyleSheet(ss)



class StyledComboBox(QW.QComboBox):
    '''
    QSS-styled reimplementation of a QComboBox, with auto-generating tooltips.
    '''
    def __init__(self, tooltip=None, parent=None):
        '''
        Constructor

        Parameters
        ----------
        tooltip : str | None, optional
            Fixed combo box tooltip. If None, the tooltip automatically changes
            when the current text changes. The default is None.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledComboBox, self).__init__(parent)

    # Set automatic tooltip update if no tooltip is provided
        if tooltip:
            self.setToolTip(tooltip)
        else:
            self.setToolTip(self.currentText())
            self.currentTextChanged.connect(self.setToolTip)

    # Set stylesheet (SS_menu in case of editable combo box)
        self.setStyleSheet(style.SS_COMBOX + style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent):
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the combobox. The event will be accepted only if SHIFT is
        pressed during the scroll.

        Parameters
        ----------
        event : QG.QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super(StyledComboBox, self).wheelEvent(event)
        else:
            event.ignore()



class StyledDoubleSpinBox(QW.QDoubleSpinBox):
    '''
    QSS-styled reimplementation of a QDoubleSpinBox.
    '''
    def __init__(self, min_value=0.0, max_value=1.0, step=0.1, decimals=2, 
                 parent=None):
        '''
        Constructor.

        Parameters
        ----------
        min_value : flaot, optional
            Minimum range value. The default is 0.0.
        max_value : float, optional
            Maximum range value. The default is 1.0.
        step : float, optional
            Step value. The default is 0.1.
        decimals : int, optional
            Number of displayed decimals. The default is 2.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledDoubleSpinBox, self).__init__(parent)

    # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

    # Set number of decimal positions
        self.setDecimals(decimals)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent):
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the spinbox. The event will be accepted only if SHIFT is
        pressed during the scroll. N.B. By default, if CTRL is also pressed the
        step is multiplied by 10.

        Parameters
        ----------
        event : QG.QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super(StyledDoubleSpinBox, self).wheelEvent(event)
        else:
            event.ignore()



class StyledListWidget(QW.QListWidget):
    '''
    QSS-styled reimplementation of a QListWidget.
    '''

    def __init__(self, ext_selection=True, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        ext_selection : bool, optional
            If extended selection mode should be enabled. The default is True.
        parent : QObject or None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledListWidget, self).__init__(parent)
    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))
    # Set extended selection mode if requested
        if ext_selection:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)


    def getItems(self):
        '''
        Return all items in the list.

        Returns
        -------
        items : list
            List of items.

        '''
        items = [self.item(row) for row in range(self.count())]
        return items
    

    def selectedRows(self):
        '''
        Return all selected rows.

        Returns
        -------
        selected_rows : list[int]
            List of selected items expressed as rows.

        '''
        selected_items = self.selectedItems()
        selected_rows = [self.row(item) for item in selected_items]
        return selected_rows
    

    def removeSelected(self):
        '''
        Remove all selected items from list.

        '''
        for idx in sorted(self.selectedRows(), reverse=True):
            self.takeItem(idx)



class StyledMenu(QW.QMenu):
    '''
    QSS-styled reimplementation of a QMenu.
    '''
    def __init__(self, parent=None):
        '''
        Constructor

        Parameters
        ----------
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledMenu, self).__init__(parent)
        self.setStyleSheet(style.SS_MENU)



class StyledRadioButton(QW.QRadioButton):
    '''
    QSS-styled reimplementation of a QRadioButton.
    '''
    def __init__(self, text='', icon: QG.QIcon|None=None, parent=None):
        '''
        Constructor

        Parameters
        ----------
        text : str
            The text of the radio button. The default is ''.
        icon : QIcon or None, optional
            The icon of the radio button. The default is None.
        parent : QObject or None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledRadioButton, self).__init__(text, parent)
        if icon is not None:
            self.setIcon(icon)
        self.setStyleSheet(style.SS_RADIOBUTTON)



class StyledScrollBar(QW.QScrollBar):
    '''
    QSS-styled reimplementation of a QScrollBar.
    '''
    def __init__(self, orientation: QC.Qt.Orientation):
        '''
        Constructor.

        Parameters
        ----------
        orientation : Qt.Orientation
            The orientation of the scroll bar.

        '''
        super(StyledScrollBar, self).__init__(orientation)

    # Set the stylesheet
        if orientation == QC.Qt.Horizontal:
            self.setStyleSheet(style.SS_SCROLLBARH)
        else:
            self.setStyleSheet(style.SS_SCROLLBARV)



class StyledSpinBox(QW.QSpinBox):
    '''
    QSS-styled reimplementation of a QSpinBox.
    '''
    def __init__(self, min_value=0, max_value=100, step=1, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        min_value : int, optional
            Minimum range value. The default is 0.
        max_value : int, optional
            Maximum range value. The default is 100.
        step : int, optional
            Step value. The default is 1.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledSpinBox, self).__init__(parent)

    # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent):
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the spinbox. The event will be accepted only if SHIFT is
        pressed during the scroll. N.B. By default, if CTRL is also pressed the
        step is multiplied by 10.

        Parameters
        ----------
        event : QG.QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super(StyledSpinBox, self).wheelEvent(event)
        else:
            event.ignore()



class StyledTable(QW.QTableWidget):
    '''
    QSS-styled reimplementation of a QTableWidget.
    '''
    def __init__(self, rows:int, cols:int, ext_selection=True, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        ext_selection : bool, optional
            If extended selection mode should be enabled. The default is True.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledTable, self).__init__(rows, cols, parent)

    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))

    # Set extended selection mode if requested
        if ext_selection:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

    # Customize corner button behaviour
        self.corner_btn = self.findChild(QW.QAbstractButton, '')
        if self.corner_btn:
            self.corner_btn.setToolTip('Select all / deselect all')
            self.corner_btn.disconnect()
            self.corner_btn.clicked.connect(self.onCornerButtonClicked)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)

    
    def onCornerButtonClicked(self):
        '''
        Select all items when not all items are currently selected. Otherwise
        deselect them all.

        '''
        tot_items = self.columnCount() + self.rowCount()
        if len(self.selectedIndexes()) < tot_items:
            self.selectAll()
        else:
            self.clearSelection()


class StyledTabWidget(QW.QTabWidget):
    '''
    QSS-styled reimplementation of a QTabWidget. Includes a reimplementation of
    the addTab function to always have a QWidget container for the tab. 
    '''
    def __init__(self, wheel_scroll=False, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        wheel_scroll : bool, optional
            Whether mouse wheel event should allow to change current tab. The 
            default is False.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(StyledTabWidget, self).__init__(parent)

        if not wheel_scroll: 
            self.setTabBar(self.UnscrollableTabBar())

    # Set stylesheet
        self.setStyleSheet(style.SS_TABWIDGET)

    def addTab(self, qobject: QC.QObject, icon: QG.QIcon|None=None, 
               title: str|None=None):
        '''
        Reimplementation of the addTab function, useful to achive consistent 
        look of the tabwidget, no matter the type of widget is set as its tab.

        Parameters
        ----------
        qobject : QObject
            Layout or widget to be added as tab. 
        icon : QG.QIcon | None, optional
            Tab icon. The default is None.
        title : str | None, optional
            Tab name. The default is None.

        '''
        if isinstance(qobject, QW.QLayout):
            tab = QW.QWidget()
            tab.setLayout(qobject)

        elif layout := qobject.layout():
            tab = qobject
        
        else:
            tab = QW.QWidget()
            layout = QW.QVBoxLayout(tab)
            layout.addWidget(qobject)

        if icon:
            super(StyledTabWidget, self).addTab(tab, icon, title)
        else:
            super(StyledTabWidget, self).addTab(tab, title)


    class UnscrollableTabBar(QW.QTabBar):
        def __init__(self, parent=None):
            super().__init__(parent)


        def wheelEvent(self, event: QG.QWheelEvent):
            '''
            Reimplements the wheelEvent to ignore it.

            Parameters
            ----------
            event : QG.QWheelEvent
                The wheel mouse event.

            '''
            event.ignore()


class StyledToolbar(QW.QToolBar):
    '''
    QSS-styled reimplementation of a QToolBar.
    '''
    def __init__(self, title='', parent=None):
        '''
        Constructor.

        Parameters
        ----------
        title : str, optional
            The name of the toolbar. The default is ''.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.
            
        '''
        super(StyledToolbar, self).__init__(title, parent)
    
    # Set icon size
        size = pref.get_setting('plots/NTBsize', 28, type=int)
        self.setIconSize(QC.QSize(size, size))

    # Set stylesheet
        self.setStyleSheet(style.SS_TOOLBAR)



class AutoUpdateComboBox(StyledComboBox):
    '''
    A reimplementation of a Styled ComboBox with a new QtSignal which triggers 
    when the combo box button is clicked. This signal can be connected to a
    custom method that allows to update the list of items in the combo box.
    '''
    clicked = QC.pyqtSignal()

    def __init__(self, parent=None, **kwargs):
        '''
        Constructor.

        Parameters
        ----------
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.
        **kwargs
            Parent class arguments (see StyledComboBox class).

        '''
        super(AutoUpdateComboBox, self).__init__(parent=parent, **kwargs)

    def showPopup(self):
        '''
        Reimplementation of the showPopup method. It just emits the new clicked
        signal.

        '''
        self.clicked.emit()
        super(AutoUpdateComboBox, self).showPopup()

    def updateItems(self, items: list):
        '''
        Populate the combo box with new items and delete the previous ones.

        Parameters
        ----------
        items : list
            List of new items (strings).
        '''
        self.clear()
        self.addItems(items)



class CBoxMapLayout(QW.QGridLayout): # deprecated! Use SampleMapsSelector instead
    cboxPressed = QC.pyqtSignal()

    def __init__(self, paths, parent=None):
        super(CBoxMapLayout, self).__init__()
        self.parent = parent

        self.setColumnStretch(0, 1)
        self.Cbox_list = []


        for idx, pth in enumerate(paths):
            cbox = QW.QCheckBox(cf.path2filename(pth))
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
                                           flags=QC.Qt.MSWindowsFixedSizeDialogHint)
        if ok and name != '':
            idx = self.sender().objectName()
            self.Cbox_list[int(idx)].setText(name)



class RadioBtnLayout(QW.QBoxLayout):
    '''
    Convenient class to group multiple radio buttons in a single layout. It 
    makes use of the QButtonGroup class to allow further useful methods to 
    easily access each button.
    '''
    selectionChanged = QC.pyqtSignal(int) # id of button

    def __init__(self, names: list[str], icons: list[QG.QIcon]|None=None, 
                 default=0, orient='vertical', parent=None):
        '''
        Constructor.

        Parameters
        ----------
        names : list
            List of radio button names.
        icons : list or None, optional
            List of button icons. Must be the same length of names. The default
            is None.
        default : int, optional
            The radio button selected by default. The default is 0.
        orient : str, optional
            Orientation of the layout. Can be 'horizontal' or 'vertical'. The
            default is 'vertical'.
        parent : QObject or None, optional
            The GUI parent of this object. The default is None.

        Raises
        ------
        NameError
            Orientation string must be one of ['horizontal', 'vertical'].
        ValueError
            If provided, icons list must have the same length of names list.

        '''
    # Set the current selected button id
        self._selected_id = default

    # Set the orientation of the layout
        if orient == 'horizontal':
            direction = QW.QBoxLayout.LeftToRight
        elif orient == 'vertical':
            direction = QW.QBoxLayout.TopToBottom
        else:
            raise NameError(f'{orient} is not a valid orientation.')
        
    # Check for same length of names and icons (if provided) lists
        if icons is not None and len(names) != len(icons):
            raise ValueError('Icons and names list have different lengths.')
        
    # Set a QButtonGroup working behind the scene
        super(RadioBtnLayout, self).__init__(direction, parent)
        self.btn_group = QW.QButtonGroup()

    # Populate the layout with styled radio buttons. Connect each radio button
    # clicked signal with a custom method that sends a new signal only when the 
    # selection has changed
        for i, n in enumerate(names):
            icon = None if icons is None else icons[i]
            btn = StyledRadioButton(n, icon)
            self.btn_group.addButton(btn, id=i)
            if i == default: 
                btn.setChecked(True)
            btn.clicked.connect(lambda _, idx=i: self.onSelect(idx))
            self.addWidget(btn)


    def onSelect(self, id_:int):
        '''
        Send the selectionChanged signal if a new button has been selected.

        Parameters
        ----------
        id_ : int
            The selected radio button id.

        '''
        if id_ != self._selected_id:
            self._selected_id = id_
            self.selectionChanged.emit(id_)


    def button(self, id_:int):
        '''
        Return the radio button with given id.

        Parameters
        ----------
        id : int
            The radio button id.

        Returns
        -------
        btn : StyledRadioButton
            The radio button object.

        '''
        btn = self.btn_group.button(id_)
        return btn

    def buttons(self):
        '''
        Return all radio buttons.

        Returns
        -------
        btns: list
            List of radio buttons.

        '''
        btns = self.btn_group.buttons()
        return btns

    def getChecked(self, as_id=False):
        '''
        Return the checked radio button.

        Parameters
        ----------
        as_id : bool, optional
            Return checked radio button id instead of object. The default is 
            False.

        Returns
        -------
        StyledRadioButton | int
            The checked radio button object or id.

        '''
        if as_id: 
            return self.btn_group.checkedId()
        else:    
            return self.btn_group.checkedButton()



class IconButton(StyledButton): # !!! deprecated, use StyledButton instead

    def __init__(self, iconPath, text=None):
        super(IconButton, self).__init__(QG.QIcon(iconPath), text)



class GroupArea(QW.QGroupBox):
    '''
    A convenient class to easily wrap a layout or a widget in a QGroupBox.
    '''
    def __init__(self, qobject, title=None, checkable=False, tight=False,
                 frame=True, align=QC.Qt.AlignHCenter, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        qobject : QObject
            A layout-like or a widget-like object.
        title : str | None, optional
            The title of the wrapping group box. The default is None.
        checkable : bool, optional
            Whether the wrapping group box is checkable. The default is False.
        tight : bool, optional
            Whether the layout of the group box should take all the available 
            space. The default is False.
        frame : bool, optional
            Whether the area should show a visible frame. This is ignored if
            the title is provided. The default is True.
        align : Qt.Alignment, optional
            The alignment of the title. Can be Qt.AlignLeft, Qt.AlignRight or
            Qt.AlignHCenter. The default is Qt.AlignHCenter.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
    # Set the title of the group box. Change the style-sheet depending on title
    # orientation.
        if title:
            super(GroupArea, self).__init__(title, parent)
            align_css = {QC.Qt.AlignLeft: 'top left',
                         QC.Qt.AlignRight: 'top right',
                         QC.Qt.AlignHCenter: 'top center'}
            title_align_ss = ('QGroupBox::title {subcontrol-position: %s;}' 
                              %(align_css[align]))
            self.setStyleSheet(style.SS_GROUPAREA_TITLE + title_align_ss)
        else:
            super(GroupArea, self).__init__(parent)
            ss = style.SS_GROUPAREA_NOTITLE
            if not frame:
                ss = ss + 'QGroupBox {border-width: 0px;}' 
            self.setStyleSheet(ss)

    # Set if the group box is checkable or not
        self.setCheckable(checkable)

    # If the qobject is not a layout, wrap it into a layout box
        if isinstance(qobject, QW.QLayout):
            LayoutBox = qobject
        else:
            LayoutBox = QW.QBoxLayout(QW.QBoxLayout.TopToBottom)
            LayoutBox.addWidget(qobject)

    # Set a tight layout if required
        if tight:
            LayoutBox.setContentsMargins(0, 0, 0, 0)
    
    # Wrap the qobject in the group box
        self.setLayout(LayoutBox)



class CollapsibleArea(QW.QWidget):
    '''
    A convenient solution to expand/collapse an entire section of the GUI. It
    includes an arrow button to switch from expanded to collapsed view.
    '''
    def __init__(self, qobject: QC.QObject, title: str, collapsed=True, 
                 parent=None):
        '''
        Constructor.

        Parameters
        ----------
        qobject : QObject
            A layout-like or a widget-like object.
        title : str
            The title of the section.
        collapsed : bool, optional
            Whether the section should be collapsed by default. The default is 
            True.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(CollapsibleArea, self).__init__(parent)

    # Set private attributes
        self._collapsed = collapsed
        self._section = qobject
        self._title = title

    # Initialize the UI and connect signals to slots
        self._init_ui()
        self._connect_slots()

    # Set collapsed view if required
        if collapsed:
            self.area.setMaximumHeight(0)
            self.arrow.setArrowType(QC.Qt.RightArrow)


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Arrow button (QToolButton)
        self.arrow = QW.QToolButton()
        self.arrow.setStyleSheet(style.SS_TOOLBUTTON)
        self.arrow.setArrowType(QC.Qt.DownArrow)
        
    # Section title (QLabel)
        self.title = QW.QLabel(self._title)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)

    # Section area (GroupArea)
        self.area = GroupArea(self._section)

    # Expand/Collapse animation (QPropertyAnimation)
        self.animation = QC.QPropertyAnimation(self.area, b'maximumHeight')
        self.animation.setDuration(90)

    # Set main layout
        layout = QW.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.arrow, 0, 0)
        layout.addWidget(self.title, 0, 1)
        layout.addWidget(LineSeparator(lw=2), 0, 2)
        layout.addWidget(self.area, 1, 0, 1, -1)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Expand/collapse section when arrow button is clicked
        self.arrow.clicked.connect(self.onArrowClicked) 


    def collapsed(self):
        '''
        Check if the section is currently collapsed or not.

        Returns
        -------
        bool
            Whether the section is collapsed.

        '''
        return self._collapsed
    

    def onArrowClicked(self):
        '''
        Slot connected to the button clicked signal from the arrow button. It 
        determines which action should be performed based on the current state
        of the section.

        '''
        self.expand() if self.collapsed() else self.collapse()


    def collapse(self):
        '''
        Collapse the section and change the arrow type.

        '''
        self._collapsed = True
        self.arrow.setArrowType(QC.Qt.RightArrow)
        self.animation.setStartValue(self.area.height())
        self.animation.setEndValue(0)
        self.animation.start()

    
    def expand(self):
        '''
        Expand the section and change the arrow type.

        '''
        self._collapsed = False
        self.arrow.setArrowType(QC.Qt.DownArrow)
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.area.sizeHint().height())
        self.animation.start()



class GroupScrollArea(QW.QScrollArea):
    '''
    A convenient class to easily wrap a layout or a widget in a QScrollArea.
    '''
    def __init__(self, qobject, title=None, hscroll=True, vscroll=True, 
                 tight=False, frame=True, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        qobject : QObject
            A layout-like or a widget-like object.
        title : str | None, optional
            If set, the qobject will be also wrapped in a GroupArea that shows
            this title. However, this paremeter is ignored if the qobject is 
            not a layout-like object. The default is None.
        hscroll : bool, optional
            Whether the horizontal scrollbar is shown. The default is True.
        vscroll : bool, optional
            Whether the vertical scrollbar is shown. The default is True.
        tight : bool, optional
            Whether the layout of the scroll area should take all the available 
            space. It is ignored if qObject is a widget. The default is False.
        frame : bool, optional
            Whether the scroll area should have a visible frame. The default is
            True.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(GroupScrollArea, self).__init__(parent)

    # Set the stylesheet
        if frame:
            self.setStyleSheet(style.SS_GROUPSCROLLAREA_FRAME)
        else:
            self.setStyleSheet(style.SS_GROUPSCROLLAREA_NOFRAME)

    # Set the scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))
        if not hscroll:
            self.setHorizontalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)
        if not vscroll:
            self.setVerticalScrollBarPolicy(QC.Qt.ScrollBarAlwaysOff)

    # If the qobject is a layout, wrap it into a QWidget or a GroupArea 
        if isinstance(qobject, QW.QLayout):
            if title is None:
                if tight: qobject.setContentsMargins(0, 0, 0, 0)
                wid = QW.QWidget()
                wid.setLayout(qobject)
            else:
                wid = GroupArea(qobject, title, tight=tight)
        else:
            wid = qobject

    # Wrap the qobject in the scroll bar.
        self.setWidget(wid)
        self.setWidgetResizable(True)



class SplitterLayout(QW.QBoxLayout):
    '''
    A ready to use layout box that comes with pre-coded splitters dividing each
    inserted wdget.
    '''
    def __init__(self, orient=QC.Qt.Horizontal, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        orient : Qt.Orientation, optional
            The orientation of the latout. The default is Qt.Horizontal.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
    # Set the orientation of the layout box
        if orient == QC.Qt.Horizontal:
            direction = QW.QBoxLayout.LeftToRight
        else:
            direction = QW.QBoxLayout.TopToBottom

    # Initialize the layout box and add a splitter to it. Here we are using the
    # super function addWidget to add the splitter, because such function has
    # been reimplemented (see below methods) and calling it would determine an
    # infinite loop where the app tries to insert the splitter inside itself.
        super(SplitterLayout, self).__init__(direction, parent)
        self.splitter = QW.QSplitter(orient)
        self.splitter.setStyleSheet(style.SS_SPLITTER)
        super(SplitterLayout, self).addWidget(self.splitter)


    def insertLayout(self, layout: QW.QLayout, index: int, stretch=0):
        '''
        Insert a layout at a given index. If index is invalid, the layout is
        inserted at the end.

        Parameters
        ----------
        layout : QLayout
            Layout to be inserted.
        index : int
            Index of insertion. 
        stretch : int, optional
            Optional stretch for the inserted layout. The default is 0.

        '''
        wid = QW.QWidget()
        wid.setLayout(layout)
        self.insertWidget(wid, index, stretch)


    def addLayout(self, layout: QW.QLayout, stretch=0):
        '''
        Add a layout after all the other items.

        Parameters
        ----------
        layout : QLayout
            Layout to be added.
        stretch : int, optional
            Optional stretch for the added layout. The default is 0.

        '''
        self.insertLayout(layout, -1, stretch)


    def addLayouts(self, layouts: list[QW.QLayout], 
                   stretches: list[int]|None=None):
        '''
        Add multiple layouts after all the other items. 

        Parameters
        ----------
        layouts : list[QLayout]
            List of layouts to be added.
        stretches : list[int] or None, optional
            List of stretches for each layout. Must have the same size of 
            layouts. If None, stretches are automatically set to 0 for each 
            layout. The default is None.

        Raises
        ------
        AssertionError
            Layouts and stretches must have the same size.

        '''
        if stretches is None:
            stretches = [0] * len(layouts)
        else:
            assert len(layouts) == len(stretches)

        for l, s in zip(layouts, stretches):
            self.addLayout(l, s)


    def insertWidget(self, widget: QW.QWidget, index: int, stretch=0):
        '''
        Insert a widget at a given index. If index is invalid, the widget is
        inserted at the end.

        Parameters
        ----------
        widget : QWidget
            Widget to be inserted.
        index : int
            Index of insertion.
        stretch : int, optional
            Optional stretch for the inserted widget. The default is 0.

        '''
        self.splitter.insertWidget(index, widget)
        self.splitter.setStretchFactor(index, stretch)


    def addWidget(self, widget: QW.QWidget, stretch=0):
        '''
        Add a widget after all the other items.

        Parameters
        ----------
        widget : QWidget
            Widget to be added.
        stretch : int, optional
            Optional stretch for the added widget. The default is 0.

        '''
        self.insertWidget(widget, -1, stretch)


    def addWidgets(self, widgets: list[QW.QWidget], 
                   stretches: list[int]|None=None):
        '''
        Add multiple widgets after all the other items. 

        Parameters
        ----------
        widgets : list[QWidget]
            List of widgets to be added.
        stretches : list[int] or None, optional
            List of stretches for each widget. Must have the same size of 
            widgets. If None, stretches are automatically set to 0 for each 
            widget. The default is None.

        Raises
        ------
        AssertionError
            Widgets and stretches must have the same size.

        '''
        if stretches is None:
            stretches = [0] * len(widgets)
        else:
            assert len(widgets) == len(stretches)

        for w, s in zip(widgets, stretches):
            self.addWidget(w, s)



class SplitterGroup(QW.QSplitter):
    '''
    Convenient class to quickly group multiple widgets in a QSplitter.
    '''
    def __init__(self, qobjects: list[QC.QObject]|None=None, 
                 stretches: list[int]|None=None, orient=QC.Qt.Horizontal):
        '''
        Constructor

        Parameters
        ----------
        qobjects : list[QObject] or None, optional
            List of layouts or widgets. If None the list will be empty. The 
            default is None.
        stretches : list[int] or None, optional
            List of stretches for each object. Must have the same size of  
            qobjects. If None, stretches are not set. The default is None.
        orient : Qt.Orientation, optional
            Orientation of the splitter. The default is Qt.Horizontal.

        Raises
        ------
        AssertionError
            Objects and stretches must have the same size.

        '''
    # Use super class to create the oriented splitter and set its stylesheet
        self.orient = orient
        super(SplitterGroup, self).__init__(self.orient)
        self.setStyleSheet(style.SS_SPLITTER)

    # Add each object to the splitter. If the object is a layout, wrap it first
    # in a QWidget
        if qobjects is None:
            qobjects = []
        for obj in qobjects:
            if not obj.isWidgetType():
                w_obj = QW.QWidget()
                w_obj.setLayout(obj)
                self.addWidget(w_obj)
            else:
                self.addWidget(obj)

    # Add stretches to each object
        if stretches is not None:
            assert len(stretches) == len(qobjects)
            for i, s in enumerate(stretches):
                self.setStretchFactor(i, s)


    def addStretchedWidget(self, widget: QW.QWidget, stretch: int, idx=-1):
        '''
        Convenient function to add a widget to the splitter with given stretch.
        If an index is provided, the insertion position can be choosen.

        Parameters
        ----------
        widget : QWidget
            Widget to be added.
        stretch : int
            Stretch factor for the added widget.
        idx : int, optional
            Index of insertion. If invalid or -1, the widget is inserted at the 
            end. The default is -1.

        '''
        self.insertWidget(idx, widget)
        self.setStretchFactor(idx, stretch)



class FramedLabel(QW.QLabel):
    '''
    A label decorated with a black squared frame. Its text is user-selectable
    and its size policy is fixed to avoid visual glitches when using monitors
    with different resolutions.
    '''
    def __init__(self, text='', parent: QC.QObject|None=None):
        '''
        Constructor.

        Parameters
        ----------
        text : str, optional
            Label text. The default is ''.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(FramedLabel, self).__init__(text, parent)
    
    # Set widget attributes
        self.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)
        self.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)

    # Set stylesheet
        self.setStyleSheet(style.SS_MENU + 'QLabel {border: 1px solid black;}')



class PathLabel(FramedLabel):
    '''
    A special type of FramedLabel that allow easy management and display of 
    file paths.
    '''
    def __init__(self, fullpath='', full_display=True, elide=True,
                 placeholder='', parent: QC.QObject|None=None):
        '''
        Constructor.

        Parameters
        ----------
        fullpath : str, optional
            The full filepath. The default is ''.
        full_display : bool, optional
            Whether the label should display the full filepath or just the file
            name. The default is True.
        elide : bool, optional
            Whether the label should automatically wrap long text. The default 
            is True.
        placeholder : str, optional
            Text to be set when the label is cleared. The default is ''.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(PathLabel, self).__init__(parent=parent)

    # Set main attributes
        self.fullpath = fullpath
        self.full_display = full_display
        self.elide = elide
        self.placeholder = placeholder
    
    # Set widget attributes
        self.setAlignment(QC.Qt.AlignLeft | QC.Qt.AlignVCenter)

    # Set stylesheet
        self.setStyleSheet(style.SS_PATHLABEL + style.SS_MENU)

    # Draw the file path
        self.display()


    @property
    def _displayedText(self):
        '''
        Internal function that returns the string that should be displayed.

        Returns
        -------
        text : str
            Displayed path.

        '''
        text = self.fullpath
        if not self.full_display:
            text = cf.path2filename(text, ext=True)
        if text == '':
            text = self.placeholder

        return text


    def display(self):
        '''
        Display the filepath in the QLabel.

        '''
        text = self._displayedText
        self.setTextElided(text) if self.elide else self.setText(text)
        self.setToolTip(self.fullpath)


    def setTextElided(self, text: str):
        '''
        Wrap long text.

        Parameters
        ----------
        text : str
            Text to be elided.

        '''
        font = pref.get_setting('main/font', QG.QFont())
        metrics = QG.QFontMetrics(font)
        elided = metrics.elidedText(text, QC.Qt.ElideRight, self.width()-2)
        self.setText(elided)


    def setPath(self, path: str|None, auto_display=True):
        '''
        Set a new filepath.

        Parameters
        ----------
        path : str | None
            The new filepath.
        auto_display : bool, optional
            Whether the new filepath should also be automatically displayed. 
            The default is True.

        '''
        if path is None: path = '*Path not found'
        self.fullpath = path
        if auto_display: self.display()


    def clearPath(self):
        '''
        Remove the current filepath.
        '''
        self.setText(self.placeholder)
        self.fullpath = ''
        self.setToolTip('')

    def resizeEvent(self, event):
        '''
        Reimplementation of the resizeEvent function. If text wrapping is 
        enabled, allows the label to elide the text adapting it to new size.

        Parameters
        ----------
        event : QEvent
            The resize event.

        '''
        event.accept()
        if self.elide:
            self.setTextElided(self._displayedText)



class PopUpProgBar(QW.QProgressDialog):
# !!! Should the parent be None by default to delete the pbar after execution? (because it has no parent alive)
    def __init__(self, parent, n_iter, label='', cancel=True, forceShow=False):
        btn_text = 'Abort' if cancel else None
        flags = QC.Qt.Tool | QC.Qt.WindowTitleHint | QC.Qt.WindowStaysOnTopHint
        super(PopUpProgBar, self).__init__(label, btn_text, 0, n_iter, parent,
                                           flags=flags)
        self.setWindowModality(QC.Qt.ApplicationModal) # experimental --> original was QC.Qt.WindowModal
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




class PulsePopUpProgBar(PopUpProgBar):
    def __init__(self, parent, label='', cancel=False):
        super(PulsePopUpProgBar, self).__init__(parent, 1, label, cancel, True)
        self.setAutoReset(False)

    def startPulse(self):
        self.setValue(1)
        self.setRange(0, 0)

    def stopPulse(self):
        self.reset()



class DecimalPointSelector(StyledComboBox):
    '''
    Convenient reimplementation of a styled combo box that allows the selection
    of the decimal point selector based on the local system settings.
    '''

    def __init__(self):
        '''
        Constructor.

        '''
        super(DecimalPointSelector, self).__init__()
        local_decimal_point = QC.QLocale().decimalPoint()
        self.addItems(['.', ','])
        self.setCurrentText(local_decimal_point)



class SeparatorSymbolSelector(StyledComboBox):
    '''
    Convenient reimplementation of a styled combo box that allows the selection
    of the separator symbol based on the local system settings.
    '''

    def __init__(self):
        '''
        Constructor.

        '''
        super(SeparatorSymbolSelector, self).__init__()
        local_separator = ',' if QC.QLocale().decimalPoint() == '.' else ';'
        self.addItems([',', ';'])
        self.setCurrentText(local_separator)



# class CsvChunkReader(): # deprecated. Moved to dataset_tools
#     '''
#     Ready to use class for reading large CSV files in chunks for better
#     performance.
#     '''
#     def __init__(self, dec: str, sep: str|None=None, chunksize=2**20//8, 
#                  pBar=False):
#         '''
#         Constructor.

#         Parameters
#         ----------
#         dec : str
#             Decimal point symbol.
#         sep : str | None, optional
#             Separator symbol. If None it is inferred. The default is None.
#         chunksize : int, optional
#             Dimension of the reading batch. The default is 2**20//8.
#         pBar : bool, optional
#             Whether a popup progress bar should be displayed during the reading
#             operation. The default is False.

#         '''
#     # Set main attributes
#         self.dec = dec
#         self.sep = sep
#         self.chunksize = chunksize
#         self.pBar = pBar


#     def read(self, filepath: str):
#         '''
#         Read the CSV file and return a pandas dataframe.

#         Parameters
#         ----------
#         filepath : str 
#             The CSV filepath.

#         Returns
#         -------
#         dataframe : Dataframe
#             Pandas Dataframe.
#         '''
#     # Set the popup progress bar if required
#         if self.pBar:
#             with open(filepath) as temp:
#                 nChunks = sum(1 for _ in temp) // self.chunksize
#             progBar = PopUpProgBar(None, nChunks + 1, 'Loading Dataset', 
#                                    cancel=False)

#     # Read the CSV
#         chunkList = []
#         with pd.read_csv(filepath, decimal=self.dec, sep=self.sep, 
#                          engine='python', chunksize=self.chunksize) as reader:
#             for n, chunk in enumerate(reader):
#                 chunkList.append(chunk)
#                 if self.pBar: progBar.setValue(n)

#     # Compile and return the pandas Dataframe
#         dataframe = pd.concat(chunkList)
#         if self.pBar: 
#             progBar.setValue(nChunks + 1)
#         return dataframe


# class KernelSelector(QW.QWidget): # deprecated. Now included within the Phase Refiner tool

#     structureChanged = QC.pyqtSignal()

#     def __init__(self, rank=2, parent=None):
#         super(KernelSelector, self).__init__()
#         self.rank=rank

#         # Shapes -> Square: 0, Circle: 1, Rhombus: 2
#         self.shape = 0
#         # self.sq_connect = True
#         self.iter = 1

#         self.structure = None
#         self.build_defaultStructure()

#         self.init_ui()

#     def init_ui(self):
#     # Kernel structure representation
#         self.grid = QW.QGridLayout()
#         self.build_grid()
#         grid_box = GroupArea(self.grid)
#         grid_box.setStyleSheet('''border: 2 solid black;
#                                   padding-top: 0px;''')

#     # # Square connectivity checkbox
#     #     self.connect_cbox = QW.QCheckBox('Square Connectivity')
#     #     self.connect_cbox.setChecked(self.sq_connect)
#     #     self.connect_cbox.stateChanged.connect(self.set_connectivity)

#     # Default kernel shapes radio buttons group
#         self.kernelShapes_btns = RadioBtnLayout(('Square', 'Circle', 'Rhombus'),
#                                                 orient='horizontal')
#         self.kernelShapes_btns.selectionChanged.connect(self.set_shape)

#     # Kernel size slider selector
#         self.kernelSize_slider = QW.QSlider(QC.Qt.Horizontal)
#         self.kernelSize_slider.setMinimum(1)
#         self.kernelSize_slider.setMaximum(5)
#         self.kernelSize_slider.setSingleStep(1)
#         self.kernelSize_slider.setSliderPosition(self.iter)
#         self.kernelSize_slider.valueChanged.connect(self.set_iterations)
#         kernelSize_form = QW.QFormLayout()
#         kernelSize_form.addRow('Kernel size', self.kernelSize_slider)

#     # Adjust main layout
#         mainLayout = QW.QVBoxLayout()
#         mainLayout.addWidget(grid_box, alignment=QC.Qt.AlignCenter)
#         # mainLayout.addWidget(self.connect_cbox)
#         mainLayout.addLayout(self.kernelShapes_btns)
#         mainLayout.addLayout(kernelSize_form)
#         self.setLayout(mainLayout)

#     def get_structure(self):
#         return self.structure

#     def updateKernel(self):
#         self.build_defaultStructure()
#         self.build_grid()
#         self.structureChanged.emit()

#     def build_defaultStructure(self):
#         # Connectivity (conn) allows to build only rhombic (1) or squared (2) structures
#         conn = 1 if self.shape == 2 else 2
#         struct = scipy.ndimage.generate_binary_structure(self.rank, conn)
#         struct = scipy.ndimage.iterate_structure(struct, self.iter)
#         # If a circle (self.shape=1) was required, transform the squared structure into circular
#         if self.shape == 1:
#             struct = self._drawCircleStructure(struct)
#         self.structure = struct

#     def _drawCircleStructure(self, baseStructure):
#         # J is the number of rows/cols of base (square) structure
#         J = baseStructure.shape[0]
#         # Compute the Euclidean distance from center (x=J//2; y=J//2)
#         Y, X = np.ogrid[:J, :J]
#         dist = np.sqrt((X - J//2)**2 + (Y - J//2)**2)
#         # Return the circle mask
#         return dist <= J/2 # J/2 = radius (in float type)

#     def set_shape(self, idx):
#         self.shape = idx
#         self.updateKernel()

#     # def set_connectivity(self, isSquared):
#     #     self.sq_connect = isSquared
#     #     self.updateKernel()

#     def set_iterations(self, value):
#         self.iter = value
#         self.updateKernel()

#     def _clear_grid(self, grid):
#         for i in reversed(range(grid.count())):
#             grid.itemAt(i).widget().setParent(None)

#     def draw_node(self, active):
#         img = QG.QImage(8, 8, QG.QImage.Format_RGB32)
#         rgb = (255,0,0) if active else (0,0,0)
#         img.fill(QG.QColor(*rgb))
#         node = QG.QPixmap.fromImage(img)
#         return node

#     def build_grid(self):
#         self._clear_grid(self.grid)
#         radius = self.structure.shape[0]
#         for row in range(radius):
#             for col in range(radius):
#                 wid = QW.QLabel()
#                 wid.setPixmap(self.draw_node(self.structure[row, col]))
#                 self.grid.addWidget(wid, row, col)

#     # def build_grid(self):
#     #     self._clear_grid(self.grid)
#     #     radius = self.structure.shape[0]
#     #     for row in range(radius):
#     #         for col in range(radius):
#     #             node = QW.QPushButton()
#     #             node.setMaximumSize(12,12)
#     #             node.setCheckable(True)
#     #             node.setStyleSheet('''QPushButton {background-color : black;}'''
#     #                                '''QPushButton::checked {background-color : red;}''')
#     #             node.setChecked(self.structure[row, col])
#     #             node.toggled.connect(lambda tgl, r=row, c=col: self.activate_node(tgl, r, c))
#     #             self.grid.addWidget(node, row, col)

#     # def activate_node(self, active, row, col):
#     #     # Avoid Erosion + recontruction freezing bug, by limiting the number of active nodes to 5
#     #     if np.count_nonzero(self.structure) <= 5 and not active:
#     #         self.sender().setChecked(True)
#     #         return
#     #     self.structure[row, col] = active
#     #     self.structureChanged.emit()
#     #     print('FUNCTION Finished')


class RichMsgBox(QW.QMessageBox): # deprecated and will be removed. Use MsgBox instead. 
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


    
class MsgBox(QW.QMessageBox):
    '''
    Convenient class to quickly construct a message box dialog with default 
    icon and buttons depending on selected 'type_'. All property can also be
    fully customized through parameters. It also includes methods to easily get
    user interaction with 'Yes' and 'No' buttons and optional checkbox.
    '''

    def __init__(self, parent, type_: str, text: str='', dtext: str='',
                 title: str|None=None, icon: QW.QMessageBox.Icon|None=None, 
                 btns: QW.QMessageBox.StandardButtons|None=None, 
                 def_btn: QW.QMessageBox.StandardButton|None=None,
                 cbox: QW.QCheckBox|None=None):
        '''
        Constructor.

        Parameters
        ----------
        parent : QObject
            The GUI parent of the message box.
        type_ : str
            Message box type. It controls the default icon and buttons if not
            specified. It must be one of 'Info' (or 'I'), 'Quest (or 'Q'),
            'Warn' (or 'W'), 'QuestWarn' (or 'QW') or 'Crit' (or 'C').
        text : str, optional
            Message box text. The default is ''.
        dtext : str, optional
            Message box detailed text. If set, a 'Show Details' button will
            also be added. The default is ''.
        title : str or None, optional
            Message box title. If None, it will be the 'parent' window title or
            'X-Min Learn' if it is invalid. The default is None.
        icon : QMessageBox.Icon or None, optional
            Message box icon. If None, it is set according to 'type_'. The
            default is None.
        btns : QMessageBox.StandardButtons or None, optional
            Message box buttons. If None, they are set according to 'type_'. 
            The default is None.
        def_btn : QMessageBox.StandardButton or None, optional
            Default selected button. If None, it is set according to 'type_'.
            The default is None.
        cbox : QCheckBox or None, optional
            If set, adds a checkbox to the message box. The default is None.

        Raises
        ------
        ValueError
            Raised if an invalid 'type_' value is set.

        '''
    # Set default icon, buttons and default button according to message type
        if type_ in ('Info', 'I'):
            icon = QW.QMessageBox.Information if icon is None else icon
            btns = QW.QMessageBox.Ok if btns is None else btns
        elif type_ in ('Quest', 'Q'):
            icon = QW.QMessageBox.Question if icon is None else icon
            btns = QW.QMessageBox.Yes | QW.QMessageBox.No if btns is None else btns
            def_btn = QW.QMessageBox.No if def_btn is None else def_btn
        elif type_ in ('Warn', 'W'):
            icon = QW.QMessageBox.Warning if icon is None else icon
            btns = QW.QMessageBox.Ok if btns is None else btns
        elif type_ in ('QuestWarn', 'QW'):
            icon = QW.QMessageBox.Warning if icon is None else icon
            btns = QW.QMessageBox.Yes | QW.QMessageBox.No if btns is None else btns
            def_btn = QW.QMessageBox.No if def_btn is None else def_btn
        elif type_ in ('Crit', 'C'):
            icon = QW.QMessageBox.Critical if icon is None else icon
            btns = QW.QMessageBox.Ok if btns is None else btns
        else:
            raise ValueError('f{type_} is an invalid message box type.')
        
    # Auto-set title if not specified
        if title is None:
            try:
                title = parent.windowTitle()
                if title == '': 
                    title = 'X-Min Learn'
            except AttributeError:
                title = 'X-Min Learn'

    # Set dialog attributes
        super(MsgBox, self).__init__(icon, title, text, btns, parent)
        self.setDefaultButton(def_btn)
        self.setDetailedText(dtext)
        self.setCheckBox(cbox)

    # Show dialog
        self.exec()


    def yes(self):
        '''
        Check if user clicked on 'Yes' button.

        Returns
        -------
        bool
            Whether user clicked on 'Yes'.

        '''
        return self.clickedButton().text() == '&Yes'
    

    def no(self):
        '''
        Check if user clicked on 'No' button.

        Returns
        -------
        bool
            Whether user clicked on 'No'.
            
        '''
        return self.clickedButton().text() == '&No'
    

    def cboxChecked(self):
        '''
        Check the state of the checkbox. If no checkbox was set, this function
        returns False.

        Returns
        -------
        bool
            Whether the checkbox was checked.

        '''
        cbox = self.checkBox()
        if cbox is None:
            return False
        else:
            return cbox.isChecked() 
        
                

class LineSeparator(QW.QFrame):
    '''
    Simple horizontal or vertical line separator.
    '''
    def __init__(self, orient='horizontal', lw=3, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        orient : str, optional
            Line orientation. The default is 'horizontal'.
        lw : int, optional
            Line width. Must be in range (1, 3). The default is 3.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        Raises
        ------
        ValueError
            Orient must be either 'horizontal' or 'vertical'.

        '''
        super(LineSeparator, self).__init__(parent)

        if orient == 'horizontal':
            self.setFrameShape(QW.QFrame.HLine)
        elif orient == 'vertical':
            self.setFrameShape(QW.QFrame.VLine)
        else:
            raise ValueError(f'{orient} is not a valid orientation.')
        
        self.setFrameShadow(QW.QFrame.Plain)
        self.setLineWidth(lw)
      


class PixelFinder(QW.QFrame):
    '''
    Custom widget that allows to zoom to a specific pixel on a canvas given its
    X and Y (or column and row) coordinates.
    '''
    def __init__(self, canvas, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        canvas : ImageCanvas, optional
            The canvas to zoom in.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(PixelFinder, self).__init__(parent)
        self.canvas = canvas
        self.setStyleSheet('''QFrame {border: 1 solid black;}''')

    # X and Y coords input (with integer validator)
        validator = QG.QIntValidator(0, 10**8)

        self.X_input = QW.QLineEdit()
        self.X_input.setStyleSheet(style.SS_MENU)
        self.X_input.setAlignment(QC.Qt.AlignHCenter)
        self.X_input.setPlaceholderText('X (C)')
        self.X_input.setValidator(validator)
        self.X_input.setToolTip('X or Column value')
        self.X_input.setMaximumWidth(50)

        self.Y_input = QW.QLineEdit()
        self.Y_input.setStyleSheet(style.SS_MENU)
        self.Y_input.setAlignment(QC.Qt.AlignHCenter)
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
        mainLayout.addWidget(self.go_btn, alignment=QC.Qt.AlignHCenter)
        mainLayout.addWidget(self.X_input, alignment=QC.Qt.AlignHCenter)
        mainLayout.addWidget(self.Y_input, alignment=QC.Qt.AlignHCenter)
        self.setLayout(mainLayout)


    def zoomToPixel(self):
        '''
        Zoom to pixel if its coordinates are valid.
        '''
        x, y = None, None
        if self.X_input.hasAcceptableInput():
            x = int(self.X_input.text())
        if self.Y_input.hasAcceptableInput():
            y = int(self.Y_input.text())

        if x is not None and y is not None:
            self.canvas.zoom_to(x, y)


    def setInputCellsMaxWidth(self, width: int):
        '''
        Set the maximum width of the coordinates input cells.

        Parameters
        ----------
        width : int
            Maximum width.

        '''
        self.X_input.setMaximumWidth(width)
        self.Y_input.setMaximumWidth(width)



class DocumentBrowser(QW.QWidget):
    '''
    Custom widget to load, display and browse a text document. It includes
    convenient search, edit and zoom functionalities.
    '''
    def __init__(self, read_only=False, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        read_only : bool, optional
            Whether document is read only or editable. The default is False.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(DocumentBrowser, self).__init__(parent)

    # Set main attributes
        self.read_only = read_only
        self._font = QG.QFont()
        self.placeholder_text = ''

    # Set GUI and connect signals to slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Text browser space
        self.browser = QW.QTextEdit(self.placeholder_text)
        self.browser.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical)) 
        self.browser.setStyleSheet(style.SS_MENU)
        self.browser.setReadOnly(self.read_only)

    # Browser toolbar
        self.tbar = StyledToolbar('Browser toolbar')

    # Edit document checkbox
        self.edit_cbox = QW.QCheckBox('Editing')
        self.edit_cbox.setChecked(not self.read_only)
        self.edit_cbox.setVisible(not self.read_only)
        self.tbar.addWidget(self.edit_cbox)

    # Search box
        self.search_box = QW.QLineEdit()
        self.search_box.setStyleSheet(style.SS_MENU)
        self.search_box.setPlaceholderText('Search')
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setMaximumWidth(100)
        self.tbar.addWidget(self.search_box)

    # Search Up Action
        self.search_up_action = self.tbar.addAction(
            QG.QIcon(r'Icons/up.png'), 'Search up')

    # Search Down Action
        self.search_down_action = self.tbar.addAction(
            QG.QIcon(r'Icons/down.png'), 'Search down')

    # Zoom in Action
        self.zoom_in_action = self.tbar.addAction(
            QG.QIcon(r'Icons/zoom_in.png'), 'Zoom in')

    # Zoom out Action
        self.zoom_out_action = self.tbar.addAction(
            QG.QIcon(r'Icons/zoom_out.png'), 'Zoom out')

    # Insert toolbar separator
        self.tbar.insertSeparator(self.zoom_in_action)

    # Adjust Main Layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        layout.addWidget(self.tbar)
        layout.addWidget(self.browser)
        self.setLayout(layout)


    def _connect_slots(self): 
        '''
        Connect signals to slots.

        '''
    # Swap between read only and edit mode when the edit checkbox changes state
        self.edit_cbox.stateChanged.connect(
            lambda state: self.browser.setReadOnly(not state))
    
    # Search text in the document
        self.search_box.editingFinished.connect(self._findTextDown)
        self.search_up_action.triggered.connect(self._findTextUp)
        self.search_down_action.triggered.connect(self._findTextDown)

    # Change the text font size (zoom in, zoom out)
        self.zoom_in_action.triggered.connect(lambda: self._alterZoom(+1))
        self.zoom_out_action.triggered.connect(lambda: self._alterZoom(-1))


    def setDoc(self, doc_path: str):
        '''
        Load a new document and display it in the browser.

        Parameters
        ----------
        doc_path : str
            Text document path.

        '''
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as log:
                doc = QG.QTextDocument(log.read(), self.browser)
                self.browser.setDocument(doc)
        else:
            self.clear()


    def setText(self, text: str):
        '''
        Set a custom text to the browser.

        Parameters
        ----------
        text : str
            Custom text.

        '''
        self.browser.setText(text)


    def clear(self):
        '''
        Clear the browser and set the placeholder text.

        '''
        self.browser.clear()
        self.browser.setPlaceholderText(self.placeholder_text)


    def setDefaultPlaceHolderText(self, text: str):
        '''
        Set the default placeholder text of the browser. This text is displayed
        when no document is loaded.

        Parameters
        ----------
        text : str
            Placeholder text.

        '''
        self.placeholder_text = text


    def _alterZoom(self, value: int):
        '''
        Change the font size of the displayed document text.

        Parameters
        ----------
        value : int
            Incremental or decremental size value. For example if +1, the font
            size will be increased by 1 pt, if -1 it will be decreased by 1 pt.

        '''
    # The new font size must in any case be in range [0pt, 80pt]
        newSize = self._font.pointSize() + value
        if 0 < newSize < 80:
            self._font.setPointSize(newSize)
            self.browser.document().setDefaultFont(self._font)


    def _findTextUp(self):
        '''
        Search text up.

        '''
        self.browser.setFocus()
        find_flag = QG.QTextDocument.FindBackward
        self.browser.find(self.search_box.text(), find_flag)


    def _findTextDown(self):
        '''
        Search text down.

        '''
        self.browser.setFocus()
        self.browser.find(self.search_box.text())



class SampleMapsSelector(QW.QWidget):
    '''
    Ready to use widget that allows loading maps from a sample and show them
    in a QTreeWidget, optionally allowing their selection through dedicated 
    checkboxes. This widget sends signals to request the maps data. This
    signals must be catched by the widget that holds such information, namely
    the DataManager.
    '''
    sampleUpdateRequested = QC.pyqtSignal()
    mapsUpdateRequested = QC.pyqtSignal(int) # index of sample
    mapsDataChanged = QC.pyqtSignal()
    mapClicked = QC.pyqtSignal(DataObject)


    def __init__(self, maps_type: str, checkable=True, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        maps_type : str
            Must be 'inmaps' to list input maps or 'minmaps' to list mineral 
            maps.
        checkable : bool, optional
            Whether the individual maps in the list
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        Raise
        -----
        ValueError
            maps_type must be 'inmaps' or 'minmaps'

        '''
        super(SampleMapsSelector, self).__init__(parent)

    # Set main attributes
        self.maps_type = maps_type
        if self.maps_type not in ('inmaps', 'minmaps'):
            raise ValueError("maps_type must be 'inmaps' or 'minmaps'")
        self.checkable = checkable

        self._init_ui()
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Sample combobox (Auto Update Combo Box)
        self.sample_combox = AutoUpdateComboBox()

    # Maps list (Tree widget)
        self.maps_list = QW.QTreeWidget()
        self.maps_list.setHeaderHidden(True)
        self.maps_list.setStyleSheet(style.SS_MENU)

    # Set layout
        main_layout = QW.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(QW.QLabel('Select sample'))
        main_layout.addWidget(self.sample_combox)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.maps_list)
        self.setLayout(main_layout)


    def _connect_slots(self): 
        '''
        Signals-slots connector.

        '''
    # Send combobox signals as custom signals
        self.sample_combox.clicked.connect(self.sampleUpdateRequested.emit)
        self.sample_combox.activated.connect(
            lambda idx: self.mapsUpdateRequested.emit(idx))
        
    # Send tree widget signals as custom signals
        self.maps_list.itemClicked.connect(lambda i: self.mapClicked.emit(i))


    def updateCombox(self, samples: list):
        '''
        Populate the samples combobox with the samples currently loaded in the
        Data Manager. This function is called by the main window when the 
        combobox is clicked.

        Parameters
        ----------
        samples : list
            List of DataGroup objects.

        '''
        samples_names = [s.text(0) for s in samples]
        self.sample_combox.updateItems(samples_names)

    
    def updateList(self, sample: DataGroup):
        '''
        Updates the list of currently loaded maps owned by the sample currently
        selected in the samples combobox. This function is called by the main 
        window when a new item is selected in the sample combobox.

        Parameters
        ----------
        sample : DataGroup
            The currently selected sample as a DataGroup.

        '''
    # Clear the input maps lists
        self.maps_list.clear()

    # Exit function if the subgroup is empty
        subgr = sample.inmaps if self.maps_type == 'inmaps' else sample.minmaps
        if subgr.isEmpty():
            return

    # Get every input map object and re-assemble them into the inmaps list
        for c in subgr.getChildren():
            item = DataObject(c.get('data'))
            if self.checkable:
                item.setCheckState(0, QC.Qt.Checked)
            self.maps_list.addTopLevelItem(item)

    # Send a signal to inform that maps data changed
        self.mapsDataChanged.emit()


    def getChecked(self):
        '''
        Get the currently checked maps data objects.

        Returns
        -------
        checked : list
            List of checked maps data objects.

        '''
        if self.checkable:
            n_maps = self.maps_list.topLevelItemCount()
            items = [self.maps_list.topLevelItem(i) for i in range(n_maps)]
            checked = [i for i in items if i.checkState(0)]
            return checked
    

    def itemCount(self):
        '''
        Return the amount of maps loaded in the maps list.

        Returns
        -------
        int
            Number of maps.

        '''
        return self.maps_list.topLevelItemCount()
    

    def currentItem(self):
        '''
        Return the currently selected map.

        Returns
        -------
        DataObject
            Currently selected map object.

        '''
        return self.maps_list.currentItem()
    

    def clear(self):
        '''
        Clear out the entire widget.

        '''
        self.sample_combox.clear()
        self.maps_list.clear()



class DescriptiveProgressBar(QW.QWidget):
    '''
    A progress bar that shows a description of the current process.
    '''
    def __init__(self, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(DescriptiveProgressBar, self).__init__(parent)

    # Description label
        self.desc = QW.QLabel()
        self.desc.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)

    # Progress bar
        self.pbar = QW.QProgressBar()
    
    # Adjust main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.desc, alignment=QC.Qt.AlignCenter)
        layout.addWidget(self.pbar)
        self.setLayout(layout)


    def setMinimum(self, minimum: int):
        '''
        Set minimum progress bar value.

        Parameters
        ----------
        minimum : int
            Minimum value.

        '''
        self.pbar.setMinimum(minimum)


    def setMaximum(self, maximum: int):
        '''
        Set maximum progress bar value.

        Parameters
        ----------
        maximum : int
            Maximum value.

        '''
        self.pbar.setMaximum(maximum)


    def setRange(self, minimum: int, maximum: int):
        '''
        Set progress bar range.

        Parameters
        ----------
        minimum : int
            Minimum value.
        maximum : int
            Maximum value

        '''
        self.pbar.setRange(minimum, maximum)


    def setUndetermined(self):
        '''
        Set the prograss bar in an undetermined state.

        '''
        self.pbar.setRange(0, 0)


    def undetermined(self):
        '''
        Check if the progress bar is in an undetermined state.

        Returns
        -------
        undet : bool
            Progress bar has undetermined state.

        '''
        undet = self.pbar.minimum() == self.pbar.maximum() == 0
        return undet


    def step(self, step_description: str):
        '''
        Increase progress bar by one step and set a new step description. If 
        progress bar is in an undetermined state, just set the description.

        Parameters
        ----------
        step_description : str
            New step description.

        '''
        self.desc.setText(step_description)
        if not self.undetermined():
            self.pbar.setValue(self.pbar.value() + 1)


    def reset(self):
        '''
        Reset the progress bar.

        '''
        self.desc.clear()
        self.pbar.reset()
        


class RandomSeedGenerator(QW.QWidget):
    '''
    Ready to use widget to set, manage and display a random seed.
    '''
    seedChanged = QC.pyqtSignal(int, int) # old seed, new seed

    def __init__(self, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(RandomSeedGenerator, self).__init__(parent)

    # Set main attributes
        self._old_seed = None
        self.seed = None

    # Set GUI and connect signals to slots
        self._init_ui()
        self._connect_slots()

    # Set a random seed
        self.randomize_seed()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Random seed line edit
        self.seed_input = QW.QLineEdit()

    # Custom validator that accepts numbers between 1 and 999999999 and 
    # empty strings. This is required to control the behaviour of the lineedit
    # when user leaves the field empty.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8})?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.seed_input.setValidator(validator)

    # Randomize seed button
        self.rand_btn = StyledButton(QG.QIcon(r'Icons/dice.png'))

    # Adjust layout
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QW.QLabel('Random seed'))
        layout.addWidget(self.seed_input, 1)
        layout.addWidget(self.rand_btn, alignment = QC.Qt.AlignRight)
        self.setLayout(layout)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Make sure that the seed is never left empty
        self.seed_input.editingFinished.connect(self.fill_empty_seed)

    # Change seed either manually or through randomization
        self.seed_input.textChanged.connect(self.on_seed_changed)
        self.rand_btn.clicked.connect(self.randomize_seed)


    def randomize_seed(self):
        '''
        Randomize seed.

        '''
        seed = np.random.default_rng().integers(1, 999999999)
        self.seed_input.setText(str(seed))


    def on_seed_changed(self, new_seed: str):
        '''
        Set a new seed.

        Parameters
        ----------
        new_seed : str
            New seed in string format.

        '''
        if new_seed != '':
            self._old_seed = self.seed
            self.seed = int(new_seed)
            self.seedChanged.emit(self._old_seed, self.seed)


    def fill_empty_seed(self):
        '''
        Fill empty seed with a new random seed.

        '''
        if not self.seed_input.text():
            self.randomize_seed()



class PercentLineEdit(QW.QFrame):
    '''
    Advanced object that allows altering an integer either through a percentage 
    value with a spinbox or directly by typing a new number in a validated line
    edit. A custom set of icons visually indicate if the new integer is bigger, 
    smaller or equal to the original one.

    '''
    valueEdited = QC.pyqtSignal(int)

    def __init__(self, base_value: int, min_perc=1, max_perc=100, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        base_value : int
            Original integer value.
        min_perc : int, optional
            Minimum allowed percentage value. It cannot be smaller than 1. The
            default is 1.
        max_perc : int, optional
            Maximum allowed percentage value. It cannot be smaller than 100. 
            The default is 100.
        parent : QObject | None, optional
            The GUI parent of this object. The default is None.

        '''
        super(PercentLineEdit, self).__init__(parent)

    # Main attributes
        self._base = base_value
        self._value = base_value
        self._min_perc = 1 if min_perc < 1 else min_perc
        self._max_perc = 100 if max_perc < 100 else max_perc

    # Widget attributes
        self.setFrameStyle(QW.QFrame.StyledPanel | QW.QFrame.Plain)
        self.setLineWidth(2)

        self._init_ui()
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # QLineEdit for direct integer input. Equipped with a regex validator that
    # accepts numbers between 1 and 10**9 as well as empty strings. This allows
    # a fine control over the behaviour of the line edit.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8}|1000000000)?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.linedit = QW.QLineEdit(str(self._value))
        self.linedit.setValidator(validator)

    # Spinbox for percentage input (Styled Spinbox)
        self.spinbox = StyledSpinBox(self._min_perc, self._max_perc)
        #self.spinbox.setFixedWidth(100)
        self.spinbox.setSuffix(' %')
        self.spinbox.setValue(100)

    # Visual increase/decrese icon indicator (QLabel)
        self.iconlbl = QW.QLabel()
        self.setIcon()

    # Reset button (Styled Button)
        self.reset_btn = StyledButton(QG.QIcon(r'Icons/refresh.png'))
        self.reset_btn.setFlat(True)

    # Adjust layout
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.iconlbl)
        layout.addWidget(self.linedit)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
        self.linedit.editingFinished.connect(self.onLineditEditingFinished)
        self.linedit.textChanged.connect(self.onLineditChanged)
        self.spinbox.valueChanged.connect(self.onSpinboxChanged)
        self.reset_btn.clicked.connect(self.resetValue)


    def setIcon(self):
        '''
        Change visual increase/decrease icon indicator based on the difference
        between base value and new input value.
        '''
        delta = self._value - self._base

        if delta > 0:
            icon = r'Icons/increase.png'
        elif delta < 0:
            icon = r'Icons/decrease.png'
        else:
            icon = r'Icons/stationary.png'

        pixmap = QG.QPixmap(icon).scaled(20, 20, QC.Qt.KeepAspectRatio)
        self.iconlbl.setPixmap(pixmap)


    def onLineditEditingFinished(self):
        '''
        Replace empty strings with original base value. Send valueEdited signal
        otherwise.

        '''
        text = self.linedit.text()
        if not text:
            self.linedit.setText(str(self._base))
        else:
            self.valueEdited.emit(int(text))


    def onLineditChanged(self, value: str):
        '''
        Store new input value and auto adjust spinbox accordingly.

        Parameters
        ----------
        value : str
            Input line edit value

        '''
        self._value = int(value) if value else self._base
        perc = self.ratio() * 100

    # GUI update
        self.adjustSpinboxPrefix(perc)
        self.setIcon()
        
    # Prevent spinbox from sending signals when auto adjusted. Prenvent 
    # overflow errors as well when perc is bigger than upper spinbox limit. 
        perc = self._max_perc if perc > self._max_perc else round(perc)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(round(perc))
        self.spinbox.blockSignals(False)


    def onSpinboxChanged(self, perc: int):
        '''
        Compute new input value and auto adjust line edit accordingly.

        Parameters
        ----------
        perc : int
            Input spinbox value.

        '''
        self._value = round(self._base * perc / 100)

    # GUI update
        self.adjustSpinboxPrefix(perc)
        self.setIcon()
        
    # Prevent line edit from sending signals when auto adjusted.
        self.linedit.blockSignals(True)
        self.linedit.setText(str(self._value))
        self.linedit.blockSignals(False)

    # Send valueEdited signal 
        self.valueEdited.emit(self._value)


    def adjustSpinboxPrefix(self, perc: int|float):
        '''
        Set a prefix to spinbox if percentage is not within its value range. 

        Parameters
        ----------
        perc : int | float
            Percentage value.

        '''
        if perc < self._min_perc:
            self.spinbox.setPrefix('<')
        elif perc > self._max_perc:
            self.spinbox.setPrefix('>')
        else:
            self.spinbox.setPrefix('')

    
    def value(self):
        '''
        Getter function for value attribute.

        Returns
        -------
        int
            Current value.

        '''
        return self._value
    

    def setValue(self, value: int):
        '''
        Setter function for value attribute.

        Parameters
        ----------
        value : int
            Integer value.

        '''
        self.linedit.setText(str(value))


    def resetValue(self):
        '''
        Reset original value.

        '''
        self.setValue(self._base)


    def percent(self):
        '''
        Get current percent value. Prefixes are not honored. Use ratio() for 
        exact ratio.

        Returns
        -------
        int
            Current percent value.

        '''
        return self.spinbox.value()


    def setPercent(self, perc: int):
        '''
        Set current percent value.

        Parameters
        ----------
        perc : int
            Percent value.

        '''
        self.spinbox.setValue(perc)
    

    def ratio(self, round_decimals: int|None = None):
        '''
        Get current value / base value ratio.

        Parameters
        ----------
        round_decimals : int | None, optional
            Round ratio to this number of decimals. If None, no rounding is 
            performed. The default is None.

        Returns
        -------
        float | int
            Current ratio.

        '''
        r = self._value / self._base
        if round_decimals is not None:
            r = round(r, round_decimals)
        return r


