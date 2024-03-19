# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:27:17 2021

@author: dagos
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
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg,
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

mpl_use('Qt5Agg')




class DraggableTool(QW.QWidget):
    '''
    The main class for all major windows of X-Min Learn, that can be dragged
    and dropped within the main window tab area. See the MainTabWidget class
    for more details.
    '''

    def __init__(self, parent=None):
        '''
        DraggableTool class constructor.

        Parameters
        ----------
        parent : qObject, optional
            The GUI parent widget. The default is None.

        Returns
        -------
        None.

        '''
        super(DraggableTool, self).__init__(parent)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)


    # def dragEnterEvent(self, e): # NOT USEFUL HERE.
    #     '''
    #     Reimplementation of the dragEnterEvent default functions. It just
    #     accepts the event if the tool is not anchored (floating tool).

    #     Parameters
    #     ----------
    #     e : dragEvent
    #         The drag event triggered by the user.

    #     Returns
    #     -------
    #     None.

    #     '''

    #     if self.is_floating():
    #         e.accept()

    def mouseMoveEvent(self, e):
        '''
        Reimplementation of the mouseMoveEvent default function. It provides a
        custom animation for the widget drag & drop action (left-click drag).

        Parameters
        ----------
        e : mouseEvent
            The mouse move event triggered by the user.

        Returns
        -------
        None.

        '''
    # Ignore the event if the tool is already anchored to parent widget
        if not self.isWindow(): return

        if e.buttons() == Qt.LeftButton:
        # Generate mime data and a pixmap for the drag & drop event
            icon = self.windowIcon()
            pixmap = icon.pixmap(icon.actualSize(QSize(32, 32)))
            mimeData = QMimeData()

        # Generate a drag event and execute it
            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setPixmap(pixmap)
            drag.exec_(Qt.MoveAction)


    def closeEvent(self, event):
        '''
        Reimplementation of the closeEvent default function. It adds a question
        dialog to confirm the closing action. It returns True if the event is
        accepted, 0 otherwise.

        Parameters
        ----------
        event : closeEvent
            The close event triggered by the user.

        Returns
        -------
        bool
            Wether or not the close event is accepted.

        '''
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         f'Close {self.windowTitle()}?',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
    # accept() or ignore() are returned as boolean output, so that the event is
    # propagated to the parent widget. The parent can be None when closed as a
    # floating window or the MainTabWidget when closed from the tab. In the
    # latter case the MainTabWidget catches the event output and use that info
    # to "choose" to close or not close the corresponding tab.
        if choice == QW.QMessageBox.Yes:
            return event.accept()
        else:
            return event.ignore()





class Pane(QW.QDockWidget):
    '''
    The main class for every pane of X-Min Learn. It is a customized version
    of a PyQt DockWidget.
    '''

    def __init__(self, qObject, title='', scroll=True):
        '''
        Pane class constructor.

        Parameters
        ----------
        qObject : qWidget or qLayout
            The central widget/layout to be displayed in the pane.
        title : str, optional
            The title of the pane. The default is ''.
        scroll : bool, optional
            Wether or not the pane should be scrollable. The default is True.

        Returns
        -------
        None.

        '''
        super(Pane, self).__init__()

        # self._temporarily_hidden = False

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.title = title
        self.setWindowTitle(self.title)

    # If the qObject is a layout, build a dummy widget to contain it first
        if isinstance(qObject, QW.QLayout):
            self.widget = QW.QWidget()
            self.widget.setLayout(qObject)
        else:
            self.widget = qObject

    # If a scrollable pane is required, a QScrollArea is set as main container
        if scroll:
            self.scroll = QW.QScrollArea()
            self.scroll.setWidgetResizable(True)
            self.scroll.setWidget(self.widget)

        # Set custom scrollbars
            self.scroll.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
            self.scroll.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))

            self.setWidget(self.scroll)
        else:
            self.setWidget(self.widget)

    # Set the style-sheet (font-weight bold is for the title text)
        self.setStyleSheet('''Pane {font-weight: bold;}'''
                           '''Pane::title {background: %s;}'''
                           '''Pane::close-button, Pane::float-button {
                                background: %s;}'''
                                %(pref.CASPER_dark, pref.IVORY))


    # def setVisible(self, visible, hide_temporarily=False):
    #     super(Pane, self).setVisible(visible)

    #     if not visible and hide_temporarily:
    #         self._temporarily_hidden = True
    #     else:
    #         self._temporarily_hidden = False


    # def isTemporarilyHidden(self):
    #     return self._temporarily_hidden



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
        self.minmaps = DataSubGroup('Mineral Maps')
        self.masks = DataSubGroup('Masks')
        # TODO self.points = DataSubGroup('Point Analysis')
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
        # TODO include points data (maybe?)
        objects = []
        for subgr in self.subgroups:
            objects.extend(subgr.getChildren())
    # Extract the shape from each object data and obtain the trending shape
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
        composite_mask : Mask object or None
            The composite mask object or None if no mask is included.

        '''
        cld = self.masks.getChildren()

        if include == 'selected':
            masks = [c.get('data') for c in cld if c.isSelected()]
        elif include == 'checked':
            masks = [c.get('data') for c in cld if c.checkState(0)]
        else:
            raise TypeError('f{include} is not a valid argument for include.')

        if mode in ('union', 'U'):
        # 0*1 = 0 --> The more masks I add, the more 0's there'll be
            func = np.prod
        elif mode in ('intersection', 'I'):
        # 0+1 = 1 --> Only overlapping holes (0's) will survive
            func = np.sum
        else:
            raise TypeError(f'{mode} is not a valid argument for mode.')

        if len(masks) == 0:
            composite_mask = None
        elif len(masks) == 1:
            composite_mask = None if ignore_single_mask else masks[0]
        else:
            mask_array = func(np.array([m.mask for m in masks]), axis=0)
            composite_mask = Mask(mask_array)

        return composite_mask


    def clear(self):
    # TODO add point data
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
        if isinstance(data, (Mask,)): # TODO add PointAnalysis class
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
            # TODO add Point Analysis here
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
        #  TODO





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



class DataManager(QW.QTreeWidget):
    '''
    A widget for loading, accessing and managing input and output data.
    '''

    updateSceneRequested = pyqtSignal(object)
    clearSceneRequested = pyqtSignal()
    rgbaChannelSet = pyqtSignal(str)


    def __init__(self):
        '''
        DataManager class constructor.

        Returns
        -------
        None.

        '''
        super(DataManager, self).__init__()

    # Set some properties of the manager and its headers
        self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.setColumnCount(2)
        # self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(QW.QHeaderView.ResizeToContents)
        # self.setHeaderLabels([''] * self.columnCount())
        self.setHeaderHidden(True)

    # Disable default editing.Item editing is forced via onEdit() function
        self.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)

    # Enable custom context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    # Connect the signals to the custom slots
        self.itemClicked.connect(self.viewData)
        # self.itemChanged.connect(self.viewData) # too much triggers
        # self.itemActivated.connect(self.viewData) # never triggers
        self.itemDoubleClicked.connect(self.onEdit)
        self.customContextMenuRequested.connect(self.showContextMenu)

    # Set custom scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(Qt.Vertical))

    # Set the style-sheet (custom icons for expanded and collapsed branches and
    # right-click menu when editing items name)
        self.setStyleSheet(pref.SS_dataManager)


    def onEdit(self, item, column=0):
        '''
        Force item editing in first column.

        Parameters
        ----------
        item : DataObject
            The item that requests editing.
        column : int, optional
            The column where edits are requested. If different than 0, editing
            will be forced to be on first column anyway. The default is 0.

        Returns
        -------
        None.

        '''
        self.editItem(item, 0)


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
    # Define a menu (Styled Menu)
        menu = StyledMenu()

    # Get the item that is clicked at <point> and the group it belongs to
        item = self.itemAt(point)
        group = self.getItemParentGroup(item)

    # Context menu on void (<point> is not on an item)
        if item is None:

        # Add new group
            new_group = QW.QAction('New sample')
            new_group.triggered.connect(self.addGroup)

        # Clear all
            clear_all = QW.QAction('Clear all')
            clear_all.triggered.connect(self.clear)
            clear_all.triggered.connect(self.clearView)

            menu.addAction(new_group)
            menu.addSeparator()
            menu.addAction(clear_all)



    # Context menu on group
        elif isinstance(item, DataGroup):

        # Load input maps
            load_inmaps = QW.QAction(QIcon('Icons/generic_add_black.png'),
                                     'Add input maps')
            load_inmaps.triggered.connect(lambda: self.loadInputMaps(item))

        # Load mineral maps
            load_minmaps = QW.QAction(QIcon('Icons/generic_add_black.png'),
                                      'Add mineral maps')
            load_minmaps.triggered.connect(lambda: self.loadMineralMaps(item))

        # Load masks
            load_masks = QW.QAction(QIcon('Icons/generic_add_black.png'),
                                    'Add mask')
            load_masks.triggered.connect(lambda: self.loadMasks(item))

        # Clear group
            clear_group = QW.QAction('Clear sample')
            clear_group.triggered.connect(self.clearGroup)

        # Delete group
            del_group = QW.QAction('Delete sample')
            del_group.triggered.connect(self.delGroup)

            menu.addActions((load_inmaps, load_minmaps, load_masks))
            menu.addSeparator()
            menu.addActions((clear_group, del_group))

    # Context menu on subgroup
        elif isinstance(item, DataSubGroup):

        # Load data
            load_data = QW.QAction(QIcon('Icons/generic_add_black.png'),
                                   'Load data')
            load_data.triggered.connect(lambda: self.loadData(item))

            menu.addAction(load_data)
            menu.addSeparator()

        # Specific actions for Masks subgroup
            if item.name == 'Masks':

            # Check all masks
                chk_mask = QW.QAction('Check all')
                chk_mask.setEnabled(not item.isEmpty())
                chk_mask.triggered.connect(lambda: self.checkMasks(1, group))

            # Uncheck all masks
                unchk_mask = QW.QAction('Uncheck all')
                unchk_mask.setEnabled(not item.isEmpty())
                unchk_mask.triggered.connect(lambda: self.checkMasks(0, group))

                menu.addActions((chk_mask, unchk_mask))
                menu.addSeparator()

        # Clear subgroup
            clear_subgroup = QW.QAction('Clear')
            clear_subgroup.triggered.connect(item.clear)
            clear_subgroup.triggered.connect(self.clearView)


            menu.addAction(clear_subgroup)

    # Context menu on data objects
        elif isinstance(item, DataObject):
        # Generic actions, useful for any data type

        # Rename item
            rename_item = QW.QAction(QIcon('Icons/rename.png'), 'Rename')
            rename_item.triggered.connect(lambda: self.onEdit(item))

        # Delete item
            del_item = QW.QAction('Remove')
            del_item.triggered.connect(self.delData)

        # Refresh data source
            refresh_source = QW.QAction(QIcon('Icons/refresh.png'),
                                        'Refresh data source')
            refresh_source.triggered.connect(self.refreshDataSource)

        # Save
            save_item = QW.QAction(QIcon('Icons/save.png'), 'Save')
            save_item.triggered.connect(lambda: self.saveData(item))

        # Save As
            save_item_as = QW.QAction(QIcon('Icons/save.png'), 'Save As...')
            save_item_as.triggered.connect(lambda: self.saveData(item, False))

            menu.addActions((rename_item, del_item))
            menu.addSeparator()
            menu.addActions((refresh_source, save_item, save_item_as))
            menu.addSeparator()

        # Specific actions when item holds Input Maps
            if item.holdsInputMap():

            # Invert Map
                invert_map = QW.QAction(QIcon('Icons/invert.png'),
                                        'Invert map')
                invert_map.triggered.connect(self.invertInputMap)
                menu.addAction(invert_map)

            # RGBA submenu
                rgba_submenu = menu.addMenu('Set as RGBA channel...')
                rgba_submenu.setIcon(QIcon('Icons/RGBA.png'))
                for c in ('R', 'G', 'B', 'A'):
                    rgba_submenu.addAction(f'{c} channel',
                                    lambda c=c: self.rgbaChannelSet.emit(c))

        # Specific actions when item holds Mineral Maps
            elif item.holdsMineralMap():

            # Export Array
                export_array = QW.QAction(QIcon('Icons/export.png'), 'Export')
                export_array.triggered.connect(lambda: self.exportMineralMap(item))
                menu.addAction(export_array)

        # Specific actions when item holds Mineral Maps
            elif item.holdsMask():

            # Invert mask
                invert_mask = QW.QAction(QIcon('Icons/invert.png'),
                                         'Invert mask')
                invert_mask.triggered.connect(self.invertMask)
                menu.addAction(invert_mask)

            # Merge masks sub-menu
                mergemask_submenu = menu.addMenu('Merge masks')
                mergemask_submenu.addAction('Union',
                                            lambda: self.mergeMasks(group,'U'))
                mergemask_submenu.addAction('Intersection',
                                            lambda: self.mergeMasks(group,'I'))

        # TODO specific actions when item hold point data


    # Deal with anything else, just for safety reasons
        else: return

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def getAllGroups(self):
        '''
        Get all the groups.

        Returns
        -------
        groups : list
            List of DataGroup objects.

        '''
        count = self.topLevelItemCount()
        groups = [self.topLevelItem(idx) for idx in range(count)]
        return groups


    def getItemParentGroup(self, item):
        '''
        Get the item's group (i.e, an instance of DataGroup).

        Parameters
        ----------
        item : DataObject or DataSubGroup or DataGroup
            The item whose group needs to be retrieved.

        Returns
        -------
        group : DataGroup or None
            The item's group. Returns None when <item> is not a valid item.

        '''
    # Item is group
        if isinstance(item, DataGroup):
            group = item
    # Item is subgroup
        elif isinstance(item, DataSubGroup):
            group = item.parent()
    # Item is data object
        elif isinstance(item, DataObject):
            group = item.parent().parent()
    # Item is invalid (safety)
        else:
            group = None

        return group


    def getSelectedGroupsIndexes(self):
        '''
        Get the indices of the selected groups (i.e., instances of DataGroup).

        Returns
        -------
        None.

        '''
        items = self.selectedItems()
    # Groups are the top level items of the DataManager.
    # IndexOfTopLevelItem returns -1 if the item is not a toplevelitem
        indexes = map(lambda i: self.indexOfTopLevelItem(i), items)
        return filter(lambda idx: idx != -1, indexes)


    def getSelectedDataObjects(self):
        '''
        Get the selected data objects (i.e., instances of DataObject).

        Returns
        -------
        data_obj : list
            The selected data objects.

        '''
        items = self.selectedItems()
        data_obj = [i for i in items if isinstance(i, DataObject)]
        return data_obj



    def addGroup(self, return_group=False):
        '''
        Add a new group (i.e., an instance of DataGroup) to the manager.

        Parameters
        ----------
        return_group : bool, optional
            Optionally return the group. The default is False.

        Returns
        -------
        None.

        '''
    # Automatically rename the group as 'New Sample' + a progressive integer id
        unnamed_groups = self.findItems('New Sample', Qt.MatchContains)
        text = 'New Sample'
        if (n := len(unnamed_groups)): text += f' ({n})'
        new_group = DataGroup(text)
        self.addTopLevelItem(new_group)
        if return_group: return new_group

    def delGroup(self):
        '''
        Remove selected groups (i.e., instances of DataGroup) from the manager.

        Returns
        -------
        None.

        '''
        selected = self.getSelectedGroupsIndexes()
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Remove selected sample(s)?',
                                          QW.QMessageBox.Yes | QW.QMessageBox.No,
                                          QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            for idx in sorted(selected, reverse=True):
                self.takeTopLevelItem(idx)

        self.clearView()


    def clearGroup(self):
        '''
        Clear the data from selected groups (i.e., instances of DataGroup).

        Returns
        -------
        None.

        '''
        selected = self.getSelectedGroupsIndexes()
        for idx in selected:
            group = self.topLevelItem(idx)
            group.clear()
        self.clearView()


    def delData(self):
        '''
        Delete the selected data objects (i.e., instances of DataObject).

        Returns
        -------
        None.

        '''
        items = self.getSelectedDataObjects()
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Remove selected data?',
                                          QW.QMessageBox.Yes | QW.QMessageBox.No,
                                          QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            for i in reversed(items):
                subgroup = i.parent()
                subgroup.delChild(i)

        self.refreshView()


    def loadData(self, subgroup):
        '''
        Load data to a subgroup (i.e., instance of DataSubGroup). This is a
        generic loading function that checks the subgroup type and then calls
        the specialized loading function.

        Parameters
        ----------
        subgroup : DataSubGroup
            The subgroup to be populated with data.

        Returns
        -------
        None.

        '''
        group = subgroup.parent()
        name = subgroup.text(0)

        if name == 'Input Maps':
            self.loadInputMaps(group)

        elif name == 'Mineral Maps':
            self.loadMineralMaps(group)

        elif name == 'Masks':
            self.loadMasks(group)

        elif name == 'Point Analysis':
            # TODO
            pass

        else: return


    def saveData(self, item, overwrite=True):
        '''
        Save the item data to file.

        Parameters
        ----------
        item : DataObject
            The item to be saved.
        overwrite : bool, optional
            Overwrite the original filepath. If False, user will be prompted to
            choose a new path. The default is True.

        Returns
        -------
        None.

        '''

        item_data = item.get('data')
        path = None

    # If overwrite is True, check if the original path still exists. If not,
    # set path to None. If it exists, double check if user really wants to
    # overwrite the file
        if overwrite:
            path = item_data.filepath
            if path is not None and exists(path):
                choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                                 'Overwrite this file?',
                                                 QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                 QW.QMessageBox.No)
                if choice == QW.QMessageBox.No: return

            else:
                path = None

    # Prompt user to select a filepath if path is None
        if path is None:
            if item.holdsInputMap():
                filext = '''Compressed ASCII map (*.gz)
                            ASCII map (*.txt)'''
            elif item.holdsMineralMap():
                filext = 'Mineral maps (*.mmp)'
            elif item.holdsMask():
                filext = 'Mask (*.msk)'
            # TODO PointAnalysis
            else: return # safety

            path, _ = QW.QFileDialog.getSaveFileName(self, 'Save Map',
                                                     pref.get_dirPath('out'),
                                                     filext)

    # Finally, if there is a valid path, save the data
        if path:
            pref.set_dirPath('out', dirname(path))
            try:
                item_data.save(path)
            # Set the item edited status to False
                item.setEdited(False)
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))


    def loadInputMaps(self, group, paths=None):
        '''
        Specialized loading function to load input maps to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list, optional
            A list of filepaths to data. The default is None.

        Returns
        -------
        None.

        '''
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load input maps',
                                                       pref.get_dirPath('in'),
                                                       'ASCII maps (*.txt *.gz)')
        if paths:
            pref.set_dirPath('in', dirname(paths[0]))
            progBar = PopUpProgBar(self, len(paths), 'Loading data')
            for n, p in enumerate(paths, start=1):
                if progBar.wasCanceled(): break
                try:
                    xmap = InputMap.load(p)
                    group.inmaps.addData(xmap)

                except Exception as e:
                    progBar.setWindowModality(Qt.NonModal)
                    RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                               f'Unexpected ASCII file:\n{p}.',
                               detailedText = repr(e))
                    progBar.setWindowModality(Qt.WindowModal)

                finally:
                    progBar.setValue(n)

            self.expandRecursively(self.indexFromItem(group))


    def loadMineralMaps(self, group, paths=None):
        '''
        Specialized loading function to load mineral maps to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list, optional
            A list of filepaths to data. The default is None.

        Returns
        -------
        None.

        '''
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load mineral maps',
                                                       pref.get_dirPath('in'),
                                                       '''Mineral maps (*.mmp)
                                                          Legacy mineral maps (*.txt *.gz)''')
        if paths:
            pref.set_dirPath('in', dirname(paths[0]))
            progBar = PopUpProgBar(self, len(paths), 'Loading data')
            for n, p in enumerate(paths, start=1):
                if progBar.wasCanceled(): break
                try:
                    mmap = MineralMap.load(p)
                # Convert legacy mineral maps to new file format (mmp)
                    if splitext(p)[1] != '.mmp':
                        mmap.save(CF.extendFileName(p, '', '.mmp'))
                    group.minmaps.addData(mmap)

                except Exception as e:
                    progBar.setWindowModality(Qt.NonModal)
                    RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                               f'Unexpected file:\n{p}.',
                               detailedText = repr(e))
                    progBar.setWindowModality(Qt.WindowModal)

                finally:
                    progBar.setValue(n)

            self.expandRecursively(self.indexFromItem(group))


    def loadMasks(self, group, paths=None):
        '''
        Specialized loading function to load masks to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list, optional
            A list of filepaths to data. The default is None.

        Returns
        -------
        None.

        '''
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load masks',
                                                       pref.get_dirPath('in'),
                                                       '''Masks (*.msk)
                                                          Text file (*.txt)''')
        if paths:
            pref.set_dirPath('in', dirname(paths[0]))
            progBar = PopUpProgBar(self, len(paths), 'Loading data')
            for n, p in enumerate(paths, start=1):
                if progBar.wasCanceled(): break
                try:
                    mask = Mask.load(p)
                    group.masks.addData(mask)

                except Exception as e:
                    progBar.setWindowModality(Qt.NonModal)
                    RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                               f'Unexpected file:\n{p}.',
                               detailedText = repr(e))
                    progBar.setWindowModality(Qt.WindowModal)

                finally:
                    progBar.setValue(n)

            self.expandRecursively(self.indexFromItem(group))


    def invertInputMap(self):
        '''
        Invert the selected input maps.

        Returns
        -------
        None.

        '''
    # Get all the selected Input Maps items
        items = [i for i in self.getSelectedDataObjects() if i.holdsInputMap()]

    # Invert the input map arrays held in each item
        progBar = PopUpProgBar(self, len(items), 'Inverting data')
        for n, i in enumerate(items, start=1):
            if progBar.wasCanceled(): break
            i.get('data').invert()
        # Set edited state to True for item
            i.setEdited(True)

            progBar.setValue(n)

        self.refreshView()


    def invertMask(self):
        '''
        Invert the selected masks.

        Returns
        -------
        None.

        '''
    # Get all the selected Masks items
        items = [i for i in self.getSelectedDataObjects() if i.holdsMask()]

    # Invert the mask arrays held in each item
        progBar = PopUpProgBar(self, len(items), 'Inverting data')
        for n, i in enumerate(items, start=1):
            if progBar.wasCanceled(): break
            i.get('data').invert()
        # Set edited state to True for item
            i.setEdited(True)

            progBar.setValue(n)

        self.refreshView()


    def mergeMasks(self, group, mode):
        '''
        Merge the selected masks into a new Mask object and add it to the
        group.

        Parameters
        ----------
        group : DataGroup object
            The group that holds the mask data.
        mode : str
            How to merge the masks. See DataManager.getCompositeMask() for more
            details.

        Returns
        -------
        None.

        '''
    # (Safety) Exit function if group is invalid
        if group is None: return
    # Create a new Mask object
        merged_mask = group.getCompositeMask(mode=mode, ignore_single_mask=True)
    # Exit function if mask is invalid
        if merged_mask is None: return
    # Append the mask to the group
        group.masks.addData(merged_mask)
    # Set the new item as edited
        group.masks.getChildren()[-1].setEdited(True)


    def checkMasks(self, checked, group):
        '''
        (Un)check all masks loaded in a group.

        Parameters
        ----------
        checked : bool
            Whether to check or uncheck the masks.
        group : DataGroup
            The group whose masks should be (un)checked.

        Returns
        -------
        None.

        '''
        checkstate = Qt.Checked if checked else Qt.Unchecked
        for child in group.masks.getChildren():
            child.setCheckState(0, checkstate)
        self.refreshView()


    def exportMineralMap(self, item):
        '''
        Export the encoded mineral map (i.e,. with mineral classes expressed
        as numerical IDs) to ASCII format. If users requests it, the encoder
        dictionary is also exported.

        Parameters
        ----------
        item : DataObject
            The data object holding the mineral map data.

        Returns
        -------
        None.

        '''

    # Safety: exit function if item does not held mineral map data
        if not item.holdsMineralMap(): return

    # Construct a question message box
        msg_cbox = QW.QCheckBox('Include translation dictionary')
        msg_cbox.setChecked(True)
        choice = RichMsgBox(self, QW.QMessageBox.Question, 'X-Min Learn',
                            'Export map as a numeric array?',
                            QW.QMessageBox.Yes | QW.QMessageBox.No,
                            QW.QMessageBox.Yes,
                            'The translation dictionary is a text file that '\
                            'holds a reference to the mineral classes linked '\
                            'with the IDs of the exported mineral map.',
                            msg_cbox)
    # Get the outpath
        if choice.clickedButton().text() == '&Yes':
            outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Export Map',
                                                        pref.get_dirPath('out'),
                                                        '''ASCII file (*.txt)''')
            if outpath:
                pref.set_dirPath('out', dirname(outpath))

            # Save the mineral map to disk
                mmap = item.get('data')
                np.savetxt(outpath, mmap.minmap_encoded, fmt='%d')

            # Also save the encoder if user requests it
                if choice.checkBox().isChecked():
                    encoder_path = CF.extendFileName(outpath, '_transDict')
                    rows, cols = mmap.shape
                    with open(encoder_path, 'w') as ep:
                        for ID, lbl in mmap.encoder.items():
                            ep.write(f'{ID} :\t{lbl}\n')
                    # Include number of rows and columns
                        ep.write(f'\nNROWS: {rows}\nNCOLS: {cols}')




    def refreshDataSource(self):
        '''
        Re-load the selected data from its original source.

        Returns
        -------
        None.

        '''
        items = self.getSelectedDataObjects()
        progBar = PopUpProgBar(self, len(items), 'Reloading data')
        for n, i in enumerate(items, start=1):
            if progBar.wasCanceled(): break
            try:
                item_data, item_name = i.get('data', 'name')
                path = item_data.filepath
                if path is None: raise FileNotFoundError
                i.setData(0, 100, item_data.load(path))
            # Change the edited state of item
                i.setEdited(False)
            except FileNotFoundError:
                progBar.setWindowModality(Qt.NonModal)
                QW.QMessageBox.critical(self, 'X-Min Learn', 'The filepath to '\
                                        f'{item_name} was deleted, removed or '\
                                        'renamed.')
                progBar.setWindowModality(Qt.WindowModal)
            finally:
                progBar.setValue(n)

        self.refreshView()


    def viewData(self, item):
        '''
        Send signals for displaying the item's data.

        Parameters
        ----------
        item : DataObject or DataSubGroup (or DataGroup)
            The data object to be displayed. If an instance of DataGroup is
            provided, exits the function.

        Returns
        -------
        None.

        '''
    # # Exit the function if item is a group
    #     if isinstance(item, DataGroup):
    #         return

    # Update the scene if item is a data group, data subgroup or data object
        if isinstance(item, (DataObject, DataSubGroup, DataGroup)):
            self.updateSceneRequested.emit(item)

    # Clear the entire scene if the item is not valid (= None)
        else:
            self.clearView()


    def refreshView(self):
        self.viewData(self.currentItem())

    def clearView(self):
        self.clearSceneRequested.emit()












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

#     # TODO Merge classes

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

    # Extract mask
        extract_mask = QW.QAction('Extract mask')
        extract_mask.triggered.connect(self.requestMaskFromClass)


    # Add actions to menu
        menu.addAction(rename)
        menu.addAction(merge)
        menu.addSeparator()
        menu.addActions([copy_color, set_color, rand_color, rand_palette])
        menu.addSeparator()
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


class MainTabWidget(StyledTabWidget):
    '''
    Central widget of the X-Min Learn window. It is a reimplementation of a
    QTabWidget, customized to accept drag and drop events of its own tabs. Such
    tabs are the major X-Min Learn windows, that can be attached to this widget
    or detached and visualized as separated windows. See the DraggableTool
    class for more details.
    '''

    def __init__(self, parent):
        '''
        MainTabWidget class constructor.

        Parameters
        ----------
        parent : QWidget
            The GUI parent of this widget.

        Returns
        -------
        None.

        '''
        super(MainTabWidget, self).__init__(parent)

    # Set properties
        self.setAcceptDrops(True)
        self.setMovable(True)
        self.setTabsClosable(True)

    # Connect signals to slots
        self._connect_slots()


    def _connect_slots(self):
        '''
        MainTabWidget class signal-slots connector.

        Returns
        -------
        None.

        '''
    # Tab 'X' button pressed --> close the tab
        self.tabCloseRequested.connect(self.closeTab)
    # Tab Double-clicked --> detach the tab
        self.tabBarDoubleClicked.connect(self.popOut)



    def addTab(self, widget):
        '''
        Reimplementation of the default addTab function. A GroupScrollArea is
        set as the <widget> container. This helps in the visualization of
        complex widgets across different sized screens.

        Parameters
        ----------
        widget : QWidget
            The widget to be added as tab.

        Returns
        -------
        None.

        '''
        icon, title = widget.windowIcon(), widget.windowTitle()
        widget = GroupScrollArea(widget)
        super(MainTabWidget, self).addTab(widget, icon, title)
        self.setCurrentWidget(widget)


    def widget(self, index):
        '''
        Reimplementation of the default widget function. Returns the widget
        held by the tab at <index> bypassing the GroupScrollArea widget that
        contains it (see addTab function for more details).

        Parameters
        ----------
        index : int
            The position of the widget in the tab bar.

        Returns
        -------
        wid : QWidget or None
            The widget in the tab page or None if <index> is out of range.

        '''
        scroll_area = self.scrollArea(index)
        wid = None if scroll_area is None else scroll_area.widget()
        return wid


    def scrollArea(self, index):
        '''
        Returns the scroll area that holds the widget.

        Parameters
        ----------
        index : int
            The position of the widget in the tab bar.

        Returns
        -------
        scroll_area : GroupScrollArea
            The scroll area that contains the widget.

        '''
        scroll_area = super(MainTabWidget, self).widget(index)
        return scroll_area


    def closeTab(self, index):
        '''
        Close the tab at index <index>. Triggers the closeEvent of the widget.

        Parameters
        ----------
        index : int
            The tab index in the tab bar.

        Returns
        -------
        None.

        '''
    # The tab is closed only if the widget closeEvent is accepted
        closed = self.widget(index).close()
        if closed:
            scroll_area = self.scrollArea(index)
            self.removeTab(index)
            scroll_area.deleteLater()


    def popOut(self, index):
        '''
        Detach the tab from the MainTabWidget and display it as a separate
        window.

        Parameters
        ----------
        index : int
            The indec of the tab to be detached.

        Returns
        -------
        None.

        '''
        wid = self.widget(index)
        scroll_area = self.scrollArea(index)
    # Only pop out draggable tools
        if isinstance(wid, DraggableTool):
            self.removeTab(index)
            wid.setParent(None)
            wid.setVisible(True)
            wid.move(0, 0)
            wid.adjustSize()
            scroll_area.deleteLater()


    def dragEnterEvent(self, e):
        '''
        Reimplementation of the default dragEnterEvent function. Customized to
        accept only DraggableTool instances.

        Parameters
        ----------
        e : dragEvent
            The dragEvent triggered by the user's drag action.

        Returns
        -------
        None.

        '''
        if isinstance(e.source(), DraggableTool):
            e.accept()

    def dropEvent(self, e):
        '''
        Reimplementation of the default dropEvent function. Customized to
        accept only DraggableTool instances.

        Parameters
        ----------
        e : dropEvent
            The dropEvent triggered by the user's drag & drop action.

        Returns
        -------
        None.

        '''
        if isinstance(e.source(), DraggableTool):
            e.setDropAction(Qt.MoveAction)
            e.accept()

        # Suppress updates temporarily for better performances
            wid = e.source()
            self.setUpdatesEnabled(False)
            self.addTab(wid)
            self.setUpdatesEnabled(True)



class HistogramViewer(QW.QWidget):
    '''
    A widget to visualize and interact with histograms of input maps data.
    '''

    def __init__(self, maps_canvas, span=True, useblit=True):
        '''
        HistogramViewer class constructor.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas displaying input maps data.
        span : bool, optional
            Wether or not the histogram can be scaled using a span selector.
            The default is True.
        useblit : bool, optional
            Matplotlib GUI-related argument for the spanner. For more details
            see matplotlib.widget.SpanSelector. The default is True.

        Returns
        -------
        None.

        '''
        super(HistogramViewer, self).__init__()

        self.maps_canvas = maps_canvas
        self.has_scaler = span

    # Histogram Canvas
        self.canvas = plots.HistogramCanvas(logscale=True, size=(3, 1.5),
                                            tight=True, wheelZoomEnabled=False,
                                            wheelPanEnabled=False)
        self.canvas.ax.get_yaxis().set_visible(False)
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.setMinimumSize(300, 200)
        # self.canvas.setFixedSize(300, 200) # TODO for better performance
        # self.canvas.setSizePolicy(QW.QSizePolicy.Minimum, QW.QSizePolicy.Minimum)

    # Histogram Navigation Toolbar
        self.navTbar = plots.NavTbar(self.canvas, self, coords=False)
        self.navTbar.fixHomeAction()
        self.navTbar.removeToolByIndex([3, 4, 5, 8, 9])

    # Toggle log scale [-> NavTbar Action]
    # TODO set a proper icon
        self.logscale_action = QW.QAction(QIcon('Icons/equalizer.png'),
                                          'Log scale', self.navTbar)
        self.logscale_action.setCheckable(True)
        self.logscale_action.setChecked(True)
        self.logscale_action.toggled.connect(self.canvas.toggle_logscale)
        self.navTbar.insertAction(self.navTbar.findChildren(QW.QAction)[2],
                                  self.logscale_action)

    # HeatMap Scaler widget for the Histogram Canvas
        self.scaler = SpanSelector(self.canvas.ax, self.onSpanSelect,
                                   'horizontal', minspan=0, useblit=useblit,
                                   interactive=True, button=MouseButton.LEFT)
        self.scaler.set_active(self.has_scaler)

    # Min and max HeatMap Scaler values line edits (--> NavTbar)
        validator = QIntValidator(0, 10**8)

        self.scaler_vmin = StyledLineEdit()
        self.scaler_vmin.setAlignment(Qt.AlignHCenter)
        self.scaler_vmin.setPlaceholderText('Min')
        self.scaler_vmin.setValidator(validator)
        self.scaler_vmin.setToolTip('Min. span value')
        self.scaler_vmin.setMaximumWidth(100)
        self.scaler_vmin.editingFinished.connect(self.setScalerExtents)

        self.scaler_vmax = StyledLineEdit()
        self.scaler_vmax.setAlignment(Qt.AlignHCenter)
        self.scaler_vmax.setPlaceholderText('Max')
        self.scaler_vmax.setValidator(validator)
        self.scaler_vmax.setToolTip('Max. span value')
        self.scaler_vmax.setMaximumWidth(100)
        self.scaler_vmax.editingFinished.connect(self.setScalerExtents)

    # Show min and max values in NavTbar only if the scaler is required
        if self.has_scaler:
            self.navTbar.addSeparator()
            self.navTbar.addWidget(self.scaler_vmin)
            self.navTbar.addWidget(self.scaler_vmax)

    # Set bin slider widget
        self.bin_slider = QW.QSlider(Qt.Horizontal)
        self.bin_slider.setSizePolicy(QW.QSizePolicy.MinimumExpanding,
                                      QW.QSizePolicy.Fixed)
        self.bin_slider.setMinimum(5)
        self.bin_slider.setMaximum(100)
        self.bin_slider.setSingleStep(5)
        self.bin_slider.setSliderPosition(50)
        self.bin_slider.valueChanged.connect(self.setHistBins)

    # Ajust Layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navTbar)
        main_layout.addWidget(QW.QLabel('Number of bins'))
        main_layout.addWidget(self.bin_slider)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)



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
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)

    # Extract action from scaler
        extract_mask = QW.QAction('Extract mask')
        extract_mask.triggered.connect(self.extractMaskFromScaler)
        extract_mask.setEnabled(self.has_scaler and self.scaler.visible)

        menu.addSeparator()
        menu.addAction(extract_mask)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def onSpanSelect(self, xmin, xmax, fromDragging=True):
        '''
        Slot triggered after interacting with the span selector. Highlights
        in the data viewer the pixels that fall within the selected area in the
        histogram.

        Parameters
        ----------
        xmin : float
            Lower range bound.
        xmax : float
            Upper range bound.
        fromDragging : bool, optional
            Whether this function was called after a dragging action within the
            histogram canvas. The default is True.

        Returns
        -------
        None.

        '''
    # Reset the image norm limits
        self.maps_canvas.update_clim()

    # Check if xmax and xmin are valid before setting the new image norm limits
        xmin, xmax = round(xmin), round(xmax)
        if xmax > xmin and self.maps_canvas.image is not None:
            self.maps_canvas.image.set_clim(xmin, xmax)
        else:
            xmin, xmax = None, None

    # If the event was triggered from a dragging action, also update the vmin
    # and vmax values in line edits (NavTbar)
        if fromDragging:
            if xmin is None and xmax is None:
                self.scaler_vmin.clear()
                self.scaler_vmax.clear()
            else:
                self.scaler_vmin.setText(str(xmin))
                self.scaler_vmax.setText(str(xmax))

    # Redraw both the histogram canvas and the data viewer canvas in any case
        self.canvas.draw()
        self.maps_canvas.draw()


    def setScalerExtents(self):
        '''
        Select a range in the histogram using the vmin and vmax line edits in
        the Navigation Toolbar. This function calls the onSpanSelect function
        without using the histogram spanner (= fromDragging set to False, see
        onSpanSelect function for more details).

        Returns
        -------
        None.

        '''
        if (self.scaler_vmin.hasAcceptableInput() and
            self.scaler_vmax.hasAcceptableInput()):
            vmin = int(self.scaler_vmin.text())
            vmax = int(self.scaler_vmax.text())
            if vmax > vmin:
                self.scaler.set_visible(True)
                self.scaler.extents = (vmin, vmax)
                self.onSpanSelect(vmin, vmax, fromDragging=False)


    def hideScaler(self):
        '''
        Hides the spanner view from the histogram canvas. Since this function
        has no canvas draw() call, it must be triggered before update_canvas().

        Returns
        -------
        None.

        '''
        self.scaler.set_visible(False)


    def setHistBins(self, value):
        '''
        Set the number of bins of the histogram.

        Parameters
        ----------
        value : int
            Number of bins.

        Returns
        -------
        None.

        '''
        self.bin_slider.setToolTip(f'Bins = {value}')
        self.canvas.set_nbins(value)


    def extractMaskFromScaler(self):
        '''
        Extract a mask from the range selected in the histogram scaler and save
        it to file.

        Returns
        -------
        None.

        '''
        vmin, vmax = self.scaler.extents
        vmin, vmax = round(vmin), round(vmax)

    # Extract the displayed array and its current mask (legacy mask)
        array, legacy_mask = self.maps_canvas.get_map(return_mask=True)

    # If the legacy mask exists, intersect it with the new mask
        mask_array = np.logical_or(array < vmin, array > vmax)
        if legacy_mask is not None:
            mask_array = mask_array + legacy_mask
        mask = Mask(mask_array)

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))


# !!! EXPERIMENTAL
    # def resizeEvent(self, e):
    #     '''
    #     Reimplementation of resizeEvent default function. It just temporarily
    #     suppress the udpates of the histogram canvas object, that is very slow
    #     to repaint during the resize events.

    #     Parameters
    #     ----------
    #     e : resizeEvent
    #         The resize event.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     self.canvas.setUpdatesEnabled(False)
    #     e.accept()
    #     self.canvas.setUpdatesEnabled(True)




class ProbabilityMapViewer(QW.QWidget):
    '''
    A widget to visualize the probability map linked with the mineral map that
    is currently displayed in the data viewer.
    '''

    def __init__(self):
        '''
        ProbabilityMapViewer class constructor.

        Returns
        -------
        None.

        '''
        super(ProbabilityMapViewer, self).__init__()

    # Canvas
        self.canvas = plots.ImageCanvas(tight=True)
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar(self.canvas, self)
        self.navTbar.fixHomeAction()
        self.navTbar.removeToolByIndex([3, 4, 8, 9])

    # Adjust layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navTbar)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)


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
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())




class RgbaCompositeMapViewer(QW.QWidget):
    '''
    A widget to visualize an RGB(A) composite map extracted from the
    combination of input maps.
    '''

    def __init__(self):
        '''
        RgbaCompositeMapViewer class constructor.

        Returns
        -------
        None.

        '''
        super(RgbaCompositeMapViewer, self).__init__()

        self.channels = ('R', 'G', 'B', 'A')

    # Canvas
        self.canvas = plots.ImageCanvas(cbar=False, tight=True)
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar(self.canvas, self)
        self.navTbar.fixHomeAction()
        self.navTbar.removeToolByIndex([3, 4, 8, 9])

    # R-G-B-A Path Labels
        self.rgba_labels = [PathLabel(full_display=False) for x in range(4)]
        channels_layout = QW.QGridLayout()

        for col, lbl in enumerate(self.rgba_labels):
            channel_name = QW.QLabel(self.channels[col])
            channel_name.setAlignment(Qt.AlignHCenter)
            channels_layout.addWidget(channel_name, 0, col)
            channels_layout.addWidget(lbl, 1, col)

    # Adjust layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navTbar)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addLayout(channels_layout)
        self.setLayout(main_layout)


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
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)
        menu.addSeparator()

    # Clear channel sub-menu
        clear_submenu = menu.addMenu('Clear channel...')

        for c in self.channels:
            clear_submenu.addAction(f'{c} channel',
                                    lambda c=c: self.clear_channel(c))
        clear_submenu.addSeparator()
        clear_submenu.addAction('Clear all',
                                lambda: self.clear_channel(*self.channels))

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def clear_channel(self, *args):
        if self.canvas.image is not None:
            rgba_map = self.canvas.image.get_array()
            for arg in args:
                # [R=0, G=0, B=0, A=1]
                idx = self.channels.index(arg)
                rgba_map[:, :, idx] = 1 if idx == 3 else 0
                self.rgba_labels[idx].clearPath()

            self.canvas.draw_heatmap(rgba_map, 'RGBA composite map')


    def set_channel(self, channel, inmap):
        '''
        Set one channel's data.

        Parameters
        ----------
        channel : str
            One of 'R', 'G', 'B' and 'A'.
        inmap : InputMap
            Input map to be set as channel <channel>.

        Returns
        -------
        None.

        '''
        assert channel in self.channels

    # Get the data and the filepath from the Input Map
        data, filepath = inmap.map, inmap.filepath

    # Get the current rgba map or build a new one
        if self.canvas.image is None:
            rgba_map = np.zeros((*inmap.shape, 4))
            rgba_map[:, :, 3] = 1   # invert A channel (opacity)
        else:
            rgba_map = self.canvas.image.get_array()

    # Exit function if the Input Map shape does not fit the RGBA map shape
        if inmap.shape != rgba_map.shape[:2]:
            err_txt = 'This map does not fit within the current RGBA map'
            return QW.QMessageBox.critical(self, 'X-Min Learn', err_txt)

    # Update the channel with the new data and update the plot
        idx = self.channels.index(channel)
        rgba_map[:, :, idx] = np.round(data/data.max(), 2)
        self.canvas.draw_heatmap(rgba_map, 'RGBA composite map')

    # Update the channel path label
        self.rgba_labels[idx].setPath(filepath)


    def clear_all(self):
        # Important: clear channel must be called before clear canvas
        self.clear_channel('R', 'G', 'B', 'A')
        self.canvas.clear_canvas()





class ModeViewer(QW.QWidget):
    '''
    A widget to visualize the modal amounts of the mineral classes occurring in
    the mineral map that is currently displayed in the Data Viewer. It includes
    an interactive legend.
    '''

    updateSceneRequested = pyqtSignal(DataObject) # current data object

    def __init__(self, map_canvas, parent=None):
        '''
        ModeViewer class constructor.

        Parameters
        ----------
        map_canvas : ImageCanvas
            The canvas where the mineral map is displayed.

        Returns
        -------
        None.

        '''
        super(ModeViewer, self).__init__(parent)

    # Set principal attributes
        self._current_data_object = None
        self.map_canvas = map_canvas

    # Initialize GUI
        self._init_ui()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self):
        '''
        ModeViewer class GUI constructor.

        Returns
        -------
        None.

        '''

    # Canvas
        self.canvas = plots.BarCanvas(orientation='h', size=(3.6, 6.4),
                                      tight=True, wheelPanEnabled=False,
                                      wheelZoomEnabled=False)
        self.canvas.setMinimumSize(200, 200)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar(self.canvas, self, coords=False)
        self.navTbar.removeToolByIndex(list(range(2,10)))

    # Show Labels Action in Navigation Toolbar (Show class %)
        self.showLabels_action = QW.QAction(QIcon('Icons/labelize.png'),
                                         'Show Amounts', self.navTbar)
        self.showLabels_action.setCheckable(True)
        self.navTbar.insertAction(self.navTbar.findChildren(QW.QAction)[10],
                                  self.showLabels_action)

    # Interactive legend
        self.legend = Legend(interactive=True)

    # Adjust layout
        mode_vbox = QW.QVBoxLayout()
        mode_vbox.addWidget(self.navTbar)
        mode_vbox.addWidget(self.canvas)

        main_layout = SplitterLayout(Qt.Vertical)
        main_layout.addLayout(mode_vbox, -1)
        main_layout.addWidget(self.legend, 1)
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        ModeViewer class signals-slots connector.

        Returns
        -------
        None.

        '''
    # Show context menu on the mode canvas after a right-click event
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Show classes amounts in the mode canvas
        self.showLabels_action.toggled.connect(self.canvas.show_amounts)

    # Connect legend signals
        self.legend.colorChangeRequested.connect(self.onColorChanged)
        self.legend.randomPaletteRequested.connect(self.onPaletteRandomized)
        self.legend.itemRenameRequested.connect(self.onClassRenamed)
        self.legend.itemsMergeRequested.connect(self.onClassMerged)
        self.legend.maskExtractionRequested.connect(self.onMaskExtracted)


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
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def _update_mode_canvas(self, minmap, title=''):
        '''
        Update the bar canvas that displays the mode data.

        Parameters
        ----------
        minmap : MineralMap
            The mineral map whose mode data must be displayed.
        title : str, optional
            The title of the bar plot. The default is ''.

        Returns
        -------
        None.

        '''
        title = self.canvas.ax.get_title() if title == '' else title
        mode_lbl, mode = zip(*minmap.get_labeled_mode().items())
        mode_col = (minmap.get_phase_color(lbl) for lbl in mode_lbl)
        self.canvas.update_canvas(mode, mode_lbl, title, mode_col)


    def update(self, data_object, title=''):
        '''
        Update all components of the ModeViewer.

        Parameters
        ----------
        data_object : DataObject
            Data object that contains a mineral map.
        title : str, optional
            The title of the mode bar plot. The default is ''.

        Returns
        -------
        None.

        '''
    # Set current data object
        if not data_object.holdsMineralMap(): return # safety
        self._current_data_object = data_object

    # Update the mode canvas
        minmap = data_object.get('data')
        self._update_mode_canvas(minmap, title)

    # Update the legend
        self.legend.update(minmap)


    def clear_all(self):
        '''
        Reset all components of the ModeViewer.

        Returns
        -------
        None.

        '''
        self._current_data_object = None
        self.canvas.clear_canvas()
        self.legend.clear()


    def onColorChanged(self, legend_item, color):
        '''
        Alter the displayed color of a class. This function propagates the
        changes to the mineral map, the map canvas, the mode bar plot and the
        legend. It also sets the linked data object as edited. The arguments of
        this function are specifically compatible with the colorChangeRequested
        signal emitted by the legend (see Legend object for more details).

        Parameters
        ----------
        legend_item : QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple
            RGB triplet. If empty, a random color is generated.

        Returns
        -------
        None.

        '''
    # Extract mineral map data
        minmap = self._current_data_object.get('data')

    # Apply the color change to mineral map (if color is empty, randomize it)
        if not len(color): color = minmap.rand_colorlist(1)[0]
        phase_name = legend_item.text(1)
        minmap.set_phase_color(phase_name, color)

    # Update the map canvas colormap
        self.map_canvas.alter_cmap(minmap.palette.values())

    # Update the mode canvas
        self._update_mode_canvas(minmap)

    # Update the legend
        self.legend.changeItemColor(legend_item, color)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onPaletteRandomized(self):
        '''
        Randomize the palette of the mineral map. This function propagates the
        changes to the mineral map, the map canvas, the mode bar plot and the
        legend. It also sets the linked data object as edited.

        Returns
        -------
        None.

        '''
    # Extract mineral map data
        minmap = self._current_data_object.get('data')

    # Apply random palette to mineral map
        rand_palette = minmap.rand_colorlist()
        minmap.set_palette(rand_palette)

    # Update the image canvas colormap
        self.map_canvas.alter_cmap(rand_palette)

    # Update the mode canvas
        self._update_mode_canvas(minmap)

    # Update the legend
        self.legend.update(minmap)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onClassRenamed(self, legend_item, new_name):
        '''
        Rename a class. This function propagates the changes to the mineral
        map, the map canvas, the mode bar plot and the legend. It also sets the
        linked data object as edited. The arguments of this function are
        specifically compatible with the itemRenameRequested signal emitted by
        the legend (see Legend object for more details).

        Parameters
        ----------
        legend_item : QTreeWidgetItem
            The legend item that requested to be renamed.
        new_name : str
            New class name.

        Returns
        -------
        None.

        '''
    # Get mineral map data
        minmap = self._current_data_object.get('data')

    # Rename the phase in the mineral map. Exit func if name is already taken)
        old_name = legend_item.text(1)
        if new_name not in minmap.get_phases():
            minmap.rename_phase(old_name, new_name)
        else:
            return QW.QMessageBox.critical(self, 'X-Min Learn',
                                           f'{new_name} is already taken.')

    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # # Update the image canvas
    #     mmap, enc, col = minmap.get_plotData()
    #     self.map_canvas.draw_discretemap(mmap, enc, col)

    # # Update the mode canvas
    #     self._update_mode_canvas(minmap)

    # # Update the legend
    #     self.legend.rename_class(legend_item, new_name)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onClassMerged(self, classes, new_name):
    # Get mineral map data
        minmap = self._current_data_object.get('data')

    # Merge phases in the mineral map
        minmap.merge_phases(classes, new_name)

    # # Update the image canvas
    #     mmap, enc, col = minmap.get_plotData()
    #     self.map_canvas.draw_discretemap(mmap, enc, col)

    # # Update the mode canvas
    #     self._update_mode_canvas(minmap)

    # # Update the legend
    #     self.legend.update(legend_item, new_name)

    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onMaskExtracted(self, classes):
        '''
        Extract a mask from a selection of mineral classes and save it to file.

        Parameters
        ----------
        classes : list
            Selected mineral classes.

        Returns
        -------
        None.

        '''
    # Extract the mask
        minmap = self._current_data_object.get('data')
        mask = ~np.isin(minmap.minmap, classes)

    # If a legacy mask exists, intersect it with the new mask
        _, legacy_mask = self.map_canvas.get_map(return_mask=True)
        if legacy_mask is not None:
            mask = mask + legacy_mask

    # Create a new Mask object
        mask = Mask(mask)

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', dirname(outpath))
            try:
                mask.save(outpath)
                return QW.QMessageBox.information(self, 'X-Min Learn',
                                                  'File saved with success')
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))




class RoiSelector(QW.QWidget):
    '''
    A widget to build, load, edit and save RoiMap objects interactively.
    '''

    rectangleSelectorUpdated = pyqtSignal()

    def __init__(self, maps_canvas, parent=None):
        '''
        RoiSelector class constructor.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas where ROIs should be drawn and displayed.
        parent : QtWidget or None, optional
            GUI parent widget of the ROI selector. The default is None.

        Returns
        -------
        None.

        '''
        super(RoiSelector, self).__init__(parent)

    # Define main attributes
        self.canvas = maps_canvas
        self.current_selection = None
        self.current_roimap = None
        self.patches = []

    # Load ROIs visual properties
        self.roi_color = pref.get_setting('class/trAreasCol', (255,0,0), tuple)
        self.roi_selcolor = pref.get_setting('class/trAreasSel', (0,0,255), tuple)
        self.roi_filled = pref.get_setting('class/trAreasFill', False, bool)

    # Initialize GUI
        self._init_ui()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self):
        '''
        RoiSelector class GUI constructor.

        Returns
        -------
        None.

        '''
    # Toolbar
        self.toolbar = QW.QToolBar('ROI toolbar', self)
        self.toolbar.setStyleSheet(pref.SS_toolbar)

    # Load ROI map [-> Toolbar Action]
        self.load_action = QW.QAction(QIcon('Icons/load.png'),
                                      'Load ROI map', self.toolbar)

    # Save ROI map [-> Toolbar Action]
        self.save_action = QW.QAction(QIcon('Icons/save.png'),
                                      'Save ROI map', self.toolbar)

    # Save ROI map as... [-> Toolbar Action]
        self.saveas_action = QW.QAction(QIcon('Icons/saveEdit.png'),
                                        'Save ROI map as...', self.toolbar)

    # ROI selector [-> Toolbar Action]
        self.rectSel = RectSel(self.canvas.ax, self.onRectSelect, btns=[1])
        self.draw_action = QW.QAction(QIcon('Icons/ROI_selection.png'),
                                      'Select ROI', self.toolbar)
        self.draw_action.setCheckable(True)

    # Add ROI [-> Toolbar Action]
        self.addroi_action = QW.QAction(QIcon('Icons/generic_add_black.png'),
                                     'Add ROI', self.toolbar)
        self.addroi_action.setEnabled(False)

    # Extract mask [-> Toolbar Action]
        # TODO add mask icon
        self.extr_mask_action = QW.QAction('Extract mask', self.toolbar)
        self.extr_mask_action.setEnabled(False)

    # ROI visual preferences [-> StyledMenu -> Toolbar Action]
        prefMenu = StyledMenu()

        self.roicolor_action = QW.QAction('Set ROIs color...', prefMenu)

        self.roiselcolor_action = QW.QAction('Set ROIs selection color...',
                                             prefMenu)

        self.roifilled_action = QW.QAction('Filled ROIs', prefMenu)
        self.roifilled_action.setCheckable(True)
        self.roifilled_action.setChecked(self.roi_filled)

        prefMenu.addAction(self.roicolor_action)
        prefMenu.addAction(self.roiselcolor_action)
        prefMenu.addAction(self.roifilled_action)

        self.pref_action = QW.QAction(QIcon('Icons/gear.png'), 'Preferences',
                                      self.toolbar)
        self.pref_action.setMenu(prefMenu)

    # Add actions to the toolbar
        self.toolbar.addActions((self.load_action, self.save_action,
                                 self.saveas_action))
        self.toolbar.addSeparator()
        self.toolbar.addActions((self.draw_action, self.addroi_action,
                                 self.extr_mask_action, self.pref_action))

    # Loaded ROI map path (Path Label)
        self.mappath = PathLabel(full_display=False)

    # Hide ROI map (Checkable Styled Button)
        self.hideroi_btn = StyledButton(QIcon('Icons/not_visible.png'))
        self.hideroi_btn.setCheckable(True)
        self.hideroi_btn.setToolTip('Hide ROI map')

    # Remove (unload) ROI map (Styled Button)
        self.unload_btn = StyledButton(QIcon(self.style().standardIcon(QW.QStyle.SP_DialogCloseButton))) # TODO icon
        self.unload_btn.setToolTip('Remove ROI map')

    # Remove ROI button [-> Corner table widget]
        self.delroi_btn = StyledButton(QIcon('Icons/generic_del.png'))

    # Roi table
        self.table = StyledTable(0, 2)
        self.table.setSelectionBehavior(QW.QAbstractItemView.SelectRows)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.table.setHorizontalHeaderLabels(['Class', 'Pixel Count'])
        self.table.horizontalHeader().setSectionResizeMode(1) # Stretch
        self.table.verticalHeader().setSectionResizeMode(3) # ResizeToContent
        self.table.setCornerWidget(self.delroi_btn)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

    # Bar plot canvas
        self.barCanvas = plots.BarCanvas(size=(3.6, 2.4), tight=True,
                                         wheelZoomEnabled=False,
                                         wheelPanEnabled=False)
        self.barCanvas.setMinimumSize(300, 300)

    # Bar plot Navigation toolbar
        self.navTbar = plots.NavTbar(self.barCanvas, self, coords=False)
        self.navTbar.removeToolByIndex(list(range(2,10)))

    # Show labels action (show class %) [-> Navigation Toolbar]
        self.showlabels_action = QW.QAction(QIcon('Icons/labelize.png'),
                                            'Show Amounts', self.navTbar)
        self.showlabels_action.setCheckable(True)
        self.navTbar.insertAction(self.navTbar.findChildren(QW.QAction)[10],
                                  self.showlabels_action)

    # Adjust Layout
        top_layout = QW.QGridLayout()
        top_layout.addWidget(self.toolbar, 0, 0, 1, -1)
        top_layout.addWidget(self.mappath, 1, 0)
        top_layout.addWidget(self.hideroi_btn, 1, 1)
        top_layout.addWidget(self.unload_btn, 1, 2)
        top_layout.addWidget(self.table, 2, 0, 1, -1)
        top_layout.setRowStretch(2, 1)
        top_layout.setColumnStretch(0, 1)

        bot_layout = QW.QVBoxLayout()
        bot_layout.addWidget(self.navTbar)
        bot_layout.addWidget(self.barCanvas, 1)

        main_layout = SplitterLayout(Qt.Vertical)
        main_layout.addLayouts((top_layout, bot_layout))
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        RoiSelector class signals-slots connector.

        Returns
        -------
        None.

        '''
    # Load ROI map from file
        self.load_action.triggered.connect(self.loadRoiMap)

    # Save ROI map to file
        self.save_action.triggered.connect(self.saveRoiMap)
        self.saveas_action.triggered.connect(lambda: self.saveRoiMap(True))

    # Toggle on/off the Rectangle Selector
        self.draw_action.toggled.connect(self.toggleRectSelect)

    # Add a new ROI
        self.addroi_action.triggered.connect(self.addRoi)

    # Extract mask from current selection
        self.extr_mask_action.triggered.connect(self.extractMaskFromSelection)

    # Set ROI color, selection color and filled property
        self.roicolor_action.triggered.connect(self.setRoiColor)
        self.roiselcolor_action.triggered.connect(self.setRoiSelectionColor)
        self.roifilled_action.toggled.connect(self.setRoiFilled)

    # Hide/show ROIs on canvas
        self.hideroi_btn.toggled.connect(self.setRoiMapHidden)

    # Remove loaded ROI map
        self.unload_btn.clicked.connect(self.unloadRoiMap)

    # Remove ROI(s)
        self.delroi_btn.clicked.connect(self.removeRoi)

    # Connect table signals (ROI selected & ROI name edited)
        self.table.itemSelectionChanged.connect(self.updatePatchSelection)
        self.table.itemChanged.connect(self.editRoiName)

    # Show percentage amounts in the bar plot
        self.showlabels_action.toggled.connect(self.barCanvas.show_amounts)

    # Show custom context menu when right-clicking on the table
        self.table.customContextMenuRequested.connect(
            self.showTableContextMenu)

    # Show custom context menu when right-clicking on the bar canvas
        self.barCanvas.customContextMenuRequested.connect(
            self.showCanvasContextMenu)


    @property
    def selectedTableIndices(self):
        '''
        Get a list of the indices of the selected ROIs in the ROIs table.

        Returns
        -------
        selectedIndices : list
            Selected indices.

        '''
        selectedItems = self.table.selectedItems()
    # List slicing to avoid rows idx repetitions (there are 2 columns in table)
        selectedIndices = [item.row() for item in selectedItems[::2]]
        return selectedIndices


    def _redraw(self):
        '''
        Redraw the canvas and update the cursor of the rectangle selector.

        Returns
        -------
        None.

        '''
        self.canvas.draw_idle()
        self.rectSel.updateCursor()


    def updateBarPlot(self):
        '''
        Update the bar canvas that displays the cumulative pixel count for each
        drawn ROI.

        Returns
        -------
        None.

        '''
        if self.current_roimap is not None:
            names, counts = zip(*self.current_roimap.class_count.items())
            self.barCanvas.update_canvas(counts, names, 'ROIs Counter')
        else:
            self.barCanvas.clear_canvas()


    def showTableContextMenu(self, point):
        '''
        Shows a context menu when right-clicking on the ROI table.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        Returns
        -------
        None.

        '''

    # Exit function when clicking outside any item
        if self.table.itemAt(point) is None: return

        menu = StyledMenu()

    # Extract mask from selected ROIs
        extract_mask = QW.QAction('Extract mask')
        extract_mask.triggered.connect(self.extractMaskFromRois)

    # Remove selected ROIs
        remove = QW.QAction(QIcon('Icons/generic_del.png'), 'Remove')
        remove.triggered.connect(self.removeRoi)

    # Add actions to menu
        menu.addAction(extract_mask)
        menu.addSeparator()
        menu.addAction(remove)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def showCanvasContextMenu(self, point):
        '''
        Shows a context menu when right-clicking on the bar plot.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        Returns
        -------
        None.

        '''
    # Get context menu from NavTbar actions
        menu = self.barCanvas.get_navigation_context_menu(self.navTbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def onRectSelect(self, eclick, erelease):
        '''
        Callback function for the rectangle selector. It is triggered when
        selection is performed by the user (left mouse button click-release).

        Parameters
        ----------
        eclick : Matplotlib MouseEvent
            Mouse click event.
        erelease : Matplotlib MouseEvent
            Mouse release event.

        Returns
        -------
        None.

        '''
        image = self.canvas.image
        if image is not None:

        # Save in memory the current selection
            map_shape = image.get_array().shape
            self.current_selection = self.rectSel.fixed_extents(map_shape)

        # Send the signal to inform that the selector has been updated
            self.rectangleSelectorUpdated.emit()


    def selectRoi(self, event):
        '''
        Callback function for picking events that can be triggered when the
        rectangle selector is active.

        Parameters
        ----------
        event : Matplotlib Pick event
            The picking event triggered by left-clicking on a ROI patch.

        Returns
        -------
        None.

        '''
        if event.mouseevent.button == MouseButton.LEFT:
            patch = event.artist
            _, patches = zip(*self.patches)
        # Select the patch in the table. This will therefore trigger the
        # UpdatePatchSelection slot
            if patch in patches:
                idx = patches.index(patch)
                self.table.selectRow(idx)


    def toggleRectSelect(self, toggled):
        '''
        Toggle on/off the rectangle selector.

        Parameters
        ----------
        toggled : bool
            Whether the rectangle selector should be toggled.

        Returns
        -------
        None.

        '''
    # Create a mpl picking event connection if the rectangle is toggled on,
    # otherwise delete it.
        if toggled:
            self.pcid = self.canvas.mpl_connect('pick_event', self.selectRoi)
        else:
            self.canvas.mpl_disconnect(self.pcid)

    # Show/hide the rectangle selector
        self.rectSel.set_active(toggled)
        self.rectSel.set_visible(toggled)
        self.rectSel.update()

    # Enable/disable the 'add Roi' and the 'extract mask' actions
        self.addroi_action.setEnabled(toggled)
        self.extr_mask_action.setEnabled(toggled)

    # When the rectangle is activated, the last selection is displayed. So we
    # need to send the signal to inform that the selector has been updated
        self.rectangleSelectorUpdated.emit()


    def updatePatchSelection(self):
        '''
        Redraw ROI patches on canvas with a different color based on their
        selection state. This function is called whenever a new ROI selection
        is performed or when a new ROI color or selection color is set. This
        function also redraws the canvas.

        Returns
        -------
        None.

        '''
        selected = self.selectedTableIndices
        for idx, (_, patch) in enumerate(self.patches):
            color = self.roi_selcolor if idx in selected else self.roi_color
            patch.set(color=CF.RGB2float([color]), lw=2+2*(idx in selected))
        self._redraw()


    def editRoiName(self, item):
        '''
        Rename ROI. This function is triggered by an itemChanged signal from
        the ROIs table.

        Parameters
        ----------
        item : QTableWidgetItem
            The table item that was edited.

        Returns
        -------
        None.

        '''
        idx = item.row()
        name = item.text()
    # Update the ROI map
        self.current_roimap.renameRoi(idx, name)
    # Update the patch
        self.editPatchAnnotation(idx, name)
    # Refresh view
        self.updateBarPlot()
        self._redraw()


    def addPatchToCanvas(self, name, bbox):
        '''
        Add a new ROI to canvas as a new patch (Rectangle) and its linked
        annotation. It does not redraw the canvas.

        Parameters
        ----------
        name : str
            The name of the ROI, displayed as annotation.
        bbox : tuple or list
            The bounding box of the ROI -> (x0, y0, width, height) where x0, y0
            are the coordinates of the top-left corner.

        Returns
        -------
        None.

        '''
    # Display the rectangle in canvas
        color = CF.RGB2float([self.roi_color])
        patch = RoiPatch(bbox, color, self.roi_filled)
        patch.set_picker(True)
        self.canvas.ax.add_patch(patch)

    # Display the text annotation in canvas
        text = RoiAnnotation(name, patch)
        self.canvas.ax.add_artist(text)

    # Set annotation and patch visibility
        visible = not self.hideroi_btn.isChecked()
        text.set_visible(visible)
        patch.set_visible(visible)

    # Store the annotation and the patch
        self.patches.append((text, patch))


    def editPatchAnnotation(self, index, text):
        '''
        Change text of a ROI patch annotation.

        Parameters
        ----------
        index : int
            The patch index in self.patches.
        text : str
            The new annotation text.

        Returns
        -------
        None.

        '''
        annotation = self.patches[index][0]
        annotation.set_text(text)


    def removePatchFromCanvas(self, index):
        '''
        Remove ROI patch from canvas and its linked annotation. It does not
        redraw the canvas.

        Parameters
        ----------
        index : int
            Index of the ROI that must be removed.

        Returns
        -------
        None.

        '''
        if len(self.patches):
            text, patch = self.patches.pop(index)
            text.remove()
            patch.remove()


    def getColorFromDialog(self, old_color):
        '''
        Show a dialog to interactively select a color.

        Parameters
        ----------
        old_color : tuple
            The dialog defaults to this color. Must be provided as RGB triplet.

        Returns
        -------
        rgb : tuple
            Selected color as RGB triplet.

        '''
        rgb = False
        col = QW.QColorDialog.getColor(initial=QColor(*old_color))
        if col.isValid():
            rgb = tuple(col.getRgb()[:-1])
        return rgb


    def setRoiColor(self):
        '''
        Set the color of the ROIs borders when they are not selected.

        Returns
        -------
        None.

        '''
        rgb = self.getColorFromDialog(self.roi_color)
        if rgb:
            pref.edit_setting('class/trAreasCol', rgb)
            self.roi_color = rgb
            self.updatePatchSelection()


    def setRoiSelectionColor(self):
        '''
        Set the color of the ROIs borders when they are selected.

        Returns
        -------
        None.

        '''
        rgb = self.getColorFromDialog(self.roi_selcolor)
        if rgb:
            pref.edit_setting('class/trAreasSel', rgb)
            self.roi_selcolor = rgb
            self.updatePatchSelection()


    def setRoiFilled(self, filled):
        '''
        Set if the ROIs should be filled or unfilled. The filling color is the
        same as the ROIs border color (selected or unselected).

        Parameters
        ----------
        filled : bool
            Whether the ROIs should be filled.

        Returns
        -------
        None.

        '''
        self.roi_filled = filled
        pref.edit_setting('class/trAreasFill', filled)
        for _, patch in self.patches:
            patch.set_fill(filled)
        self._redraw()


    def addRoi(self):
        '''
        Wrapper function to easily add a new ROI with the extents of the
        currently selected rectangle. The function also redraws the canvas.

        Returns
        -------
        None.

        '''
    # Exit function if canvas is empty
        image = self.canvas.image
        if image is None: return

    # Get ROI bbox. If bbox is invalid (=None) exit function.
        map_array = image.get_array()
        map_shape = map_array.shape
        bbox = self.rectSel.fixed_rect_bbox(map_shape)
        if bbox is None: return

    # If no ROI map is loaded, then create a new one.
        if self.current_roimap is None:
            self.current_roimap = RoiMap.fromShape(map_shape)
            self.mappath.setPath('*Unsaved ROI map')

    # Send a warning if a ROI map is loaded and has different shape of the map
    # currently displayed in the canvas.
        elif self.current_roimap.shape != map_shape:
            warn_text = 'Warning: different map shapes detected. Drawing ROIs '\
                        'on top of different sized maps leads to unpredictable '\
                        'behaviours. Proceed anyway?'
            btns = QW.QMessageBox.Yes | QW.QMessageBox.No
            choice = QW.QMessageBox.warning(self, 'X-Min Learn', warn_text,
                                            btns, QW.QMessageBox.No)
        # Exit function if user does not want to procede
            if choice == QW.QMessageBox.No: return

    # Show the dialog to type the ROI name
        name, ok = QW.QInputDialog.getText(self, 'X-Min Learn',
                                           'Type name (max 8 ASCII characters)')
    # Proceed only if the new name is an ASCII <= 8 characters string
        if ok and 0 < len(name) < 9 and name.isascii():
            area = self.current_roimap.bboxArea(bbox)
        # Add to roimap
            self.current_roimap.addRoi(name, bbox)
        # Add to table
            self.addRoiToTable(name, area)
        # Add to patches list and canvas
            self.addPatchToCanvas(name, bbox)
        # Refresh view
            self.updateBarPlot()
            self._redraw()


    def addRoiToTable(self, name, pixel_count):
        '''
        Append ROI to the ROIs table as a new row.

        Parameters
        ----------
        name : str
            ROI name.
        pixel_count : int
            ROI area in pixels.

        Returns
        -------
        None.

        '''
    # Define a new table entry (2 TableWidgetItems, one for each table column)
        i0 = QW.QTableWidgetItem(name)
        i1 = QW.QTableWidgetItem(str(pixel_count))
        i1.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) # not editable
    # Append new row to the table
        new_row = self.table.rowCount()
        self.table.insertRow(new_row)
    # Populate columns with the respective TableWidgetItem (name, pixel_count).
    # Block temporarily the table signals to avoid triggering itemChanged().
        self.table.blockSignals(True)
        self.table.setItem(new_row, 0, i0)
        self.table.setItem(new_row, 1, i1)
        self.table.blockSignals(False)


    def setRoiMapHidden(self, hidden):
        '''
        Show/hide all ROIs displayed in the canvas. The function also redraws
        the canvas.

        Parameters
        ----------
        hidden : bool
            Whether the ROIs should be hidden.

        Returns
        -------
        None.

        '''
        for text, patch in self.patches:
            text.set_visible(not hidden)
            patch.set_visible(not hidden)
        self._redraw()


    def extractMaskFromSelection(self):
        '''
        Extract and save a mask from current selection.

        Returns
        -------
        None.

        '''
    # Exit function if image is empty
        if self.canvas.image is None: return

    # Extract the displayed array shape and its current mask (legacy mask)
        array, legacy_mask = self.canvas.get_map(return_mask=True)
        shape = array.shape

    # Exit function if there is no valid selection
        extents = self.rectSel.fixed_extents(shape, fmt='xy') # x0,x1, y0,y1
        if extents is None: return

    # Initialize a new Mask of 1's and invert it to draw 'holes' using extents
        mask = Mask.fromShape(shape, fillwith=1)
        mask.invertRegion(extents)

    # If the legacy mask exists, intersect it with the new mask
        if legacy_mask is not None:
            mask.mask = mask.mask + legacy_mask

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', dirname(outpath))
            try:
                mask.save(outpath)
                return QW.QMessageBox.information(self, 'X-Min Learn',
                                                  'File saved with success')
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))


    def extractMaskFromRois(self):
        '''
        Extract and save a mask from selected ROIs.

        Returns
        -------
        None.

        '''
    # Get the indices of the selected ROIs. Exit function if no ROI is selected
        selected = self.selectedTableIndices
        if not len(selected): return

    # Initialize a new Mask of 1's with the shape of the current ROI map
        shape = self.current_roimap.shape
        mask = Mask.fromShape(shape, fillwith=1)

    # Use the extents of the selected ROIs to draw 'holes' (0's) on the mask
        for idx in selected:
            roi_bbox = self.current_roimap.rois[idx][1]
            extents = self.current_roimap.bboxToExtents(roi_bbox)
            mask.invertRegion(extents)

    # If there is a loaded image that has the same shape of the current ROI map
    # and it has a legacy mask, intersect it with the new mask
        if self.canvas.image is not None:
            array, legacy_mask = self.canvas.get_map(return_mask=True)
            if array.shape == shape and legacy_mask is not None:
                mask.mask = mask.mask + legacy_mask

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', dirname(outpath))
            try:
                mask.save(outpath)
                return QW.QMessageBox.information(self, 'X-Min Learn',
                                                  'File saved with success')
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))

    def removeRoi(self):
        '''
        Wrapper function to easily remove selected ROIs. This function requires
        a confirm from the user. The function also redraws the canvas.

        Returns
        -------
        None.

        '''
    # Get the indices of the selected ROIs. Exit function if no ROI is selected
        selected = self.selectedTableIndices
        if not len(selected): return

    # Ask for confirmation
        btns = QW.QMessageBox.Yes | QW.QMessageBox.No
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Remove selected ROI?',
                                         btns, QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:

            for row in sorted(selected, reverse=True):
            # Remove from roimap
                self.current_roimap.delRoi(row)
            # Remove from table
                self.table.removeRow(row)
            # Remove from patches list and canvas
                self.removePatchFromCanvas(row)
        # Refresh view
            self.updateBarPlot()
            self._redraw()


    def removeCurrentRoiMap(self):
        '''
        Remove the current ROI map, reset the ROIs table and all the patches
        from the canvas. The function does not redraw the canvas.

        Returns
        -------
        None.

        '''
        self.current_roimap = None
        self.mappath.clear()
        self.table.setRowCount(0)
    # Remove all patches from canvas
        for idx in reversed(range(len(self.patches))):
            self.removePatchFromCanvas(idx)


    def loadRoiMap(self):
        '''
        Wrapper function to easily load a new ROI map. If a previous ROI map
        exists, this function removes it, after user confirm. The function also
        redraws the canvas.

        Returns
        -------
        None.

        '''
    # Show a warning if a ROI map was already loaded
        if self.current_roimap is not None:
            warn_text = 'Loading a new ROI map will discard any unsaved changes '\
                        'made to the current ROI map. Proceed anyway?'
            choice = QW.QMessageBox.warning(self, 'X-Min Learn', warn_text,
                                            QW.QMessageBox.Yes | QW.QMessageBox.No,
                                            QW.QMessageBox.No)
        # Exit function if user does not want to procede
            if choice == QW.QMessageBox.No: return

    # Get new ROI map filepath
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Load ROI map',
                                                 pref.get_dirPath('in'),
                                                 'ROI maps (*.rmp)')
        if path:
            pref.set_dirPath('in', dirname(path))
            progbar = PopUpProgBar(self, 4, 'Loading data', cancel=False)
            progbar.setValue(0)

        # Remove old (current) ROI map
            self.removeCurrentRoiMap()
            progbar.increase()

        # Load new ROI map
            try:
                self.current_roimap = RoiMap.load(path)
                self.mappath.setPath(path)
                progbar.increase()
            except Exception as e:
                progbar.reset()
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  f'Unexpected file:\n{path}',
                                  detailedText = repr(e))

        # Populate the canvas and the ROIs table with the loaded ROIs
            for name, bbox in self.current_roimap.rois:
                area = self.current_roimap.bboxArea(bbox)
                self.addRoiToTable(name, area)
                self.addPatchToCanvas(name, bbox)
            progbar.increase()

        # Refresh view
            self.updateBarPlot()
            self._redraw()
            progbar.increase()


    def unloadRoiMap(self):
        '''
        Wrapper function to easily remove from memory the currently loaded ROI
        map, if it exists. This function requires a confirm from the user. The
        function also redraws the canvas.

        Returns
        -------
        None.

        '''
        if self.current_roimap is not None:
            choice = QW.QMessageBox.warning(self, 'X-Min Learn',
                                            'Remove current ROI map? Any '\
                                            'unsaved change will be discarded.',
                                             QW.QMessageBox.Yes |
                                             QW.QMessageBox.No,
                                             QW.QMessageBox.No)
            if choice == QW.QMessageBox.Yes:
                self.removeCurrentRoiMap()
                self.updateBarPlot()
                self._redraw()


    def saveRoiMap(self, saveAs=False):
        '''
        Save the current ROI map to file. If the file already exists, it will
        be overwritten. ROI map can still be saved as a new file if <saveAs> is
        True.

        Parameters
        ----------
        saveAs : bool, optional
            Whether the ROI map should be saved to a new file. The default is
            False.

        Returns
        -------
        None.

        '''
    # Exit function if the current ROI map does not exist
        if self.current_roimap is None: return

    # Save the ROI map to a new file if it was requested (saveAs = True) and/or
    # if it was never saved before (= it has not a valid filepath). Otherwise,
    # save it to its current filepath (overwrite).
        if not saveAs and (path := self.current_roimap.filepath) is not None:
            outpath = path
        else:
            outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save ROI map',
                                                        pref.get_dirPath('out'),
                                                        '''ROI map file (*.rmp)''')
        if outpath:
            pref.set_dirPath('out', dirname(outpath))
            try:
                self.current_roimap.save(outpath)
                self.mappath.setPath(outpath)
                return QW.QMessageBox.information(self, 'X-Min Learn',
                                                  'File saved with success')
            except Exception as e:
                return RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                  'An error occurred while saving the file',
                                  detailedText = repr(e))




















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



class PolySel(PolygonSelector): # TODO future improvement to ROIs

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
    A rectangle selector widget customized to be interactive and to support
    pre-defined callback functions.
    '''
    def __init__(self, ax, onselect, useblit=True, btns=None):
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
            default is None.

        Returns
        -------
        None.

        '''
    # Customize the appearence of the rectangle selector and of its handles
        rect_props = dict(fc=pref.BLACK_PEARL, ec=pref.BLACK_PEARL, alpha=0.8,
                          fill=True)
        handle_props = dict(mfc=pref.SAN_MARINO, mec=pref.BLACK_PEARL,
                            alpha=1)

    # drag_from_anywhere=True causes a rendering glitch when a former selection
    # is active, you draw a new one and, without releasing the left button, you
    # start resizing it.
        kwargs = {'minspanx': 1,
                  'minspany': 1,
                  'useblit': useblit,
                  'props': rect_props,
                  'spancoords': 'data',
                  'button': btns,
                  'grab_range': 10,
                  'handle_props': handle_props,
                  'interactive': True,
                  'drag_from_anywhere': False}

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
    A Span Selector widget to scale heatmaps interactively to a min-max range.
    <select_ax> -> ax from wich to select the range.
    <heatmap> -> HeatMapCanvas object
    '''
    def __init__(self, select_ax, heatmap, useblit=True):
        super(HeatmapScaler, self).__init__(select_ax, self.onselect, 'horizontal',
                                            minspan=1, useblit=useblit, interactive=True,
                                            button=MouseButton.LEFT)
        self.select_canvas = select_ax.figure.canvas
        self.target_canvas = heatmap
        self.set_active(True)

    def onselect(self, xmin, xmax):
        self.target_canvas.update_clim(None)
        if xmax - xmin > 1:
            self.target_canvas.img.set_clim(round(xmin), round(xmax))

        self.select_canvas.draw()
        self.target_canvas.draw()

    def hide(self):
        self.set_visible(False)
        self.select_canvas.draw()





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





