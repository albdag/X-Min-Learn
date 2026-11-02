# -*- coding: utf-8 -*-
"""
Created on Tue Mar  26 11:25:14 2024

@author: albdag
"""

from collections.abc import Collection
import json
import os
from typing import Literal, cast, get_args

from PyQt5.QtCore import pyqtSignal, QMimeData, QPoint, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5 import QtWidgets as QW

import numpy as np

from base import *
import config_schemas
import convenient_functions as cf
import custom_widgets as CW
import image_analysis_tools as iatools
import dialogs
import plots
import settings
import style


class Pane(QW.QDockWidget):

    def __init__(
        self,
        widget: QW.QWidget,
        title: str = '',
        icon: QIcon | None = None,
        scroll: bool = True,
        tip: str = ''
    ) -> None:
        '''
        The base class for every pane of X-Min Learn.

        Parameters
        ----------
        widget : QWidget
            The widget to be displayed in the pane.
        title : str, optional
            The title of the pane. The default is ''.
        icon : QIcon, optional
            The icon of the pane, that is displayed in the panes toolbar. The
            default is None.
        scroll : bool, optional
            Whether or not the pane should be scrollable. The default is True.
        tip : str, optional
            A short pane description to be displayed in the status bar of the
            main window. The default is ''.

        '''
        super().__init__()
    
    # Set widget properties
        self.setWindowTitle(title)
        self.setAllowedAreas(Qt.DockWidgetArea(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        ))
    
    # Set main attributes
        self.title = title
        self.icon = icon
        self.tip = tip

    # Wrap the widget in a scroll area if requested
        if scroll:
            scroll_area = CW.GroupScrollArea(widget)
            scroll_area.setObjectName('PaneScrollArea') # Name for custom qss
            self.setWidget(scroll_area)
        else:
            self.setWidget(widget)

    # Set the style-sheet 
        self.setStyleSheet(style.SS_PANE)


    def widget(self) -> QW.QWidget:
        '''
        Reimplemented 'widget' method that ensures the actual pane's widget
        is returned rather than the scroll area that wraps it.

        Returns
        -------
        QWidget
            The widget of the pane.

        '''
        widget: CW.GroupScrollArea | QW.QWidget = super().widget()
        if isinstance(widget, CW.GroupScrollArea):
            return widget.wrappedObject()
        return widget



type DataManagerWidgetItem = CW.DataGroup | CW.DataSubGroup | CW.DataObject
class DataManager(QW.QTreeWidget):

    clearSceneRequested = pyqtSignal()
    updateSceneRequested = pyqtSignal(object)
    dataModified = pyqtSignal()
    rgbaChannelSet = pyqtSignal(str)
    
    PALETTE_MIMETYPE = 'application/x-palette+json'

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A widget for loading, accessing and managing input and output data.
        
        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        RuntimeError
            Raised if the header initialization fails.

        '''
        super().__init__(parent)

    # Set main properties of the manager
        self.setSelectionMode(QW.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setColumnCount(2)

    # Set header properties
        if (header := self.header()) is None:
            raise RuntimeError("Failed to initialize header.")
        header.setSectionResizeMode(QW.QHeaderView.ResizeMode.ResizeToContents)
        self.setHeaderHidden(True)

    # Disable default editing; item editing is forced via 'onEdit' method
        self.setEditTriggers(QW.QAbstractItemView.EditTrigger.NoEditTriggers)

    # Enable custom context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    # Set custom scrollbars
        self.setHorizontalScrollBar(CW.StyledScrollBar(style.O_HORIZONTAL))
        self.setVerticalScrollBar(CW.StyledScrollBar(style.O_VERTICAL))

    # Set the style-sheet 
        self.setStyleSheet(style.SS_DATAMANAGER)

    # Connect signals to custom slots
        self._connect_slots()


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # User's mouse interactions signals
        self.itemClicked.connect(self.viewData)
        # self.itemChanged.connect(self.viewData) # too much triggers
        # self.itemActivated.connect(self.viewData) # does not trigger enough
        self.itemDoubleClicked.connect(self.onEdit)
        self.customContextMenuRequested.connect(self.showContextMenu)

    # Forward model signals to dataModified signal
        if (model := self.model()) is not None:
            model.dataChanged.connect(self.dataModified)
            model.rowsRemoved.connect(self.dataModified)


    def onEdit(self, item: DataManagerWidgetItem, col: int = 0) -> None:
        '''
        Force item editing in first column. DataSubGroup objects are excluded
        from editing.

        Parameters
        ----------
        item : DataGroup, DataSubGroup or DataObject
            The item that requests editing.
        col : int, optional
            The column where edits are requested. This parameter is unused: if
            different than 0, editing will still be forced to be on the first 
            column. The default is 0.

        '''
        if isinstance(item, (CW.DataGroup, CW.DataObject)):
            self.editItem(item, 0)


    def showContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Define a menu (Menu)
        menu = QW.QMenu(self)
        menu.setStyleSheet(style.SS_MENU)

    # Get the item that is clicked at 'point' and the group it belongs to
        item = cast(DataManagerWidgetItem | None, self.itemAt(point))
        group = self.getItemParentGroup(item)

    #(A) CONTEXT MENU ON VOID =================================================
        if item is None or group is None: #'point' is not on a valid item

        # Only allow adding a new group or clearing all groups
            menu.addAction(style.getIcon('CIRCLE_ADD'), 'New sample', self.addGroup)
            menu.addSeparator()
            menu.addAction(style.getIcon('REMOVE'), 'Remove all', self.clearAll)

    #(B) CONTEXT MENU ON GROUP ================================================
        elif isinstance(item, CW.DataGroup):

        #--------------------------[ Group Actions ]---------------------------
        #|Load input maps
            load_imap = QW.QAction(style.getIcon('STACK'), 'Input maps', menu)
            load_imap.triggered.connect(lambda: self.loadData(item.inmaps))
        #|Load mineral maps
            load_mmap = QW.QAction(style.getIcon('MINERAL'), 'Mineral maps', menu)
            load_mmap.triggered.connect(lambda: self.loadData(item.minmaps))
        #|Load masks
            load_mask = QW.QAction(style.getIcon('MASK'), 'Masks', menu)
            load_mask.triggered.connect(lambda: self.loadData(item.masks))
        #|Rename group
            rename_group = QW.QAction(style.getIcon('RENAME'), 'Rename', menu)
            rename_group.triggered.connect(lambda: self.onEdit(item))
        #|Clear selected groups
            clear_group = QW.QAction(style.getIcon('CLEAR'), 'Clear', menu)
            clear_group.triggered.connect(self.clearSelectedGroups)
        #|Delete selected groups
            delete_group = QW.QAction(style.getIcon('REMOVE'), 'Remove', menu)
            delete_group.triggered.connect(self.delSelectedGroups)

        #--------------------------[ Group Submenus ]--------------------------
        #|Import data
            import_submenu = QW.QMenu('Import...', menu)
            import_submenu.setIcon(style.getIcon('IMPORT'))
            import_submenu.addActions((load_imap, load_mmap, load_mask))

        #--------------------------[ Populate Menu ]---------------------------
            menu.addMenu(import_submenu)
            menu.addSeparator()
            menu.addActions((rename_group, clear_group, delete_group))

    #(C) CONTEXT MENU ON SUBGROUP =============================================
        elif isinstance(item, CW.DataSubGroup):

        #-------------------------[ Subgroup Actions ]-------------------------
        #|Load data
            load_data = QW.QAction(style.getIcon('IMPORT'), 'Import', menu)
            load_data.triggered.connect(lambda: self.loadData(item))
        #|Clear selected subgroups
            clear_subgroup = QW.QAction(style.getIcon('CLEAR'), 'Clear', menu)
            clear_subgroup.triggered.connect(self.clearSelectedSubgroups)

        #--------------------------[ Populate Menu ]---------------------------
            menu.addAction(load_data)
            menu.addSeparator()
            menu.addAction(clear_subgroup)

        #---------------[ Special Additions For MASKS Subgroup ]---------------
            if item.name == 'Masks':
            #[Mask Subgroup Actions]
            #|Check all masks
                chk_mask = QW.QAction('Check all', menu)
                chk_mask.setEnabled(not item.isEmpty())
                chk_mask.triggered.connect(lambda: self.checkMasks(True, group))
            #|Uncheck all masks
                unchk_mask = QW.QAction('Uncheck all', menu)
                unchk_mask.setEnabled(not item.isEmpty())
                unchk_mask.triggered.connect(lambda: self.checkMasks(False, group))
                
            #[Add To Menu]
                menu.addSeparator()
                menu.addActions((chk_mask, unchk_mask))

    #(D) CONTEXT MENU ON DATA OBJECTS =========================================
        elif isinstance(item, CW.DataObject):
            item_data = (components := item.getComponents())['data']

        #-----------------------[ Data Object Actions ]------------------------
        #|Rename item
            rename_item = QW.QAction(style.getIcon('RENAME'), 'Rename', menu)
            rename_item.triggered.connect(lambda: self.onEdit(item))
        #|Delete item
            delete_item = QW.QAction(style.getIcon('REMOVE'), 'Remove', menu)
            delete_item.triggered.connect(self.delData)
        #|Move item up
            move_up = QW.QAction(style.getIcon('ARROW_UP'), 'Move up', menu)
            move_up.triggered.connect(lambda: self.moveItemUp(item))
        #|Move item down
            move_down = QW.QAction(style.getIcon('ARROW_DOWN'), 'Move down', menu)
            move_down.triggered.connect(lambda: self.moveItemDown(item))
        #|Move item to (data-driven list of actions)
            move_to_actions: list[QW.QAction] = []
            for g in self.getAllGroups():
                act = QW.QAction(g.name, menu)
                act.setEnabled(g != group)
                act.triggered.connect(lambda _, g=g: self.moveItemTo(item, g))
                move_to_actions.append(act)
        #|Fix item's data source
            fix_source = QW.QAction(style.getIcon('FIX'), 'Fix source', menu)
            fix_source.setEnabled(components['not_found'])
            fix_source.triggered.connect(lambda: self.fixDataSource(item))
        #|Refresh item's data source
            refresh = QW.QAction(style.getIcon('REFRESH'), 'Refresh source', menu)
            refresh.triggered.connect(self.refreshDataSource)
        #|Save item
            save = QW.QAction(style.getIcon('SAVE'), 'Save', menu)
            save.setEnabled(item_data is not None)
            save.triggered.connect(lambda: self.saveData(item))
        #|Save item as...
            save_as = QW.QAction(style.getIcon('SAVE_AS'), 'Save As...', menu)
            save_as.setEnabled(item_data is not None)
            save_as.triggered.connect(lambda: self.saveData(item, False))

        #----------------------[ Data Object Subgroups ]-----------------------
        #|Move to sample...
            move_submenu = QW.QMenu('Move to sample...', menu)
            move_submenu.addActions(move_to_actions)

        #--------------------------[ Populate Menu ]---------------------------
            menu.addActions((rename_item, delete_item))
            menu.addSeparator()
            menu.addActions((move_up, move_down))
            menu.addMenu(move_submenu)
            menu.addSeparator()
            menu.addActions((fix_source, refresh, save, save_as))

        #---------------[ Special Additions For INPUT MAP Item ]---------------
            if isinstance(item_data, InputMap):
            #[Input Map Item Actions]
            #|Invert input map
                invert_imap = QW.QAction(style.getIcon('INVERT'), 'Invert', menu)
                invert_imap.triggered.connect(self.invertInputMap)
            #|Set RGBA channels (list of actions)
                rgba_actions: list[QW.QAction] = []
                for c in ('R', 'G', 'B', 'A'):
                    act = QW.QAction(f'{c} channel', menu)
                    act.triggered.connect(lambda _, c=c: self.rgbaChannelSet.emit(c))
                    rgba_actions.append(act)

            #[Input Map Item Submenus]
            #|Set as RGBA channel
                rgba_submenu = QW.QMenu('Set as RGBA channel...', menu)
                rgba_submenu.setIcon(style.getIcon('RGBA'))
                rgba_submenu.addActions(rgba_actions)

            #[Add To Menu]
                menu.addSeparator()
                menu.addAction(invert_imap)
                menu.addMenu(rgba_submenu)

        #--------------[ Special Additions For MINERAL MAP Item ]--------------
            elif isinstance(item_data, MineralMap):
            #[Mineral Map Item Actions]
            #|Randomize palette
                randpal = QW.QAction(
                    style.getIcon('RANDOMIZE_COLOR'), 'Randomize', menu)
                randpal.triggered.connect(lambda: self.randomizePalette(item))
            #|Copy palette
                copypal = QW.QAction(style.getIcon('COPY'), 'Copy')
                copypal.triggered.connect(
                    lambda: self.copyPaletteToClipboard(item))
            #|Paste palette
                pastepal = QW.QAction(style.getIcon('PASTE'), 'Paste', menu)
                pastepal.triggered.connect(
                    lambda: self.pastePaletteFromClipboard(item))
            #|Export array
                export = QW.QAction(style.getIcon('EXPORT'), 'Export map', menu)
                export.triggered.connect(lambda: self.exportMineralMap(item))

            #[Mineral Map Item Submenus]
            #|Palette
                palette_submenu = QW.QMenu('Palette', menu)
                palette_submenu.setIcon(style.getIcon('PALETTE'))
                palette_submenu.addActions((randpal, copypal, pastepal))

            #[Add To Menu]
                menu.addSeparator()
                menu.addMenu(palette_submenu)
                menu.addAction(export)

        #-----------------[ Special Additions For MASK Item ]------------------
            elif isinstance(item_data, Mask):
            #[Mask Item Actions]
            #|Invert mask
                invert_mask = QW.QAction(style.getIcon('INVERT'), 'Invert', menu)
                invert_mask.triggered.connect(self.invertMask)
            #|Merge masks with intersection
                intersect = QW.QAction('Intersection', menu)
                intersect.triggered.connect(lambda: self.mergeMasks(group, 'I'))
            #|Merge masks with union
                union = QW.QAction('Union', menu)
                union.triggered.connect(lambda: self.mergeMasks(group, 'U'))

            #[Mask Item Submenus]
            #|Merge masks
                merge_submenu = QW.QMenu('Merge', menu)
                merge_submenu.addActions((intersect, union))

            #[Add To Menu]
                menu.addSeparator()
                menu.addAction(invert_mask)
                menu.addMenu(merge_submenu)

        #--------------[ Special Additions For POINT DATA Item ]---------------
        #TODO

    #(E) SAFETY FALLBACK ======================================================
        else:
            return

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.mapToGlobal(point))


    def get_group(self, idx: int) -> CW.DataGroup | None:
        '''
        Get the group at index 'idx'.

        Parameters
        ----------
        idx : int
            The index of the group to be retrieved.

        Returns
        -------
        DataGroup or None
            The group at index 'idx' or None if 'idx' is invalid.

        '''
        if isinstance(item := self.topLevelItem(idx), CW.DataGroup):
            return item
        return None


    def getAllGroups(self) -> list[CW.DataGroup]:
        '''
        Get all the groups.

        Returns
        -------
        groups : list[DataGroup]
            List of groups.

        '''
        count = self.topLevelItemCount()
        items = [self.topLevelItem(idx) for idx in range(count)]
        groups = [i for i in items if isinstance(i, CW.DataGroup)]
        return groups


    def getItemParentGroup(
        self, item: DataManagerWidgetItem | None) -> CW.DataGroup | None:
        '''
        Get the item's group.

        Parameters
        ----------
        item : DataObject, DataSubGroup or DataGroup or None
            The item whose group needs to be retrieved.

        Returns
        -------
        group : DataGroup or None
            The item's group. Returns None when 'item' is not a valid item or
            it is not owned by any group.

        '''
    # Item is group
        if isinstance(item, CW.DataGroup):
            group = item
    # Item is subgroup
        elif isinstance(item, CW.DataSubGroup):
            group = item.group()
    # Item is data object
        elif isinstance(item, CW.DataObject):
            subgroup = item.subgroup()
            group = None if subgroup is None else subgroup.group()
    # Item is invalid (safety)
        else:
            group = None

        return group


    def getSelectedGroupsIndexes(self) -> list[int]:
        '''
        Get the indices of the selected groups.

        Returns
        -------
        groups_idx: list[int]
            List of indices of selected groups.

        '''
        items = self.selectedItems()
    # Groups are the top-level items of the DataManager; indexOfTopLevelItem 
    # returns -1 if the item is not a top-level item (i.e., is not a group)
        indexes = map(lambda i: self.indexOfTopLevelItem(i), items)
        groups_idx = [idx for idx in indexes if idx != -1]
        return groups_idx
    

    def getSelectedSubgroups(self) -> list[CW.DataSubGroup]:
        '''
        Get the selected data subgroups.

        Returns
        -------
        subgroups: list[DataSubGroup]
            List of selected data subgroups.

        '''
        items = self.selectedItems()
        subgroups = [i for i in items if isinstance(i, CW.DataSubGroup)]
        return subgroups


    def getSelectedDataObjects(self) -> list[CW.DataObject]:
        '''
        Get the selected data objects.

        Returns
        -------
        data_obj : list[DataObject]
            List of selected data objects.

        '''
        items = self.selectedItems()
        data_obj = [i for i in items if isinstance(i, CW.DataObject)]
        return data_obj
    

    def getAllDataObjects(self) -> list[CW.DataObject]:
        '''
        Get all data object from all groups in a single list.

        Returns
        -------
        list[DataObject]
            List of all data objects.

        '''
        objects = []
        for group in self.getAllGroups():
            objects.extend(group.getAllDataObjects())
        return objects


    def addGroup(self, name: str | None = None) -> CW.DataGroup:
        '''
        Add a new group to the manager and return it.

        Parameters
        ----------
        name : str or None, optional
            Name to assign to group. If None, a default name is assigned. The
            default is None.

        Returns
        -------
        DataGroup.

        '''
    # If name is None, rename the group as 'New Sample' + progressive ID
        if name is None:
            unnamed = self.findItems('New Sample', Qt.MatchFlag.MatchContains)
            name = 'New Sample'
            if (n := len(unnamed)): 
                name += f' ({n})'
    
    # Add a new DataGroup as a top level item of the Data Manager
        new_group = CW.DataGroup(name)
        self.addTopLevelItem(new_group)
        return new_group


    def take_group(self, idx: int) -> CW.DataGroup | None:
        '''
        Remove the group at index 'idx' and return it.

        Parameters
        ----------
        idx : int
            The index of the group to be removed.

        Returns
        -------
        DataGroup or None
            The removed group or None if 'idx' is invalid.

        '''
        if isinstance(item := self.takeTopLevelItem(idx), CW.DataGroup):
            return item
        return None 
        

    def delSelectedGroups(self) -> None:
        '''
        Remove selected groups.

        '''
    # Check for user confirm
        msgbox = CW.MsgBox(self, 'Quest', 'Remove selected sample(s)?')
        if msgbox.is_choice('No'):
            return

    # Delete groups and clear their data to destroy possible broken references
        for idx in sorted(self.getSelectedGroupsIndexes(), reverse=True):
            if (group := self.take_group(idx)) is not None:
                group.clear()
        self.refreshView()


    def clearSelectedGroups(self) -> None:
        '''
        Clear data from selected groups.

        '''
    # Check for user confirm
        msgbox = CW.MsgBox(self, 'Quest', 'Clear selected sample(s)?')
        if msgbox.is_choice('No'):
            return

        for idx in self.getSelectedGroupsIndexes():
            if (group := self.get_group(idx)) is not None:
                group.clear()
        self.refreshView()

    
    def moveItemTo(self, item: CW.DataObject, dst_group: CW.DataGroup) -> None:
        '''
        Move 'item' from its original group to another existent group 
        'dst_group'.

        Parameters
        ----------
        item : CW.DataObject
            Item to be moved.
        dst_group : CW.DataGroup
            Destination group. 

        '''
    # Do nothing if source group is invalid or is the destination group
        src_group = self.getItemParentGroup(item)
        if src_group is None or src_group == dst_group:
            return

    # Do nothing if the source subgroup is invalid (safety)
        if (src_subgr := item.subgroup()) is None:
            return
        
    # Delete data from source subgroup and append it to destination subgroup 
        dtype = src_subgr.datatype
        src_subgr.delChild(item)
        dst_subgr = [s for s in dst_group.subgroups if s.datatype == dtype][0]
        dst_subgr.addChild(item)

    # Check for unfitting maps shapes on both source and destination groups
        src_group.setShapeWarnings()
        dst_group.setShapeWarnings()

    # Force viewing the moved item to provide better feedback
        self.setCurrentItem(item)
        self.refreshView()


    def moveItemUp(self, item: CW.DataObject) -> None:
        '''
        Move 'item' up by one position within its subgroup.

        Parameters
        ----------
        item : CW.DataObject
            Item to be moved.

        '''
    # Do nothing if the group or the subgroup are invalid
        subgr = item.subgroup()
        if self.getItemParentGroup(item) is None or subgr is None:
            return
    
    # Move item within its subgroup
        subgr.moveChildUp(item)

    # Force viewing the moved item to provide better feedback
        self.setCurrentItem(item)
        self.refreshView()


    def moveItemDown(self, item: CW.DataObject) -> None:
        '''
        Move 'item' down by one position within its subgroup.

        Parameters
        ----------
        item : CW.DataObject
            Item to be moved.

        '''
    # Do nothing if the group or the subgroup are invalid
        subgr = item.subgroup()
        if self.getItemParentGroup(item) is None or subgr is None:
            return
    
    # Move item within its subgroup
        subgr.moveChildDown(item)

    # Force viewing the moved item to provide better feedback
        self.setCurrentItem(item)
        self.refreshView()
        

    def clearSelectedSubgroups(self) -> None:
        '''
        Clear data from selected subgroups.

        '''
    # Check for user confirm
        msgbox = CW.MsgBox(self, 'Quest', 'Clear selected data?')
        if msgbox.is_choice('No'):
            return
    
    # Clear out subgroups
        subgroups = self.getSelectedSubgroups()
        groups = [self.getItemParentGroup(s) for s in subgroups]
        for subgr in subgroups:
            subgr.takeChildren()

    # Check for unfitting maps shapes within the samples
        for idx in set((self.indexOfTopLevelItem(g) for g in groups)):
            if (group := self.get_group(idx)) is not None:
                group.setShapeWarnings()

        self.refreshView()


    def delData(self) -> None:
        '''
        Delete the selected data objects.

        '''
    # Check for user confirm
        msgbox = CW.MsgBox(self, 'Quest', 'Remove selected data?')
        if msgbox.is_choice('No'):
            return
    
    # Delete data objects
        items = self.getSelectedDataObjects()
        groups = [self.getItemParentGroup(i) for i in items]
        for i in reversed(items):
            if (subgr := i.subgroup()) is not None:
                subgr.delChild(i)

    # Check for unfitting maps shapes within the samples
        for idx in set((self.indexOfTopLevelItem(g) for g in groups)):
            if (group := self.get_group(idx)) is not None:
                group.setShapeWarnings()

        self.refreshView()


    def loadData(
        self,
        subgroup: CW.DataSubGroup,
        paths: Collection[str] | None = None
    ) -> None:
        '''
        Load data objects to a subgroup. This is a wrapper loading method that 
        checks the subgroup type and then calls the specialized loading method.

        Parameters
        ----------
        subgroup : DataSubGroup
            The subgroup to be populated with data.
        paths : Collection[str] or None, optional
            A collection of filepaths to data. If None, user will be prompt to
            load them from disk. The default is None.

        '''
    # Do nothing if group is invalid (safety)
        if (group := subgroup.group()) is None:
            return

    # Call the dedicated load method based on the subgroup 
        match subgroup.name:
            case 'Input Maps':
                self.loadInputMaps(group, paths)
            case 'Mineral Maps':
                self.loadMineralMaps(group, paths)
            case 'Masks':
                self.loadMasks(group, paths)
            # case 'Point Analysis' TODO
            case _: # safety
                return  
        
    # Check for unfitting maps shapes
        group.setShapeWarnings()


    def saveData(self, item: CW.DataObject, overwrite: bool = True) -> None:
        '''
        Save the item data to file.

        Parameters
        ----------
        item : DataObject
            The item to be saved.
        overwrite : bool, optional
            Overwrite the original filepath. If False, user will be prompted to
            choose a new path. The default is True.

        '''
    # Do nothing if item data is invalid (safety)
        components = item.getComponents()
        item_data, path = components['data'], components['filepath']
        if item_data is None:
            return

    # Select filepath if overwrite is not requested (= saveAs), path is None or
    # path is invalid (moved, renamed, deleted)
        if not overwrite or path is None or not item.filepathValid():
            if isinstance(item_data, InputMap):
                filext = 'Compressed ASCII map (*.gz);;ASCII map (*.txt)'
            elif isinstance(item_data, MineralMap):
                filext = 'Mineral maps (*.mmp)'
            elif isinstance(item_data, Mask):
                filext = 'Mask (*.msk)'
            # add PointAnalysis
            else: 
                return # safety

            path = CW.FileDialog(self, 'save', 'Save Map', filext).get_path()
            if not path:
                return
    
    # Otherwise, double check if user really wants to overwrite the file 
        else:
            text = f'Source data ({path}) will be irreversibly altered. Proceed?'
            msgbox = CW.MsgBox(self, 'QuestWarn', text)
            if msgbox.is_choice('No'):
                return

    # Save the data
        try:
            item_data.save(path)
        # Ensure that DataObjects is populated with updated data and filepath
            item.setObjectData(item_data) 
        # All edits have been saved, so the edited status is set to False
            item.setEdited(False)
            
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save file.', str(e))
            return

        finally:
            self.refreshView()
    

    def unsavedData(self) -> list[CW.DataObject]:
        '''
        Get a list of all Data Objects that contain unsaved data.

        Returns
        -------
        list[CW.DataObject]
            List of edited Data Objects.

        '''
        all_obj = self.getAllDataObjects()
        return [o for o in all_obj if o.getComponents()['is_edited']]


    def loadInputMaps(
        self,
        group: CW.DataGroup,
        paths: Collection[str] | None = None
    ) -> None:
        '''
        Specialized loading method to load input maps to a group.

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : Collection[str] or None, optional
            A collection of filepaths to data. If None, user will be prompt to
            load them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            ftypes = 'ASCII maps (*.txt *.gz)'
            paths = CW.FileDialog(self, 'open', 'Load Maps', ftypes, True).get_paths()
            if not paths:
                return

    # Data loading routine
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        errors = []
        for n, p in enumerate(paths, start=1):
            try:
                xmap = InputMap.load(p)
                group.inmaps.addData(xmap)
        
            except FileNotFoundError:
                # Add item with "not found" status
                item = CW.DataObject(None)
                item.setInvalidFilepath(p)
                group.inmaps.addChild(item)

            except Exception as e:
                errors.append((p, e))

            finally:
                pbar.setValue(n)

    # Send detailed error message if any file failed to be loaded
        if n_err := len(errors):
            text = f'A total of {n_err} file(s) failed to load.'
            dtext = '\n\n'.join((f'{path}: {exc}' for path, exc in errors))
            CW.MsgBox(self, 'Crit', text, dtext)

    # Auto expand group
        self.expandRecursively(self.indexFromItem(group))


    def loadMineralMaps(
        self,
        group: CW.DataGroup,
        paths: Collection[str] | None = None
    ) -> None:
        '''
        Specialized loading method to load mineral maps to a group.

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : Collection[str] or None, optional
            A collection of filepaths to data. If None, user will be prompt to
            load them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            ftypes = 'Mineral maps (*.mmp);;ASCII maps (*.txt *.gz)'
            paths = CW.FileDialog(self, 'open', 'Load Maps', ftypes, True).get_paths()
            if not paths:
                return
    
    # Data loading routine
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        errors = []
        for n, p in enumerate(paths, start=1):
            try:
                mmap = MineralMap.load(p)
                # Ask user confirm if a large number of classes is detected
                count_warn = settings.manager.get('data/warn_phase_count')
                if (cnt := mmap.phase_count()) > count_warn:
                    txt = f'{cnt} mineral classes detected for {p}. Proceed?'
                    dtxt = '\n'.join(mmap.get_phases())
                    # Make pbar the parent of msgbox for it to be responsive
                    msgbox = CW.MsgBox(pbar, 'QW', txt, dtxt, 'X-Min Learn')
                    if msgbox.is_choice('No'):
                        continue
                # Convert ASCII-style maps to .mmp
                if mmap.is_ascii_style():
                    mmap.save(cf.extend_filename(p, '', '.mmp'))
                group.minmaps.addData(mmap)
        
            except FileNotFoundError:
                # Add item with "not found" status
                item = CW.DataObject(None)
                item.setInvalidFilepath(p)
                group.minmaps.addChild(item)

            except Exception as e:
                errors.append((p, e))

            finally:
                pbar.setValue(n)

    # Send detailed error message if any file failed to be loaded
        if n_err := len(errors):
            text = f'A total of {n_err} file(s) failed to load.'
            dtext = '\n\n'.join((f'{path}: {exc}' for path, exc in errors))
            CW.MsgBox(self, 'Crit', text, dtext)

    # Auto expand group
        self.expandRecursively(self.indexFromItem(group))


    def loadMasks(
        self,
        group: CW.DataGroup,
        paths: Collection[str] | None = None
    ) -> None:
        '''
        Specialized loading method to load masks to a group.

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : Collection[str] or None, optional
            A collection of filepaths to data. If None, user will be prompt to
            load them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            ft = 'Masks (*.msk);;Text file (*.txt)'
            paths = CW.FileDialog(self, 'open', 'Load Masks', ft, True).get_paths()
            if not paths:
                return
    
    # Data loading routine
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        errors = []
        for n, p in enumerate(paths, start=1):
            try:
                mask = Mask.load(p)
                group.masks.addData(mask)
        
            except FileNotFoundError:
                # Add item with "not found" status
                item = CW.DataObject(None)
                item.setInvalidFilepath(p)
                group.masks.addChild(item)

            except Exception as e:
                errors.append((p, e))

            finally:
                pbar.setValue(n)

    # Send detailed error message if any file failed to be loaded
        if n_err := len(errors):
            text = f'A total of {n_err} file(s) failed to load.'
            dtext = '\n\n'.join((f'{path}: {exc}' for path, exc in errors))
            CW.MsgBox(self, 'Crit', text, dtext)

    # Auto expand group
        self.expandRecursively(self.indexFromItem(group))


    def invertInputMap(self) -> None:
        '''
        Invert the selected input maps.

        '''
    # Invert the input map array held in each selected and valid item
        items = self.getSelectedDataObjects()
        pbar = CW.PopUpProgBar(self, len(items), 'Inverting data')
        for n, i in enumerate(items, start=1):
            if (inmap := i.getInputMapData()) is not None:
                inmap.invert()
                i.setEdited(True)
            pbar.setValue(n)

        self.refreshView()


    def invertMask(self) -> None:
        '''
        Invert the selected masks.

        '''
    # Invert the mask array held in each selected and valid item
        items = self.getSelectedDataObjects()
        pbar = CW.PopUpProgBar(self, len(items), 'Inverting data')
        for n, i in enumerate(items, start=1):
            if (mask := i.getMaskData()) is not None:
                mask.invert()
                i.setEdited(True)
            pbar.setValue(n)

        self.refreshView()


    def mergeMasks(
            self, group: CW.DataGroup, mode: iatools.BinaryMergeModes) -> None:
        '''
        Merge the selected masks into a new Mask object and add it to the
        group.

        Parameters
        ----------
        group : DataGroup
            The group that holds the mask data.
        mode : BinaryMergeModes
            How to merge the masks. See 'getCompositeMask' method for more 
            details.

        '''
    # Exit function if group is invalid (safety)
        if group is None: 
            return
    
    # Exit function if composite mask is invalid
        merged_mask = group.getCompositeMask(mode=mode, ignore_single_mask=True)
        if merged_mask is None: 
            return
        
    # Append the mask to the group and set it as edited
        group.masks.addData(merged_mask)
        group.masks.getChildren()[-1].setEdited(True)


    def checkMasks(self, checked: bool, group: CW.DataGroup) -> None:
        '''
        Check or uncheck all masks loaded in a group.

        Parameters
        ----------
        checked : bool
            Whether to check or uncheck the masks.
        group : DataGroup
            The group whose masks should be checked or unchecked.

        '''
        for child in group.masks.getChildren():
            child.setChecked(checked)
        self.refreshView()

    
    def copyPaletteToClipboard(self, item: CW.DataObject) -> None:
        '''
        Copy the palette of the mineral map to the clipboard.

        Parameters
        ----------
        item : CW.DataObject
            The data object holding the mineral map data.

        '''
    # Exit function if item does not contain mineral map data (safety)
        if (minmap := item.getMineralMapData()) is None: 
            return
    
    # Try converting the palette to JSON bytes data
        colors = list(minmap.palette.values())
        try:
            data = json.dumps(colors).encode('utf-8')
        except TypeError as e:
            CW.MsgBox(self, 'Crit', 'Invalid palette data', str(e))
            return

    # Set JSON data as MIME data for the clipboard
        mimedata = QMimeData()
        mimedata.setData(self.PALETTE_MIMETYPE, data)
        if (clipboard := QW.qApp.clipboard()) is not None:
            clipboard.setMimeData(mimedata)


    def pastePaletteFromClipboard(self, item: CW.DataObject) -> None:
        '''
        Paste palette from the clipboard to the mineral map.

        Parameters
        ----------
        item : CW.DataObject
            The data object holding the mineral map data.

        '''
    # Exit function if item does not contain mineral map data (safety)
        if (minmap := item.getMineralMapData()) is None: 
            return

    # Exit function if app fails to retrieve the clipboard (safety)
        if (clipboard := QW.qApp.clipboard()) is None:
            return
        
    # Exit function if clipboard does not contain palette data (safety)
        mimedata = clipboard.mimeData()
        if mimedata is None or not mimedata.hasFormat(self.PALETTE_MIMETYPE):
            return
    
    # Try loading the palette data as list of RGB tuples from the clipboard
        try:
            data = mimedata.data(self.PALETTE_MIMETYPE).data().decode('utf-8')
            colors = [tuple(rgb) for rgb in json.loads(data)]
        except json.JSONDecodeError as e:
            CW.MsgBox(self, 'Crit', 'Invalid palette data in clipboard', str(e))
            return
    
    # Set the palette to the mineral map and mark it as edited
        minmap.set_palette(colors)
        item.setEdited(True)
        self.refreshView()


    def randomizePalette(self, item: CW.DataObject) -> None:
        '''
        Randomize mineral map's palette.

        Parameters
        ----------
        item : CW.DataObject
            The data object holding the mineral map data.

        '''
    # Exit function if item does not contain mineral map data (safety)
        if (minmap := item.getMineralMapData()) is None: 
            return
        
    # Randomize the mineral map's palette and mark it as edited
        minmap.set_palette(minmap.rand_colorlist())
        item.setEdited(True)
        self.refreshView()


    def exportMineralMap(self, item: CW.DataObject) -> None:
        '''
        Export the encoded mineral map (i.e,. with mineral classes expressed
        as numerical IDs) to ASCII format. If users requests it, the encoder
        dictionary is also exported.

        Parameters
        ----------
        item : DataObject
            The data object holding the mineral map data.

        '''
    # Exit function if item does not contain mineral map data (safety)
        if (minmap := item.getMineralMapData()) is None: 
            return

    # Ask for user confirm
        text =  'Export map as a numeric array?'
        det_text = (
            'The translation dictionary is a text file holding a reference to '
            'the mineral classes linked with IDs of the exported mineral map.'
        )
        msg_cbox = QW.QCheckBox('Include translation dictionary')
        msg_cbox.setChecked(True)
        msgbox = CW.MsgBox(self, 'Quest', text, det_text, cbox=msg_cbox)
        if msgbox.is_choice('No'):
            return
        
    # Do nothing if the outpath is invalid or the file dialog is canceled
        ftype = 'ASCII file (*.txt)'
        outpath = CW.FileDialog(self, 'save', 'Save Map', ftype).get_path()
        if not outpath:
            return
        
    # Save the mineral map to disk
        np.savetxt(outpath, minmap.minmap_encoded, fmt='%d')

    # Also save the encoder if user requests it
        if msgbox.cboxChecked():
            encoder_path = cf.extend_filename(outpath, '_transDict')
            rows, cols = minmap.shape
            with open(encoder_path, 'w') as ep:
                for id_, lbl in minmap.encoder.items():
                    ep.write(f'{id_} :\t{lbl}\n')
            # Include number of rows and columns
                ep.write(f'\nNROWS: {rows}\nNCOLS: {cols}')


    def fixDataSource(self, item: CW.DataObject) -> None:
        '''
        Repair the data source of 'item' by selecting a valid filepath. This 
        method attempts to automatically fix all the invalid items in the same
        group of 'item' using the parent folder of the selected filepath as a
        reference. 

        Parameters
        ----------
        item : CW.DataObject
            Item to be repaired.

        '''
    # Exit function if item has no valid subgroup or group (safety)
        if (subgr := item.subgroup()) is None or (group := subgr.group()) is None:
            return

    # Identify correct item data type by looking at its subgroup, because item
    # can contain None data if they failed to be loaded
        match subgr.name:
            case 'Input Maps':
                ftype = 'ASCII maps (*.txt *.gz)'
            case 'Mineral Maps':
                ftype = 'Mineral maps (*.mmp);;ASCII maps (*.txt *.gz)'
            case 'Masks':
                ftype = 'Masks (*.msk);;Text file (*.txt)'
            case _:
                return
    
    # Do nothing if path is invalid or file dialog is canceled
        path = CW.FileDialog(self, 'open', 'Fix Source', ftype).get_path()
        if not path:
            return
    
    # Fix item file source and data
        try:
        # Setting object data also sets its filepath to data.filepath
            if item.getComponents()['data'] is None:
                item.setObjectData(subgr.datatype.load(path))
        # Setting object path also changes its data to the data stored in path
            else:
                item.setFilepath(path)

        # Toggle off the 'not_found' and 'edited' status 
            item.setNotFound(False)
            item.setEdited(False)
            
        except Exception as e:
            CW.MsgBox(self, 'Crit', f'Unexpected file: {path}', str(e))
            return

    # Try fixing data objects in the same group that are also "not found" by 
    # checking all the files in the same root folder of the loaded file
        root_fld = os.path.dirname(path)
        available_files = os.listdir(root_fld)
        for obj in group.getAllDataObjects():
        # Skip invalid/unsuitable objects
            components = obj.getComponents()
            if (not components['not_found']
                or (obj_path := components['filepath']) is None
                or (obj_subgr := obj.subgroup()) is None
            ): 
                continue
        # Get object info
            obj_data = components['data']
            obj_fname = cf.path2filename(obj_path)
            obj_type = obj_subgr.datatype
       
            for f in available_files:
            # Skip files with invalid extensions or unfitting filename
                f_name, f_ext = os.path.splitext(f)
                if f_ext.lower() in obj_type._FILEXT and f_name == obj_fname:
                    f_path = os.path.join(root_fld, f)
                else:
                    continue
            # Fix object file source and data
                try:
                    if obj_data is None:
                    # Setting data also sets filepath to data.filepath
                        obj.setObjectData(obj_type.load(f_path))
                    else:
                    # Setting path also changes data to the data stored in path
                        obj.setFilepath(f_path)

                # Toggle off the 'not_found' and 'edited' status
                    obj.setNotFound(False)
                    obj.setEdited(False)

                except:
                    continue

    # Check for maps shapes warnings within the group and refresh view
        group.setShapeWarnings()
        self.refreshView()

                    
    def refreshDataSource(self) -> None:
        '''
        Reload the selected data from its original source.

        '''
        items = self.getSelectedDataObjects()
        groups = [self.getItemParentGroup(i) for i in items]
        pbar = CW.PopUpProgBar(self, len(items), 'Reloading data')
        errors = []
        for n, i in enumerate(items, start=1):
            comps = i.getComponents()
            data, path, name = comps['data'], comps['filepath'], comps['name']
            try:
                # None path indicates unsaved data and cannot be refreshed
                if path is None: 
                    continue
                path_exists = os.path.exists(path)
                i.setNotFound(not path_exists)
                if path_exists and (subgr := i.subgroup()) is not None:
                    data = subgr.datatype if data is None else data
                    i.setObjectData(data.load(path))
                    i.setEdited(False)

            except Exception as e:
                errors.append((name, path, e))

            finally:
                pbar.setValue(n)

    # Check for unfitting maps shapes within the samples and refresh view
        for idx in set((self.indexOfTopLevelItem(g) for g in groups)):
            if (group := self.get_group(idx)) is not None:
                group.setShapeWarnings()

        self.refreshView()

    # Send detailed error message if any file failed to be refreshed
        if n_err := len(errors):
            text = f'A total of {n_err} file(s) failed to load.'
            dtext = '\n\n'.join((f'{fn} ({fp}): {ex}' for fn, fp, ex in errors))
            CW.MsgBox(self, 'Crit', text, dtext)


    def viewData(self, item: DataManagerWidgetItem | None) -> None:
        '''
        Send signal for displaying the item's data.

        Parameters
        ----------
        item : DataManagerWidgetItem | None
            The data object to be displayed.

        '''
    # Request to clear the scene if the item is not valid
        if item is None:
            self.clearSceneRequested.emit()
    # Request to update the scene if item is a valid object
        else:
            self.updateSceneRequested.emit(item)


    def refreshView(self) -> None:
        '''
        Request a re-rendering of the currently displayed item.

        '''
        item = cast(DataManagerWidgetItem | None, self.currentItem())
        self.viewData(item)


    def clearAll(self) -> None:
        '''
        Remove all groups.

        '''
        msgbox = CW.MsgBox(self, 'Quest', 'Remove all samples?')
        if msgbox.is_choice('Yes'):
            self.clear()
            self.clearSceneRequested.emit()

    
    def resetConfig(self) -> None:
        '''
        Reset the Data Manager to its default state and configuration.

        '''
        self.loadConfig(config_schemas.DATAMANAGER_DEFAULT_CONFIG)
            

    def getConfig(self) -> config_schemas.DataManagerConfig:
        '''
        Take a snapshot of the current state and configuration of the Data
        Manager.

        Returns
        -------
        config : DataManagerConfig
            Current state and configuration of Data Manager.

        '''
        config: config_schemas.DataManagerConfig = {}
        for group in self.getAllGroups():
        # Initialize config schema for group
            config[group.name] = {
                'Input Maps': [],
                'Mineral Maps': [],
                'Masks': []
            }
        # Populate each subgroup with corresponding data
            for subgr in group.subgroups:
                attributes = ['name', 'filepath']
                if subgr.name == 'Masks':
                    attributes.append('checkstate')

                data: list[config_schemas.DataObjectSchema] = []
                for obj in subgr.getChildren():
                # Get required data object components
                    components = obj.getComponents()
                    to_save = tuple(components[a] for a in attributes)
                # Don't save data objects with null filepath
                    if to_save[1] is not None:
                        data.append(to_save)

                config[group.name][subgr.name] = data
        
        return config


    def loadConfig(self, config: config_schemas.DataManagerConfig) -> None:
        '''
        Set the state and configuration of the Data Manager to those provided
        in 'config'.

        Parameters
        ----------
        config : DataManagerConfig
            Stored state and configuration of Data Manager.

        '''
    # Clear out the manager from previous data
        self.clear()
    
    # Add each sample and populate them with data
        for sample, data in config.items():
            group = self.addGroup(sample)
        
        # Populate with input maps data
            if len(data['Input Maps']):
                inmap_names, inmap_paths = zip(*data['Input Maps'])
                self.loadData(group.inmaps, inmap_paths)
                for idx, child in enumerate(group.inmaps.getChildren()):
                    child.setName(inmap_names[idx])

        # Populate with mineral maps data
            if len(data['Mineral Maps']):
                minmap_names, minmap_paths = zip(*data['Mineral Maps'])
                self.loadData(group.minmaps, minmap_paths)
                for idx, child in enumerate(group.minmaps.getChildren()):
                    child.setName(minmap_names[idx])

        # Populate with masks data
            if len(data['Masks']):
                mask_names, mask_paths, mask_checkstates = zip(*data['Masks'])
                self.loadData(group.masks, mask_paths)
                for idx, child in enumerate(group.masks.getChildren()):
                    child.setName(mask_names[idx])
                    child.setChecked(mask_checkstates[idx])



class HistogramViewer(QW.QWidget):

    scalerRangeChanged = pyqtSignal()

    def __init__(
        self,
        maps_canvas: plots.ImageCanvas,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A widget to visualize and interact with histograms of input maps data.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas displaying input maps data.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Define main attributes
        self.maps_canvas = maps_canvas

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Histogram Canvas
        self.canvas = plots.HistogramCanvas(
            logscale=True, size=(3, 1.5), wheel_zoom=False, wheel_pan=False)
        self.canvas.ax.get_yaxis().set_visible(False)
        self.canvas.setMinimumSize(300, 200)

    # HeatMap Scaler widget
        self.scaler = plots.HeatmapScaler(
            self.canvas.ax, self.onSpanSelect, btns=[1])

    # Navigation Toolbar
        self.navtbar = plots.NavTbar.histCanvasDefault(self.canvas, self)

    # HeatMap scaler toolbar
        self.scaler_tbar = QW.QToolBar('Histogram scaler toolbar')
        self.scaler_tbar.setStyleSheet(style.SS_TOOLBAR)

    # Toggle scaler action [-> Heatmap scaler toolbar]
        self.scaler_action = QW.QAction(
            style.getIcon('RANGE'), 'Enable scaler', self)
        self.scaler_action.setCheckable(True)
    
    # Extract mask action [-> Heatmap scaler toolbar]
        self.mask_action = QW.QAction(
            style.getIcon('ADD_MASK'), 'Extract mask', self)
        self.mask_action.setEnabled(False)

    # Min and Max scaler values input widgets [-> Heatmap scaler toolbar]
        self.scaler_vmin = QW.QSpinBox()
        self.scaler_vmin.setToolTip('Min. span value')
        self.scaler_vmax = QW.QSpinBox()
        self.scaler_vmax.setToolTip('Max. span value')
        for wid in (self.scaler_vmin, self.scaler_vmax):
            wid.setMaximum(2**16)
            wid.setSingleStep(1)
            wid.setMaximumWidth(100)
            wid.setEnabled(False)

    # Range warning icon [-> Heatmap scaler toolbar]
        warn_pixmap = QPixmap(str(style.ICONS.get('WARNING')))
        warn_lbl = QW.QLabel(self.scaler_tbar)
        warn_lbl.setPixmap(warn_pixmap.scaled(20, 20, style.ARM_KEEP))
        warn_lbl.setSizePolicy(style.SP_MAXIMUM, style.SP_MAXIMUM)
        warn_lbl.setToolTip('Lower limit is bigger than upper limit.')
        self.warn_icon = QW.QWidgetAction(self)
        self.warn_icon.setDefaultWidget(warn_lbl)
        self.warn_icon.setVisible(False)
    
    # Add actions and widgets to Heatmap scaler toolbar
        self.scaler_tbar.addAction(self.scaler_action)
        self.scaler_tbar.addAction(self.mask_action)
        self.scaler_tbar.addWidget(self.scaler_vmin)
        self.scaler_tbar.addWidget(self.scaler_vmax)
        self.scaler_tbar.addAction(self.warn_icon)

    # Bin slider widget
        self.bin_slider = QW.QSlider(style.O_HORIZONTAL)
        self.bin_slider.setSizePolicy(style.SP_MINIMUMEXPANDING, style.SP_FIXED)
        self.bin_slider.setMinimum(5)
        self.bin_slider.setMaximum(100)
        self.bin_slider.setSingleStep(5)
        self.bin_slider.setSliderPosition(50)

    # Bins selection layout
        bins_form = QW.QFormLayout()
        bins_form.addRow('Bins', self.bin_slider)

    # Adjust Layout
        main_layout = QW.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.navtbar)
        main_layout.addWidget(self.scaler_tbar)
        main_layout.addLayout(bins_form)
        main_layout.addWidget(self.canvas, stretch=2)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Context menu on histogram canvas
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Toggle scaler 
        self.scaler_action.toggled.connect(self.onScalerToggled)

    # Extract mask from scaler selection
        self.mask_action.triggered.connect(self.extractMaskFromScaler)

    # Set scaler extent when interacting with vmin and vmax inputs
        self.scaler_vmin.valueChanged.connect(self.onScalerRangeChanged)      
        self.scaler_vmax.valueChanged.connect(self.onScalerRangeChanged)

    # Number of bins selection
        self.bin_slider.valueChanged.connect(self.setHistBins)


    def showContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navtbar)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.canvas.mapToGlobal(point))


    def onScalerToggled(self, toggled: bool) -> None:
        '''
        Slot triggered when scaler is toggled on/off from the scaler toolbar.

        Parameters
        ----------
        toggled : bool
            Toggle state of the scaler.

        '''
    # Change state and visibility of the span selector
        self.scaler.set_active(toggled)
        self.scaler.set_visible(toggled)

    # Enable/disable scaler toolbar actions and widgets
        self.mask_action.setEnabled(toggled)
        self.scaler_vmin.setEnabled(toggled)
        self.scaler_vmax.setEnabled(toggled)

    # Update the scaler extents
        if not self.canvas.is_empty():
            if toggled:
                self.updateScalerExtents(update_span=False)
            else:
                self.applyScaling(None, None)

            self.canvas.draw()


    def onScalerRangeChanged(self) -> None:
        '''
        Slot triggered when the upper or the lower bound of the scaler are 
        modified from the respective widgets in the scaler toolbar.

        '''
        self.updateScalerExtents(update_span=True)
        self.scalerRangeChanged.emit()


    def onSpanSelect(self, vmin: float, vmax: float) -> None:
        '''
        Slot triggered after interacting with the span selector. Highlights
        in the data viewer the pixels that fall within the selected area in the
        histogram.

        Parameters
        ----------
        vmin : float
            Lower range bound.
        vmax : float
            Upper range bound.

        '''
    # Do nothing if the histogram canvas is empty
        if self.canvas.is_empty():
            return

    # Adjust values to integers; empty ranges are converted to (0, 0)
        vmin, vmax = round(vmin), round(vmax)
        if vmin == vmax:
            vmin, vmax = 0, 0

    # Update vmin and vmax values in the scaler toolbar while blocking their 
    # signals temporarily to avoid loops with 'onScalerRangeChanged' method
        self.scaler_vmin.blockSignals(True)
        self.scaler_vmax.blockSignals(True)
        self.scaler_vmin.setValue(vmin)
        self.scaler_vmax.setValue(vmax)
        self.scaler_vmin.blockSignals(False)
        self.scaler_vmax.blockSignals(False)

    # Update scaler extents
        self.updateScalerExtents(update_span=False)
        self.scalerRangeChanged.emit()


    def applyScaling(self, vmin: int | None, vmax: int | None) -> None:
        '''
        Set the maps canvas clims to 'vmin' and 'vmax'.

        Parameters
        ----------
        vmin : int or None
            Lower range value. If None, the clims are reset.
        vmax : int or None
            Upper range value. If None, the clims are reset.

        '''
    # Set vmin and vmax to None if one of them lies outside the map data range
        if vmin is not None and vmax is not None:
            array = self.maps_canvas.get_array()
            if array is None or vmin >= array.max() or vmax <= array.min():
                vmin, vmax = None, None

    # Update clims
        self.maps_canvas.update_clim(vmin, vmax)


    def updateScalerExtents(self, update_span: bool = True) -> None:
        '''
        Select a range in the histogram using the vmin and vmax line edits in
        the Navigation Toolbar.

        Parameters
        ----------
        update_span : bool, optional
            Whether to update the span selector as well. The default is True.

        '''
    # Do nothing if the histogram canvas is empty
        if self.canvas.is_empty():
            return
    
    # Retrieve upper and lower range values from the line edits
        vmin = self.scaler_vmin.value()
        vmax = self.scaler_vmax.value()

    # Deal with invalid range values
        if vmax > vmin:
            self.applyScaling(vmin, vmax)
            self.warn_icon.setVisible(False)
        else:
            self.applyScaling(None, None)
            if vmax < vmin:
                self.warn_icon.setVisible(True)

    # Also update the span selector widget if requested
        if update_span:
            if vmax > vmin:
                self.scaler.extents = (vmin, vmax)
                self.scaler.set_visible(True)
            else:
                self.scaler.set_visible(False)
            
            self.canvas.draw()


    def hideScaler(self) -> None:
        '''
        Hides the spanner view from the histogram canvas. Warning: this method
        does not redraw the canvas.

        '''
        self.scaler.set_visible(False)


    def setHistBins(self, value: int) -> None:
        '''
        Set the number of bins of the histogram.

        Parameters
        ----------
        value : int
            Number of bins.

        '''
        self.bin_slider.setToolTip(f'Bins = {value}')
        self.canvas.set_bins(value)


    def extractMaskFromScaler(self) -> None:
        '''
        Extract a mask from the range selected in the histogram scaler and save
        it to file.

        '''
    # Do nothing if no map is displayed in the maps canvas (array invalid)
        arr, legacy_mask = self.maps_canvas.get_unmasked_array(return_mask=True)
        if arr is None:
            return
    
    # Get histogram scaler extents
        vmin, vmax = self.scaler.extents
        vmin, vmax = round(vmin), round(vmax)

    # If the legacy mask exists, merge it with the new mask
        mask_array = np.logical_or(arr < vmin, arr > vmax)
        if legacy_mask is not None:
            mode = settings.manager.get('data/mask_merging_rule')
            mask_array = iatools.binary_merge([mask_array, legacy_mask], mode)
        mask = Mask(mask_array)

    # Save mask file
        ftype = 'Mask (*.msk)'
        outpath = CW.FileDialog(self, 'save', 'Save Mask', ftype).get_path()
        if not outpath:
            return
        try:
            mask.save(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))
            return
            

    def resetConfig(self) -> None:
        '''
        Reset the Histogram Viewer to its default state and configuration.

        '''
        self.loadConfig(config_schemas.HISTOGRAM_DEFAULT_CONFIG)


    def getConfig(self) -> config_schemas.HistogramViewerConfig:
        '''
        Take a snapshot of the current state and configuration of the Histogram
        Viewer.

        Returns
        -------
        config : HistogramViewerConfig
            Current state and configuration of Histogram Viewer.

        '''
        config: config_schemas.HistogramViewerConfig = {
            'Canvas': {
                'Logscale': self.canvas.get_logscale(),
                'Bins': self.canvas.get_bins()
            },
            'NavTbar': {
                'Visible': self.navtbar.showToolbarAction().isChecked(),
                'ActiveMode': self.navtbar.mode.value
            },
            'Scaler': {
                'Enabled': self.scaler_action.isChecked(),
                'Extents': (
                    round(self.scaler.extents[0]), 
                    round(self.scaler.extents[1])
                )
            }
        }
        return config


    def loadConfig(self, config: config_schemas.HistogramViewerConfig) -> None:
        '''
        Set the state and configuration of the Histogram Viewer to those 
        provided in 'config'.

        Parameters
        ----------
        config : HistogramViewerConfig
            Stored state and configuration of Histogram Viewer.

        '''
    # Extract configs
        logscale = config['Canvas']['Logscale']
        nbins = config['Canvas']['Bins']
        show_ntbar = config['NavTbar']['Visible']
        ntbar_mode = config['NavTbar']['ActiveMode']
        scaler_enabled = config['Scaler']['Enabled']
        vmin, vmax = config['Scaler']['Extents']

    # Update navigation toolbar actions
        self.navtbar.showToolbarAction().setChecked(show_ntbar)
        self.navtbar.setVisible(show_ntbar)
        self.navtbar.log_action.setChecked(logscale)
        self.navtbar.mode = type(self.navtbar.mode)(ntbar_mode)
        self.navtbar._update_buttons_checked()

    # Update scaler and scaler toolbar actions
        self.scaler_action.setChecked(scaler_enabled)
        self.scaler_vmin.setValue(vmin)
        self.scaler_vmax.setValue(vmax)
        self.scaler.extents = (vmin, vmax)

    # Update number of bins
        self.bin_slider.setSliderPosition(nbins)
        self.setHistBins(nbins)



class ModeViewer(CW.StyledTabWidget):

    updateSceneRequested = pyqtSignal(CW.DataObject) # current data object

    def __init__(
        self,
        map_canvas: plots.ImageCanvas,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A widget to visualize the modal amounts of the mineral classes that 
        occurr in the mineral map currently displayed in the Data Viewer. It
        includes an interactive legend.

        Parameters
        ----------
        map_canvas : ImageCanvas
            The canvas where the mineral map is displayed.
        parent : QtWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent=parent)

    # Set principal attributes
        self._current_data_object: CW.DataObject | None = None
        self.map_canvas = map_canvas

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.
        
        '''
    # Interactive legend
        self.legend = CW.Legend(context_menu=True)

    # Canvas
        self.canvas = plots.BarCanvas(
            orientation='h', size=(3.6, 6.4), wheel_zoom=False, wheel_pan=False)
        self.canvas.fig.set_linewidth(0)
        self.canvas.setMinimumSize(200, 350)

    # Navigation Toolbar
        self.navtbar = plots.NavTbar.barCanvasDefault(self.canvas, self)
    
    # Wrap canvas and navigation toolbar in a vertical box layout
        plot_vbox = QW.QVBoxLayout()
        plot_vbox.addWidget(self.navtbar)
        plot_vbox.addWidget(self.canvas)

    # Add tabs to the Mode Viewer
        legend_idx = self.wrap_tab(
            self.legend, style.getIcon('LEGEND'), title=None, tight=True)
        plot_idx = self.wrap_tab(
            plot_vbox, style.getIcon('PLOT'), title=None, tight=True)
        self.setTabToolTip(legend_idx, 'Legend')
        self.setTabToolTip(plot_idx, 'Bar plot')


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.
        
        '''

    # Show context menu on the mode canvas after a right-click event
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Connect legend signals
        self.legend.colorChangeRequested.connect(self.onColorChanged)
        self.legend.itemRenameRequested.connect(self.onClassRenamed)
        self.legend.itemsMergeRequested.connect(self.onClassMerged)
        self.legend.itemHighlightRequested.connect(self.onItemHighlighted)
        self.legend.maskExtractionRequested.connect(self.onMaskExtracted)


    def showContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navtbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.canvas.mapToGlobal(point))


    def _update_mode_canvas(self, minmap: MineralMap, title: str = '') -> None:
        '''
        Update the bar canvas that displays the mode data.

        Parameters
        ----------
        minmap : MineralMap
            The mineral map whose mode data must be displayed.
        title : str, optional
            The title of the bar plot. The default is ''.

        '''
        title = self.canvas.ax.get_title() if title == '' else title
        mode_lbl, mode = zip(*minmap.get_labeled_mode().items())
        mode_col = [minmap.get_phase_color(lbl) for lbl in mode_lbl]
        self.canvas.update_canvas((mode,), mode_lbl, mode_col, title=title)


    def get_current_minmap(self) -> MineralMap | None:
        '''
        Get the mineral map held by the currently displayed data object.

        Returns
        -------
        minmap : MineralMap or None
            The mineral map of the currently displayed data object or None if
            no valid mineral map is found.

        '''
        if self._current_data_object is None:
            return
        return self._current_data_object.getMineralMapData()


    def update_viewer(self, data_object: CW.DataObject, title: str = '') -> None:
        '''
        Update all components of the ModeViewer.

        Parameters
        ----------
        data_object : DataObject
            Data object that contains a mineral map.
        title : str, optional
            The title of the mode bar plot. The default is ''.

        '''
    # Do nothing if data object has no valid mineral map data (safety)
        if (minmap := data_object.getMineralMapData()) is None:
            return
        
    # Set current data object
        self._current_data_object = data_object

    # Update the mode canvas
        self._update_mode_canvas(minmap, title)

    # Update the legend
        self.legend.refresh(minmap)


    def clearAll(self) -> None:
        '''
        Reset all components of the ModeViewer.

        '''
        self._current_data_object = None
        self.canvas.clear_canvas()
        self.legend.clear()


    def onColorChanged(
        self,
        item: QW.QTreeWidgetItem,
        color: tuple[int, int, int]
    ) -> None:
        '''
        Alter the displayed color of a class. This method propagates the
        changes to the mineral map, the map canvas, the mode bar plot and the
        legend. The arguments of this method are specifically compatible with
        'colorChangeRequested' signal emitted by the legend (see 'Legend' class
        for more details).

        Parameters
        ----------
        item : QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple[int, int, int]
            RGB triplet. If empty, a random color is generated.

        '''
    # Do nothing if current data object is invalid or contains invalid mineral
    # map data (safety)
        if self._current_data_object is None:
            return
        if (minmap := self._current_data_object.getMineralMapData()) is None:
            return

    # Apply the color change to mineral map (if color is empty, randomize it)
        if not len(color):
            color = minmap.rand_colorlist(1)[0]
        phase_name = item.text(1)
        minmap.set_phase_color(phase_name, color)

    # Update the map canvas colormap, the mode canvas and the legend
        self.map_canvas.alter_colormap(minmap.palette.values())
        self.canvas.change_bar_color(phase_name, color)
        self.legend.changeItemColor(item, color)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onClassRenamed(self, item: QW.QTreeWidgetItem, new_name: str) -> None:
        '''
        Rename a class. This method propagates the changes to the mineral
        map, the map canvas, the mode bar plot and the legend. The arguments of 
        this method are specifically compatible with 'itemRenameRequested' 
        signal emitted by the legend (see 'Legend' class for more details).

        Parameters
        ----------
        item : QTreeWidgetItem
            The legend item that requested to be renamed.
        new_name : str
            New class name.

        '''
    # Do nothing if current data object is invalid or contains invalid mineral
    # map data (safety)
        if self._current_data_object is None:
            return
        if (minmap := self._current_data_object.getMineralMapData()) is None:
            return

    # Rename the phase in the mineral map
        old_name = item.text(1)
        minmap.rename_phase(old_name, new_name)
            
    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onClassMerged(self, classes: list[str], new_name: str) -> None:
        '''
        Merge two or more classes into a new one. This method propagates the 
        changes to the mineral map, the map canvas, the mode bar plot and the 
        legend. The arguments of this method are specifically compatible with
        'itemsMergeRequested' signal emitted by the legend (see 'Legend' class
        for more details).

        Parameters
        ----------
        classes : list[str]
            List of class names.
        new_name : str
            New name for the merged class.

        '''
    # Do nothing if current data object is invalid or contains invalid mineral
    # map data (safety)
        if self._current_data_object is None:
            return
        if (minmap := self._current_data_object.getMineralMapData()) is None:
            return
        
    # Merge phases in the mineral map
        minmap.merge_phases(classes, new_name)

    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onItemHighlighted(self, toggled: bool, item: QW.QTreeWidgetItem) -> None:
        '''
        Highlight on or off the selected mineral class in the map canvas. The 
        arguments of this method are specifically compatible with 
        'itemHighlightRequested' signal emitted by the legend (see 'Legend' 
        class for more details).

        Parameters
        ----------
        toggled : bool
            Highlight state of the class.
        item : QW.QTreeWidgetItem
            The legend item that requested to be highlighted.

        '''
    # Do nothing if current mineral map is invalid (safety)
        if (minmap := self.get_current_minmap()) is None:
            return

    # Change phase visibility based on toggled state
        if toggled:
            vmin = vmax = minmap.as_id(item.text(1))
        else:
            vmin = vmax = None

        self.map_canvas.update_clim(vmin, vmax)


    def onMaskExtracted(self, classes: list[str]) -> None:
        '''
        Extract a mask from a selection of mineral classes and save it to file.

        Parameters
        ----------
        classes : list[str]
            Selected mineral classes.

        '''
    # Do nothing if current mineral map is invalid (safety)
        if (minmap := self.get_current_minmap()) is None:
            return
        
    # Extract the mask
        mask = ~np.isin(minmap.minmap, classes)

    # If a legacy mask exists, merge it with the new mask
        _, legacy_mask = self.map_canvas.get_unmasked_array(return_mask=True)
        if legacy_mask is not None:
            mode = settings.manager.get('data/mask_merging_rule')
            mask = iatools.binary_merge([mask, legacy_mask], mode)

    # Create a new Mask object
        mask = Mask(mask)

    # Save mask file
        ftype = 'Mask (*.msk)'
        outpath = CW.FileDialog(self, 'save', 'Save Mask', ftype).get_path()
        if not outpath:
            return
        try:
            mask.save(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))
            return
            

    def resetConfig(self) -> None:
        '''
        Reset the Mode Viewer to its default state and configuration.

        '''
        self.loadConfig(config_schemas.MODE_DEFAULT_CONFIG)


    def getConfig(self) -> config_schemas.ModeViewerConfig:
        '''
        Take a snapshot of the current state and configuration of the Mode
        Viewer.

        Returns
        -------
        config : ModeViewerConfig 
            Current state and configuration of the Mode Viewer.

        '''
        config: config_schemas.ModeViewerConfig = {
            'NavTbar': {
                'Visible': self.navtbar.showToolbarAction().isChecked(),
                'Labelize': self.navtbar.lbl_action.isChecked()
            }
        }
        return config
    

    def loadConfig(self, config: config_schemas.ModeViewerConfig) -> None:
        '''
        Set the state and configuration of the Mode Viewer to those provided in
        'config'.

        Parameters
        ----------
        config : ModeViewerConfig
            Stored state and configuration of Mode Viewer.

        '''
        show_ntbar = config['NavTbar']['Visible']
        labelize = config['NavTbar']['Labelize']
        self.navtbar.showToolbarAction().setChecked(show_ntbar)
        self.navtbar.setVisible(show_ntbar)
        self.navtbar.lbl_action.setChecked(labelize)



class RoiEditor(QW.QWidget):

    rectangleSelectorUpdated = pyqtSignal()
    autoroiSamplesRequested = pyqtSignal()
    roiDataAltered = pyqtSignal()

    def __init__(
        self,
        maps_canvas: plots.ImageCanvas,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A widget to build, load, edit and save RoiMap objects interactively.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas where ROIs should be drawn and displayed.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Define main attributes
        self.canvas = maps_canvas
        self.current_selection: tuple[int, int, int, int] | None = None
        self._current_roimap: RoiMap | None = None
        self.patches: list[tuple[plots.RoiAnnotation, plots.RoiPatch]] = []

    # ROI selector widget 
        self.rect_sel = plots.RectSel(self.canvas.ax, self.onRectSelect, btns=[1])

    # Load ROIs visual properties
        hex_roi_color = settings.manager.get('plots/roi_color')
        hex_roi_selcolor = settings.manager.get('plots/roi_selcolor')
        self.roi_filled = settings.manager.get('plots/roi_filled')
        self.roi_color = iatools.hex2rgb(hex_roi_color)
        self.roi_selcolor = iatools.hex2rgb(hex_roi_selcolor)

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Toolbar
        self.toolbar = QW.QToolBar('ROI toolbar', self)
        self.toolbar.setStyleSheet(style.SS_TOOLBAR)
        
    #[Define Toolbar Actions]
    #|Load ROI map
        self.load_action = QW.QAction(
            style.getIcon('IMPORT'), 'Import ROI map', self)
    #|Save ROI map
        self.save_action = QW.QAction(
            style.getIcon('SAVE'), 'Save ROI map', self)
    #|Save ROI map as... 
        self.saveas_action = QW.QAction(
            style.getIcon('SAVE_AS'), 'Save ROI map as', self)
    #|Auto detect ROI
        self.autoroi_action = QW.QAction(
            style.getIcon('ROI_SEARCH'), 'Auto detect ROI', self)
    #|Toggle ROI selection
        self.draw_action = QW.QAction(
            style.getIcon('ROI'), 'Draw ROI', self)
        self.draw_action.setCheckable(True)
    #|Add ROI
        self.addroi_action = QW.QAction(
            style.getIcon('ADD_ROI'), 'Add ROI', self)
        self.addroi_action.setEnabled(False)
    #|Extract mask
        self.extr_mask_action = QW.QAction(
            style.getIcon('ADD_MASK'), 'Extract selection mask', self)
        self.extr_mask_action.setEnabled(False)
    #|Set ROI outline color
        self.roicolor_action = QW.QAction('Color...', self)
    #|Set ROI outline selection color
        self.roiselcolor_action = QW.QAction('Selection color...', self)
    #|Toggle filled ROI color 
        self.roifilled_action = QW.QAction('Filled', self)
        self.roifilled_action.setCheckable(True)
        self.roifilled_action.setChecked(self.roi_filled)

    #[Define Toolbar Menu-Actions]
    #|ROI visual preferences 
        pref_menu = QW.QMenu(self.toolbar)
        pref_menu.setStyleSheet(style.SS_MENU)
        pref_menu.addActions((
            self.roicolor_action,
            self.roiselcolor_action,
            self.roifilled_action
        ))
        self.pref_action = QW.QAction(
            style.getIcon('GEAR'), 'ROI settings', self) 
        self.pref_action.setMenu(pref_menu)

    # Populate toolbar
        self.toolbar = QW.QToolBar('ROI toolbar', self)
        self.toolbar.setStyleSheet(style.SS_TOOLBAR)
        self.toolbar.addActions((
            self.load_action,
            self.save_action,
            self.saveas_action
        ))
        self.toolbar.addSeparator()
        self.toolbar.addActions((
            self.autoroi_action,
            self.draw_action,
            self.addroi_action,
            self.extr_mask_action,
            self.pref_action
        ))

    # Auto detect ROI dialog
        self.autoroi_dial = dialogs.AutoRoiDetector()

    # Loaded ROI map path (Path Label)
        self.mappath = CW.PathLabel(full_display=False)

    # Hide ROI map (Checkable Styled Button)
        self.hideroi_btn = CW.StyledButton(style.getIcon('HIDDEN'))
        self.hideroi_btn.setCheckable(True)
        self.hideroi_btn.setToolTip('Hide ROI map')

    # Remove (unload) ROI map (Styled Button)
        self.unload_btn = CW.StyledButton(style.getIcon('CLEAR'))
        self.unload_btn.setToolTip('Clear ROI map')

    # Roi table
        self.table = CW.StyledTable(0, 2)
        selection_behaviour = QW.QAbstractItemView.SelectionBehavior.SelectRows
        self.table.setSelectionBehavior(selection_behaviour)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.setHorizontalHeaderLabels(['Class', 'Pixel Count'])
        self.table.h_header.setSectionResizeMode(style.HRM_STRETCH)
        self.table.v_header.setSectionResizeMode(style.HRM_RESIZETOCONTENTS)        

    # Bar plot canvas
        self.barCanvas = plots.BarCanvas(
            size=(3.6, 2.4), wheel_zoom=False, wheel_pan=False)
        self.barCanvas.set_decimal_precision(0)
        self.barCanvas.fig.set_linewidth(0)
        self.barCanvas.setMinimumSize(300, 300)

    # Bar plot Navigation toolbar
        self.navtbar = plots.NavTbar.barCanvasDefault(self.barCanvas, self)
        
    # Wrap bar plot and its navigation toolbar in a vbox layout
        barplot_vbox = QW.QVBoxLayout()
        barplot_vbox.addWidget(self.navtbar)
        barplot_vbox.addWidget(self.barCanvas)

    # ROI visualizer (Styled Tab Widget -> [ROI table | bar plot])
        roi_visualizer = CW.StyledTabWidget()
        table_idx = roi_visualizer.wrap_tab(
            self.table, style.getIcon('TABLE'), title=None, tight=True)
        plot_idx = roi_visualizer.wrap_tab(
            barplot_vbox, style.getIcon('PLOT'), title=None, tight=True)
        roi_visualizer.setTabToolTip(table_idx, 'Table')
        roi_visualizer.setTabToolTip(plot_idx, 'Bar plot')

    # Adjust Layout
        main_layout = QW.QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toolbar, 0, 0, 1, -1)
        main_layout.addWidget(self.mappath, 1, 0)
        main_layout.addWidget(self.hideroi_btn, 1, 1)
        main_layout.addWidget(self.unload_btn, 1, 2)
        main_layout.addWidget(roi_visualizer, 2, 0, 1, -1)
        main_layout.setColumnStretch(0, 1)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Load ROI map from file
        self.load_action.triggered.connect(lambda: self.loadRoiMap())

    # Save ROI map to file
        self.save_action.triggered.connect(self.saveRoiMap)
        self.saveas_action.triggered.connect(lambda: self.saveRoiMap(True))

    # Launch ROI detector dialog
        self.autoroi_action.triggered.connect(self.autoroi_dial.show)

    # Manage ROI detector dialog signals
        self.autoroi_dial.roiDrawingRequested.connect(self.addAutoRoi)
        self.autoroi_dial.roimapUpdateRequested.connect(
            self.on_autoroi_roimap_requested)
        self.autoroi_dial.sampleUpdateRequested.connect(
            self.autoroiSamplesRequested)

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

    # Manage table signals
        self.table.itemSelectionChanged.connect(self.updatePatchSelection)
        self.table.itemChanged.connect(self.onRoiNameEdited)
        self.table.customContextMenuRequested.connect(self.showTableContextMenu)
        if (model := self.table.model()) is not None:
            model.dataChanged.connect(self.roiDataAltered)
            model.rowsRemoved.connect(self.roiDataAltered)

    # Show custom context menu when right-clicking on the bar canvas
        self.barCanvas.customContextMenuRequested.connect(
            self.showCanvasContextMenu)


    @property
    def current_roimap(self) -> RoiMap:
        '''
        Get the current ROI map.

        Returns
        -------
        roimap : RoiMap
            Current ROI map.

        Raises
        ------
        TypeError
            Raised if current ROI map is None.

        '''
        if (roimap := self._current_roimap) is None:
            raise TypeError('Current ROI map is None.')
        return roimap


    def set_current_roimap(self, roimap: RoiMap):
        '''
        Set a new current ROI map.

        Parameters
        ----------
        roimap : RoiMap
            New ROI map.

        '''
        self._current_roimap = roimap


    def has_roimap(self) -> bool:
        '''
        Check if a ROI map is currently set.

        Returns
        -------
        bool
            Whether a ROI map is set or not.

        '''
        return self._current_roimap is not None


    @property
    def selectedTableIndices(self) -> list[int]:
        '''
        Get a list of the indices of the selected ROIs in the ROIs table.

        Returns
        -------
        selectedIndices : list[int]
            Selected indices.

        '''
        selectedItems = self.table.selectedItems()
    # List slicing to avoid rows idx repetitions (there are 2 columns in table)
        selectedIndices = [item.row() for item in selectedItems[::2]]
        return selectedIndices
    

    @property
    def currentRoiMapPath(self) -> str | None:
        '''
        Get the filepath to the current ROI map.

        Returns
        -------
        str or None
            Filepath to current ROI map.

        '''
        if not self.has_roimap():
            return None
        return self.current_roimap.filepath


    def on_autoroi_roimap_requested(self) -> None:
        '''
        Actions to perform when AutoROI dialog request a ROI map update.

        '''
        roimap = None if not self.has_roimap() else self.current_roimap
        self.autoroi_dial.set_current_roimap(roimap)



    def _redraw(self) -> None:
        '''
        Redraw the canvas and update the cursor of the rectangle selector.

        '''
        self.canvas.draw_idle()
        self.rect_sel.updateCursor()


    def updateBarPlot(self) -> None:
        '''
        Update the bar canvas that displays the cumulative pixel count for each
        drawn ROI.

        '''
    # Clear the canvas if the RoiMap is None or empty
        if not self.has_roimap() or not (cc := self.current_roimap.class_count):
            self.barCanvas.clear_canvas()
    # Otherwise update the canvas
        else:
            names, counts = zip(*cc.items())
            self.barCanvas.update_canvas((counts,), names, title='ROIs Counter')


    def showTableContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu when right-clicking on the ROI table.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Exit function when clicking outside any item
        if self.table.itemAt(point) is None: 
            return

        menu = QW.QMenu(self.table)
        menu.setStyleSheet(style.SS_MENU)

    # Extract mask from selected ROIs
        menu.addAction(
            style.getIcon('ADD_MASK'), 'ROI mask', self.extractMaskFromRois)
        
    # Separator
        menu.addSeparator()

    # Remove selected ROIs
        menu.addAction(style.getIcon('REMOVE'), 'Remove', self.removeRoi)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.table.mapToGlobal(point))


    def showCanvasContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu when right-clicking on the bar plot.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.barCanvas.get_navigation_context_menu(self.navtbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.barCanvas.mapToGlobal(point))


    def onRectSelect(
        self, 
        eclick: plots.mpl_backend_bases.MouseEvent,
        erelease: plots.mpl_backend_bases.MouseEvent
    ) -> None:
        '''
        Callback function for the rectangle selector. It is triggered when
        selection is performed by the user (left mouse button click-release).

        Parameters
        ----------
        eclick : Matplotlib MouseEvent
            Mouse click event.
        erelease : Matplotlib MouseEvent
            Mouse release event.

        '''
    # Save in memory the current selection
        if (array := self.canvas.get_array()) is not None:
            rows, col = array.shape
            self.current_selection = self.rect_sel.fixed_extents((rows, col))

    # Send the signal to inform that the selector has been updated
        self.rectangleSelectorUpdated.emit()


    def selectRoi(self, event: plots.mpl_backend_bases.Event) -> None:
        '''
        Callback function for picking events that can be triggered when the
        rectangle selector is active.

        Parameters
        ----------
        event : Matplotlib Event
            The picking event triggered by clicking on a ROI patch.

        '''
        # Ignore event if not a PickEvent
        if not isinstance(event, plots.mpl_backend_bases.PickEvent):
            return

        # Only process left mouse click
        if event.mouseevent.button == 1:
            patch = event.artist
            _, patches = zip(*self.patches)

        # Select the patch in the table. This will therefore trigger the signal
        # "updatePatchSelection"
            if patch in patches:
                idx = patches.index(patch)
                self.table.selectRow(idx)


    def toggleRectSelect(self, toggled: bool) -> None:
        '''
        Toggle on/off the rectangle selector.

        Parameters
        ----------
        toggled : bool
            Toggle state.

        '''
    # Create a mpl picking event connection if the rectangle is toggled on,
    # otherwise delete it
        if toggled:
            self.pcid = self.canvas.mpl_connect('pick_event', self.selectRoi)
        else:
            self.canvas.mpl_disconnect(self.pcid)

    # Show/hide the rectangle selector
        self.rect_sel.set_active(toggled)
        self.rect_sel.set_visible(toggled)
        self.rect_sel.update()

    # Enable/disable the 'add Roi' and the 'extract mask' actions
        self.addroi_action.setEnabled(toggled)
        self.extr_mask_action.setEnabled(toggled)

    # When the rectangle is activated, the last selection is displayed; so we
    # need to send the signal to inform that the selector has been updated
        self.rectangleSelectorUpdated.emit()


    def updatePatchSelection(self) -> None:
        '''
        Redraw ROI patches on canvas with a different color based on their
        selection state. This must be called whenever a new ROI selection is
        performed or when a new ROI color or selection color is set.

        '''
        selected = self.selectedTableIndices
        for idx, (_, patch) in enumerate(self.patches):
            col = self.roi_selcolor if idx in selected else self.roi_color
            (mpl_col,) = plots.rgb_to_float((col,))
            patch.set(color=mpl_col, linewidth=2 + 2 * (idx in selected))
        self._redraw()


    def onRoiNameEdited(self, item: QW.QTableWidgetItem) -> None:
        '''
        Edit the name of the ROI patch when its linked 'item' in the ROI table
        is renamed.

        Parameters
        ----------
        item : QTableWidgetItem
            The table item that was edited.

        '''
    # Update the ROI map
        idx = item.row()
        name = item.text()
        self.current_roimap.rename_roi(idx, name)

    # Update the patch
        self.editPatchAnnotation(idx, name)

    # Refresh view
        self.updateBarPlot()
        self._redraw()


    def addPatchToCanvas(self, name: str, bbox: tuple[float, float, int, int]) -> None:
        '''
        Add a new ROI to canvas as a new patch and its linked annotation. 
        Warning: this method does not redraw the canvas.

        Parameters
        ----------
        name : str
            The name of the ROI, displayed as annotation.
        bbox : tuple[float, float, int, int]
            The bounding box of the ROI -> (x0, y0, width, height) where x0, y0
            are the coordinates of the top-left corner.

        '''
    # Display the rectangle in canvas
        patch = plots.RoiPatch(bbox, self.roi_color, self.roi_filled)
        patch.set_picker(True)
        self.canvas.ax.add_patch(patch)

    # Display the text annotation in canvas
        text = plots.RoiAnnotation(name, patch)
        self.canvas.ax.add_artist(text)

    # Set annotation and patch visibility
        visible = not self.hideroi_btn.isChecked()
        text.set_visible(visible)
        patch.set_visible(visible)

    # Store the annotation and the patch
        self.patches.append((text, patch))


    def editPatchAnnotation(self, index: int, text: str) -> None:
        '''
        Change text of a ROI patch annotation.

        Parameters
        ----------
        index : int
            The patch index in 'self.patches'.
        text : str
            The new annotation text.

        '''
        annotation = self.patches[index][0]
        annotation.set_text(text)


    def removePatchFromCanvas(self, index: int) -> None:
        '''
        Remove ROI patch from canvas and its linked annotation. It does not
        redraw the canvas.

        Parameters
        ----------
        index : int
            Index of the ROI that must be removed.

        '''
        if len(self.patches):
            text, patch = self.patches.pop(index)
            text.remove()
            patch.remove()


    def getColorFromDialog(
        self,
        old_color: tuple[int, int, int]
    ) -> tuple[int, int, int] | None:
        '''
        Show a dialog to interactively select a color.

        Parameters
        ----------
        old_color : tuple[int, int, int]
            The dialog defaults to this color. Must be provided as RGB triplet.

        Returns
        -------
        tuple[int, int, int] or None
            Selected color as RGB triplet or None if selected color is invalid.

        '''
        col = QW.QColorDialog.getColor(QColor(*old_color), self)
        if col.isValid():
            return (col.red(), col.green(), col.blue())
        return None


    def setRoiColor(self) -> None:
        '''
        Set the color of the ROIs borders when they are not selected.

        '''
        rgb = self.getColorFromDialog(self.roi_color)
        if rgb is not None:
            settings.manager.set('plots/roi_color', iatools.rgb2hex(rgb))
            self.roi_color = rgb
            self.updatePatchSelection()


    def setRoiSelectionColor(self) -> None:
        '''
        Set the color of the ROIs borders when they are selected.

        '''
        rgb = self.getColorFromDialog(self.roi_selcolor)
        if rgb is not None:
            settings.manager.set('plots/roi_selcolor', iatools.rgb2hex(rgb))
            self.roi_selcolor = rgb
            self.updatePatchSelection()


    def setRoiFilled(self, filled: bool) -> None:
        '''
        Set if the ROIs should be filled or unfilled. The filling color is the
        same as the ROIs border color (selected or unselected).

        Parameters
        ----------
        filled : bool
            Whether the ROIs should be filled.

        '''
        self.roi_filled = filled
        settings.manager.set('plots/roi_filled', filled)
        for _, patch in self.patches:
            patch.set_fill(filled)
        self._redraw()
        

    def addRoi(self) -> None:
        '''
        Wrapper method to easily add a new ROI with the extents of the 
        currently selected rectangle. This method also redraws the canvas.

        '''
    # Exit function if canvas is empty (invalid map array)
        if (map_array := self.canvas.get_array()) is None: 
            return

    # Get ROI bbox. Exit function if bbox is invalid (= None)
        rows, cols = map_shape = map_array.shape
        bbox = self.rect_sel.fixed_rect_bbox((rows, cols))
        if bbox is None: 
            return

    # If no ROI map is loaded, then create a new one.
        if not self.has_roimap():
            self.set_current_roimap(RoiMap.from_shape((rows, cols)))
            self.mappath.setPath('*Unsaved ROI map')

    # Send a warning if a ROI map is loaded and has different shape of the map
    # currently displayed in the canvas.
        elif self.current_roimap.shape != map_shape:
            warn_text = (
                'Warning: different map shapes detected. Drawing ROIs will '
                'lead to unpredictable results. Proceed anyway?'
            )
            msgbox = CW.MsgBox(self, 'QuestWarn', warn_text)

        # Exit function if user does not want to procede
            if msgbox.is_choice('No'):
                return

    # Prevent drawing overlapping ROIs
        if (roimap := self.current_roimap).bbox_overlaps(bbox):
            CW.MsgBox(self, 'Crit', 'ROIs cannot overlap.')
            return

    # Show the dialog to type the ROI name
        text = 'Type name (max 8 ASCII characters)'
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), text)

    # Proceed only if the new name is an ASCII <= 8 characters string
        if ok and 0 < len(name) < 9 and name.isascii():
            area = roimap.bbox_area(bbox)
        # Add to roimap
            roimap.add_roi(name, bbox)
        # Add to table
            self.addRoiToTable(name, area)
        # Add to patches list and canvas
            self.addPatchToCanvas(name, bbox)
        # Refresh view
            self.updateBarPlot()
            self._redraw()


    def addRoiToTable(self, name: str, pixel_count: int) -> None:
        '''
        Append ROI to the ROIs table as a new row.

        Parameters
        ----------
        name : str
            ROI name.
        pixel_count : int
            ROI area in pixels.

        '''
    # Define a new table entry (2 TableWidgetItems, one for each table column)
        i0 = QW.QTableWidgetItem(name)
        i1 = QW.QTableWidgetItem(str(pixel_count))
        f = Qt.ItemFlag(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        i1.setFlags(f) # make item not editable

    # Append new row to the table
        new_row = self.table.rowCount()
        self.table.insertRow(new_row)

    # Populate columns with the respective TableWidgetItem (name, pixel_count).
    # Block temporarily the table signals to avoid triggering itemChanged().
        self.table.blockSignals(True)
        self.table.setItem(new_row, 0, i0)
        self.table.setItem(new_row, 1, i1)
        self.table.blockSignals(False)


    def setRoiMapHidden(self, hidden: bool) -> None:
        '''
        Show/hide all ROIs displayed in the canvas. This method also redraws
        the canvas.

        Parameters
        ----------
        hidden : bool
            Whether the ROIs should be hidden.

        '''
        for text, patch in self.patches:
            text.set_visible(not hidden)
            patch.set_visible(not hidden)
        self._redraw()


    def extractMaskFromSelection(self) -> None:
        '''
        Extract and save a mask from current selection.

        '''
    # Exit function if image is empty (invalid array)
        array, legacy_mask = self.canvas.get_unmasked_array(return_mask=True)
        if array is None: 
            return

    # Exit function if there is no valid selection
        rows, cols = array.shape
        extents = self.rect_sel.fixed_extents((rows, cols), fmt='xy') # x0,x1, y0,y1
        if extents is None: 
            return

    # Initialize a new Mask of 1's and invert it to draw 'holes' using extents
        mask = Mask.from_shape((rows, cols), fillwith=1)
        mask.invert_region(extents)

    # If the legacy mask exists, merge it with the new mask
        if legacy_mask is not None:
            mode = settings.manager.get('data/mask_merging_rule')
            mask.mask = iatools.binary_merge([mask.mask, legacy_mask], mode)

    # Save mask file
        ftype = 'Mask (*.msk)'
        outpath = CW.FileDialog(self, 'save', 'Save Mask', ftype).get_path()
        if not outpath:
            return
        try:
            mask.save(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


    def extractMaskFromRois(self) -> None:
        '''
        Extract and save a mask from selected ROIs.

        '''
    # Exit function if no ROI is selected
        if not len(selected := self.selectedTableIndices): 
            return

    # Initialize a new Mask of 1's with the shape of the current ROI map
        roimap = self.current_roimap
        rows, cols = shape = roimap.shape
        mask = Mask.from_shape((rows, cols), fillwith=1)

    # Use the extents of the selected ROIs to draw 'holes' (0's) on the mask
        for idx in selected:
            roi_bbox = roimap.roilist[idx][1]
            extents = roimap.bbox_to_extents(roi_bbox)
            mask.invert_region(extents)

    # If there is a loaded image that has the same shape of the current ROI map
    # and it has a legacy mask, merge it with the new mask
        arr, legacy_mask = self.canvas.get_unmasked_array(return_mask=True)
        if arr is not None and arr.shape == shape and legacy_mask is not None:
            mode = settings.manager.get('data/mask_merging_rule')
            mask.mask = iatools.binary_merge([mask.mask, legacy_mask], mode)

    # Save mask file
        ftype = 'Mask (*.msk)'
        outpath = CW.FileDialog(self, 'save', 'Save Mask', ftype).get_path()
        if not outpath:
            return
        try:
            mask.save(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


    def removeRoi(self) -> None:
        '''
        Wrapper method to easily remove selected ROIs. This method requires a
        user confirm. This method also redraws the canvas.

        '''        
    # Exit function if no ROI is selected
        if not len(selected := self.selectedTableIndices): 
            return

    # Ask for user confirmation
        roimap = self.current_roimap
        msgbox = CW.MsgBox(self, 'Quest', 'Remove selected ROI?')
        if msgbox.is_choice('Yes'):
            for row in sorted(selected, reverse=True):
            # Remove from ROI map
                roimap.del_roi(row)
            # Remove from table
                self.table.removeRow(row)
            # Remove from patches list and canvas
                self.removePatchFromCanvas(row)
        # Refresh view
            self.updateBarPlot()
            self._redraw()


    def removeCurrentRoiMap(self) -> None:
        '''
        Remove the current ROI map, reset the ROIs table and all the patches
        from the canvas. Warning: this method does not redraw the canvas.

        '''
        self._current_roimap = None
        self.mappath.clearPath()
        self.table.setRowCount(0)

    # Remove all patches from canvas
        for idx in reversed(range(len(self.patches))):
            self.removePatchFromCanvas(idx)


    def loadRoiMap(self, path: str | None = None) -> None:
        '''
        Wrapper method to easily load a new ROI map. If a previous ROI map
        exists, it will be removed after user confirm. This method also redraws
        the canvas.

        Parameters
        ----------
        path : str or None, optional
            Filepath to ROI map. If None, user will be prompt to load it from
            disk. The default is None.

        '''
    # Show a warning if a ROI map was already loaded
        if self.has_roimap():
            warn_text = (
                'Loading a new ROI map will discard any unsaved changes made '
                'to the current ROI map. Proceed anyway?'
            )
            msgbox = CW.MsgBox(self, 'QuestWarn', warn_text)

        # Exit function if user does not want to procede
            if msgbox.is_choice('No'):
                return

        # Do nothing if filepath is invalid or the file dialog is canceled
        if path is None:
            ftype = 'ROI maps (*.rmp)'
            path = CW.FileDialog(self, 'open', 'Load ROI', ftype).get_path()
            if not path:
                return
        
    # Remove old (current) ROI map
        pbar = CW.PopUpProgBar(self, 4, 'Loading data')
        self.removeCurrentRoiMap()
        pbar.increase()

    # Load new ROI map
        try:
            self.set_current_roimap(RoiMap.load(path))
            self.mappath.setPath(path)
            pbar.increase()
        except Exception as e:
            pbar.reset()
            CW.MsgBox(self, 'C', f'Unexpected file:\n{path}', str(e))
            return

    # Populate the canvas and the ROIs table with the loaded ROIs
        for name, bbox in (roimap := self.current_roimap).roilist:
            area = roimap.bbox_area(bbox)
            self.addRoiToTable(name, area)
            self.addPatchToCanvas(name, bbox)
        pbar.increase()

    # Refresh view
        self.updateBarPlot()
        self._redraw()
        pbar.increase()


    def unloadRoiMap(self) -> None:
        '''
        Wrapper method to easily remove from memory the currently loaded ROI
        map, if it exists. This method requires a user confirm if ROI map has
        unsaved data. This method also redraws the canvas.

        '''
    # Exit function if no ROI map is loaded
        if not self.has_roimap():
            return
    
    # Ask for user confirmation
        if self.isRoiMapUnsaved():
            warn_text = 'Remove current ROI map? Unsaved changes will be lost.'
            msgbox = CW.MsgBox(self, 'QuestWarn', warn_text)
            if msgbox.is_choice('No'):
                return
    
    # Unload ROI map and redraw the canvas
        self.removeCurrentRoiMap()
        self.updateBarPlot()
        self._redraw()


    def saveRoiMap(self, save_as: bool = False) -> None:
        '''
        Save the current ROI map to file. If the file already exists, it will
        be overwritten. ROI map can still be saved as a new file if 'save_as'
        is True.

        Parameters
        ----------
        save_as : bool, optional
            Whether the ROI map should be saved to a new file. The default is
            False.

        '''
    # Exit function if the current ROI map does not exist
        if not self.has_roimap(): 
            return

    # Save ROI map to a new file if it was requested (save_as = True) and/or
    # if it was never saved before (= it has not a valid filepath). Otherwise,
    # save it to its current filepath (overwrite).
        if not save_as and (path := self.currentRoiMapPath) is not None:
            outpath = path
        else:
            ftype = 'ROI maps (*.rmp)'
            outpath = CW.FileDialog(self, 'save', 'Save ROI', ftype).get_path()
            if not outpath:
                return
 
        try:
            self.current_roimap.save(outpath)
            self.mappath.setPath(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save ROI map.', str(e))


    def isRoiMapUnsaved(self):
        '''
        Check if current ROI map has unsaved data.

        Returns
        -------
        bool
            Whether current ROI map data is unsaved.

        '''
    # If no ROI map is loaded, it cannot be unsaved
        if not self.has_roimap():
            return False
    
    # If current ROI map has no valid filepath, it is unsaved
        if (roimap_path := self.currentRoiMapPath) is None:
            return True
        
    # If current ROI map has a valid but broken filepath, it is unsaved
        try:
            saved_roimap = RoiMap.load(roimap_path)
        except Exception: # path deleted, moved, renamed or with broken data
            return True 

    # Check if saved ROI map data differs from current ROI map data
        return saved_roimap != self.current_roimap


    def updateDetectorSamplesList(self, samples: list[CW.DataGroup]) -> None:
        '''
        Update the list of available samples within the "ROI detector" dialog.
        This method is called by the main window, by accessing sample data from
        the Data Manager.

        Parameters
        ----------
        samples : list[CW.DataGroup]
            List of available samples.

        '''
        self.autoroi_dial.maps_selector.updateCombox(samples)


    def addAutoRoi(self, auto_roimap: RoiMap) -> None:
        '''
        Populate the current ROI map with the auto-traced ROIs that are stored
        in 'auto_roimap'. This should only be called when the "ROI detector"
        dialog is closed.

        Parameters
        ----------
        auto_roimap : RoiMap
            ROI map whose ROIs have been traced automatically.

        '''
    # Create a new ROI map if there was no existent ROI map
        if not self.has_roimap():
            self.mappath.setPath('*Unsaved ROI map')
            self.set_current_roimap(auto_roimap)
    
    # Otherwise add the new ROIs to the current ROI map
        else:
            # We can save some time by switching off safe mode. We are already
            # sure that the new ROIs do NOT overlap with the current ones.
            self.current_roimap.overwrite_roimap(auto_roimap, safe=False)

    # Add new ROIs to table and canvas
        for name, bbox in auto_roimap.roilist:
            area = auto_roimap.bbox_area(bbox)
            self.addRoiToTable(name, area)
            self.addPatchToCanvas(name, bbox)
    
    # Refresh view
        self.updateBarPlot()
        self._redraw()


    def resetConfig(self) -> None:
        '''
        Reset the ROI Editor to its default state and configuration.

        '''
        self.loadConfig(config_schemas.ROIEDITOR_DEFAULT_CONFIG)


    def getConfig(self) -> config_schemas.RoiEditorConfig:
        '''
        Take a snapshot of the current state and configuration of the ROI 
        Editor.

        Returns
        -------
        config : RoiEditorConfig
            Current state and configuration of ROI Editor.

        '''
        config: config_schemas.RoiEditorConfig = {
            'Selector': {
                'Extents': self.rect_sel.extents,
                'FixedExtents': self.current_selection,
                'Active': self.rect_sel.active
            },
            'RoiMap': {
                'Path': self.currentRoiMapPath,
                'Visible': not self.hideroi_btn.isChecked()
            },
            'NavTbar': {
                'Visible': self.navtbar.showToolbarAction().isChecked(),
                'Labelize': self.navtbar.lbl_action.isChecked()
            }
        }
        return config


    def loadConfig(self, config: config_schemas.RoiEditorConfig) -> None:
        '''
        Set the state and configuration of the ROI Editor to those provided in
        'config'.

        Parameters
        ----------
        config : RoiEditorConfig
            Stored state and configuration of ROI Editor.

        '''
    # Extract configs
        extents = config['Selector']['Extents']
        fixed_extents = config['Selector']['FixedExtents']
        selector_active = config['Selector']['Active']
        roimap_path = config['RoiMap']['Path']
        roimap_visible = config['RoiMap']['Visible']
        show_ntbar = config['NavTbar']['Visible']
        labelize = config['NavTbar']['Labelize']

    # Update ROI selector (requires ImageCanvas' extents to be updated first)
        self.rect_sel.extents = extents
        self.current_selection = fixed_extents
        self.draw_action.setChecked(selector_active)
        self.rect_sel.set_active(selector_active)
        self.rect_sel.set_visible(selector_active)

    # Update ROI map and linked table and bar plot
        self.removeCurrentRoiMap()
        self.hideroi_btn.setChecked(not roimap_visible)

        if roimap_path is None:
            self.barCanvas.clear_canvas()        
            self._redraw()
        else:
            self.loadRoiMap(roimap_path) # this also updates canvas and barplot

    # Update bar plot navigation toolbar
        self.navtbar.showToolbarAction().setChecked(show_ntbar)
        self.navtbar.setVisible(show_ntbar)
        self.navtbar.lbl_action.setChecked(labelize)



class ProbabilityMapViewer(QW.QWidget):

    probabilityRangeChanged = pyqtSignal()

    def __init__(
        self,
        maps_canvas: plots.ImageCanvas,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A widget to visualize the probability map linked with the mineral map 
        that is currently displayed in the data viewer..

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas that displays the mineral map which is linked to the
            probability map displayed in this widget.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Define main attribute
        self.maps_canvas = maps_canvas

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor

        '''
    # Canvas
        self.canvas = plots.ImageCanvas()
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navtbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # View Range toolbar
        self.rangeTbar = QW.QToolBar('Probability range toolbar')
        self.rangeTbar.setStyleSheet(style.SS_TOOLBAR)

    # Toggle range selection [-> View Range Toolbar]
        self.toggle_range_action = QW.QAction(
            style.getIcon('RANGE'), 'Set range', self)
        self.toggle_range_action.setCheckable(True)

    # Extract mask from range [-> View Range Toolbar]
        self.mask_action = QW.QAction(
            style.getIcon('ADD_MASK'), 'Extract mask', self)
        self.mask_action.setEnabled(False)

    # Range values inputs [-> View Range Toolbar]
        self.min_input = QW.QDoubleSpinBox()
        self.min_input.setValue(0.0)
        self.min_input.setToolTip('Min. range value')
        self.max_input = QW.QDoubleSpinBox()
        self.max_input.setValue(1.0)
        self.max_input.setToolTip('Max. range value')
        for wid in (self.min_input, self.max_input):
            wid.setMaximum(1.0)
            wid.setSingleStep(0.01)
            wid.setMaximumWidth(100)
            wid.setEnabled(False)

    # Range warning icon [-> Heatmap scaler toolbar]
        warn_pixmap = QPixmap(str(style.ICONS.get('WARNING')))
        warn_lbl = QW.QLabel(self.rangeTbar)
        warn_lbl.setPixmap(warn_pixmap.scaled(20, 20, style.ARM_KEEP))
        warn_lbl.setSizePolicy(style.SP_MAXIMUM, style.SP_MAXIMUM)
        warn_lbl.setToolTip('Lower limit is bigger or equal to upper limit.')
        self.warn_icon = QW.QWidgetAction(self)
        self.warn_icon.setDefaultWidget(warn_lbl)
        self.warn_icon.setVisible(False)

    # Add actions and widgets to View Range Toolbar
        self.rangeTbar.addAction(self.toggle_range_action)
        self.rangeTbar.addAction(self.mask_action)
        self.rangeTbar.addWidget(self.min_input)
        self.rangeTbar.addWidget(self.max_input)
        self.rangeTbar.addAction(self.warn_icon)

    # Adjust layout
        main_layout = QW.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.navtbar)
        main_layout.addWidget(self.rangeTbar)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector
        
        '''
    # Context menu on canvas
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Toggle view range action
        self.toggle_range_action.toggled.connect(self.onViewRangeToggled)

    # Set min and max range actions
        self.min_input.valueChanged.connect(self.onViewRangeChanged)
        self.max_input.valueChanged.connect(self.onViewRangeChanged)

    # Extract mask action
        self.mask_action.triggered.connect(self.extractMaskFromRange)


    def showContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navtbar)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.canvas.mapToGlobal(point))


    def onViewRangeToggled(self, toggled: bool) -> None:
        '''
        Slot triggered when the view range is toggled on/off in the probability
        range toolbar.

        Parameters
        ----------
        toggled : bool
            Toggle state of the view range.

        '''
    # Enable/disable all functions in the View Range Toolbar
        self.mask_action.setEnabled(toggled)
        self.min_input.setEnabled(toggled)
        self.max_input.setEnabled(toggled)
    
    # Change the view range or reset it to default values
        if toggled:
            self.updateViewRange()
        else:
            self.canvas.update_clim()


    def onViewRangeChanged(self) -> None:
        '''
        Slot triggered when the upper or the lower bound of the probability
        view range are modified from the respective widgets in the probability
        range toolbar.

        '''
        self.updateViewRange()
        self.probabilityRangeChanged.emit()
    

    def updateViewRange(self) -> None:
        '''
        Change the view range (clims) of the probability map.

        '''
    # Exit function if no probability map is displayed
        if self.canvas.is_empty():
            return
    
    # If range values are correct change the clims, otherwise send warning
        vmin, vmax = self.min_input.value(), self.max_input.value()
        if vmin < vmax:
            self.canvas.update_clim(vmin, vmax)
            self.warn_icon.setVisible(False)
        else:
            self.warn_icon.setVisible(True)


    def extractMaskFromRange(self) -> None:
        '''
        Extract mask from current view range.

        '''
    # Exit function if no probability map is displayed (array invalid)
        if (pmap_array := self.canvas.get_unmasked_array()) is None:
            return
    
    # Get the range values from the view range toolbar widgets
        vmin, vmax = self.min_input.value(), self.max_input.value()
        
    # Extract the current mask (legacy mask)
        _, legacy_mask = self.maps_canvas.get_unmasked_array(return_mask=True)

    # If the legacy mask exists, merge it with the new mask
        mask_array = np.logical_or(pmap_array < vmin, pmap_array > vmax)
        if legacy_mask is not None:
            mode = settings.manager.get('data/mask_merging_rule')
            mask_array = iatools.binary_merge([mask_array, legacy_mask], mode)
        mask = Mask(mask_array)

    # Save mask file
        ftype = 'Mask (*.msk)'
        outpath = CW.FileDialog(self, 'save', 'Save Mask', ftype).get_path()
        if not outpath:
            return
        try:
            mask.save(outpath)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


    def resetConfig(self) -> None:
        '''
        Reset the Probability Map Viewer to its default state and configuration.

        '''
        self.loadConfig(config_schemas.PROBABILITY_DEFAULT_CONFIG)


    def getConfig(self) -> config_schemas.ProbabilityViewerConfig:
        '''
        Take a snapshot of the current state and configuration of the 
        Probability Map Viewer.

        Returns
        -------
        config : ProbabilityViewerConfig
            Current state and configuration of Probability Map Viewer.

        '''
        config: config_schemas.ProbabilityViewerConfig = {
            'NavTbar': {
                'Visible': self.navtbar.showToolbarAction().isChecked(),
                'ActiveMode': self.navtbar.mode.value
            },
            'ViewRange': {
                'Enabled': self.toggle_range_action.isChecked(),
                'Range': (self.min_input.value(), self.max_input.value())
            }
        }
        return config
    

    def loadConfig(self, config: config_schemas.ProbabilityViewerConfig) -> None:
        '''
        Set the state and configuration of the Probability Map Viewer to those
        provided in 'config'.

        Parameters
        ----------
        config : ProbabilityViewerConfig
            Stored state and configuration of Probability Map Viewer.

        '''
    # Extract configs
        show_ntbar = config['NavTbar']['Visible']
        ntbar_mode = config['NavTbar']['ActiveMode']
        view_range_enabled = config['ViewRange']['Enabled']
        vmin, vmax = config['ViewRange']['Range']

    # Update navigation toolbar actions
        self.navtbar.showToolbarAction().setChecked(show_ntbar)
        self.navtbar.setVisible(show_ntbar)
        self.navtbar.mode = type(self.navtbar.mode)(ntbar_mode)
        self.navtbar._update_buttons_checked()

    # Update view range toolbar actions
        self.toggle_range_action.setChecked(view_range_enabled)
        self.min_input.setValue(vmin)
        self.max_input.setValue(vmax)


RgbaChannelName = Literal['R', 'G', 'B', 'A']
class RgbaCompositeMapViewer(QW.QWidget):

    channels = get_args(RgbaChannelName)
    rgbaModified = pyqtSignal()

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A widget to visualize an RGB(A) composite map extracted from the
        combination of input maps.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Canvas
        self.canvas = plots.ImageCanvas(cbar=False)
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navtbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # R-G-B-A Path Labels
        self.rgba_lbls = [CW.PathLabel(full_display=False) for _ in range(4)]
        channels_layout = QW.QGridLayout()

        for col, lbl in enumerate(self.rgba_lbls):
            channel_name = QW.QLabel(self.channels[col])
            channel_name.setAlignment(style.A_HCENTER)
            channels_layout.addWidget(channel_name, 0, col)
            channels_layout.addWidget(lbl, 1, col)

    # Adjust layout
        main_layout = QW.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.navtbar)
        main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addLayout(channels_layout)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector
        
        '''
    # Context menu on canvas
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)


    def showContextMenu(self, point: QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navtbar)
        menu.addSeparator()

    # Define the "Clear channel" sub-menu
        clear_submenu = QW.QMenu('Clear channel...', menu)
        menu.addMenu(clear_submenu)

    #[Populate sub-menu]
    #|Clear each individual channel
        for c in self.channels:
            clear_submenu.addAction(
                f'{c} channel', lambda c=c: self.clearChannel(c))
    #|Separator
        clear_submenu.addSeparator()
    #|Clear all channels
        clear_submenu.addAction('Clear all', self.clearAllChannels)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.canvas.mapToGlobal(point))


    def clearChannel(self, channel: RgbaChannelName) -> None:
        '''
        Clear provided RGBA channel.

        Parameters
        ----------
        channel : RgbaChannelName
            Channel to be cleared. Valid channels are 'R', 'G', 'B' or 'A'.

        '''
    # Do nothing if canvas is empty (invalid RGBA map array)
        if (rgba_map_array := self.canvas.get_array()) is None:
            return
        
    # Default "empty" values for each RGBA channel are: R=0, B=0, G=0, A=1
        idx = self.channels.index(channel)
        rgba_map_array[:, :, idx] = 1 if idx == 3 else 0
        self.rgba_lbls[idx].clearPath()

    # If no loaded channel is left, clear the canvas; otherwise redraw it
        if all(map(lambda lbl: lbl.fullpath == '', self.rgba_lbls)):
            self.canvas.clear_canvas()
        else:
            self.canvas.draw_heatmap(rgba_map_array, 'RGBA composite map')
        
    # Emit signal
        self.rgbaModified.emit()


    def clearAllChannels(self) -> None:
        '''
        Clear all RGBA channels.

        '''
    # Do nothing if canvas is empty
        if self.canvas.is_empty():
            return
        
    # Clear canvas and all path labels
        self.canvas.clear_canvas()
        for lbl in self.rgba_lbls:
            lbl.clearPath()

    # Emit signal
        self.rgbaModified.emit()


    def setChannel(self, channel: RgbaChannelName, inmap: InputMap) -> None:
        '''
        Set one channel's data.

        Parameters
        ----------
        channel : RgbaChannelName
            One of 'R', 'G', 'B' or 'A'.
        inmap : InputMap
            Input map to be set as channel 'channel'.

        '''
    # Do nothing if 'channel' is not a valid channel (safety)
        if channel not in self.channels:
            return

    # Get the current rgba map array or build a new one
        if (rgba_map_array := self.canvas.get_array()) is None:
            rgba_map_array = np.zeros((*inmap.shape, 4))
            rgba_map_array[:, :, 3] = 1   # invert A channel (opacity)

    # Exit function if the Input Map shape does not fit the RGBA map shape
        if inmap.shape != rgba_map_array.shape[:2]:
            err_txt = 'This map does not fit within the current RGBA map.'
            CW.MsgBox(self, 'Crit', err_txt)
            return

    # Update the channel with the new data and update the plot
        idx = self.channels.index(channel)
        data = inmap.map
        rgba_map_array[:, :, idx] = np.round(data/data.max(), 2)
        self.canvas.draw_heatmap(rgba_map_array, 'RGBA composite map')

    # Update the channel path label
        self.rgba_lbls[idx].setPath(inmap.filepath)

    # Send signal
        self.rgbaModified.emit()


    def resetConfig(self) -> None:
        '''
        Reset RGBA Composite Map Viewer to its default state and configuration.

        '''
        self.loadConfig(config_schemas.RGBA_VIEWER_CONFIG)


    def getConfig(self) -> config_schemas.RgbaViewerConfig:
        '''
        Take a snapshot of the current state and configuration of the RGBA
        Composite Map Viewer.

        Returns
        -------
        config : RgbaViewerConfig
            Current state and configuration of RGBA Composite Map Viewer.

        '''
        config: config_schemas.RgbaViewerConfig = {
            'NavTbar': {
                'Visible': self.navtbar.showToolbarAction().isChecked(),
                'ActiveMode': self.navtbar.mode.value
            },
            'Channels': {
                'R': self.rgba_lbls[0].fullpath,
                'G': self.rgba_lbls[1].fullpath,
                'B': self.rgba_lbls[2].fullpath,
                'A': self.rgba_lbls[3].fullpath
            }
        }
        return config


    def loadConfig(self, config: config_schemas.RgbaViewerConfig) -> None:
        '''
        Set the state and configuration of the RGBA Composite Map Viewer to 
        those provided in 'config'.

        Parameters
        ----------
        config : RgbaViewerConfig
            Stored state and configuration of RGBA Composite Map Viewer.

        '''
    # Update navigation toolbar actions
        show_ntbar = config['NavTbar']['Visible']
        ntbar_mode = config['NavTbar']['ActiveMode']
        self.navtbar.showToolbarAction().setChecked(show_ntbar)
        self.navtbar.setVisible(show_ntbar)
        self.navtbar.mode = type(self.navtbar.mode)(ntbar_mode)
        self.navtbar._update_buttons_checked()

    # Update R, G, B, A channels
        self.clearAllChannels()
        for ch in self.channels:
            path = config['Channels'][ch]
            if os.path.exists(path):
                try:
                    self.setChannel(ch, InputMap.load(path))
                except ValueError: # triggered by InputMap.load()
                    continue