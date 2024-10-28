# -*- coding: utf-8 -*-
"""
Created on Tue Mar  26 11:25:14 2024

@author: albdag
"""

import os

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QCursor, QIcon, QPixmap
from PyQt5 import QtWidgets as QW

import numpy as np

from _base import InputMap, Mask, MineralMap, RoiMap
import convenient_functions as cf
import custom_widgets as CW
import image_analysis_tools as iatools
import dialogs
import plots
import preferences as pref



class Pane(QW.QDockWidget):
    '''
    The main class for every pane of X-Min Learn. It is a customized version
    of a QDockWidget.
    '''

    def __init__(self, widget: QW.QWidget, title='', icon=None, scroll=True):
        '''
        Constructor.

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
            Wether or not the pane should be scrollable. The default is True.

        '''
        super(Pane, self).__init__()
    
    # Set widget properties
        self.setWindowTitle(title)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    
    # Set main attributes
        self.title = title
        self.icon = icon
        self._scrollable = scroll

    # Wrap the widget into a container widget (scroll area or group area)
        if scroll:
            scroll_area = CW.GroupScrollArea(widget)
            scroll_area.setStyleSheet(None)
            scroll_area.setObjectName('PaneScrollArea') # Name for custom qss
            self.setWidget(scroll_area)

        else:
            group_area = CW.GroupArea(widget, tight=True)
            group_area.setStyleSheet(None)
            group_area.setObjectName('PaneGroupArea')  # Name for custom qss
            self.setWidget(group_area)

    # Set the style-sheet 
        self.setStyleSheet(pref.SS_pane)


    def trueWidget(self):
        '''
        Convenient function to return the actual pane widget and not just its
        container widget, which is returned when invoking the default widget()
        function.

        Returns
        -------
        QWidget
            The widget of the pane.

        '''
        if self._scrollable:
            return self.widget().widget()
        else:
            return self.widget().layout().itemAt(0).widget()




class DataManager(QW.QTreeWidget):
    '''
    A widget for loading, accessing and managing input and output data.

    '''

    updateSceneRequested = pyqtSignal(object)
    clearSceneRequested = pyqtSignal()
    rgbaChannelSet = pyqtSignal(str)


    def __init__(self, parent=None):
        '''
        Constructor.
        
        parent : QtWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''

        super(DataManager, self).__init__(parent)

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
        self.setHorizontalScrollBar(CW.StyledScrollBar(Qt.Horizontal))
        self.setVerticalScrollBar(CW.StyledScrollBar(Qt.Vertical))

    # Set the style-sheet (custom icons for expanded and collapsed branches and
    # right-click menu when editing items name)
        self.setStyleSheet(pref.SS_dataManager)


    def onEdit(self, item, column=0):
        '''
        Force item editing in first column. DataSubGroup objects are excluded
        from editing.

        Parameters
        ----------
        item : DataGroup, DataSubGroup or DataObject
            The item that requests editing.
        column : int, optional
            The column where edits are requested. If different than 0, editing
            will be forced to be on first column anyway. The default is 0.

        '''
        if isinstance(item, (CW.DataGroup, CW.DataObject)):
            self.editItem(item, 0)


    def showContextMenu(self, point):
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Define a menu (Styled Menu)
        menu = CW.StyledMenu()

    # Get the item that is clicked at <point> and the group it belongs to
        item = self.itemAt(point)
        group = self.getItemParentGroup(item)

    # CONTEXT MENU ON VOID (<point> is not on an item)
        if item is None:

        # Add new group
            menu.addAction(QIcon(r'Icons/generic_add.png'), 'New sample',
                           self.addGroup)
        
        # Separator
            menu.addSeparator()

        # Clear all
            menu.addAction(QIcon(r'Icons/remove.png'), 'Delete all', 
                           self.clearAll)

    # CONTEXT MENU ON GROUP
        elif isinstance(item, CW.DataGroup):

        # Load data submenu
            load_submenu = menu.addMenu(QIcon(r'Icons/import.png'), 'Load...')
        # - Input maps
            load_submenu.addAction(QIcon(r'Icons/inmap.png'), 'Input maps', 
                                   lambda: self.loadInputMaps(item))
        # - Mineral maps
            load_submenu.addAction(QIcon(r'Icons/minmap.png'), 'Mineral maps',
                                   lambda: self.loadMineralMaps(item))
        # - Masks
            load_submenu.addAction(QIcon(r'Icons/mask.png'), 'Masks',
                                   lambda: self.loadMasks(item))
            
        # Separator
            menu.addSeparator()

        # Rename group
            menu.addAction(QIcon(r'Icons/rename.png'), 'Rename',
                           lambda: self.onEdit(item))

        # Clear group
            menu.addAction(QIcon(r'Icons/clear.png'), 'Clear', self.clearGroup)

        # Delete group
            menu.addAction(QIcon(r'Icons/remove.png'), 'Delete', self.delGroup)

    # CONTEXT MENU ON SUBGROUP
        elif isinstance(item, CW.DataSubGroup):

        # Load data
            menu.addAction(QIcon(r'Icons/generic_add.png'), 'Load', 
                           lambda: self.loadData(item))
        
        # Separator
            menu.addSeparator()

        # Specific actions for MASKS subgroup
            if item.name == 'Masks':

            # Check all masks
                chk_mask = menu.addAction('Check all')
                chk_mask.setEnabled(not item.isEmpty())
                chk_mask.triggered.connect(lambda: self.checkMasks(1, group))

            # Uncheck all masks
                unchk_mask = menu.addAction('Uncheck all')
                unchk_mask.setEnabled(not item.isEmpty())
                unchk_mask.triggered.connect(lambda: self.checkMasks(0, group))

            # Separator
                menu.addSeparator()

        # Clear subgroup
            clear_subgroup = menu.addAction(QIcon(r'Icons/clear.png'), 'Clear')
            clear_subgroup.triggered.connect(item.clear)
            clear_subgroup.triggered.connect(self.clearView)

    # CONTEXT MENU ON DATA OBJECTS
        elif isinstance(item, CW.DataObject):

        # Rename item
            menu.addAction(QIcon(r'Icons/rename.png'), 'Rename',
                           lambda: self.onEdit(item))

        # Delete item
            menu.addAction(QIcon(r'Icons/remove.png'), 'Remove', self.delData)

        # Separator
            menu.addSeparator()

        # Refresh data source
            menu.addAction(QIcon(r'Icons/refresh.png'), 'Refresh data source',
                           self.refreshDataSource)

        # Save
            menu.addAction(QIcon(r'Icons/save.png'), 'Save',
                           lambda: self.saveData(item))

        # Save As
            menu.addAction(QIcon(r'Icons/save_as.png'), 'Save As...',
                           lambda: self.saveData(item, False))

        # Separator
            menu.addSeparator()

        # Specific actions when item holds INPUT MAPS
            if item.holdsInputMap():

            # Invert Map
                menu.addAction(QIcon(r'Icons/invert.png'), 'Invert',
                               self.invertInputMap)

            # RGBA submenu
                rgba_submenu = menu.addMenu(QIcon(r'Icons/rgba_2.png'),
                                            'Set as RGBA channel...')
                for c in ('R', 'G', 'B', 'A'):
                    rgba_submenu.addAction(
                        f'{c} channel', lambda c=c: self.rgbaChannelSet.emit(c))

        # Specific actions when item holds MINERAL MAPS
            elif item.holdsMineralMap():

            # Export Array
                menu.addAction(QIcon(r'Icons/export.png'), 'Export map',
                               lambda: self.exportMineralMap(item))

        # Specific actions when item holds MASKS
            elif item.holdsMask():

            # Invert mask
                menu.addAction(QIcon(r'Icons/invert.png'), 'Invert mask',
                               self.invertMask)

            # Merge masks sub-menu
                mergemask_submenu = menu.addMenu('Merge masks')
                mergemask_submenu.addAction('Union', 
                                            lambda: self.mergeMasks(group,'U'))
                mergemask_submenu.addAction('Intersection',
                                            lambda: self.mergeMasks(group,'I'))

        # add specific actions when item holds point data


    # Deal with anything else, just for safety reasons
        else: 
            return

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
        if isinstance(item, CW.DataGroup):
            group = item
    # Item is subgroup
        elif isinstance(item, CW.DataSubGroup):
            group = item.parent()
    # Item is data object
        elif isinstance(item, CW.DataObject):
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
        groups_idx: list
            List of indices of selected groups.

        '''
        items = self.selectedItems()
    # Groups are the top level items of the DataManager.
    # IndexOfTopLevelItem returns -1 if the item is not a toplevelitem
        indexes = map(lambda i: self.indexOfTopLevelItem(i), items)
        groups_idx = filter(lambda idx: idx != -1, indexes)
        return  groups_idx


    def getSelectedDataObjects(self):
        '''
        Get the selected data objects (i.e., instances of DataObject).

        Returns
        -------
        data_obj : list
            The selected data objects.

        '''
        items = self.selectedItems()
        data_obj = [i for i in items if isinstance(i, CW.DataObject)]
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
        DataGroup or None.

        '''
    # Automatically rename the group as 'New Sample' + a progressive integer id
        unnamed_groups = self.findItems('New Sample', Qt.MatchContains)
        text = 'New Sample'
        if (n := len(unnamed_groups)): 
            text += f' ({n})'
        new_group = CW.DataGroup(text)
        self.addTopLevelItem(new_group)
        if return_group: 
            return new_group
        

    def delGroup(self):
        '''
        Remove selected groups (i.e., instances of DataGroup) from the 
        manager.

        '''

        selected = self.getSelectedGroupsIndexes()
        choice = CW.MsgBox(self, 'Quest', 'Remove selected sample(s)?')
        if choice.yes():
            for idx in sorted(selected, reverse=True):
                self.takeTopLevelItem(idx)

        self.clearView()


    def clearGroup(self):
        '''
        Clear the data from selected groups (i.e., instances of DataGroup).

        '''
        selected = self.getSelectedGroupsIndexes()
        for idx in selected:
            group = self.topLevelItem(idx)
            group.clear()
        self.clearView()


    def delData(self):
        '''
        Delete the selected data objects (i.e., instances of DataObject).

        '''
        items = self.getSelectedDataObjects()
        choice = CW.MsgBox(self, 'Quest', 'Remove selected data?')
        if choice.yes():
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
            # implement
            pass

        else: return


    def saveData(self, item: CW.DataObject, overwrite=True):
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

        item_data = item.get('data')
        path = None

    # If overwrite is True, check if the original path still exists. If not,
    # set path to None. If it exists, double check if user really wants to
    # overwrite the file
        if overwrite:
            path = item_data.filepath
            if path is not None and os.path.exists(path):
                choice = CW.MsgBox(self, 'Quest', 'Overwrite this file?')
                if choice.no(): 
                    return
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
            # add PointAnalysis
            else: 
                return # safety

            path, _ = QW.QFileDialog.getSaveFileName(self, 'Save Map',
                                                     pref.get_dirPath('out'),
                                                     filext)

    # Finally, if there is a valid path, save the data
        if path:
            pref.set_dirPath('out', os.path.dirname(path))
            try:
                item_data.save(path)
            # Set the item edited status to False
                item.setEdited(False)
            except Exception as e:
                return CW.MsgBox(self, 'Crit', 'Failed to save file.', str(e))


    def loadInputMaps(self, group: CW.DataGroup, paths: list|None=None):
        '''
        Specialized loading function to load input maps to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list or None, optional
            A list of filepaths to data. If None, user will be prompt to load
            them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load input maps',
                                                       pref.get_dirPath('in'),
                                                       'ASCII maps (*.txt *.gz)')
        if not paths:
            return
        
        pref.set_dirPath('in', os.path.dirname(paths[0]))
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        for n, p in enumerate(paths, start=1):
            if pbar.wasCanceled(): 
                break
            try:
                xmap = InputMap.load(p)
                group.inmaps.addData(xmap)

            except Exception as e:
                pbar.setWindowModality(Qt.NonModal)
                CW.MsgBox(self, 'Crit', f'Unexpected file:\n{p}.', str(e))
                pbar.setWindowModality(Qt.WindowModal)

            finally:
                pbar.setValue(n)

        self.expandRecursively(self.indexFromItem(group))


    def loadMineralMaps(self, group: CW.DataGroup, paths: list|None=None):
        '''
        Specialized loading function to load mineral maps to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list or None, optional
            A list of filepaths to data. If None, user will be prompt to load
            them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load mineral maps',
                                                       pref.get_dirPath('in'),
                                                       '''Mineral maps (*.mmp)
                                                          Legacy mineral maps (*.txt *.gz)''')
        if not paths:
            return
        
        pref.set_dirPath('in', os.path.dirname(paths[0]))
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        for n, p in enumerate(paths, start=1):
            if pbar.wasCanceled(): 
                break
            try:
                mmap = MineralMap.load(p)
                # Convert legacy mineral maps to new file format (mmp)
                if mmap.is_obsolete():
                    mmap.save(cf.extend_filename(p, '', '.mmp'))
                group.minmaps.addData(mmap)

            except Exception as e:
                pbar.setWindowModality(Qt.NonModal)
                CW.MsgBox(self, 'Crit', f'Unexpected file:\n{p}.', str(e))
                pbar.setWindowModality(Qt.WindowModal)

            finally:
                pbar.setValue(n)

        self.expandRecursively(self.indexFromItem(group))


    def loadMasks(self, group: CW.DataGroup, paths: list|None=None):
        '''
        Specialized loading function to load masks to a group (i.e., an
        instance of DataGroup).

        Parameters
        ----------
        group : DataGroup
            The group that will contain the data.
        paths : list or None, optional
            A list of filepaths to data. If None, user will be prompt to load
            them from disk. The default is None.

        '''
    # Do nothing if paths are invalid or file dialog is canceled
        if paths is None:
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load masks',
                                                       pref.get_dirPath('in'),
                                                       '''Masks (*.msk)
                                                          Text file (*.txt)''')
        if not paths:
            return
        
        pref.set_dirPath('in', os.path.dirname(paths[0]))
        pbar = CW.PopUpProgBar(self, len(paths), 'Loading data')
        for n, p in enumerate(paths, start=1):
            if pbar.wasCanceled(): 
                break
            try:
                mask = Mask.load(p)
                group.masks.addData(mask)

            except Exception as e:
                pbar.setWindowModality(Qt.NonModal)
                CW.MsgBox(self, 'Crit', f'Unexpected file:\n{p}.', str(e))
                pbar.setWindowModality(Qt.WindowModal)

            finally:
                pbar.setValue(n)

        self.expandRecursively(self.indexFromItem(group))


    def invertInputMap(self):
        '''
        Invert the selected input maps.

        '''
    # Get all the selected Input Maps items
        items = [i for i in self.getSelectedDataObjects() if i.holdsInputMap()]

    # Invert the input map arrays held in each item
        progBar = CW.PopUpProgBar(self, len(items), 'Inverting data')
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

        '''
    # Get all the selected Masks items
        items = [i for i in self.getSelectedDataObjects() if i.holdsMask()]

    # Invert the mask arrays held in each item
        progBar = CW.PopUpProgBar(self, len(items), 'Inverting data')
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

        '''
        checkstate = Qt.Checked if checked else Qt.Unchecked
        for child in group.masks.getChildren():
            child.setCheckState(0, checkstate)
        self.refreshView()


    def exportMineralMap(self, item: CW.DataObject):
        '''
        Export the encoded mineral map (i.e,. with mineral classes expressed
        as numerical IDs) to ASCII format. If users requests it, the encoder
        dictionary is also exported.

        Parameters
        ----------
        item : DataObject
            The data object holding the mineral map data.

        '''

    # Safety: exit function if item does not held mineral map data
        if not item.holdsMineralMap(): 
            return

    # Do nothing if user choice is NO
        text =  'Export map as a numeric array?'
        det_text = 'The translation dictionary is a text file that holds a '\
                   'reference to the mineral classes linked with the IDs of '\
                   'the exported mineral map.'
        msg_cbox = QW.QCheckBox('Include translation dictionary')
        msg_cbox.setChecked(True)
        choice = CW.MsgBox(self, 'Quest', text, det_text, cbox=msg_cbox)
        if choice.no():
            return
        
    # Do nothing if the outpath is invalid or the file dialog is canceled
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Export Map',
                                                    pref.get_dirPath('out'),
                                                    '''ASCII file (*.txt)''')
        if not outpath:
            return
        
        pref.set_dirPath('out', os.path.dirname(outpath))

    # Save the mineral map to disk
        mmap = item.get('data')
        np.savetxt(outpath, mmap.minmap_encoded, fmt='%d')

    # Also save the encoder if user requests it
        if choice.cboxChecked():
            encoder_path = cf.extend_filename(outpath, '_transDict')
            rows, cols = mmap.shape
            with open(encoder_path, 'w') as ep:
                for id_, lbl in mmap.encoder.items():
                    ep.write(f'{id_} :\t{lbl}\n')
            # Include number of rows and columns
                ep.write(f'\nNROWS: {rows}\nNCOLS: {cols}')


    def refreshDataSource(self):
        '''
        Re-load the selected data from its original source.

        '''
        items = self.getSelectedDataObjects()
        pbar = CW.PopUpProgBar(self, len(items), 'Reloading data')
        for n, i in enumerate(items, start=1):
            if pbar.wasCanceled(): 
                break
            try:
                item_data, item_name = i.get('data', 'name')
                path = item_data.filepath
                if path is None: raise FileNotFoundError
                i.setData(0, 100, item_data.load(path))
            # Change the edited state of item
                i.setEdited(False)
            except FileNotFoundError:
                pbar.setWindowModality(Qt.NonModal)
                err = f'Filepath to {item_name} was deleted, moved or renamed.'
                CW.MsgBox(self, 'Crit', err)
                pbar.setWindowModality(Qt.WindowModal)
            finally:
                pbar.setValue(n)

        self.refreshView()


    def viewData(self, item):
        '''
        Send signals for displaying the item's data.

        Parameters
        ----------
        item : DataObject or DataSubGroup (or DataGroup)
            The data object to be displayed. If an instance of DataGroup is
            provided, exits the function.

        '''
    # # Exit the function if item is a group
    #     if isinstance(item, DataGroup):
    #         return

    # Update the scene if item is a data group, data subgroup or data object
        objects = (CW.DataObject, CW.DataSubGroup, CW.DataGroup)
        if isinstance(item, objects):
            self.updateSceneRequested.emit(item)

    # Clear the entire scene if the item is not valid (= None)
        else:
            self.clearView()


    def refreshView(self):
        self.viewData(self.currentItem())


    def clearView(self):
        self.clearSceneRequested.emit()


    def clearAll(self):
        choice = CW.MsgBox(self, 'Quest', 'Remove all samples?')
        if choice.yes():
            self.clear()
            self.clearView()



class HistogramViewer(QW.QWidget):
    '''
    A widget to visualize and interact with histograms of input maps data.
    '''

    def __init__(self, maps_canvas, parent=None):
        '''
        HistogramViewer class constructor.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas displaying input maps data.
        parent : QtWidget or None, optional
            The GUI parent of this widget. The default is None.

        Returns
        -------
        None.

        '''
        super(HistogramViewer, self).__init__(parent)

    # Define main attributes
        self.maps_canvas = maps_canvas

    # Initialize GUI
        self._init_ui()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Histogram Canvas
        self.canvas = plots.HistogramCanvas(logscale=True, size=(3, 1.5),
                                            wheel_zoom=False, wheel_pan=False)
        self.canvas.ax.get_yaxis().set_visible(False)
        self.canvas.setMinimumSize(300, 200)
        # self.canvas.setFixedSize(300, 200) # for better performance
        # self.canvas.setSizePolicy(QW.QSizePolicy.Minimum, QW.QSizePolicy.Minimum)

    # HeatMap Scaler widget
        self.scaler = plots.HeatmapScaler(self.canvas.ax, self.onSpanSelect)

    # Navigation Toolbar
        self.navtbar = plots.NavTbar.histCanvasDefault(self.canvas, self)

    # HeatMap scaler toolbar
        self.scaler_tbar = CW.StyledToolbar('Histogram scaler toolbar')

    # Toggle scaler action [-> Heatmap scaler toolbar]
        self.scaler_action = self.scaler_tbar.addAction(
            QIcon(r'Icons/range.png'), 'Enable scaler')
        self.scaler_action.setCheckable(True)
    
    # Extract mask action [-> Heatmap scaler toolbar]
        self.mask_action = self.scaler_tbar.addAction(
            QIcon(r'Icons/add_mask.png'), 'Extract mask')
        self.mask_action.setEnabled(False)

    # Min and Max scaler values input widgets [-> Heatmap scaler toolbar]
        self.scaler_vmin = QW.QSpinBox()
        self.scaler_vmin.setToolTip('Min. span value')
        self.scaler_vmax = QW.QSpinBox()
        self.scaler_vmax.setToolTip('Max. span value')
        for wid in (self.scaler_vmin, self.scaler_vmax):
            wid.setMaximum(2**16)
            wid.setSingleStep(10)
            wid.setMaximumWidth(100)
            wid.setEnabled(False)

    # Range warning icon [-> Heatmap scaler toolbar]
        warn_pixmap = QPixmap(r'Icons/warnIcon.png')
        warn_lbl = QW.QLabel()
        warn_lbl.setPixmap(warn_pixmap.scaled(20, 20, Qt.KeepAspectRatio))
        warn_lbl.setSizePolicy(QW.QSizePolicy.Maximum, QW.QSizePolicy.Maximum)
        warn_lbl.setToolTip('Lower limit cannot be greater than upper limit.')
    
    # Add widgets to Heatmap scaler toolbar
        self.scaler_tbar.addWidget(self.scaler_vmin)
        self.scaler_tbar.addWidget(self.scaler_vmax) 
        self.warn_icon = self.scaler_tbar.addWidget(warn_lbl)
        self.warn_icon.setVisible(False) 

    # Bin slider widget
        self.bin_slider = QW.QSlider(Qt.Horizontal)
        self.bin_slider.setSizePolicy(QW.QSizePolicy.MinimumExpanding,
                                      QW.QSizePolicy.Fixed)
        self.bin_slider.setMinimum(5)
        self.bin_slider.setMaximum(100)
        self.bin_slider.setSingleStep(5)
        self.bin_slider.setSliderPosition(50)

    # Bins selection layout
        bins_form = QW.QFormLayout()
        bins_form.addRow('Bins', self.bin_slider)

    # Adjust Layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navtbar)
        main_layout.addWidget(self.scaler_tbar)
        main_layout.addLayout(bins_form)
        main_layout.addWidget(self.canvas, stretch=2)
        self.setLayout(main_layout)


    def _connect_slots(self):
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
        self.scaler_vmin.valueChanged.connect(lambda: self.setScalerExtents())
        self.scaler_vmax.valueChanged.connect(lambda: self.setScalerExtents())

    # Number of bins selection
        self.bin_slider.valueChanged.connect(self.setHistBins)


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
        menu = self.canvas.get_navigation_context_menu(self.navtbar)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def onScalerToggled(self, toggled):
        self.scaler.set_active(toggled)
        self.scaler.set_visible(toggled)
        self.mask_action.setEnabled(toggled)
        self.scaler_vmin.setEnabled(toggled)
        self.scaler_vmax.setEnabled(toggled)


        if not self.canvas.is_empty():
            if toggled:
                self.setScalerExtents(update_span=False)
            else:
                self.applyScaling(None, None)

            self.canvas.draw()




    def onSpanSelect(self, vmin, vmax):
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

        Returns
        -------
        None.

        '''
        if self.canvas.is_empty():
            return

        vmin, vmax = round(vmin), round(vmax)
        if vmin == vmax:
            vmin, vmax = 0, 0

        self.scaler_vmin.blockSignals(True)
        self.scaler_vmax.blockSignals(True)

        self.scaler_vmin.setValue(vmin)
        self.scaler_vmax.setValue(vmax)

        self.scaler_vmin.blockSignals(False)
        self.scaler_vmax.blockSignals(False)

        self.setScalerExtents(update_span=False)






    def applyScaling(self, vmin, vmax):
        if vmin is not None and vmax is not None:
            array = self.maps_canvas.image.get_array()
            if vmin >= array.max() or vmax <= array.min():
                vmin, vmax = None, None
        self.maps_canvas.update_clim(vmin, vmax)
        self.maps_canvas.draw()





    def setScalerExtents(self, update_span=True):
        '''
        Select a range in the histogram using the vmin and vmax line edits in
        the Navigation Toolbar. This function calls the onSpanSelect function
        without using the histogram spanner (= fromDragging set to False, see
        onSpanSelect function for more details).

        Returns
        -------
        None.

        '''
        if self.canvas.is_empty():
            return
        
        vmin = self.scaler_vmin.value()
        vmax = self.scaler_vmax.value()


        
        if vmax > vmin:
            self.applyScaling(vmin, vmax)
            self.warn_icon.setVisible(False)
        else:
            self.applyScaling(None, None)
            if vmax < vmin:
                self.warn_icon.setVisible(True)

        if update_span:
            if vmax > vmin:
                self.scaler.extents = (vmin, vmax)
                self.scaler.set_visible(True)
            else:
                self.scaler.set_visible(False)
            
            self.canvas.draw()




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
    # Do nothing if no map is displayed in the maps canvas
        if self.maps_canvas.is_empty():
            return
    
    # Get histogram scaler extents
        vmin, vmax = self.scaler.extents
        vmin, vmax = round(vmin), round(vmax)

    # Extract the displayed array and its current mask (legacy mask)
        array, legacy_mask = self.maps_canvas.get_map(return_mask=True)

    # If the legacy mask exists, intersect it with the new mask
        mask_array = np.logical_or(array < vmin, array > vmax)
        if legacy_mask is not None:
            mask_array = iatools.binary_merge([mask_array, legacy_mask], 'I')
        mask = Mask(mask_array)

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                return CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


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



class ModeViewer(CW.StyledTabWidget):
    '''
    A widget to visualize the modal amounts of the mineral classes occurring in
    the mineral map that is currently displayed in the Data Viewer. It includes
    an interactive legend.
    
    '''

    updateSceneRequested = pyqtSignal(CW.DataObject) # current data object

    def __init__(self, map_canvas, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        map_canvas : ImageCanvas
            The canvas where the mineral map is displayed.
        parent : QtWidget or None, optional
            The GUI parent of this widget. The default is None.

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
        GUI constructor.
        
        '''
    # Interactive legend
        self.legend = CW.Legend(context_menu=True)

    # Canvas
        self.canvas = plots.BarCanvas(orientation='h', size=(3.6, 6.4),
                                      wheel_zoom=False, wheel_pan=False)
        self.canvas.setMinimumSize(200, 350)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar.barCanvasDefault(self.canvas, self)
    
    # Wrap canvas and navigation toolbar in a vertical box layout
        plot_vbox = QW.QVBoxLayout()
        plot_vbox.addWidget(self.navTbar)
        plot_vbox.addWidget(self.canvas)

    # Add tabs to the Mode Viewer
        self.addTab(self.legend, QIcon(r'Icons/legend.png'), None)
        self.addTab(plot_vbox, QIcon(r'Icons/plot.png'), None)
        self.setTabToolTip(0, 'Legend')
        self.setTabToolTip(1, 'Bar plot')


    def _connect_slots(self):
        '''
        Signals-slots connector.
        
        '''

    # Show context menu on the mode canvas after a right-click event
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Connect legend signals
        self.legend.colorChangeRequested.connect(self.onColorChanged)
        self.legend.randomPaletteRequested.connect(self.onPaletteRandomized)
        self.legend.itemRenameRequested.connect(self.onClassRenamed)
        self.legend.itemsMergeRequested.connect(self.onClassMerged)
        self.legend.itemHighlightRequested.connect(self.onItemHighlighted)
        self.legend.maskExtractionRequested.connect(self.onMaskExtracted)


    def showContextMenu(self, point):
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

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

        '''
        self._current_data_object = None
        self.canvas.clear_canvas()
        self.legend.clear()


    def onColorChanged(self, legend_item:QW.QTreeWidgetItem, color:tuple):
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


    def onClassRenamed(self, legend_item: QW.QTreeWidgetItem, new_name: str):
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

        '''
    # Rename the phase in the mineral map
        minmap = self._current_data_object.get('data')
        old_name = legend_item.text(1)
        minmap.rename_phase(old_name, new_name)
            
    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onClassMerged(self, classes:list, new_name:str):
        '''
        Merge two or more classes into a new one. This function propagates the 
        changes to the mineral map, the map canvas, the mode bar plot and the 
        legend. It also sets the linked data object as edited. The arguments of 
        this function are specifically compatible with the itemsMergeRequested 
        signal emitted by the legend (see Legend object for more details).

        Parameters
        ----------
        classes : list
            List of class names.
        new_name : str
            New name for the merged class.

        '''
    # Merge phases in the mineral map
        minmap = self._current_data_object.get('data')
        minmap.merge_phases(classes, new_name)

    # Request update scene
        self.updateSceneRequested.emit(self._current_data_object)

    # Set the current data object as edited
        self._current_data_object.setEdited(True)


    def onItemHighlighted(self, toggled:bool, legend_item:QW.QTreeWidgetItem):
        '''
        Highlight on/off the selected mineral class in the map canvas. The 
        arguments of this function are specifically compatible with the 
        itemHighlightRequested signal emitted by the legend (see Legend object 
        for more details).

        Parameters
        ----------
        toggled : bool
            Highlight on/off
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be highlighted.

        '''
        if toggled:
            minmap = self._current_data_object.get('data')
            phase_id = minmap.as_id(legend_item.text(1))
            vmin, vmax = phase_id - 0.5, phase_id + 0.5
        else:
            vmin, vmax = None, None

        self.map_canvas.update_clim(vmin, vmax)
        self.map_canvas.draw()


    def onMaskExtracted(self, classes: list):
        '''
        Extract a mask from a selection of mineral classes and save it to file.

        Parameters
        ----------
        classes : list
            Selected mineral classes.

        '''
    # Extract the mask
        minmap = self._current_data_object.get('data')
        mask = ~np.isin(minmap.minmap, classes)

    # If a legacy mask exists, intersect it with the new mask
        _, legacy_mask = self.map_canvas.get_map(return_mask=True)
        if legacy_mask is not None:
            mask = iatools.binary_merge([mask, legacy_mask], 'I')

    # Create a new Mask object
        mask = Mask(mask)

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                return CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))



class RoiEditor(QW.QWidget):
    '''
    A widget to build, load, edit and save RoiMap objects interactively.

    '''

    rectangleSelectorUpdated = pyqtSignal()

    def __init__(self, maps_canvas, parent=None):
        '''
        Constructor.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas where ROIs should be drawn and displayed.
        parent : QtWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super(RoiEditor, self).__init__(parent)

    # Define main attributes
        self.canvas = maps_canvas
        self.current_selection = None
        self.current_roimap = None
        self.patches = []

    # ROI selector widget 
        self.rect_sel = plots.RectSel(self.canvas.ax, self.onRectSelect)

    # Load ROIs visual properties
        self.roi_color = pref.get_setting('class/trAreasCol', (255, 0, 0),
                                           tuple)
        self.roi_selcolor = pref.get_setting('class/trAreasSel', (0, 0, 255),
                                              tuple)
        self.roi_filled = pref.get_setting('class/trAreasFill', False, bool)

    # Initialize GUI
        self._init_ui()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Toolbar
        self.toolbar = CW.StyledToolbar('ROI toolbar')

    # Load ROI map [-> Toolbar Action]
        self.load_action = self.toolbar.addAction(
            QIcon(r'Icons/import.png'), 'Import ROI map')

    # Save ROI map [-> Toolbar Action]
        self.save_action = self.toolbar.addAction(
            QIcon(r'Icons/save.png'), 'Save ROI map')

    # Save ROI map as... [-> Toolbar Action]
        self.saveas_action = self.toolbar.addAction(
            QIcon(r'Icons/save_as.png'), 'Save ROI map as')
        
    # Auto detect ROI [-> Toolbar Action]
        self.autoroi_action = self.toolbar.addAction(
            QIcon(r'Icons/roi_detection.png'), 'Auto detect ROI')
        
    # Auto detect ROI dialog
        self.autoroi_dial = dialogs.AutoRoiDetector()

    # Toggle ROI selection [-> Toolbar Action]
        self.draw_action = self.toolbar.addAction(
            QIcon(r'Icons/roi_selection.png'), 'Draw ROI')
        self.draw_action.setCheckable(True)

    # Add ROI [-> Toolbar Action]
        self.addroi_action = self.toolbar.addAction(
            QIcon(r'Icons/add_roi.png'), 'Add ROI')
        self.addroi_action.setEnabled(False)

    # Extract mask [-> Toolbar Action]
        self.extr_mask_action = self.toolbar.addAction(
            QIcon(r'Icons/add_mask.png'), 'Extract selection mask')
        self.extr_mask_action.setEnabled(False)

    # ROI visual preferences [-> Toolbar Menu-Action]
        pref_menu = CW.StyledMenu()
    # - Set ROI outline color action
        self.roicolor_action = pref_menu.addAction('Color...')
    # - Set ROI outline selection color action 
        self.roiselcolor_action = pref_menu.addAction('Selection color...')
    # - Toggle filled ROI color action 
        self.roifilled_action = pref_menu.addAction('Filled')
        self.roifilled_action.setCheckable(True)
        self.roifilled_action.setChecked(self.roi_filled)
    # - Add menu-action to toolbar
        self.pref_action = QW.QAction(QIcon(r'Icons/gear.png'), 'ROI settings')
        self.pref_action.setMenu(pref_menu)
        self.toolbar.addAction(self.pref_action)

    # Insert separator into the toolbar
        self.toolbar.insertSeparator(self.autoroi_action)

    # Loaded ROI map path (Path Label)
        self.mappath = CW.PathLabel(full_display=False)

    # Hide ROI map (Checkable Styled Button)
        self.hideroi_btn = CW.StyledButton(QIcon(r'Icons/not_visible.png'))
        self.hideroi_btn.setCheckable(True)
        self.hideroi_btn.setToolTip('Hide ROI map')

    # Remove (unload) ROI map (Styled Button)
        self.unload_btn = CW.StyledButton(QIcon(r'Icons/clear.png')) 
        self.unload_btn.setToolTip('Clear ROI map')

    # Remove ROI button [-> Corner table widget]
        self.delroi_btn = CW.StyledButton(QIcon(r'Icons/remove.png'))
        self.delroi_btn.setFlat(True)

    # Roi table
        self.table = CW.StyledTable(0, 2)
        self.table.setSelectionBehavior(QW.QAbstractItemView.SelectRows)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.table.setHorizontalHeaderLabels(['Class', 'Pixel Count'])
        self.table.horizontalHeader().setSectionResizeMode(1) # Stretch
        self.table.verticalHeader().setSectionResizeMode(3) # ResizeToContent
        self.table.setCornerWidget(self.delroi_btn)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)

    # Bar plot canvas
        self.barCanvas = plots.BarCanvas(size=(3.6, 2.4), wheel_zoom=False, 
                                         wheel_pan=False)
        self.barCanvas.setMinimumSize(300, 300)

    # Bar plot Navigation toolbar
        self.navTbar = plots.NavTbar.barCanvasDefault(self.barCanvas, self)
        
    # Wrap bar plot and its navigation toolbar in a vbox layout
        barplot_vbox = QW.QVBoxLayout()
        barplot_vbox.addWidget(self.navTbar)
        barplot_vbox.addWidget(self.barCanvas)

    # ROI visualizer (Styled Tab Widget -> [ROI table | bar plot])
        roi_visualizer = CW.StyledTabWidget()
        roi_visualizer.addTab(self.table, QIcon(r'Icons/table.png'), None)
        roi_visualizer.addTab(barplot_vbox, QIcon(r'Icons/plot.png'), None)
        roi_visualizer.setTabToolTip(0, 'Table')
        roi_visualizer.setTabToolTip(1, 'Bar plot')

    # Adjust Layout
        main_layout = QW.QGridLayout()
        main_layout.addWidget(self.toolbar, 0, 0, 1, -1)
        main_layout.addWidget(self.mappath, 1, 0)
        main_layout.addWidget(self.hideroi_btn, 1, 1)
        main_layout.addWidget(self.unload_btn, 1, 2)
        main_layout.addWidget(roi_visualizer, 2, 0, 1, -1)
        main_layout.setColumnStretch(0, 1)
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Load ROI map from file
        self.load_action.triggered.connect(self.loadRoiMap)

    # Save ROI map to file
        self.save_action.triggered.connect(self.saveRoiMap)
        self.saveas_action.triggered.connect(lambda: self.saveRoiMap(True))

    # Launch ROI detector dialog
        self.autoroi_action.triggered.connect(self.autoroi_dial.show)

    # Connect ROI detector dialog signals
        self.autoroi_dial.requestRoiMap.connect(
            lambda: self.autoroi_dial.set_current_roimap(self.current_roimap))
        self.autoroi_dial.drawingRequested.connect(self.addAutoRoi)

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

        '''
        self.canvas.draw_idle()
        self.rect_sel.updateCursor()


    def updateBarPlot(self):
        '''
        Update the bar canvas that displays the cumulative pixel count for each
        drawn ROI.

        '''
    # Clear the canvas if the RoiMap is None or empty
        if not self.current_roimap or not self.current_roimap.class_count:
            self.barCanvas.clear_canvas()
    # Otherwise update the canvas
        else:
            names, counts = zip(*self.current_roimap.class_count.items())
            self.barCanvas.update_canvas(counts, names, 'ROIs Counter')


    def showTableContextMenu(self, point):
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

        menu = QW.QMenu()
        menu.setStyleSheet(pref.SS_menu)

    # Extract mask from selected ROIs
        menu.addAction(QIcon(r'Icons/add_mask.png'), 'ROI mask',
                       self.extractMaskFromRois)
        
    # Separator
        menu.addSeparator()

    # Remove selected ROIs
        menu.addAction(QIcon(r'Icons/remove.png'), 'Remove', self.removeRoi)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def showCanvasContextMenu(self, point):
        '''
        Shows a context menu when right-clicking on the bar plot.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

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

        '''
        image = self.canvas.image
        if image is not None:

        # Save in memory the current selection
            map_shape = image.get_array().shape
            self.current_selection = self.rect_sel.fixed_extents(map_shape)

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

        '''
        if event.mouseevent.button == 1: # left mouse button
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

        '''
    # Create a mpl picking event connection if the rectangle is toggled on,
    # otherwise delete it.
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

    # When the rectangle is activated, the last selection is displayed. So we
    # need to send the signal to inform that the selector has been updated
        self.rectangleSelectorUpdated.emit()


    def updatePatchSelection(self):
        '''
        Redraw ROI patches on canvas with a different color based on their
        selection state. This function is called whenever a new ROI selection
        is performed or when a new ROI color or selection color is set. This
        function also redraws the canvas.

        '''
        selected = self.selectedTableIndices
        for idx, (_, patch) in enumerate(self.patches):
            col = self.roi_selcolor if idx in selected else self.roi_color
            patch.set(color=plots.rgb_to_float([col]), lw=2+2*(idx in selected))
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
        self.current_roimap.rename_roi(idx, name)

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

        '''
    # Display the rectangle in canvas
        color = plots.rgb_to_float([self.roi_color])
        patch = plots.RoiPatch(bbox, color, self.roi_filled)
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


    def editPatchAnnotation(self, index, text):
        '''
        Change text of a ROI patch annotation.

        Parameters
        ----------
        index : int
            The patch index in self.patches.
        text : str
            The new annotation text.

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


    def addAutoRoi(self, auto_roimap):
    
    # Create a new ROI map if there was no existent ROI map
        if self.current_roimap is None:
            self.mappath.setPath('*Unsaved ROI map')
            self.current_roimap = auto_roimap
    
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
        bbox = self.rect_sel.fixed_rect_bbox(map_shape)
        if bbox is None: return

    # If no ROI map is loaded, then create a new one.
        if self.current_roimap is None:
            self.current_roimap = RoiMap.from_shape(map_shape)
            self.mappath.setPath('*Unsaved ROI map')

    # Send a warning if a ROI map is loaded and has different shape of the map
    # currently displayed in the canvas.
        elif self.current_roimap.shape != map_shape:
            warn_text = 'Warning: different map shapes detected. Drawing '\
                        'ROIs on top of different sized maps leads to '\
                        'unpredictable behaviours. Proceed anyway?'
            choice = CW.MsgBox(self, 'QuestWarn', warn_text)

        # Exit function if user does not want to procede
            if choice.no():
                return

    # Prevent drawing overlapping ROIs
        if self.current_roimap.bbox_overlaps(bbox):
            return CW.MsgBox(self, 'Crit', 'ROIs cannot overlap.')

    # Show the dialog to type the ROI name
        text = 'Type name (max 8 ASCII characters)'
        name, ok = QW.QInputDialog.getText(self, 'X-Min Learn', text)

    # Proceed only if the new name is an ASCII <= 8 characters string
        if ok and 0 < len(name) < 9 and name.isascii():
            area = self.current_roimap.bbox_area(bbox)
        # Add to roimap
            self.current_roimap.add_roi(name, bbox)
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

        '''
        for text, patch in self.patches:
            text.set_visible(not hidden)
            patch.set_visible(not hidden)
        self._redraw()


    def extractMaskFromSelection(self):
        '''
        Extract and save a mask from current selection.

        '''
    # Exit function if image is empty
        if self.canvas.is_empty(): 
            return

    # Extract the displayed array shape and its current mask (legacy mask)
        array, legacy_mask = self.canvas.get_map(return_mask=True)
        shape = array.shape

    # Exit function if there is no valid selection
        extents = self.rect_sel.fixed_extents(shape, fmt='xy') # x0,x1, y0,y1
        if extents is None: 
            return

    # Initialize a new Mask of 1's and invert it to draw 'holes' using extents
        mask = Mask.from_shape(shape, fillwith=1)
        mask.invert_region(extents)

    # If the legacy mask exists, intersect it with the new mask
        if legacy_mask is not None:
            mask.mask = iatools.binary_merge([mask.mask, legacy_mask], 'I')

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                return CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


    def extractMaskFromRois(self):
        '''
        Extract and save a mask from selected ROIs.

        '''
    # Get the indices of the selected ROIs. Exit function if no ROI is selected
        selected = self.selectedTableIndices
        if not len(selected): return

    # Initialize a new Mask of 1's with the shape of the current ROI map
        shape = self.current_roimap.shape
        mask = Mask.from_shape(shape, fillwith=1)

    # Use the extents of the selected ROIs to draw 'holes' (0's) on the mask
        for idx in selected:
            roi_bbox = self.current_roimap.roilist[idx][1]
            extents = self.current_roimap.bbox_to_extents(roi_bbox)
            mask.invert_region(extents)

    # If there is a loaded image that has the same shape of the current ROI map
    # and it has a legacy mask, intersect it with the new mask
        if not self.canvas.is_empty():
            array, legacy_mask = self.canvas.get_map(return_mask=True)
            if array.shape == shape and legacy_mask is not None:
                mask.mask = iatools.binary_merge([mask.mask, legacy_mask], 'I')

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                return CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))


    def removeRoi(self):
        '''
        Wrapper function to easily remove selected ROIs. This function requires
        a confirm from the user. The function also redraws the canvas.

        '''
    # Get the indices of the selected ROIs. Exit function if no ROI is selected
        selected = self.selectedTableIndices
        if not len(selected): return

    # Ask for confirmation
        choice = CW.MsgBox(self, 'Quest', 'Remove selected ROI?')
        if choice.yes():

            for row in sorted(selected, reverse=True):
            # Remove from roimap
                self.current_roimap.del_roi(row)
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

        '''
    # Show a warning if a ROI map was already loaded
        if self.current_roimap is not None:
            warn_text = 'Loading a new ROI map will discard any unsaved '\
                        'changes made to the current ROI map. Proceed anyway?'
            choice = CW.MsgBox(self, 'QuestWarn', warn_text)

        # Exit function if user does not want to procede
            if choice.no():
                return

    # Do nothing if filepath is invalid or the file dialog is canceled
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Load ROI map',
                                                 pref.get_dirPath('in'),
                                                 'ROI maps (*.rmp)')
        if not path:
            return
        
        pref.set_dirPath('in', os.path.dirname(path))
        progbar = CW.PopUpProgBar(self, 4, 'Loading data', cancel=False)
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
            return CW.MsgBox(self, 'C', f'Unexpected file:\n{path}', str(e))

    # Populate the canvas and the ROIs table with the loaded ROIs
        for name, bbox in self.current_roimap.roilist:
            area = self.current_roimap.bbox_area(bbox)
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

        '''
        if self.current_roimap is not None:
            warn_text = 'Remove current ROI map? Unsaved changes will be lost.'
            choice = CW.MsgBox(self, 'QuestWarn', warn_text)
            if choice.yes():
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

        '''
    # Exit function if the current ROI map does not exist
        if self.current_roimap is None: 
            return

    # Save the ROI map to a new file if it was requested (saveAs = True) and/or
    # if it was never saved before (= it has not a valid filepath). Otherwise,
    # save it to its current filepath (overwrite).
        if not saveAs and (path := self.current_roimap.filepath) is not None:
            outpath = path
        else:
            outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save ROI map',
                                                        pref.get_dirPath('out'),
                                                        'ROI map file (*.rmp)')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                self.current_roimap.save(outpath)
                self.mappath.setPath(outpath)
            except Exception as e:
                CW.MsgBox(self, 'Crit', 'Failed to save ROI map.', str(e))



class ProbabilityMapViewer(QW.QWidget):
    '''
    A widget to visualize the probability map linked with the mineral map that
    is currently displayed in the data viewer.

    '''

    def __init__(self, maps_canvas):
        '''
        Constructor.

        Parameters
        ----------
        maps_canvas : ImageCanvas
            The canvas that displays the mineral map which is linked to the
            probability map displayed in this widget.

        '''
        super(ProbabilityMapViewer, self).__init__()

        self.maps_canvas = maps_canvas

        self._init_ui()
        self._connect_slots()

    def _init_ui(self):
        '''
        GUI constructor

        '''
    # Canvas
        self.canvas = plots.ImageCanvas()
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # View Range toolbar
        self.rangeTbar = CW.StyledToolbar('Probability range toolbar')

    # Toggle range selection [-> View Range Toolbar]
        self.toggle_range_action = self.rangeTbar.addAction(
            QIcon(r'Icons/range.png'), 'Set range')
        self.toggle_range_action.setCheckable(True)

    # Extract mask from range [-> View Range Toolbar]
        self.mask_action = self.rangeTbar.addAction(
            QIcon(r'Icons/add_mask.png'), 'Extract mask')
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
            wid.setSingleStep(0.05)
            wid.setMaximumWidth(100)
            wid.setEnabled(False)

    # Range warning icon [-> Heatmap scaler toolbar]
        warn_pixmap = QPixmap(r'Icons/warnIcon.png')
        warn_lbl = QW.QLabel()
        warn_lbl.setPixmap(warn_pixmap.scaled(20, 20, Qt.KeepAspectRatio))
        warn_lbl.setSizePolicy(QW.QSizePolicy.Maximum, QW.QSizePolicy.Maximum)
        warn_lbl.setToolTip('Lower limit cannot be greater or equal than '\
                            'upper limit.')

    # Add widgets to View Range Toolbar
        self.rangeTbar.addWidget(self.min_input)
        self.rangeTbar.addWidget(self.max_input)
        self.warn_icon = self.rangeTbar.addWidget(warn_lbl)
        self.warn_icon.setVisible(False)

    # Adjust layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navTbar)
        main_layout.addWidget(self.rangeTbar)
        main_layout.addWidget(self.canvas)
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        Signals-slots connector
        
        '''
    # Context menu on canvas
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)

    # Toggle view range action
        self.toggle_range_action.toggled.connect(self.onViewRangeToggled)

    # Set min and max range actions
        self.min_input.valueChanged.connect(self.setViewRange)
        self.max_input.valueChanged.connect(self.setViewRange)

    # Extract mask action
        self.mask_action.triggered.connect(self.extractMaskFromRange)


    def showContextMenu(self, point):
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def onViewRangeToggled(self, toggled):
        '''
        Actions to be performed when the set view range action is toggled.

        Parameters
        ----------
        toggled : bool
            Toggled state of the action.

        '''
    # Enable/disable all functions in the View Range Toolbar
        self.mask_action.setEnabled(toggled)
        self.min_input.setEnabled(toggled)
        self.max_input.setEnabled(toggled)
    
    # Change the view range or reset it to default values
        if toggled:
            self.setViewRange()
        else:
            self.canvas.update_clim()
            self.canvas.draw()
    

    def setViewRange(self):
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
            self.canvas.draw()
            self.warn_icon.setVisible(False)
        else:
            self.warn_icon.setVisible(True)


    def extractMaskFromRange(self):
        '''
        Extract mask from current view range.

        '''
    # Exit function if no probability map is displayed
        if self.canvas.is_empty():
            return
    
    # Get the range values from the view range toolbar widgets
        vmin, vmax = self.min_input.value(), self.max_input.value()
        
    # Get the probability map array
        array = self.canvas.get_map()
        
    # Extract the current mask (legacy mask)
        _, legacy_mask = self.maps_canvas.get_map(return_mask=True)

    # If the legacy mask exists, intersect it with the new mask
        mask_array = np.logical_or(array < vmin, array > vmax)
        if legacy_mask is not None:
            mask_array = iatools.binary_merge([mask_array, legacy_mask], 'I')
        mask = Mask(mask_array)

    # Save mask file
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save mask',
                                                    pref.get_dirPath('out'),
                                                    '''Mask file (*.msk)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            try:
                mask.save(outpath)
            except Exception as e:
                CW.MsgBox(self, 'Crit', 'Failed to save mask.', str(e))
            


class RgbaCompositeMapViewer(QW.QWidget):
    '''
    A widget to visualize an RGB(A) composite map extracted from the
    combination of input maps.

    '''

    def __init__(self):
        '''
        Constructor.

        '''
        super(RgbaCompositeMapViewer, self).__init__()

        self.channels = ('R', 'G', 'B', 'A')

    # Canvas
        self.canvas = plots.ImageCanvas(cbar=False)
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # R-G-B-A Path Labels
        self.rgba_lbls = [CW.PathLabel(full_display=False) for _ in range(4)]
        channels_layout = QW.QGridLayout()

        for col, lbl in enumerate(self.rgba_lbls):
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

        '''
    # Get context menu from NavTbar actions
        menu = self.canvas.get_navigation_context_menu(self.navTbar)
        menu.addSeparator()

    # Clear channel sub-menu
        clear_submenu = menu.addMenu('Clear channel...')
    # - Clear each individual channel
        for c in self.channels:
            clear_submenu.addAction(
                f'{c} channel', lambda c=c: self.clear_channel(c))
    # - Separator
        clear_submenu.addSeparator()
    # - Clear all channels
        clear_submenu.addAction(
            'Clear all', lambda: self.clear_channel(*self.channels))

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QCursor.pos())


    def clear_channel(self, *args):
        if not self.canvas.is_empty():
            rgba_map = self.canvas.image.get_array()
            for arg in args:
                # [R=0, G=0, B=0, A=1]
                idx = self.channels.index(arg)
                rgba_map[:, :, idx] = 1 if idx == 3 else 0
                self.rgba_lbls[idx].clearPath()

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

        '''
        assert channel in self.channels

    # Get the data and the filepath from the Input Map
        data, filepath = inmap.map, inmap.filepath

    # Get the current rgba map or build a new one
        if self.canvas.is_empty():
            rgba_map = np.zeros((*inmap.shape, 4))
            rgba_map[:, :, 3] = 1   # invert A channel (opacity)
        else:
            rgba_map = self.canvas.image.get_array()

    # Exit function if the Input Map shape does not fit the RGBA map shape
        if inmap.shape != rgba_map.shape[:2]:
            err_txt = 'This map does not fit within the current RGBA map.'
            return CW.MsgBox(self, 'Crit', err_txt)

    # Update the channel with the new data and update the plot
        idx = self.channels.index(channel)
        rgba_map[:, :, idx] = np.round(data/data.max(), 2)
        self.canvas.draw_heatmap(rgba_map, 'RGBA composite map')

    # Update the channel path label
        self.rgba_lbls[idx].setPath(filepath)


    def clear_all(self):
        '''
        Clear all channels.

        '''
        # Important: clear channel must be called before clear canvas
        self.clear_channel('R', 'G', 'B', 'A')
        self.canvas.clear_canvas()