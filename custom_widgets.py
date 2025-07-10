# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 12:27:17 2021

@author: albdag
"""

import os

from PyQt5 import QtWidgets as QW
from PyQt5 import QtGui as QG
from PyQt5 import QtCore as QC

import numpy as np

from _base import *
import convenient_functions as cf
import image_analysis_tools as iatools
import preferences as pref
import style


class DataGroup(QW.QTreeWidgetItem):

    def __init__(self, name: str) -> None:
        '''
        A specialized TreeWidgetItem for the Data Manager pane (for details see
        'docks.DataManager' class). This is the the top-level item (i.e., with
        highest hierarchy) of the Data Manager, holding full rock samples. It
        includes three fixed sub-groups (see 'DataSubGroup' class for details):
        'Input Maps', 'Minearal Maps' and 'Masks'.

        Parameters
        ----------
        name : str
            The name of the group.

        '''
        super().__init__()

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
        self.inmaps.setIcon(0, style.getIcon('STACK'))
        self.minmaps = DataSubGroup('Mineral Maps')
        self.minmaps.setIcon(0, style.getIcon('MINERAL'))
        self.masks = DataSubGroup('Masks')
        self.masks.setIcon(0, style.getIcon('MASK'))
        # add self.points = DataSubGroup('Point Analysis')
        self.subgroups = (self.inmaps, self.minmaps, self.masks)
        self.addChildren(self.subgroups)


    @property
    def name(self) -> str:
        '''
        Return the displayed name of the group.

        Returns
        -------
        str
            Name of the group.

        '''
        return self.text(0)

    
    def getAllDataObjects(self) -> list['DataObject']:
        '''
        Return the data objects from all subgroups in a single list.

        Returns
        -------
        objects : list[DataObject]
            All data objects

        '''
        objects = []
        # include points data (maybe?)
        for subgr in self.subgroups:
            objects.extend(subgr.getChildren())
        return objects


    def setShapeWarnings(self) -> None:
        '''
        Set a warning state to every loaded data object whose shape differs
        from the sample overall trending shape.

        '''
    # Collect in a single list the objects data from all subgroups
        objects = self.getAllDataObjects()
        obj_data = [o.get('data') for o in objects]

    # Extract the shape from each object data and obtain the trending shape  
        if any(obj_data): # returns False if all data in the list are None
            shapes = (d.shape for d in obj_data if d is not None)
            trend_shape = cf.most_frequent(shapes)

        # Set a warning state to each object whose data shape differs from trend
            for idx, data in enumerate(obj_data):
                warn = data is not None and data.shape != trend_shape
                objects[idx].setWarning(warn)


    def getCompositeMask(
        self,
        include: str = 'selected',
        mode: str = 'union',
        ignore_single_mask: bool = False
    ) -> Mask | None:
        '''
        Get the mask array resulting from merging all the checked or selected
        masks loaded in this group.

        Parameters
        ----------
        include : str, optional
            Whether to merge the 'selected' or 'checked' maps. The default is
            'selected'.
        mode : str, optional
            How the composite mask should be constructed. If 'intersection' or
            'I', it is the product of all the masks. If 'union' or 'U', it is
            the sum of all the masks. The default is 'union'.
        ignore_single_mask : bool, optional
            Whether the method should not return a composite mask if only one
            mask is selected/checked. The default is False.

        Returns
        -------
        comp_mask : Mask or None
            The composite mask object or None if no mask is included.

        Raises
        ------
        TypeError
            Raised if "include" is not 'selected' or 'checked'.
        TypeError
            Raised if "mode" is not 'intersection' ('I') or 'union' ('U').

        '''
        cld = self.masks.getChildren()

        match include:
            case 'selected':
                masks = [c.get('data') for c in cld if c.isSelected()]
            case 'checked':
                masks = [c.get('data') for c in cld if c.checkState(0)]
            case _:
                raise TypeError(f'Invalid "include" argument: {include}.')

        if len(masks) == 0:
            comp_mask = None
        elif len(masks) == 1:
            comp_mask = None if ignore_single_mask else masks[0]
        else:
            comp_mask = Mask(
                iatools.binary_merge([m.mask for m in masks], mode))

        return comp_mask


    def clear(self) -> None:
        '''
        Remove data loaded in this group.

        '''
        for subgr in self.subgroups:
            subgr.takeChildren()



class DataSubGroup(QW.QTreeWidgetItem):

    datatype_dict = {
        'Input Maps': InputMap,
        'Mineral Maps': MineralMap,
        'Masks': Mask
    }
    
    def __init__(self, name: str) -> None:
        '''
        A specialized TreeWidgetItem for the Data Manager pane (for details see
        'docks.DataManager' class). This widget can hold different data types
        within the same group (see 'DataGroup' class), in the form of Data 
        Objects (see 'DataObject' class). Their main function is to provide a
        better organization and visualization of data in the Data Manager.

        Parameters
        ----------
        name : str
            The name of the subgroup.

        '''
        super().__init__()

    # Set main attributes
        self.name = name
        self.datatype = self.datatype_dict.get(name, None)

    # Set the flags. Data subgroups can be selected but cannot be edited
        self.setFlags(QC.Qt.ItemIsSelectable | QC.Qt.ItemIsUserCheckable |
                      QC.Qt.ItemIsEnabled)

    # Set the font as underlined
        font = self.font(0)
        font.setItalic(True)
        self.setFont(0, font)
        self.setText(0, name)


    def isEmpty(self) -> bool:
        '''
        Check if this DataSubGroup object is populated with data or not.

        Returns
        -------
        empty : bool
            Whether or not the object is empty.

        '''
        empty = self.childCount() == 0
        return empty


    def addData(self, data: InputMap | MineralMap | Mask):
        '''
        Add data to the subgroup in the form of data objects.

        Parameters
        ----------
        data : InputMap, MineralMap or Mask.
            The data to be added to the subgroup.

        '''
        if isinstance(data, self.datatype):
            self.addChild(DataObject(data))


    def delChild(self, child: 'DataObject') -> None:
        '''
        Remove child DataObject from the subgroup.

        Parameters
        ----------
        child : DataObject
            The child to be removed
            
        '''
        self.takeChild(self.indexOfChild(child))


    def moveChildUp(self, child: 'DataObject') -> None:
        '''
        Move child DataObject up by one position.

        Parameters
        ----------
        child : DataObject
            The child to be moved.

        '''
    # Do nothing if child is already in top position 
        idx = self.indexOfChild(child)
        if idx == 0:
            return
        
    # Remove and re-insert child 
        self.takeChild(idx)
        self.insertChild(idx - 1, child)


    def moveChildDown(self, child: 'DataObject') -> None:
        '''
        Move child DataObject down by one position.

        Parameters
        ----------
        child : DataObject
            The child to be moved.

        '''
    # Do nothing if child is already in last position 
        idx = self.indexOfChild(child)
        if idx == self.childCount() - 1:
            return
        
    # Remove and re-insert child 
        self.takeChild(idx)
        self.insertChild(idx + 1, child)


    def getChildren(self) -> list['DataObject']:
        '''
        Get all the DataObject items owned by this subgroup.

        Returns
        -------
        children : List[DataObject]
            List of DataObject children.

        '''
        children = [self.child(idx) for idx in range(self.childCount())]
        return children
    

    def group(self) -> DataGroup | None:
        '''
        Get the parent group that holds this subgroup or None if it hasn't got
        one.

        Returns
        -------
        DataGroup or None
            Parent group.

        '''
        return self.parent()



class DataObject(QW.QTreeWidgetItem):
   
    MainData = InputMap | MineralMap | Mask # add point analysis
    DataComponent = MainData | str | bool | QG.QIcon | QC.Qt.CheckState | None

    def __init__(self, data: MainData | None) -> None:
        '''
        A specialized TreeWidgetItem for the Data Manager pane (for details see
        'docks.DataManager' class). This widget is a data container, enabling
        convenient access to all data components (e.g., status, icons, arrays,
        filepaths), stored and organized using custom columns and item roles.
        The widget type is set to 'UserType' (see QTreeWidgetItem for details),
        unlocking more customization options for the developer.

        Parameters
        ----------
        data : InputMap, MineralMap, Mask or None
            The data stored in the Data Object.

        '''
    # Set the data object type to 'User Type' for more customization options
        super().__init__(type = QW.QTreeWidgetItem.UserType)

    # Set the flags. A data object is selectable and editable by the user.
        self.setFlags(QC.Qt.ItemIsSelectable | QC.Qt.ItemIsUserCheckable |
                      QC.Qt.ItemIsEnabled | QC.Qt.ItemIsEditable)
        
    # Set status stack [102 = not_found, 101 = edited, 0 = Empty status] 
        self._status_stack = [102, 101, 0] 

    # Set data components with custom roles. A custom role is a user-role (int
    # in [0, 256]), that does not overwrite any default Qt role.
        self.setData(0, 100, data) # Object data - CustomRole (100)
        self.setData(0, 110, None if data is None else data.filepath) # Object filepath - CustomRole (110)
        self.setData(0, 0, self.generateDisplayName()) # Display name - DisplayRole (0)
    # Set object file status types
        self.setData(0, 101, False) # File edited status - CustomRole (101)
        self.setData(0, 102, False) # File missing status - CustomRole (102)
        self.setData(0, 1, QG.QIcon()) # Status icon - DecorationRole (1)
        self.setData(0, 3, '') # Status tooltip - ToolTipRole (3)
    # Set data shape warning state
        self.setData(0, 103, False) # Data shape warning - CustomRole (103)
        self.setData(1, 1, QG.QIcon()) # Warning icon - DecorationRole (1)
        self.setData(1, 3, '') # Warning tooltip - ToolTipRole (= 3)
    # Set the "checked" state for togglable data (Masks and Points [in future])
        if isinstance(data, (Mask,)): # add PointAnalysis class
            self.setData(0, 10, QC.Qt.Unchecked) # CheckedRole (10)


    def setInvalidFilepath(self, path: str) -> None:
        '''
        Invalidate object by setting its filepath to invalid filepath 'path'
        and its status to "not found". This method is useful to set a pointer
        to data that has been deleted, removed or renamed, by keeping its
        original filepath.

        Parameters
        ----------
        filepath : str
            The invalid filepath. It must not link to an existent file.

        '''
        if not os.path.exists(path):
            self.setData(0, 110, path) # set invalid filepath
            self.setData(0, 0, self.generateDisplayName()) # auto generate name
            self.setNotFound(True) # set as 'not found'


    def filepathValid(self) -> bool:
        '''
        Whether the Data Object filepath is valid. A filepath is valid if it
        exists or if it is None, which indicates an unsaved datum.

        Returns
        -------
        bool
            Whether the filepath is valid or not.

        '''
        filepath = self.get('filepath')
        return filepath is None or os.path.exists(filepath)
    

    def generateDisplayName(self) -> str:
        '''
        Generate a display name for the object based on its filepath. If the
        object has an invalid filepath, a generic name will be generated.

        Returns
        -------
        name : str
            Display name.

        '''
        data, filepath = self.get('data', 'filepath')
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


    def holdsInputMap(self) -> bool:
        '''
        Check if the object holds input map data.

        Returns
        -------
        bool
            Whether or not the object holds input map data.

        '''
        return isinstance(self.get('data'), InputMap)


    def holdsMineralMap(self) -> bool:
        '''
        Check if the object holds mineral map data.

        Returns
        -------
        bool
            Wether or not the object holds mineral map data.

        '''
        return isinstance(self.get('data'), MineralMap)


    def holdsMap(self) -> bool:
        '''
        Check if the object holds generic map data.

        Returns
        -------
        bool
            Wether or not the object holds map data.

        '''
        return self.holdsInputMap() or self.holdsMineralMap()


    def holdsMask(self) -> bool:
        '''
        Check if the object holds mask data.

        Returns
        -------
        bool
            Wether or not the object holds mask data.

        '''
        return isinstance(self.get('data'), Mask)


    # def holdsPointsData(self):


    def get(self, *args: str) -> list[DataComponent] | DataComponent:
        '''
        Convenient method to get data components.

        Parameters
        ----------
        *args : str
            One or multiple components keys.

        Returns
        -------
        out : list[DataComponent] or DataComponent
            Requested data components. It returns the first element of the list
            if just one data component is requested.

        Raises
        ------
        KeyError
            Raised if an invalid argument is passed.

        '''
    # Define a dictionary holding all the object's comppnents
        components = {
            'data' : self.data(0, 100),
            'filepath' : self.data(0, 110),
            'name' : self.data(0, 0),
            'is_edited' : self.data(0, 101),
            'not_found' : self.data(0, 102),
            'has_warning' : self.data(0, 103),
            'status_icon' : self.data(0, 1),
            'warn_icon' : self.data(1, 1),
            'status_tip' : self.data(0, 3),
            'warn_tip' : self.data(1, 3),
            'checked' : self.data(0, 10)
        }

    # Raises a KeyError if invalid arg is passed
        out = [components[a] for a in args]
        if len(out) == 1: out = out[0]
        return out
    

    def setObjectData(self, data: MainData):
        '''
        Set 'data' as the Data Object main data. This method also updates the 
        filepath.

        Parameters
        ----------
        data : InputMap, MineralMap or Mask
            The object main data.
            
        '''
        if isinstance(data, (InputMap, MineralMap, Mask)):  # add POINTS
            self.setData(0, 100, data)
            self.setData(0, 110, data.filepath)


    def setFilepath(self, path: str) -> None:
        '''
        Set 'path' as new Data Object filepath. This method also changes the
        data.

        Parameters
        ----------
        path : str
            New filepath.

        Raises
        ------
        Exception
            Raised if the filepath is not valid.

        '''      
    # Do nothing if data is None
        data = self.get('data')  
        if data is None:
            return

    # Load new data from filepath. If filepath is invalid for the current data
    # type, this will raise an error.
        data = data.load(path)
        
    # If we are here, data loading was successfull and we can set new object's
    # data and filepath
        self.setData(0, 100, data)
        self.setData(0, 110, path)


    def setName(self, name: str) -> None:
        '''
        Set Data Object displayed name as 'name'.

        Parameters
        ----------
        name : str
            Displayed name.

        '''
        self.setData(0, 0, name)
        

    def setEdited(self, toggle: bool) -> None:
        '''
        Toggle on/off the file edited status of Data Object both internally and 
        visually (icon and tooltip).

        Parameters
        ----------
        toggle : bool
            Toggle on/off file edited status.

        '''
    # Set the 'is_edited' component
        self.setData(0, 101, toggle)
    # Show/hide the save status icon and tooltip
        self._toggleStatus(101, toggle)


    def setNotFound(self, toggle: bool) -> None:
        '''
        Toggle on/off the file not found status of this object both internally 
        and visually (icon and tooltip).

        Parameters
        ----------
        toggle : bool
            Toggle on/off file not found status.

        '''
    # Set the 'not_found' component
        self.setData(0, 102, toggle)
    # Show/hide the not found status icon and tooltip
        self._toggleStatus(102, toggle)
    # Set empty object data if 'not found'
        if toggle and self.get('data') is not None:
            self.setData(0, 100, None)


    def _toggleStatus(self, status: int, toggle: bool) -> None:
        '''
        Visually toggle on or off the provided status. This method alters the
        status stack, so that if 'status' is toggled on, it will be placed as
        the last (hence visible) element of the stack, otherwise it will be
        placed as the first element of the stack. This allows for stackable
        status with a "last toggled on is visible" rule. 

        Parameters
        ----------
        status : int
            Status to be toggled on/off.
        toggle : bool
            Toggle state.

        '''
    # Do nothing if 'status' is invalid.
        if not status in self._status_stack:
            return
        
    # Move status to last position in stack if toggled else at the beginning
        idx = len(self._status_stack) - 1 if toggle else 0
        status = self._status_stack.pop(self._status_stack.index(status))
        self._status_stack.insert(idx, status) 
        
    # Visually show the last status in the stack (show status icon and tooltip)
        last = self._status_stack[-1]

        if last == 0: # Empty status
            icon = QG.QIcon()
            tooltip = ''
        elif last == 101: # File edited status
            icon = style.getIcon('EDIT')
            tooltip = 'Edits not saved'
        elif last == 102: # File not found status
            icon = style.getIcon('FILE_ERROR')
            tooltip = 'File was deleted, moved or renamed'
        else: # invalid status, should not be possible
            return
        
        self.setData(0, 1, icon)
        self.setData(0, 3, tooltip)


    def setWarning(self, warning: bool) -> None:
        '''
        Toggle on/off the shape warning state of Data Object both internally 
        and visually (icon and tooltip).

        Parameters
        ----------
        warning : bool
            Shape warning state.

        '''
    # Set the 'has_warning' attribute
        self.setData(1, 103, warning)
    # Show/hide the warning icon
        icon = style.getIcon('WARNING') if warning else QG.QIcon()
        self.setData(1, 1, icon)
    # Show/hide the warning tooltip
        text = 'Unfitting shapes' if warning else ''
        self.setData(1, 3, text)


    def setChecked(self, checked: bool) -> None:
        '''
        Set the check state of Data Object to 'checked'.

        Parameters
        ----------
        checked : bool
            Check state. Only checked (=True) or unchecked (=False) states are
            accepted.

        '''
        checkstate = QC.Qt.Checked if checked else QC.Qt.Unchecked
        self.setData(0, 10, checkstate)


    def subgroup(self) -> DataSubGroup | None:
        '''
        Return the parent subgroup that holds this Data Object or None if it 
        hasn't got one.

        Returns
        -------
        DataSubGroup or None
            Parent subgroup.

        '''
        return self.parent()



class SampleMapsSelector(QW.QWidget):

    sampleUpdateRequested = QC.pyqtSignal()
    mapsUpdateRequested = QC.pyqtSignal(int) # index of sample
    mapsDataChanged = QC.pyqtSignal()
    mapClicked = QC.pyqtSignal(DataObject)

    def __init__(
        self,
        maps_type: str,
        checkable: bool = True,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        Ready to use widget that allows loading maps from a sample and show
        them in a QTreeWidget, optionally enabling their selection through
        dedicated checkboxes. This widget sends signals to request maps data.
        Such signals must be catched by the widget that holds the information,
        namely the DataManager.

        Parameters
        ----------
        maps_type : str
            Must be 'inmaps' to list input maps or 'minmaps' to list mineral 
            maps.
        checkable : bool, optional
            Whether the individual maps in the list can be selected through
            checkboxes. The default is True.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        ValueError
            Raised if "maps_type" is not 'inmaps' or 'minmaps'.

        '''
        super().__init__(parent)

    # Set main attributes
        if maps_type not in ('inmaps', 'minmaps'):
            raise ValueError(f'Invalid "maps_type" argument: {maps_type}.')
        self.maps_type = maps_type
        self.checkable = checkable

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Sample combobox (Auto Update Combo Box)
        self.sample_combox = AutoUpdateComboBox()

    # Maps list (Tree Widget)
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


    def _connect_slots(self) -> None: 
        '''
        Signals-slots connector.

        '''
    # Send combobox signals as custom signals
        self.sample_combox.clicked.connect(self.sampleUpdateRequested.emit)
        self.sample_combox.activated.connect(
            lambda idx: self.mapsUpdateRequested.emit(idx))
        
    # Send tree widget signals as custom signals
        self.maps_list.itemClicked.connect(lambda i: self.mapClicked.emit(i))


    def updateCombox(self, samples: list[DataGroup]) -> None:
        '''
        Populate the samples combobox with the samples currently loaded in the
        Data Manager. This method is called by the main window when the combo
        box is clicked.

        Parameters
        ----------
        samples : list[DataGroup]
            List of DataGroup objects.

        '''
        samples_names = [s.text(0) for s in samples]
        self.sample_combox.updateItems(samples_names)

    
    def updateList(self, sample: DataGroup) -> None:
        '''
        Updates the list of currently loaded maps owned by the sample currently
        selected in the samples combobox. This method is called by the main 
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
                item.setChecked(True)
            self.maps_list.addTopLevelItem(item)

    # Send a signal to inform that maps data changed
        self.mapsDataChanged.emit()


    def getChecked(self) -> list[DataObject]:
        '''
        Get the currently checked maps.

        Returns
        -------
        checked : list[DataObject]
            List of checked maps.

        '''
        if self.checkable:
            n_maps = self.maps_list.topLevelItemCount()
            items = [self.maps_list.topLevelItem(i) for i in range(n_maps)]
            checked = [i for i in items if i.checkState(0)]
            return checked
    

    def itemCount(self) -> int:
        '''
        Return the amount of maps loaded in the maps list.

        Returns
        -------
        int
            Number of maps.

        '''
        return self.maps_list.topLevelItemCount()
    

    def currentItem(self) -> DataObject:
        '''
        Return the currently selected map.

        Returns
        -------
        DataObject
            Currently selected map object.

        '''
        return self.maps_list.currentItem()
    

    def clear(self) -> None:
        '''
        Clear out the entire widget.

        '''
        self.sample_combox.clear()
        self.maps_list.clear()



class Legend(QW.QTreeWidget):

    colorChangeRequested = QC.pyqtSignal(QW.QTreeWidgetItem, tuple) # item, col
    randomPaletteRequested = QC.pyqtSignal()
    itemRenameRequested = QC.pyqtSignal(QW.QTreeWidgetItem, str) # item, name
    itemsMergeRequested = QC.pyqtSignal(list, str) # list of classes, name
    itemHighlightRequested = QC.pyqtSignal(bool, QW.QTreeWidgetItem) # on/off, item
    maskExtractionRequested = QC.pyqtSignal(list) # list of classes

    def __init__(
        self,
        amounts: bool = True,
        context_menu: bool = False,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        An interactive legend object for displaying and editing the names and  
        palette colors of mineral phases within a mineral map. Optionally, the 
        mineral modal amounts are displayed as well. It optionally includes a
        right-click context menu action for advanced interactions. This object
        sends signals to notify each edit request, which must be catched and
        handled by other widgets to be effective.

        Parameters
        ----------
        amounts : bool, optional
            Include classes amounts (percentage) in the legend. The default is
            True.
        context_menu : bool, optional
            Whether a context menu should popup when right-clicking on a legend
            item. The default is False.
        parent : QWidget or None, optional
            GUI parent widget of this widget. The default is None.

        '''
        super().__init__(parent)

    # Define main attributes
        self._highlighted_item = None
        self.amounts = amounts
        self.has_context_menu = context_menu
        self.precision = pref.get_setting('data/decimal_precision')
        
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


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.itemDoubleClicked.connect(self.onDoubleClick)
        if self.has_context_menu:
            self.customContextMenuRequested.connect(self.showContextMenu)


    def showContextMenu(self, point: QC.QPoint) -> None:
        '''
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''

    # Get the item that is clicked from 'point' and define a menu
        i = self.itemAt(point)
        if i is None: return
        menu = QW.QMenu()
        menu.setStyleSheet(style.SS_MENU)

    # Rename class
        menu.addAction(style.getIcon('RENAME'), 'Rename',
                       lambda: self.requestClassRename(i))

    # Merge classes
        merge = menu.addAction(
            style.getIcon('MERGE'), 'Merge', self.requestClassMerge)
        merge.setEnabled(len(self.selectedItems()) > 1)

    # Separator
        menu.addSeparator()
 
    # Copy current color HEX string
        menu.addAction('Copy color', lambda: self.copyColorHexToClipboard(i))

    # Change color
        menu.addAction(style.getIcon('PALETTE'), 'Set color',
                       lambda: self.requestColorChange(i))

    # Randomize color
        menu.addAction(style.getIcon('RANDOMIZE_COLOR'), 'Random color',
                       lambda: self.requestRandomColorChange(i))

    # Randomize palette
        menu.addAction(style.getIcon('RANDOMIZE_COLOR'), 'Randomize all',
                       self.randomPaletteRequested.emit)
        
    # Separator
        menu.addSeparator()

    # Higlight item
        highlight = QW.QAction(style.getIcon('HIGHLIGHT'), 'Highlight')
        highlight.setCheckable(True)
        highlight.setChecked(i == self._highlighted_item)
        highlight.toggled.connect(lambda t: self.requestItemHighlight(t, i))
        menu.addAction(highlight)

    # Extract mask
        menu.addAction(style.getIcon('ADD_MASK'), 'Extract mask',
                       self.requestMaskFromClass)

    # Show the menu in the same spot where the user triggered the event
        menu.exec(QG.QCursor.pos())


    def onDoubleClick(self, item: QW.QTreeWidgetItem, column: int) -> None:
        '''
        Wrapper method that triggers different actions depending on which 
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


    def copyColorHexToClipboard(self, item: QW.QTreeWidgetItem) -> None:
        '''
        Copy the selected phase color to the clipboard as a HEX string.

        Parameters
        ----------
        item : QTreeWidgetItem
            The selected phase item.

        '''
        clipboard = QW.qApp.clipboard()
        clipboard.setText(item.whatsThis(0))


    def requestColorChange(self, item: QW.QTreeWidgetItem) -> None:
        '''
        Request to change the color of a class by sending a signal. The signal
        must be catched and handled by the widget that contains the legend.
        This method is also triggered when double-clicking on the item icon.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.

        '''
    # Get the old color (as HEX string) and the new color (as RGB tuple)
        old_col = item.whatsThis(0)
        new_col = QW.QColorDialog.getColor(QG.QColor(old_col), self)

    # Emit the signal
        if new_col.isValid():
            rgb = tuple(new_col.getRgb()[:-1])
            self.colorChangeRequested.emit(item, rgb)


    def requestRandomColorChange(self, item: QW.QTreeWidgetItem) -> None:
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


    def changeItemColor(
        self,
        item: QW.QTreeWidgetItem,
        color: tuple[int, int, int]
    ) -> None:
        '''
        Change the color of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item whose color must be changed.
        color : tuple[int, int, int]
            RGB triplet.

        '''
    # Set the new color to the legend item
        item.setIcon(0, ColorIcon(color))

    # Also set the new whatsThis string
        item.setWhatsThis(0, iatools.rgb2hex(color))


    def requestClassRename(self, item: QW.QTreeWidgetItem) -> None:
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
        label = 'Rename class (max. 8 ASCII characters)'
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), label,
                                           text=old_name)
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


    def renameClass(self, item: QW.QTreeWidgetItem, name: str) -> None:
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


    def requestClassMerge(self) -> None:
        '''
        Request to merge two or more mineral classes. The signal must be
        catched and handled by the widget that contains the legend.

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
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), text)
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


    def requestItemHighlight(self, toggled: bool, item: QW.QTreeWidgetItem) -> None:
        '''
        Request to highlight the selected mineral class. Highlight means to
        show ONLY the selected class in map. The signal must be catched and 
        handled by the widget that contains the legend.

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


    def requestMaskFromClass(self) -> None:
        '''
        Request to extract a mask from the selected mineral classes. The signal
        must be catched and handled by the widget that contains the legend.

        '''
        classes = [i.text(1) for i in self.selectedItems()]
        self.maskExtractionRequested.emit(classes)


    def setPrecision(self, value: int) -> None:
        '''
        Set the number of decimals of the class amounts.

        Parameters
        ----------
        value : int
            Number of decimals to be shown in the legend.

        '''
        self.precision = value
        # self.update()


    def addClass(self, name: str, color: tuple[int, int, int], amount: float) -> None:
        '''
        Add a new mineral class to the legend.

        Parameters
        ----------
        name : str
            Class name.
        color : tuple[int, int, int]
            Class color as RGB triplet. 
        amount : float
            Class modal amount. This value is ignored if legend amounts are not
            enabled.

        '''
        item = QW.QTreeWidgetItem(self)
        item.setIcon(0, ColorIcon(color))               # icon [column 0]
        item.setWhatsThis(0, iatools.rgb2hex(color))    # HEX string ['virtual' column 0]
        item.setText(1, name)                           # class name [column 1]
        if self.amounts:                                # amounts (optional) [column 2]
            amount = round(amount, self.precision)
            item.setText(2, f'{amount}%')


    def hasClass(self, name: str) -> bool:
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


    def update(self, mineral_map: MineralMap) -> None:
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



class ColorIcon(QG.QIcon):

    def __init__(
        self,
        color: tuple[int, int, int] | str,
        edgecolor: str = style.IVORY,
        lw: int = 1,
        size: tuple[int, int] = (64, 64)
    ) -> None:
        '''
        Convenient class to generate a colored icon. Very useful in legends.

        Parameters
        ----------
        color : tuple[int, int, int] or str
            RGB triplet or HEX color string.
        edgecolor : str, optional
            Color of the icon border, expressed as a HEX string. The default is 
            #F9F9F4 (IVORY).
        lw : int, optional
            Border line width. The default is 1.
        size : tuple[int, int], optional
            Icon size. The default is (64, 64).

        Raises
        ------
        TypeError
            Raised if color is not a valid type.

        '''
    # Set main attributes
        self.color = color
        self.height, self.width = size

    # Create a pixmap
        pixmap = QG.QPixmap(self.height, self.width)
        if isinstance(self.color, tuple):
            pixmap.fill(QG.QColor(*self.color))
        elif isinstance(self.color, str):
            pixmap.fill(QG.QColor(self.color))
        else:
            raise TypeError(f'Color must be tuple or str, not {type(color)}')
            
    # Add border
        painter = QG.QPainter(pixmap)
        pen = QG.QPen(QG.QColor(edgecolor))
        pen.setWidth(lw)
        painter.setPen(pen)
        # -1 to make the border inside the pixmap
        painter.drawRect(0, 0, self.width - 1, self.height - 1)  
        painter.end()

    # Create the icon using the pixmap
        super().__init__(pixmap)



class StyledButton(QW.QPushButton):

    def __init__(
        self,
        icon: QG.QIcon | None = None,
        text: str | None = None,
        bg : str | None = None,
        text_padding: int = 1,
        parent: QW.QWidget | None = None
     ) -> None:
        '''
        QSS-styled reimplementation of a QPushButton, adding extra convenient
        parameters.

        Parameters
        ----------
        icon : QIcon or None, optional
            Button icon. The default is None.
        text : str or None, optional
            Button text. The default is None.
        bg : str or None, optional
            Background color of the button. If None the default color is used.
            The default is None.
        text_padding : int, optional
            Adds some space between the icon and the text. The default is 1.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)
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
        if bg:
            ss += 'QPushButton {background-color: %s; font: bold;}' %(bg)
        self.setStyleSheet(ss)



class StyledComboBox(QW.QComboBox):

    def __init__(
        self,
        tooltip: str | None = None,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QComboBox, adding auto-generating
        tooltips and wheel event control.

        Parameters
        ----------
        tooltip : str or None, optional
            Fixed combo box tooltip. If None, the tooltip automatically changes
            when the current text changes. The default is None.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set automatic tooltip update if no tooltip is provided
        if tooltip:
            self.setToolTip(tooltip)
        else:
            self.setToolTip(self.currentText())
            self.currentTextChanged.connect(self.setToolTip)

    # Set stylesheet (SS_menu in case of editable combo box)
        self.setStyleSheet(style.SS_COMBOX + style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent) -> None:
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the combobox. The event will be accepted only if SHIFT is
        pressed during the scroll.

        Parameters
        ----------
        event : QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super().wheelEvent(event)
        else:
            event.ignore()



class StyledDoubleSpinBox(QW.QDoubleSpinBox):

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
        decimals: int = 2,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QDoubleSpinBox, adding extra 
        convenient parameters and wheel event control.

        Parameters
        ----------
        min_value : float, optional
            Minimum range value. The default is 0.0.
        max_value : float, optional
            Maximum range value. The default is 1.0.
        step : float, optional
            Step value. The default is 0.1.
        decimals : int, optional
            Number of displayed decimals. The default is 2.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

    # Set number of decimal positions
        self.setDecimals(decimals)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent) -> None:
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the spinbox. The event will be accepted only if SHIFT is
        pressed during the scroll. N.B. By default, if CTRL is also pressed the
        step is multiplied by 10.

        Parameters
        ----------
        event : QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super().wheelEvent(event)
        else:
            event.ignore()



class StyledListWidget(QW.QListWidget):

    def __init__(self, ext_sel: bool = True, parent: QW.QWidget | None = None) -> None:
        '''
        QSS-styled reimplementation of a QListWidget, adding extra convenient
        methods.

        Parameters
        ----------
        ext_sel : bool, optional
            If extended selection mode should be enabled. The default is True.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))

    # Set extended selection mode if requested
        if ext_sel:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)


    def getItems(self) -> list[QW.QListWidgetItem]:
        '''
        Return all items in the list.

        Returns
        -------
        items : list[QListWidgetItem]
            List of items.

        '''
        items = [self.item(row) for row in range(self.count())]
        return items
    

    def selectedRows(self) -> list[int]:
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
    

    def removeSelected(self) -> None:
        '''
        Remove all selected items from list.

        '''
        for idx in sorted(self.selectedRows(), reverse=True):
            self.takeItem(idx)



class StyledRadioButton(QW.QRadioButton):

    def __init__(
        self,
        text: str = '',
        icon: QG.QIcon | None = None,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QRadioButton, adding extra convenient
        parameters.

        Parameters
        ----------
        text : str
            The text of the radio button. The default is ''.
        icon : QIcon or None, optional
            The icon of the radio button. The default is None.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(text, parent)
        if icon is not None:
            self.setIcon(icon)
        self.setStyleSheet(style.SS_RADIOBUTTON)



class StyledScrollBar(QW.QScrollBar):

    def __init__(
        self,
        orientation: QC.Qt.Orientation,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QScrollBar, with dynamic stylesheet
        assignment based on its orientation.

        Parameters
        ----------
        orientation : Qt.Orientation
            The orientation of the scroll bar.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(orientation, parent)

    # Set the stylesheet
        if orientation == QC.Qt.Horizontal:
            self.setStyleSheet(style.SS_SCROLLBARH)
        else:
            self.setStyleSheet(style.SS_SCROLLBARV)



class StyledSpinBox(QW.QSpinBox):

    def __init__(
        self,
        min_value: int = 0,
        max_value: int = 100,
        step: int = 1,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QSpinBox, adding extra convenient
        parameters and wheel event control.

        Parameters
        ----------
        min_value : int, optional
            Minimum range value. The default is 0.
        max_value : int, optional
            Maximum range value. The default is 100.
        step : int, optional
            Step value. The default is 1.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)


    def wheelEvent(self, event: QG.QWheelEvent) -> None:
        '''
        Reimplementation of the wheelEvent to better control mouse wheel scroll
        activation of the spinbox. The event will be accepted only if SHIFT is
        pressed during the scroll. N.B. By default, if CTRL is also pressed the
        step is multiplied by 10.

        Parameters
        ----------
        event : QWheelEvent
            The mouse wheel event.

        '''
        modifiers = event.modifiers()
        if modifiers & QC.Qt.ShiftModifier:
            super(StyledSpinBox, self).wheelEvent(event)
        else:
            event.ignore()



class StyledTable(QW.QTableWidget):

    def __init__(
        self,
        rows: int,
        cols: int,
        ext_sel: bool = True,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QTableWidget, adding extra convenient
        parameters and better corner button behaviour.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        ext_sel : bool, optional
            If extended selection mode should be enabled. The default is True.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(rows, cols, parent)

    # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(QC.Qt.Horizontal))
        self.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical))

    # Set extended selection mode if requested
        if ext_sel:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

    # Customize corner button behaviour
        self.corner_btn = self.findChild(QW.QAbstractButton, '')
        if self.corner_btn:
            self.corner_btn.setToolTip('Select all / deselect all')
            self.corner_btn.disconnect()
            self.corner_btn.clicked.connect(self.onCornerButtonClicked)

    # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)

    
    def onCornerButtonClicked(self) -> None:
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

    def __init__(
        self,
        wheel_scroll: bool = False,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        QSS-styled reimplementation of a QTabWidget, adding a reimplementation
        of the 'addTab' method to always have a QWidget container for the tab.

        Parameters
        ----------
        wheel_scroll : bool, optional
            Whether mouse wheel event should allow to change current tab. The 
            default is False.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set tabs unscrollable via mouse wheel if requested 
        if not wheel_scroll: 
            self.setTabBar(self.UnscrollableTabBar())

    # Set stylesheet
        self.setStyleSheet(style.SS_TABWIDGET)


    def addTab(
        self,
        qobject: QW.QLayout | QW.QWidget,
        icon: QG.QIcon | None = None,
        title: str | None = None
    ) -> None:
        '''
        Reimplementation of the 'addTab' method, useful to achive consistent 
        look of the tabwidget, no matter the type of widget is set as its tab.

        Parameters
        ----------
        qobject : QLayout or QWidget
            A layout-like or widget-like object to be added as tab. 
        icon : QIcon or None, optional
            Tab icon. The default is None.
        title : str or None, optional
            Tab name. The default is None.

        Raises
        ------
        TypeError
            Raised when qobject is not a QWidget or a QLayout.

        '''
        # If qobject is a layout, tab is a QWidget with layout set to qobject
        if isinstance(qobject, QW.QLayout):
            tab = QW.QWidget()
            tab.setLayout(qobject)

        # If qobject is a widget that contains a layout, tab is the qobject.
        # If it does not contain a layout, tab is a QWidget, containing a
        # QVBoxLayout, containing the qobject.
        elif isinstance(qobject, QW.QWidget):
            if qobject.layout():
                tab = qobject
            else:
                tab = QW.QWidget()
                layout = QW.QVBoxLayout(tab)
                layout.addWidget(qobject)

        # Raise TypeError if qobject is neither a QWidget nor a QLayout
        else:
            raise TypeError(f'Invalid qobject type: {type(qobject)}.')


        # Add the tab with or without an icon
        if icon:
            super().addTab(tab, icon, title)
        else:
            super().addTab(tab, title)


    class UnscrollableTabBar(QW.QTabBar):

        def __init__(self, parent: QW.QWidget | None = None) -> None:
            '''
            Dedicated reimplementation of a QTabBar for the StyledTabWidget.
            It disables tab scrolling with mouse wheel.

            Parameters
            ----------
            parent : QW.QWidget or None, optional
                The GUI parent of this widget. The default is None.

            '''
            super().__init__(parent)


        def wheelEvent(self, event: QG.QWheelEvent) -> None:
            '''
            Reimplements the wheelEvent to ignore it.

            Parameters
            ----------
            event : QWheelEvent
                The wheel mouse event.

            '''
            event.ignore()



class AutoUpdateComboBox(StyledComboBox):

    clicked = QC.pyqtSignal()

    def __init__(self, parent: QW.QWidget | None = None, **kwargs) -> None:
        '''
        A reimplementation of a Styled ComboBox, adding a signal which triggers 
        when the combo box button is clicked. This signal can be connected to a
        custom method that allows to update the list of items in the combo box.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.
        **kwargs
            Parent class arguments (see StyledComboBox class).

        '''
        super().__init__(parent=parent, **kwargs)
        
    def showPopup(self) -> None:
        '''
        Reimplementation of the showPopup method. It just emits the new clicked
        signal.

        '''
        self.clicked.emit()
        super().showPopup()


    def updateItems(self, items: list[str]) -> None:
        '''
        Populate the combo box with new items and delete the previous ones.

        Parameters
        ----------
        items : list[str]
            List of new items.

        '''
        self.clear()
        self.addItems(items)



class RadioBtnLayout(QW.QBoxLayout):

    selectionChanged = QC.pyqtSignal(int) # id of button

    def __init__(
        self,
        names: list[str],
        icons: list[QG.QIcon] | None = None,
        default: int = 0,
        orient: str = 'vertical',
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        Convenient class to group multiple radio buttons in a single layout. It 
        makes use of the QButtonGroup class to easily access each button.

        Parameters
        ----------
        names : list[str]
            List of radio button names.
        icons : list[QIcon] or None, optional
            List of button icons. Must be the same length of names. The default
            is None.
        default : int, optional
            Id of the radio button selected by default. The default is 0.
        orient : str, optional
            Orientation of the layout. Can be 'horizontal' ('h') or 'vertical'
            ('v'). The default is 'vertical'.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        NameError
            Orient must be one of ['horizontal', 'h', 'vertical', 'v'].
        ValueError
            If provided, icons list must have the same length of names list.

        '''
    # Set the current selected button id
        self._selected_id = default

    # Set the orientation of the layout
        match orient:
            case 'horizontal' | 'h':
                direction = QW.QBoxLayout.LeftToRight
            case 'vertical' | 'v':
                direction = QW.QBoxLayout.TopToBottom
            case _:
                raise NameError(f'{orient} is not a valid orientation.')
        
    # Check for same length of names and icons (if provided) lists
        if icons is not None and len(names) != len(icons):
            raise ValueError('Icons and names list have different lengths.')
        
    # Set a QButtonGroup working behind the scene
        super().__init__(direction, parent)
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


    def onSelect(self, id_: int) -> None:
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


    def button(self, id_: int) -> StyledRadioButton:
        '''
        Return the radio button with given id.

        Parameters
        ----------
        id : int
            The radio button id.

        Returns
        -------
        btn : StyledRadioButton
            The radio button with id 'id_'.

        '''
        btn = self.btn_group.button(id_)
        return btn


    def buttons(self) -> list[StyledRadioButton]:
        '''
        Return all radio buttons.

        Returns
        -------
        btns: list[StyledRadioButton]
            All radio buttons.

        '''
        btns = self.btn_group.buttons()
        return btns


    def getChecked(self, as_id: bool = False) -> StyledRadioButton | int:
        '''
        Return the checked radio button.

        Parameters
        ----------
        as_id : bool, optional
            Return checked radio button id instead of object. The default is 
            False.

        Returns
        -------
        StyledRadioButton or int
            The checked radio button object or its id.

        '''
        if as_id: 
            return self.btn_group.checkedId()
        else:    
            return self.btn_group.checkedButton()



class GroupArea(QW.QGroupBox):

    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str | None = None,
        checkable: bool = False,
        tight: bool = False,
        frame: bool = True,
        align: QC.Qt.Alignment = QC.Qt.AlignHCenter,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A convenient class to easily wrap a layout or a widget in a QGroupBox.

        Parameters
        ----------
        qobject : QLayout or QWidget
            A layout-like or a widget-like object.
        title : str or None, optional
            The title of the group box. The default is None.
        checkable : bool, optional
            Whether the group box is checkable. The default is False.
        tight : bool, optional
            Whether the layout of the group box should take all the available 
            space. The default is False.
        frame : bool, optional
            Whether the area should show a visible frame. This is ignored if
            the title is provided. The default is True.
        align : Qt.Alignment, optional
            The alignment of the title. Can be Qt.AlignLeft, Qt.AlignRight or
            Qt.AlignHCenter. The default is Qt.AlignHCenter.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        TypeError
            Raised if an invalid "qobject" argument is passed.
        ValueError
            Raised if an invalid "align" argument is passed.

        '''
    # Set the title of the group box. Change the style-sheet depending on title
    # orientation.
        if title:
            super().__init__(title, parent)

            match align:
                case QC.Qt.AlignLeft:
                    align_css = 'top left'
                case QC.Qt.AlignRight:
                    align_css = 'top right'
                case QC.Qt.AlignHCenter:
                    align_css = 'top center'
                case _:
                    valid = ("Qt.AlignLeft", "Qt.AlignRight", "Qt.AlignHCenter")
                    raise ValueError(f'Argument "align" can only be {valid}.')

            title_ss = (f'QGroupBox::title {{subcontrol-position: {align_css};}}')
            self.setStyleSheet(style.SS_GROUPAREA_TITLE + title_ss)
        
        else:
            super().__init__(parent)
            ss = style.SS_GROUPAREA_NOTITLE
            if not frame:
                ss += 'QGroupBox {border-width: 0px;}' 
            self.setStyleSheet(ss)

    # Set if the group box is checkable or not
        self.setCheckable(checkable)

    # If the qobject is a widget, wrap it into a layout box
        if isinstance(qobject, QW.QLayout):
            layout_box = qobject

        elif isinstance(qobject, QW.QWidget):  
            layout_box = QW.QBoxLayout(QW.QBoxLayout.TopToBottom)
            layout_box.addWidget(qobject)
        
        else: # raise error for invalid qobject
            raise TypeError(f'Invalid "qobject" type: {type(qobject)}.')

    # Set a tight layout if required
        if tight:
            layout_box.setContentsMargins(0, 0, 0, 0)
    
    # Wrap the qobject in the group box
        self.setLayout(layout_box)



class CollapsibleArea(QW.QWidget):

    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str,
        collapsed: bool = True,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A convenient widget to expand/collapse an entire section of the GUI. It
        includes an arrow button to switch from expanded to collapsed view.

        Parameters
        ----------
        qobject : QLayout or QWidget
            A layout-like or a widget-like object.
        title : str
            The title of the section.
        collapsed : bool, optional
            Whether the section should be collapsed by default. The default is 
            True.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set private attributes
        self._collapsed = collapsed
        self._section = qobject
        self._title = title

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    # Set collapsed view if required
        if collapsed:
            self.area.setMaximumHeight(0)
            self.arrow.setArrowType(QC.Qt.RightArrow)


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Arrow button (Tool Button)
        self.arrow = QW.QToolButton()
        self.arrow.setStyleSheet(style.SS_TOOLBUTTON)
        self.arrow.setArrowType(QC.Qt.DownArrow)
        
    # Section title (Label)
        self.title = QW.QLabel(self._title)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)

    # Section area (Group Area)
        self.area = GroupArea(self._section)

    # Expand/Collapse animation (Property Animation)
        self.animation = QC.QPropertyAnimation(self.area, b'maximumHeight')
        self.animation.setDuration(90)

    # Set main layout
        layout = QW.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.arrow, 0, 0)
        layout.addWidget(self.title, 0, 1)
        layout.addWidget(LineSeparator(), 0, 2)
        layout.addWidget(self.area, 1, 0, 1, -1)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Expand/collapse section when arrow button is clicked
        self.arrow.clicked.connect(self.onArrowClicked) 


    def collapsed(self) -> bool:
        '''
        Check if the section is currently collapsed or not.

        Returns
        -------
        bool
            Whether the section is collapsed.

        '''
        return self._collapsed
    

    def onArrowClicked(self) -> None:
        '''
        Slot connected to the button clicked signal from the arrow button. It 
        determines which action should be performed based on the current state
        of the section.

        '''
        self.expand() if self.collapsed() else self.collapse()


    def collapse(self) -> None:
        '''
        Collapse the section and change the arrow type.

        '''
        self._collapsed = True
        self.arrow.setArrowType(QC.Qt.RightArrow)
        self.animation.setStartValue(self.area.height())
        self.animation.setEndValue(0)
        self.animation.start()

    
    def expand(self) -> None:
        '''
        Expand the section and change the arrow type.

        '''
        self._collapsed = False
        self.arrow.setArrowType(QC.Qt.DownArrow)
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.area.sizeHint().height())
        self.animation.start()



class GroupScrollArea(QW.QScrollArea):

    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str | None = None,
        hscroll: bool = True,
        vscroll: bool = True,
        tight: bool = False,
        frame: bool = True,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A convenient class to easily wrap a layout or a widget in a QScrollArea.

        Parameters
        ----------
        qobject : QLayout or QWidget
            A layout-like or a widget-like object.
        title : str or None, optional
            If set, the qobject will be also wrapped in a GroupArea that shows
            this title. Ignored if qobject is a widget. The default is None.
        hscroll : bool, optional
            Whether the horizontal scrollbar is shown. The default is True.
        vscroll : bool, optional
            Whether the vertical scrollbar is shown. The default is True.
        tight : bool, optional
            Whether the layout of the scroll area should take all the available 
            space. Ignored if qobject is a widget. The default is False.
        frame : bool, optional
            Whether the scroll area should have a visible frame. The default is
            True.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        TypeError
            Raised if an invalid "qobject" argument is passed.

        '''
        super().__init__(parent)

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
                if tight:
                    qobject.setContentsMargins(0, 0, 0, 0)
                wid = QW.QWidget()
                wid.setLayout(qobject)
            else:
                wid = GroupArea(qobject, title, tight=tight)
        
        elif isinstance(qobject, QW.QWidget):
            wid = qobject

        else: # raise error if qobject is invalid
            raise TypeError(f'Invalid "qobject" of type {type(qobject)}.')

    # Wrap the qobject in the scroll bar.
        self.setWidget(wid)
        self.setWidgetResizable(True)



class SplitterLayout(QW.QBoxLayout):

    def __init__(
        self,
        orient: QC.Qt.Orientation = QC.Qt.Horizontal,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A ready to use layout box that comes with pre-coded splitters dividing
        each inserted wdget.

        Parameters
        ----------
        orient : Qt.Orientation, optional
            The orientation of the latout. The default is Qt.Horizontal.
        parent : QWidget or None, optional
            The GUI parent of this layout. The default is None.

        '''
    # Set the orientation of the layout box
        if orient == QC.Qt.Horizontal:
            direction = QW.QBoxLayout.LeftToRight
        else:
            direction = QW.QBoxLayout.TopToBottom

    # Initialize the layout box and add a splitter to it. Here we are using the
    # super method 'addWidget' to add the splitter, because such method has
    # been reimplemented (see below methods), and calling it would determine an
    # infinite loop where the app tries to insert the splitter inside itself.
        super().__init__(direction, parent)
        self.splitter = QW.QSplitter(orient)
        self.splitter.setOpaqueResize(pref.get_setting('GUI/smooth_animation'))
        self.splitter.setStyleSheet(style.SS_SPLITTER)
        super().addWidget(self.splitter)


    def insertLayout(self, layout: QW.QLayout, index: int, stretch: int = 0) -> None:
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


    def addLayout(self, layout: QW.QLayout, stretch: int = 0) -> None:
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


    def addLayouts(
        self,
        layouts: list[QW.QLayout],
        stretches: list[int] | None = None
    ) -> None:
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


    def insertWidget(self, widget: QW.QWidget, index: int, stretch: int = 0) -> None:
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


    def addWidget(self, widget: QW.QWidget, stretch: int = 0) -> None:
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


    def addWidgets(
        self,
        widgets: list[QW.QWidget], 
        stretches: list[int] | None = None
    ) -> None:
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

    def __init__(
        self,
        qobjects: list[QW.QLayout | QW.QWidget] | None = None, 
        stretches: list[int] | None = None,
        orient: QC.Qt.Orientation = QC.Qt.Horizontal
    ) -> None:
        '''
        A convenient class to quickly group multiple widgets in a QSplitter.

        Parameters
        ----------
        qobjects : list[QLayout or QWidget] or None, optional
            List of layout-like or widget-like objects. If None, the list will 
            be passed as empty. The default is None.
        stretches : list[int] or None, optional
            List of stretches for each object. Must have the same size of  
            qobjects. If None, stretches are not set. The default is None.
        orient : Qt.Orientation, optional
            Orientation of the splitter. The default is Qt.Horizontal.

        Raises
        ------
        TypeError
            Raised if an invalid object is passed in the "qobjects" argument.
        AssertionError
            Raised if "qobjects" and "stretches" have different sizes.

        '''
    # Use super class to create the oriented splitter and set its stylesheet
        self.orient = orient
        super().__init__(self.orient)
        self.setOpaqueResize(pref.get_setting('GUI/smooth_animation'))
        self.setStyleSheet(style.SS_SPLITTER)

    # Add each object to the splitter. If the object is a layout, wrap it first
    # in a QWidget
        if qobjects is None:
            qobjects = []

        for obj in qobjects:
            if isinstance(obj, QW.QLayout):
                w_obj = QW.QWidget()
                w_obj.setLayout(obj)
                self.addWidget(w_obj)

            elif isinstance(obj, QW.QWidget):
                self.addWidget(obj)
            
            else: # raise error if object is invalid
                raise TypeError(f'Invalid type in "qobjects": {type(obj)}')

    # Add stretches to each object
        if stretches is not None:
            assert len(stretches) == len(qobjects)
            for i, s in enumerate(stretches):
                self.setStretchFactor(i, s)


    def addStretchedWidget(
        self,
        widget: QW.QWidget,
        stretch: int,
        idx: int = -1
    ) -> None:
        '''
        Convenient method to add a widget to the splitter with given stretch.
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

    def __init__(self, text: str = '', parent: QW.QWidget | None = None) -> None:
        '''
        A label decorated with a squared frame. Its text is user-selectable and
        its size policy is fixed to avoid visual glitches when using monitor
        with different resolutions.

        Parameters
        ----------
        text : str, optional
            Label text. The default is ''.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(text, parent)
    
    # Set widget attributes
        self.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)
        self.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)

    # Set stylesheet
        self.setStyleSheet(style.SS_MENU + 'QLabel {border: 1px solid black;}')


    def setFrameColor(self, color: str | tuple[int, int, int]) -> None:
        '''
        Change the color of the label frame.

        Parameters
        ----------
        color : str or tuple[int, int, int]
            Color as HEX string or RGB triplet.

        '''
    # Convert RGB triplet to HEX string
        if isinstance(color, tuple):
            color = iatools.rgb2hex(color)
        
        ss = f'QLabel {{border-color: {color};}}'
        self.setStyleSheet(self.styleSheet() + ss)


    def setFrameWidth(self, width: int) -> None:
        '''
        Change the width of the label frame.

        Parameters
        ----------
        width : int
            Frame width.

        '''
        ss = f'QLabel {{border-width: {width}px;}}'
        self.setStyleSheet(self.styleSheet() + ss)



class PathLabel(FramedLabel):

    def __init__(
        self,
        fullpath: str = '',
        full_display: bool = True,
        elide: bool = True,
        placeholder: str = '',
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        A special type of FramedLabel which allows easy management and display
        of file paths.

        Parameters
        ----------
        fullpath : str, optional
            The full filepath. The default is ''.
        full_display : bool, optional
            Whether the label should display the full filepath. If False, just
            the file name is displayed instead. The default is True.
        elide : bool, optional
            Whether the label should automatically wrap long text. The default 
            is True.
        placeholder : str, optional
            Text to be displayed when the label is cleared. The default is ''.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent=parent)

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
    def _displayedText(self) -> str:
        '''
        Internal method that returns the string that should be displayed.

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


    def display(self) -> None:
        '''
        Display the filepath in the QLabel.

        '''
        text = self._displayedText
        self.setTextElided(text) if self.elide else self.setText(text)
        self.setToolTip(self.fullpath)


    def setTextElided(self, text: str) -> None:
        '''
        Wrap long text.

        Parameters
        ----------
        text : str
            Text to be elided.

        '''
        metrics = self.fontMetrics()
        elided = metrics.elidedText(text, QC.Qt.ElideRight, self.width() - 2)
        self.setText(elided)


    def setPath(self, path: str | None, auto_display: bool = True) -> None:
        '''
        Set a new filepath.

        Parameters
        ----------
        path : str or None
            The new filepath. If None, the text "*Path not found" will be
            displayed.
        auto_display : bool, optional
            Whether the new filepath should also be automatically displayed. 
            The default is True.

        '''
        if path is None: path = '*Path not found'
        self.fullpath = path
        if auto_display: self.display()


    def clearPath(self) -> None:
        '''
        Remove the current filepath.

        '''
        self.setText(self.placeholder)
        self.fullpath = ''
        self.setToolTip('')


    def resizeEvent(self, event: QG.QResizeEvent):
        '''
        Reimplementation of the 'resizeEvent' method. If text wrapping is 
        enabled, allows the label to elide the text adapting it to new size.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.

        '''
        event.accept()
        if self.elide:
            self.setTextElided(self._displayedText)



class DescriptiveProgressBar(QW.QWidget):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A progress bar that shows a description of the current process.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Description label (Label)
        self.desc = QW.QLabel()
        self.desc.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)

    # Progress bar (Progress Bar)
        self.pbar = QW.QProgressBar()
    
    # Adjust main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.desc, alignment=QC.Qt.AlignCenter)
        layout.addWidget(self.pbar)
        self.setLayout(layout)


    def setMinimum(self, minimum: int) -> None:
        '''
        Set minimum progress bar value.

        Parameters
        ----------
        minimum : int
            Minimum value.

        '''
        self.pbar.setMinimum(minimum)


    def setMaximum(self, maximum: int) -> None:
        '''
        Set maximum progress bar value.

        Parameters
        ----------
        maximum : int
            Maximum value.

        '''
        self.pbar.setMaximum(maximum)


    def setRange(self, minimum: int, maximum: int) -> None:
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


    def setUndetermined(self) -> None:
        '''
        Set the prograss bar in an undetermined state.

        '''
        self.pbar.setRange(0, 0)


    def undetermined(self) -> bool:
        '''
        Check if the progress bar is in an undetermined state.

        Returns
        -------
        undet : bool
            Progress bar has undetermined state.

        '''
        undet = self.pbar.minimum() == self.pbar.maximum() == 0
        return undet


    def step(self, step_description: str) -> None:
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


    def reset(self) -> None:
        '''
        Reset the progress bar.

        '''
        self.desc.clear()
        self.pbar.reset()



class PopUpProgBar(QW.QProgressDialog):

    def __init__(
        self,
        parent: QW.QWidget | None,
        n_iter: int,
        label: str = '',
        auto_delete: bool = True
    ) -> None:
        '''
        A customized modal progress bar dialog, that automatically shows and 
        hides itself. Ideal for quick, non-threaded operations. 

        Parameters
        ----------
        parent : QWidget or None
            The GUI parent of this widget. If not None, the dialog will popup
            centered with it.
        n_iter : int
            Total number of operations to be performed. It is set as the upper
            range value of the progress bar.
        label : str, optional
            The text label to show in the dialog. The default is ''.
        auto_delete : bool, optional
            Whether the dialog should be deleted after it has been hidden. The
            default is True.

        Example
        -------
        To display another dialog over this progress bar dialog while keeping 
        it responsive, set this dialog as its parent:

        # Construct the progress bar dialog
        parent_widget = PyQt5.QtWidgets.QWidget()
        n_iter = 5
        pbar = PopUpProgBar(parent_widget, n_iter)

        # Run a processing iteration
        result = 0
        for i in range(n_iter):
            try:
                result += 1/i
            except ZeroDivisionError:
                # Set 'pbar' as the message box's parent <<--[ RELEVANT CODE ]
                MsgBox(pbar, 'Info', f'{i} caused an error. Skipped.')
            finally:
                pbar.increase()

        '''
    # Initialize custom progress dialog. Cancel button is forced to be None.
        flags = QC.Qt.Tool | QC.Qt.WindowTitleHint | QC.Qt.WindowStaysOnTopHint
        super().__init__(label, None, 0, n_iter, parent, flags=flags)

    # Set attributes
        self.auto_delete = auto_delete
    
    # Set widget properties
        self.setMinimumDuration(0)
        self.setValue(0)
        self.setWindowModality(QC.Qt.ApplicationModal) 
        self.setWindowTitle('Please wait...')


    def increase(self) -> None:
        '''
        Convenient method to increase the value of the progress bar by one.

        '''
        self.setValue(self.value() + 1)


    def hideEvent(self, event: QG.QHideEvent) -> None:
        '''
        Reimplementation of the default 'hideEvent' method. If 'auto_delete' is
        True, this requests the destruction of the progress dialog immediately 
        after it is hidden.

        Parameters
        ----------
        event : QG.QHideEvent
            The triggered hide event.

        '''
        event.accept()
        if self.auto_delete:
            self.deleteLater()


    def reject(self) -> None:
        '''
        We do not support the PopUpProgBar dialog being rejected via default 
        triggers such as pressing the ESC key or clicking the 'X' button. Thus,
        we reimplement the default 'reject' method so that it returns nothing.

        '''
        return


class PulsePopUpProgBar(PopUpProgBar):

    def __init__(self, parent: QW.QWidget | None, **kwargs) -> None:
        '''
        Special variant of the PopUpProgBar class, that shows a progress dialog
        with undetermined state (a.k.a. pulse progress bar). Ideal when dealing
        with one or more operations whose lenght can be extremely variable.

        Parameters
        ----------
        parent : QWidget or None
            The GUI parent of this widget. If not None, the dialog will popup
            centered with it.
        
        **kwargs
            Parent class arguments (see 'PopUpProgBar' class).
            
        '''
        super().__init__(parent, 1, **kwargs)
        self.setAutoReset(False)


    def startPulse(self) -> None:
        '''
        Start pulsing the progress bar.

        '''
        self.setValue(1)
        self.setRange(0, 0)


    def stopPulse(self) -> None:
        '''
        Stop pulsing and reset the progress bar.

        '''
        self.reset()



class DecimalPointSelector(StyledComboBox):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        Convenient reimplementation of a Styled Combo Box, that allows the
        selection of the decimal point character.

        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent=parent)
        self.addItems(['.', ','])
        self.setCurrentText(QC.QLocale().decimalPoint())



class SeparatorSymbolSelector(StyledComboBox):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        Convenient reimplementation of a Styled Combo Box, that allows the
        selection of the CSV separator character.

        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent=parent)
        local_separator = ',' if QC.QLocale().decimalPoint() == '.' else ';'
        self.addItems([',', ';'])
        self.setCurrentText(local_separator)



class MsgBox(QW.QMessageBox):

    def __init__(
        self,
        parent: QW.QWidget | None,
        kind: str,
        text: str = '',
        dtext: str = '',
        title: str | None = None,
        icon: QW.QMessageBox.Icon | None = None,
        btns: QW.QMessageBox.StandardButtons | None = None,
        def_btn: QW.QMessageBox.StandardButton | None = None,
        cbox: QW.QCheckBox | None = None
    ) -> None:
        '''
        Convenient class to quickly construct a message box dialog with default 
        icon and buttons depending on selected 'kind'. All of its properties
        can be fully customized through parameters. It also includes methods to
        easily get user interactions with 'Yes'/'No' buttons and checkboxes.

        Parameters
        ----------
        parent : QWidget or None
            The GUI parent of this widget. If not None, the dialog will popup
            centered with it.
        kind : str
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
            Message box icon. If None, it is set according to 'kind'. The
            default is None.
        btns : QMessageBox.StandardButtons or None, optional
            Message box buttons. If None, they are set according to 'kind'. 
            The default is None.
        def_btn : QMessageBox.StandardButton or None, optional
            Default selected button. If None, it is set according to 'kind'.
            The default is None.
        cbox : QCheckBox or None, optional
            If set, adds a checkbox to the message box. The default is None.

        Raises
        ------
        ValueError
            Raised if an invalid 'kind' value is set.

        '''
    # Set default icon, buttons and default button according to message type
        question_buttons = QW.QMessageBox.Yes | QW.QMessageBox.No
        match kind:
            case 'Info' | 'I':
                icon = QW.QMessageBox.Information if icon is None else icon
                btns = QW.QMessageBox.Ok if btns is None else btns
            case 'Quest' | 'Q':
                icon = QW.QMessageBox.Question if icon is None else icon
                btns = question_buttons if btns is None else btns
                def_btn = QW.QMessageBox.No if def_btn is None else def_btn
            case 'Warn' | 'W':
                icon = QW.QMessageBox.Warning if icon is None else icon
                btns = QW.QMessageBox.Ok if btns is None else btns
            case 'QuestWarn' | 'QW':
                icon = QW.QMessageBox.Warning if icon is None else icon
                btns = question_buttons if btns is None else btns
                def_btn = QW.QMessageBox.No if def_btn is None else def_btn
            case 'Crit' | 'C':
                icon = QW.QMessageBox.Critical if icon is None else icon
                btns = QW.QMessageBox.Ok if btns is None else btns
            case _:
                raise ValueError('f{kind} is an invalid message box kind.')
        
    # Auto-set title if not specified
        if title is None:
            try:
                title = parent.windowTitle()
                if title == '': 
                    title = 'X-Min Learn'
            except AttributeError:
                title = 'X-Min Learn'

    # Set dialog attributes
        super().__init__(icon, title, text, btns, parent)
        self.setDefaultButton(def_btn)
        self.setDetailedText(dtext)
        self.setCheckBox(cbox)

    # Set righ-click menu stylesheet
        self.setStyleSheet(style.SS_MENU)

    # Show dialog
        self.exec()


    def yes(self) -> bool:
        '''
        Check if user clicked on 'Yes' button.

        Returns
        -------
        bool
            Whether user clicked on 'Yes'.

        '''
        return self.clickedButton().text() == '&Yes'
    

    def no(self) -> bool:
        '''
        Check if user clicked on 'No' button.

        Returns
        -------
        bool
            Whether user clicked on 'No'.
            
        '''
        return self.clickedButton().text() == '&No'
    

    def cboxChecked(self) -> bool:
        '''
        Check the state of the checkbox. If no checkbox was set, this method
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
        
                

class FileDialog(QW.QFileDialog):

    def __init__(self,
        parent: QW.QWidget | None,
        action: str,
        caption: str = '',
        filters: str = '',
        multifile: bool = False,
        set_default_folder: bool = True
    ) -> None:
        '''
        Convenient class for executing open/save file dialogs. It automatically
        sets its home directory to the application's current default input or
        output directory.

        Parameters
        ----------
        parent : QW.QWidget or None
            The GUI parent of this widget. If not None, the dialog will popup
            centered with it.
        action : str
            If 'open' (or 'O'), executes an open file dialog; if 'save' (or 
            'S'), executes a save file dialog.
        caption : str, optional
            The dialog's caption (title). The default is ''.
        filters : str, optional
            Accepted file types. If '', all types are valid. This parameter is
            ignored if 'action' is 'save' (or 'S') and 'multifile' is True. The
            default is ''.
        multifile : bool, optional
            Whether multiple files should be opened or saved. When 'action' is
            'save' (or 'S') an this parameter is True, an open existent folder
            dialog will be executed, to select an output folder for multiple
            files to be saved to. The default is False.
        set_default_folder : bool, optional
            Whether the application's current default input or output folder 
            should be updated. The default is True.

        Raises
        ------
        ValueError
            Raised if 'action' is not one of ('open', 'O', 'save' or 'S').

        '''
    # Set dialog attributes depending on required action
        match action:
            case 'open' | 'O':
                dir_type = 'in'
                accept_mode = QW.QFileDialog.AcceptOpen
                if multifile:
                    file_mode = QW.QFileDialog.ExistingFiles
                else:
                    file_mode = QW.QFileDialog.ExistingFile
 
            case 'save' | 'S':
                dir_type = 'out'
                if multifile:
                    accept_mode = QW.QFileDialog.AcceptOpen
                    file_mode = QW.QFileDialog.Directory
                    filters = None
                else:
                    accept_mode = QW.QFileDialog.AcceptSave
                    file_mode = QW.QFileDialog.AnyFile
            
            case _:
                raise ValueError(f'Invalid "action" argument: {action}')

    # Fix home directory if non existent 
        home_dir = pref.get_dir(dir_type)
        if not os.path.exists(home_dir):
            home_dir = '.\\'

    # Construct dialog
        super().__init__(parent, caption, home_dir, filters)
        if file_mode == QW.QFileDialog.Directory:
            self.setOption(QW.QFileDialog.ShowDirsOnly, True)
            self.setOption(QW.QFileDialog.DontResolveSymlinks, True)

        self.setFileMode(file_mode)
        self.setAcceptMode(accept_mode)
        self.setViewMode(QW.QFileDialog.Detail)

    # Execute dialog. If required, and the dialog is accepted, update the app's 
    # default input or output folder.
        if self.exec() and set_default_folder:
            path = self.selectedFiles()[0]
            folder = path if file_mode == 2 else os.path.dirname(path)
            pref.set_dir(dir_type, folder)


    def get(self) -> list[str] | str | None:
        '''
        Get the filepath(s) returned by the dialog.

        Returns
        -------
        list[str] or str or None
            A list of user-selected filepaths, or a single filepath if dialog
            accepts the selection of just one file. If the dialog was canceled,
            None is returned.

        '''
        if self.result(): # returns 0 if dialog was canceled, 1 if accepted
            paths = self.selectedFiles()
            multipaths = self.fileMode() == QW.QFileDialog.ExistingFiles
            return paths if multipaths else paths[0]
        return None



class LineSeparator(QW.QFrame):

    def __init__(
        self,
        orient: str = 'horizontal',
        lw: int = 2,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        Simple horizontal or vertical line separator.

        Parameters
        ----------
        orient : str, optional
            Line orientation. Must be 'horizontal' ('h') or 'vertical' ('v').
            The default is 'horizontal'.
        lw : int, optional
            Line width. Can be set to 1, 2 or 3. The default is 2.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        ValueError
            Orient must be either 'horizontal' ('h') or 'vertical' ('v').

        '''
        super().__init__(parent)

        match orient:
            case 'horizontal' | 'h':
                self.setFrameShape(QW.QFrame.HLine)
            case 'vertical' | 'v':
                self.setFrameShape(QW.QFrame.VLine)
            case _:
                raise ValueError(f'{orient} is not a valid orientation.')
        
        self.setFrameShadow(QW.QFrame.Plain)
        self.setLineWidth(lw)
      


class CoordinatesFinder(QW.QFrame):

    coordinatesRequested = QC.pyqtSignal(int, int) # x coord, y coord

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        Convenient widget that allows to find specific X, Y (or column, row) 
        coordinates. Expecially useful to find a specific pixel in an image.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)
        self.setStyleSheet('QFrame {border: 1 solid black;}')

    # X and Y coords input (Line Edit with Integer Validator)
        validator = QG.QIntValidator(0, 10**8)

        self.x_input = QW.QLineEdit()
        self.x_input.setStyleSheet(style.SS_MENU)
        self.x_input.setAlignment(QC.Qt.AlignHCenter)
        self.x_input.setPlaceholderText('X (C)')
        self.x_input.setValidator(validator)
        self.x_input.setToolTip('X or Column value')
        self.x_input.setMaximumWidth(50)

        self.y_input = QW.QLineEdit()
        self.y_input.setStyleSheet(style.SS_MENU)
        self.y_input.setAlignment(QC.Qt.AlignHCenter)
        self.y_input.setPlaceholderText('Y (R)')
        self.y_input.setValidator(validator)
        self.y_input.setToolTip('Y or Row value')
        self.y_input.setMaximumWidth(50)

    # Go to pixel button (Styled Button)
        self.go_btn = StyledButton(style.getIcon('BULLSEYE'))
        self.go_btn.setFlat(True)
        self.go_btn.clicked.connect(self.onButtonClicked)

    # Adjust widget layout
        mainLayout = QW.QHBoxLayout()
        mainLayout.setContentsMargins(5, 2, 5, 2) # l, t, r, b
        mainLayout.addWidget(self.go_btn, alignment=QC.Qt.AlignHCenter)
        mainLayout.addWidget(self.x_input, alignment=QC.Qt.AlignHCenter)
        mainLayout.addWidget(self.y_input, alignment=QC.Qt.AlignHCenter)
        self.setLayout(mainLayout)


    def onButtonClicked(self) -> None:
        '''
        Request coordinates if they are valid.

        '''
        x, y = None, None
        if self.x_input.hasAcceptableInput():
            x = int(self.x_input.text())
        if self.y_input.hasAcceptableInput():
            y = int(self.y_input.text())

        if x is not None and y is not None:
            self.coordinatesRequested.emit(x, y)


    def setInputCellsMaxWidth(self, width: int) -> None:
        '''
        Set the maximum width of the coordinates input cells.

        Parameters
        ----------
        width : int
            Maximum width.

        '''
        self.x_input.setMaximumWidth(width)
        self.y_input.setMaximumWidth(width)



class DocumentBrowser(QW.QWidget):

    def __init__(
        self,
        readonly: bool = False,
        toolbar: bool = True,
        placeholder: str = '',
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        Custom widget to load, display and browse a text document. It includes
        convenient search, edit and zoom functionalities.

        Parameters
        ----------
        readonly : bool, optional
            Whether document is read only or editable. The default is False.
        toolbar : bool, optional
            Whether a toolbar with search, edit and zoom functionalities should
            be visible. The default is True.
        placeholder : str, optional
            Default placeholder text, which is displayed when browser is empty.
            The default is ''.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set main attributes
        self.readonly = readonly
        self.toolbar_visible = toolbar
        self._font = QG.QFont()
        self.placeholder_text = placeholder

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Text browser space (Text Edit)
        self.browser = QW.QTextEdit()
        self.browser.setVerticalScrollBar(StyledScrollBar(QC.Qt.Vertical)) 
        self.browser.setStyleSheet(style.SS_MENU)
        self.browser.setReadOnly(self.readonly)
        self.browser.setPlaceholderText(self.placeholder_text)

    # Browser toolbar (Toolbar)
        self.tbar = QW.QToolBar('Browser toolbar')
        self.tbar.setStyleSheet(style.SS_TOOLBAR)
        self.tbar.setVisible(self.toolbar_visible)

    # Zoom in (Action) [-> Browser Toolbar]
        self.zoom_in_action = self.tbar.addAction(
            style.getIcon('ZOOM_IN'), 'Zoom in')

    # Zoom out (Action) [-> Browser Toolbar]
        self.zoom_out_action = self.tbar.addAction(
            style.getIcon('ZOOM_OUT'), 'Zoom out')
        
    # Separator [-> Browser Toolbar]
        self.tbar.addSeparator()

    # Search Up (Action) [-> Browser Toolbar]
        self.search_up_action = self.tbar.addAction(
            style.getIcon('CHEVRON_UP'), 'Search up')

    # Search Down (Action) [-> Browser Toolbar]
        self.search_down_action = self.tbar.addAction(
            style.getIcon('CHEVRON_DOWN'), 'Search down')
        
    # Search box (Line Edit) [-> Browser Toolbar]
        self.search_box = QW.QLineEdit()
        self.search_box.setStyleSheet(style.SS_MENU)
        self.search_box.setPlaceholderText('Search')
        self.search_box.setClearButtonEnabled(True)
        self.tbar.addWidget(self.search_box)

    # Add optional separator if editing is enabled [-> Browser Toolbar]
        self.tbar.addSeparator().setVisible(not self.readonly)

    # Edit document (Check Box) [-> Browser Toolbar]
        self.edit_cbox = QW.QCheckBox('Editing')
        self.edit_cbox.setChecked(not self.readonly)
        self.edit_cbox.setVisible(not self.readonly)
        self.tbar.addWidget(self.edit_cbox)

    # Adjust Main Layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        layout.addWidget(self.tbar)
        layout.addWidget(self.browser)
        self.setLayout(layout)


    def _connect_slots(self) -> None: 
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


    def setDoc(self, doc_path: str) -> None:
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


    def setText(self, text: str) -> None:
        '''
        Set a custom text to the browser.

        Parameters
        ----------
        text : str
            Custom text.

        '''
        self.browser.setText(text)


    def clear(self) -> None:
        '''
        Clear the browser and set the placeholder text.

        '''
        self.browser.clear()
        self.browser.setPlaceholderText(self.placeholder_text)


    def setDefaultPlaceHolderText(self, text: str) -> None:
        '''
        Set the default placeholder text of the browser. This text is displayed
        when no document is loaded.

        Parameters
        ----------
        text : str
            Placeholder text.

        '''
        self.placeholder_text = text


    def _alterZoom(self, value: int) -> None:
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


    def _findTextUp(self) -> None:
        '''
        Search text up.

        '''
        self.browser.setFocus()
        find_flag = QG.QTextDocument.FindBackward
        self.browser.find(self.search_box.text(), find_flag)


    def _findTextDown(self) -> None:
        '''
        Search text down.

        '''
        self.browser.setFocus()
        self.browser.find(self.search_box.text())



class RandomSeedGenerator(QW.QWidget):

    seedChanged = QC.pyqtSignal(int, int) # old seed, new seed

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        Convenient widget to set, manage and display random seeds.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Set main attributes
        self._old_seed = None
        self.seed = None

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    # Set a random seed
        self.randomizeSeed()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Random seed line edit
        self.seed_input = QW.QLineEdit()
        self.seed_input.setStyleSheet(style.SS_MENU)

    # Custom validator that accepts numbers between 1 and 999999999 and 
    # empty strings. This is required to control the behaviour of the Line Edit
    # when user leaves the field empty.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8})?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.seed_input.setValidator(validator)

    # Randomize seed button (Styled Button)
        self.rand_btn = StyledButton(style.getIcon('DICE'))

    # Adjust layout
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QW.QLabel('Random seed'))
        layout.addWidget(self.seed_input, 1)
        layout.addWidget(self.rand_btn, alignment = QC.Qt.AlignRight)
        self.setLayout(layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Make sure that the seed is never left empty
        self.seed_input.editingFinished.connect(self.fillEmptySeed)

    # Change seed either manually or through randomization
        self.seed_input.textChanged.connect(self.onSeedChanged)
        self.rand_btn.clicked.connect(self.randomizeSeed)


    def randomizeSeed(self) -> None:
        '''
        Randomize seed.

        '''
        seed = np.random.default_rng().integers(1, 999999999)
        self.seed_input.setText(str(seed))


    def onSeedChanged(self, new_seed: str) -> None:
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


    def fillEmptySeed(self) -> None:
        '''
        Fill empty seed with a new random seed.

        '''
        if not self.seed_input.text():
            self.randomizeSeed()



class PercentLineEdit(QW.QFrame):

    valueEdited = QC.pyqtSignal(int)

    def __init__(
        self,
        base_value: int,
        min_perc: int = 1,
        max_perc: int = 100,
        parent: QW.QWidget | None = None
    ) -> None:
        '''
        Advanced widget that allows altering an integer using either a percent
        value with a spinbox, or by typing the new number in a validated Line
        Edit. A custom set of icons visually indicate if the new integer is 
        bigger, smaller or equal to the original one.

        Parameters
        ----------
        base_value : int
            Original integer value.
        min_perc : int, optional
            Minimum allowed percentage value. It cannot be smaller than 1. The
            default is 1.
        max_perc : int, optional
            Maximum allowed percentage value. It cannot be bigger than 100. 
            The default is 100.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Main attributes
        self._base = base_value
        self._value = base_value
        self._min_perc = 1 if min_perc < 1 else min_perc
        self._max_perc = 100 if max_perc < 100 else max_perc

    # Widget attributes
        self.setFrameStyle(QW.QFrame.StyledPanel | QW.QFrame.Plain)
        self.setLineWidth(2)

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Line Edit for direct integer input, equipped with a regex validator that
    # accepts numbers between 1 and 10**9 as well as empty strings. This allows
    # a finer control over the behaviour of the line edit.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8}|1000000000)?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.linedit = QW.QLineEdit(str(self._value))
        self.linedit.setValidator(validator)

    # Percentage input (Styled Spin Box)
        self.spinbox = StyledSpinBox(self._min_perc, self._max_perc)
        #self.spinbox.setFixedWidth(100)
        self.spinbox.setSuffix(' %')
        self.spinbox.setValue(100)

    # Visual increase/decrese icon indicator (Label)
        self.iconlbl = QW.QLabel()
        self.setIcon()

    # Reset button (Styled Button)
        self.reset_btn = StyledButton(style.getIcon('REFRESH'))
        self.reset_btn.setFlat(True)

    # Adjust layout
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.iconlbl)
        layout.addWidget(self.linedit)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.spinbox)
        self.setLayout(layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.linedit.editingFinished.connect(self.onLineditEditingFinished)
        self.linedit.textChanged.connect(self.onLineditChanged)
        self.spinbox.valueChanged.connect(self.onSpinboxChanged)
        self.reset_btn.clicked.connect(self.resetValue)


    def setIcon(self) -> None:
        '''
        Change visual increase/decrease icon indicator based on the difference
        between base value and new input value.
        '''
        delta = self._value - self._base

        if delta > 0:
            icon = str(style.ICONS.get('CARET_UP_GREEN'))
        elif delta < 0:
            icon = str(style.ICONS.get('CARET_DOWN_RED'))
        else:
            icon = str(style.ICONS.get('CARET_DOUBLE_YELLOW'))

        pixmap = QG.QPixmap(icon).scaled(20, 20, QC.Qt.KeepAspectRatio)
        self.iconlbl.setPixmap(pixmap)


    def onLineditEditingFinished(self) -> None:
        '''
        Replace empty strings with original base value. Send valueEdited signal
        otherwise.

        '''
        text = self.linedit.text()
        if not text:
            self.linedit.setText(str(self._base))
        else:
            self.valueEdited.emit(int(text))


    def onLineditChanged(self, value: str)  -> None:
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
    # overflow errors as well when "perc" is bigger than upper spinbox limit. 
        perc = self._max_perc if perc > self._max_perc else round(perc)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(round(perc))
        self.spinbox.blockSignals(False)


    def onSpinboxChanged(self, perc: int) -> None:
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


    def adjustSpinboxPrefix(self, perc: int | float) -> None:
        '''
        Set a prefix to spinbox if percentage is not within its value range. 

        Parameters
        ----------
        perc : int or float
            Percentage value.

        '''
        if perc < self._min_perc:
            self.spinbox.setPrefix('<')
        elif perc > self._max_perc:
            self.spinbox.setPrefix('>')
        else:
            self.spinbox.setPrefix('')

    
    def value(self) -> int:
        '''
        Getter method for value attribute.

        Returns
        -------
        int
            Current value.

        '''
        return self._value
    

    def setValue(self, value: int) -> None:
        '''
        Setter method for value attribute.

        Parameters
        ----------
        value : int
            Integer value.

        '''
        self.linedit.setText(str(value))


    def resetValue(self) -> None:
        '''
        Reset original value.

        '''
        self.setValue(self._base)


    def percent(self) -> int:
        '''
        Get current percent value. Prefixes are not honored. Use ratio() for 
        exact ratio.

        Returns
        -------
        int
            Current percent value.

        '''
        return self.spinbox.value()


    def setPercent(self, perc: int) -> None:
        '''
        Set current percent value.

        Parameters
        ----------
        perc : int
            Percent value.

        '''
        self.spinbox.setValue(perc)
    

    def ratio(self, round_decimals: int | None = None) -> float | int:
        '''
        Get current [value / base value] ratio.

        Parameters
        ----------
        round_decimals : int or None, optional
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



class DatasetDesigner(StyledTable):
    
    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A specialized Styled Table Widget used in the DatasetBuilder tool to
        allow user-friendly creation of ground truth datasets.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(0, 4, parent=parent)

    # Set selection behaviour
        self.setSelectionBehavior(QW.QAbstractItemView.SelectRows)

    # Initialize horizontal headers
        self.columns = ['', 'Input Maps', '', 'Mineral Map']
        self.setHorizontalHeaderLabels(self.columns)
        self.resizeHorizontalHeader()


    def resizeHorizontalHeader(self) -> None:
        '''
        Resize column headers to fill all the available space. Fill-row column,
        separator column and mineral map column will instead resize to their 
        content.

        '''
    # Stretch = 1, ResizeToContent = 3
        header = self.horizontalHeader()
        header.setSectionResizeMode(1) # All columns
        header.setSectionResizeMode(0, 3) # Fill-row 
        header.setSectionResizeMode(self.columnCount() - 2, 3) # Separator
        header.setSectionResizeMode(self.columnCount() - 1, 3) # Mineral Map


    def setFillRowButton(self, row: int) -> None:
        '''
        Initialize the first column of given 'row' to include the "fill row" 
        button, which allows user to import an entire row of feature data.

        Parameters
        ----------
        row : int
            Index of row.

        '''
        btn = StyledButton(style.getIcon('CIRCLE_ADD'))
        btn.setToolTip('Load multiple features at once')
        btn.clicked.connect(self.fillRow)
        self.setCellWidget(row, 0, btn)


    def setRowWidgets(self, row: int) -> None:
        '''
        Initialize the cell widgets of given 'row' as StatusFileLoader widgets
        (see StatusFileLoader class for more details).

        Parameters
        ----------
        row : int
            Index of row.

        '''
        n_columns = self.columnCount()

    # Initialize every input map column with InputMap file filter
        for col in range(1, n_columns - 2):
            wid = StatusFileLoader('ASCII maps (*.txt *.gz)')
            self.setCellWidget(row, col, wid)

    # Initialize mineral map column with MineralMap file filter
        wid = StatusFileLoader('Mineral maps (*.mmp);;ASCII maps (*.txt *.gz)')
        self.setCellWidget(row, n_columns - 1, wid)

    # Set resize mode of vertical header to ResizeToContent(3)
        self.verticalHeader().setSectionResizeMode(row, 3)


    def refresh(self, features: list[str]) -> None:
        '''
        Reset and re-initialize the table with new features names.

        Parameters
        ----------
        features : list[str]
            List of features names.

        '''
    # Clear all and set new column names
        self.clear()
        self.setColumnCount(len(features) + 3)
        self.columns = [''] + features + ['', 'Mineral Map']
        self.setHorizontalHeaderLabels(self.columns)
        self.resizeHorizontalHeader()

    # Initialize the table
        for row in range(self.rowCount()):
            self.setFillRowButton(row)
            self.setRowWidgets(row)


    def addRow(self) -> None:
        '''
        Add a row to the table.

        '''
        row = self.rowCount()
        self.insertRow(row)
        self.setFillRowButton(row)
        self.setRowWidgets(row)


    def delRow(self, row: int) -> None:
        '''
        Remove 'row' from the table.

        Parameters
        ----------
        row : int
            Row to be removed. 

        '''
        if 0 <= row < self.rowCount():
            self.removeRow(row)


    def delLastRow(self) -> None:
        '''
        Remove last row from the table

        '''
        self.delRow(self.rowCount() - 1)


    def fillRow(self) -> None: # Should we instead skip name fitting and have free imports?
        '''
        Try to fill an entire row of Input Maps by automatically import and 
        identify multiple files based on their name fitting with column names.

        '''
    # Do nothing if path is invalid or file dialog is canceled
        ftype = 'ASCII maps (*.txt *.gz)'
        paths = FileDialog(self, 'O', 'Load Maps', ftype, multifile=True).get()
        if not paths:
            return
        
    # Try matching maps with loaded file names
        required_maps = self.columns[1:-2]
        pbar = PopUpProgBar(self, len(paths), 'Loading maps')
        bad_files = []
        for n, p in enumerate(paths, start=1):
            try:
                matching_col = cf.guessMap(cf.path2filename(p), required_maps) # !!! find a more elegant solution
            # If a filename matches with a column, then add it
                if matching_col is not None:
                    row = self.indexAt(self.sender().pos()).row()
                    col = self.columns.index(matching_col)
                    self.cellWidget(row, col).addFile(p)

            except FileNotFoundError:
                bad_files.append(p)
            finally:
                pbar.setValue(n)
    
    # Send error if one or more files could not be read
        if len(bad_files):
            text = 'One or more files have been deleted, removed or renamed.'
            MsgBox(self, 'Crit', text, '\n'.join(bad_files))


    def getRowData(self, row: int) -> tuple[list[InputMap], MineralMap | None]:
        '''
        Extract map data from the given row.

        Parameters
        ----------
        row : int
            Index of row.

        Returns
        -------
        input_maps : list[InputMap]
            List of valid input maps.
        mineral_map : MineralMap or None
            Mineral map or None if invalid.

        '''
        n_columns = self.columnCount()
        input_maps, mineral_map = [], None

        pbar = PopUpProgBar(self, n_columns - 1, 'Loading data')

        for col in range(1, n_columns):
            try:
                # Skip separator column
                if col == n_columns - 2: 
                    continue
                wid = self.cellWidget(row, col)
                path = wid.filepath
                # Extract Mineral Map
                if col == n_columns - 1:
                    mineral_map = MineralMap.load(path)
                # Extract Input Map
                else: 
                    input_maps.append(InputMap.load(path))
            except:
                wid.setStatus('Invalid')
            finally:
                pbar.setValue(col)

        return input_maps, mineral_map



class StatusFileLoader(QW.QWidget):

    def __init__(self, filext: str = '', parent: QW.QWidget | None = None) -> None:
        '''
        Interactive widget useful to load filepaths and provide visual feedback
        on their status (valid, invalid or with warnings).

        Parameters
        ----------
        filext : str
            File extension filter. An empty string means all files. The default
            is ''.
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)

    # Define main attributes
        self.filter = filext
        self.filepath = None
        self.status = 'Invalid'

    # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Add file (Styled Button)
        self.add_btn = StyledButton(style.getIcon('CIRCLE_ADD_GREEN'))
        self.add_btn.setFlat(True)

    # Remove file (Styled Button)
        self.del_btn = StyledButton(style.getIcon('CIRCLE_DEL_RED'))
        self.del_btn.setFlat(True)
        self.del_btn.setEnabled(False)

    # Status file path (Path Label)
        self.status_path = PathLabel(full_display=False)
        self.status_path.setFrameWidth(2)
        self.status_path.setFrameColor(style.BAD_RED)

    # Adjust layout
        layout = QW.QGridLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.add_btn, 0, 0, QC.Qt.AlignBottom)
        layout.addWidget(self.del_btn, 0, 1, QC.Qt.AlignBottom)
        layout.addWidget(self.status_path, 1, 0, 1, -1)
        self.setLayout(layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.add_btn.clicked.connect(self.loadFile)
        self.del_btn.clicked.connect(self.removeFile)


    def loadFile(self) -> None:
        '''
        Load file from an open file dialog.

        '''
        path = FileDialog(self, 'open', 'Load File', self.filter).get()
        if path:
            self.addFile(path)


    def addFile(self, path: str) -> None:
        '''
        Add filepath 'path'. This method can be used to add a path directly 
        without showing an open file dialog.

        Parameters
        ----------
        path : str
            Filepath to be added.

        '''
    # Raise error if filepath is invalid
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}') 

        self.filepath = path
        self.status_path.setPath(path)
        self.setStatus('Valid')
        self.add_btn.setEnabled(False)
        self.del_btn.setEnabled(True)


    def removeFile(self) -> None:
        '''
        Remove loaded filepath.

        '''
        self.filepath = None
        self.status_path.clearPath()
        self.setStatus('Invalid')
        self.add_btn.setEnabled(True)
        self.del_btn.setEnabled(False)


    def getStatus(self) -> str:
        '''
        Get the current file status.

        Returns
        -------
        str
            Current status. Can be 'Valid', 'Invalid' or 'Warning'.

        '''
        return self.status


    def setStatus(self, status: str) -> None:
        '''
        Set current status of file to 'status'. This visually changes the color
        of the label frame to green (Valid), red (Invalid) or yellow (Warning).

        Parameters
        ----------
        status : str
            Required status. It must be 'Valid', 'Invalid' or 'Warning'.

        Raises
        ------
        ValueError
            Raised if status is an invalid string.

        '''
        match status:
            case 'Valid':
                self.status_path.setFrameColor(style.OK_GREEN)
            case 'Invalid':
                self.status_path.setFrameColor(style.BAD_RED)
            case 'Warning':
                self.status_path.setFrameColor(style.WARN_YELLOW)
            case _:
                raise ValueError(f'Invalid status: {status}.')
        
        self.status = status