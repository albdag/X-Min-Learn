"""
Defines custom GUI components that extend standard PyQt widgets.

Created on Wed Apr 28 12:27:17 2021
@author: albdag
"""

import os
from collections.abc import Collection, Sequence
from typing import ClassVar, Literal, TypedDict

import numpy as np
from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW

import convenient_functions as cf
import image_analysis_tools as iatools
import settings
import style
from base import *


class DataGroup(QW.QTreeWidgetItem):
    def __init__(self, name: str) -> None:
        """
        A specialized TreeWidgetItem for the Data Manager pane (for
        details see 'docks.DataManager' class). This is the top-level
        item of the Data Manager, holding full rock samples. It includes
        three fixed sub-groups: 'Input Maps', 'Minearal Maps' and
        'Masks' (see 'DataSubGroup' class for details).

        Parameters
        ----------
        name : str
            The name of the group.
        """
        super().__init__()

        # Set data group as selectable and its name as editable
        self.setFlags(
            QC.Qt.ItemFlag(
                QC.Qt.ItemFlag.ItemIsSelectable
                | QC.Qt.ItemFlag.ItemIsUserCheckable
                | QC.Qt.ItemFlag.ItemIsEnabled
                | QC.Qt.ItemFlag.ItemIsEditable
            )
        )

        # Set the font as bold
        font = self.font(0)
        font.setBold(True)
        self.setFont(0, font)
        self.setText(0, name)

        # Add the subgroups
        self.inmaps = DataSubGroup("Input Maps")
        self.inmaps.setIcon(0, style.getIcon("STACK"))
        self.minmaps = DataSubGroup("Mineral Maps")
        self.minmaps.setIcon(0, style.getIcon("MINERAL"))
        self.masks = DataSubGroup("Masks")
        self.masks.setIcon(0, style.getIcon("MASK"))
        # add self.points = DataSubGroup('Point Analysis')
        self.subgroups = (self.inmaps, self.minmaps, self.masks)
        self.addChildren(self.subgroups)

    @property
    def name(self) -> str:
        """
        Return the displayed name of the group.

        Returns
        -------
        str
            Name of the group.
        """
        return self.text(0)

    def getAllDataObjects(self) -> list["DataObject"]:
        """
        Return the data objects from all subgroups in a single list.

        Returns
        -------
        objects : list[DataObject]
            All data objects
        """
        objects = []
        # include points data (maybe?)
        for subgr in self.subgroups:
            objects.extend(subgr.getChildren())
        return objects

    def setShapeWarnings(self) -> None:
        """
        Set a warning state to every loaded data object whose shape
        differs from the sample overall trending shape.

        """
        # Get the trending shape from all data objects in the group
        objects = self.getAllDataObjects()
        obj_data = [o.getComponents()["data"] for o in objects]
        if any(obj_data):  # False if all data in the list are None
            shapes = (d.shape for d in obj_data if d is not None)
            trend_shape = cf.most_frequent(shapes)

            # Set a warning state to object with unfitting data shape
            for idx, data in enumerate(obj_data):
                warn = data is not None and data.shape != trend_shape
                objects[idx].setWarning(warn)

    def getCompositeMask(
        self,
        include: Literal["checked", "selected"] = "selected",
        mode: iatools.BinaryMergeModes = "union",
        ignore_single_mask: bool = False,
    ) -> Mask | None:
        """
        Get the mask array resulting from merging all the checked or
        selected masks loaded in this group.

        Parameters
        ----------
        include : Literal["checked", "selected"], optional
            Whether to merge the checked or the selected maps. The
            default is "selected".
        mode : BinaryMergeModes, optional
            How the composite mask should be constructed. If
            "intersection" or "I", it is the product of all the masks.
            If "union" or "U", it is the sum of all the masks. The
            default is "union".
        ignore_single_mask : bool, optional
            Whether the method should not return a composite mask when
            only one mask is selected/checked. The default is False.

        Returns
        -------
        comp_mask : Mask | None
            The composite mask object or None, if no mask is included.

        Raises
        ------
        TypeError
            Raised if 'include' is not "selected" or "checked".
        TypeError
            Raised if 'mode' is not "intersection" ("I") or "union"
            ("U").
        """
        cld = self.masks.getChildren()

        match include:
            case "selected":
                items = [c for c in cld if c.isSelected()]
            case "checked":
                items = [c for c in cld if c.getComponents()["checkstate"]]
            case _:
                raise TypeError(f'Invalid include argument: "{include}".')

        masks = [m for i in items if (m := i.getMaskData()) is not None]
        if len(masks) == 0:
            comp_mask = None
        elif len(masks) == 1:
            comp_mask = None if ignore_single_mask else masks[0]
        else:  # Raises TypeError if mode is invalid
            comp_mask = Mask(
                iatools.binary_merge([m.mask for m in masks], mode)
            )

        return comp_mask

    def clear(self) -> None:
        """
        Remove data loaded in this group.

        """
        for subgr in self.subgroups:
            subgr.takeChildren()


class DataSubGroup(QW.QTreeWidgetItem):
    def __init__(
        self, name: Literal["Input Maps", "Mineral Maps", "Masks"]
    ) -> None:
        """
        A specialized TreeWidgetItem for the Data Manager pane (for
        details see 'docks.DataManager' class). This widget can hold
        different data types within the same group (see 'DataGroup'
        class), in the form of Data Objects (see 'DataObject' class).
        Their main function is to provide a better organization and
        visualization of data in the Data Manager.

        Parameters
        ----------
        name : Literal["Input Maps", "Mineral Maps", "Masks"]
            The name of the subgroup.

        Raises
        ------
        ValueError
            Raised if 'name' is not a valid name.
        """
        super().__init__()

        # Set main attributes
        self.name = name
        match name:
            case "Input Maps":
                self.datatype = InputMap
            case "Mineral Maps":
                self.datatype = MineralMap
            case "Masks":
                self.datatype = Mask
            case _:
                raise ValueError(f"{name} is not a valid name for a subgroup.")

        # Set subgroup as selectable but NOT editable
        self.setFlags(
            QC.Qt.ItemFlag(
                QC.Qt.ItemFlag.ItemIsSelectable
                | QC.Qt.ItemFlag.ItemIsUserCheckable
                | QC.Qt.ItemFlag.ItemIsEnabled
            )
        )

        # Set the font as underlined
        font = self.font(0)
        font.setItalic(True)
        self.setFont(0, font)
        self.setText(0, name)

    def isEmpty(self) -> bool:
        """
        Check if the subgroup is populated with data or not.

        Returns
        -------
        empty : bool
            Whether or not the object is empty.
        """
        empty = self.childCount() == 0
        return empty

    def addData(self, data: InputMap | MineralMap | Mask) -> None:
        """
        Add data to the subgroup in the form of data objects.

        Parameters
        ----------
        data : InputMap | MineralMap | Mask.
            The data to be added to the subgroup.
        """
        if isinstance(data, self.datatype):
            self.addChild(DataObject(data))

    def delChild(self, child: "DataObject") -> None:
        """
        Remove child DataObject from the subgroup.

        Parameters
        ----------
        child : DataObject
            The child to be removed.
        """
        self.takeChild(self.indexOfChild(child))

    def moveChildUp(self, child: "DataObject") -> None:
        """
        Move child DataObject up by one position.

        Parameters
        ----------
        child : DataObject
            The child to be moved.
        """
        # Exit function if child is already in top position
        idx = self.indexOfChild(child)
        if idx == 0:
            return

        # Remove and re-insert child
        self.takeChild(idx)
        self.insertChild(idx - 1, child)

    def moveChildDown(self, child: "DataObject") -> None:
        """
        Move child DataObject down by one position.

        Parameters
        ----------
        child : DataObject
            The child to be moved.
        """
        # Exit function if child is already in last position
        idx = self.indexOfChild(child)
        if idx == self.childCount() - 1:
            return

        # Remove and re-insert child
        self.takeChild(idx)
        self.insertChild(idx + 1, child)

    def getChildren(self) -> list["DataObject"]:
        """
        Get all the DataObject items owned by this subgroup.

        Returns
        -------
        data_objects : List[DataObject]
            List of DataObject children.
        """
        children = [self.child(idx) for idx in range(self.childCount())]
        data_objects = [c for c in children if isinstance(c, DataObject)]
        return data_objects

    def group(self) -> DataGroup | None:
        """
        Get the parent group that holds this subgroup.

        Returns
        -------
        DataGroup | None
            Parent group or None, if this subgroup hasn't got one.
        """
        if isinstance(parent := self.parent(), DataGroup):
            return parent
        return


class DataObjectComponents(TypedDict):
    data: InputMap | MineralMap | Mask | None
    filepath: str | None
    name: str
    is_edited: bool
    not_found: bool
    has_warning: bool
    status_icon: QG.QIcon
    warn_icon: QG.QIcon
    status_tip: str
    warn_tip: str
    checkstate: QC.Qt.CheckState


class DataObject(QW.QTreeWidgetItem):
    def __init__(self, data: InputMap | MineralMap | Mask | None) -> None:
        """
        A specialized TreeWidgetItem for the Data Manager pane (for
        details see 'docks.DataManager' class). This widget is a data
        container, enabling convenient access to all data components
        (e.g., status, icons, arrays, filepaths), stored and organized
        using custom columns and item roles.

        Parameters
        ----------
        data : InputMap | MineralMap | Mask | None
            The data stored in the Data Object.
        """
        # Set data object type to 'User Type' for more customizations
        super().__init__(type=QW.QTreeWidgetItem.ItemType.UserType)

        # Set data object as selectable and editable by the user
        self.setFlags(
            QC.Qt.ItemFlag(
                QC.Qt.ItemFlag.ItemIsSelectable
                | QC.Qt.ItemFlag.ItemIsUserCheckable
                | QC.Qt.ItemFlag.ItemIsEnabled
                | QC.Qt.ItemFlag.ItemIsEditable
            )
        )

        # Set status stack [102 = not_found, 101 = edited, 0 = None]
        self._status_stack = [102, 101, 0]

        # Set data components with custom user roles (int in [0, 256]),
        # that do not overwrite default Qt roles
        #|Object data - CustomRole (100)
        self.setData(0, 100, data)
        #|Object filepath - CustomRole (110)
        self.setData(0, 110, None if data is None else data.filepath)
        #|Display name - DisplayRole (0)
        self.setData(0, 0, self.generateDisplayName())
        # Set object file status types
        #|File edited status - CustomRole (101)
        self.setData(0, 101, False)
        #|File missing status - CustomRole (102)
        self.setData(0, 102, False)
        #|Status icon - DecorationRole (1)
        self.setData(0, 1, QG.QIcon())
        #|Status tooltip - ToolTipRole (3)
        self.setData(0, 3, "")
        # Set data shape warning state
        #|Data shape warning - CustomRole (103)
        self.setData(0, 103, False)
        #|Warning icon - DecorationRole (1)
        self.setData(1, 1, QG.QIcon())
        #|Warning tooltip - ToolTipRole (3)
        self.setData(1, 3, "")
        # Set check state for togglable data (Masks and Points [future])
        if isinstance(data, (Mask,)):  # add PointAnalysis class
            #|Check state - CheckedRole (10)
            self.setData(0, 10, QC.Qt.CheckState.Unchecked)

    def setInvalidFilepath(self, path: str) -> None:
        """
        Invalidate object by setting its filepath to an invalid filepath
        'path' and its status to "not found". This method is useful to
        set a pointer to data that has been deleted, removed or renamed,
        by keeping its original filepath.

        Parameters
        ----------
        filepath : str
            The invalid filepath. It must NOT link to an existing file.
        """
        if not os.path.exists(path):
            # Set invalid filepath
            self.setData(0, 110, path)
            # Set auto-generated name
            self.setData(0, 0, self.generateDisplayName())
            # Set as 'not found'
            self.setNotFound(True)

    def filepathValid(self) -> bool:
        """
        Whether the Data Object filepath is valid. A filepath is valid
        if it exists or if it is None, which indicates an unsaved datum.

        Returns
        -------
        bool
            Whether the filepath is valid or not.
        """
        filepath = self.getComponents()["filepath"]
        return filepath is None or os.path.exists(filepath)

    def generateDisplayName(self) -> str:
        """
        Generate a display name for the object based on its filepath. If
        the object has an invalid filepath, a generic name will be
        generated.

        Returns
        -------
        name : str
            Display name.
        """
        components = self.getComponents()
        data, filepath = components["data"], components["filepath"]
        if filepath is None:
            if isinstance(data, (InputMap, MineralMap)):
                name = "Unnamed map"
            elif isinstance(data, Mask):
                name = "Unnamed mask"
            # add Point Analysis here
            else:
                name = "Unnamed item"  # safety
        else:
            name = cf.path2filename(filepath)

        return name

    def getInputMapData(self) -> InputMap | None:
        """
        Retrieve the input map data contained in the object.

        Returns
        -------
        InputMap | None
            Input map data or None, if the object does not contain it.
        """
        if isinstance(data := self.data(0, 100), InputMap):
            return data
        return

    def getMineralMapData(self) -> MineralMap | None:
        """
        Retrieve the mineral map data contained in the object.

        Returns
        -------
        MineralMap | None
            Mineral map data or None, if the object does not contain it.
        """
        if isinstance(data := self.data(0, 100), MineralMap):
            return data
        return

    def getMaskData(self) -> Mask | None:
        """
        Retrieve the mask data contained in the object.

        Returns
        -------
        Mask | None
            Mask data or None, if the object does not contain it.
        """
        if isinstance(data := self.data(0, 100), Mask):
            return data
        return

    # def getPointsData(self):

    def getComponents(self) -> DataObjectComponents:
        """
        Get all Data Object components.

        Returns
        -------
        components : DataObjectComponents
            Data Object components as a strongly-typed dictionary.
        """
        components: DataObjectComponents = {
            "data": self.data(0, 100),
            "filepath": self.data(0, 110),
            "name": self.data(0, 0),
            "is_edited": self.data(0, 101),
            "not_found": self.data(0, 102),
            "has_warning": self.data(0, 103),
            "status_icon": self.data(0, 1),
            "warn_icon": self.data(1, 1),
            "status_tip": self.data(0, 3),
            "warn_tip": self.data(1, 3),
            "checkstate": self.data(0, 10),
        }
        return components

    def setObjectData(self, data: InputMap | MineralMap | Mask) -> None:
        """
        Set 'data' as the Data Object main data. This also updates the
        filepath.

        Parameters
        ----------
        data : InputMap | MineralMap | Mask
            The object main data.
        """
        if isinstance(data, (InputMap, MineralMap, Mask)):  # add POINTS
            self.setData(0, 100, data)
            self.setData(0, 110, data.filepath)

    def setFilepath(self, path: str) -> None:
        """
        Set 'path' as new Data Object filepath. This also updates the
        data.

        Parameters
        ----------
        path : str
            New filepath.
        """
        # Exit function if data is None
        if (data := self.getComponents()["data"]) is None:
            return

        # Load new data from filepath. If filepath contains an invalid
        # data type, this will raise an error.
        data = data.load(path)

        # Set new object's data and filepath
        self.setData(0, 100, data)
        self.setData(0, 110, path)

    def setName(self, name: str) -> None:
        """
        Set Data Object displayed name as 'name'.

        Parameters
        ----------
        name : str
            Displayed name.
        """
        self.setData(0, 0, name)

    def setEdited(self, toggle: bool) -> None:
        """
        Toggle on/off the file edited status of Data Object both
        internally and visually (icon and tooltip).

        Parameters
        ----------
        toggle : bool
            Toggle state of the 'edited' status.
        """
        # Set the 'is_edited' component
        self.setData(0, 101, toggle)
        # Show/hide the save status icon and tooltip
        self._toggleStatus(101, toggle)

    def setNotFound(self, toggle: bool) -> None:
        """
        Toggle on/off the file not found status of this object both
        internally and visually (icon and tooltip).

        Parameters
        ----------
        toggle : bool
            Toggle state of the 'file not found' status.
        """
        # Set the 'not_found' component
        self.setData(0, 102, toggle)
        # Show/hide the not found status icon and tooltip
        self._toggleStatus(102, toggle)
        # Set empty object data if 'not found'
        if toggle and self.getComponents()["data"] is not None:
            self.setData(0, 100, None)

    def _toggleStatus(self, status: int, toggle: bool) -> None:
        """
        Visually toggle on or off the provided status. This alters the
        status stack, so that if 'status' is toggled on, it will be
        placed as the last (hence visible) element of the stack,
        otherwise it will be placed as the first element of the stack.
        This allows the status icon and tooltip to always show the last
        toggled-on status.

        Parameters
        ----------
        status : int
            Status to be toggled on/off.
        toggle : bool
            Toggle state.
        """
        # Exit function if 'status' is invalid.
        if not status in self._status_stack:
            return

        # Move status to last or first stack position based on 'toggle'
        idx = len(self._status_stack) - 1 if toggle else 0
        status = self._status_stack.pop(self._status_stack.index(status))
        self._status_stack.insert(idx, status)

        # Visually show the last status in the stack (icon and tooltip)
        match self._status_stack[-1]:
            case 0:  # Empty status
                icon = QG.QIcon()
                tooltip = ""
            case 101:  # File edited status
                icon = style.getIcon("EDIT")
                tooltip = "Edits not saved"
            case 102:  # File not found status
                icon = style.getIcon("FILE_ERROR")
                tooltip = "File was deleted, moved or renamed"
            case _:  # invalid status, should not be possible
                return

        self.setData(0, 1, icon)
        self.setData(0, 3, tooltip)

    def setWarning(self, warning: bool) -> None:
        """
        Toggle on/off the shape warning state of Data Object both
        internally and visually (icon and tooltip).

        Parameters
        ----------
        warning : bool
            Shape warning state.
        """
        # Set the 'has_warning' attribute
        self.setData(1, 103, warning)
        # Show/hide the warning icon
        icon = style.getIcon("WARNING") if warning else QG.QIcon()
        self.setData(1, 1, icon)
        # Show/hide the warning tooltip
        text = "Unfitting shapes" if warning else ""
        self.setData(1, 3, text)

    def setChecked(self, checked: bool) -> None:
        """
        Set the check state of Data Object. Only the "Checked" and the
        "Unchecked" states can be set.

        Parameters
        ----------
        checked : bool
            Check state (True = "Checked", False = "Unchecked").
        """
        c = QC.Qt.CheckState.Checked if checked else QC.Qt.CheckState.Unchecked
        self.setData(0, 10, c)

    def subgroup(self) -> DataSubGroup | None:
        """
        Return the parent subgroup that holds this Data Object.

        Returns
        -------
        subgroup : DataSubGroup | None
            Parent subgroup or None, if this Data Object hasn't got one.
        """
        if isinstance(subgroup := self.parent(), DataSubGroup):
            return subgroup
        return


class SampleMapsSelector(QW.QWidget):
    updateRequested = QC.pyqtSignal()
    mapsListUpdated = QC.pyqtSignal()
    mapClicked = QC.pyqtSignal(DataObject)
    mapChanged = QC.pyqtSignal(DataObject)

    def __init__(
        self,
        maps_type: Literal["inmaps", "minmaps"],
        checkable: bool = True,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        Ready to use widget that allows loading maps from a sample and
        show them in a QTreeWidget, optionally enabling their selection
        through dedicated checkboxes. This widget sends signals to
        request maps data. These signals must be catched by the widget
        that holds the information, namely the DataManager.

        Parameters
        ----------
        maps_type : Literal["inmaps", "minmaps"]
            Use "inmaps" to list input maps or "minmaps" to list mineral
            maps.
        checkable : bool, optional
            Whether the individual maps in the list can be selected
            through checkboxes. The default is True.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        ValueError
            Raised if 'maps_type' is not "inmaps" or "minmaps".
        """
        super().__init__(parent)

        # Check for valid 'maps_type' argument
        if maps_type not in ("inmaps", "minmaps"):
            raise ValueError(f'Invalid "maps_type" argument: {maps_type}.')

        # Set main attributes
        self.maps_type = maps_type
        self.checkable = checkable
        self._available_samples: list[DataGroup] = []
        self._current_sample: DataGroup | None = None

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Sample combobox (Auto Update Combo Box)
        self.sample_combox = AutoUpdateComboBox()

        # Maps list (Tree Widget)
        self.maps_list = QW.QTreeWidget()
        self.maps_list.setHeaderHidden(True)
        self.maps_list.setStyleSheet(style.SS_MENU)

        # Set layout
        main_layout = QW.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(QW.QLabel("Select sample"))
        main_layout.addWidget(self.sample_combox)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.maps_list)
        self.setLayout(main_layout)

    def _connect_slots(self) -> None:
        """
        Signals-slots connector.

        """
        # Forward custom signals
        self.sample_combox.clicked.connect(self.updateRequested)
        self.maps_list.itemClicked.connect(lambda i: self.mapClicked.emit(i))
        self.maps_list.itemChanged.connect(lambda i: self.mapChanged.emit(i))

        # Populate the maps list when a new sample is selected in the combobox
        self.sample_combox.activated.connect(self.updateMapsList)

    def updateCombox(self, samples: list[DataGroup]) -> None:
        """
        Populate the samples combobox with the available 'samples'. This
        method must be managed by the parent widget.

        Parameters
        ----------
        samples : list[DataGroup]
            List of DataGroup objects.
        """
        samples_names = [s.text(0) for s in samples]
        self.sample_combox.updateItems(samples_names)
        self._available_samples = samples
        self._current_sample = None

    def updateMapsList(self, sample_idx: int) -> None:
        """
        Populate the maps list with the maps owned by the sample at
        index 'sample_idx' inside the combobox.

        Parameters
        ----------
        sample_idx : int
            Index of the sample in the combo-box.
        """
        # Ensure the sample index is valid (safety)
        if not 0 <= sample_idx < len(self._available_samples):
            return

        # Clear the input maps lists and set current sample
        self.maps_list.clear()
        sample = self._available_samples[sample_idx]
        self._current_sample = sample

        # Add valid input map objects to the inmaps list
        subgr = sample.inmaps if self.maps_type == "inmaps" else sample.minmaps
        for c in subgr.getChildren():
            components = c.getComponents()
            if (data := components["data"]) is not None:
                item = DataObject(data)
                item.setName(components["name"])
                if self.checkable:
                    item.setChecked(True)
                self.maps_list.addTopLevelItem(item)

        # Send a signal to inform that maps data changed
        self.mapsListUpdated.emit()

    def getMap(self, idx: int) -> DataObject | None:
        """
        Get map data object at index 'idx'.

        Parameters
        ----------
        idx : int
            Index of map.

        Returns
        -------
        DataObject | None
            Map data object at given index or None, if index is invalid.

        """
        item = self.maps_list.topLevelItem(idx)
        if isinstance(item, DataObject) or item is None:
            return item
        return

    def getMaps(self) -> list[DataObject]:
        """
        Get all map data objects.

        Returns
        -------
        list[DataObject]
            List of all maps.
        """
        maps = [self.getMap(i) for i in range(self.mapsCount())]
        return [m for m in maps if m is not None]

    def getCheckedMaps(self) -> list[DataObject]:
        """
        Get the currently checked maps. Returns all maps if not
        checkable.

        Returns
        -------
        checked : list[DataObject]
            List of checked maps.
        """
        maps = self.getMaps()
        if self.checkable:
            maps = [m for m in maps if m.checkState(0)]
        return maps

    def mapsCount(self) -> int:
        """
        Return the amount of maps loaded in the maps list.

        Returns
        -------
        int
            Number of maps.
        """
        return self.maps_list.topLevelItemCount()

    def currentMap(self) -> DataObject | None:
        """
        Return the currently selected map.

        Returns
        -------
        DataObject | None
            Selected map object or None, if no map is selected.
        """
        if isinstance(item := self.maps_list.currentItem(), DataObject):
            return item
        return

    def currentSample(self) -> DataGroup | None:
        """
        Return the currently selected sample.

        Returns
        -------
        DataGroup | None
            Selected sample or None, if no sample is selected.
        """
        return self._current_sample

    def clear(self) -> None:
        """
        Clear out the entire widget.

        """
        self.sample_combox.clear()
        self.maps_list.clear()


class Legend(QW.QTreeWidget):
    colorChangeRequested = QC.pyqtSignal(QW.QTreeWidgetItem, tuple)  # i, color
    itemRenameRequested = QC.pyqtSignal(QW.QTreeWidgetItem, str)  # i, name
    itemsMergeRequested = QC.pyqtSignal(list, str)  # classes, new name
    itemHighlightRequested = QC.pyqtSignal(bool, QW.QTreeWidgetItem)  # high, i
    maskExtractionRequested = QC.pyqtSignal(list)  # classes

    def __init__(
        self,
        amounts: bool = True,
        context_menu: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        An interactive legend object for displaying and editing the
        names and display colors of mineral phases within a mineral map.
        Optionally, the mineral modal amounts are displayed as well. It
        includes an optional right-click context menu action for
        advanced interactions. It sends signals to notify each edit
        request, which must be catched and handled by other widgets to
        be effective.

        Parameters
        ----------
        amounts : bool, optional
            Include classes amounts (percentage) in the legend. The
            default is True.
        context_menu : bool, optional
            Whether a context menu should popup when right-clicking on a
            legend item. The default is False.
        parent : QWidget | None, optional
            GUI parent widget of this widget. The default is None.

        Raises
        ------
        RuntimeError
            Raised if the initialization of the header fails.
        """
        super().__init__(parent)

        # Define main attributes
        self._highlighted_item = None
        self.amounts = amounts
        self.has_context_menu = context_menu
        self.precision = settings.manager.get("data/decimal_precision")

        # Set header properties
        if (header := self.header()) is None:
            raise RuntimeError("Failed to initialize headers.")
        self._header = header
        self.setColumnCount(2 + self.amounts)
        self.setHeaderLabels(["Color", "Class"] + ["Amount"] * self.amounts)
        self._header.setSectionResizeMode(style.HRM_INTERACTIVE)

        # Set selection mode
        selection_mode = QW.QAbstractItemView.SelectionMode.ExtendedSelection
        self.setSelectionMode(selection_mode)

        # Item editing is only allowed via 'requestClassRename' method
        self.setEditTriggers(QW.QAbstractItemView.EditTrigger.NoEditTriggers)

        # Set custom scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(style.O_HORIZONTAL))
        self.setVerticalScrollBar(StyledScrollBar(style.O_VERTICAL))

        # Enable custom context menu
        self.setContextMenuPolicy(QC.Qt.ContextMenuPolicy.CustomContextMenu)

        # Set stylesheet (right-click menu when editing items name)
        self.setStyleSheet(style.SS_MENU)

        # Connect signals with slots
        self._connect_slots()

    def _connect_slots(self) -> None:
        """
        Signals-slots connector.

        """
        self.itemDoubleClicked.connect(self.onDoubleClick)
        if self.has_context_menu:
            self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point: QC.QPoint) -> None:
        """
        Shows a context menu with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event.
        """

        # Exit funtion if 'point' is not on any item (on void)
        if (item := self.itemAt(point)) is None:
            return

        # Create menu
        menu = QW.QMenu()
        menu.setStyleSheet(style.SS_MENU)

        # [Define Actions]
        #|Rename class
        rename = QW.QAction(style.getIcon("RENAME"), "Rename", menu)
        rename.triggered.connect(lambda: self.requestClassRename(item))
        #|Merge classes
        merge = QW.QAction(style.getIcon("MERGE"), "Merge", menu)
        merge.setEnabled(len(self.selectedItems()) > 1)
        merge.triggered.connect(self.requestClassMerge)
        #|Copy color HEX string
        copy_col = QW.QAction(style.getIcon("COPY"), "Copy color HEX", menu)
        copy_col.triggered.connect(lambda: self.copyColorHexToClipboard(item))
        #|Set color
        set_col = QW.QAction(style.getIcon("PALETTE"), "Set color", menu)
        set_col.triggered.connect(lambda: self.requestColorChange(item))
        #|Randomize color
        rand_col = QW.QAction(
            style.getIcon("RANDOMIZE_COLOR"), "Randomize color", menu
        )
        rand_col.triggered.connect(lambda: self.requestRandomColorChange(item))
        #|Higlight item
        highlight = QW.QAction(style.getIcon("HIGHLIGHT"), "Highlight", menu)
        highlight.setCheckable(True)
        highlight.setChecked(item == self._highlighted_item)
        highlight.toggled.connect(lambda t: self.requestItemHighlight(t, item))
        #|Extract mask
        extr_mask = QW.QAction(style.getIcon("ADD_MASK"), "Extract mask", menu)
        extr_mask.triggered.connect(self.requestMaskFromClass)

        # Populate menu
        menu.addActions((rename, merge))
        menu.addSeparator()
        menu.addActions((copy_col, set_col, rand_col))
        menu.addSeparator()
        menu.addActions((highlight, extr_mask))

        # Show the menu where the user triggered the event
        menu.exec(self.mapToGlobal(point))

    def onDoubleClick(self, item: QW.QTreeWidgetItem, column: int) -> None:
        """
        Wrapper method that triggers different actions depending on
        which column is double-clicked.

        Parameters
        ----------
        item : QTreeWidgetItem
            The legend item that was double-clicked.
        column : int
            The column that was double-clicked.
        """
        if column == 0:
            self.requestColorChange(item)
        elif column == 1:
            self.requestClassRename(item)
        else:
            self.requestItemHighlight(item != self._highlighted_item, item)

    def copyColorHexToClipboard(self, item: QW.QTreeWidgetItem) -> None:
        """
        Copy the selected phase color to the clipboard as a HEX string.

        Parameters
        ----------
        item : QTreeWidgetItem
            The selected phase item.
        """
        if (clipboard := QW.qApp.clipboard()) is not None:
            clipboard.setText(item.whatsThis(0))

    def requestColorChange(self, item: QW.QTreeWidgetItem) -> None:
        """
        Request to change the color of a class. The signal must be
        catched and handled by the widget that contains the legend.
        This method is also triggered when double-clicking on the item
        icon.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.
        """
        # Get the old color (HEX string) and the new color (RGB tuple)
        old_col = item.whatsThis(0)
        new_col = QW.QColorDialog.getColor(QG.QColor(old_col), self)

        # Emit the signal
        if new_col.isValid():
            rgb = tuple(new_col.getRgb()[:-1])
            self.colorChangeRequested.emit(item, rgb)

    def requestRandomColorChange(self, item: QW.QTreeWidgetItem) -> None:
        """
        Request to randomize the color of a class. The signal must be
        catched and handled by the widget that contains the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests the color change.
        """
        self.colorChangeRequested.emit(item, ())

    def changeItemColor(
        self, item: QW.QTreeWidgetItem, color: tuple[int, int, int]
    ) -> None:
        """
        Change the color of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item whose color must be changed.
        color : tuple[int, int, int]
            RGB triplet.
        """
        # Set the new color to the legend item
        item.setIcon(0, ColorIcon(color))

        # Also set the new whatsThis string
        item.setWhatsThis(0, iatools.rgb2hex(color))

    def requestClassRename(self, item: QW.QTreeWidgetItem) -> None:
        """
        Request to change the name of a class. The signal must be
        catched and handled by the widget that contains the legend. This
        method is also triggered when double-clicking on the item name.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item that requests to be renamed.
        """
        # Deny renaming protected '_ND'_ class
        old_name = item.text(1)
        if old_name == "_ND_":
            MsgBox(self, "Crit", "This class cannot be renamed.")
            return

        # Exit function if dialog is canceled or class is not renamed
        label = "Rename class (max. 8 ASCII characters)"
        name, ok = QW.QInputDialog.getText(
            self, self.windowTitle(), label, text=old_name
        )
        if not ok or name == old_name:
            return

        # Deny renaming to protected '_ND_' class
        elif name == "_ND_":
            MsgBox(self, "Crit", '"_ND_" is a protected class name.')

        # Deny renaming if the name already exists
        elif self.hasClass(name):
            MsgBox(self, "Crit", f"{name} is already taken.")

        # Deny renaming if name is NOT an ASCII <= 8 characters string
        elif 0 < len(name) <= 8 and name.isascii():
            self.itemRenameRequested.emit(item, name)
        else:
            MsgBox(self, "Crit", "Invalid name.")

    def renameClass(self, item: QW.QTreeWidgetItem, name: str) -> None:
        """
        Change the name of the item in the legend.

        Parameters
        ----------
        item : QTreeWidgetItem
            The item to be renamed.
        name : str
            New item name.
        """
        # Set the new name to the legend item
        item.setText(1, name)

    def requestClassMerge(self) -> None:
        """
        Request to merge two or more mineral classes. The signal must be
        catched and handled by the widget that contains the legend.

        """
        # Exit function if less than 2 classes are selected
        classes = [i.text(1) for i in self.selectedItems()]
        if len(classes) < 2:
            return

        # Deny renaming protected '_ND'_ class
        if "_ND_" in classes:
            MsgBox(self, "Crit", '"_ND_" class cannot be merged.')
            return

        # Exit function if dialog is canceled
        text = f"Merge {classes} in a new class (max. 8 ASCII characters):"
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), text)
        if not ok:
            return

        # Deny renaming to protected '_ND_' class
        elif name == "_ND_":
            MsgBox(self, "Crit", '"_ND_" is a protected class name.')

        # Deny renaming if name already exists outside selected classes
        elif name not in classes and self.hasClass(name):
            MsgBox(self, "Crit", f"{name} is already taken.")

        # Deny renaming if name is not an ASCII <= 8 characters string
        elif 0 < len(name) <= 8 and name.isascii():
            self.itemsMergeRequested.emit(classes, name)
        else:
            MsgBox(self, "Crit", "Invalid name.")

    def requestItemHighlight(
        self, toggled: bool, item: QW.QTreeWidgetItem
    ) -> None:
        """
        Request to highlight the selected mineral class, i.e., to show
        ONLY the selected class in map. The signal must be catched and
        handled by the widget that contains the legend. This method is
        also triggered when double-clicking on the class percentage.

        Parameters
        ----------
        toggled : bool
            Highlight state.
        item : QTreeWidgetItem
            The item to be highlighted.
        """
        # Change the current highlighted item attribute
        self._highlighted_item = item if toggled else None

        # Send the request signal
        self.itemHighlightRequested.emit(toggled, item)

    def requestMaskFromClass(self) -> None:
        """
        Request to extract a mask from the selected mineral classes. The
        signal must be catched and handled by the widget that contains
        the legend.

        """
        classes = [i.text(1) for i in self.selectedItems()]
        self.maskExtractionRequested.emit(classes)

    def setPrecision(self, value: int) -> None:
        """
        Set the number of decimals of the class amounts.

        Parameters
        ----------
        value : int
            Number of decimals to be shown in the legend.
        """
        self.precision = value
        # self.update()

    def addClass(
        self, name: str, color: tuple[int, int, int], amount: float
    ) -> None:
        """
        Add a new mineral class to the legend.

        Parameters
        ----------
        name : str
            Class name.
        color : tuple[int, int, int]
            Class color as RGB triplet.
        amount : float
            Class modal amount. This value is ignored if legend amounts
            are not enabled.
        """
        item = QW.QTreeWidgetItem(self)
        # Color icon [column 0]
        item.setIcon(0, ColorIcon(color))
        # Color HEX string ['virtual' column 0]
        item.setWhatsThis(0, iatools.rgb2hex(color))
        # Mineral class name [column 1]
        item.setText(1, name)
        # Mineral mode (optional) [column 2]
        if self.amounts:
            amount = round(amount, self.precision)
            item.setText(2, f"{amount}%")

    def hasClass(self, name: str) -> bool:
        """
        Check if the given class is already displayed in the legend. The
        search is done through the class name.

        Parameters
        ----------
        name : str
            Class name to check.

        Returns
        -------
        bool
            Whether the legend already contains an entry with the given
            name.
        """
        items = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        class_names = [i.text(1) for i in items if i is not None]
        return name in class_names

    def refresh(self, mineral_map: MineralMap) -> None:
        """
        Updates the legend.

        Parameters
        ----------
        mineral_map : MineralMap
            The Mineral Map object linked to the current view of the
            legend.
        """
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
        self._header.resizeSections(style.HRM_RESIZETOCONTENTS)


class ColorIcon(QG.QIcon):
    def __init__(
        self,
        color: tuple[int, int, int] | str,
        edgecolor: tuple[int, int, int] | str = style.IVORY,
        lw: int = 1,
        size: tuple[int, int] = (64, 64),
    ) -> None:
        """
        A simple plain colored icon, especially useful in legends.

        Parameters
        ----------
        color : tuple[int, int, int] | str
            Icon color as RGB triplet or HEX string.
        edgecolor : tuple[int, int, int] | str, optional
            Color of the icon border as RGB triplet or HEX string. The
            default is #F9F9F4 (IVORY).
        lw : int, optional
            Border line width. The default is 1.
        size : tuple[int, int], optional
            Icon size. The default is (64, 64).

        Raises
        ------
        TypeError
            Raised if 'color' is not a valid type.
        TypeError
            Raised if 'edgecolor' is not a valid type.
        """
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
            err = f"Color must be tuple or str, not {type(color)}"
            raise TypeError(err)

        # Add border
        if isinstance(edgecolor, tuple):
            pen = QG.QPen(QG.QColor(*edgecolor))
        elif isinstance(edgecolor, str):
            pen = QG.QPen(QG.QColor(edgecolor))
        else:
            err = f"Edgecolor must be tuple or str, not {type(edgecolor)}"
            raise TypeError(err)

        pen.setWidth(lw)
        painter = QG.QPainter(pixmap)
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
        bg: str | None = None,
        inner_gap: int = 1,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QPushButton, adding extra
        customization parameters.

        Parameters
        ----------
        icon : QIcon | None, optional
            Button icon. The default is None.
        text : str | None, optional
            Button text. The default is None.
        bg : str | None, optional
            Background color of the button. If None, the default color
            is used. The default is None.
        inner_gap : int, optional
            Space between the icon and the text. The default is 1.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)
        self.setSizePolicy(style.SP_PREFERRED, style.SP_FIXED)
        self.gap = inner_gap

        # Set icon and text
        if icon is not None:
            self.setIcon(icon)
        self.setText(text)

        # Overwrite default background color if requested
        ss = style.SS_BUTTON
        if bg:
            ss += "QPushButton {background-color: %s; font: bold;}" % (bg)  # noqa: UP031
        self.setStyleSheet(ss)

    def setText(self, text: str | None) -> None:
        """
        Reimplementation of the parent method, that adds a gap between
        the button's icon and the button's text.

        Parameters
        ----------
        text : str | None
            Button's text.
        """
        if text is not None and not self.icon().isNull():
            text = text.rjust(self.gap + len(text))
        return super().setText(text)


class StyledComboBox(QW.QComboBox):
    def __init__(
        self, tooltip: str | None = None, parent: QW.QWidget | None = None
    ) -> None:
        """
        QSS-styled reimplementation of a QComboBox, adding wheel event
        control and auto-generating tooltips.

        Parameters
        ----------
        tooltip : str | None, optional
            Fixed combo box tooltip. If None, the tooltip automatically
            changes when the current text changes. The default is None.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set automatic tooltip update if no tooltip is provided
        if tooltip:
            self.setToolTip(tooltip)
        else:
            self.setToolTip(self.currentText())
            self.currentTextChanged.connect(self.setToolTip)

        # Set stylesheet (SS_menu in case of editable combo box)
        self.setStyleSheet(style.SS_COMBOX + style.SS_MENU)

    def wheelEvent(self, e: QG.QWheelEvent | None) -> None:
        """
        Reimplementation of the parent method to better control mouse
        wheel scroll activation of the combobox. The event is accepted
        only if SHIFT is pressed during the scroll.

        Parameters
        ----------
        event : QWheelEvent | None
            The mouse wheel event.
        """
        if e is not None:
            if bool(e.modifiers() & QC.Qt.KeyboardModifier.ShiftModifier):
                super().wheelEvent(e)
            else:
                e.ignore()


class StyledDoubleSpinBox(QW.QDoubleSpinBox):
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        step: float = 0.1,
        decimals: int = 2,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QDoubleSpinBox, adding extra
        customization parameters and wheel event control.

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
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

        # Set number of decimal positions
        self.setDecimals(decimals)

        # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)

    def wheelEvent(self, e: QG.QWheelEvent | None) -> None:
        """
        Reimplementation of the parent method to better control mouse
        wheel scroll activation of the spinbox. The event is accepted
        only if SHIFT is pressed during the scroll. By default, if CTRL
        is also pressed the step is multiplied by 10.

        Parameters
        ----------
        event : QWheelEvent | None
            The mouse wheel event.
        """
        if e is not None:
            if bool(e.modifiers() & QC.Qt.KeyboardModifier.ShiftModifier):
                super().wheelEvent(e)
            else:
                e.ignore()


class StyledListWidget(QW.QListWidget):
    def __init__(
        self, ext_sel: bool = True, parent: QW.QWidget | None = None
    ) -> None:
        """
        QSS-styled reimplementation of a QListWidget, adding extra
        convenient methods.

        Parameters
        ----------
        ext_sel : bool, optional
            If extended selection mode should be enabled. The default is
            True.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(style.O_HORIZONTAL))
        self.setVerticalScrollBar(StyledScrollBar(style.O_VERTICAL))

        # Set extended selection mode if requested
        if ext_sel:
            sel_mode = QW.QAbstractItemView.SelectionMode.ExtendedSelection
            self.setSelectionMode(sel_mode)

    def getItems(self) -> list[QW.QListWidgetItem]:
        """
        Return all items in the list.

        Returns
        -------
        items : list[QListWidgetItem]
            List of items.
        """
        items = [self.item(row) for row in range(self.count())]
        items = [i for i in items if i is not None]
        return items

    def selectedRows(self) -> list[int]:
        """
        Return all selected rows.

        Returns
        -------
        selected_rows : list[int]
            List of selected row indices.
        """
        selected_items = self.selectedItems()
        selected_rows = [self.row(item) for item in selected_items]
        return selected_rows

    def removeSelected(self) -> None:
        """
        Remove all selected items from list.

        """
        for idx in sorted(self.selectedRows(), reverse=True):
            self.takeItem(idx)


class StyledRadioButton(QW.QRadioButton):
    def __init__(
        self,
        text: str = "",
        icon: QG.QIcon | None = None,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QRadioButton, adding extra
        customization parameters.

        Parameters
        ----------
        text : str
            The text of the radio button. The default is ''.
        icon : QIcon | None, optional
            The icon of the radio button. The default is None.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(text, parent)
        if icon is not None:
            self.setIcon(icon)
        self.setStyleSheet(style.SS_RADIOBUTTON)


class StyledScrollBar(QW.QScrollBar):
    def __init__(
        self, orientation: QC.Qt.Orientation, parent: QW.QWidget | None = None
    ) -> None:
        """
        QSS-styled reimplementation of a QScrollBar, with dynamic
        stylesheet assignment based on its orientation.

        Parameters
        ----------
        orientation : Qt.Orientation
            The orientation of the scroll bar.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(orientation, parent)

        # Set the stylesheet
        if orientation == QC.Qt.Orientation.Horizontal:
            self.setStyleSheet(style.SS_SCROLLBARH)
        else:
            self.setStyleSheet(style.SS_SCROLLBARV)


class StyledSpinBox(QW.QSpinBox):
    def __init__(
        self,
        min_value: int = 0,
        max_value: int = 100,
        step: int = 1,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QSpinBox, adding wheel event
        control and extra customization parameters.

        Parameters
        ----------
        min_value : int, optional
            Minimum range value. The default is 0.
        max_value : int, optional
            Maximum range value. The default is 100.
        step : int, optional
            Step value. The default is 1.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set range and single step values
        self.setRange(min_value, max_value)
        self.setSingleStep(step)

        # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)

    def wheelEvent(self, e: QG.QWheelEvent | None) -> None:
        """
        Reimplementation of the parent method to better control mouse
        wheel scroll activation of the spinbox. The event is accepted
        only if SHIFT is pressed during the scroll. By default, if CTRL
        is also pressed the step is multiplied by 10.

        Parameters
        ----------
        event : QWheelEvent | None
            The mouse wheel event.
        """
        if e is not None:
            if bool(e.modifiers() & QC.Qt.KeyboardModifier.ShiftModifier):
                super().wheelEvent(e)
            else:
                e.ignore()


class StyledTable(QW.QTableWidget):
    def __init__(
        self,
        rows: int,
        cols: int,
        ext_sel: bool = True,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QTableWidget, adding extra
        customization parameters and dedicated corner button behaviour.

        Parameters
        ----------
        rows : int
            Number of rows.
        cols : int
            Number of columns.
        ext_sel : bool, optional
            If extended selection mode should be enabled. The default is
            True.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        RuntimeError
            Raised if the intialization of horizontal header fails.
        RuntimeError
            Raised if the intialization of vertical header fails.
        """
        super().__init__(rows, cols, parent)

        # Initialize horizontal and vertical headers
        if (h_header := self.horizontalHeader()) is None:
            raise RuntimeError("Horizontal header initialization failed.")
        self.h_header = h_header

        if (v_header := self.verticalHeader()) is None:
            raise RuntimeError("Vertical header initialization failed.")
        self.v_header = v_header

        # Set custom scroll bars
        self.setHorizontalScrollBar(StyledScrollBar(style.O_HORIZONTAL))
        self.setVerticalScrollBar(StyledScrollBar(style.O_VERTICAL))

        # Set extended selection mode if requested
        if ext_sel:
            self.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)

        # Customize corner button behaviour
        self.corner_btn = self.findChild(QW.QAbstractButton, "")
        if self.corner_btn:
            self.corner_btn.setToolTip("Select all / deselect all")
            self.corner_btn.disconnect()
            self.corner_btn.clicked.connect(self.onCornerButtonClicked)

        # Set stylesheet (context menu)
        self.setStyleSheet(style.SS_MENU)

    def onCornerButtonClicked(self) -> None:
        """
        Select all items when not all items are currently selected.
        Otherwise deselect them all.

        """
        tot_items = self.columnCount() + self.rowCount()
        if len(self.selectedIndexes()) < tot_items:
            self.selectAll()
        else:
            self.clearSelection()


class StyledTabWidget(QW.QTabWidget):
    def __init__(
        self,
        tabs_align: Literal["c", "center", "l", "left", "r", "right"] = "left",
        wheel_scroll: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        QSS-styled reimplementation of a QTabWidget, with dedicated
        methods to improve the visual consistency of added tabs.

        Parameters
        ----------
        tabs_align : Literal["c", "center", "l", "left", "r", "right"], optional
            Tabs alignment in the tab bar. The default is "left".
        wheel_scroll : bool, optional
            Whether mouse wheel event should allow to change the current
            tab. The default is False.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        RuntimeError
            Raised if the initialization of the tab bar fails.
        ValueError
            Raised if 'tabs_align' is not a valid string.
        """
        super().__init__(parent)

        # Set tabs unscrollable via mouse wheel if requested
        if wheel_scroll:
            if (tabbar := self.tabBar()) is None:
                raise RuntimeError("Failed to initialize tab bar.")
            self.tabbar = tabbar
        else:
            self.tabbar = self.UnscrollableTabBar()
            self.setTabBar(self.tabbar)

        # Set stylesheet
        match tabs_align:
            case "left" | "l":
                ss_align = "QTabWidget::tab-bar {alignment: left;}"
            case "right" | "r":
                ss_align = "QTabWidget::tab-bar {alignment: right;}"
            case "center" | "c":
                ss_align = "QTabWidget::tab-bar {alignment: center;}"
            case _:
                err = f"Invalid tabs_align argument: '{tabs_align}'"
                raise ValueError(err)

        self.setStyleSheet(style.SS_TABWIDGET + ss_align)

    def wrap_tab(
        self,
        qobject: QW.QLayout | QW.QWidget,
        icon: QG.QIcon | None = None,
        title: str | None = None,
        tight: bool = False,
    ) -> int:
        """
        Wrap 'qobject' in a GroupArea and add it to the tab widget. This
        method is useful to achive a consistent look of the tab widget,
        regardless of the type of object that is set as its tab.

        Parameters
        ----------
        qobject : QLayout | QWidget
            A layout-like or widget-like object to be added as tab.
        icon : QIcon | None, optional
            Tab icon. The default is None.
        title : str | None, optional
            Tab name. The default is None.
        tight : bool, optional
            Whether the tab's layout should take all the available
            space. The default is False.

        Raises
        ------
        TypeError
            Raised when 'qobject' is not a QWidget or a QLayout.
        """
        # Ensure that 'qobject' is a valid widget or layout
        if not isinstance(qobject, (QW.QLayout, QW.QWidget)):
            raise TypeError(f'Invalid "qobject" type: {type(qobject)}.')

        # Set a GroupArea as the wrapper widget for 'qobject'
        has_frame = self.tabbar.documentMode()
        tab = GroupArea(qobject, frame=has_frame, tight=tight)
        if has_frame:
            tab.setStyleSheet(None)
            tab.setObjectName("FramedWrapper")  # Name for custom qss

        # Add the tab with or without an icon
        if icon:
            return super().addTab(tab, icon, title)
        else:
            return super().addTab(tab, title)

    def setTabBarExpanding(self) -> None:
        """
        A workaround to display expanding tabs, without compromising the
        style of the Tab Widget pane.

        """
        self.tabbar.setExpanding(True)
        self.tabbar.setDocumentMode(True)
        self.tabbar.setUsesScrollButtons(False)
        ss = "QTabWidget::pane {padding: 0px; border: 0px; margin: 0px;}"
        self.setStyleSheet(self.styleSheet() + ss)

    def currentWidget(self) -> QW.QWidget | None:
        """
        Reimplementation of the parent method that ensures to get the
        object held by the currently selected tab and not its wrapper
        widget.

        Returns
        -------
        QW.QWidget | None
            Widget in the current tab or None, if no tab is selected.
        """
        return self.widget(self.currentIndex())

    def widget(self, index: int) -> QW.QWidget | None:
        """
        Reimplementation of the parent method that ensures to get the
        object held by the tab at index 'index' and not its wrapper
        widget.

        Parameters
        ----------
        index : int
            Index of tab.

        Returns
        -------
        QWidget | None
            Widget in the tab or None, if the index is invalid.
        """
        wrapper: QW.QWidget | GroupArea | None = super().widget(index)
        if isinstance(wrapper, GroupArea):
            wrapped_object = wrapper.wrappedObject()
            if isinstance(wrapped_object, QW.QWidget):
                return wrapped_object
        return wrapper

    class UnscrollableTabBar(QW.QTabBar):
        def __init__(self, parent: QW.QWidget | None = None) -> None:
            """
            Dedicated reimplementation of a QTabBar specifically for a
            StyledTabWidget. It disables tab scrolling with mouse wheel.

            Parameters
            ----------
            parent : QW.QWidget | None, optional
                The GUI parent of this widget. The default is None.
            """
            super().__init__(parent)

        def wheelEvent(self, event: QG.QWheelEvent | None) -> None:
            """
            Reimplement the parent method to always ignore it.

            Parameters
            ----------
            event : QWheelEvent | None
                The wheel mouse event.
            """
            if event is not None:
                event.ignore()


class AutoUpdateComboBox(StyledComboBox):
    clicked = QC.pyqtSignal()

    def __init__(self, parent: QW.QWidget | None = None, **kwargs) -> None:
        """
        A Styled ComboBox that emits a signal when the combo box button
        is clicked. This signal can be connected to a custom method that
        allows to update the list of items in the combo box.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        **kwargs
            Parent class arguments (see StyledComboBox class).
        """
        super().__init__(parent=parent, **kwargs)

    def showPopup(self) -> None:
        """
        Reimplementation of the parent method. It emits the new clicked
        signal just before showing the pop up.

        """
        self.clicked.emit()
        super().showPopup()

    def updateItems(self, items: list[str]) -> None:
        """
        Refresh the combo box list of items.

        Parameters
        ----------
        items : list[str]
            List of new items.
        """
        self.clear()
        self.addItems(items)


class RadioBtnLayout(QW.QBoxLayout):
    selectionChanged = QC.pyqtSignal(int)  # id of button

    def __init__(
        self,
        names: Sequence[str],
        icons: Sequence[QG.QIcon] | None = None,
        default: int = 0,
        orient: Literal["h", "horizontal", "v", "vertical"] = "vertical",
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        Convenient class to group multiple radio buttons in a single
        layout. It makes use of the QButtonGroup class to quickly access
        each button.

        Parameters
        ----------
        names : Sequence[str]
            Sequence of radio button names.
        icons : Sequence[QIcon] | None, optional
            Sequence of button icons. If provided, it must be the same
            length of 'names'. The default is None.
        default : int, optional
            Id of the button selected by default. The default is 0.
        orient : Literal["h", "horizontal", "v", "vertical"], optional
            Orientation of the layout. The default is "vertical".
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        ValueError
            Raised if 'orient' is an invalid string.
        ValueError
            Raised if 'names' and 'icons' lists have different lengths.
        """
        # Set the current selected button id
        self._selected_id = default

        # Set the orientation of the layout
        match orient:
            case "horizontal" | "h":
                direction = QW.QBoxLayout.Direction.LeftToRight
            case "vertical" | "v":
                direction = QW.QBoxLayout.Direction.TopToBottom
            case _:
                raise ValueError(f'"{orient}" is not a valid orientation.')

        # Check for same length of names and icons (if provided) lists
        if icons is not None and len(names) != len(icons):
            raise ValueError("Icons and names list have different lengths.")

        # Set a QButtonGroup working behind the scene
        super().__init__(direction, parent)
        self.btn_group = QW.QButtonGroup()

        # Populate layout with radio buttons and connect their signals
        for i, n in enumerate(names):
            icon = None if icons is None else icons[i]
            btn = StyledRadioButton(n, icon)
            self.btn_group.addButton(btn, id=i)
            if i == default:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, idx=i: self.onSelect(idx))
            self.addWidget(btn)

    def onSelect(self, id_: int) -> None:
        """
        Send the 'selectionChanged' signal if a new button has been
        selected.

        Parameters
        ----------
        id_ : int
            The selected radio button id.
        """
        if id_ != self._selected_id:
            self._selected_id = id_
            self.selectionChanged.emit(id_)

    def button(self, id_: int) -> QW.QAbstractButton | None:
        """
        Return the radio button with given id.

        Parameters
        ----------
        id : int
            The radio button id.

        Returns
        -------
        btn : StyledRadioButton
            The radio button with id 'id_'.
        """
        btn = self.btn_group.button(id_)
        return btn

    def buttons(self) -> list[QW.QAbstractButton]:
        """
        Return all radio buttons.

        Returns
        -------
        btns: list[QAbstractButton]
            All radio buttons.
        """
        btns = self.btn_group.buttons()
        return btns

    def get_checked(self) -> QW.QAbstractButton:
        """
        Return the checked radio button.

        Returns
        -------
        StyledRadioButton
            The checked radio button.
        """
        return self.btn_group.checkedButton()

    def get_checked_id(self) -> int:
        """
        Return the checked radio button id.

        Returns
        -------
        int
            The checked radio button id.
        """
        return self.btn_group.checkedId()


class GroupArea(QW.QGroupBox):
    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str | None = None,
        checkable: bool = False,
        frame: bool = True,
        tight: bool | None = None,
        align: Literal["c", "center", "l", "left", "r", "right"] = "c",
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        A convenient class to easily wrap a layout or a widget in a
        QGroupBox.

        Parameters
        ----------
        qobject : QLayout | QWidget
            A layout-like or a widget-like object.
        title : str | None, optional
            The title of the group box. The default is None.
        checkable : bool, optional
            Whether the group box is checkable. The default is False.
        frame : bool, optional
            Whether the area should show a visible frame. This is forced
            to be True if "title" is provided. The default is True.
        tight : bool | None, optional
            Whether the layout of the group box should take all the
            available space. If None, this is automatically set based on
            "title" and "frame". The default is None.
        align : Literal["c", "center", "l", "left", "r", "right"], optional
            The alignment of the title. The default is "c".
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        TypeError
            Raised if an invalid 'qobject' argument is passed.
        ValueError
            Raised if an invalid 'align' argument is passed.
        """
        # Set the style, based on 'title', 'align' and 'frame'
        if title:
            super().__init__(title, parent)
            match align:
                case "left" | "l":
                    align_css = "top left"
                case "right" | "r":
                    align_css = "top right"
                case "center" | "c":
                    align_css = "top center"
                case _:
                    raise ValueError(f'Invalid "align" argument: {align}.')

            ss = f"QGroupBox::title {{subcontrol-position: {align_css};}}"
            self.setStyleSheet(style.SS_GROUPAREA_TITLE + ss)

        else:
            super().__init__(parent)
            if frame:
                self.setStyleSheet(style.SS_GROUPAREA_BASE)
            else:
                self.setStyleSheet(style.SS_GROUPAREA_NOFRAME)

        # Set if the GroupBox is checkable or not
        self.setCheckable(checkable)

        # Set a layout according to 'qobject' type
        if isinstance(qobject, QW.QLayout):
            layout = qobject
        elif isinstance(qobject, QW.QWidget):
            layout = QW.QVBoxLayout()
            layout.setObjectName("GroupAreaWrapperLayout")
            layout.addWidget(qobject)
        else:  # raise error if 'qobject' is invalid
            raise TypeError(f'Invalid "qobject" type: {type(qobject)}.')

        # Set a tight layout, if required
        if tight or (tight is None and not title and not frame):
            layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def wrappedObject(self) -> QW.QLayout | QW.QWidget | None:
        """
        Return the object wrapped in the group area.

        Returns
        -------
        QW.QLayout | QW.QWidget | None
            Original object wrapped in the group area.
        """
        # Return None if the layout is invalid (safety)
        if (layout := self.layout()) is None:
            return None

        # If the layout is the original object, return it
        if layout.objectName() != "GroupAreaWrapperLayout":
            return layout

        # If the layout is a wrapper, get and return its wrapped object
        if (item := layout.itemAt(0)) is not None:
            return item.widget()

        # Safety fallback
        return None


class CollapsibleArea(QW.QWidget):
    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str,
        frame: bool = True,
        collapsed: bool = True,
        animated: bool = True,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        A convenient widget to expand/collapse an entire section of the
        GUI. It includes an arrow button to switch from expanded to
        collapsed view.

        Parameters
        ----------
        qobject : QLayout | QWidget
            A layout-like or a widget-like object.
        title : str
            The title of the section.
        frame : bool, optional
            Whether the section area should have a frame. The default is
            True.
        collapsed : bool, optional
            Whether the section should be collapsed by default. The
            default is True.
        animated : bool, optional
            Whether a simple animation should be played when switching
            views. Set this to False if certain widgets, like matplotlib
            canvas, send rendering errors or warnings.
            The default is True.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set main attributes
        self._framed = frame
        self._collapsed = collapsed
        self._animated = animated
        self._section = qobject
        self._title = title

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

        # Set collapsed view if required
        if collapsed:
            self.area.setMaximumHeight(0)
            self.arrow.setArrowType(QC.Qt.ArrowType.RightArrow)

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Arrow button (Tool Button)
        self.arrow = QW.QToolButton()
        self.arrow.setStyleSheet(style.SS_TOOLBUTTON)
        self.arrow.setArrowType(QC.Qt.ArrowType.DownArrow)

        # Section title (Label)
        self.title = QW.QLabel(self._title)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)

        # Section area (Group Area)
        # 'frame' also influences 'tight' (see GroupArea for details)
        self.area = GroupArea(self._section, frame=self._framed, tight=None)

        # Expand/Collapse animation (Property Animation)
        self.animation = QC.QPropertyAnimation(self.area, b"maximumHeight")
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
        """
        Signals-slots connector.

        """
        # Expand/collapse section when arrow button is clicked
        self.arrow.clicked.connect(self.onArrowClicked)

    def collapsed(self) -> bool:
        """
        Check if the section is currently collapsed or not.

        Returns
        -------
        bool
            Whether the section is collapsed.

        """
        return self._collapsed

    def onArrowClicked(self) -> None:
        """
        Slot connected to the button clicked signal from the arrow
        button. It determines which action should be performed based on
        the current state of the section.

        """
        self.expand() if self.collapsed() else self.collapse()

    def collapse(self) -> None:
        """
        Collapse the section and change the arrow type.

        """
        self._collapsed = True
        self.arrow.setArrowType(QC.Qt.ArrowType.RightArrow)

        if self._animated:
            self.animation.setStartValue(self.area.height())
            self.animation.setEndValue(0)
            self.animation.start()
        else:
            self.area.setVisible(False)

    def expand(self) -> None:
        """
        Expand the section and change the arrow type.

        """
        self._collapsed = False
        self.arrow.setArrowType(QC.Qt.ArrowType.DownArrow)

        if self._animated:
            self.animation.setStartValue(0)
            self.animation.setEndValue(self.area.sizeHint().height())
            self.animation.start()
        else:
            self.area.setVisible(True)


class GroupScrollArea(QW.QScrollArea):
    def __init__(
        self,
        qobject: QW.QLayout | QW.QWidget,
        title: str | None = None,
        title_align: Literal["c", "center", "l", "left", "r", "right"] = "c",
        hscroll: bool = True,
        vscroll: bool = True,
        frame: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        A convenient class to easily wrap a layout or a widget in a
        scrollable area.

        Parameters
        ----------
        qobject : QLayout | QWidget
            A layout-like or a widget-like object.
        title : str | None, optional
            If set, 'qobject' will be wrapped in a GroupArea that shows
            this title. The default is None.
        title_align : Literal["c", "center", "l", "left", "r", "right"], optional
            The alignment of the title. The default is "c".
        hscroll : bool, optional
            Show the horizontal scrollbar. The default is True.
        vscroll : bool, optional
            Show the vertical scrollbar. The default is True.
        frame : bool, optional
            Whether the scroll area should have a visible frame. The
            default is False.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        TypeError
            Raised if an invalid 'qobject' argument is passed.
        """
        super().__init__(parent)

        # Set the stylesheet
        if frame:
            self.setStyleSheet(style.SS_GROUPSCROLLAREA_FRAME)
        else:
            self.setStyleSheet(style.SS_GROUPSCROLLAREA_NOFRAME)

        # Set the scrollbars
        self.setHorizontalScrollBar(StyledScrollBar(style.O_HORIZONTAL))
        self.setVerticalScrollBar(StyledScrollBar(style.O_VERTICAL))
        scroll_as_needed = QC.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        scroll_always_off = QC.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        h_scroll_policy = scroll_as_needed if hscroll else scroll_always_off
        v_scroll_policy = scroll_as_needed if vscroll else scroll_always_off
        self.setHorizontalScrollBarPolicy(h_scroll_policy)
        self.setVerticalScrollBarPolicy(v_scroll_policy)

        # Set a layout according to 'qobject' type
        if isinstance(qobject, QW.QLayout):
            layout = qobject
        elif isinstance(qobject, QW.QWidget):
            layout = QW.QVBoxLayout()
            layout.setObjectName("GroupScrollAreaWrapperLayout")
            layout.addWidget(qobject)
        else:  # raise error if 'qobject' is invalid
            raise TypeError(f'Invalid "qobject" of type {type(qobject)}.')

        # Set a wrapper widget
        if title is None:
            wrapper = QW.QWidget()
            layout.setContentsMargins(0, 0, 0, 0)
            wrapper.setLayout(layout)
        else:
            wrapper = GroupArea(layout, title, align=title_align)

        # Insert the wrapper widget in the scroll area
        wrapper.setObjectName("GroupScrollAreaWrapper")  # Name for custom qss
        self.setWidget(wrapper)
        self.setWidgetResizable(True)

    def wrappedObject(self) -> QW.QLayout | QW.QWidget | None:
        """
        Return the object wrapped in the scroll area.

        Returns
        -------
        QW.QLayout | QW.QWidget | None
            Original object wrapped in the scroll area.
        """
        # Return None if the wrapper is invalid (safety)
        wrapper: QW.QWidget | GroupArea | None = self.widget()
        if wrapper is None:
            return None

        # If wrapper is GroupArea, use its dedicated method
        if isinstance(wrapper, GroupArea):
            return wrapper.wrappedObject()

        # If wrapper is QWidget and its layout is the original object,
        # return it
        lyt = wrapper.layout()
        if (
            lyt is not None
            and lyt.objectName() != "GroupScrollAreaWrapperLayout"
        ):
            return lyt

        # If wrapper is QWidget and its layout is a wrapper layout, get
        # and return its wrapped object (which is a widget)
        if lyt is not None and (item := lyt.itemAt(0)) is not None:
            return item.widget()

        # Safety fallback
        return None


class SplitterLayout(QW.QBoxLayout):
    def __init__(
        self,
        spacing: int = 5,
        orient: Literal["h", "horizontal", "v", "vertical"] = "horizontal",
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        A ready to use layout box that comes with pre-coded splitters
        dividing each inserted widget.

        Parameters
        ----------
        spacing : int, optional
            Spacing between widgets within the layout. The default is 5.
        orient : Literal["h", "horizontal", "v", "vertical"], optional
            The orientation of the layout. The default is "horizontal".
        parent : QWidget | None, optional
            The GUI parent of this layout. The default is None.

        Raises
        ------
        ValueError
            Raised if 'orient' is not a valid string.
        """
        # Set main attributes
        self.splitter_spacing = spacing
        self.orientation = orient

        # Set the orientation of the layout and the splitter
        match self.orientation:
            case "horizontal" | "h":
                layout_direction = QW.QBoxLayout.Direction.LeftToRight
                splitter_orientation = style.O_HORIZONTAL
            case "vertical" | "v":
                layout_direction = QW.QBoxLayout.Direction.TopToBottom
                splitter_orientation = style.O_VERTICAL
            case _:
                raise ValueError(f'"{orient}" is not a valid orientation.')

        # Initialize the layout and add the splitter. Here we are using
        # the super method 'addWidget' to add the splitter, because it
        # has been reimplemented (see below), and calling it would cause
        # infinite recursions.
        super().__init__(layout_direction, parent)
        self.splitter = QW.QSplitter(splitter_orientation)
        opaque = settings.manager.get("GUI/smooth_animation")
        self.splitter.setOpaqueResize(opaque)
        self.splitter.setStyleSheet(style.SS_SPLITTER)
        super().addWidget(self.splitter)

    def _getLayoutMargins(self, layout: QW.QLayout, index: int) -> QC.QMargins:
        """
        Adjust content margins of given 'layout' at the given 'index' to
        ensure a visually consistent spacing of items in the splitter.

        Parameters
        ----------
        layout : QLayout
            Layout whose margins should be adjusted.
        index : int
            Positional index of the layout in the splitter.

        Returns
        -------
        QMargins
            Adjusted content margins.
        """
        margins = layout.contentsMargins()
        max_index = self.splitter.count()
        match self.orientation:
            case "horizontal" | "h":
                if index == 0:  # only at left of a splitter
                    margins.setRight(margins.right() + self.splitter_spacing)
                elif index == max_index:  # only at right of a splitter
                    margins.setLeft(margins.left() + self.splitter_spacing)
                else:  # between two splitters
                    margins.setRight(margins.right() + self.splitter_spacing)
                    margins.setLeft(margins.left() + self.splitter_spacing)
            case "vertical" | "v":
                if index == 0:  # only at top of a splitter
                    margins.setBottom(margins.bottom() + self.splitter_spacing)
                elif index == max_index:  # only at bottom of a splitter
                    margins.setTop(margins.top() + self.splitter_spacing)
                else:  # between two splitters
                    margins.setBottom(margins.bottom() + self.splitter_spacing)
                    margins.setTop(margins.top() + self.splitter_spacing)
        return margins

    def insertLayout(
        self, index: int, layout: QW.QLayout | None, stretch: int = 0
    ) -> None:
        """
        Reimplementation of the parent method, that ensures that the
        'layout' is inserted in the main splitter widget at 'index'
        after being wrapped inside a helper widget. If index is invalid,
        the layout is inserted at the end of the splitter.

        Parameters
        ----------
        index : int
            Index of insertion.
        layout : QLayout | None
            Layout to be inserted. If None, this method does nothing.
        stretch : int, optional
            Optional stretch for the inserted layout. The default is 0.
        """
        if layout is not None:
            wrapper = QW.QWidget()
            layout.setContentsMargins(self._getLayoutMargins(layout, index))
            wrapper.setLayout(layout)
            self.splitter.insertWidget(index, wrapper)
            self.splitter.setStretchFactor(index, stretch)

    def addLayout(self, layout: QW.QLayout | None, stretch: int = 0) -> None:
        """
        Reimplementation of the parent method, that ensures that the
        'layout' is appended to the main splitter widget after being
        wrapped inside a helper widget.

        Parameters
        ----------
        layout : QLayout | None
            Layout to be added. If None, this method does nothing.
        stretch : int, optional
            Optional stretch for the added layout. The default is 0.
        """
        if layout is not None:
            index = self.splitter.count()
            wrapper = QW.QWidget()
            layout.setContentsMargins(self._getLayoutMargins(layout, index))
            wrapper.setLayout(layout)
            self.splitter.addWidget(wrapper)
            self.splitter.setStretchFactor(index, stretch)

    def insertWidget(
        self,
        index: int,
        widget: QW.QWidget | None,
        stretch: int = 0,
        alignment: QC.Qt.AlignmentFlag = QC.Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        """
        Reimplementation of the parent method, that ensures that the
        'widget' is inserted in the main splitter widget at 'index'
        after being wrapped inside a helper widget. If index is invalid,
        the widget is inserted at the end.

        Parameters
        ----------
        index : int
            Index of insertion.
        widget : QWidget | None
            Widget to be inserted. If None, this method does nothing.
        stretch : int, optional
            Optional stretch for the inserted widget. The default is 0.
        alignment : AlignmentFlag
            Unused. Only required for a compatible method override.
        """
        if widget is not None:
            wrapper = QW.QWidget()
            layout = QW.QVBoxLayout()
            layout.addWidget(widget)
            layout.setContentsMargins(self._getLayoutMargins(layout, index))
            wrapper.setLayout(layout)
            self.splitter.insertWidget(index, wrapper)
            self.splitter.setStretchFactor(index, stretch)

    def addWidget(
        self,
        a0: QW.QWidget | None,
        stretch: int = 0,
        alignment: QC.Qt.AlignmentFlag = QC.Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        """
        Reimplementation of the parent method, that ensures that the
        'widget' is appended to the main splitter widget after being
        wrapped inside a helper widget.

        Parameters
        ----------
        a0 : QWidget | None
            Widget to be added. If None, this method does nothing.
        stretch : int, optional
            Optional stretch for the added widget. The default is 0.
        alignment : AlignmentFlag
            Unused. Only required for a compatible method override.
        """
        if self.widget is not None:
            index = self.splitter.count()
            wrapper = QW.QWidget()
            layout = QW.QVBoxLayout()
            layout.addWidget(a0)
            layout.setContentsMargins(self._getLayoutMargins(layout, index))
            wrapper.setLayout(layout)
            self.splitter.addWidget(wrapper)
            self.splitter.setStretchFactor(index, stretch)


class SplitterGroup(QW.QSplitter):
    def __init__(
        self,
        qobjects: Collection[QW.QLayout | QW.QWidget] | None = None,
        stretches: Collection[int] | None = None,
        orient: QC.Qt.Orientation = QC.Qt.Orientation.Horizontal,
    ) -> None:
        """
        A convenient class to quickly group multiple widgets in a
        QSplitter.

        Parameters
        ----------
        qobjects : Collection[QLayout | QWidget] | None, optional
            Collection of layout-like or widget-like objects. If None,
            an empty splitter group is initialized. The default is None.
        stretches : Collection[int] | None, optional
            Collection of stretch values for each object. It must have
            the same size of 'qobjects'. If None, stretches are not set.
            The default is None.
        orient : Qt.Orientation, optional
            Orientation of the splitter. The default is Qt.Horizontal.

        Raises
        ------
        TypeError
            Raised if 'qobjects' contains an invalid object.
        AssertionError
            Raised if 'qobjects' and 'stretches' have different sizes.
        """
        # Use super class to create the splitter and set its stylesheet
        self.orient = orient
        super().__init__(self.orient)
        self.setOpaqueResize(settings.manager.get("GUI/smooth_animation"))
        self.setStyleSheet(style.SS_SPLITTER)

        # Add objects to the splitter; wrap layout objects in QWidgets
        if qobjects is None:
            qobjects = []

        for obj in qobjects:
            if isinstance(obj, QW.QLayout):
                w_obj = QW.QWidget()
                w_obj.setLayout(obj)
                self.addWidget(w_obj)

            elif isinstance(obj, QW.QWidget):
                self.addWidget(obj)

            else:  # raise error if object is invalid
                raise TypeError(f'Invalid type in "qobjects": {type(obj)}')

        # Add stretches to each object
        if stretches is not None:
            assert len(stretches) == len(qobjects)
            for i, s in enumerate(stretches):
                self.setStretchFactor(i, s)

    def addStretchedWidget(
        self, widget: QW.QWidget, stretch: int, idx: int = -1
    ) -> None:
        """
        Add a widget to the splitter with given stretch. If an index is
        provided, the insertion position can be choosen.

        Parameters
        ----------
        widget : QWidget
            Widget to be added.
        stretch : int
            Stretch factor for the added widget.
        idx : int, optional
            Index of insertion. If invalid or -1, the widget is inserted
            at the end. The default is -1.
        """
        self.insertWidget(idx, widget)
        self.setStretchFactor(idx, stretch)


class FramedLabel(QW.QLabel):
    def __init__(
        self, text: str = "", parent: QW.QWidget | None = None
    ) -> None:
        """
        A label decorated with a squared frame. Its text is selectable
        and its size policy is fixed to avoid visual glitches when using
        screens with different resolutions.

        Parameters
        ----------
        text : str, optional
            Label text. The default is ''.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(text, parent)

        # Set widget attributes
        self.setSizePolicy(style.SP_IGNORED, style.SP_FIXED)
        text_selectable = QC.Qt.TextInteractionFlag.TextSelectableByMouse
        self.setTextInteractionFlags(text_selectable)

        # Set stylesheet
        self.setStyleSheet(style.SS_MENU + "QLabel {border: 1px solid black;}")

    def setFrameColor(self, color: str | tuple[int, int, int]) -> None:
        """
        Change the color of the label frame.

        Parameters
        ----------
        color : str | tuple[int, int, int]
            Color as a HEX string or an RGB triplet.
        """
        # Convert RGB triplet to HEX string
        if isinstance(color, tuple):
            color = iatools.rgb2hex(color)

        ss = f"QLabel {{border-color: {color};}}"
        self.setStyleSheet(self.styleSheet() + ss)

    def setFrameWidth(self, width: int) -> None:
        """
        Change the width of the label frame.

        Parameters
        ----------
        width : int
            Frame width.
        """
        ss = f"QLabel {{border-width: {width}px;}}"
        self.setStyleSheet(self.styleSheet() + ss)


class PathLabel(FramedLabel):
    def __init__(
        self,
        fullpath: str = "",
        full_display: bool = True,
        elide: bool = True,
        placeholder: str = "",
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        A special type of FramedLabel which allows easy management and
        display of file paths.

        Parameters
        ----------
        fullpath : str, optional
            The full filepath. The default is ''.
        full_display : bool, optional
            Whether the label should display the full filepath. If
            False, only the file name is displayed. The default is True.
        elide : bool, optional
            Whether the label should automatically wrap long text. The
            default is True.
        placeholder : str, optional
            Text to be displayed when the label is cleared. The default
            is ''.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent=parent)

        # Set main attributes
        self.fullpath = fullpath
        self.full_display = full_display
        self.elide = elide
        self.placeholder = placeholder

        # Set widget attributes
        self.setAlignment(QC.Qt.AlignmentFlag(style.A_LEFT | style.A_VCENTER))

        # Set stylesheet
        self.setStyleSheet(style.SS_PATHLABEL + style.SS_MENU)

        # Draw the file path
        self.display()

    @property
    def _displayedText(self) -> str:
        """
        Return the path string to be displayed.

        Returns
        -------
        text : str
            Displayed path.
        """
        text = self.fullpath
        if not self.full_display:
            text = cf.path2filename(text, ext=True)
        if text == "":
            text = self.placeholder

        return text

    def display(self) -> None:
        """
        Display the filepath in the QLabel.

        """
        text = self._displayedText
        self.setTextElided(text) if self.elide else self.setText(text)
        self.setToolTip(self.fullpath)

    def setTextElided(self, text: str) -> None:
        """
        Wrap long text.

        Parameters
        ----------
        text : str
            Text to be elided.
        """
        metrics = self.fontMetrics()
        w = self.width() - 2
        elided = metrics.elidedText(text, QC.Qt.TextElideMode.ElideRight, w)
        self.setText(elided)

    def setPath(self, path: str | None, auto_display: bool = True) -> None:
        """
        Set a new filepath.

        Parameters
        ----------
        path : str | None
            The new filepath. If None, the text "*Path not found" will
            be displayed.
        auto_display : bool, optional
            Whether the new filepath should also be automatically
            displayed. The default is True.
        """
        self.fullpath = "*Path not found" if path is None else path
        if auto_display:
            self.display()

    def clearPath(self) -> None:
        """
        Remove the current filepath.

        """
        self.setText(self.placeholder)
        self.fullpath = ""
        self.setToolTip("")

    def resizeEvent(self, a0: QG.QResizeEvent | None) -> None:
        """
        Reimplementation of the parent method. If text wrapping is
        enabled, this allows the label to elide the text, adapting it to
        the new size.

        Parameters
        ----------
        a0 : QResizeEvent | None
            The resize event.
        """
        if a0 is not None:
            a0.accept()
            if self.elide:
                self.setTextElided(self._displayedText)


class DescriptiveProgressBar(QW.QWidget):
    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        A progress bar that shows a description of the current process.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Description label (Label)
        self.desc = QW.QLabel()
        self.desc.setSizePolicy(style.SP_IGNORED, style.SP_FIXED)

        # Progress bar (Progress Bar)
        self.pbar = QW.QProgressBar()

        # Adjust main layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.desc, alignment=style.A_CENTER)
        layout.addWidget(self.pbar)
        self.setLayout(layout)

    def setMinimum(self, minimum: int) -> None:
        """
        Set minimum progress bar value.

        Parameters
        ----------
        minimum : int
            Minimum value.
        """
        self.pbar.setMinimum(minimum)

    def setMaximum(self, maximum: int) -> None:
        """
        Set maximum progress bar value.

        Parameters
        ----------
        maximum : int
            Maximum value.
        """
        self.pbar.setMaximum(maximum)

    def setRange(self, minimum: int, maximum: int) -> None:
        """
        Set progress bar range.

        Parameters
        ----------
        minimum : int
            Minimum value.
        maximum : int
            Maximum value
        """
        self.pbar.setRange(minimum, maximum)

    def setUndetermined(self) -> None:
        """
        Set the prograss bar in an undetermined state.

        """
        self.pbar.setRange(0, 0)

    def undetermined(self) -> bool:
        """
        Check if the progress bar is in an undetermined state.

        Returns
        -------
        bool
            If the progress bar has an undetermined state.
        """
        return self.pbar.minimum() == self.pbar.maximum() == 0

    def step(self, step_description: str) -> None:
        """
        Increase progress bar by one step and update its description. If
        progress bar is in an undetermined state, only the description
        is changed.

        Parameters
        ----------
        step_description : str
            New step description.
        """
        self.desc.setText(step_description)
        if not self.undetermined():
            self.pbar.setValue(self.pbar.value() + 1)

    def reset(self) -> None:
        """
        Reset the progress bar.

        """
        self.desc.clear()
        self.pbar.reset()


class PopUpProgBar(QW.QProgressDialog):
    def __init__(
        self,
        parent: QW.QWidget | None,
        n_iter: int,
        label: str = "",
        auto_delete: bool = True,
    ) -> None:
        """
        A customized modal progress bar dialog, that automatically shows
        and hides itself. Ideal for quick, non-threaded operations.

        Parameters
        ----------
        parent : QWidget | None
            The GUI parent of this widget. If not None, the dialog will
            popup centered with it.
        n_iter : int
            Total number of operations to be performed. It determines
            the upper range value of the progress bar.
        label : str, optional
            The text label to show in the dialog. The default is ''.
        auto_delete : bool, optional
            Whether the dialog should be deleted after being hidden. The
            default is True.

        Example
        -------
        To display another dialog over this progress bar dialog while
        keeping it responsive, set this dialog as its parent:

        Construct the progress bar dialog
        >>> parent_widget = PyQt5.QtWidgets.QWidget()
        >>> n_iter = 5
        >>> pbar = PopUpProgBar(parent_widget, n_iter)

        Run a processing iteration
        >>> result = 0
        >>> for i in range(n_iter):
                try:
                    result += 1 / i
                except ZeroDivisionError:
                    # [RELEVANT CODE]: set 'pbar' as the message box's parent
                    MsgBox(pbar, 'Info', f'{i} caused an error. Skipped.')
                finally:
                    pbar.increase()
        """
        # Initialize a custom progress dialog without a 'Cancel' button
        flags = QC.Qt.WindowType(
            QC.Qt.WindowType.Tool
            | QC.Qt.WindowType.WindowTitleHint
            | QC.Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(label, None, 0, n_iter, parent, flags=flags)

        # Set attributes
        self.auto_delete = auto_delete

        # Set widget properties
        self.setWindowModality(QC.Qt.WindowModality.ApplicationModal)
        self.setWindowTitle("Please wait...")
        self.setMinimumDuration(0)
        self.setValue(0)
        if self.auto_delete:
            self.setAttribute(QC.Qt.WidgetAttribute.WA_DeleteOnClose, True)

    def increase(self) -> None:
        """
        Increase the value of the progress bar by one.

        """
        self.setValue(self.value() + 1)

    def reset(self) -> None:
        """
        Reimplementation of the parent method. If 'auto_delete' is True,
        this triggers window closure so 'WA_DeleteOnClose' executes.

        """
        super().reset()
        if self.auto_delete:
            self.close()

    def reject(self) -> None:
        """
        Reimplementation of the parent method, that prevents dialog
        rejection triggers to work (e.g., pressing the ESC key or the
        'X' button).

        """
        return


class PulsePopUpProgBar(PopUpProgBar):
    def __init__(self, parent: QW.QWidget | None, **kwargs) -> None:
        """
        A variant of the 'PopUpProgBar' class, that shows a progress
        dialog with undetermined state. Ideal when dealing with one or
        more operations whose lenght can be extremely variable.

        Parameters
        ----------
        parent : QWidget | None
            The GUI parent of this widget. If not None, the dialog will
            popup centered with it.

        **kwargs
            Parent class arguments (see 'PopUpProgBar' class).
        """
        super().__init__(parent, 1, **kwargs)
        self.setAutoReset(False)

    def startPulse(self) -> None:
        """
        Start pulsing the progress bar.

        """
        self.setValue(1)
        self.setRange(0, 0)

    def stopPulse(self) -> None:
        """
        Stop pulsing and reset the progress bar.

        """
        self.reset()


class DecimalPointSelector(StyledComboBox):
    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        A Styled Combo Box that allows the selection of the preferred
        decimal point character.

        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent=parent)
        self.addItems([".", ","])
        self.setCurrentText(QC.QLocale().decimalPoint())

        # Avoid the combo box being squeezed too much by layouts
        self.setMinimumContentsLength(3)


class SeparatorSymbolSelector(StyledComboBox):
    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        A Styled Combo Box, that allows the selection of the preferred
        CSV separator character.

        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent=parent)
        local_separator = "," if QC.QLocale().decimalPoint() == "." else ";"
        self.addItems([",", ";"])
        self.setCurrentText(local_separator)

        # Avoid the combo box being squeezed too much by layouts
        self.setMinimumContentsLength(3)


class MsgBox(QW.QMessageBox):
    _BUTTONS: ClassVar[dict[str, QW.QMessageBox.StandardButton]] = {
        "Ok": QW.QMessageBox.StandardButton.Ok,
        "Open": QW.QMessageBox.StandardButton.Open,
        "Save": QW.QMessageBox.StandardButton.Save,
        "Cancel": QW.QMessageBox.StandardButton.Cancel,
        "Close": QW.QMessageBox.StandardButton.Close,
        "Discard": QW.QMessageBox.StandardButton.Discard,
        "Apply": QW.QMessageBox.StandardButton.Apply,
        "Reset": QW.QMessageBox.StandardButton.Reset,
        "RestoreDefaults": QW.QMessageBox.StandardButton.RestoreDefaults,
        "Help": QW.QMessageBox.StandardButton.Help,
        "SaveAll": QW.QMessageBox.StandardButton.SaveAll,
        "Yes": QW.QMessageBox.StandardButton.Yes,
        "YesToAll": QW.QMessageBox.StandardButton.YesToAll,
        "No": QW.QMessageBox.StandardButton.No,
        "NoToAll": QW.QMessageBox.StandardButton.NoToAll,
        "Abort": QW.QMessageBox.StandardButton.Abort,
        "Retry": QW.QMessageBox.StandardButton.Retry,
        "Ignore": QW.QMessageBox.StandardButton.Ignore,
    }
    _ICONS: ClassVar[dict[str, QW.QMessageBox.Icon]] = {
        "Question": QW.QMessageBox.Icon.Question,
        "Information": QW.QMessageBox.Icon.Information,
        "Warning": QW.QMessageBox.Icon.Warning,
        "Critical": QW.QMessageBox.Icon.Critical,
    }

    def __init__(
        self,
        parent: QW.QWidget | None,
        kind: Literal["Info", "Quest", "QuestWarn", "Warn", "Crit"],
        text: str = "",
        dtext: str = "",
        title: str | None = None,
        icon: QW.QMessageBox.Icon | None = None,
        btns: QW.QMessageBox.StandardButton | None = None,
        def_btn: QW.QMessageBox.StandardButton | None = None,
        cbox: QW.QCheckBox | None = None,
    ) -> None:
        """
        Convenient class to quickly construct a message box dialog with
        default icon and buttons depending on selected 'kind'. Most of
        its properties can be customized through parameters. It also
        includes methods to check user interactions with buttons and
        checkboxes.

        Parameters
        ----------
        parent : QWidget | None
            The GUI parent of this widget. If not None, the dialog will
            popup centered with it.
        kind : Literal["Info", "Quest", "QuestWarn", "Warn", "Crit"]
            Message box type, determining its default icon and buttons.
        text : str, optional
            Message box text. The default is "".
        dtext : str, optional
            Message box detailed text. If set, a 'Show Details' button
            will also be added. The default is "".
        title : str | None, optional
            Message box title. If None, it will be the 'parent' window
            title or 'X-Min Learn', if invalid. The default is None.
        icon : QMessageBox.Icon | None, optional
            Message box icon. If None, it is set according to 'kind'.
            The default is None.
        btns : QMessageBox.StandardButton | None, optional
            Message box buttons. If None, they are set according to
            'kind'. The default is None.
        def_btn : QMessageBox.StandardButton | None, optional
            Default selected button. If None, it is set according to
            'kind'. The default is None.
        cbox : QCheckBox | None, optional
            If set, the checkbox is added to the message box. The
            default is None.

        Raises
        ------
        ValueError
            Raised if an invalid 'kind' value is set.
        """
        # Set default icon and buttons according to 'kind'
        question_buttons = QW.QMessageBox.StandardButton(
            self._BUTTONS["Yes"] | self._BUTTONS["No"]
        )
        match kind:
            case "Info":
                icon = self._ICONS["Information"] if icon is None else icon
                btns = self._BUTTONS["Ok"] if btns is None else btns
            case "Quest":
                icon = self._ICONS["Question"] if icon is None else icon
                btns = question_buttons if btns is None else btns
                def_btn = self._BUTTONS["No"] if def_btn is None else def_btn
            case "QuestWarn":
                icon = self._ICONS["Warning"] if icon is None else icon
                btns = question_buttons if btns is None else btns
                def_btn = self._BUTTONS["No"] if def_btn is None else def_btn
            case "Warn":
                icon = self._ICONS["Warning"] if icon is None else icon
                btns = self._BUTTONS["Ok"] if btns is None else btns
            case "Crit":
                icon = self._ICONS["Critical"] if icon is None else icon
                btns = self._BUTTONS["Ok"] if btns is None else btns
            case _:
                raise ValueError(f'"{kind}" is an invalid message box kind.')

        # Auto-set title if not specified
        if title is None:
            try:
                if parent is None or (title := parent.windowTitle()) == "":
                    title = "X-Min Learn"
            except AttributeError:
                title = "X-Min Learn"

        # Set dialog attributes
        super().__init__(icon, title, text, btns, parent)
        self.setDefaultButton(def_btn)
        self.setDetailedText(dtext)
        self.setCheckBox(cbox)

        # Set righ-click menu stylesheet
        self.setStyleSheet(style.SS_MENU)

        # Show dialog
        self.exec()

    def _clicked_standard_button(self) -> QW.QMessageBox.StandardButton:
        """
        Helper method to return the clicked button as a StandardButton
        enum.

        Returns
        -------
        QMessageBox.StandardButton
            Clicked button as a StandardButton.
        """
        if (btn := self.clickedButton()) is None:
            return self._BUTTONS["Cancel"]
        return self.standardButton(btn)

    def is_choice(self, button_key: str) -> bool:
        """
        Check if the clicked button matches a specific button.

        Parameters
        ----------
        button_key : str
            String key matching a standard button. Check the
            'self._BUTTONS' class attribute for a list of valid keys.

        Returns
        -------
        bool
            Whether the user choice matches the specified button.
        """
        clicked = self._clicked_standard_button()
        return self._BUTTONS.get(button_key) == clicked

    def cboxChecked(self) -> bool:
        """
        Check the state of the checkbox. If no checkbox was set, this
        method returns False.

        Returns
        -------
        bool
            Whether the checkbox was checked.
        """
        cbox = self.checkBox()
        if cbox is None:
            return False
        else:
            return cbox.isChecked()


class FileDialog(QW.QFileDialog):
    def __init__(
        self,
        parent: QW.QWidget | None,
        action: Literal["open", "O", "save", "S"],
        caption: str = "",
        filters: str | None = "",
        multifile: bool = False,
        set_default_folder: bool = True,
    ) -> None:
        """
        Convenient class for executing open/save file dialogs. It
        automatically sets its home directory to the application's
        current default input or output directory.

        Parameters
        ----------
        parent : QW.QWidget | None
            The GUI parent of this widget. If not None, the dialog will
            popup centered with it.
        action : Literal["open", "O", "save", "S"]
            If "open" (or "O"), an open file dialog is executed. If
            "save" (or "S"), a save dialog is executed.
        caption : str, optional
            The dialog's caption (title). The default is "".
        filters : str | None, optional
            Accepted file types. If "" or None, all types are valid.
            This parameter is ignored if 'action' is "save" (or "S") and
            'multifile' is True. The default is "".
        multifile : bool, optional
            Whether multiple files should be opened or saved. When
            'action' is "save" (or "S") and this parameter is True, the
            dialog allows the selection of an output folder for multiple
            files to be saved to. The default is False.
        set_default_folder : bool, optional
            Whether the application's current default input or output
            folder should be updated. The default is True.

        Raises
        ------
        ValueError
            Raised if 'action' is not a valid string.
        """
        # Set dialog attributes depending on required action
        match action:
            case "open" | "O":
                dir_type = "in"
                accept_mode = QW.QFileDialog.AcceptMode.AcceptOpen
                if multifile:
                    file_mode = QW.QFileDialog.FileMode.ExistingFiles
                else:
                    file_mode = QW.QFileDialog.FileMode.ExistingFile

            case "save" | "S":
                dir_type = "out"
                if multifile:
                    accept_mode = QW.QFileDialog.AcceptMode.AcceptOpen
                    file_mode = QW.QFileDialog.FileMode.Directory
                    filters = None
                else:
                    accept_mode = QW.QFileDialog.AcceptMode.AcceptSave
                    file_mode = QW.QFileDialog.FileMode.AnyFile

            case _:
                raise ValueError(f'Invalid "action" argument: {action}')

        # Fix home directory if not existing
        home_dir = settings.manager.get(f"system/{dir_type}_path")
        if not os.path.exists(home_dir):
            home_dir = settings.DOC_DIR

        # Construct dialog
        super().__init__(parent, caption, home_dir, filters)
        if file_mode == QW.QFileDialog.FileMode.Directory:
            self.setOption(QW.QFileDialog.Option.ShowDirsOnly, True)
            self.setOption(QW.QFileDialog.Option.DontResolveSymlinks, True)

        self.setFileMode(file_mode)
        self.setAcceptMode(accept_mode)
        self.setViewMode(QW.QFileDialog.ViewMode.Detail)

        # If dialog is accepted, update default I/O folder, if requested
        if self.exec() and set_default_folder:
            path = self.selectedFiles()[0]
            folder = path if file_mode == 2 else os.path.dirname(path)
            settings.manager.set(f"system/{dir_type}_path", folder)

    def returns_multipaths(self) -> bool:
        """
        Check if the dialog is set up to return a list of multiple
        filepaths.

        Returns
        -------
        bool
            Whether the dialog returns multiple filepaths.
        """
        return self.fileMode() == QW.QFileDialog.FileMode.ExistingFiles

    def get_path(self) -> str | None:
        """
        Get the single filepath returned by the dialog. This method
        cannot be called if the dialog was initialized with 'multifile'
        = True.

        Returns
        -------
        str | None
            A single user-selected filepath or None, if the dialog was
            canceled.

        Raises
        ------
        RuntimeError
            Raised if a single path is requested from a multi-file
            dialog.
        """
        # Raise error if this is a multi-file dialog
        if self.returns_multipaths():
            raise RuntimeError(
                'Cannot call "get_path()" from a multi-file dialog. '
                'Use "get_paths()" instead.'
            )

        # Ensure that dialog was not canceled and that path is valid
        if self.result() and (paths := self.selectedFiles()):
            return paths[0]  # TODO consider using pathlib
        return None

    def get_paths(self) -> list[str] | None:
        """
        Get the multiple filepaths returned by the dialog. This method
        cannot be called if the dialog was initialized with 'multifile'
        = False.

        Returns
        -------
        str | None
            All user-selected filepaths or None, if the dialog was
            canceled.

        Raises
        ------
        RuntimeError
            Raised if multiple paths are requested from a single-file
            dialog.
        """
        # Raise error if this is a single-file dialog
        if not self.returns_multipaths():
            raise RuntimeError(
                'Cannot call "get_paths()" from a single-file dialog. '
                'Use "get_path()" instead.'
            )

        # Ensure that dialog was not canceled and that path is valid
        if self.result():
            return self.selectedFiles()  # TODO consider using pathlib
        return None


class LineSeparator(QW.QFrame):
    def __init__(
        self,
        orient: Literal["h", "horizontal", "v", "vertical"] = "horizontal",
        lw: int = 2,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        Simple horizontal or vertical line separator.

        Parameters
        ----------
        orient : Literal["h", "horizontal", "v", "vertical"], optional
            Line orientation. Must be "horizontal" ("h") or "vertical"
            ("v"). The default is "horizontal".
        lw : int, optional
            Line width. Can be set to 1, 2 or 3. The default is 2.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.

        Raises
        ------
        ValueError
            Raised if 'orient' is an invalid string.
        """
        super().__init__(parent)

        match orient:
            case "horizontal" | "h":
                self.setFrameShape(QW.QFrame.Shape.HLine)
            case "vertical" | "v":
                self.setFrameShape(QW.QFrame.Shape.VLine)
            case _:
                raise ValueError(f'"{orient}" is not a valid orientation.')

        self.setFrameShadow(QW.QFrame.Shadow.Plain)
        self.setLineWidth(lw)


class CoordinatesFinder(QW.QFrame):
    coordinatesRequested = QC.pyqtSignal(int, int)  # x coord, y coord

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        Convenient widget that allows to find specific X, Y (or column,
        row) coordinates. Especially useful to find a specific pixel in
        an image.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)
        self.setStyleSheet("QFrame {border: 1 solid black;}")

        # X and Y coords input (Line Edit with Integer Validator)
        validator = QG.QIntValidator(0, 10**8)

        self.x_input = QW.QLineEdit()
        self.x_input.setStyleSheet(style.SS_MENU)
        self.x_input.setAlignment(style.A_HCENTER)
        self.x_input.setPlaceholderText("X (C)")
        self.x_input.setValidator(validator)
        self.x_input.setToolTip("X or Column value")
        self.x_input.setMaximumWidth(50)

        self.y_input = QW.QLineEdit()
        self.y_input.setStyleSheet(style.SS_MENU)
        self.y_input.setAlignment(style.A_HCENTER)
        self.y_input.setPlaceholderText("Y (R)")
        self.y_input.setValidator(validator)
        self.y_input.setToolTip("Y or Row value")
        self.y_input.setMaximumWidth(50)

        # Go to pixel button (Styled Button)
        self.go_btn = StyledButton(style.getIcon("BULLSEYE"))
        self.go_btn.setFlat(True)
        self.go_btn.setToolTip("Zoom to pixel")
        self.go_btn.clicked.connect(self.onButtonClicked)

        # Adjust widget layout
        mainLayout = QW.QHBoxLayout()
        mainLayout.setContentsMargins(5, 2, 5, 2)  # l, t, r, b
        mainLayout.addWidget(self.go_btn, alignment=style.A_HCENTER)
        mainLayout.addWidget(self.x_input, alignment=style.A_HCENTER)
        mainLayout.addWidget(self.y_input, alignment=style.A_HCENTER)
        self.setLayout(mainLayout)

    def onButtonClicked(self) -> None:
        """
        Request coordinates if they are valid.

        """
        x, y = None, None
        if self.x_input.hasAcceptableInput():
            x = int(self.x_input.text())
        if self.y_input.hasAcceptableInput():
            y = int(self.y_input.text())

        if x is not None and y is not None:
            self.coordinatesRequested.emit(x, y)

    def setInputCellsMaxWidth(self, width: int) -> None:
        """
        Set the maximum width of the coordinates input cells.

        Parameters
        ----------
        width : int
            Maximum width.
        """
        self.x_input.setMaximumWidth(width)
        self.y_input.setMaximumWidth(width)


class DocumentBrowser(QW.QWidget):
    def __init__(
        self,
        readonly: bool = True,
        tbar: bool = True,
        placeholder: str = "",
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        Custom widget to load, display and browse a text document. It
        includes search, edit and zoom functionalities.

        Parameters
        ----------
        readonly : bool, optional
            Whether document is read-only. The default is True.
        tbar : bool, optional
            Whether a toolbar with search, edit and zoom functionalities
            should be visible. The default is True.
        placeholder : str, optional
            Default placeholder text, which is displayed when browser is
            empty. The default is "".
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set main attributes
        self.readonly = readonly
        self.toolbar_visible = tbar
        self.placeholder_text = placeholder

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Text browser space
        self.browser = QW.QTextEdit()
        self.browser.setVerticalScrollBar(StyledScrollBar(style.O_VERTICAL))
        self.browser.setStyleSheet(style.SS_MENU)
        self.browser.setReadOnly(self.readonly)
        self.browser.setPlaceholderText(self.placeholder_text)

        # [Browser toolbar actions]
        #|Zoom In
        self.zoom_in_action = QW.QAction(
            style.getIcon("ZOOM_IN"), "Zoom in", self
        )
        #|Zoom Out
        self.zoom_out_action = QW.QAction(
            style.getIcon("ZOOM_OUT"), "Zoom out", self
        )
        #|Search Up
        self.search_up_action = QW.QAction(
            style.getIcon("CHEVRON_UP"), "Search up", self
        )
        #|Search Down
        self.search_down_action = QW.QAction(
            style.getIcon("CHEVRON_DOWN"), "Search down", self
        )

        # [Browser toolbar widget-actions]
        #|Search box (Line Edit)
        self.search_box = QW.QLineEdit(self)
        self.search_box.setStyleSheet(style.SS_MENU)
        self.search_box.setPlaceholderText("Search")
        self.search_box.setClearButtonEnabled(True)
        #|Edit document (Check Box)
        self.edit_cbox = QW.QCheckBox("Editing", self)
        self.edit_cbox.setChecked(not self.readonly)
        self.edit_cbox.setVisible(not self.readonly)

        # Browser toolbar
        self.tbar = QW.QToolBar("Browser toolbar")
        self.tbar.setStyleSheet(style.SS_TOOLBAR)
        self.tbar.setVisible(self.toolbar_visible)
        self.tbar.addActions((self.zoom_in_action, self.zoom_out_action))
        self.tbar.addSeparator()
        self.tbar.addActions((self.search_up_action, self.search_down_action))
        self.tbar.addWidget(self.search_box)
        if not self.readonly:
            self.tbar.addSeparator()
            self.tbar.addWidget(self.edit_cbox)

        # Adjust Main Layout
        layout = QW.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        layout.addWidget(self.tbar)
        layout.addWidget(self.browser)
        self.setLayout(layout)

    def _connect_slots(self) -> None:
        """
        Connect signals to slots.

        """
        # Swap between read only and edit mode
        self.edit_cbox.stateChanged.connect(
            lambda state: self.browser.setReadOnly(not state)
        )

        # Search text in the document
        self.search_box.editingFinished.connect(self._findTextDown)
        self.search_up_action.triggered.connect(self._findTextUp)
        self.search_down_action.triggered.connect(self._findTextDown)

        # Change the text font size (zoom in, zoom out)
        self.zoom_in_action.triggered.connect(lambda: self.browser.zoomIn(1))
        self.zoom_out_action.triggered.connect(lambda: self.browser.zoomOut(1))

    def setDoc(self, doc_path: str) -> None:
        """
        Load a new document and display it in the browser.

        Parameters
        ----------
        doc_path : str
            Text document path.
        """
        if os.path.exists(doc_path):
            with open(doc_path, "r") as log:
                doc = QG.QTextDocument(log.read(), self.browser)
                self.browser.setDocument(doc)
        else:
            self.clear()

    def setText(self, text: str) -> None:
        """
        Set a custom text to the browser.

        Parameters
        ----------
        text : str
            Custom text.
        """
        self.browser.setText(text)

    def clear(self) -> None:
        """
        Clear the browser and set the placeholder text.

        """
        self.browser.clear()
        self.browser.setPlaceholderText(self.placeholder_text)

    def setDefaultPlaceHolderText(self, text: str) -> None:
        """
        Set the default placeholder text of the browser. This text is
        displayed when no document is loaded.

        Parameters
        ----------
        text : str
            Placeholder text.
        """
        self.placeholder_text = text

    def _findTextUp(self) -> None:
        """
        Search text up.

        """
        self.browser.setFocus()
        find_flag = QG.QTextDocument.FindFlag.FindBackward
        self.browser.find(self.search_box.text(), find_flag)

    def _findTextDown(self) -> None:
        """
        Search text down.

        """
        self.browser.setFocus()
        self.browser.find(self.search_box.text())


class RandomSeedGenerator(QW.QWidget):
    seedChanged = QC.pyqtSignal(int, int)  # old seed, new seed

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        Convenient widget to set, manage and display random seeds.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Set main attributes
        self._old_seed = None
        self.seed: int = 0  # placeholder, will be set by randomizeSeed

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

        # Set a random seed
        self.randomizeSeed()

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Random seed line edit
        self.seed_input = QW.QLineEdit()
        self.seed_input.setStyleSheet(style.SS_MENU)

        # Custom validator that accepts numbers between 1 and 999999999
        # and empty strings. This is required to control the behaviour
        # of the Line Edit when user leaves the field empty.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8})?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.seed_input.setValidator(validator)

        # Randomize seed button (Styled Button)
        self.rand_btn = StyledButton(style.getIcon("DICE"))

        # Adjust layout
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QW.QLabel("Seed"))
        layout.addWidget(self.seed_input, 1)
        layout.addWidget(self.rand_btn, alignment=style.A_RIGHT)
        self.setLayout(layout)

    def _connect_slots(self) -> None:
        """
        Signals-slots connector.

        """
        # Make sure that the seed is never left empty
        self.seed_input.editingFinished.connect(self.fillEmptySeed)

        # Change seed either manually or through randomization
        self.seed_input.textChanged.connect(self.onSeedChanged)
        self.rand_btn.clicked.connect(self.randomizeSeed)

    def randomizeSeed(self) -> None:
        """
        Randomize seed.

        """
        seed = np.random.default_rng().integers(1, 999999999)
        self.seed_input.setText(str(seed))

    def onSeedChanged(self, new_seed: str) -> None:
        """
        Set a new seed.

        Parameters
        ----------
        new_seed : str
            New seed in string format.
        """
        if new_seed != "":
            self._old_seed = self.seed
            self.seed = int(new_seed)
            self.seedChanged.emit(self._old_seed, self.seed)

    def fillEmptySeed(self) -> None:
        """
        Fill empty seed with a new random seed.

        """
        if not self.seed_input.text():
            self.randomizeSeed()


class PercentLineEdit(QW.QFrame):
    valueEdited = QC.pyqtSignal(int)

    def __init__(
        self,
        base_value: int,
        min_perc: int = 1,
        max_perc: int = 100,
        parent: QW.QWidget | None = None,
    ) -> None:
        """
        Advanced widget that allows altering an integer using either a
        percent value with a spinbox, or by typing the new number in a
        validated Line Edit. A custom set of icons visually indicate if
        the new integer is bigger, smaller or equal to the original one.

        Parameters
        ----------
        base_value : int
            Original integer value.
        min_perc : int, optional
            Minimum allowed percentage value. It cannot be smaller than
            1. The default is 1.
        max_perc : int, optional
            Maximum allowed percentage value. It cannot be bigger than
            100. The default is 100.
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Main attributes
        self._base = base_value
        self._value = base_value
        self._min_perc = max(min_perc, 1)
        self._max_perc = min(max_perc, 100)

        # Widget attributes
        frame_style = QW.QFrame.Shape.StyledPanel | QW.QFrame.Shadow.Plain
        self.setFrameStyle(frame_style)
        self.setLineWidth(2)

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Line Edit for direct integer input, equipped with a regex
        # validator that accepts numbers between 1 and 10**9 as well as
        # empty strings. This allows a finer control over the behaviour
        # of the line edit.
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8}|1000000000)?$")
        validator = QG.QRegularExpressionValidator(regex)
        self.linedit = QW.QLineEdit(str(self._value))
        self.linedit.setValidator(validator)

        # Percentage input (Styled Spin Box)
        self.spinbox = StyledSpinBox(self._min_perc, self._max_perc)
        # self.spinbox.setFixedWidth(100)
        self.spinbox.setSuffix(" %")
        self.spinbox.setValue(100)

        # Visual increase/decrese icon indicator (Label)
        self.iconlbl = QW.QLabel()
        self.setIcon()

        # Reset button (Styled Button)
        self.reset_btn = StyledButton(style.getIcon("REFRESH"))
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
        """
        Signals-slots connector.

        """
        self.linedit.editingFinished.connect(self.onLineditEditingFinished)
        self.linedit.textChanged.connect(self.onLineditChanged)
        self.spinbox.valueChanged.connect(self.onSpinboxChanged)
        self.reset_btn.clicked.connect(self.resetValue)

    def setIcon(self) -> None:
        """
        Change visual increase/decrease icon indicator based on the
        difference between base value and new input value.

        """
        delta = self._value - self._base

        if delta > 0:
            icon = str(style.ICONS.get("CARET_UP_GREEN"))
        elif delta < 0:
            icon = str(style.ICONS.get("CARET_DOWN_RED"))
        else:
            icon = str(style.ICONS.get("CARET_DOUBLE_YELLOW"))

        pixmap = QG.QPixmap(icon).scaled(20, 20, style.ARM_KEEP)
        self.iconlbl.setPixmap(pixmap)

    def onLineditEditingFinished(self) -> None:
        """
        Actions to perform when user finishes editing the line edit.

        """
        text = self.linedit.text()
        # Replace an empty string with original base value
        if not text:
            self.linedit.setText(str(self._base))
        # Send the 'valueEdited' signal otherwise
        else:
            self.valueEdited.emit(int(text))

    def onLineditChanged(self, value: str) -> None:
        """
        Actions to perform when the value of the line edit is changed.

        Parameters
        ----------
        value : str
            Input line edit value
        """
        # Store new value
        self._value = int(value) if value else self._base

        # Update UI
        perc = self.ratio() * 100
        self.adjustSpinboxPrefix(perc)
        self.setIcon()

        # Adjust value of spin box, preventing it from sending signals
        perc = min(round(perc), self._max_perc)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(perc)
        self.spinbox.blockSignals(False)

    def onSpinboxChanged(self, perc: int) -> None:
        """
        Actions to perform when the value of the spin box is changed.

        Parameters
        ----------
        perc : int
            Input spinbox value.
        """
        # Compute new value
        self._value = round(self._base * perc / 100)

        # GUI update
        self.adjustSpinboxPrefix(perc)
        self.setIcon()

        # Adjust value of line edit, preventing it from sending signals
        self.linedit.blockSignals(True)
        self.linedit.setText(str(self._value))
        self.linedit.blockSignals(False)

        # Send valueEdited signal
        self.valueEdited.emit(self._value)

    def adjustSpinboxPrefix(self, perc: float) -> None:
        """
        Set a prefix to spinbox if percentage is not within its value
        range.

        Parameters
        ----------
        perc : float
            Percentage value.
        """
        if perc < self._min_perc:
            self.spinbox.setPrefix("<")
        elif perc > self._max_perc:
            self.spinbox.setPrefix(">")
        else:
            self.spinbox.setPrefix("")

    def value(self) -> int:
        """
        Getter method for value attribute.

        Returns
        -------
        int
            Current value.
        """
        return self._value

    def setValue(self, value: int) -> None:
        """
        Setter method for value attribute.

        Parameters
        ----------
        value : int
            Integer value.
        """
        self.linedit.setText(str(value))

    def resetValue(self) -> None:
        """
        Reset original value.

        """
        self.setValue(self._base)

    def percent(self) -> int:
        """
        Get current percent value. Prefixes are not honored. Use 'ratio'
        method for exact ratio.

        Returns
        -------
        int
            Current percent value.
        """
        return self.spinbox.value()

    def setPercent(self, perc: int) -> None:
        """
        Set current percent value.

        Parameters
        ----------
        perc : int
            Percent value.
        """
        self.spinbox.setValue(perc)

    def ratio(self, round_decimals: int | None = None) -> float | int:
        """
        Get current [value / base value] ratio.

        Parameters
        ----------
        round_decimals : int | None, optional
            Round ratio to this number of decimals. If None, no rounding
            is performed. The default is None.

        Returns
        -------
        float | int
            Current ratio.
        """
        r = self._value / self._base
        if round_decimals is not None:
            r = round(r, round_decimals)
        return r


class StatusFileLoader(QW.QWidget):
    _STATUS_DICT: ClassVar[dict[str, int]] = {
        "Invalid": 0,
        "Valid": 1,
        "Warning": 2,
    }

    def __init__(
        self, filext: str = "", parent: QW.QWidget | None = None
    ) -> None:
        """
        Interactive widget useful to load filepaths and provide visual
        feedback on their status (valid, invalid or with warnings).

        Parameters
        ----------
        filext : str
            File extension filter. An empty string means all files. The
            default is "".
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(parent)

        # Define main attributes
        self.filter = filext
        self.filepath: str | None = None
        self._status: Literal[0, 1, 2] = 0

        # Initialize GUI and connect its signals to slots
        self._init_ui()
        self._connect_slots()

    def _init_ui(self) -> None:
        """
        GUI constructor.

        """
        # Add file (Styled Button)
        self.add_btn = StyledButton(style.getIcon("CIRCLE_ADD_GREEN"))
        self.add_btn.setFlat(True)

        # Remove file (Styled Button)
        self.del_btn = StyledButton(style.getIcon("CIRCLE_DEL_RED"))
        self.del_btn.setFlat(True)
        self.del_btn.setEnabled(False)

        # Status file path (Path Label)
        self.status_path = PathLabel(full_display=False)
        self.status_path.setFrameWidth(2)
        self.status_path.setFrameColor(style.BAD_RED)

        # Adjust layout
        layout = QW.QGridLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.add_btn, 0, 0, style.A_BOTTOM)
        layout.addWidget(self.del_btn, 0, 1, style.A_BOTTOM)
        layout.addWidget(self.status_path, 1, 0, 1, -1)
        self.setLayout(layout)

    def _connect_slots(self) -> None:
        """
        Signals-slots connector.

        """
        self.add_btn.clicked.connect(self.loadFile)
        self.del_btn.clicked.connect(self.removeFile)

    @property
    def status(self) -> str:
        """
        Get the current file status as string.

        Returns
        -------
        str
            Current status.
        """
        return cf.get_dictkey(self._STATUS_DICT, self._status)

    def loadFile(self) -> None:
        """
        Load file from an open file dialog.

        """
        path = FileDialog(self, "open", "Load File", self.filter).get_path()
        if path:
            self.addFile(path)

    def addFile(self, path: str) -> None:
        """
        Add filepath 'path'. This method can be used to add a path
        directly without showing an open file dialog.

        Parameters
        ----------
        path : str
            Filepath to be added.
        """
        # Raise error if filepath is invalid
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        self.filepath = path
        self.status_path.setPath(path)
        self.setStatus("Valid")
        self.add_btn.setEnabled(False)
        self.del_btn.setEnabled(True)

    def removeFile(self) -> None:
        """
        Remove loaded filepath.

        """
        self.filepath = None
        self.status_path.clearPath()
        self.setStatus("Invalid")
        self.add_btn.setEnabled(True)
        self.del_btn.setEnabled(False)

    def setStatus(
        self, status: Literal["Invalid", "Valid", "Warning"]
    ) -> None:
        """
        Set current status of file to 'status'. This visually changes
        the color of the label frame to red ("Invalid"), green ("Valid")
        or yellow ("Warning").

        Parameters
        ----------
        status : Literal["Invalid", "Valid", "Warning"]
            File status.

        Raises
        ------
        ValueError
            Raised if status is an invalid string.
        """
        match status:
            case "Invalid":
                status_id = 0
                self.status_path.setFrameColor(style.BAD_RED)
            case "Valid":
                status_id = 1
                self.status_path.setFrameColor(style.OK_GREEN)
            case "Warning":
                status_id = 2
                self.status_path.setFrameColor(style.WARN_YELLOW)
            case _:
                raise ValueError(f"Invalid status: {status}.")

        self._status = status_id


class DatasetDesigner(StyledTable):
    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        A specialized Styled Table Widget used in the DatasetBuilder tool to
        allow user-friendly creation of ground truth datasets.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this widget. The default is None.
        """
        super().__init__(0, 4, parent=parent)

        # Set selection behaviour
        selection_behaviour = QW.QAbstractItemView.SelectionBehavior.SelectRows
        self.setSelectionBehavior(selection_behaviour)

        # Set horizontal header properties
        self.columns = ["", "Input Maps", "", "Mineral Map"]
        self.setHorizontalHeaderLabels(self.columns)
        self.resizeHorizontalHeader()

    def resizeHorizontalHeader(self) -> None:
        """
        Resize column headers to fill all the available space. Fill-row
        column, separator column and mineral map column will instead
        resize to their content.

        """
        # Stretch all columns
        self.h_header.setSectionResizeMode(style.HRM_STRETCH)
        # Resize to contents the fill-row, the minmap and the separator columns
        cnt = self.columnCount()
        self.h_header.setSectionResizeMode(0, style.HRM_RESIZETOCONTENTS)
        self.h_header.setSectionResizeMode(cnt - 2, style.HRM_RESIZETOCONTENTS)
        self.h_header.setSectionResizeMode(cnt - 1, style.HRM_RESIZETOCONTENTS)

    def setFillRowButton(self, row: int) -> None:
        """
        Initialize the first column of given 'row' to include the
        'fill row' button, which allows user to import an entire row of
        feature data.

        Parameters
        ----------
        row : int
            Index of row.
        """
        btn = StyledButton(style.getIcon("CIRCLE_ADD"))
        btn.setToolTip("Load multiple features at once")
        btn.clicked.connect(self.fillRow)
        self.setCellWidget(row, 0, btn)

    def setRowWidgets(self, row: int) -> None:
        """
        Initialize the cell widgets of given 'row' as StatusFileLoader
        widgets (see 'StatusFileLoader' class for more details).

        Parameters
        ----------
        row : int
            Index of row.
        """
        n_columns = self.columnCount()

        # Initialize every input map column with InputMap file filter
        for col in range(1, n_columns - 2):
            wid = StatusFileLoader("ASCII maps (*.txt *.gz)")
            self.setCellWidget(row, col, wid)

        # Initialize mineral map column with MineralMap file filter
        wid = StatusFileLoader("Mineral maps (*.mmp);;ASCII maps (*.txt *.gz)")
        self.setCellWidget(row, n_columns - 1, wid)

        # Set resize mode of vertical header
        self.v_header.setSectionResizeMode(row, style.HRM_RESIZETOCONTENTS)

    def refresh(self, features: list[str]) -> None:
        """
        Reset and re-initialize the table with new features names.

        Parameters
        ----------
        features : list[str]
            List of features names.
        """
        # Clear all and set new column names
        self.clear()
        self.setColumnCount(len(features) + 3)
        self.columns = [""] + features + ["", "Mineral Map"]
        self.setHorizontalHeaderLabels(self.columns)
        self.resizeHorizontalHeader()

        # Initialize the table
        for row in range(self.rowCount()):
            self.setFillRowButton(row)
            self.setRowWidgets(row)

    def addRow(self) -> None:
        """
        Add a row to the table.

        """
        row = self.rowCount()
        self.insertRow(row)
        self.setFillRowButton(row)
        self.setRowWidgets(row)

    def delRow(self, row: int) -> None:
        """
        Remove 'row' from the table.

        Parameters
        ----------
        row : int
            Row to be removed.
        """
        if 0 <= row < self.rowCount():
            self.removeRow(row)

    def delLastRow(self) -> None:
        """
        Remove last row from the table

        """
        self.delRow(self.rowCount() - 1)

    def fillRow(self) -> None:
        # Should we skip name fitting and have free imports?
        """
        Try to fill an entire row of Input Maps by automatically import
        and identify multiple files based on their name fitting with
        column names.

        """
        # Exit function if row cannot be retrieved from sender button
        if not isinstance(sender := self.sender(), StyledButton):
            return
        row = self.indexAt(sender.pos()).row()

        # Exit function if path is invalid or file dialog is canceled
        ftype = "ASCII maps (*.txt *.gz)"
        paths = FileDialog(
            self, "O", "Load Maps", ftype, multifile=True
        ).get_paths()
        if not paths:
            return

        # Try matching maps with loaded file names
        required_maps = self.columns[1:-2]
        pbar = PopUpProgBar(self, len(paths), "Loading maps")
        fail: list[str] = []
        for n, p in enumerate(paths, start=1):
            try:
                matching_col = cf.match_map(cf.path2filename(p), required_maps)
                # If a filename matches with a column, then add it
                if matching_col is not None:
                    col = self.columns.index(matching_col)
                    if (loader := self.get_file_loader(row, col)) is not None:
                        loader.addFile(p)
            except FileNotFoundError:
                fail.append(p)
            finally:
                pbar.setValue(n)

        # Send error if one or more files could not be read
        if len(fail):
            text = "One or more files have been deleted, removed or renamed."
            MsgBox(self, "Crit", text, "\n".join(fail))

    def get_file_loader(self, row: int, col: int) -> StatusFileLoader | None:
        """
        Return the file loader widget at 'row', 'col'.

        Parameters
        ----------
        row : int
            Cell row.
        col : int
            Cell column.

        Returns
        -------
        StatusFileLoader | None
            The cell widget if its type is 'StatusFileLoader', otherwise
            None.
        """
        if isinstance(wid := self.cellWidget(row, col), StatusFileLoader):
            return wid
        return None

    def getRowData(self, row: int) -> tuple[list[InputMap], MineralMap | None]:
        """
        Extract map data from the given row.

        Parameters
        ----------
        row : int
            Index of row.

        Returns
        -------
        input_maps : list[InputMap]
            List of valid input maps.
        mineral_map : MineralMap | None
            Mineral map or None, if invalid.
        """
        n_columns = self.columnCount()
        input_maps: list[InputMap] = []
        mineral_map: MineralMap | None = None
        pbar = PopUpProgBar(self, n_columns - 1, "Loading data")

        for col in range(1, n_columns):
            # Skip if not containing a file loader widget (= separator)
            if (loader := self.get_file_loader(row, col)) is None:
                pass
            # Invalidate loader if it contains an invalid path (safety)
            elif (path := loader.filepath) is None:
                loader.setStatus("Invalid")
            # Load Input or Mineral Map; invalidate loader if failed
            else:
                try:
                    if col == n_columns - 1:
                        mineral_map = MineralMap.load(path)
                    else:
                        input_maps.append(InputMap.load(path))
                except (OSError, ValueError, TypeError):
                    loader.setStatus("Invalid")

            pbar.setValue(col)

        return (input_maps, mineral_map)


class MapsMatcher(StyledTable):
    mapsChanged = QC.pyqtSignal()
    selectorClicked = QC.pyqtSignal(int)  # index of combo box

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        """
        Interactive widget for matching the input maps (or features)
        required for running custom machine learning models. Its signals
        must be properly connected to enable all its functionalities.

        Parameters
        ----------
        parent : QWidget | None, optional
            The GUI parent of this class. The default is None.
        """
        # Set widget properties
        super().__init__(0, 2, ext_sel=False, parent=parent)
        self.setEditTriggers(QW.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QW.QAbstractItemView.SelectionMode.NoSelection)

        # Set horizontal header properties
        self.h_header.setSectionResizeMode(style.HRM_RESIZETOCONTENTS)
        self.h_header.setStretchLastSection(True)
        self.h_header.setHidden(True)

    def setMaps(self, maps_names: list[str]) -> None:
        """
        Populate the table with the required input maps names. The left
        column displays each map's name. The right column is populated
        with Auto-Update ComboBoxes that allow the interactive selection
        of matching input maps.

        Parameters
        ----------
        maps_names : list[str]
            List of required maps names.
        """
        self.clear()
        self.setRowCount(len(maps_names))
        for row, name in enumerate(maps_names):
            i0 = QW.QTableWidgetItem(name)  # map name
            i1 = AutoUpdateComboBox()  # interactive selector
            i1.clicked.connect(lambda idx=row: self.selectorClicked.emit(idx))
            self.setItem(row, 0, i0)
            self.setCellWidget(row, 1, i1)
        self.mapsChanged.emit()

    def updateSelector(self, row: int, items: list[str]) -> None:
        """
        Populate the selector at 'row' with 'items'.

        Parameters
        ----------
        row : int
            Row index of the selector.
        items : list[str]
            List of maps names available for matching.
        """
        self.getSelector(row).updateItems(items)

    def autoFillSelectors(self, items: list[str]):
        """
        Attempt to automatically fill selectors based on matching maps
        names. Each individual selector is also updated. This method
        should be called every time the available Input Maps change.

        Parameters
        ----------
        items : list[str]
            List of maps names available for matching.
        """
        for row in range(self.rowCount()):
            # Update each selector with 'items'
            selector = self.getSelector(row)
            selector.updateItems(items)

            # Set index of selector. If -1, an empty text is displayed.
            map_name = self.getMapName(row)
            matching = cf.match_map(map_name, items)
            index = -1 if matching is None else items.index(matching)
            selector.setCurrentIndex(index)

    def getMapName(self, row: int) -> str:
        """
        Get map name at 'row'.

        Parameters
        ----------
        row : int
            Row index of the map.

        Returns
        -------
        map_name : str
            Name of the map at 'row'.

        Raises
        ------
        ValueError
            Raised if 'row' is an invalid row index.
        """
        if (item := self.item(row, 0)) is None:
            raise ValueError(f"Invalid row index: {row}")
        map_name = item.text()
        return map_name

    def getSelector(self, row: int) -> AutoUpdateComboBox:
        """
        Get the selector at 'row'.

        Parameters
        ----------
        row : int
            Row index of the selector.

        Returns
        -------
        selector : AutoUpdateComboBox
            Selector at 'row'.

        Raises
        ------
        ValueError
            Raised if 'row' is an invalid row index.
        """
        if (selector := self.cellWidget(row, 1)) is None:
            raise ValueError(f"Invalid row index: {row}")
        return selector

    def getMatchedMaps(self) -> list[str]:
        """
        Get a list of all matched maps names, i.e., maps chosen within
        each selector. Empty selectors return empty strings.

        Returns
        -------
        list[str]
            List of selected matching maps.
        """
        rows = range(self.rowCount())
        return [self.getSelector(r).currentText() for r in rows]

    def getMatchedIndices(self) -> list[int]:
        """
        Get a list of all matched maps' indices, i.e., the indices of
        selected maps in reference to each selector's list of available
        maps. Empty selectors return index -1.

        Returns
        -------
        list[int]
            List of selected matching maps' indices.
        """
        rows = range(self.rowCount())
        return [self.getSelector(r).currentIndex() for r in rows]
