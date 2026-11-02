# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:16:01 2021

@author: albdag
"""

from collections.abc import Sequence
import os
from datetime import datetime
from typing import Literal, cast, get_args, overload
import webbrowser

import numpy as np

from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW

from base import *
import convenient_functions as cf
import custom_widgets as CW
import dataset_tools as dtools
import image_analysis_tools as iatools
import machine_learning_tools as mltools
import plots
import settings
import style
import threads


class DraggableTool(QW.QWidget):

    closed = QC.pyqtSignal()
    tabified = QC.pyqtSignal()

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        The base class for all major tools of X-Min Learn. It implements a drag
        & drop mechanic that allows the tool to be docked in the main window or
        to be displayed as a floating window. See MainTabWidget class for more
        details.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this widget. The default is None.

        '''
        super().__init__(parent)
        self.floating_size = QC.QSize(1600, 900)
        self.resize(self.floating_size)
        self.setAttribute(QC.Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setAttribute(QC.Qt.WidgetAttribute.WA_DeleteOnClose, True)


    def isFloating(self) -> bool:
        '''
        Check if the tool is floating (i.e., is not tabbed).

        Returns
        -------
        bool
            Whether the tool is floating.

        '''
        return self.parent() == None


    def mouseMoveEvent(self, a0: QG.QMouseEvent | None) -> None:
        '''
        Reimplementation of the 'mouseMoveEvent' method. It provides a custom
        animation for the widget drag & drop action (left-click drag).

        Parameters
        ----------
        a0 : QMouseEvent | None
            The mouse move event triggered by the user.

        '''
    # Ignore the event if the tool is tabified
        if not self.isFloating():
            return
        
        if a0 is not None and a0.buttons() == QC.Qt.MouseButton.LeftButton:
            # Generate mime data and a pixmap for the drag & drop event
                icon = self.windowIcon()
                pixmap = icon.pixmap(icon.actualSize(QC.QSize(32, 32)))
                mimeData = QC.QMimeData()

            # Generate a drag event and execute it
                drag = QG.QDrag(self)
                drag.setMimeData(mimeData)
                drag.setPixmap(pixmap)
                drag.exec_(QC.Qt.DropAction.MoveAction)

    # deprecated
    # def killReferences(self) -> None:
    #     '''
    #     Remove all the references that keep the tool alive. This method must be
    #     called to properly close the tool.

    #     '''
    #     for child in self.findChildren(QW.QWidget):
    #         child.disconnect()


    def is_safe_to_close(self) -> bool:
        '''
        Ask user confirmation to close the tool.

        Returns
        -------
        bool
            Whether the user confirms to close the tool or not.

        '''
        text = f'Close {self.windowTitle()}? Any unsaved data will be lost.'
        msgbox = CW.MsgBox(self, 'Quest', text)
        return msgbox.is_choice('Yes')


    def force_close(self) -> None:
        '''
        Close the tool without asking for user confirmation. This method is 
        meant only for programmatically close a tool.

        '''
        # Close after resetting the close event to default (skip user confirm)
        self.closeEvent = super().closeEvent
        self.close()


    def closeEvent(self, a0: QG.QCloseEvent | None) -> None:
        '''
        Reimplementation of the default 'closeEvent' method. It adds a question
        dialog to confirm the closing action. It sends the 'closed' signal when
        the close event is accepted.

        Parameters
        ----------
        a0 : QG.QCloseEvent | None
            The close event.

        '''
        if a0 is None:
            return

        if self.is_safe_to_close():
            a0.accept()
            # Defer signal emission to next event loop pass
            QC.QTimer.singleShot(0, self.closed.emit)
        else:
            a0.ignore()



class MineralClassifier(DraggableTool):

    def __init__(self) -> None:
        '''
        One of the main tools of X-Min Learn, that allows the classification of
        input maps using pre-trained eager ML models, ROI-based lazy algorithms
        and/or clustering algorithms. It also enables sub-phase identification.

        '''
        super().__init__()

    # Set tool title and icon
        self.setWindowTitle('Mineral Classifier')
        self.setWindowIcon(style.getIcon('MINERAL_CLASSIFIER'))

    # Initialize main attributes
        self._mask: Mask | None = None
        self._nodata_color: tuple[int, int, int] = (0, 0, 0)

    # Initialize classification state and attributes
        self._isBusyClassifying: bool = False
        self._current_classifier: mltools._ClassifierBase | None = None

    # Initialize GUI and connect its signals with slots 
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                                DATA PANEL 
#  -------------------------------------------------------------------------  #
        
    # Input maps selector (Sample Maps Selector) 
        self.inmaps_selector = CW.SampleMapsSelector('inmaps')

    # Remove mask button (Styled Button)
        self.del_mask_btn = CW.StyledButton(style.getIcon('CLEAR'))
        self.del_mask_btn.setToolTip('Remove current mask')

    # Loaded mask (Path Label)
        self.mask_pathlbl = CW.PathLabel(full_display=False)

    # Mask warning icon (Label)
        warn_icon = QG.QPixmap(str(style.ICONS.get('WARNING')))
        self.mask_warn = QW.QLabel()
        self.mask_warn.setPixmap(warn_icon.scaled(24, 24, style.ARM_KEEP))
        self.mask_warn.setToolTip('Invalid shape detected. Mask not applied.')
        self.mask_warn.setVisible(False)

    # Load mask from file (Radio Button) [Default choice]
        self.mask_radbtn_1 = QW.QRadioButton('From mask file')
        self.mask_radbtn_1.setStyleSheet(style.SS_RADIOBUTTON)
        self.mask_radbtn_1.setChecked(True)

    # Mask selector (Auto Update Combo Box)
        self.mask_combox = CW.AutoUpdateComboBox()
    
    # Load mask button (Styled Button)
        self.load_mask_btn = CW.StyledButton(style.getIcon('IMPORT'))
        self.load_mask_btn.setToolTip('Load from file...')
        
    # Get mask from classificaton result (Radio Button)
        self.mask_radbtn_2 = QW.QRadioButton('From classification result')
        self.mask_radbtn_2.setStyleSheet(style.SS_RADIOBUTTON)

    # Classification result selector (Auto Update Combo Box)
        self.minmap_combox = CW.AutoUpdateComboBox()
        self.minmap_combox.setEnabled(False)

    # Mineral Class selector (Combo Box)
        self.class_combox = CW.StyledComboBox()
        self.class_combox.setEnabled(False)

    # Mineral maps list (Tree Widget)
        self.minmaps_list = QW.QTreeWidget()
        self.minmaps_list.setHeaderHidden(True)
        no_edit_triggers = QW.QAbstractItemView.EditTrigger.NoEditTriggers
        self.minmaps_list.setEditTriggers(no_edit_triggers)
        if (vscrollbar := self.minmaps_list.verticalScrollBar()) is not None:
            vscrollbar.setStyleSheet(style.SS_SCROLLBARV)
        self.minmaps_list.setStyleSheet(style.SS_MENU)
        
    #[Mineral maps toolbar actions]
    #|Save mineral map
        self.save_minmap_action = QW.QAction(
            style.getIcon('SAVE_AS'), 'Save mineral map as...', self)
    #|Delete mineral map
        self.del_minmap_action = QW.QAction(
            style.getIcon('REMOVE'), 'Remove mineral map', self)
        
    # Mineral maps toolbar (Toolbar)
        self.minmaps_tbar = QW.QToolBar('Mineral maps toolbar')
        self.minmaps_tbar.setOrientation(style.O_VERTICAL)
        self.minmaps_tbar.setStyleSheet(style.SS_TOOLBAR)
        self.minmaps_tbar.addAction(self.save_minmap_action)
        self.minmaps_tbar.addAction(self.del_minmap_action)

    # Mineral maps legend (Legend)
        self.legend = CW.Legend()
        selection_mode = QW.QAbstractItemView.SelectionMode.SingleSelection
        self.legend.setSelectionMode(selection_mode)

    # Mineral maps bar canvas (Bar Canvas)
        self.barplot = plots.BarCanvas(
            orientation='h', size=(3, 5.3), wheel_zoom=False, wheel_pan=False)
        self.barplot.fig.set_linewidth(0)
    
    # Mineral maps bar canvas navigation toolbar (Navigation Toolbar)
        self.barplot_navtbar = plots.NavTbar.barCanvasDefault(
            self.barplot, self)

    # Silhouette canvas (Silhouette Canvas)
        self.silscore_canvas = plots.SilhouetteCanvas(
            size=(3, 5.3), wheel_zoom=False, wheel_pan=False)
        self.silscore_canvas.fig.set_linewidth(0)
        self.silscore_canvas.setMinimumHeight(400)

    # Silhouette canvas navigation toolbar (Navigation Toolbar)
        self.silscore_navtbar = plots.NavTbar(
            self.silscore_canvas, self, coords=False)
        self.silscore_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Silhouette average score (Label)
        self.silscore_lbl = QW.QLabel('-')
        
    # Calinski-Harabasz Index (CHI) score (Label)
        self.chiscore_lbl = QW.QLabel('-')

    # Davies-Bouldin Index (DBI) score (Label)
        self.dbiscore_lbl = QW.QLabel('-')

#  -------------------------------------------------------------------------  #
#                              CLASSIFIER PANEL 
#  -------------------------------------------------------------------------  #

    # Classifier panel (Styled Tab Widget)
        self.classifier_tabwid = CW.StyledTabWidget()
        self.classifier_tabwid.setTabBarExpanding()
        self.pre_train_tab = self.PreTrainedClassifierTab()
        self.roi_based_tab = self.RoiBasedClassifierTab()
        self.unsuperv_tab = self.UnsupervisedClassifierTab()
        self.classifier_tabwid.wrap_tab(self.pre_train_tab, title='PRE-TRAINED')
        self.classifier_tabwid.wrap_tab(self.roi_based_tab, title='ROI-BASED')
        self.classifier_tabwid.wrap_tab(self.unsuperv_tab, title='UNSUPERVISED')

    # Classification progress bar (Descriptive Progress Bar)
        self.progbar = CW.DescriptiveProgressBar()

    # Classify button (Styled Button)
        self.classify_btn = CW.StyledButton(text='CLASSIFY', bg=style.OK_GREEN)
        self.classify_btn.setEnabled(False)

    # Interrupt classification process button (StyledButton)
        self.stop_btn = CW.StyledButton(text='STOP', bg=style.BAD_RED)

#  -------------------------------------------------------------------------  #
#                                VIEWER PANEL 
#  -------------------------------------------------------------------------  #

    # Maps viewer (Image Canvas)
        self.maps_viewer = plots.ImageCanvas()
        self.maps_viewer.setMinimumHeight(100)

    # Viewer Navigation Toolbar (Navigation Toolbar)
        self.viewer_navtbar = plots.NavTbar.imageCanvasDefault(
            self.maps_viewer, self)

    # Confidence value input (Spin Box)
        self.conf_spbox = QW.QSpinBox()
        self.conf_spbox.setToolTip('Confidence')
        self.conf_spbox.setRange(0, 100)
        self.conf_spbox.setSingleStep(1)
        self.conf_spbox.setValue(50)
        self.conf_spbox.setEnabled(False)

    # Confidence Label (Label)
        conf_lbl = QW.QLabel('Confidence threshold')

    # Confidence slider (Slider)
        self.conf_slider = QW.QSlider(style.O_HORIZONTAL)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setTracking(False)
        self.conf_slider.setSliderPosition(50)
        self.conf_slider.setEnabled(False)

#  -------------------------------------------------------------------------  #
#                                 LAYOUT 
#  -------------------------------------------------------------------------  #
        
    # Data panel (Styled Tab Widget)
        
        # - Input data tab (Splitter Group) 
        mask_grid = QW.QGridLayout()
        mask_grid.setVerticalSpacing(15)
        mask_grid.addWidget(self.mask_warn, 0, 0)
        mask_grid.addWidget(self.mask_pathlbl, 0, 1, 1, 2)
        mask_grid.addWidget(self.del_mask_btn, 0, 3)
        mask_grid.addWidget(self.mask_radbtn_1, 1, 0, 1, -1)
        mask_grid.addWidget(self.mask_combox, 2, 0, 1, 3)
        mask_grid.addWidget(self.load_mask_btn, 2, 3)
        mask_grid.addWidget(self.mask_radbtn_2, 3, 0, 1, -1)
        mask_grid.addWidget(self.minmap_combox, 4, 0, 1, 2)
        mask_grid.addWidget(QW.QLabel('Output map'), 4, 2, 1, 2)
        mask_grid.addWidget(self.class_combox, 5, 0, 1, 2)
        mask_grid.addWidget(QW.QLabel('Class'), 5, 2, 1, 2)
        mask_grid.setColumnStretch(1, 1)
        self.mask_group = CW.GroupArea(mask_grid, 'Mask', checkable=True)
        self.mask_group.setChecked(False)

        input_data_split = CW.SplitterGroup(
            (self.inmaps_selector, self.mask_group), (1, 0), style.O_VERTICAL)

        # - Output data tab (Splitter Group)
        barplot_vbox = QW.QVBoxLayout()
        barplot_vbox.addWidget(self.barplot_navtbar)
        barplot_vbox.addWidget(self.barplot)

        scores_grid = QW.QGridLayout()
        scores_grid.setRowStretch(1, 1)
        scores_grid.addWidget(self.silscore_navtbar, 0, 0, 1, -1)
        scores_grid.addWidget(self.silscore_canvas, 1, 0, 1, -1)
        scores_grid.addWidget(QW.QLabel('Average silhouette score'), 2, 0)
        scores_grid.addWidget(self.silscore_lbl, 2, 1, style.A_RIGHT)
        scores_grid.addWidget(QW.QLabel('Calinski-Harabasz Index'), 3, 0)
        scores_grid.addWidget(self.chiscore_lbl, 3, 1, style.A_RIGHT)
        scores_grid.addWidget(QW.QLabel('Davies-Bouldin Index'), 4, 0)
        scores_grid.addWidget(self.dbiscore_lbl, 4, 1, style.A_RIGHT)

        graph_tabwid = CW.StyledTabWidget()
        legend_idx = graph_tabwid.wrap_tab(
            self.legend, style.getIcon('LEGEND'), title=None, tight=True)
        plot_idx = graph_tabwid.wrap_tab(
            barplot_vbox, style.getIcon('PLOT'), title=None, tight=True)
        scores_idx = graph_tabwid.wrap_tab(
            scores_grid, style.getIcon('SCORES'), title=None, tight=True)
        graph_tabwid.setTabToolTip(legend_idx, 'Legend')
        graph_tabwid.setTabToolTip(plot_idx, 'Bar plot')
        graph_tabwid.setTabToolTip(scores_idx, 'Clustering scores')

        minmaps_list_hbox = QW.QHBoxLayout()
        minmaps_list_hbox.setContentsMargins(0, 0, 0, 0)
        minmaps_list_hbox.addWidget(self.minmaps_tbar)
        minmaps_list_hbox.addWidget(self.minmaps_list)

        output_data_split = CW.SplitterGroup(
            (minmaps_list_hbox, graph_tabwid), (1, 0), style.O_VERTICAL)
        
        # - Tab Widget
        self.data_tabwid = CW.StyledTabWidget()
        self.data_tabwid.setTabBarExpanding()
        self.data_tabwid.wrap_tab(
            input_data_split, style.getIcon('STACK'), title='INPUT MAPS')
        self.data_tabwid.wrap_tab(
            output_data_split, style.getIcon('MINERAL'), title='OUTPUT MAPS')

    # Classifier settings panel (Collapsible Area)
        class_group = CW.CollapsibleArea(
            qobject=self.classifier_tabwid,
            title='Classifier settings',
            frame=False,
            collapsed=False
        )
        
    # Main layout
        left_grid = QW.QGridLayout()
        left_grid.addWidget(self.data_tabwid, 0, 0, 1, -1)
        left_grid.addWidget(self.progbar, 1, 0, 1, -1)
        left_grid.addWidget(self.classify_btn, 2, 0)
        left_grid.addWidget(self.stop_btn, 2, 1)
        left_scroll = CW.GroupScrollArea(left_grid)

        right_grid = QW.QGridLayout()
        right_grid.setRowStretch(2, 1)
        right_grid.addWidget(class_group, 0, 0, 1, -1)
        right_grid.addWidget(self.viewer_navtbar, 1, 0, 1, -1)
        right_grid.addWidget(self.maps_viewer, 2, 0, 1, -1)
        right_grid.addWidget(conf_lbl, 3, 0, 1, -1)
        right_grid.addWidget(self.conf_spbox, 4, 0)
        right_grid.addWidget(self.conf_slider, 4, 1)
        right_scroll = CW.GroupScrollArea(right_grid)

        main_layout = CW.SplitterLayout()
        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(right_scroll, 1)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Actions to be performed when input maps are updated
        self.inmaps_selector.mapsListUpdated.connect(self.onInmapsDataUpdated)

    # Display input maps when clicked from the input selector
        self.inmaps_selector.mapClicked.connect(self.showInputMap)
    
    # Actions related to maps matcher widget in PreTrainedClassifierTab
        self.inmaps_selector.mapChanged.connect(self.autoFillMatcher)
        self.pre_train_tab.matcher.mapsChanged.connect(self.autoFillMatcher)
        self.pre_train_tab.matcher.selectorClicked.connect(
            self.updateMatcherSelector)
        
    # Hide/show masks on input maps when mask group is enabled/disabled
        self.mask_group.clicked.connect(self.refreshInputMap)

    # Mask combobox actions
        self.mask_combox.clicked.connect(self.updateMaskCombox)
        self.mask_combox.activated.connect(self.getMaskFromSample)

    # Enable/disable mask radio buttons actions. We only check the toggle state
    # of one of them, since they are mutually exclusive.
        self.mask_radbtn_1.toggled.connect(self.onMaskRadioButtonToggled)

    # Load/remove masks actions
        self.del_mask_btn.clicked.connect(self.removeMask)
        self.load_mask_btn.clicked.connect(self.loadMaskFromFile)

    # Actions to select a mask from mineral phase (comboboxes)
        self.minmap_combox.clicked.connect(self.updateMineralMapsCombox)
        self.minmap_combox.activated.connect(self.updateClassesCombox)
        self.class_combox.textActivated.connect(self.getMaskFromClass)

    # Add/remove ROI maps in canvas when requested by the RoiBasedClassifierTab
        self.roi_based_tab.addRoiMapRequested.connect(self.addRoiMap)
        self.roi_based_tab.removeRoiMapRequested.connect(self.removeRoiMap)

    # Actions to be performed when a mineral map is clicked
        self.minmaps_list.itemClicked.connect(self.showMineralMap)

    # Save and remove mineral maps actions
        self.save_minmap_action.triggered.connect(self.saveMineralMap)
        self.del_minmap_action.triggered.connect(self.removeMineralMap)

    # Connect legend signals (change item color, rename item, highlight item)
        self.legend.colorChangeRequested.connect(self.changeClassColor)
        self.legend.itemRenameRequested.connect(self.renameClass)
        self.legend.itemHighlightRequested.connect(self.highlightClass)

    # Run mineral classification when classify button is clicked
        self.classify_btn.clicked.connect(self.startClassification)

    # Interrupt mineral classification when stop button is clicked
        self.stop_btn.clicked.connect(self.stopClassification)

    # Show custom context menu when right-clicking on the maps canvas
        self.maps_viewer.customContextMenuRequested.connect(
            self.showMapsViewerContextMenu)
        
    # Change probability threshold with spinbox and scaler
        self.conf_spbox.valueChanged.connect(self.setConfidenceThreshold)
        self.conf_slider.valueChanged.connect(self.setConfidenceThreshold)


    def onInmapsDataUpdated(self) -> None:
        '''
        Actions to be performed when Input Maps are updated in the selector.

        '''
        self.maps_viewer.clear_canvas()
        self.updateClassifyButtonState()
        self.autoFillMatcher()


    def autoFillMatcher(self) -> None:
        '''
        Attempt to automatically match the input features of matcher widget.

        '''
        checked_maps = self.inmaps_selector.getCheckedMaps()
        maps_names = [i.getComponents()['name'] for i in checked_maps]
        self.pre_train_tab.matcher.autoFillSelectors(maps_names)


    def updateMatcherSelector(self, index: int) -> None:
        '''
        Update the items of matcher widget's selector at 'index' with currently
        checked input maps. 

        Parameters
        ----------
        index : int
            Index of selector to be updated.

        '''
        checked_maps = self.inmaps_selector.getCheckedMaps()
        maps_names = [i.getComponents()['name'] for i in checked_maps]
        self.pre_train_tab.matcher.updateSelector(index, maps_names)

    
    def onMaskRadioButtonToggled(self, toggled: bool) -> None:
        '''
        Manage the GUI visualization of the input mask options. When one option
        is toggled, the other is disabled. This method is called just by one of
        the two radio buttons (arbitrarily the first one - see '_connect_slots'
        method), since they are mutually exclusive.

        Parameters
        ----------
        toggled : bool
            State of radio button.

        '''
    # Assuming checking against the toggle state of 'self.mask_radbtn_1'
        self.mask_combox.setEnabled(toggled)
        self.load_mask_btn.setEnabled(toggled)
        self.minmap_combox.setEnabled(not toggled)
        self.class_combox.setEnabled(not toggled)
        

    def updateMaskCombox(self) -> None:
        '''
        Populate the combobox that allows the selection of a mask from the
        currently loaded sample.

        '''
    # Clear the combobox and exit function if no sample is selected
        sample = self.inmaps_selector.currentSample()
        if sample is None:
            self.mask_combox.clear()
            return
    
    # Update the combobox with the names of the masks in the selected sample
        items = [c.getComponents()['name'] for c in sample.masks.getChildren()]
        self.mask_combox.updateItems(items)
        

    def updateMineralMapsCombox(self) -> None:
        '''
        Populate the combobox that allows the seletion of a mineral map for 
        extracting a mask.

        '''
        count = self.minmaps_list.topLevelItemCount()
        items = [self.minmaps_list.topLevelItem(idx) for idx in range(count)]
        self.minmap_combox.updateItems([i.text(0) for i in items if i is not None])


    def updateClassesCombox(self, idx: int) -> None:
        '''
        Populate the combobox that allows the selection of a mineral phase to 
        use as input mask.

        Parameters
        ----------
        idx : int
            The choosen mineral map index.

        '''
    # Do nothing if mineral map data object is invalid
        if (mmap_obj := self.getMineralMap(idx)) is None:
            return

    # Do nothing if mineral map data object has invalid data
        if (minmap := mmap_obj.getMineralMapData()) is None:
            return
         
        self.class_combox.clear()
        self.class_combox.addItems(minmap.get_phases())


    def updateClassifyButtonState(self) -> None:
        '''
        Toggle on/off the CLASSIFY button.

        '''
        inmaps_count = self.inmaps_selector.mapsCount()
        self.classify_btn.setEnabled(inmaps_count > 0)


    def showMapsViewerContextMenu(self, point: QC.QPoint) -> None:
        '''
        Shows a context menu with custom actions in the map viewer.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        '''
    # Get context menu from NavTbar actions
        menu = self.maps_viewer.get_navigation_context_menu(
            self.viewer_navtbar)
    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.maps_viewer.mapToGlobal(point))


    def showInputMap(self, item: CW.DataObject | None) -> None:
        '''
        Display input map item data in the maps viewer. If a mask is loaded, 
        the input map is masked accordingly.

        Parameters
        ----------
        item : DataObject or None
            The data object that holds current input map data. If None the maps
            viewer is cleared.

        '''
    # Clear maps viewer and exit function if no item is selected
        if item is None:
            self.maps_viewer.clear_canvas()
            return

    # Exit function if item has invalid input map data
        components = item.getComponents()
        if not isinstance(inmap := components['data'], InputMap):
            return 

    # Check whether data should be masked or not
        if self.mask_group.isChecked() and self._mask is not None:
            mask_available = True
            array = inmap.get_masked(self._mask.mask)
        else:
            mask_available = False
            array = inmap.map
            
    # "get_masked" returns an unmasked array if mask shape is invalid. If that
    # happens, show the mask warning icon.
        visible = mask_available and not np.ma.is_masked(array)
        self.mask_warn.setVisible(visible)

    # Disable confidence spinbox and slider
        self.conf_spbox.setEnabled(False)
        self.conf_slider.setEnabled(False)

    # Plot the map
        map_name = components['name']
        sample_name = self.inmaps_selector.sample_combox.currentText()
        title = f'{sample_name} - {map_name}'
        self.maps_viewer.draw_heatmap(array, title)

    
    def refreshInputMap(self) -> None:
        '''
        Re-render the currently selected input map in the maps viewer. This is
        useful when a mask is loaded/removed or shown/hidden.

        '''
    # Re-render input map only if already displayed in the maps viewer
        if self.maps_viewer.has_heatmap():
            self.showInputMap(self.inmaps_selector.currentMap())


    def loadMaskFromFile(self) -> None:
        '''
        Load an input mask from file.

        '''
    # Do nothing if path is invalid or file dialog is canceled
        ftypes = 'Mask (*.msk);;Text file (*.txt)'
        path = CW.FileDialog(self, 'open', 'Load Mask', ftypes).get_path()
        if not path:
            return
    
    # Try loading mask
        try:
            mask = Mask.load(path)     
        except Exception as e:
            CW.MsgBox(self, 'Crit', f'Unexpected file:\n{path}.', str(e))
            return

    # Update GUI
        self.mask_pathlbl.setPath(path)
        self.mask_combox.clear()
        self.set_mask(mask)


    def getMaskFromSample(self, mask_idx: int) -> None:
        '''
        Extract from the current sample the mask at index 'mask_idx'.

        Parameters
        ----------
        mask_idx : int
            Index of the mask.

        '''
    # Exit function if no valid sample is selected (safety)
        if (curr_sample := self.inmaps_selector.currentSample()) is None:
            return
        
    # Exit function if an invalid index is provided
        sample_masks = curr_sample.masks.getChildren()
        if mask_idx < 0 or mask_idx > len(sample_masks) - 1:
            return
    
    # Extract mask
        components = sample_masks[mask_idx].getComponents()
        mask_data, mask_path = components['data'], components['filepath']
        if isinstance(mask_data, Mask):
            self.mask_pathlbl.setPath(mask_path)
            self.set_mask(mask_data)


    def getMaskFromClass(self, class_name: str) -> None:
        '''
        Extract a mask from the choosen mineral class of the choosen mineral 
        map.

        Parameters
        ----------
        class_name : str
            Name of the choosen class.

        '''
    # Attempt to match mask combobox item name with available mineral map names 
        minmap_name = self.minmap_combox.currentText()
        match_flag = QC.Qt.MatchFlag.MatchExactly
        items = self.minmaps_list.findItems(minmap_name, match_flag, 0) # TODO: not ideal, find something more solid
        if len(items) and isinstance(mmap_obj := items[0], CW.DataObject):
            if (mmap := mmap_obj.getMineralMapData()) is not None:
                mask = Mask(mmap.minmap != class_name)
                self.mask_pathlbl.setPath(f'{minmap_name}/{class_name}')
                self.set_mask(mask)
                return

    # Clear mask if no valid matching mineral map is found 
        self.minmap_combox.clear()
        self.class_combox.clear()
        self.removeMask()
        CW.MsgBox(self, 'Crit', 'This mineral map is no longer available.')


    def set_mask(self, mask: Mask) -> None:
        '''
        Set a mask for classification. Masks are only rendered on top of Input 
        Maps and not on top of Mineral Maps.

        Parameters
        ----------
        mask : Mask
            The mask object.

        '''
        self._mask = mask
        self.refreshInputMap()


    def removeMask(self) -> None:
        '''
        Remove classification mask.

        '''
        self._mask = None
        self.mask_pathlbl.clearPath()
        self.refreshInputMap()


    def addRoiMap(self, roimap: RoiMap) -> None:
        '''
        Render ROI map's patches and annotations in the maps canvas.

        Parameters
        ----------
        roimap : RoiMap
            The ROI map to be rendered.

        '''
        color = iatools.hex2rgb(settings.manager.get('plots/roi_color'))
        filled = settings.manager.get('plots/roi_filled')

        for name, bbox in roimap.roilist:
            patch = plots.RoiPatch(bbox, color, filled)
            text = plots.RoiAnnotation(name, patch)
            self.maps_viewer.ax.add_patch(patch)
            self.maps_viewer.ax.add_artist(text)

        self.maps_viewer.draw_idle()


    def removeRoiMap(self) -> None:
        '''
        Remove all ROI map's patches and annotations from the maps canvas.

        '''
        for child in self.maps_viewer.ax.get_children():
            if isinstance(child, (plots.RoiPatch, plots.RoiAnnotation)):
                child.remove()

        self.maps_viewer.draw_idle()


    def showMineralMap(self, item: CW.DataObject | None) -> None:
        '''
        Display mineral map item data in the maps viewer and update the legend,
        the bar plot and the clustering scores.

        Parameters
        ----------
        item : DataObject or None
            The data object that holds current input map data.

        '''
    # Clear all and exit function if no item is selected
        if item is None:
            self.legend.clear()
            self.barplot.clear_canvas()
            self.maps_viewer.clear_canvas()
            self.silscore_canvas.clear_canvas()
            self.silscore_lbl.setText('-')
            self.chiscore_lbl.setText('-')
            self.dbiscore_lbl.setText('-')
            return

    # Exit function if item has invalid mineral map data
        components = item.getComponents()
        if not isinstance(minmap := components['data'], MineralMap):
            return
            
    # Enable confidence spinbox and slider
        self.conf_spbox.setEnabled(True)
        self.conf_slider.setEnabled(True)

    # Alter minmap data with the confidence threshold
        minmap_thr = self._thresholdMineralMap(minmap)

    # Use item name as title
        title = components['name']

    # Update the legend
        self.legend.refresh(minmap_thr)
    
    # Update the bar plot
        lbls, mode = zip(*minmap_thr.get_labeled_mode().items())
        mode_col = [minmap_thr.get_phase_color(lbl) for lbl in lbls]
        self.barplot.update_canvas((mode,), lbls, mode_col, title=title)

    # Update clustering scores using original (non-thresholded) mineral map
        sil_avg, sil_clust, chi, dbi = minmap.get_clustering_scores()
        if sil_avg and sil_clust:
            self.silscore_canvas.update_canvas(
                sil_clust, sil_avg, title, minmap.palette)
        else:
            self.silscore_canvas.clear_canvas()

    # Round scores according to global decimal precision value
        precision = settings.manager.get('data/decimal_precision')
        sil_lbl = '-' if sil_avg is None else str(round(sil_avg, precision))
        chi_lbl = '-' if chi is None else str(round(chi, precision))
        dbi_lbl = '-' if dbi is None else str(round(dbi, precision))
        self.silscore_lbl.setText(sil_lbl)
        self.chiscore_lbl.setText(chi_lbl)
        self.dbiscore_lbl.setText(dbi_lbl)

    # Update the maps viewer
        mmap, enc, col = minmap_thr.get_plot_data()
        self.maps_viewer.draw_discretemap(mmap, enc, col, title)


    def currentMineralMap(self) -> CW.DataObject | None:
        '''
        Get the currently selected mineral map data object.

        Returns
        -------
        DataObject or None
            Current selected mineral map data object or None if no mineral map
            is selected.

        '''
        item = self.minmaps_list.currentItem()
        if isinstance(item, CW.DataObject):
            return item
        return None


    def getMineralMap(self, idx: int) -> CW.DataObject | None:
        '''
        Get mineral map data object at index 'idx'.

        Parameters
        ----------
        idx : int
            Index of mineral map.

        Returns
        -------
        CW.DataObject or None
            Mineral map data object at given index or None if index is invalid.
            
        '''
        item = self.minmaps_list.topLevelItem(idx)
        if isinstance(item, CW.DataObject):
            return item
        return


    def saveMineralMap(self) -> None: 
        '''
        Save selected mineral map to file.

        '''
    # Exit function if filepath is invalid
        ftype = 'Mineral map (*.mmp)'
        path = CW.FileDialog(self, 'save', 'Save Map', ftype).get_path()
        if not path:
            return

    # Exit function if no valid mineral map data object is currently selected
        if (curr_obj := self.currentMineralMap()) is None:
            return

    # Exit function if mineral map data object has invalid mineral map data
        if (minmap := curr_obj.getMineralMapData()) is None:
            return

    # Attempt to save the mineral map filtered by confidence threshold 
        thresh = self._thresholdMineralMap(minmap)
        try:
            thresh.save(path)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save map.', str(e))


    def removeMineralMap(self) -> None: 
        '''
        Remove the selected mineral map from the list of classified mineral 
        maps.

        '''
        current = self.currentMineralMap()
        item_idx = self.minmaps_list.indexOfTopLevelItem(current)
        choice = CW.MsgBox(self, 'QuestWarn', 'Remove selected map?')
        if choice.yes():
            self.minmaps_list.takeTopLevelItem(item_idx)
            new_displayed_item = self.getMineralMap(item_idx - 1)
            self.minmaps_list.setCurrentItem(new_displayed_item)
            self.showMineralMap(new_displayed_item)


    def setConfidenceThreshold(self, value: int) -> None:
        '''
        Change the probability threshold value. This method dinamically redraws
        the mineral map as well.

        Parameters
        ----------
        value : int
            Threshold value between 0 and 100.

        '''
        if self.sender() == self.conf_spbox:
            other_wid = self.conf_slider
        else:
            other_wid = self.conf_spbox

        other_wid.blockSignals(True)
        other_wid.setValue(value)
        other_wid.blockSignals(False)

    # Re-draw the mineral map
        self.showMineralMap(self.currentMineralMap())


    def _thresholdMineralMap(self, minmap: MineralMap) -> MineralMap:
        '''
        Return a thresholded version of a mineral map. The returned version
        will display _ND_ pixels where the associated probability score is
        below the current confidence threshold value.

        Parameters
        ----------
        minmap : MineralMap
            Original mineral map.

        Returns
        -------
        minmap_thr: MineralMap
            Thresholded version of the mineral map.

        '''
    # Clone mineral map
        minmap_thr = minmap.copy()
    
    # Use confidence threshold to alter the cloned mineral map 
        thresh = self.conf_slider.value() / 100.
        minmap_thr.edit_minmap(minmap_thr._with_nodata(thresh))

    # Set special color to ND data (bugfix for constantly changing _ND_ color)
        if minmap_thr.has_phase('_ND_'):
            minmap_thr.set_phase_color('_ND_', self._nodata_color)

        return minmap_thr
    

    def changeClassColor(self, legend_item: QW.QTreeWidgetItem, color: tuple) -> None:
        '''
        Alter the displayed color of a mineral class. This method propagates
        the changes to the mineral map, the map canvas, the mode bar plot, the 
        legend and the silhouette plot. The arguments of this method are 
        specifically compatible with the 'colorChangeRequested' signal emitted
        by the legend (see 'Legend' class for more details). 

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple
            RGB triplet.

        '''
    # Do nothing if no valid mineral map data object is currently selected
        if (curr_obj := self.currentMineralMap()) is None:
            return

    # Do nothing if mineral map data object has invalid mineral map data
        if (minmap := curr_obj.getMineralMapData()) is None:
            return
        
    # Update the phase color in the mineral map. However, if _ND_ color was 
    # changed, update the _nodata_color attribute instead
        class_name = legend_item.text(1)
        if class_name == '_ND_':
            self._nodata_color = color
        else:
            minmap.set_phase_color(class_name, color)

    # Update mineral map, legend, bar plot and silhouette plot
        palette = self._thresholdMineralMap(minmap).palette
        self.maps_viewer.alter_colormap(palette.values())
        self.legend.changeItemColor(legend_item, color)
        self.barplot.change_bar_color(class_name, color)
        self.silscore_canvas.alter_colors(palette)


    def renameClass(self, legend_item: QW.QTreeWidgetItem, new_name: str) -> None:
        '''
        Rename a mineral class. This method propagates the changes to the 
        mineral map, the map canvas, the mode bar plot and the legend. The 
        arguments of this method are specifically compatible with the 
        'itemRenameRequested' signal emitted by the legend (see 'Legend' class
        for more details).

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be renamed
        new_name : str
            New class name.

        '''
    # Do nothing if no valid mineral map data object is currently selected
        if (curr_obj := self.currentMineralMap()) is None:
            return

    # Do nothing if mineral map data object has invalid mineral map data
        if (minmap := curr_obj.getMineralMapData()) is None:
            return
        
    # Rename phase in the current mineral map 
        old_name = legend_item.text(1)
        minmap.rename_phase(old_name, new_name)
    
    # Re-draw mineral map, legend, barplot and silhouette plot
        self.showMineralMap(curr_obj)


    def highlightClass(self, toggled: bool, legend_item: QW.QTreeWidgetItem) -> None:
        '''
        Highlight on/off the selected mineral class in the map canvas. The 
        arguments of this method are specifically compatible with the 
        'itemHighlightRequested' signal emitted by the legend (see 'Legend'
        class for more details).

        Parameters
        ----------
        toggled : bool
            Highlight on/off
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be highlighted.

        '''
    # Do nothing if a mineral map is not currently displayed in the viewer
        if not self.maps_viewer.has_discretemap():
            return

    # Do nothing if no valid mineral map data object is currently selected
        if (curr_obj := self.currentMineralMap()) is None:
            return

    # Do nothing if mineral map data object has invalid mineral map data
        if (minmap := curr_obj.getMineralMapData()) is None:
            return

    # Change phase visibility based on toggled state
        if toggled:
        # We need to operate on the thresholded version of the mineral map
            thresh = self._thresholdMineralMap(minmap)
            vmin = vmax = thresh.as_id(legend_item.text(1))
        else:
            vmin = vmax = None

        self.maps_viewer.update_clim(vmin, vmax)


    def startClassification(self) -> None:
        '''
        Launch a classification process.

        '''
    # Do not allow multiple classification processes at once
        if self._isBusyClassifying:
            CW.MsgBox(self, 'C', 'Cannot run multiple classifications.')
            return
        
    # Build the input maps stack
        checked = self.inmaps_selector.getCheckedMaps()
        inmaps = [i for c in checked if (i := c.getInputMapData()) is not None]
        mask = self._mask if self.mask_group.isChecked() else None
        input_stack = InputMapStack(inmaps, mask)
        
    # Check for valid stack size
        if not len(input_stack):
            CW.MsgBox(self, 'Crit', 'Select at least one map.')
            return
    
    # Check for maps perfect overlapping
        if not input_stack.maps_fit():
            CW.MsgBox(self, 'Crit', 'Input maps do not overlap.')
            return
        
    # Check that mask (if present) has correct shape
        if not input_stack.mask_fit():
            CW.MsgBox(self, 'C', 'The selected mask has invalid shape.')
            return

    # Get the classifier and launch the classification thread
        if (active_tab := self.classifier_tabwid.currentWidget()) is not None:
            csf = active_tab.getClassifier(input_stack)
            if isinstance(csf, mltools._ClassifierBase):
                csf.thread.taskInitialized.connect(self.progbar.step)
                csf.thread.workInterrupted.connect(self._endClassification)
                csf.thread.workFinished.connect(self._parseClassifierResult)

                self.progbar.setRange(0, csf.classification_steps)
                self._current_classifier = csf
                self._isBusyClassifying = True

                csf.startThreadedClassification()


    def _parseClassifierResult(self, result: tuple, success: bool) -> None:
        '''
        Parse the result of the classification thread. If the classification 
        was successful, this method shows the mineral map in the canvas.

        Parameters
        ----------
        result : tuple
            Classification thread result.
        success : bool
            Whether the classification thread succeeded or not.

        '''
        if success and self._current_classifier is not None:  
        # Parse the classification result appropriately
            if self._current_classifier.kind == 'Unsupervised':
                pred, prob, sil_avg, sil_clust, chi, dbi = result
                pred = pred.astype(MineralMap._DTYPE_STR, copy=False)
            else:   
                pred, prob = result
                sil_avg, sil_clust, chi, dbi = None, None, None, None
            shape = self._current_classifier.map_shape
            mask = self._current_classifier.input_stack.mask

        # Reshape mineral map and probability map when mask is absent
            if mask is None:
                mmap = pred.reshape(shape)
                pmap = prob.reshape(shape)

        # Reshape mineral map and probability map when mask is present
            else:
                rows, cols = (mask.mask == 0).nonzero()
                mmap = np.empty(shape, dtype=MineralMap._DTYPE_STR)
                mmap[:, :] = '_MSK_'
                mmap[rows, cols] = pred
                pmap = np.ones(shape)
                pmap[rows, cols] = prob

        # Create a new item in minmaps list and populate it with a Mineral Map
            mineral_map = MineralMap(mmap, pmap)
            mineral_map.set_clustering_scores(sil_avg, sil_clust, chi, dbi)
            item = CW.DataObject(mineral_map)
            dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = f'{self._current_classifier.name} [{dt}]'
            item.setText(0, name)
            item.setToolTip(0, name)
            self.minmaps_list.addTopLevelItem(item)

        # Force show the output maps tab and display the mineral map
            self.data_tabwid.setCurrentIndex(1) 
            self.minmaps_list.setCurrentItem(item)
            self.showMineralMap(item)

        else:
            CW.MsgBox(self, 'Crit', 'Classification failed.', str(result[0]))

        self._endClassification(success)


    def stopClassification(self) -> None:
        '''
        Interrupt the current classification thread.

        '''
        if self._current_classifier is not None:
            self.progbar.setUndetermined()
            self.progbar.step('Interrupting classification')
            self._current_classifier.thread.requestInterruption()


    def _endClassification(self, success: bool = False) -> None:
        '''
        Internally and visually exit from a classification thread session.

        Parameters
        ----------
        success : bool, optional
            Whether the classification thread ended with success. The default 
            is False.

        '''
        self._isBusyClassifying = False
        self._current_classifier = None
        self.progbar.reset()

    # If pbar is left undetermined, it visually seems that process hasn't stop
        if self.progbar.undetermined():
            self.progbar.setMaximum(1)

        if success:
            CW.MsgBox(self, 'Info', 'Classification completed successfully.')

    # deprecated
    # def killReferences(self) -> None:
    #     '''
    #     Reimplementation of the 'killReferences' method from the parent class.
    #     It also stops any active classification thread.

    #     '''
    #     if self._isBusyClassifying:
    #         self.stopClassification()
    #     super().killReferences()

    def is_safe_to_close(self) -> bool:
        '''
        Reimplementation of the 'is_safe_to_close' method. Changes the close
        confirmation question if a classification thread is currently active.
        It stops the thread if user confirms to close the tool.

        Returns
        -------
        bool
            Whether the user confirms to close the tool or not.

        '''
        if self._isBusyClassifying:
            text = (
                f'A classification process is active. Close {self.windowTitle()}'
                ' anyway? The process will be interrupted and any unsaved data'
                ' will be lost.'
            )
            msgbox = CW.MsgBox(self, 'Quest', text)
            if close := msgbox.is_choice('Yes'):
                self.stopClassification()
            return close
        
        else:
            return super().is_safe_to_close()


    def force_close(self) -> None:
        '''
        Reimplementation of the 'force_close' method. It also stops any active
        classification thread.

        '''
        if self._isBusyClassifying:
            self.stopClassification()
        return super().force_close()


    class PreTrainedClassifierTab(QW.QWidget):

        def __init__(self, parent: QW.QWidget | None = None) -> None:
            '''
            First tab of the Mineral Classifier's tab widget. It allows loading
            a pre-trained machine learning model.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attributes
            self.model: mltools.EagerModel | None = None

        # Initialize GUI and connect its signals with slots
            self._init_ui()
            self._connect_slots()


        def _init_ui(self) -> None:
            '''
            GUI constructor.

            '''
        # Load Model (Styled Button)
            self.load_btn = CW.StyledButton(
                style.getIcon('IMPORT'), 'Load model')
            self.load_btn.setToolTip('Load a pre-trained ML model')

        # Loaded model path (Path Label)
            self.model_path = CW.PathLabel(full_display=False)

        # Loaded model information (Document Browser)
            placeholder_text = 'Unable to retrieve model information.'
            self.model_info = CW.DocumentBrowser()
            self.model_info.setDefaultPlaceHolderText(placeholder_text)

        # Maps selector (Maps Matcher)
            self.matcher = CW.MapsMatcher()

        # Adjust main layout
            left_vbox = QW.QVBoxLayout()
            left_vbox.addWidget(self.load_btn)
            left_vbox.addWidget(self.model_path)
            left_vbox.addWidget(self.model_info)

            right_vbox = QW.QVBoxLayout()
            right_vbox.addWidget(QW.QLabel('Match input features'))
            right_vbox.addWidget(self.matcher)
            
            main_layout = CW.SplitterLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addLayout(left_vbox)
            main_layout.addLayout(right_vbox)
            self.setLayout(main_layout)


        def _connect_slots(self) -> None:
            '''
            Signals-slots connector.

            '''
            self.load_btn.clicked.connect(self.loadModel)


        def loadModel(self) -> None:
            '''
            Load a pre-trained model from file.

            '''
        # Do nothing if path is invalid or file dialog is canceled
            ftype = 'PyTorch model (*.pth)'
            path = CW.FileDialog(self, 'open', 'Load Model', ftype).get_path()
            if not path:
                return

            try:
                self.model = mltools.EagerModel.load(path)
            except Exception as e:
                CW.MsgBox(self, 'Crit', 'Incompatible model.', str(e))
                return

        # Model's path changed if model was converted to a newer version
            path = self.model.filepath
            self.model_path.setPath(path)
            logpath = '' if path is None else self.model.generate_log_path(path)

        # If model log was deleted or moved, ask for rebuilding it
            if not os.path.exists(logpath):
                quest_text = 'Unable to find model log file. Rebuild it?'
                choice = CW.MsgBox(self, 'Quest', quest_text)
                if choice.yes():
                    ext_log = settings.manager.get('data/extended_model_log')
                    self.model.save_log(logpath, extended=ext_log)

        # Load log file. No error will be raised if it does not exist
            self.model_info.setDoc(logpath)

        # Populate the matcher widget
            self.matcher.setMaps(self.model.features)


        def getClassifier(self, input_stack: InputMapStack
        ) -> mltools.ModelBasedClassifier | None:
            '''
            Return the classifier stored in the loaded pre-trained model. This
            method also checks if the input maps required by the model are
            present.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.

            Returns
            -------
            ModelBasedClassifier or None
                The pre-trained classifier or None if input maps data is not 
                properly provided.

            '''
        # Check if model is loaded
            if self.model == None:
                CW.MsgBox(self, 'Crit', 'A pre-trained model is required.')
                return

        # Check if all required input maps are matched
            idx = self.matcher.getMatchedIndices()
            empty_rows = [n for n, i in enumerate(idx) if i == -1]
            if empty_rows:
                missing = [self.matcher.getMapName(r) for r in empty_rows]
                text = 'One or more features have not been matched.'
                CW.MsgBox(self, 'Crit', text, ', '.join(missing))
                return
            
        # Send warning if maps have been matched to multiple features
            maps = self.matcher.getMatchedMaps()
            dupes = {maps[n] for n, i in enumerate(idx) if idx.count(i) > 1}
            if dupes:
                text = 'One or more maps match multiple features. Proceed?'
                if CW.MsgBox(self, 'QuestWarn', text, ', '.join(dupes)).no():
                    return
            
        # Reorder input maps to fit the order of model's features
            input_stack.reorder(idx)
            return mltools.ModelBasedClassifier(input_stack, self.model)


    class RoiBasedClassifierTab(QW.QWidget):

        addRoiMapRequested = QC.pyqtSignal(RoiMap)
        removeRoiMapRequested = QC.pyqtSignal()

        def __init__(self, parent: QW.QWidget | None = None) -> None:
            '''
            Second tab of the Mineral Classifier's tab widget. It allows the
            selection of a ROI-based classifier.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attributes
            self._algorithms = ('K-Nearest Neighbors', )
            self._roimap: RoiMap | None = None

        # Initialize GUI and connect its signals with slots
            self._init_ui()
            self._connect_slots()


        def _init_ui(self) -> None:
            '''
            GUI constructor.

            '''
        # Load ROI map (Styled Button)
            self.load_btn = CW.StyledButton(
                style.getIcon('IMPORT'), 'Load ROI map')
            self.load_btn.setToolTip('Load training ROI data')

        # Remove (unload) ROI map (Styled Button)
            self.unload_btn = CW.StyledButton(style.getIcon('CLEAR'))
            self.unload_btn.setToolTip('Remove ROI map')
            
        # Loaded model path (Path Label)
            self.roimap_path = CW.PathLabel(full_display=False)

        # Canvas for preview of loaded ROI map (ImageCanvas)
            self.roi_canvas = plots.ImageCanvas(
                binary=True,
                cbar=False,
                size=(2, 1.5),
                layout='constrained',
                wheel_zoom=False,
                wheel_pan=False
            )
            self.roi_canvas.fig.patch.set(facecolor=style.CASPER_LIGHT)
            self.roi_canvas.ax.xaxis.set_visible(False)
            self.roi_canvas.ax.yaxis.set_visible(False)
            self.roi_canvas.setMinimumHeight(50)

        # Include pixel proximity (Check Box)
            self.pixprox_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
            self.pixprox_cbox.setToolTip('Use pixel coords as input features')
            self.pixprox_cbox.setChecked(False)

        # Use parallel computation (Check Box)
            multithread_tip = 'Distribute computation across multiple processes'
            self.multithread_cbox = QW.QCheckBox('Parallel computation')
            self.multithread_cbox.setToolTip(multithread_tip)
            self.multithread_cbox.setChecked(False)

        # Algorithm selection (Styled Combo Box)
            self.algm_combox = CW.StyledComboBox()
            self.algm_combox.addItems(self._algorithms)

        # Algorithms Panel (Stacked Widget)
            self.algm_panel = QW.QStackedWidget()


        #---------------K-NEAREST NEIGHBORS ALGORITHM SETTINGS----------------#
        # N. of neighbors  (Styled Spin Box)
            self.knn_nneigh_spbox = CW.StyledSpinBox(1, 100)
            self.knn_nneigh_spbox.setValue(5)

        # Weight of neighbors (Styled Combo Box)
            knn_tip = 'Should neighbors weighting be uniform or distance-based?'
            self.knn_weight_combox = CW.StyledComboBox()
            self.knn_weight_combox.addItems(['Uniform', 'Distance'])
            self.knn_weight_combox.setToolTip(knn_tip)

        # Add KNN widgets to the Algorithm Panel
            knn_layout = QW.QFormLayout()
            knn_layout.addRow('N. of neighbours', self.knn_nneigh_spbox)
            knn_layout.addRow('Neighbors weight', self.knn_weight_combox)
            knn_group = CW.GroupArea(knn_layout)
            
            self.algm_panel.addWidget(knn_group)
        #---------------------------------------------------------------------#

        # Adjust main layout
            left_grid = QW.QGridLayout()
            left_grid.setColumnStretch(1, 1)
            left_grid.setRowStretch(2, 1)
            left_grid.addWidget(self.load_btn, 0, 0, 1, -1)
            left_grid.addWidget(self.unload_btn, 1, 0, 1, 1)
            left_grid.addWidget(self.roimap_path, 1, 1)
            left_grid.addWidget(self.roi_canvas, 2, 0, 1, -1)
            left_grid.addWidget(self.pixprox_cbox, 3, 0, 1, -1)
            left_grid.addWidget(self.multithread_cbox, 4, 0, 1, -1)

            right_vbox = QW.QVBoxLayout()
            right_vbox.addWidget(QW.QLabel('Select algorithm'))
            right_vbox.addWidget(self.algm_combox)
            right_vbox.addWidget(self.algm_panel)

            main_layout = CW.SplitterLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addLayout(left_grid)
            main_layout.addLayout(right_vbox)
            self.setLayout(main_layout)


        def _connect_slots(self) -> None:
            '''
            Signals-slots connector.

            '''
        # Load ROI map from file
            self.load_btn.clicked.connect(self.loadRoiMap)

        # Remove (unload) ROI map
            self.unload_btn.clicked.connect(self.unloadRoiMap)

        # Select a different ROI-based algorithm
            self.algm_combox.currentTextChanged.connect(self.switchAlgorithm)


        def loadRoiMap(self) -> None:
            '''
            Load a new ROI map from file and request its rendering in the 
            Mineral Classifier's maps canvas.

            '''
        # Do nothing if path is invalid or file dialog is canceled
            ftype = 'ROI maps (*.rmp)'
            path = CW.FileDialog(self, 'open', 'Load ROI', ftype).get_path()
            if not path:
                return 
            
            pbar = CW.PopUpProgBar(self, 3, 'Loading data')

        # Try loading the ROI map. Exit function if something goes wrong
            try:
                new_roimap = RoiMap.load(path)
                pbar.increase()
            except Exception as e:
                pbar.reset()
                CW.MsgBox(self, 'C', f'Unexpected file:\n{path}.', str(e))
                return

        # Remove previous ROI map from canvas
            self.unloadRoiMap()
            pbar.increase()

        # Add the new ROI map and populate the canvas with its ROIs
            self._roimap = new_roimap
            self.addRoiMapRequested.emit(new_roimap)
            self.roimap_path.setPath(path)
            self.roi_canvas.draw_heatmap(self._roimap.map != '')
            pbar.increase()


        def unloadRoiMap(self) -> None:
            '''
            Unload current ROI map and request its removal from the Mineral
            Classifier's maps canvas.

            '''
            if self._roimap is not None:
                self._roimap = None
                self.roimap_path.clearPath()
                self.roi_canvas.clear_canvas()
                self.removeRoiMapRequested.emit()
                

        def switchAlgorithm(self, algorithm: str) -> None:
            '''
            Swap widget in the algorithms panel based on selected algorithm.

            Parameters
            ----------
            algorithm : str
                Selected algorithm.

            '''
            idx = self._algorithms.index(algorithm)
            self.algm_panel.setCurrentIndex(idx)


        def getClassifier(self, input_stack: InputMapStack
        ) -> mltools.RoiBasedClassifier | None: 
            '''
            Return the ROI-based classifier with the selected parameters. This
            fuction also checks if the extension the ROI map is compatible with
            the extensions of the input maps.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.

            Returns
            -------
            RoiBasedClassifier or None
                The ROI-based classifier or None if the ROI map is absent or it
                has incompatible extension.

            '''
            roimap = self._roimap
            prox = self.pixprox_cbox.checkState()
            algm = self.algm_combox.currentText()

            if roimap is None:
                CW.MsgBox(self, 'Crit', 'A ROI map is required.')
                return None

        # Check if ROI map extension differs from input maps extensions
            if input_stack.maps_shape != roimap.shape:
                text = 'ROI map and sample extensions are different. Proceed?'
                choice = CW.MsgBox(self, 'QuestWarn', text)
                if choice.no():
                    return None

            if algm == 'K-Nearest Neighbors':
                nneigh = self.knn_nneigh_spbox.value()
                weight = self.knn_weight_combox.currentText().lower()
                njobs = -1 if self.multithread_cbox.checkState() else 1
                args = (input_stack, roimap, nneigh, weight)
                kwargs = {'n_jobs': njobs, 'pixel_proximity': prox}
                return mltools.KNearestNeighbors(*args, **kwargs)

            else:
                return None


    class UnsupervisedClassifierTab(QW.QWidget):

        def __init__(self, parent: QW.QWidget | None = None) -> None:
            '''
            Third tab of the Mineral Classifier's tab widget. It allows the
            selection of an unsupervised classifier.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attribute
            self._algorithms = ('K-Means', )

        # Initialize GUI and connect its signals with slots
            self._init_ui()
            self._connect_slots()


        def _init_ui(self) -> None:
            '''
            GUI constructor.

            '''
        # Seed generator widget 
            self.seed_generator = CW.RandomSeedGenerator()

        # Compute silhouette score (Check Box)
            self.silscore_cbox = QW.QCheckBox('Silhouette score')
            self.silscore_cbox.setChecked(True)
            self.silscore_cbox.setToolTip('This may take some time')

        # Silhouette score subset ratio (Styled Spin Box)
            ratio_tip = 'Analyze a portion of data to reduce computation time'
            self.silscore_ratio_spbox = CW.StyledSpinBox()
            self.silscore_ratio_spbox.setSuffix(' %')
            self.silscore_ratio_spbox.setValue(25)
            self.silscore_ratio_spbox.setToolTip(ratio_tip)

        # Compute Calinski-Harabasz Index (Check Box)
            self.chiscore_cbox = QW.QCheckBox('Calinski-Harabasz Index')
            self.chiscore_cbox.setChecked(True)

        # Compute Davies-Bouldin Index (Check Box)
            self.dbiscore_cbox = QW.QCheckBox('Davies-Bouldin Index')
            self.dbiscore_cbox.setChecked(True)

        # Include pixel proximity (Check Box)
            self.pixprox_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
            self.pixprox_cbox.setToolTip('Use pixel coords as input features')
            self.pixprox_cbox.setChecked(False)

        # Use parallel computation (Check Box)
            multithread_tip = 'Distribute computation across multiple processes'
            self.multithread_cbox = QW.QCheckBox('Parallel computation')
            self.multithread_cbox.setToolTip(multithread_tip)
            self.multithread_cbox.setChecked(False)

        # Algorithm selection (Styled Combo Box)
            self.algm_combox = CW.StyledComboBox()
            self.algm_combox.addItems(self._algorithms)

        # Algorithms Panel (Stacked Widget)
            self.algm_panel = QW.QStackedWidget()


        #--------------------- K-MEANS ALGORITHM WIDGETS ---------------------#
        # N. of clusters (Styled Spin Box)
            self.kmeans_nclust_spbox = CW.StyledSpinBox(min_value=2)
            self.kmeans_nclust_spbox.setValue(8)

        # Add K-Means widgets to the Algorithm Panel
            kmeans_layout = QW.QFormLayout()
            kmeans_layout.addRow('N. of clusters', self.kmeans_nclust_spbox)
            kmeans_group = CW.GroupArea(kmeans_layout)

            self.algm_panel.addWidget(kmeans_group)
        #---------------------------------------------------------------------#

        # Adjust layout
            scores_grid = QW.QFormLayout()
            scores_grid.setContentsMargins(9, 9, 9, 9)
            scores_grid.addRow(self.silscore_cbox)
            scores_grid.addRow('Silhouette ratio', self.silscore_ratio_spbox)
            scores_grid.addRow(self.chiscore_cbox)
            scores_grid.addRow(self.dbiscore_cbox)
            scores_group = CW.GroupArea(
                scores_grid, 'Clustering scores', checkable=True, align='left')
            scores_group.setChecked(False)

            left_vbox = QW.QVBoxLayout()
            left_vbox.addWidget(self.seed_generator)
            left_vbox.addWidget(scores_group, 1)
            left_vbox.addWidget(self.pixprox_cbox)
            left_vbox.addWidget(self.multithread_cbox)

            right_vbox = QW.QVBoxLayout()
            right_vbox.addWidget(QW.QLabel('Select algorithm'))
            right_vbox.addWidget(self.algm_combox)
            right_vbox.addWidget(self.algm_panel)

            main_layout = CW.SplitterLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addLayout(left_vbox)
            main_layout.addLayout(right_vbox)
            self.setLayout(main_layout)


        def _connect_slots(self) -> None:
            '''
            Signals-slots connector.

            '''
        # Disable silhouette data ratio when silhouette score is unchecked
            self.silscore_cbox.stateChanged.connect(
                self.silscore_ratio_spbox.setEnabled)
            
        # Select a different clustering algorithm
            self.algm_combox.currentTextChanged.connect(self.switchAlgorithm)


        def switchAlgorithm(self, algorithm: str) -> None:
            '''
            Swap widget in the algorithms panel based on selected algorithm.

            Parameters
            ----------
            algorithm : str
                Selected algorithm.

            '''
            idx = self._algorithms.index(algorithm)
            self.algm_panel.setCurrentIndex(idx)


        def getClassifier(self, input_stack: InputMapStack
        ) -> mltools.UnsupervisedClassifier | None:
            '''
            Return the unsupervised classifier with the selected parameters.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.

            Returns
            -------
            UnsupervisedClassifier or None
                The unsupervised classifier.

            '''
        # Common clustering parameters
            seed = self.seed_generator.seed
            sil = self.silscore_cbox.isEnabled() & self.silscore_cbox.isChecked()
            chi = self.chiscore_cbox.isEnabled() & self.chiscore_cbox.isChecked()
            dbi = self.dbiscore_cbox.isEnabled() & self.dbiscore_cbox.isChecked()
            kwargs = {
                'n_jobs': -1 if self.multithread_cbox.checkState() else 1,
                'pixel_proximity': self.pixprox_cbox.checkState(),
                'sil_score': sil,
                'chi_score': chi,
                'dbi_score': dbi,
                'sil_ratio': self.silscore_ratio_spbox.value() / 100
            }
        
        # Algorithm-specific parameters
            algm = self.algm_combox.currentText()

            if algm == 'K-Means':
                n_clust = self.kmeans_nclust_spbox.value()
                args = (input_stack, seed, n_clust)
                return mltools.KMeans(*args, **kwargs)

            else:
                return None



class DatasetBuilder(DraggableTool):

    _elements = {
        'Ac': 'Actinium (Z=89)',     'Ag': 'Silver (Z=47)',      
        'Al': 'Aluminum (Z=13)',     'Ar': 'Argon (Z=18)',      
        'As': 'Arsenic (Z=33)',      'At': 'Astatine (Z=85)',   
        'Au': 'Gold (Z=79)',         'B': 'Boron (Z=5)',        
        'Ba': 'Barium (Z=56)',       'Be': 'Beryllium (Z=4)',   
        'Bi': 'Bismuth (Z=83)',      'Br': 'Bromine (Z=35)',    
        'C': 'Carbon (Z=6)',         'Ca': 'Calcium (Z=20)',    
        'Cd': 'Cadmium (Z=48)',      'Ce': 'Cerium (Z=58)',     
        'Cl': 'Chlorine (Z=17)',     'Co': 'Cobalt (Z=27)',     
        'Cr': 'Chromium (Z=24)',     'Cs': 'Cesium (Z=55)',     
        'Cu': 'Copper (Z=29)',       'Dy': 'Dysprosium (Z=66)', 
        'Er': 'Erbium (Z=68)',       'Eu': 'Europium (Z=63)',   
        'F': 'Fluorine (Z=9)',       'Fe': 'Iron (Z=26)',       
        'Fr': 'Francium (Z=87)',     'Ga': 'Gallium (Z=31)',    
        'Gd': 'Gadolinium (Z=64)',   'Ge': 'Germanium (Z=32)',  
        'H': 'Hydrogen (Z=1)',       'He': 'Helium (Z=2)',      
        'Hf': 'Hafnium (Z=72)',      'Hg': 'Mercury (Z=80)',    
        'Ho': 'Holmium (Z=67)',      'I': 'Iodine (Z=53)',      
        'In': 'Indium (Z=49)',       'Ir': 'Iridium (Z=77)',    
        'K': 'Potassium (Z=19)',     'Kr': 'Krypton (Z=36)',    
        'La': 'Lanthanum (Z=57)',    'Li': 'Lithium (Z=3)',     
        'Lu': 'Lutetium (Z=71)',     'Mg': 'Magnesium (Z=12)',  
        'Mn': 'Manganese (Z=25)',    'Mo': 'Molybdenum (Z=42)', 
        'N': 'Nitrogen (Z=7)',       'Na': 'Sodium (Z=11)',     
        'Nb': 'Niobium (Z=41)',      'Nd': 'Neodymium (Z=60)',  
        'Ne': 'Neon (Z=10)',         'Ni': 'Nickel (Z=28)',     
        'O': 'Oxygen (Z=8)',         'Os': 'Osmium (Z=76)',     
        'P': 'Phosphorus (Z=15)',    'Pa': 'Protactinium (Z=91)',
        'Pb': 'Lead (Z=82)',         'Pd': 'Palladium (Z=46)',  
        'Pm': 'Promethium (Z=61)',   'Po': 'Polonium (Z=84)',   
        'Pr': 'Praseodymium (Z=59)', 'Pt': 'Platinum (Z=78)',   
        'Ra': 'Radium (Z=88)',       'Rb': 'Rubidium (Z=37)',   
        'Re': 'Rhenium (Z=75)',      'Rh': 'Rhodium (Z=45)',    
        'Rn': 'Radon (Z=86)',        'Ru': 'Ruthenium (Z=44)',  
        'S': 'Sulfur (Z=16)',        'Sb': 'Antimony (Z=51)',   
        'Sc': 'Scandium (Z=21)',     'Se': 'Selenium (Z=34)',   
        'Si': 'Silicon (Z=14)',      'Sm': 'Samarium (Z=62)',   
        'Sn': 'Tin (Z=50)',          'Sr': 'Strontium (Z=38)',  
        'Ta': 'Tantalum (Z=73)',     'Tb': 'Terbium (Z=65)',    
        'Tc': 'Technetium (Z=43)',   'Te': 'Tellurium (Z=52)',  
        'Th': 'Thorium (Z=90)',      'Ti': 'Titanium (Z=22)',   
        'Tl': 'Thallium (Z=81)',     'Tm': 'Thulium (Z=69)',    
        'U': 'Uranium (Z=92)',       'V': 'Vanadium (Z=23)',    
        'W': 'Tungsten (Z=74)',      'Xe': 'Xenon (Z=54)',      
        'Y': 'Yttrium (Z=39)',       'Yb': 'Ytterbium (Z=70)',  
        'Zn': 'Zinc (Z=30)',         'Zr': 'Zirconium (Z=40)' 
    }

    def __init__(self) -> None:
        '''
        One of the main tools of X-Min Learn, that allows the semi-automated
        compilation of human-readable, machine-friendly ground truth datasets.

        '''
        super().__init__()

    # Set tool title and icon
        self.setWindowTitle('Dataset Builder')
        self.setWindowIcon(style.getIcon('DATASET_BUILDER'))

    # Set main attributes
        self._dataset: dtools.GroundTruthDataset | None = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Input grid of chemical elements (Styled Buttons) [-> GroupScrollArea]
        self.elements_btns: list[CW.StyledButton] = []
        elem_grid = QW.QGridLayout()
        n_cols = 9
        for n, (k, v) in enumerate(self._elements.items()):
            elem_btn = CW.StyledButton(text=k)
            elem_btn.setToolTip(v)
            elem_btn.setCheckable(True)
            self.elements_btns.append(elem_btn)
            row, col = n // n_cols, n % n_cols
            elem_grid.addWidget(elem_btn, row, col)
        elem_scroll = CW.GroupScrollArea(elem_grid)

    # Custom feature entry (Line Edit)
        self.custom_entry = QW.QLineEdit()
        self.custom_entry.setStyleSheet(style.SS_MENU)
        self.custom_entry.setPlaceholderText('Custom feature name')

    # Selected elements list (Styled List Widget)
        self.feature_list = CW.StyledListWidget()
        self.feature_list.setSortingEnabled(True)

    # Delete feature (Styled Button)
        self.delfeat_btn = CW.StyledButton(style.getIcon('REMOVE'))
        self.delfeat_btn.setToolTip('Remove selected features')

    # Refresh Dataset Designer (Styled Button)
        self.refresh_btn = CW.StyledButton(style.getIcon('TICK'))
        self.refresh_btn.setToolTip('Refresh Dataset Designer')

    # Dataset designer (Dataset Designer)
        self.designer = CW.DatasetDesigner()
        self.designer.setMinimumHeight(500)

    # Designer progressbar (Progress Bar)
        self.designer_pbar = QW.QProgressBar()

    #[Designer actions]
    #|Expand designer's rows
        self.expand_action = QW.QAction(
            style.getIcon('ROW_ADD'),'Expand rows', self)
    #|Reduce designer's rows
        self.reduce_action = QW.QAction(
            style.getIcon('ROW_DEL'), 'Reduce rows', self)
    #|Remove selected rows
        self.delrows_action = QW.QAction(
            style.getIcon('REMOVE'), 'Remove selected rows', self)
    #|Compile dataset (Widget Action)
        self.compile_btn = CW.StyledButton(
            style.getIcon('DATASET_BUILDER'), 'COMPILE')

    # Designer toolbar (Styled Toolbar)
        self.designer_tbar = QW.QToolBar('Designer toolbar')
        self.designer_tbar.setEnabled(False)
        self.designer_tbar.setStyleSheet(style.SS_TOOLBAR)
        self.designer_tbar.addActions((
            self.expand_action,
            self.reduce_action,
            self.delrows_action
        ))
        self.designer_tbar.addSeparator()
        self.designer_tbar.addWidget(self.compile_btn)

    # Class refinement list (Styled List Widget)
        self.refine_list = CW.StyledListWidget()
        self.refine_list.setSortingEnabled(True)

    # Rename selected class (Styled Button)
        self.rename_class_btn = CW.StyledButton(style.getIcon('RENAME'))
        self.rename_class_btn.setToolTip('Rename')
        
    # Delete selected class (Styled Button)
        self.delete_class_btn = CW.StyledButton(style.getIcon('REMOVE'))
        self.delete_class_btn.setToolTip('Delete')

    # Merge selected classes (Styled Button)
        self.merge_class_btn = CW.StyledButton(style.getIcon('MERGE'))
        self.merge_class_btn.setToolTip('Merge')
        
    # Dataset preview area (Document Browser)
        self.preview = CW.DocumentBrowser(tbar=False)

    # CSV decimal point selector (Decimal Point Selector)
        self.decimal_combox = CW.DecimalPointSelector()

    # CSV separator character selector (Separator Symbol Selector)
        self.separator_combox = CW.SeparatorSymbolSelector()

    # Split large dataset (Check Box)
        split_tip = (
            'Split dataset into multiple CSV files if the number of lines '
            'exceeds Microsoft Excel rows limit (about 1 million)'
        )
        self.split_cbox = QW.QCheckBox('Split dataset')
        self.split_cbox.setToolTip(split_tip)
        self.split_cbox.setChecked(False)

    # Save dataset (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'Save')

    # Adjust layout
        # Input features group (Group Area)
        infeat_grid = QW.QGridLayout()
        infeat_grid.addWidget(self.custom_entry, 0, 0, 1, -1)
        infeat_grid.addWidget(self.feature_list, 1, 0, 1, -1)
        infeat_grid.addWidget(self.delfeat_btn, 2, 0, 1, 1)
        infeat_grid.addWidget(self.refresh_btn, 2, 1, 1, 1)
        
        infeat_vsplit = CW.SplitterLayout(orient='vertical')
        infeat_vsplit.addWidget(elem_scroll, 0)
        infeat_vsplit.addLayout(infeat_grid, 1)
        infeat_group = CW.GroupArea(infeat_vsplit, 'Input features')

        # Output preferences group (Group Area)
        outpref_form = QW.QFormLayout()
        outpref_form.setVerticalSpacing(10)
        outpref_form.addRow('CSV decimal point', self.decimal_combox)
        outpref_form.addRow('CSV separator', self.separator_combox)
        outpref_form.addRow(self.split_cbox)
        outpref_form.addRow(self.save_btn)
        outpref_group = CW.GroupArea(outpref_form, 'Output preferences')

        # Dataset designer group (Group Area)
        designer_grid = QW.QGridLayout()
        designer_grid.setHorizontalSpacing(15)
        designer_grid.addWidget(self.designer, 0, 0, 1, -1)
        designer_grid.addWidget(self.designer_tbar, 1, 0)
        designer_grid.addWidget(self.designer_pbar, 1, 1)
        designer_group = CW.GroupArea(designer_grid, 'Dataset Designer')
   
        # Dataset refinement group (GroupArea)
        refine_grid = QW.QGridLayout()
        refine_grid.setColumnStretch(4, 1)
        refine_grid.setColumnMinimumWidth(3, 15)
        refine_grid.addWidget(self.refine_list, 0, 0, 1, 3)
        refine_grid.addWidget(self.preview, 0, 4, -1, 1)
        refine_grid.addWidget(self.rename_class_btn, 1, 0, 1, 1)
        refine_grid.addWidget(self.delete_class_btn, 1, 1, 1, 1)
        refine_grid.addWidget(self.merge_class_btn, 1, 2, 1, 1)
        refine_group = CW.GroupArea(refine_grid, 'Dataset Refinement')

        # Set main layout
        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(20)
        left_vbox.addWidget(infeat_group)
        left_vbox.addWidget(outpref_group)
        left_scroll = CW.GroupScrollArea(left_vbox)

        right_vbox = QW.QVBoxLayout()
        right_vbox.setSpacing(20)
        right_vbox.addWidget(designer_group)
        right_vbox.addWidget(refine_group)
        right_scroll = CW.GroupScrollArea(right_vbox)

        main_layout = CW.SplitterLayout()
        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(right_scroll, 1)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Connection of each element button
        for btn in self.elements_btns:
            btn.toggled.connect(self.onElementButtonToggled)
            
    # Input features signals
        self.custom_entry.returnPressed.connect(self.addCustomFeature)
        self.delfeat_btn.clicked.connect(self.removeSelectedFeatures)
        self.refresh_btn.clicked.connect(self.refreshDesigner)

    # Dataset Designer signals
        self.expand_action.triggered.connect(self.designer.addRow)
        self.reduce_action.triggered.connect(self.designer.delLastRow)
        self.delrows_action.triggered.connect(self.removeSelectedRows)
        self.compile_btn.clicked.connect(self.compileDataset)
        
    # Dataset refinement signals
        self.rename_class_btn.clicked.connect(self.renameClass)
        self.delete_class_btn.clicked.connect(self.deleteClass)
        self.merge_class_btn.clicked.connect(self.mergeClass)

    # Output dataset signals
        self.save_btn.clicked.connect(self.saveDataset)


    @property
    def dataset(self) -> dtools.GroundTruthDataset:
        '''
        Get the current ground truth dataset.

        Returns
        -------
        dataset : dtools.GroundTruthDataset
            Current dataset.

        Raises
        ------
        TypeError
            Raised if the current dataset is None.

        '''
        if (dataset := self._dataset) is None:
            raise TypeError('Current dataset is None.')
        return dataset


    def set_dataset(self, dataset: dtools.GroundTruthDataset) -> None:
        '''
        Set the new current ground truth dataset.

        Parameters
        ----------
        dataset : dtools.GroundTruthDataset
            New dataset.

        '''
        self._dataset = dataset


    def has_dataset(self) -> bool:
        '''
        Check if a ground truth dataset currently exists.

        Returns
        -------
        bool
            Whether a dataset exists or not.

        '''
        return self._dataset is not None


    def onElementButtonToggled(self, toggled: bool) -> None:
        '''
        Actions to be performed when an element button is toggled on/off.

        Parameters
        ----------
        toggled : bool
            Toggle state of the button.

        '''
    # Exit function if sender is invalid (safety)
        if not isinstance(sender := self.sender(), CW.StyledButton):
            return
        element = sender.text()

    # Add element to input features list if button is toggled on
        if toggled:
            self.feature_list.addItem(element)

    # Remove element from input features list if button is toggled off
        else:
            match_flag = QC.Qt.MatchFlag.MatchExactly
            item = self.feature_list.findItems(element, match_flag)[0]
            row = self.feature_list.row(item)
            self.feature_list.takeItem(row)


    def addCustomFeature(self) -> None:
        '''
        Add a new user-typed feature to input features list.

        '''
    # Retrieve the feature name and then clear the entry widget to free it for
    # a new entry to be written
        name = self.custom_entry.text()
        self.custom_entry.clear()

    # Do nothing if custom feature name is empty
        if name == '':
            return
        
    # Do nothing if name is already present in the features list
        if len(self.feature_list.findItems(name, QC.Qt.MatchFlag.MatchExactly)):
            return

    # If name is a valid chemical element, toggle on the linked button.
    # Otherwise, just add a new item.
        if name in (keys := self._elements.keys()):
            self.elements_btns[list(keys).index(name)].setChecked(True)
        else:
            self.feature_list.addItem(name)


    def removeSelectedFeatures(self) -> None:
        '''
        Remove selected features from the input features list.

        '''
    # Do nothing if no feature is selected
        selected = self.feature_list.selectedItems()
        if not len(selected):
            return
    
    # If feature name is a valid chemical element, toggle off the linked button
        for item in selected:
            if (name := item.text()) in (keys := self.elements.keys()):
                self.elements_btns[list(keys).index(name)].setChecked(False)

    # Remove features
        self.feature_list.removeSelected()


    def refreshDesigner(self) -> None:
        '''
        Confirm the selected input features and populate the Dataset Designer.

        '''
    # Deny refreshing if no input feature has been added
        if not self.feature_list.count():
            CW.MsgBox(self, 'Crit', 'Feature list cannot be empty.')
            return
        
        text = 'Refresh Dataset Designer? Any progress will be lost.'
        choice = CW.MsgBox(self, 'QuestWarn', text)
        if choice.yes():
            features = [item.text() for item in self.feature_list.getItems()]
            self.designer.refresh(features)
            self.designer_tbar.setEnabled(True)


    def removeSelectedRows(self) -> None:
        '''
        Remove selected rows from the Dataset Designer.
        
        '''
        selected_rows = {idx.row() for idx in self.designer.selectedIndexes()}
        for row in sorted(selected_rows, reverse=True):
            self.designer.delRow(row)


    def compileDataset(self) -> None:
        '''
        Extract data from the loaded samples and compile a new ground truth
        dataset.

        '''
        samples_data: list[dtools.GroundTruthDataset] = []
        nrows = self.designer.rowCount()
        nfeat = self.designer.columnCount() - 3 # -fillrow, -separator, -minmap

    # Deny compiling dataset if the Dataset Designer is empty
        if nrows == 0:
            CW.MsgBox(self, 'Crit', 'Dataset Designer is empty.')
            return

        self.designer_pbar.setRange(0, nrows)
        for row in range(nrows): # iterate through each row (= sample)
            inmaps, minmap = self.designer.getRowData(row)
        
        # Deny compiling dataset if maps are missing or invalid
            if len(inmaps) < nfeat or minmap is None:
                self.designer_pbar.reset()
                CW.MsgBox(self, 'Crit', f'Invalid maps in row {row+1}.')
                return
            
        # Deny compiling dataset if maps have different shapes in same sample
            shapes = [map_.shape for map_ in inmaps + [minmap]]
            if len(set(shapes)) > 1:
                common_shape = cf.most_frequent(shapes)
                map_fit = [shp == common_shape for shp in shapes]

                # Set warning status to loader widgets with unfitting maps
                for n, is_fitting in enumerate(map_fit, start=1):
                    if not is_fitting:
                        # +1 skips separator (needed for last map: mineral map)
                        col = n + 1 if n == len(map_fit) else n
                        loader = self.designer.get_file_loader(row, col)
                        if loader is not None:
                            loader.setStatus('Warning')

                # Send informative error message
                self.designer_pbar.reset()
                txt = (
                    f'A total of {map_fit.count(False)} maps do not fit the '
                    f'shape of sample in row {row + 1}.\n'
                    'Tip: they are highlighted in yellow.'
                )
                CW.MsgBox(self, 'Crit', txt)
                return
        
        # Construct and store a new GroundTruthDataset
            instack = InputMapStack(inmaps)
            columns = self.designer.columns[1:-2] + ['Class']
            gtd = dtools.GroundTruthDataset.from_maps(instack, minmap, columns)
            samples_data.append(gtd)
            self.designer_pbar.setValue(row + 1)

    # Create a single dataset and remove data with the protected '_ND_' class
        self.set_dataset(dtools.GroundTruthDataset.from_existent(samples_data))
        self.dataset.remove_where(-1, ('_ND_', ))

    # Conclude dataset compilation operations
        self.updateUniqueClasses()
        self.updatePreview()
        self.designer_pbar.reset()
        CW.MsgBox(self, 'Info', 'Dataset successfully compiled.')

 
    def updateUniqueClasses(self) -> None:
        '''
        Update the list of unique dataset classes in the dataset refiner list.

        '''
        if self.has_dataset():
            self.refine_list.clear()
            unique_classes = [str(c) for c in self.dataset.column_unique(-1)]
            self.refine_list.addItems(unique_classes)


    def updatePreview(self) -> None:
        '''
        Update the dataset preview.

        '''
        if self.has_dataset():
            df = repr((dataset := self.dataset).dataframe)
            class_count = dataset.column_count(-1)
            cnt = '\n'.join(f'{k} = {v}' for k, v in class_count.items())
            t = f'DATAFRAME PREVIEW\n\n{df}\n\n\nPER-CLASS DATA COUNT\n\n{cnt}'
            self.preview.setText(t)


    def renameClass(self) -> None:
        '''
        Rename a class in the dataset.

        '''
    # Exit function if no classes are selected
        selected = self.refine_list.selectedItems()
        if (n_selected := len(selected)) == 0:
            return
        
    # Deny renaming if more than one class is selected
        elif n_selected > 1:
            CW.MsgBox(self, 'Warn', 'Rename one class at a time.')
            return
    
    # Exit function if the dialog is canceled or the class is not renamed
        old_name = selected[0].text()
        label = 'Rename class (max. 8 ASCII characters):'
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), label,
                                           text=old_name)
        if not ok or name == old_name:
            return
        
    # Deny renaming to protected '_ND_' class
        if name == '_ND_':
            CW.MsgBox(self, 'Crit', '"_ND_" is a protected class name.')
            return
        
    # Deny renaming if the name already exists
        if name in (dataset := self.dataset).column_unique(-1):
            CW.MsgBox(self, 'Crit', f'{name} is already taken.')
            return

    # Deny renaming if the new name is not an ASCII <= 8 characters string
        if not 0 < len(name) <= 8 or not name.isascii():
            CW.MsgBox(self, 'Crit', 'Invalid name.')
            return

    # If we are here, the new name is valid: conclude renaming operations
        pbar = CW.PopUpProgBar(self, 2, 'Editing dataset')
        dataset.rename_target(old_name, name)
        pbar.increase()
        self.updateUniqueClasses()
        self.updatePreview()
        pbar.increase()


    def deleteClass(self) -> None:
        '''
        Remove rows of data with the selected classes from the dataset.

        '''
    # Exit function if no data is selected
        if not len(selected := self.refine_list.selectedItems()):
            return

        msgbox = CW.MsgBox(self, 'Quest', 'Remove selected classes?')
        if msgbox.is_choice('Yes'):
            targets = [item.text() for item in selected]
            pbar = CW.PopUpProgBar(self, 2, 'Editing dataset')
            self.dataset.remove_where(-1, targets)
            pbar.increase()
            self.updateUniqueClasses()
            self.updatePreview()
            pbar.increase()


    def mergeClass(self) -> None:
        '''
        Unify two or more classes in the dataset under a new name.

        '''
    # Exit function if no classes are selected
        selected = self.refine_list.selectedItems()
        if (n_selected := len(selected)) == 0:
            return
        
    # Deny merging if only one class is selected
        if n_selected == 1:
            CW.MsgBox(self, 'Warn', 'Select at least two classes.')
            return
        
    # Exit function if the dialog is canceled
        targets = [item.text() for item in selected]
        text = f'Merge {targets} in a new class (max. 8 ASCII characters):'
        name, ok = QW.QInputDialog.getText(self, self.windowTitle(), text)
        if not ok:
            return
        
    # Deny renaming to protected '_ND_' class
        if name == '_ND_':
            CW.MsgBox(self, 'Crit', '"_ND_" is a protected class name.')
            return
    
    # Deny renaming if the name already exists (excluding selected classes)
        dataset = self.dataset
        if name not in targets and name in dataset.column_unique(-1):
            CW.MsgBox(self, 'Crit', f'{name} is already taken.')
            return

    # Deny renaming if the new name is not an ASCII <= 8 characters string
        if not 0 < len(name) <= 8 and not name.isascii():
            CW.MsgBox(self, 'Crit', 'Invalid name.')
            return
    
    # If we are here, the name is valid. Conclude merging operations
        pbar = CW.PopUpProgBar(self, 2, 'Editing dataset')
        dataset.merge_targets(targets, name)
        pbar.increase()
        self.updateUniqueClasses()
        self.updatePreview()
        pbar.increase()


    def saveDataset(self) -> None:
        '''
        Save the ground truth dataset as one or multiple CSV file(s).

        '''
    # Deny saving dataset if no dataset is compiled
        if not self.has_dataset():
            CW.MsgBox(self, 'Crit', 'No dataset compiled.')
            return
        
    # Exit function if outpath is invalid or file dialog is canceled
        ftype = 'CSV (*.csv)'
        path = CW.FileDialog(self, 'save', 'Save Dataset', ftype).get_path()
        if not path:
            return

        pbar = CW.PulsePopUpProgBar(self, label='Saving dataset')
        pbar.startPulse()
        dec = self.decimal_combox.currentText()
        sep = self.separator_combox.currentText()
        self.dataset.save(path, sep, dec, split=self.split_cbox.isChecked())
        pbar.stopPulse()
        CW.MsgBox(self, 'Info', 'Dataset succesfully saved.')



class ModelLearner(DraggableTool):

    # Balancing strategies
    BalancingStrategyName = Literal[
        'Current',
        'Min',
        'Max',
        'Mean',
        'Median',
        'Custom value',
        'Custom multi-value'
    ]
    _balancing_strategies = get_args(BalancingStrategyName)

    # Available over-sampling and under-sampling algorithms
    _os_list = (
        'None',
        'SMOTE',
        'BorderlineSMOTE',
        'ADASYN'
    )
    _us_list = (
        'None',
        'RandUS',
        'NearMiss',
        'TomekLinks',
        'ENN-all', 
        'ENN-mode',
        'NCR'
    )

    def __init__(self) -> None:
        '''
        One of the main tools of X-Min Learn, that allows the developing of
        eager supervised machine learning models.

        '''
        super().__init__()

    # Set widget attributes
        self.setWindowTitle('Model Learner')
        self.setWindowIcon(style.getIcon('MODEL_LEARNER'))

    # Set main attributes
        # Ground-truth dataset
        self._dataset: dtools.GroundTruthDataset | None = None
        self.dataset_reader = dtools.CsvChunkReader(QC.QLocale().decimalPoint())

        # Train set balancing operation tracker
        self.balancing_info: list[dtools.BalancingInfo] = []

        # External threads
        self.balancing_thread = threads.BalancingThread()
        self.balancing_thread.setObjectName('Balancing')
        self.learning_thread = threads.LearningThread()
        self.learning_thread.setObjectName('Learning')

        # Machine learning model, network architecture and optimizer
        self._model: mltools.EagerModel | None = None 
        self.network: mltools.NeuralNetwork | None = None 
        self.optimizer: mltools.Optimizer | None = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None: 
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                       RANDOM SEED GENERATOR WIDGET
#  -------------------------------------------------------------------------  #
    # Random seed generator group (Group Area)
        self.seed_generator = CW.RandomSeedGenerator()
        seed_group = CW.GroupArea(self.seed_generator, 'Random seed generator')

#  -------------------------------------------------------------------------  #
#                       GROUND TRUTH DATASET WIDGETS 
#  -------------------------------------------------------------------------  #
    # Load dataset button (Styled Button)
        self.load_dataset_btn = CW.StyledButton(
            style.getIcon('IMPORT'), 'Load dataset')

    # CSV decimal character selector
        self.csv_dec_selector = CW.DecimalPointSelector()

    # Loaded dataset path (Path Label)
        self.dataset_path_lbl = CW.PathLabel(
            full_display=False, placeholder='No dataset loaded')

    # Loaded dataset preview area (Document Browser)
        self.dataset_preview = CW.DocumentBrowser(tbar=False)

    # Ground truth dataset group (Collapsible Area)
        dataset_grid = QW.QGridLayout()
        dataset_grid.setHorizontalSpacing(20)
        dataset_grid.setVerticalSpacing(10)
        dataset_grid.addWidget(self.load_dataset_btn, 0, 0, 1, 2)
        dataset_grid.addWidget(QW.QLabel('CSV decimal point'), 0, 2)
        dataset_grid.addWidget(self.csv_dec_selector, 0, 3)
        dataset_grid.addWidget(self.dataset_path_lbl, 1, 0, 1, -1)
        dataset_grid.addWidget(self.dataset_preview, 2, 0, 1, -1)
        dataset_group = CW.CollapsibleArea(
            dataset_grid, collapsed=False, title='Ground truth dataset')
        
#  -------------------------------------------------------------------------  #
#                            PARENT MODEL WIDGETS 
#  -------------------------------------------------------------------------  #

    # Load parent model button (Styled Button)
        self.load_pmodel_btn = CW.StyledButton(
            style.getIcon('IMPORT'), 'Load model')
        self.load_pmodel_btn.setEnabled(False)

    # Remove loaded parent model button (Styled Button)
        self.unload_pmodel_btn = CW.StyledButton(
            style.getIcon('CLEAR'), 'Remove model')

    # Parent model path (Path Label)
        self.pmodel_path = CW.PathLabel(
            full_display=False, placeholder='No model loaded')
        
    # Parent model preview (Document Browser)
        self.pmodel_preview = CW.DocumentBrowser()

    # Parent model group (Collapsible Area)
        pmodel_grid = QW.QGridLayout()
        pmodel_grid.setHorizontalSpacing(20)
        pmodel_grid.setVerticalSpacing(10)
        pmodel_grid.addWidget(self.load_pmodel_btn, 0, 0, 1, 1)
        pmodel_grid.addWidget(self.unload_pmodel_btn, 0, 1, 1, 1)
        pmodel_grid.addWidget(self.pmodel_path, 1, 0, 1, -1)
        pmodel_grid.addWidget(self.pmodel_preview, 2, 0, 1, -1)
        pmodel_group = CW.CollapsibleArea(
            pmodel_grid, title='Update existent model')

#  -------------------------------------------------------------------------  #
#                        HYPERPARAMETERS INPUT WIDGETS 
#  -------------------------------------------------------------------------  #

    # Learning rate input (Styled Double Spin Box)
        self.lr_spbox = CW.StyledDoubleSpinBox(
            min_value=0.00001, max_value=10, step=0.01, decimals=5)
        self.lr_spbox.setValue(0.01)

    # Weight decay input (Styled Double Spin Box)
        self.wd_spbox = CW.StyledDoubleSpinBox(
            max_value=10, step=0.01, decimals=5)
        self.wd_spbox.setValue(0)

    # Momentum input (Styled Double Spin Box)
        self.mtm_spbox = CW.StyledDoubleSpinBox(
            max_value=10, step=0.01, decimals=5)
        self.mtm_spbox.setValue(0)

    # Epochs input (Styled Spin Box)
        self.epochs_spbox = CW.StyledSpinBox(max_value=10**8)
        self.epochs_spbox.setValue(100)

    # Batch size input (editable Styled Combo Box)
        combox_tooltip = 'A Batch Size of 0 means a single batch'
        default_batch_sizes = [str(2 ** n) for n in range(5, 13)]
        self.batch_combox = CW.StyledComboBox(tooltip=combox_tooltip)
        self.batch_combox.addItem('0')
        self.batch_combox.addItems(default_batch_sizes)
        self.batch_combox.setCurrentIndex(0)
        self.batch_combox.setEditable(True)
        self.batch_combox.setValidator(QG.QIntValidator(0, 2**24))
        self.batch_combox.setInsertPolicy(QW.QComboBox.InsertPolicy.NoInsert)

    # Hyperparameters group (Collapsible Area)
        hparam_form = QW.QFormLayout()
        hparam_form.setSpacing(10)
        hparam_form.addRow('Learning Rate', self.lr_spbox)
        hparam_form.addRow('Weight Decay', self.wd_spbox)
        hparam_form.addRow('Momentum', self.mtm_spbox)
        hparam_form.addRow('Epochs', self.epochs_spbox)
        hparam_form.addRow('Batch Size', self.batch_combox)
        hparam_group = CW.CollapsibleArea(hparam_form, title='Hyperparameters')
        
#  -------------------------------------------------------------------------  #
#                      LEARNING PREFERENCES INPUT WIDGETS 
#  -------------------------------------------------------------------------  #

    # Feature mapping (Check Box)
        self.feat_mapping_cbox = QW.QCheckBox('Polynomial mapping')
        self.feat_mapping_cbox.setToolTip('Enable polynomial feature mapping')

    # Polynomial mapping degree (Styled Spin Box)
        self.poly_deg_spbox = CW.StyledSpinBox(min_value=2, max_value=5)
        self.poly_deg_spbox.setValue(2)
        self.poly_deg_spbox.setEnabled(False)
        self.poly_deg_spbox.setToolTip('Polynomial degree')

    # Algorithm (Styled Combo Box)
        algm_list = ('Softmax Regression',)
        self.algm_combox = CW.StyledComboBox()
        self.algm_combox.addItems(algm_list)

    # Optimization function (Styled Combo Box)
        optim_list = ('SGD',)
        self.optim_combox = CW.StyledComboBox()
        self.optim_combox.addItems(optim_list)

    # Learning graphic update rate (Line Edit)
        self.plots_update_rate = QW.QLineEdit()
        self.plots_update_rate.setValidator(QG.QIntValidator(1, 10**8))
        self.plots_update_rate.setText('10')
        self.plots_update_rate.setStyleSheet(style.SS_MENU)

    # Number of workers (Styled Spin Box)
        workers_tip = 'Parallel computations. Used only if Batch Size > 0'
        max_workers = mltools.num_cores() // 2 # safety
        self.workers_spbox = CW.StyledSpinBox(max_value=max_workers)
        self.workers_spbox.setValue(0)
        self.workers_spbox.setToolTip(workers_tip)

    # Use GPU acceleration (Check Box)
        self.cuda_cbox = QW.QCheckBox('Use GPU acceleration')
        self.cuda_cbox.setChecked(mltools.cuda_available())
        self.cuda_cbox.setEnabled(mltools.cuda_available())

    # Learning preferences group (Collapsible Area)
        pref_form = QW.QFormLayout()
        pref_form.setSpacing(10)
        pref_form.addRow(self.feat_mapping_cbox, self.poly_deg_spbox)
        pref_form.addRow('Algorithm', self.algm_combox)
        pref_form.addRow('Optimization function', self.optim_combox)
        pref_form.addRow('Graphics refresh rate', self.plots_update_rate)
        pref_form.addRow('Number of workers', self.workers_spbox)
        pref_form.addRow(self.cuda_cbox)
        pref_group = CW.CollapsibleArea(pref_form, title='Learning preferences')

#  -------------------------------------------------------------------------  #
#                   START, STOP, TEST, SAVE, PROGBAR WIDGETS 
#  -------------------------------------------------------------------------  #

    # Start learning button (Styled Button)
        self.start_learn_btn = CW.StyledButton(text='LEARN', bg=style.OK_GREEN)
        self.start_learn_btn.setToolTip('Start learning session')
        self.start_learn_btn.setEnabled(False)

    # Stop learning button (Styled Button)
        self.stop_learn_btn = CW.StyledButton(text='STOP', bg=style.BAD_RED)
        self.stop_learn_btn.setToolTip('Stop learning session')
        self.stop_learn_btn.setEnabled(False)

    # Test model button (Styled Button)
        self.test_model_btn = CW.StyledButton(
            style.getIcon('TEST'), 'TEST MODEL')
        self.test_model_btn.setEnabled(False)

    # Save model button (Styled Button)
        self.save_model_btn = CW.StyledButton(
            style.getIcon('SAVE'), 'SAVE MODEL')
        self.save_model_btn.setEnabled(False)

    # Progress bar (Progress Bar)
        self.learning_pbar = QW.QProgressBar()

#  -------------------------------------------------------------------------  #
#                       TRAIN, VALIDATION, TEST WIDGETS 
#  -------------------------------------------------------------------------  #

    # Train set ratio selector (Styled Spin Box)
        self.train_ratio_spbox = CW.StyledSpinBox(1, 98)
        self.train_ratio_spbox.setSuffix(' %')
        self.train_ratio_spbox.setValue(50)

    # Validation set ratio selector (Styled Spin Box)
        self.valid_ratio_spbox = CW.StyledSpinBox(1, 98)
        self.valid_ratio_spbox.setSuffix(' %')
        self.valid_ratio_spbox.setValue(25)

    # Test set ratio selector (Styled Spin Box)
        self.test_ratio_spbox = CW.StyledSpinBox(1, 98)
        self.test_ratio_spbox.setSuffix(' %')
        self.test_ratio_spbox.setValue(25)

    # Split ground truth dataset button (Styled Button)
        split_tip = 'Split dataset into train, validation and test sets'
        self.split_dataset_btn = CW.StyledButton(
            text='SPLIT', bg=style.OK_GREEN)
        self.split_dataset_btn.setToolTip(split_tip)
        self.split_dataset_btn.setEnabled(False)

    # Train, Validation and Test sets PieChart (Pie Canvas)
        self.subsets_pie = plots.PieCanvas(wheel_pan=False, wheel_zoom=False)
        self.subsets_pie.fig.set_facecolor(style.CASPER_LIGHT)
        self.subsets_pie.setMinimumSize(200, 200)

    # Train, Validation & Test sets barchart (Bar Canvas)
        self.subsets_barplot = plots.BarCanvas(
            wheel_pan=False, wheel_zoom=False)
        self.subsets_barplot.fig.set_facecolor(style.CASPER_LIGHT)
        self.subsets_barplot.setMinimumSize(200, 300)

    # Train set class visualizer (Styled List Widget)
        self.train_class_list = CW.StyledListWidget(ext_sel=False)
        self.train_class_list.setMinimumWidth(100)

    # Train set per-class current and total counter labels (Framed Labels)
        self.train_curr_count_lbl = CW.FramedLabel('')
        self.train_tot_count_lbl = CW.FramedLabel('')

    # Validation set class visualizer (Styled List Widget)
        self.valid_class_list = CW.StyledListWidget(ext_sel=False)
        self.valid_class_list.setMinimumWidth(100)

    # Validation set per-class current and total counter labels (Framed Labels)
        self.valid_curr_count_lbl = CW.FramedLabel('')
        self.valid_tot_count_lbl = CW.FramedLabel('')

    # Test set class visualizer (Styled List Widget)
        self.test_class_list = CW.StyledListWidget(ext_sel=False)
        self.test_class_list.setMinimumWidth(100)

    # Test set per-class current and total counter labels (Framed Labels)
        self.test_curr_count_lbl = CW.FramedLabel('')
        self.test_tot_count_lbl = CW.FramedLabel('')

    # Split Train, validation, test subset widgets group (Group Area)
        counters_grid = QW.QGridLayout()
        counters_grid.setContentsMargins(20, 5, 0, 0)
        counters_grid.setHorizontalSpacing(20)
        counters_grid.setVerticalSpacing(5)
        counters_grid.addWidget(QW.QLabel('Train'), 0, 0, style.A_HCENTER)
        counters_grid.addWidget(QW.QLabel('Validation'), 0, 1, style.A_HCENTER)
        counters_grid.addWidget(QW.QLabel('Test'), 0, 2, style.A_HCENTER)
        counters_grid.addWidget(self.train_class_list, 1, 0)
        counters_grid.addWidget(self.valid_class_list, 1, 1)
        counters_grid.addWidget(self.test_class_list, 1, 2)
        counters_grid.addWidget(self.train_curr_count_lbl, 2, 0)
        counters_grid.addWidget(self.valid_curr_count_lbl, 2, 1)
        counters_grid.addWidget(self.test_curr_count_lbl, 2, 2)
        counters_grid.addWidget(self.train_tot_count_lbl, 3, 0)
        counters_grid.addWidget(self.valid_tot_count_lbl, 3, 1)
        counters_grid.addWidget(self.test_tot_count_lbl, 3, 2)
        counters_grid.setRowStretch(1, -1)

        split_grid = QW.QGridLayout()
        split_grid.setSpacing(20)
        split_grid.addWidget(QW.QLabel('Train set'), 0, 0)
        split_grid.addWidget(self.train_ratio_spbox, 0, 1)
        split_grid.addWidget(QW.QLabel('Validation set'), 1, 0)
        split_grid.addWidget(self.valid_ratio_spbox, 1, 1)
        split_grid.addWidget(QW.QLabel('Test set'), 2, 0)
        split_grid.addWidget(self.test_ratio_spbox, 2, 1)
        split_grid.addWidget(self.split_dataset_btn, 3, 0, 1, 2)
        split_grid.addWidget(self.subsets_pie, 0, 2, 4, 1)
        split_grid.addWidget(self.subsets_barplot, 4, 0, 1, 3)
        split_grid.addLayout(counters_grid, 0, 3, -1, 1)
        split_grid.setColumnStretch(3, -1)
        split_grid.setRowStretch(4, -1)
        split_group = CW.GroupArea(split_grid, 'Split dataset')

#  -------------------------------------------------------------------------  #
#                       BALANCING OPERATIONS WIDGETS 
#  -------------------------------------------------------------------------  #

    # Balancing help button (Styled Button)
        self.balancing_help_btn = CW.StyledButton(style.getIcon('INFO'))
        self.balancing_help_btn.setMaximumSize(30, 30)
        self.balancing_help_btn.setFlat(True)
        self.balancing_help_btn.setToolTip('More info about dataset balancing')

    # Oversampling strategy warning icon (Label)
        os_warn = QG.QPixmap(str(style.ICONS.get('WARNING')))
        self.os_warn_icon = QW.QLabel()
        self.os_warn_icon.setPixmap(os_warn.scaled(16, 16, style.ARM_KEEP))
        self.os_warn_icon.setToolTip('Sample count may differ from requested')
        self.os_warn_icon.hide()

    # Oversampling algorithm selector (Styled Combo Box)
        self.os_combox = CW.StyledComboBox()
        self.os_combox.addItems(self._os_list)

    # Oversampling K-neighbours selector (Styled Spin Box)
        kn_tip = 'Nearest neighbours to construct synthetic samples'
        self.k_neigh_spbox = CW.StyledSpinBox()
        self.k_neigh_spbox.setValue(5)
        self.k_neigh_spbox.setToolTip(kn_tip)

    # Oversampling M-neighbours selector (Styled Spin Box)
        mn_tip = 'Nearest neighbours to determine if a minority sample is in danger'
        self.m_neigh_spbox = CW.StyledSpinBox()
        self.m_neigh_spbox.setValue(10)
        self.m_neigh_spbox.setToolTip(mn_tip)
        
    # Undersampling strategy warning icon (Label)
        us_warn = QG.QPixmap(str(style.ICONS.get('WARNING')))
        self.us_warn_icon = QW.QLabel()
        self.us_warn_icon.setPixmap(us_warn.scaled(16, 16, style.ARM_KEEP))
        self.us_warn_icon.setToolTip('Sample count may differ from requested')
        self.us_warn_icon.hide()

    # Undersampling algorithm selector (Styled Combo Box)
        self.us_combox = CW.StyledComboBox()
        self.us_combox.addItems(self._us_list)

    # Undersampling N-neighbours selector (Styled Spin Box)
        self.n_neigh_spbox = CW.StyledSpinBox()
        self.n_neigh_spbox.setValue(3)
        self.n_neigh_spbox.setToolTip('Size of the neighbourhood to consider')

    # Strategy selector (Styled Combo Box)
        self.strategy_combox = CW.StyledComboBox()
        self.strategy_combox.addItems(self._balancing_strategies)

    # Strategy value (Line Edit)
        self.strategy_value = QW.QLineEdit()
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8}|1000000000)$") # 1 - 10^9
        self.strategy_value.setValidator(QG.QRegularExpressionValidator(regex))
        self.strategy_value.setStyleSheet(style.SS_MENU)

    # Strategy percentage (Line Edit)
        self.strategy_percent = QW.QLineEdit()
        regex = QC.QRegularExpression(r"^(?:[1-9]|\d{2,3}|1000)$") # 1 - 1000
        self.strategy_percent.setValidator(QG.QRegularExpressionValidator(regex))
        self.strategy_percent.setText('100')
        self.strategy_percent.setStyleSheet(style.SS_MENU)

    # Use parallel computation (Check Box)
        self.balancing_multicore_cbox = QW.QCheckBox('Parallel computation')
        self.balancing_multicore_cbox.setToolTip(
            'Distribute computation across multiple processes')
        self.balancing_multicore_cbox.setChecked(False)

    # Balancing preview table (Styled Table)
        self.balancing_table = CW.StyledTable(0, 4)
        header_resize_mode = style.HRM_RESIZETOCONTENTS
        self.balancing_table.h_header.setSectionResizeMode(header_resize_mode)
        self.balancing_table.h_header.setStretchLastSection(True)
        self.balancing_table.v_header.setSectionResizeMode(header_resize_mode)
        h_headers_labels = ('Class', 'Original', 'Current', 'After balancing')
        self.balancing_table.setHorizontalHeaderLabels(h_headers_labels)

    # Start balancing button (Styled Button)
        self.start_balancing_btn = CW.StyledButton(
            text='Start', bg=style.OK_GREEN)
        self.start_balancing_btn.setToolTip('Start balancing session')

    # Stop balancing button (Styled Button)
        self.stop_balancing_btn = CW.StyledButton(
            text='Stop', bg=style.BAD_RED)
        self.stop_balancing_btn.setToolTip('Stop balancing session')

    # Cancel balancing button (Styled Button)
        self.canc_balancing_btn = CW.StyledButton(
            style.getIcon('CLEAR'), 'Cancel')
        self.canc_balancing_btn.setToolTip('Cancel balancing results')

    # Balancing operations progress bar (Descriptive Progress Bar)
        self.balancing_pbar = CW.DescriptiveProgressBar()
        self.balancing_pbar.setMaximum(5)

    # Balancing group (Group Area)
        balancing_grid = QW.QGridLayout()
        balancing_grid.setSpacing(10)
        balancing_grid.addWidget(self.balancing_help_btn, 0, 0, style.A_LEFT)
        balancing_grid.addWidget(QW.QLabel('Over-Sampling'), 1, 0, 1, 2)
        balancing_grid.addWidget(self.os_warn_icon, 1, 2, style.A_RIGHT)
        balancing_grid.addWidget(CW.LineSeparator(), 2, 0, 1, 3)
        balancing_grid.addWidget(self.os_combox, 3, 0)
        balancing_grid.addWidget(QW.QLabel('K neigh.'), 3, 1, style.A_RIGHT)
        balancing_grid.addWidget(self.k_neigh_spbox, 3, 2)
        balancing_grid.addWidget(QW.QLabel('M neigh.'), 4, 1, style.A_RIGHT)
        balancing_grid.addWidget(self.m_neigh_spbox, 4, 2)
        balancing_grid.addWidget(QW.QLabel('Under-Sampling'), 6, 0, 1, 2)
        balancing_grid.addWidget(self.us_warn_icon, 6, 2, style.A_RIGHT)
        balancing_grid.addWidget(CW.LineSeparator(), 7, 0, 1, 3)
        balancing_grid.addWidget(self.us_combox, 8, 0)
        balancing_grid.addWidget(QW.QLabel('N neigh.'), 8, 1, style.A_RIGHT)
        balancing_grid.addWidget(self.n_neigh_spbox, 8, 2)
        balancing_grid.addWidget(QW.QLabel('Strategy'), 10, 0, 1, 3)
        balancing_grid.addWidget(CW.LineSeparator(), 11, 0, 1, 3)
        balancing_grid.addWidget(self.strategy_combox, 12, 0, 1, 3)
        balancing_grid.addWidget(QW.QLabel('Unique value'), 13, 0)
        balancing_grid.addWidget(self.strategy_value, 13, 1, 1, 2)
        balancing_grid.addWidget(QW.QLabel('Unique percentage'), 14, 0)
        balancing_grid.addWidget(self.strategy_percent, 14, 1, 1, 2)
        balancing_grid.addWidget(self.balancing_multicore_cbox, 15, 0, 1, 3)
        balancing_grid.addWidget(self.balancing_pbar, 16, 0, 1, 3)
        balancing_grid.addWidget(self.canc_balancing_btn, 17, 0)
        balancing_grid.addWidget(self.start_balancing_btn, 17, 1)
        balancing_grid.addWidget(self.stop_balancing_btn, 17, 2)
        balancing_grid.setColumnMinimumWidth(3, 20)
        balancing_grid.addWidget(self.balancing_table, 0, 4, -1, 1)
        balancing_grid.setColumnStretch(3, -1)

        self.balancing_group = CW.GroupArea(
            balancing_grid, checkable=True, title='Balance train set')
        self.balancing_group.setChecked(False)
        self.balancing_group.setEnabled(False)

#  -------------------------------------------------------------------------  #
#                         LEARNING PROGRESSION PLOTS 
#  -------------------------------------------------------------------------  #

    # Loss curves canvas (Curve Canvas)
        self.loss_plot = plots.CurveCanvas(
            'Loss plot',
            'Epochs',
            'Loss', 
            layout='tight',
            wheel_pan=False,
            wheel_zoom=False
        )
        self.loss_plot.setMinimumSize(500, 400)

    # Loss curves Navigation Toolbar (Navigation Toolbar)
        self.loss_navtbar = plots.NavTbar(self.loss_plot, self)
        self.loss_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.loss_navtbar.fixHomeAction()

    # Loss info labels (Framed Labels)
        self.train_loss_lbl = CW.FramedLabel('')
        self.valid_loss_lbl = CW.FramedLabel('')

    # Accuracy curves canvas (Curve Canvas)
        self.accuracy_plot = plots.CurveCanvas(
            'Accuracy plot',
            'Epochs',
            'Accuracy',
            layout='tight',
            wheel_pan=False,
            wheel_zoom=False
        )
        self.accuracy_plot.setMinimumSize(500, 400)

    # Accuracy curves Navigation Toolbar (Navigation Toolbar)
        self.accuracy_navtbar = plots.NavTbar(self.accuracy_plot, self)
        self.accuracy_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.accuracy_navtbar.fixHomeAction()

    # Accuracy info labels (Framed Labels)
        self.train_accuracy_lbl = CW.FramedLabel('')
        self.valid_accuracy_lbl = CW.FramedLabel('')

    # Learning progression plots group (Group Area)
        loss_grid = QW.QGridLayout()
        loss_grid.setSpacing(15)
        loss_grid.addWidget(self.loss_navtbar, 0, 0, 1, -1)
        loss_grid.addWidget(self.loss_plot, 1, 0, 1, -1)
        loss_grid.addWidget(QW.QLabel('Train'), 2, 0)
        loss_grid.addWidget(self.train_loss_lbl, 2, 1)
        loss_grid.addWidget(QW.QLabel('Validation'), 2, 2)
        loss_grid.addWidget(self.valid_loss_lbl, 2, 3)
        loss_grid.setColumnStretch(1, 1)
        loss_grid.setColumnStretch(3, 1)

        accuracy_grid = QW.QGridLayout()
        accuracy_grid.setSpacing(15)
        accuracy_grid.addWidget(self.accuracy_navtbar, 0, 0, 1, -1)
        accuracy_grid.addWidget(self.accuracy_plot, 1, 0, 1, -1)
        accuracy_grid.addWidget(QW.QLabel('Train'), 2, 0)
        accuracy_grid.addWidget(self.train_accuracy_lbl, 2, 1)
        accuracy_grid.addWidget(QW.QLabel('Validation'), 2, 2)
        accuracy_grid.addWidget(self.valid_accuracy_lbl, 2, 3)
        accuracy_grid.setColumnStretch(1, 1)
        accuracy_grid.setColumnStretch(3, 1)

        learn_plot_tabwid = CW.StyledTabWidget()
        learn_plot_tabwid.setTabBarExpanding()
        learn_plot_tabwid.wrap_tab(
            loss_grid, style.getIcon('LOSS'), 'LOSS PLOT')
        learn_plot_tabwid.wrap_tab(
            accuracy_grid, style.getIcon('ACCURACY'), 'ACCURACY PLOT')
        learn_group = CW.GroupArea(learn_plot_tabwid, 'Learning progression')

#  -------------------------------------------------------------------------  #
#                        CONFUSION MATRICES WIDGETS 
#  -------------------------------------------------------------------------  #

    # Train Confusion Matrix canvas (Confusion Matrix Canvas)
        self.train_confmat = plots.ConfMatCanvas(
            'Train set', layout='tight', wheel_pan=False, wheel_zoom=False)
        self.train_confmat.setMinimumSize(600, 600)

    # Train Confusion Matrix Navigation Toolbar (Navigation Toolbar)
        self.train_cm_navtbar = plots.NavTbar(
            self.train_confmat, self, coords=False)
        self.train_cm_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.train_cm_navtbar.fixHomeAction()

    # Annotation as percentage action [--> Train CM Navigation ToolBar]
        self.train_cm_perc_action = QW.QAction(
            style.getIcon('PERCENT'), 'Annotations as percentage')
        self.train_cm_perc_action.setCheckable(True)
        self.train_cm_perc_action.setChecked(True)
        self.train_cm_navtbar.insertAction(2, self.train_cm_perc_action)
        self.train_cm_navtbar.insertSeparator(2)

    # Train F1 scores labels (Framed Labels)
        self.train_f1_macro_lbl = CW.FramedLabel('')
        self.train_f1_weight_lbl = CW.FramedLabel('')

    # Validation Confusion Matrix area (Confusion Matrix Canvas)
        self.valid_confmat = plots.ConfMatCanvas(
            'Validation set', layout='tight', wheel_pan=False, wheel_zoom=False)
        self.valid_confmat.setMinimumSize(600, 600)

    # Validation Confusion Matrix Navigation Toolbar (Navigation Toolbar)
        self.valid_cm_navtbar = plots.NavTbar(
            self.valid_confmat, self, coords=False)
        self.valid_cm_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.valid_cm_navtbar.fixHomeAction()

    # Annotation as percentage action [--> Validation CM Navigation Toolbar]
        self.valid_cm_perc_action = QW.QAction(
            style.getIcon('PERCENT'), 'Annotations as percentage')
        self.valid_cm_perc_action.setCheckable(True)
        self.valid_cm_perc_action.setChecked(True)
        self.valid_cm_navtbar.insertAction(2, self.valid_cm_perc_action)
        self.valid_cm_navtbar.insertSeparator(2)

    # Validation F1 scores label (Framed Labels)
        self.valid_f1_macro_lbl = CW.FramedLabel('')
        self.valid_f1_weight_lbl = CW.FramedLabel('')

    # Test Confusion Matrix (Confusion Matrix Canvas)
        self.test_confmat = plots.ConfMatCanvas(
            'Test set', layout='tight', wheel_pan=False, wheel_zoom=False)
        self.test_confmat.setMinimumSize(600, 600)

    # Test Confusion Matrix Navigation Toolbar (Navigation Toolbar)
        self.test_cm_navtbar = plots.NavTbar(
            self.test_confmat, self, coords=False)
        self.test_cm_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.test_cm_navtbar.fixHomeAction()

    # Annotation as percentage action [--> Test CM Navigation Toolbar]
        self.test_cm_perc_action = QW.QAction(
            style.getIcon('PERCENT'), 'Annotations as percentage')
        self.test_cm_perc_action.setCheckable(True)
        self.test_cm_perc_action.setChecked(True)
        self.test_cm_navtbar.insertAction(2, self.test_cm_perc_action)
        self.test_cm_navtbar.insertSeparator(2)

    # Test score labels (Framed Labels)
        self.test_accuracy_lbl = CW.FramedLabel('')
        self.test_f1_macro_lbl = CW.FramedLabel('')
        self.test_f1_weight_lbl = CW.FramedLabel('')

    # Confusion matrix group (Group Area)
        tr_cm_grid = QW.QGridLayout()
        tr_cm_grid.setSpacing(15)
        tr_cm_grid.addWidget(self.train_cm_navtbar, 0, 0, 1, -1)
        tr_cm_grid.addWidget(self.train_confmat, 1, 0, 1, -1)
        tr_cm_grid.addWidget(QW.QLabel('F1 score - Average'), 2, 0)
        tr_cm_grid.addWidget(self.train_f1_macro_lbl, 2, 1)
        tr_cm_grid.addWidget(QW.QLabel('F1 score - Weighted'), 2, 2)
        tr_cm_grid.addWidget(self.train_f1_weight_lbl, 2, 3)
        tr_cm_grid.setColumnStretch(1, 1)
        tr_cm_grid.setColumnStretch(3, 1)

        vd_cm_grid = QW.QGridLayout()
        vd_cm_grid.setSpacing(15)
        vd_cm_grid.addWidget(self.valid_cm_navtbar, 0, 0, 1, -1)
        vd_cm_grid.addWidget(self.valid_confmat, 1, 0, 1, -1)
        vd_cm_grid.addWidget(QW.QLabel('F1 score - Average'), 2, 0)
        vd_cm_grid.addWidget(self.valid_f1_macro_lbl, 2, 1)
        vd_cm_grid.addWidget(QW.QLabel('F1 score - Weighted'), 2, 2)
        vd_cm_grid.addWidget(self.valid_f1_weight_lbl, 2, 3)
        vd_cm_grid.setColumnStretch(1, 1)
        vd_cm_grid.setColumnStretch(3, 1)

        ts_cm_grid = QW.QGridLayout()
        ts_cm_grid.setSpacing(15)
        ts_cm_grid.addWidget(self.test_cm_navtbar, 0, 0, 1, -1)
        ts_cm_grid.addWidget(self.test_confmat, 1, 0, 1, -1)
        ts_cm_grid.addWidget(QW.QLabel('Accuracy'), 2, 0)
        ts_cm_grid.addWidget(self.test_accuracy_lbl, 2, 1)
        ts_cm_grid.addWidget(QW.QLabel('F1 score - Average'), 2, 2)
        ts_cm_grid.addWidget(self.test_f1_macro_lbl, 2, 3)
        ts_cm_grid.addWidget(QW.QLabel('F1 score - Weighted'), 2, 4)
        ts_cm_grid.addWidget(self.test_f1_weight_lbl, 2, 5)
        ts_cm_grid.setColumnStretch(1, 1)
        ts_cm_grid.setColumnStretch(3, 1)
        ts_cm_grid.setColumnStretch(5, 1)

        confmat_tabwid = CW.StyledTabWidget()
        confmat_tabwid.setTabBarExpanding()
        confmat_tabwid.wrap_tab(
            tr_cm_grid, style.getIcon('TRAIN_SET'), 'TRAIN SET')
        confmat_tabwid.wrap_tab(
            vd_cm_grid, style.getIcon('VALIDATION_SET'), 'VALIDATION SET')
        confmat_tabwid.wrap_tab(
            ts_cm_grid, style.getIcon('TEST_SET'), 'TEST SET')
        confmat_group = CW.GroupArea(confmat_tabwid, 'Model evaluation')

#  -------------------------------------------------------------------------  #
#                              ADJUST LAYOUT 
#  -------------------------------------------------------------------------  #

    # Start-stop-test-save buttons and progress bar (Main controls) layout
        main_ctrl_grid = QW.QGridLayout()
        main_ctrl_grid.addWidget(self.learning_pbar, 0, 0, 1, -1)
        main_ctrl_grid.addWidget(self.start_learn_btn, 1, 0)
        main_ctrl_grid.addWidget(self.stop_learn_btn, 1, 1)
        main_ctrl_grid.addWidget(self.test_model_btn, 2, 0)
        main_ctrl_grid.addWidget(self.save_model_btn, 2, 1)

    # Adjust options panel layout (left Group Scroll Area)
        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(15)
        left_vbox.addWidget(seed_group)
        left_vbox.addWidget(dataset_group)
        left_vbox.addWidget(pmodel_group)
        left_vbox.addWidget(hparam_group)
        left_vbox.addWidget(pref_group)
        left_vbox.addStretch(1)
        left_vbox.addLayout(main_ctrl_grid)
        left_scroll = CW.GroupScrollArea(left_vbox)

    # Adjust learning panels layout (right Group Scroll Area)
        right_vbox = QW.QVBoxLayout()
        right_vbox.setSpacing(30)
        right_vbox.addWidget(split_group)
        right_vbox.addWidget(self.balancing_group)
        right_vbox.addWidget(learn_group)
        right_vbox.addWidget(confmat_group)
        right_scroll = CW.GroupScrollArea(right_vbox)

    # Main Layout
        main_layout = CW.SplitterLayout()
        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(right_scroll, 1)
        self.setLayout(main_layout)
        

    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Balancing thread signals
        self.balancing_thread.taskInitialized.connect(self.balancing_pbar.step)
        self.balancing_thread.workInterrupted.connect(self._endBalancingSession)
        self.balancing_thread.workFinished.connect(self._parseBalancingResult)

    # Learning thread signals
        self.learning_thread.iterCompleted.connect(self.learning_pbar.setValue)
        self.learning_thread.renderRequested.connect(self._drawLearningScores)
        self.learning_thread.taskFinished.connect(self._parseLearningResult)

    # Manage seed changes
        self.seed_generator.seedChanged.connect(self.onSeedChanged)

    # Ground truth dataset signals
        self.csv_dec_selector.currentTextChanged.connect(
            self.dataset_reader.set_decimal)
        self.load_dataset_btn.clicked.connect(self.loadGroundTruthDataset)

    # Load / unload existent model
        self.load_pmodel_btn.clicked.connect(self.loadParentModel)
        self.unload_pmodel_btn.clicked.connect(self.removeParentModel)

    # Update the learning plots refresh rate when the number of epochs changes
        self.epochs_spbox.valueChanged.connect(
            lambda value: self.plots_update_rate.setText(str(value // 10)))

    # Prevent empty values in batch size combobox
        self.batch_combox.editTextChanged.connect(
            lambda t: self.batch_combox.setCurrentIndex(0) if not t else None)

    # Enable poly degree spinbox when polynomial feature mapping is checked
        self.feat_mapping_cbox.stateChanged.connect(
            lambda state: self.poly_deg_spbox.setEnabled(state))
    
    # Start, stop, test, save model
        self.start_learn_btn.clicked.connect(self.startLearningSession)
        self.stop_learn_btn.clicked.connect(self.stopLearningSession)
        self.test_model_btn.clicked.connect(self.testModel)
        self.save_model_btn.clicked.connect(self.saveModel)
    
    # Adjust subsets ratios so that they always sum to 100 when one changes
        for spbox in (self.train_ratio_spbox, self.valid_ratio_spbox, 
                      self.test_ratio_spbox):
            spbox.valueChanged.connect(self._fixSubsetsRatios)

    # Split ground truth dataset into train, validation and test substes
        self.split_dataset_btn.clicked.connect(self.splitDataset)

    # Show number of class instances in subsets
        self.train_class_list.itemClicked.connect(self.countClassInstances) 
        self.valid_class_list.itemClicked.connect(self.countClassInstances) 
        self.test_class_list.itemClicked.connect(self.countClassInstances)

    # Balancing info help button links to imbalanced learn official website
        self.balancing_help_btn.clicked.connect(self.on_balancing_help_clicked)

    # Show/hide over-sampling warning icon based on selected OS algorithm
        self.os_combox.currentTextChanged.connect(
            lambda t: self.os_warn_icon.setHidden(t != 'ADASYN'))

    # Enable m-neighbors spinbox only when OS algorithm is BorderlineSMOTE
        self.os_combox.currentTextChanged.connect(
            lambda t: self.m_neigh_spbox.setEnabled(t == 'BorderlineSMOTE'))
    
    # Show/hide under-sampling warning icon based on selected US algorithm
        us_no_warn = ('None', 'RandUS', 'NearMiss')
        self.us_combox.currentTextChanged.connect(
            lambda t: self.us_warn_icon.setHidden(t in us_no_warn))
        
    # Enable n-neighbors spinbox when US algorithm is not in the following list
        us_no_nus = ('RandUS', 'TomekLinks')
        self.us_combox.currentTextChanged.connect(
            lambda t: self.n_neigh_spbox.setEnabled(t not in us_no_nus))
    
    # Set strategy based on the selection made in the strategy combobox 
        self.strategy_combox.currentTextChanged.connect(self.parseStrategy)

    # Set strategy to 'Custom value' if strategy value is user-modified
        self.strategy_value.textEdited.connect(
            lambda: self.strategy_combox.setCurrentText('Custom value'))
        
    # Set strategy to 'Custom multi-value' if strategy percent is user-modified
        self.strategy_percent.textEdited.connect(
            lambda: self.strategy_combox.setCurrentText('Custom multi-value'))

    # Update last column of balancing table if strategy value / percent changes
        self.strategy_value.textChanged.connect(self.fillAfterBalancingColumn)
        self.strategy_percent.textChanged.connect(self.fillAfterBalancingColumn)

    # Connect start-stop-cancel balancing buttons functions
        self.start_balancing_btn.clicked.connect(self.startBalancingSession)
        self.stop_balancing_btn.clicked.connect(self.stopBalancingSession)
        self.canc_balancing_btn.clicked.connect(self.cancelBalancingResults)
    
    # Toggle on/off percentages values on confusion matrices
        self.train_cm_perc_action.toggled.connect(self.toggleConfMatPercent) 
        self.valid_cm_perc_action.toggled.connect(self.toggleConfMatPercent) 
        self.test_cm_perc_action.toggled.connect(self.toggleConfMatPercent)


    @property
    def dataset(self) -> dtools.GroundTruthDataset:
        '''
        Get the currently loaded ground truth dataset.

        Returns
        -------
        dataset : dtools.GroundTruthDataset
            Current dataset.

        Raises
        ------
        TypeError
            Raised if the current dataset is None.

        '''
        if (dataset := self._dataset) is None:
            raise TypeError('Current dataset is None.')
        return dataset


    def set_dataset(self, dataset: dtools.GroundTruthDataset) -> None:
        '''
        Set a new ground truth dataset.

        Parameters
        ----------
        dataset : dtools.GroundTruthDataset
            New dataset.

        '''
        self._dataset = dataset


    def has_dataset(self) -> bool:
        '''
        Check if a ground truth dataset is currently loaded.

        Returns
        -------
        bool
            Whether a dataset exists or not.

        '''
        return self._dataset is not None


    @property
    def model(self) -> mltools.EagerModel:
        '''
        Get the current machine learning model.

        Returns
        -------
        model : mltools.EagerModel
            Current model.

        Raises
        ------
        TypeError
            Raised if the current model is None.

        '''
        if (model := self._model) is None:
            raise TypeError('Current model is None.')
        return model


    def set_model(self, model: mltools.EagerModel) -> None:
        '''
        Set a new machine learning model.

        Parameters
        ----------
        model : mltools.EagerModel
            New model.

        '''
        self._model = model


    def has_model(self) -> bool:
        '''
        Check if a machine learning model currently exists.

        Returns
        -------
        bool
            Whether a model exists or not.

        '''
        return self._model is not None


    def _reset(self) -> None:
        '''
        Restore the Model Learner to its original state. Only the loaded ground
        truth dataset and the random seed are preserved.

        '''
    # Reset dataset proterties
        self.dataset.reset()
    
    # Reset model 
        self._model = None 

    # Remove parent model
        self.removeParentModel()

    # Reset split dataset widgets
        self.train_class_list.clear()
        self.valid_class_list.clear()
        self.test_class_list.clear()
        self.subsets_pie.clear_canvas()
        self.subsets_barplot.clear_canvas()

    # Reset balancing
        self.balancing_info.clear()
        self.resetBalancingGroup()
        self.balancing_group.setEnabled(False)

    # Reset learning session graphics
        self.loss_plot.clear_canvas()
        self.accuracy_plot.clear_canvas()
        self.train_confmat.clear_canvas()
        self.valid_confmat.clear_canvas()
        self.test_confmat.clear_canvas()

    # Reset scores labels
        for metric in ('loss', 'accuracy', 'F1_macro', 'F1_weighted'):
            for name in ('Train', 'Validation', 'Test'):
                self.updateScoreLabel(name, metric, None)

    # Disable "start", "stop", "test" and "save" buttons
        self.start_learn_btn.setEnabled(False)
        self.stop_learn_btn.setEnabled(False)
        self.test_model_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)


    def _threadRunning(self) -> tuple[bool, str | None]:
        '''
        Check if an external thread is currently running.

        Returns
        -------
        tuple[bool, str or None] 
            (True, thread name) if a thread is running, otherwise (True, None).

        '''
        for thread in (self.balancing_thread, self.learning_thread):
            if thread.isRunning():
                return (True, thread.objectName())
        else:
            return (False, None)


    def on_balancing_help_clicked(self) -> None:
        '''
        Actions to perform when the balancing help button is clicked.

        '''
        # Open official imbalanced-learn user guide
        link = 'https://imbalanced-learn.org/stable/user_guide.html#user-guide'
        webbrowser.open(link)
        

    def onSeedChanged(self, old_seed: int, new_seed: int) -> None:
        '''
        Determine whether or not a random seed change should be allowed based
        on the current state of the Model Learner. 

        Parameters
        ----------
        old_seed : int
            Random seed before seed change event.
        new_seed : int
            Random seed after seed change event. This argument is not used by
            this method; it is included for compatibility with 'seedChanged'
            signal (see 'custom_widgets.RandomSeedGenerator' for details).

        '''
    # Prevent changing seed when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            self.seed_generator.blockSignals(True)
            self.seed_generator.seed_input.setText(str(old_seed))
            self.seed_generator.blockSignals(False)
            text = f'Cannot change seed while a {thr_name} Session is active'
            CW.MsgBox(self, 'Crit', text)
            return
    
    # Ask confirmation if a dataset was already processed
        if self.has_dataset() and self.dataset.are_subsets_split():
            text = 'Current progress will be lost if seed is changed. Confirm?'
            choice = CW.MsgBox(self, 'QuestWarn', text)
            if choice.no():
                self.seed_generator.blockSignals(True)
                self.seed_generator.seed_input.setText(str(old_seed))
                self.seed_generator.blockSignals(False)
                return
            else:
                self._reset()


    def loadGroundTruthDataset(self) -> None:
        '''
        Load ground truth dataset from file. This method launches the dataset
        chunk reader thread.

        '''
    # Prevent loading dataset when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot load dataset while a {thr_name} Session is active'
            CW.MsgBox(self, 'Crit', text)
            return
    
    # Do nothing if path is invalid or the dialog is canceled
        ftype = 'CSV (*.csv)'
        path = CW.FileDialog(self, 'open', 'Load Dataset', ftype).get_path()
        if not path:
            return
    
    # Ask confirmation if a dataset was already processed
        if self.has_dataset():
            text = 'Load a new dataset? Current progress will be lost.'
            choice = CW.MsgBox(self, 'QuestWarn', text)
            if choice.no():
                return
            else:
                self._reset()

    # Set up a temporary popup progress bar
        n_chunks = self.dataset_reader.chunks_number(path)
        pbar = CW.PopUpProgBar(self, n_chunks, 'Loading dataset')
        
    # Connect dataset reader thread signals with popup progress bar
        self.dataset_reader.thread.iterCompleted.connect(pbar.setValue)
        self.dataset_reader.thread.taskFinished.connect(pbar.reset)
        self.dataset_reader.thread.taskFinished.connect(
            self._parseDatasetReaderResult)
    
    # Update current dataset path 
        self.dataset_path_lbl.setPath(path)

    # Launch CSV chunk reader thread
        self.dataset_reader.read_threaded(path)


    def _parseDatasetReaderResult(self, result: tuple, success: bool) -> None:
        '''
        Parse the result of the dataset chunk reader thread and compile ground
        truth dataset if it is succesful.

        Parameters
        ----------
        result : tuple
            Result of the dataset chunk reader thread.
        success : bool
            Whether the thread ended succesfully.

        '''
        if success:
            try:
            # Compile ground truth dataset
                dataframe = self.dataset_reader.combine_chunks(result)
                path = self.dataset_path_lbl.fullpath
                self.set_dataset(dtools.GroundTruthDataset(dataframe, path))
            # Update dataset preview
                head1, head2 = 'DATAFRAME PREVIEW', '\nPER-CLASS DATA COUNT'
                class_count = self.dataset.column_count(-1)
                cnt = '\n'.join(f'{k} = {v}' for k, v in class_count.items())
                text = '\n\n'.join((head1, repr(dataframe), head2, cnt))
                self.dataset_preview.setText(text)
            # Enable "Split dataset" and "Load previous model" buttons
                self.load_pmodel_btn.setEnabled(True)
                self.split_dataset_btn.setEnabled(True)
                return

            except Exception as e:
                err_txt = str(e)

        else:
            err_txt = str(result[0])

        # If we are here, either the thread or the dataset compilation failed
        self._dataset = None
        self.dataset_preview.clear()
        self.dataset_path_lbl.clearPath()
        CW.MsgBox(self, 'Crit', 'Loading dataset failed.', err_txt)


    def _fixSubsetsRatios(self, new_value: int) -> None:
        '''
        Adjust subsets ratios when one of them is changed, so that they always
        sum to 100.

        Parameters
        ----------
        new_value : int
            Altered subset value.

        '''
        altered = self.sender()

    # If train ratio was changed, validation ratio will adapt
        if altered == self.train_ratio_spbox:
            adapting, fixed = self.valid_ratio_spbox, self.test_ratio_spbox
    # If validation ratio was changed, test ratio will adapt
        elif altered == self.valid_ratio_spbox:
            adapting, fixed = self.test_ratio_spbox, self.train_ratio_spbox
    # If test ratio was changed, train ratio will adapt
        else:
            adapting, fixed = self.train_ratio_spbox, self.valid_ratio_spbox

    # Adjust adapting subset
        adapting.blockSignals(True)
        adapting.setValue(100 - fixed.value() - new_value)
        adapting.blockSignals(False)

    # If sum is still not 100, it means that the adapting ratio has reached its 
    # minimum value (=1). So we also change the fixed ratio.
        if new_value + adapting.value() + fixed.value() != 100:
            fixed.blockSignals(True)
            fixed.setValue(100 - adapting.value() - new_value)
            fixed.blockSignals(False)


    def splitDataset(self) -> None:
        '''
        Split dataset into train, validation and test sets.

        '''
    # Prevent splitting dataset when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot split dataset while a {thr_name} Session is active.'
            CW.MsgBox(self, 'Crit', text)
            return

    # Ask for overwriting previous balancing results
        if self.balancing_info:
            text = 'Any existent balancing result will be discarded. Continue?'
            msgbox = CW.MsgBox(self, 'Quest', text)
            if not msgbox.is_choice('Yes'):
                return
            
    # Reset balancing info
        self.balancing_info.clear()

    # Reset dataset properties if split was already performed
        if (dataset := self.dataset).are_subsets_split():
            dataset.reset()

    # Split features [X] from targets [Y] expressed as labels
        pbar = CW.PopUpProgBar(self, 4, 'Splitting dataset')
        try:
            x_dtype, y_dtype = InputMap._DTYPE, MineralMap._DTYPE_STR
            dataset.split_features_targets(xtype=x_dtype, ytype=y_dtype)
            pbar.increase()

    # Update encoder. Inherit from parent model encoder if present.    
            parent_enc = {}
            if (pmodel_path := self.pmodel_path.fullpath) != '':
                parent_enc = mltools.EagerModel.load(pmodel_path).encoder
            dataset.update_encoder(parent_enc)
            pbar.increase()

    # Split dataset into train, validation and test sets
            tr_ratio = self.train_ratio_spbox.value() / 100
            vd_ratio = self.valid_ratio_spbox.value() / 100
            ts_ratio = self.test_ratio_spbox.value() / 100
            seed = self.seed_generator.seed
            dataset.split_subsets(tr_ratio, vd_ratio, ts_ratio, seed) 
            pbar.increase()

        except Exception as e:
            pbar.reset()
            CW.MsgBox(self, 'Crit', 'Unexpected data type', str(e))
            return

    # Update GUI elements
        for subset in ('Train', 'Validation', 'Test'):
            self.updateSubsetCounterWidget(subset)
        self.updateSubsetsPlots()
        self.resetBalancingGroup()
        self._initBalancingTable()
        self.balancing_group.setEnabled(True)
        self.start_learn_btn.setEnabled(True)
        self.test_model_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        pbar.increase()


    def updateSubsetCounterWidget(
        self,
        subset: Literal['Train', 'Validation', 'Test']
    ) -> None:
        '''
        Update list widget of a subset class counter.

        Parameters
        ----------
        subset : Literal['Train', 'Validation', 'Test']
            Name of the subset.

        '''
        match subset:
            case 'Train':
                classes, counts = zip(*self.dataset.train_counter.items())
                self.train_class_list.clear()
                self.train_class_list.addItems(sorted(classes))
                self.train_curr_count_lbl.setText('')
                self.train_tot_count_lbl.setText(f'Tot = {sum(counts)}')
            case 'Validation':
                classes, counts = zip(*self.dataset.valid_counter.items())
                self.valid_class_list.clear()
                self.valid_class_list.addItems(sorted(classes))
                self.valid_curr_count_lbl.setText('')
                self.valid_tot_count_lbl.setText(f'Tot = {sum(counts)}')
            case 'Test':
                classes, counts = zip(*self.dataset.test_counter.items())
                self.test_class_list.clear()
                self.test_class_list.addItems(sorted(classes))
                self.test_curr_count_lbl.setText('')
                self.test_tot_count_lbl.setText(f'Tot = {sum(counts)}')


    def updateSubsetsPlots(self) -> None:
        '''
        Update the pie chart and the bar chart displaying the train, validation 
        and test subsets.

        '''
    # Update Pie chart
        tr = self.train_ratio_spbox.value()
        vd = self.valid_ratio_spbox.value()
        ts = self.test_ratio_spbox.value()
        lbls = ('Train', 'Validation', 'Test')
        self.subsets_pie.update_canvas((tr, vd, ts), labels=lbls)

    # Update Bar chart
        dataset = self.dataset
        bar_data = [list(cnt.values()) for cnt in dataset.counters()]
        tickslbl = dataset.encoder.keys()
        self.subsets_barplot.update_canvas(bar_data, tickslbl)


    def countClassInstances(self, item: QW.QListWidgetItem) -> None:
        '''
        Count instances of selected class. The target subset is automatically
        identified using sender().

        Parameters
        ----------
        item : QW.QListWidgetItem
            Subset list's item containing the class name as text.

        '''
        # Get counter and label based on target subset (sender)
        match self.sender():
            case self.train_class_list:
                counter = self.dataset.train_counter
                label = self.train_curr_count_lbl
            case self.valid_class_list:
                counter = self.dataset.valid_counter
                label = self.valid_curr_count_lbl
            case self.test_class_list:
                counter = self.dataset.test_counter
                label = self.test_curr_count_lbl
            case _:
                return

        count = counter[item.text()]
        label.setText(f'{item.text()} = {count}')


    def cancelBalancingResults(self) -> None: 
        '''
        Cancel all balancing operation and restore original train, validation
        and test subsets.

        '''
    # Exit if dataset is not split (safety, should be impossible) or not balanced
        dataset = self.dataset
        if not dataset.are_subsets_split() or not self.balancing_info:
            return
        
    # Ask for user confirm
        text = 'Any existent balancing result will be discarded. Continue?'
        choice = CW.MsgBox(self, 'Quest', text)
        if choice.no(): 
            return
        
    # Discard balancing results
        dataset.discard_balancing()
        self.balancing_info.clear()

    # Set ratios of subsets spin-boxes to original values 
        train_ratio, valid_ratio, test_ratio = dataset.orig_subsets_ratios
        self.train_ratio_spbox.setValue(round(train_ratio * 100))
        self.valid_ratio_spbox.setValue(round(valid_ratio * 100))
        self.test_ratio_spbox.setValue(round(test_ratio * 100))

    # Update only the train subset counter, as balancing only affects train set
        self.updateSubsetCounterWidget('Train')

    # Update other GUI elements
        self.updateSubsetsPlots()
        self.resetBalancingGroup()
        self._initBalancingTable()

    
    def stopBalancingSession(self) -> None:
        '''
        Interrupt balancing session. This is visually shown by setting the 
        balancing progress bar to an undetermined state.

        '''
        if self.balancing_thread.isRunning():
            self.balancing_pbar.setUndetermined()
            self.balancing_pbar.step('Interrupting session')
            self.balancing_thread.requestInterruption()


    def startBalancingSession(self) -> None:
        '''
        Start a new Balancing Session. This method launches the balancing 
        external thread.

        '''
    # Prevent starting a new balancing session when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = (
                f'Cannot start a new Balancing Session while a {thr_name} '
                'Session is active.'
            )
            CW.MsgBox(self, 'Crit', text)
            return
        
    # Get cell (widgets) data from balancing table
        dataset = self.dataset
        classes = [c.text() for c in self.getBalancingTableCellsByColumn(0)]
        classes_ids = [dataset.encoder.get(c, -1) for c in classes]
        line_edits = self.getBalancingTableCellsByColumn(3)
        values = [l.value() for l in line_edits]
        ratios = [l.ratio(1) for l in line_edits]

    # Get over-sampling and under-sampling parameters
        strategy: dtools.BalancingStrategy = dict(zip(classes_ids, values))
        ovs = self.os_combox.currentText()
        uds = self.us_combox.currentText()
        seed = self.seed_generator.seed
        kos = self.k_neigh_spbox.value()
        mos = self.m_neigh_spbox.value()
        nus = self.n_neigh_spbox.value()
        njobs = (-1) ** self.balancing_multicore_cbox.isChecked()

    # Ask for user confirm
        w: list[str] = [] # list of warnings
        if ovs == 'None':
            ovs = None
            w.append('Over-Sampling not selected. Increased classes ignored.')
        elif max(ratios) > 5:
            w.append((
                'High after-balancing value(s) detected (>500 %). Aggressive '
                'over-sampling can cause model overfitting!'
            ))

        if uds == 'None':
            uds = None
            w.append('Under-Sampling not selected. Decreased classes ignored.')
        elif min(ratios) < 0.2:
            w.append((
                'Low after-balancing value(s) detected (<20 %). Aggressive '
                'under-sampling can cause information loss!'
            ))

        icon = QW.QMessageBox.Icon.Warning if w else QW.QMessageBox.Icon.Question
        note = '\n\n'.join(w) if w else 'No additional info.'
        text = (
            f'You have {len(w)} warning(s). Click "Show Details" for more info.'
            '\n\nLaunch Balancing Session?'
        )
        msgbox = CW.MsgBox(self, 'Quest', text, note, icon=icon)
        
    # Run the balancing session
        if msgbox.is_choice('Yes'):
            self.canc_balancing_btn.setEnabled(False)
            self.balancing_pbar.reset()
            self.balancing_pbar.setUndetermined()

            f1 = dataset.parse_balancing_strategy
            f2 = dataset.oversample
            f3 = dataset.undersample
            f4 = dataset.shuffle
            pipeline = (f1, f2, f3, f4)
            params = (strategy, seed, ovs, uds, kos, mos, nus, njobs)

            self.balancing_thread.set_pipeline(pipeline, *params)
            self.balancing_thread.start()


    def _parseBalancingResult(self, result: tuple, success: bool) -> None:
        '''
        Parse the result of the balancing thread. If the Balancing Session was
        successful, this method applies the balancing to the train set.

        Parameters
        ----------
        result : tuple
            Balancing thread result.
        success : bool
            Whether the balancing thread succeeded or not.

        '''
    # Use a determined state for the balancing progress bar
        self.balancing_pbar.reset()
        self.balancing_pbar.setRange(0, 4)

        if success:
        # Parse the result if balancing operations succeded
            x_bal, y_bal, info = result
            (dataset := self.dataset).apply_balancing(x_bal, y_bal)

        # Update train, validation and test sets ratios spin boxes
            self.balancing_pbar.step('Updating subsets')
            tr_ratio, vd_ratio, ts_ratio = dataset.current_subsets_ratios()
            self.train_ratio_spbox.setValue(round(tr_ratio * 100))
            self.valid_ratio_spbox.setValue(round(vd_ratio * 100))
            self.test_ratio_spbox.setValue(round(ts_ratio * 100))
        
        # Update only the train counter widget, as balancing only affects train
            self.updateSubsetCounterWidget('Train')

        # Add new balancing info to the balancing operation tracker
            self.balancing_pbar.step('Tracking balancing info')
            info['New TVT ratios'] = (
                round(tr_ratio, 2), 
                round(vd_ratio, 2), 
                round(ts_ratio, 2)
            )
            self.balancing_info.append(info)

        # Update pie chart and bar chart plots
            self.balancing_pbar.step('Updating plots')
            self.updateSubsetsPlots()

        # Re-initialize the balancing table
            self.balancing_pbar.step('Updating table')
            self._initBalancingTable()
        
        else:
        # Forward error if balancing operations failed
            CW.MsgBox(self, 'Crit', 'Balancing failed.', str(result[0]))
            
    # End balancing session in any case
        self._endBalancingSession(success)


    def _endBalancingSession(self, success: bool = False) -> None:
        '''
        Internally and visually exit from a balancing thread session.

        Parameters
        ----------
        success : bool, optional
            Whether the balancing thread ended with success. The default is
            False.

        '''
        self.canc_balancing_btn.setEnabled(True)
        self.balancing_pbar.reset()

    # If pbar is left undetermined, it visually seems that process hasn't stop
        if self.balancing_pbar.undetermined():
            self.balancing_pbar.setMaximum(4)
        
        if success:
            CW.MsgBox(self, 'Info', 'Balancing session concluded succesfully.')


    def _initBalancingTable(self) -> None:
        '''
        Initialize the balancing table by populating it with current train set 
        data. 

        '''
    # Get train set data and set the appropriate number of rows in the table
        dataset = self.dataset
        names, curr_counts = zip(*dataset.train_counter.items())
        n_rows = len(names)
        self.balancing_table.setRowCount(n_rows)

        for row in range(n_rows):
        # 1st col item: class name 
            name = names[row]
            i0 = QW.QTableWidgetItem(name)
        # 2nd col item: original class count before any balancing operation
            orig_count = dataset.orig_train_counter[name]
            i1 = QW.QTableWidgetItem(str(orig_count)) 
        # 3rd col item: current class count 
            curr_count = curr_counts[row]
            i2 = QW.QTableWidgetItem(str(curr_count))
        # Set such items as not editable (only selectable and enabled)
            flags = QC.Qt.ItemFlag.ItemIsSelectable | QC.Qt.ItemFlag.ItemIsEnabled
            for col, i in enumerate((i0, i1, i2)):
                i.setFlags(QC.Qt.ItemFlag(flags))
                self.balancing_table.setItem(row, col, i)
        # 4th col item: expected class count after next balancing operation.
        # This item is a PercentLineEdit connected with a function that sets 
        # the strategy value to 'Custom multi-value' when the item is edited. 
        # The starting value of this item is initialized as the current class 
        # count (= 3rd column value).
            i3 = CW.PercentLineEdit(curr_count, min_perc=1, max_perc=1000)
            i3.valueEdited.connect(
                lambda: self.strategy_combox.setCurrentText('Custom multi-value'))
            self.balancing_table.setCellWidget(row, 3, i3)


    def resetBalancingGroup(self) -> None:
        '''
        Reset balancing group widgets.

        '''
        self.balancing_table.clearContents()
        self.balancing_table.setRowCount(0)
        self.strategy_combox.blockSignals(True)
        self.strategy_combox.setCurrentText('Current')
        self.strategy_combox.blockSignals(False)
        self.strategy_value.clear()
        self.strategy_percent.setText('100')


    @overload # when 'col' is 0, 1 or 2 
    def getBalancingTableCellsByColumn(
        self, col: Literal[0, 1, 2]) -> list[QW.QTableWidgetItem]: ...

    @overload # when 'col' is 3 
    def getBalancingTableCellsByColumn(
        self, col: Literal[3]) -> list[CW.PercentLineEdit]: ...

    def getBalancingTableCellsByColumn(
        self,
        col: Literal[0, 1, 2, 3]
    ) -> list[QW.QTableWidgetItem] | list[CW.PercentLineEdit]:
        '''
        Convenient method to get all cells in a given column of the balancing
        table. 

        Parameters
        ----------
        col : Literal[0, 1, 2, 3]
            Column index.

        Returns
        -------
        list[QW.QTableWidgetItem] | list[CW.PercentLineEdit]
            List of cells (or cell widgets) in the given column. The type of 
            the elements in the list can be 'PercentLineEdit' for column 3, or
            'QTableWidgetItem' for columns 0, 1 and 2.
        
        Raises
        ------
        ValueError
            Raised if 'col' is an invalid column index.

        '''
        rows = range(self.balancing_table.rowCount())
        match col:
            case 0 | 1 | 2: # "Class", "Original" and "Current" columns 
                items = [self.balancing_table.item(r, col) for r in rows]
            case 3: # "After Balancing" column
                items = [self.balancing_table.cellWidget(r, col) for r in rows]
            case _:
                raise ValueError(f'Invalid column index: {col}.')

        return [i for i in items if i is not None]


    def parseStrategy(self, strategy: BalancingStrategyName) -> None:
        '''
        Apply different visual modification to the widgets of the balancing
        group depending on the strategy that has been selected. This is mainly
        done to provide feedback and/or information about the choosen strategy.

        Parameters
        ----------
        strategy : BalancingStrategyName
            Selected balancing strategy.

        '''
        match strategy:
            case 'Current': # Set the strategy percent to 100 %
                self.strategy_percent.setText('100')
                self.strategy_value.clear()
            case 'Min': # Set the strategy value to the minimum counter value
                n = min(self.dataset.train_counter.values())
                self.strategy_value.setText(str(n))
                self.strategy_percent.clear()
            case 'Max': # Set the strategy value to the maximum counter value
                n = max(self.dataset.train_counter.values())
                self.strategy_value.setText(str(n))
                self.strategy_percent.clear()
            case 'Mean': # Set the strategy value to the average counter value
                n = np.mean(list(self.dataset.train_counter.values()))
                self.strategy_value.setText(str(round(n)))
                self.strategy_percent.clear()
            case 'Median': # Set the strategy value to the median counter value
                n = np.median(list(self.dataset.train_counter.values()))
                self.strategy_value.setText(str(round(n)))
                self.strategy_percent.clear()
            case 'Custom value': # Clear out strategy percent
                self.strategy_percent.clear()
            case 'Custom multi-value': # Clear out strategy value
                self.strategy_value.clear()
        

    def fillAfterBalancingColumn(self, value: str | None) -> None:
        '''
        Fill the last column ('After Balancing') of the balancing table with
        a unique balancing strategy custom value or percentage. 

        Parameters
        ----------
        value : str or None
            The balancing strategy value or percentage as a string. If None or 
            a non-numeric string is passed, this method does nothing.

        '''
    # Exit function if an invalid value is passed 
        if value is None or not value.isnumeric():
            return

    # Get all percent line edit widgets (table cell widgets)
        cell_widgets = self.getBalancingTableCellsByColumn(3)

    # Determine how to fill cell widgets based on the sender
        match self.sender():
            case self.strategy_value:
                for c in cell_widgets:
                    c.setValue(int(value))
            case self.strategy_percent:
                for c in cell_widgets:
                    c.setPercent(int(value))
            case _:
                return


    def loadParentModel(self) -> None:
        '''
        Load parent ML model, allowing users to train a new model starting from
        its pre-trained parameters.

        '''
    # Prevent loading parent model when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot load model while a {thr_name} Session is active'
            CW.MsgBox(self, 'Crit', text)
            return

    # Do nothing if path is invalid or file dialog is canceled
        ftype = 'PyTorch model (*.pth)'
        path = CW.FileDialog(self, 'open', 'Load Model', ftype).get_path()
        if not path:
            return

    # Ask confirmation if a dataset was already processed
        if (dataset := self.dataset).are_subsets_split():
            text = 'Current progress will be lost if model is loaded. Confirm?'
            choice = CW.MsgBox(self, 'QuestWarn', text)
            if choice.no():
                return
            else:
                self._reset()
        
        pbar = CW.PopUpProgBar(self, 4, 'Loading Model')
        
        try:
        # Import parent model
            pmodel = mltools.EagerModel.load(path)
            path = pmodel.filepath # refresh path in case of version update
            pbar.increase()

        # Check if parent model shares the same features of loaded dataset
            if pmodel.features != dataset.columns_names()[:-1]:
                err = 'Model and loaded dataset have incompatible features.'
                raise ValueError(err)
            
        # Check if parent model shares the same targets of loaded dataset
            if pmodel.targets != dataset.column_unique(-1):
                err = 'Model and loaded dataset have incompatible target classes.'
                raise ValueError(err)
            pbar.increase()

        # Show parent model preview. Ask for rebuilding missing model log file.
            logpath = '' if path is None else pmodel.generate_log_path(path)
            if not os.path.exists(logpath):
                quest_text = 'Unable to find model log file. Rebuild it?'
                choice = CW.MsgBox(pbar, 'Quest', quest_text)
                if choice.yes():
                    ext_log = settings.manager.get('data/extended_model_log')
                    pmodel.save_log(logpath, extended=ext_log)
            self.pmodel_preview.setDoc(logpath) # Raises no error if missing
            pbar.increase()

        # Update GUI
            lr, wd, mtm, _, _ = pmodel.hyperparameters
            poly_degree = pmodel.poly_degree
            self.pmodel_path.setPath(path)
            self.seed_generator.seed_input.setText(str(pmodel.seed))
            self.lr_spbox.setValue(lr)
            self.wd_spbox.setValue(wd)
            self.mtm_spbox.setValue(mtm)
            self.feat_mapping_cbox.setChecked(poly_degree > 1)
            self.feat_mapping_cbox.setEnabled(False)
            self.poly_deg_spbox.setValue(poly_degree) # if 1, is ignored
            self.poly_deg_spbox.setEnabled(False)
            self.algm_combox.setCurrentText(pmodel.algorithm)
            self.algm_combox.setEnabled(False)
            self.optim_combox.setCurrentText(pmodel.optimizer)
            pbar.increase()
            CW.MsgBox(self, 'Info', 'Model loaded successfully.')
        
        except Exception as e:
            pbar.reset()
            self.removeParentModel()
            CW.MsgBox(self, 'Crit', 'Failed to load model.', str(e))


    def removeParentModel(self) -> None:
        '''
        Remove loaded parent model, allowing users to train a new model without
        pre-trained model parameters.

        '''
        self.pmodel_path.clearPath()
        self.pmodel_preview.clear()
        # User can select again polynomial feature mapping and algorithm
        self.feat_mapping_cbox.setEnabled(True)
        self.feat_mapping_cbox.setChecked(False)
        self.algm_combox.setEnabled(True)


    def updateScoreLabel(self, subset: str, metric: str, score: float | None) -> None:
        '''
        Update a label that displays a specific learning score.

        Parameters
        ----------
        subset : str
            Name of the subset. Must be one of ('Train', 'Validation', 'Test').
        metric : str
            Name of the score. Must be one of ('loss', 'accuracy', 'F1_macro', 
            'F1_weighted').
        score : float or None
            Value of the score. If None the label is cleared.

        '''
        names = ('Train', 'Validation', 'Test')
        metrics = ('loss', 'accuracy', 'F1_macro', 'F1_weighted')

        labels = (
            # loss
            (self.train_loss_lbl, self.valid_loss_lbl, None), 
            # accuracy                            
            (self.train_accuracy_lbl, self.valid_accuracy_lbl, self.test_accuracy_lbl),
            # f1 macro
            (self.train_f1_macro_lbl, self.valid_f1_macro_lbl, self.test_f1_macro_lbl),
            # f1 weighted
            (self.train_f1_weight_lbl, self.valid_f1_weight_lbl, self.test_f1_weight_lbl)
        )
        col = names.index(subset)
        row = metrics.index(metric)
        widget = labels[row][col]

        if widget:
            text = '' if score is None else '{:.9f}'.format(score)
            widget.setText(text)

    
    def stopLearningSession(self) -> None:
        '''
        Interrupt learning session. This is visually shown by setting the 
        learning progress bar to an undetermined state.

        '''
        if self.learning_thread.isRunning():
            self.learning_pbar.setRange(0, 0)
            self.learning_thread.requestInterruption()


    def startLearningSession(self) -> None:
        '''
        Start a new Learning Session. This method launches the learning 
        external thread.

        '''
    # Exit function if dataset has an invalid filepath (safety)
        if (dataset := self.dataset).filepath is None:
            return

    # Prevent starting a new learning session when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = (
                f'Cannot start a new Learning Session while a {thr_name} '
                'Session is active.'
            )
            CW.MsgBox(self, 'Crit', text)
            return
        
    # Check that subsets contain at least one instance of each target class
        tr_cls = {k for k, v in dataset.train_counter.items() if v}
        vd_cls = {k for k, v in dataset.valid_counter.items() if v}
        ts_cls = {k for k, v in dataset.test_counter.items() if v}

        if xor := ((tr_cls ^ vd_cls) | (tr_cls ^ ts_cls) | (vd_cls ^ ts_cls)): 
            err = 'Some classes have 0 instances in one or more subsets.'
            CW.MsgBox(self, 'Crit', err, ', '.join(xor))
            return

    # Update GUI
        self.learning_pbar.setRange(0, 4)
        self.learning_pbar.setTextVisible(False)
        self.start_learn_btn.setEnabled(False)
        self.stop_learn_btn.setEnabled(True)
        self.test_model_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)

    # Get learning hyperparameters and preferences
        parent_path = self.pmodel_path.fullpath
        lr = self.lr_spbox.value()
        wd = self.wd_spbox.value()
        mtm = self.mtm_spbox.value()
        epochs = self.epochs_spbox.value()
        batch_size = int(self.batch_combox.currentText())
        workers = self.workers_spbox.value()
        poly_feat_mapping = self.feat_mapping_cbox.isChecked()
        poly_deg = self.poly_deg_spbox.value() if poly_feat_mapping else 1
        algm_name = self.algm_combox.currentText()
        optim_name = self.optim_combox.currentText()
        uprate = int(self.plots_update_rate.text())
        device = 'cuda' if self.cuda_cbox.isChecked() else 'cpu'

    # If parent model exists, update some of its variables
        if parent_path:
            self.set_model(mltools.EagerModel.load(parent_path))
            var_dict = (model := self.model).variables
            var_dict['optimizer'] = optim_name
            var_dict['device'] = device
            var_dict['parent_model_path'] = parent_path
            var_dict['dataset_path'] = dataset.filepath
            var_dict['tvt_ratios'] = dataset.orig_subsets_ratios
            var_dict['balancing_info'] = self.balancing_info
            var_dict['learning_rate'] = lr
            var_dict['weight_decay'] = wd
            var_dict['momentum'] = mtm
            var_dict['batch_size'] = batch_size

    # If parent model does not exist, initialize a new empty one from scratch
        else:
            self.set_model(mltools.EagerModel.initialize_empty())
            var_dict = (model := self.model).variables
            var_dict['algorithm'] = algm_name
            var_dict['optimizer'] = optim_name
            var_dict['input_features'] = dataset.columns_names()[:-1]
            var_dict['class_encoder'] = dataset.encoder
            var_dict['device'] = device
            var_dict['seed'] = self.seed_generator.seed
            var_dict['parent_model_path'] = parent_path
            var_dict['dataset_path'] = dataset.filepath
            var_dict['tvt_ratios'] = dataset.orig_subsets_ratios
            var_dict['balancing_info'] = self.balancing_info
            var_dict['polynomial_degree'] = poly_deg
            var_dict['learning_rate'] = lr
            var_dict['weight_decay'] = wd
            var_dict['momentum'] = mtm
            var_dict['batch_size'] = batch_size

        self.learning_pbar.setValue(1)

    # Map features from linear to polynomial (get original data if degree=1)
        x_train, x_valid = dataset.x_train, dataset.x_valid
        degree = model.poly_degree
        x_train = mltools.map_polinomial_features(x_train, degree)
        x_valid = mltools.map_polinomial_features(x_valid, degree)

        self.learning_pbar.setValue(2)

    # Convert train and validation features and targets to torch Tensors
        x_train = mltools.array2tensor(x_train, 'float32')
        y_train = mltools.array2tensor(dataset.y_train, 'uint8')
        x_valid = mltools.array2tensor(x_valid, 'float32')
        y_valid = mltools.array2tensor(dataset.y_valid, 'uint8')

    # Normalize feature data using parent standards or by calculating new ones
        x_mean, x_stdev = var_dict['standards'] # empty tensors if model is new
        if x_mean.size(dim=0) > 0 and x_stdev.size(dim=0) > 0:
            x_train_norm, _ = mltools.norm_data(x_train, x_mean, x_stdev)
        else:
            x_train_norm, (x_mean, x_stdev) = mltools.norm_data(x_train)
            var_dict['standards'] = (x_mean, x_stdev)

        x_valid_norm, _ = mltools.norm_data(x_valid, x_mean, x_stdev)
        
        self.learning_pbar.setValue(3)

    # Prepare network and optimizer. If a parent model is provided, load its 
    # network and optimizer state dicts
        network = model.get_network_architecture()
        network.to(device)
        optimizer = model.get_optimizer(network)
        var_dict['loss_function'] = network._loss_name


        if len(parent_model_state_dict := var_dict['model_state_dict']):
            network.load_state_dict(parent_model_state_dict)
        if len(parent_optimizer_state_dict := var_dict['optimizer_state_dict']):
            optimizer.load_state_dict(parent_optimizer_state_dict)

        self.learning_pbar.setValue(4)

    # Reset progress bar range
        parent_epochs = var_dict['epochs']
        e_min = parent_epochs if parent_epochs else 0
        e_max = e_min + epochs
        self.learning_pbar.reset()
        self.learning_pbar.setRange(e_min, e_max)
        self.learning_pbar.setTextVisible(True)

    # Adjust graphics update rate
        if uprate == 0:
            uprate = e_max # prevent ZeroDivisionError when update graphics
        elif uprate / epochs < 0.02:
            uprate = epochs * 0.02 # prevent too fast updates that crashes the thread(?)

    # Initialize loss and accuracy lists and plots
        loss_lists = var_dict['loss_list']
        acc_lists = var_dict['accuracy_list']
        for plot in (self.loss_plot, self.accuracy_plot):
            plot.clear_canvas()

    # Start the learning thread
        if batch_size > 0:
            self.learning_thread.set_task(
                lambda: network.batch_learn(
                    mltools.DataLoader(x_train_norm, y_train, batch_size, workers),
                    mltools.DataLoader(x_valid_norm, y_valid, batch_size, workers),
                    optimizer,
                    device
                )
            )
        else:
            self.learning_thread.set_task(
                lambda: network.learn(
                    x_train_norm, 
                    y_train, x_valid_norm, 
                    y_valid, 
                    optimizer, 
                    device
                )                
            )

        self.learning_thread.set_params(e_min, e_max, uprate, *loss_lists, *acc_lists)
        self.learning_thread.start()

    # Store network and optimizer attributes (TODO improve by making "network" 
    # a stored attribute of EagerModel class)
        self.network = network
        self.optimizer = optimizer


    def _parseLearningResult(self, result: tuple, success: bool) -> None:
        '''
        Parse the result of the learning thread. If the Learning Session was
        successful, this method populates the train and validation confusion
        matrices.

        Parameters
        ----------
        result : tuple
            Learning thread result.
        success : bool
            Whether the learning thread succeeded or not.

        '''
        self.learning_pbar.reset()
        
    # Predict full train and validation sets if learning session succeded
        if success and self.network is not None and self.optimizer is not None: 
            tr_loss_list, vd_loss_list, tr_acc_list, vd_acc_list = result
            self.learning_pbar.setRange(0, 2)
            self.learning_pbar.setTextVisible(False)

            dataset = self.dataset
            var_dict = (model := self.model).variables
            d = var_dict['polynomial_degree']
            x_mean, x_stdev = var_dict['standards']
            device = var_dict['device']
            x_train = mltools.map_polinomial_features(dataset.x_train, d)
            x_valid = mltools.map_polinomial_features(dataset.x_valid, d)
            x_train = mltools.array2tensor(x_train, 'float32')
            x_valid = mltools.array2tensor(x_valid, 'float32')
            x_train, _ = mltools.norm_data(x_train, x_mean, x_stdev)
            x_valid, _ = mltools.norm_data(x_valid, x_mean, x_stdev)
            train_pred = self.network.predict(x_train.to(device))[1].cpu()
            valid_pred = self.network.predict(x_valid.to(device))[1].cpu()

        # Update train and validation F1 scores
            f1m_tr = mltools.f1_score(dataset.y_train, train_pred, 'macro')
            f1m_vd = mltools.f1_score(dataset.y_valid, valid_pred, 'macro')
            f1w_tr = mltools.f1_score(dataset.y_train, train_pred, 'weighted')
            f1w_vd = mltools.f1_score(dataset.y_valid, valid_pred, 'weighted')
            self.updateScoreLabel('Train', 'F1_macro', f1m_tr)
            self.updateScoreLabel('Validation', 'F1_macro', f1m_vd)
            self.updateScoreLabel('Train', 'F1_weighted', f1w_tr)
            self.updateScoreLabel('Validation', 'F1_weighted', f1w_vd)
            self.learning_pbar.setValue(1)

        # Populate variables (accuracy and f1 scores of test set are placeholders)
            var_dict['epochs'] = len(tr_loss_list)
            var_dict['accuracy'] = (tr_acc_list[-1], vd_acc_list[-1], 0.)
            var_dict['loss'] = (tr_loss_list[-1], vd_loss_list[-1], None)
            var_dict['f1_scores'] = ((f1m_tr, f1m_vd, 0.), (f1w_tr, f1w_vd, 0.))
            var_dict['accuracy_list'] = (tr_acc_list, vd_acc_list)
            var_dict['loss_list'] = (tr_loss_list, vd_loss_list)
            var_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            var_dict['model_state_dict'] = self.network.state_dict()

        # Update Confusion Matrices
            self.updateConfusionMatrix('Train', train_pred)
            self.updateConfusionMatrix('Validation', valid_pred)
            self.learning_pbar.setValue(2)

        # Update GUI
            self.test_model_btn.setEnabled(True)
            self.learning_pbar.reset()
            self.learning_pbar.setTextVisible(True)
            CW.MsgBox(self, 'Info', 'Learning Session completed successfully.')

    # Forward error if learning session failed
        else:
            CW.MsgBox(self, 'Crit', 'Learning Session failed.', str(result[0]))

    # Update Start-Stop-Learn buttons on exit in any case
        self.start_learn_btn.setEnabled(True)
        self.stop_learn_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)


    def _drawLearningScores(
        self,
        scores: tuple[list[float], list[float], list[float], list[float]]
    ) -> None:
        '''
        Update loss and accuracy curves, as well as score labels.

        Parameters
        ----------
        scores : tuple[list[float], list[float], list[float], list[float]]
            A tuple containing loss and accuracy scores for both train and
            validation sets, in the order: train loss, validation loss, train
            accuracy, validation accuracy.


        '''
        tr_loss, vd_loss, tr_acc, vd_acc = scores

    # Update the loss and accuracy curves
        self.updateLearningCurve('loss', tr_loss, vd_loss)
        self.updateLearningCurve('accuracy', tr_acc, vd_acc)

    # Update score labels widgets
        self.updateScoreLabel('Train', 'loss', tr_loss[-1])
        self.updateScoreLabel('Validation', 'loss', vd_loss[-1])
        self.updateScoreLabel('Train', 'accuracy', tr_acc[-1])
        self.updateScoreLabel('Validation', 'accuracy', vd_acc[-1])


    def updateLearningCurve(
        self, 
        plot: Literal['accuracy', 'loss'],
        y_train: Sequence[float | int],
        y_valid: Sequence[float | int]
    ) -> None:
        '''
        Update loss or accuracy plot.

        Parameters
        ----------
        plot : Literal['accuracy', 'loss']
            Type of scores to update.
        y_train : Sequence[float | int]
            Loss or accuracy values for train set.
        y_valid : Sequence[float | int]
            Loss or accuracy values for validation set.

        Raises
        ------
        ValueError
            Raised if plot is not 'loss' or 'accuracy'.

        '''
        match plot:
            case 'loss':
                canvas = self.loss_plot
            case 'accuracy':
                canvas = self.accuracy_plot
            case _:
                raise ValueError(f'Invalid "plot" argument: {plot}.')
        
        x_data = range(len(y_train))
        curves = [(x_data, y_train), (x_data, y_valid)]
        canvas.update_canvas(curves, [f'Train {plot}', f'Validation {plot}'])


    def updateConfusionMatrix(
        self,
        subset: Literal['Train', 'Validation', 'Test'],
        preds: mltools.TensorLike | None
    ) -> None:
        '''
        Populate confusion matrix of 'subset' with predicted classes data.

        Parameters
        ----------
        subset : Literal['Train', 'Validation', 'Test']
            Name of the subset.
        preds : TensorLike | None
            Predicted classes.

        '''
        # Exit function if predictions are invalid (safety)
        if preds is None:
            return

        dataset = self.dataset
        match subset:
            case 'Train':
                true = dataset.y_train
                perc = self.train_cm_perc_action.isChecked()
                canvas = self.train_confmat
            case 'Validation':
                true = dataset.y_valid
                perc = self.valid_cm_perc_action.isChecked()
                canvas = self.valid_confmat
            case 'Test':
                true = dataset.y_test
                perc = self.test_cm_perc_action.isChecked()
                canvas = self.test_confmat
            case _:
                return

        lbls, ids = zip(*dataset.encoder.items())
        confmat = mltools.confusion_matrix(true, preds, ids)
        canvas.update_canvas(confmat, lbls, perc)


    def toggleConfMatPercent(self, toggled: bool) -> None:
        '''
        Toggle on/off confusion matrix values as percentages. The matrix canvas
        is automatically identified using sender().

        Parameters
        ----------
        toggled : bool
            Whether to show values as percentages.

        '''
    # Determine the matrix canvas from the action that called this method
        match self.sender():
            case self.train_cm_perc_action:
                self.train_confmat.annotate(toggled)
            case self.valid_cm_perc_action:
                self.valid_confmat.annotate(toggled)
            case self.test_cm_perc_action:
                self.test_confmat.annotate(toggled)
            case _:
                return # safety


    def testModel(self) -> None:
        '''
        Test the trained model on test set.

        '''
    # Exit function if network is invalid (safety)
        if self.network is None:
            return

    # Ask for user confirmation
        text = (
            'Once tested, this model should not be further trained on the '
            'same train set. Proceed?'
        ) 
        msgbox = CW.MsgBox(self, 'QuestWarn', text)
        if msgbox.is_choice('Yes'):
            self.learning_pbar.setRange(0, 4)

        # Predict targets on test set
            dataset = self.dataset
            var_dict = self.model.variables
            d = var_dict['polynomial_degree']
            device = var_dict['device']
            x_mean, x_stdev = var_dict['standards']

            x_test = mltools.map_polinomial_features(dataset.x_test, d)
            x_test = mltools.array2tensor(x_test, 'float32')
            x_test, _ = mltools.norm_data(x_test, x_mean, x_stdev)
            test_pred = self.network.predict(x_test.to(device))[1].cpu()
            self.learning_pbar.setValue(1)

        # Compute and store test accuracy
            test_acc = mltools.accuracy_score(dataset.y_test, test_pred)
            train_acc, valid_acc, _ = var_dict['accuracy']
            var_dict['accuracy'] = (train_acc, valid_acc, test_acc)
            self.updateScoreLabel('Test', 'accuracy', test_acc)
            self.learning_pbar.setValue(2)

        # Compute and store test F1 scores
            f1m_ts = mltools.f1_score(dataset.y_test, test_pred, 'macro')
            f1w_ts = mltools.f1_score(dataset.y_test, test_pred, 'weighted')
            self.updateScoreLabel('Test', 'F1_macro', f1m_ts)
            self.updateScoreLabel('Test', 'F1_weighted', f1w_ts)
            ((m_tr, m_vd, _), (w_tr, w_vd, _)) = var_dict['f1_scores']
            var_dict['f1_scores'] = ((m_tr, m_vd, f1m_ts), (w_tr, w_vd, f1w_ts))
            self.learning_pbar.setValue(3)

        # Update test Confusion Matrix 
            self.updateConfusionMatrix('Test', test_pred)
            self.learning_pbar.setValue(4)

        # Update GUI
            self.save_model_btn.setEnabled(True)
            
        # Reset progress bar and end testing session with success
            self.learning_pbar.reset()
            CW.MsgBox(self, 'Info', 'Model tested succesfully.')


    def saveModel(self) -> None:
        '''
        Save model to file. Also saves a model log.

        '''
        ftype = 'PyTorch model (*.pth)'
        path = CW.FileDialog(self, 'save', 'Save Model', ftype).get_path()
        if not path:
            return
        
        try:
            log_path = (model := self.model).generate_log_path(path)
            extended_log = settings.manager.get('data/extended_model_log')
            model.save(path, log_path, extended_log)
            CW.MsgBox(self, 'Info', 'Model saved successfully.')
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save model.', str(e))

    # deprecated
    # def killReferences(self) -> None:
    #     '''
    #     Reimplementation of the 'killReferences' method from the parent class.
    #     It also stops any active balancing or learning thread.

    #     '''
    #     running, _ = self._threadRunning()
    #     if running:
    #         self.stopBalancingSession()
    #         self.stopLearningSession()
    #     super().killReferences()

    def is_safe_to_close(self) -> bool:
        '''
        Reimplementation of the 'is_safe_to_close' method. Changes the close
        confirmation question if a learning thread or a balancing thread is 
        currently active. It stops the thread if user confirms to close the
        tool.

        Returns
        -------
        bool
            Whether the user confirms to close the tool or not.

        '''
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = (
                f'A {thr_name} Session is active. Close {self.windowTitle()}'
                ' anyway? The process will be interrupted and any unsaved data'
                ' will be lost.'
            )
            msgbox = CW.MsgBox(self, 'Quest', text)
            if close := msgbox.is_choice('Yes'):
                self.stopBalancingSession()
                self.stopLearningSession()
            return close

        else:
            return super().is_safe_to_close()


    def force_close(self) -> None:
        '''
        Reimplementation of the 'force_close' method. It also stops any active
        balancing or learning thread.

        '''
        thr_active, _ = self._threadRunning()
        if thr_active:
            self.stopBalancingSession()
            self.stopLearningSession()
        return super().force_close()



class PhaseRefiner(DraggableTool):

    # Advanced refiner properties
    _morph_algorithms = get_args(iatools.BinaryMorphologyAlgorithms)
    _variance_colors = (
        (0, 0, 0),       # 0 -> background preserved
        (255, 0, 0),     # 1 -> phase removed
        (0, 255, 0),     # 2 -> phase added
        (255, 255, 255)  # 3 -> phase preserved
    )
    
    def __init__(self) -> None:
        '''
        One of the main tools of X-Min Learn, which allows the application of
        post-processing refinement operations on classified mineral maps.

        '''
        super().__init__()

    # Set widget attributes
        self.setWindowTitle('Phase Refiner')
        self.setWindowIcon(style.getIcon('PHASE_REFINER'))

    # Set main attributes
        self._refiner_mode = 0 # Basic: 0, Advanced: 1'
        self._minmap: MineralMap | None = None
        self._minmap_backup: MineralMap | None = None

    # Set attributes for advanced refiner
        self._phase: str | None = None
        self._roi_extents: tuple[int, int, int, int] | None = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                           SAMPLE SELECTION WIDGETS
#  -------------------------------------------------------------------------  #

    # Mineral map selector (Sample Maps Selector)
        self.minmap_selector = CW.SampleMapsSelector('minmaps', False)

    # Select mineral map (Styled Button)
        self.select_map_btn = CW.StyledButton(text='Select map')

    # Current selected mineral map (Path Label)
        self.selected_map_lbl = CW.PathLabel(
            full_display=False, placeholder='No map selected')

    # Sample selection group (Collapsible Area)
        sample_vbox = QW.QVBoxLayout()
        sample_vbox.addWidget(self.minmap_selector)
        sample_vbox.addWidget(self.select_map_btn)
        sample_vbox.addWidget(self.selected_map_lbl)
        sample_group = CW.CollapsibleArea(
            sample_vbox, 'Map selection', collapsed=False)

#  -------------------------------------------------------------------------  #
#                              KERNEL WIDGETS
#  -------------------------------------------------------------------------  #

    # Kernel shape selector (Radio Buttons Group)
        shapes = ('square', 'circle', 'diamond')
        shp_icons = [style.getIcon(s.upper()) for s in shapes]
        self.kern_shape_btns = CW.RadioBtnLayout(['']*3, shp_icons, orient='h')
        for shp, btn in zip(shapes, self.kern_shape_btns.buttons()):
            btn.setObjectName(shp)
            btn.setToolTip(shp.capitalize())

    # Kernel size selector (Styled Spin Box)
        self.kern_size_spbox = CW.StyledSpinBox(3, 49, 2)
        self.kern_size_spbox.setToolTip('Must be an odd value')

    # Skip border (Check Box)
        self.skip_border_cbox = QW.QCheckBox('Skip borders')
        self.skip_border_cbox.setToolTip('Preserve map borders')

    # Kernel group (Group Area)
        kern_form = QW.QFormLayout()
        kern_form.setSpacing(15)
        kern_form.addRow('Shape', self.kern_shape_btns)
        kern_form.addRow('Size', self.kern_size_spbox)
        kern_form.addRow(self.skip_border_cbox)
        kern_group = CW.CollapsibleArea(kern_form, 'Kernel')

#  -------------------------------------------------------------------------  #
#                              SETTINGS WIDGETS
#  -------------------------------------------------------------------------  #

    # NoData phase selector (Auto-Update Combo Box) [-> Basic Refiner]
        nd_combox_tip = 'Allows finer control of NoData'
        self.nd_combox = CW.AutoUpdateComboBox(tooltip=nd_combox_tip)
        self.nd_combox.setPlaceholderText('No selection') # Not working in Qt 5.15.2 due to a Qt BUG

    # NoData phase clear button (Styled Button) [-> Basic Refiner]
        self.nd_clear_btn = CW.StyledButton(style.getIcon('CLEAR'))

    # NoData threshold (Styled Double Spin Box) [-> Basic Refiner]
        thresh_tip = 'Output NoData class if its pixels exceed this ratio'
        self.nd_thresh_spbox = CW.StyledDoubleSpinBox(max_value=0.99)
        self.nd_thresh_spbox.setValue(0.5)
        self.nd_thresh_spbox.setToolTip(thresh_tip)
        self.nd_thresh_spbox.setEnabled(False)

    # Algorithm selection (Styled Combo Box) [-> Advanced Refiner]
        self.algm_combox = CW.StyledComboBox()
        self.algm_combox.addItems(self._morph_algorithms)

    # Algorithm warning icon (Label) [-> Advanced Refiner]
        warn_icon = QG.QPixmap(str(style.ICONS.get('WARNING')))
        self.algm_warn = QW.QLabel()
        self.algm_warn.setPixmap(warn_icon.scaled(16, 16, style.ARM_KEEP))
        self.algm_warn.setToolTip('The selected algorithm ignore ROIs')
        self.algm_warn.hide()

    # Invert mask (Check Box) [-> Advanced Refiner]
        self.invert_mask_cbox = QW.QCheckBox('Invert mask')

    # Invert ROI (Check Box) [-> Advanced Refiner]
        self.invert_roi_cbox = QW.QCheckBox('Invert ROI')

    # Removed pixels behaviour (Styled Combo Box) [-> Advanced Refiner]
        del_pixels_tip = 'How should removed pixels be reclassified?'
        self.del_pixels_combox = CW.StyledComboBox(del_pixels_tip)
        self.del_pixels_combox.addItems(('_ND_', 'Nearest class'))
    
    # [Basic Refiner] settings group (Group Area)
        basic_settings_grid = QW.QGridLayout()
        basic_settings_grid.setAlignment(style.A_TOP)
        basic_settings_grid.setVerticalSpacing(10)
        basic_settings_grid.addWidget(QW.QLabel('NoData class'), 0, 0, 1, 1)
        basic_settings_grid.addWidget(self.nd_combox, 0, 1, 1, 1)
        basic_settings_grid.addWidget(self.nd_clear_btn, 0, 2, 1, 1)
        basic_settings_grid.addWidget(QW.QLabel('Threshold'), 1, 0, 1, 1)
        basic_settings_grid.addWidget(self.nd_thresh_spbox, 1, 1, 1, -1)
        basic_settings_grid.setColumnStretch(0, 1)
        basic_settings_grid.setColumnStretch(1, 1)
        basic_settings_group = CW.GroupArea(basic_settings_grid, frame=False)

    # [Advanced Refiner] settings group (Group Area)
        advan_settings_grid = QW.QGridLayout()
        advan_settings_grid.setAlignment(style.A_TOP)
        advan_settings_grid.setVerticalSpacing(10)
        advan_settings_grid.addWidget(QW.QLabel('Algorithm'), 0, 0, 1, 1)
        advan_settings_grid.addWidget(self.algm_warn, 0, 1, 1, 1)
        advan_settings_grid.addWidget(self.algm_combox, 0, 2, 1, 1)
        advan_settings_grid.addWidget(self.invert_mask_cbox, 1, 0, 1, -1)
        advan_settings_grid.addWidget(self.invert_roi_cbox, 2, 0, 1, -1)
        advan_settings_grid.addWidget(QW.QLabel('Removed as'), 3, 0, 1, 1)
        advan_settings_grid.addWidget(self.del_pixels_combox, 3, 2, 1, 1)
        advan_settings_grid.setColumnStretch(0, 1)
        advan_settings_grid.setColumnStretch(2, 1)
        advan_settings_group = CW.GroupArea(advan_settings_grid, frame=False)

    # Settings container widget (Stacked Widget)
        self.settings_stack = QW.QStackedWidget()
        self.settings_stack.addWidget(basic_settings_group)
        self.settings_stack.addWidget(advan_settings_group)
        settings_group = CW.CollapsibleArea(self.settings_stack, 'Settings')

#  -------------------------------------------------------------------------  #
#                             LEGEND WIDGET
#  -------------------------------------------------------------------------  #

    # Legend group (Group Area)
        self.legend = CW.Legend()
        selection_mode = QW.QAbstractItemView.SelectionMode.SingleSelection
        self.legend.setSelectionMode(selection_mode)
        legend_group = CW.GroupArea(self.legend, 'Legend')

#  -------------------------------------------------------------------------  #
#                    CONTROL WIDGETS (APPLY-CANCEL-SAVE)
#  -------------------------------------------------------------------------  #

    # Apply refinement (Styled Button)
        self.apply_btn = CW.StyledButton(text='APPLY', bg=style.OK_GREEN)
        self.apply_btn.setToolTip('Apply filter')

    # Cancel refinements (Styled Button)
        self.cancel_btn = CW.StyledButton(text='CANCEL', bg=style.BAD_RED)
        self.cancel_btn.setToolTip('Revert all edits')

    # Save refinements (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'SAVE')
        self.save_btn.setToolTip('Save refined mineral map')

    # Control widgets layout
        ctrl_widgets_grid = QW.QGridLayout()
        ctrl_widgets_grid.addWidget(self.apply_btn, 0, 0, 1, 1)
        ctrl_widgets_grid.addWidget(self.cancel_btn, 0, 1, 1, 1)
        ctrl_widgets_grid.addWidget(self.save_btn, 1, 0, 1, -1)


#  -------------------------------------------------------------------------  #
#                       BASIC REFINER RENDERING WIDGETS
#  -------------------------------------------------------------------------  #

    # Original map canvas (Image Canvas) 
        self.orig_canvas_basic = plots.ImageCanvas(cbar=False)
        self.orig_canvas_basic.fig.suptitle('Original', size='large')
        self.orig_canvas_basic.setMinimumSize(500, 500)
        
    # Navigation toolbar of original map canvas (Navigation Toolbar)
        self.orig_navtbar_basic = plots.NavTbar.imageCanvasDefault(
            self.orig_canvas_basic, self)
        
    # Current map canvas (Image Canvas) 
        self.curr_canvas_basic = plots.ImageCanvas(cbar=False)
        self.curr_canvas_basic.fig.suptitle('Current', size='large')
        self.curr_canvas_basic.setMinimumSize(500, 500)
        self.curr_canvas_basic.share_axis(self.orig_canvas_basic.ax)

    # Navigation toolbar of current map canvas (Navigation Toolbar) 
        self.curr_navtbar_basic = plots.NavTbar.imageCanvasDefault(
            self.curr_canvas_basic, self)

    # Original map mode barplot (Bar Canvas)
        self.orig_barplot = plots.BarCanvas(wheel_pan=False, wheel_zoom=False)
        self.orig_barplot.setMinimumHeight(200)

    # Navigation toolbar of the original map mode barplot (Navigation Toolbar)
        self.orig_barplot_navtbar = plots.NavTbar.barCanvasDefault(
            self.orig_barplot, self, orient=style.O_VERTICAL)

    # Current map mode barplot (Bar Canvas)
        self.curr_barplot = plots.BarCanvas(wheel_pan=False, wheel_zoom=False)
        self.curr_barplot.setMinimumHeight(200)

    # Navigation toolbar of the current map mode barplot (Navigation Toolbar)
        self.curr_barplot_navtbar = plots.NavTbar.barCanvasDefault(
            self.curr_barplot, self, orient=style.O_VERTICAL)

    # Basic Refiner rendering group
        basic_refiner_grid = QW.QGridLayout()
        basic_refiner_grid.setHorizontalSpacing(10)
        basic_refiner_grid.setRowStretch(1, 1)
        basic_refiner_grid.addWidget(self.orig_navtbar_basic, 0, 0, 1, 2)
        basic_refiner_grid.addWidget(self.curr_navtbar_basic, 0, 2, 1, 2)
        basic_refiner_grid.addWidget(self.orig_canvas_basic, 1, 0, 1, 2)
        basic_refiner_grid.addWidget(self.curr_canvas_basic, 1, 2, 1, 2)
        basic_refiner_grid.addWidget(self.orig_barplot_navtbar, 2, 0, 1, 1)
        basic_refiner_grid.addWidget(self.orig_barplot, 2, 1, 1, 1)
        basic_refiner_grid.addWidget(self.curr_barplot, 2, 2, 1, 1)
        basic_refiner_grid.addWidget(self.curr_barplot_navtbar, 2, 3, 1, 1)
        
#  -------------------------------------------------------------------------  #
#                     ADVANCED REFINER RENDERING WIDGETS
#  -------------------------------------------------------------------------  #

    # Current map canvas (Image Canvas) 
        self.curr_canvas_advan = plots.ImageCanvas(cbar=False)
        self.curr_canvas_advan.fig.suptitle('Current', size='large')
        self.curr_canvas_advan.setMinimumSize(500, 500)
    
    # Navigation toolbar of current map canvas (Navigation Toolbar) 
        self.curr_navtbar_advan = plots.NavTbar.imageCanvasDefault(
            self.curr_canvas_advan, self)

    # Reset current phase (Action) [-> Navigation toolbar of current map]
        self.reset_phase_action = QW.QAction(
            style.getIcon('REFRESH'), 'Reset current phase')
        self.curr_navtbar_advan.insertSeparator(10)
        self.curr_navtbar_advan.insertAction(10, self.reset_phase_action)
        
    # Preview map canvas (Image Canvas)
        self.prev_canvas_advan = plots.ImageCanvas(cbar=False)
        self.prev_canvas_advan.fig.suptitle('Refined preview', size='large')
        self.prev_canvas_advan.setMinimumSize(500, 500)
        self.prev_canvas_advan.share_axis(self.curr_canvas_advan.ax)

    # Navigation toolbar of preview map canvas (Navigation Toolbar)
        self.prev_navtbar_advan = plots.NavTbar.imageCanvasDefault(
            self.prev_canvas_advan, self)
        
    # Draw ROI (Action) [-> Navigation Toolbar of preview map]
        self.draw_roi_action = QW.QAction(style.getIcon('ROI'), 'Draw ROI')
        self.draw_roi_action.setCheckable(True)
        self.prev_navtbar_advan.insertSeparator(10)
        self.prev_navtbar_advan.insertAction(10, self.draw_roi_action)

    # ROI selector widget (Rectangle Selector)
        self.roi_sel = plots.RectSel(
            self.prev_canvas_advan.ax, self.onRoiDrawn, btns=[1])

    # Original phase info (Group Area)
        orig_info_name = QW.QLabel()
        orig_info_pixels = QW.QLabel()
        orig_info_mode = QW.QLabel()
        orig_info_form = QW.QFormLayout()
        orig_info_form.setVerticalSpacing(5)
        orig_info_form.addRow('Phase: ', orig_info_name)
        orig_info_form.addRow('Pixels: ', orig_info_pixels)
        orig_info_form.addRow('Amount: ', orig_info_mode)
        orig_info_group = CW.GroupArea(orig_info_form, 'Original')

    # Current phase info (Group Area) 
        curr_info_name = QW.QLabel()
        curr_info_pixels = QW.QLabel()
        curr_info_mode = QW.QLabel()
        curr_info_form = QW.QFormLayout()
        curr_info_form.setVerticalSpacing(5)
        curr_info_form.addRow('Phase: ', curr_info_name)
        curr_info_form.addRow('Pixels: ', curr_info_pixels)
        curr_info_form.addRow('Amount: ', curr_info_mode)
        curr_info_group = CW.GroupArea(curr_info_form, 'Current')

    # Preview phase info (Group Area)
        prev_info_name = QW.QLabel()
        prev_info_pixels = QW.QLabel()
        prev_info_mode = QW.QLabel()
        prev_info_icon = QW.QLabel()
        prev_info_added = QW.QLabel()
        prev_info_added.setStyleSheet('QLabel {color: green;}')
        prev_info_removed = QW.QLabel()
        prev_info_removed.setStyleSheet('QLabel {color: red;}')

        prev_info_form = QW.QFormLayout()
        prev_info_form.setVerticalSpacing(5)
        prev_info_form.addRow('Phase: ', prev_info_name)
        prev_info_form.addRow('Pixels: ', prev_info_pixels)
        prev_info_form.addRow('Amount: ', prev_info_mode)

        prev_info_grid = QW.QGridLayout()
        prev_info_grid.setColumnStretch(0, 2)
        prev_info_grid.setColumnStretch(1, 1)
        prev_info_grid.setColumnStretch(2, 1)
        prev_info_grid.addLayout(prev_info_form, 0, 0, -1, 1)
        align = QC.Qt.AlignmentFlag(style.A_HCENTER | style.A_TOP)
        prev_info_grid.addWidget(prev_info_icon, 0, 1, 1, -1, align)
        prev_info_grid.addWidget(prev_info_added, 1, 1, 1, 1, align)
        prev_info_grid.addWidget(prev_info_removed, 1, 2, 1, 1, align)

        prev_info_group = CW.GroupArea(prev_info_grid, 'Preview')

    # Convenient advanced info labels list 
        self.advan_info = (
            orig_info_name, orig_info_pixels, orig_info_mode,
            curr_info_name, curr_info_pixels, curr_info_mode,
            prev_info_name, prev_info_pixels, prev_info_mode,
            prev_info_icon, prev_info_added, prev_info_removed
        )

    # [Advanced Refiner] rendering group
        advan_refiner_grid = QW.QGridLayout()
        advan_refiner_grid.setHorizontalSpacing(10)
        advan_refiner_grid.setRowStretch(1, 1)
        advan_refiner_grid.addWidget(self.curr_navtbar_advan, 0, 0, 1, 2)
        advan_refiner_grid.addWidget(self.prev_navtbar_advan, 0, 2, 1, 2)
        advan_refiner_grid.addWidget(self.curr_canvas_advan, 1, 0, 1, 2)
        advan_refiner_grid.addWidget(self.prev_canvas_advan, 1, 2, 1, 2)
        advan_refiner_grid.addWidget(orig_info_group, 2, 0, 1, 1)
        advan_refiner_grid.addWidget(curr_info_group, 2, 1, 1, 1)
        advan_refiner_grid.addWidget(prev_info_group, 2, 2, 1, 2)


#  -------------------------------------------------------------------------  #
#                             ADJUST LAYOUT
#  -------------------------------------------------------------------------  #

    # Left panel layout
        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(15)
        left_vbox.addWidget(sample_group)
        left_vbox.addWidget(kern_group)
        left_vbox.addWidget(settings_group)
        left_vbox.addWidget(legend_group, 1)
        left_vbox.addLayout(ctrl_widgets_grid)
        left_scroll = CW.GroupScrollArea(left_vbox)

    # Right panel layout
        self.mode_tabwid = CW.StyledTabWidget()
        basic_idx = self.mode_tabwid.wrap_tab(
            basic_refiner_grid, style.getIcon('CUBE'), 'BASIC MODE')
        advan_idx = self.mode_tabwid.wrap_tab(
            advan_refiner_grid, style.getIcon('GEAR'), 'ADVANCED MODE')
        basic_tip = 'Quick and simple refinement with mode filter'
        advan_tip = 'Class-wise advanced refinement with multiple binary filters'
        self.mode_tabwid.setTabToolTip(basic_idx, basic_tip)
        self.mode_tabwid.setTabToolTip(advan_idx, advan_tip)
        right_scroll = CW.GroupScrollArea(self.mode_tabwid)

    # Main layout
        main_layout = CW.SplitterLayout()
        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(right_scroll, 1)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # New map selected 
        self.select_map_btn.clicked.connect(self.onMapSelected)

    # Legend
        self.legend.itemSelectionChanged.connect(self.onPhaseSelected)
        self.legend.colorChangeRequested.connect(self.changePhaseColor)
        self.legend.itemRenameRequested.connect(self.renamePhase)
        self.legend.itemHighlightRequested.connect(self.highlightPhase)

    # Kernel selection
        self.kern_size_spbox.editingFinished.connect(self.ensureOddKernelSize)
        self.kern_shape_btns.selectionChanged.connect(self.onKernelChanged)
        self.kern_size_spbox.valueChanged.connect(self.onKernelChanged)
        self.skip_border_cbox.stateChanged.connect(self.onKernelChanged)

    # NoData control settings [Basic Refiner]
        self.nd_combox.clicked.connect(self.updateNoDataComboBox)
        self.nd_combox.currentIndexChanged.connect(
            lambda i: self.nd_thresh_spbox.setEnabled(i != -1))
        self.nd_clear_btn.clicked.connect(
            lambda: self.nd_combox.setCurrentIndex(-1))
        
    # Algorithm and advanced settings [Advanced Refiner]
        self.algm_combox.currentTextChanged.connect(self.onAlgorithmChanged)
        self.invert_mask_cbox.stateChanged.connect(self.updateAdvancedPreview)
        self.invert_roi_cbox.stateChanged.connect(self.updateAdvancedPreview)

    # Refiner mode swapping 
        self.mode_tabwid.currentChanged.connect(self.onRefinerModeSwapped)

    # Advanced toolbar actions (reset phase and draw ROI) [Advanced Refiner]
        self.draw_roi_action.toggled.connect(self.toggleRoiSelector)
        self.reset_phase_action.triggered.connect(self.resetPhase)
    
    # Control buttons
        self.apply_btn.clicked.connect(self.applyRefinement)
        self.cancel_btn.clicked.connect(self.resetMap)
        self.save_btn.clicked.connect(self.saveMap)


    @property
    def minmap(self) -> MineralMap:
        '''
        Get the current mineral map.

        Returns
        -------
        minmap : MineralMap
            Current mineral map.

        Raises
        ------
        TypeError
            Raised if the current mineral map is None.

        '''
        if (minmap := self._minmap) is None:
            raise TypeError('Mineral map is None.')
        return minmap


    def set_minmap(self, minmap: MineralMap) -> None:
        '''
        Set the current mineral map.

        Parameters
        ----------
        minmap : MineralMap
            New mineral map.
        
        '''
        self._minmap = minmap


    def has_minmap(self) -> bool:
        '''
        Check if a mineral map is currently selected.

        Returns
        -------
        bool
            Whether a mineral map is currently selected or not.

        '''
        return self._minmap is not None


    @property
    def minmap_backup(self) -> MineralMap:
        '''
        Get the current mineral map backup.

        Returns
        -------
        backup : MineralMap
            Current mineral map backup.

        Raises
        ------
        TypeError
            Raised if the current mineral map backup is None.

        '''
        if (backup := self._minmap_backup) is None:
            raise TypeError('Mineral map backup is None')
        return backup


    def set_minmap_backup(self, minmap: MineralMap) -> None:
        '''
        Set the current mineral map backup.

        Parameters
        ----------
        minmap : MineralMap
            New mineral map backup.
        
        '''
        self._minmap_backup = minmap
        
        
    def onMapSelected(self) -> None:
        '''
        Actions to be performed when a mineral map is selected.

        '''
    # Exit function if selection is invalid
        selected = self.minmap_selector.currentMap()
        if selected is None or (mmap := selected.getMineralMapData()) is None:
            return
        
    # Ask for user confirm if a mineral map is currently selected
        if self.has_minmap():
            text = 'Change map? Unsaved edits on current map will be lost.'
            choice = CW.MsgBox(self, 'QuestWarn', text)
            if choice.no():
                return
    
    # Change current mineral map with the one selected
        self.changeCurrentMap(mmap)
  

    def changeCurrentMap(self, minmap: MineralMap) -> None:
        '''
        Change the current mineral map with 'minmap'.

        Parameters
        ----------
        minmap : MineralMap
            New mineral map.

        '''
    # Reset refiner attributes
        self.set_minmap(minmap.copy())
        self.set_minmap_backup(minmap.copy())
        self._phase = None
    
    # Update widgets
        self.selected_map_lbl.setPath(minmap.filepath)
        self.legend.refresh(minmap)
        self.nd_combox.clear()

    # Render refiner widgets
        if self._refiner_mode == 0:
            self.renderBasic()
        else:
            self.renderAdvanced()

    
    def onPhaseSelected(self) -> None:
        '''
        Actions to be performed when a mineral phase is selected (left-clicked)
        in the legend.

        '''
    # Change the current phase attribute
        selected = self.legend.selectedItems()
        if len(selected):
            self._phase = selected[0].text(1)

    # Render only advanced refiner widgets
        if self._refiner_mode == 1:
            self.renderAdvanced()


    def changePhaseColor(
        self,
        legend_item: QW.QTreeWidgetItem,
        color: tuple[int, int, int]
    ) -> None:
        '''
        Alter the displayed color of a mineral phase. This method propagates
        the changes to the current map, the backup map, the basic refiner 
        widgets and the legend. The arguments of this method are specifically
        compatible with the 'colorChangeRequested' signal emitted by the legend 
        (see 'Legend' class for more details). 

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple[int, int, int]
            RGB triplet.

        '''
        # Update the item color in the legend
        self.legend.changeItemColor(legend_item, color)

        # Update phase color in current and backup maps and in basic widgets
        is_basic_mode = self._refiner_mode == 0
        phase = legend_item.text(1)

        if self.minmap_backup.has_phase(phase):
            self.minmap_backup.set_phase_color(phase, color)
            if is_basic_mode:
                colorlist_bkp = self.minmap_backup.palette.values()
                self.orig_canvas_basic.alter_colormap(colorlist_bkp)
                self.orig_barplot.change_bar_color(phase, color)

        if self.minmap.has_phase(phase):
            self.minmap.set_phase_color(phase, color)
            if is_basic_mode:
                colorlist = self.minmap.palette.values()
                self.curr_canvas_basic.alter_colormap(colorlist)
                self.curr_barplot.change_bar_color(phase, color)


    def renamePhase(self, legend_item: QW.QTreeWidgetItem, new_name: str) -> None:
        '''
        Rename a mineral phase. This method propagates the changes to the
        current map, the backup map, the refiner widgets and the legend. The 
        arguments of this method are specifically compatible with the 
        'itemRenameRequested' signal emitted by the legend (see 'Legend' class
        for more details).

        Parameters
        ----------
        legend_item : QTreeWidgetItem
            The legend item that requested to be renamed.
        new_name : str
            New class name.

        '''
    # Renamed phase is also the currently selected phase, so the current phase
    # attribute must be updated to avoid errors.
        self._phase = new_name

    # Rename phase in original and current mineral maps
        old_name = legend_item.text(1)
        if self.minmap_backup.has_phase(old_name):
            self.minmap_backup.rename_phase(old_name, new_name)
        if self.minmap.has_phase(old_name):
            self.minmap.rename_phase(old_name, new_name)

    # Rename phase name shown in legend
        self.legend.renameClass(legend_item, new_name)
    
    # Render refiner widgets
        if self._refiner_mode == 0:
            self.renderBasic()
        else:
            self.renderAdvanced()


    def highlightPhase(self, toggled: bool, legend_item: QW.QTreeWidgetItem) -> None:
        '''
        Highlight on/off the selected mineral phase in the basic refiner plots.
        The arguments of this method are specifically compatible with the 
        'itemHighlightRequested' signal emitted by the legend (see 'Legend'
        class for more details).

        Parameters
        ----------
        toggled : bool
            Highlight on/off
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be highlighted.

        '''
    # Exit function if in advanced mode
        if self._refiner_mode == 1:
            return

        if toggled:
            phase = legend_item.text(1) 

        # Compute current mineral map clims
            if self.minmap.has_phase(phase):
                vmin = vmax = self.minmap.as_id(phase)
            else:
                vmin = vmax = -1 # shows empty canvas
        
        # Compute original mineral map clims
            if self.minmap_backup.has_phase(phase):
                vmin_bkp = vmax_bkp = self.minmap_backup.as_id(phase)
            else:
                vmin_bkp = vmax_bkp = -1 # shows empty canvas

        else:
            vmin = vmax = None
            vmin_bkp = vmax_bkp = None

    # Update clims and render plots
        self.orig_canvas_basic.update_clim(vmin_bkp, vmax_bkp)
        self.curr_canvas_basic.update_clim(vmin, vmax)


    def updateLegendAmounts(self) -> None:
        '''
        Update the amounts of the mineral phases currently displayed in the
        legend. We use this method instead of 'legend.update()', which would 
        be the ideal method, because we do not want to remove from the legend
        the mineral phases that have been cleared out from the mineral map as a
        result of refinement operations. 

        '''
        prec = self.legend.precision
        for idx in range(self.legend.topLevelItemCount()):
            if (item := self.legend.topLevelItem(idx)) is None:
                continue

        # If a phase was completely removed from the currently edited mineral
        # map, just set its amount to 0.
            phase = item.text(1)
            if self.minmap.has_phase(phase):
                amount = round(self.minmap.get_phase_amount(phase), prec)
            else:
                amount = 0.0

            item.setText(2, f'{amount}%')


    def ensureOddKernelSize(self) -> None:
        '''
        Ensure that user-typed kernel size value is always odd.

        '''
        size = self.kern_size_spbox.value()
        if not size % 2:
            self.kern_size_spbox.setValue(size - 1)


    def onKernelChanged(self) -> None:
        '''
        Actions to be performed when kernel parameters are changed. 

        '''
        if self._refiner_mode == 1: 
            self.updateAdvancedPreview()


    def updateNoDataComboBox(self) -> None:
        '''
        Auto-update method for the NoData selection combo box.

        '''
        if self.minmap is not None:
            self.nd_combox.updateItems(self.minmap.get_phases())


    def onAlgorithmChanged(self, algorithm: str) -> None:
        '''
        Actions to be performed when the advanced algorithm is changed. 

        Parameters
        ----------
        algorithm : str
            Algorithm name.

        '''
        self.algm_warn.setVisible(algorithm == 'Fill Holes')
        self.updateAdvancedPreview()


    def onRefinerModeSwapped(self, mode: int) -> None:
        '''
        Actions to be performed when the refiner mode (basic or advanced) is 
        changed through the dedicated tab widget.

        Parameters
        ----------
        mode : int
            Refiner mode. Can only be '0' (Basic) or '1' (Advanced).

        '''

        self._refiner_mode = mode

    # Swap settings stack
        self.settings_stack.setCurrentIndex(mode)

    # Render refiner widgets
        if mode == 0:
            self.renderBasic()
        else:
            self.renderAdvanced()


    def renderBasic(self) -> None:
        '''
        Render basic refiner widgets, which include original and current maps
        canvases and corresponding bar plots.

        '''
    # Clear widgets if no mineral map is currently selected
        if not self.has_minmap():
            return self.clearView()
    
    # Update maps canvases
        mmap_enc_bkp, enc_bkp, col_bkp = self.minmap_backup.get_plot_data()
        mmap_enc, enc, col = self.minmap.get_plot_data()
        self.orig_canvas_basic.draw_discretemap(mmap_enc_bkp, enc_bkp, col_bkp)
        self.curr_canvas_basic.draw_discretemap(mmap_enc, enc, col)

    # Update bar plots
        lbl_bkp, mod_bkp = zip(*self.minmap_backup.get_labeled_mode().items())
        bar_col_bkp = [self.minmap_backup.get_phase_color(l) for l in lbl_bkp]
        lbl, mod = zip(*self.minmap.get_labeled_mode().items())
        bar_col = [self.minmap.get_phase_color(l) for l in lbl]
        self.orig_barplot.update_canvas((mod_bkp,), lbl_bkp, bar_col_bkp)
        self.curr_barplot.update_canvas((mod,), lbl, bar_col)
        
    
    def renderAdvanced(self) -> None:
        '''
        Render advanced refiner widgets, which include current and preview
        phase maps canvases and info widgets.

        '''
    # Clear widgets if no mineral map or mineral phase is currently selected
        if not self.has_minmap() or self._phase is None:
            return self.clearView()
        
    # Update current phase map canvas
        phasemap = self.minmap.minmap == self._phase
        encoder = {0: 'Background', 1: self._phase}
        colors = [(0, 0, 0), (255, 255, 255)]
        self.curr_canvas_advan.draw_discretemap(phasemap, encoder, colors, '')
        
    # Update preview phase map canvas and info widgets
        self.updateAdvancedPreview()


    def updateAdvancedPreview(self) -> None:
        '''
        Update advanced preview of current phase and advanced info widgets.

        '''
    # Do nothing if no mineral map or mineral phase is currently selected
        if not self.has_minmap() or self._phase is None:
            return
        
    # Compute refinement operations
        current_map = (self.minmap.minmap == self._phase)
        refined_map = self.computeAdvancedRefinement(current_map)
        variance_map = 2 * refined_map + current_map

    # Update preview canvas
        encoder = {
            0: 'Background',
            1: f'Removed {self._phase}', 
            2: f'Added {self._phase}',
            3: self._phase
        }
        self.prev_canvas_advan.draw_discretemap(
            variance_map, encoder, self._variance_colors, '')

    # Compute original phase info
        orig_pixels = np.count_nonzero((
            self.minmap_backup.minmap == self._phase))
        if orig_pixels == 0:
            orig_amount = 0.0
        else:
            orig_amount = self.minmap_backup.get_phase_amount(self._phase)

    # Compute current phase info
        curr_pixels = np.count_nonzero(current_map)
        if curr_pixels == 0:
            curr_amount = 0.0
        else:
            curr_amount = self.minmap.get_phase_amount(self._phase)
    
    # Compute preview phase info
        prev_pixels = np.count_nonzero(refined_map)
        prev_amount = 100 * prev_pixels / refined_map.size

        if curr_pixels > prev_pixels:
            icon = str(style.ICONS.get('CARET_DOWN_RED'))
        elif curr_pixels < prev_pixels:
            icon = str(style.ICONS.get('CARET_UP_GREEN'))
        else:
            icon = str(style.ICONS.get('CARET_DOUBLE_YELLOW'))

        prev_pixmap = QG.QPixmap(icon).scaled(48, 48, style.ARM_KEEP)
        prev_added = np.count_nonzero(variance_map == 2)
        prev_removed = np.count_nonzero(variance_map == 1)

    # Update info widgets
        prec = settings.manager.get('data/decimal_precision')
        # original name
        self.advan_info[0].setText(self._phase)
        # original pixels                  
        self.advan_info[1].setText(str(orig_pixels))
        # original amount  
        self.advan_info[2].setText(f'{round(orig_amount, prec)} %')
        # current name
        self.advan_info[3].setText(self._phase)
        # current pixels
        self.advan_info[4].setText(str(curr_pixels))
        # current amount
        self.advan_info[5].setText(f'{round(curr_amount, prec)} %')
        # preview name
        self.advan_info[6].setText(self._phase)
        # preview pixels
        self.advan_info[7].setText(str(prev_pixels))
        # preview amount
        self.advan_info[8].setText(f'{round(prev_amount, prec)} %')
        # preview icon
        self.advan_info[9].setPixmap(prev_pixmap)
        # preview added pixels
        self.advan_info[10].setText(f'+{prev_added}')
        # preview removed pixels
        self.advan_info[11].setText(f'-{prev_removed}')


    def clearView(self) -> None:
        '''
        Clear refiner widgets.

        '''
    # Basic Refiner cleaning operations 
        if self._refiner_mode == 0: 
            self.orig_canvas_basic.clear_canvas()
            self.curr_canvas_basic.clear_canvas()
            self.orig_barplot.clear_canvas()
            self.curr_barplot.clear_canvas()

    # Advanced Refiner cleaning operations 
        else:
            self.curr_canvas_advan.clear_canvas()
            self.curr_canvas_advan.ax.set_title('Select a phase', pad=-100)
            self.prev_canvas_advan.clear_canvas()
            self.prev_canvas_advan.ax.set_title('Select a phase', pad=-100)
            for lbl in self.advan_info:
                lbl.clear()


    def applyRefinement(self) -> None:
        '''
        Wrapper method to apply basic or advanced refinement operations.

        '''
    # Basic refinement operations
        if self._refiner_mode == 0:
            self.refineBasic()
            self.renderBasic()

    # Advanced refinement operations
        else:
            self.refineAdvanced()
            self.renderAdvanced()
        
    # Force edit legend amounts
        self.updateLegendAmounts()

    # Clear NoData combobox if its currently selected NaN phase was removed
    # from the mineral map after this refinement operation. 
        if not self.minmap.has_phase(self.nd_combox.currentText()):
            self.nd_combox.clear()


    def refineBasic(self) -> None:
        '''
        Edit the current mineral map using the result of basic refinement 
        operations.

        '''
        pbar = CW.PopUpProgBar(self, 3, 'Applying filter')

    # Compute basic refinement operations. An encoded refined map is returned.
        pbar.increase()
        refined_encoded = self.computeBasicRefinement()

    # Decode the refined mineral map
        refined = np.empty(refined_encoded.shape, dtype=self.minmap._DTYPE_STR)
        for _id, lbl in self.minmap.encoder.items():
            refined[refined_encoded == _id] = lbl
        pbar.increase()

    # Apply edits to current mineral map
        self.minmap.edit_minmap(refined, alter_probmap=True)
        pbar.increase()

        
    def refineAdvanced(self) -> None:
        '''
        Edit the current mineral map using the result of advanced refinement
        operations.

        '''
    # Get variance map from the advanced preview to compute phase masks
        preview = self.prev_canvas_advan.get_unmasked_array()
        removed_pixels_mask = preview == 1
        added_pixels_mask = preview == 2
    
    # Remove deleted phase pixels and add new phase pixels
        refined_map = self.replaceRemovedPixels(removed_pixels_mask)
        refined_map[added_pixels_mask] = self._phase

    # Edit current mineral map
        self.minmap.edit_minmap(refined_map, alter_probmap=True)

    # Add "_ND_" to legend if it was not present but has now been added to the 
    # current mineral map after calling the 'replaceRemovedPixels' method
        if self.minmap.has_phase('_ND_') and not self.legend.hasClass('_ND_'):
            color = self.minmap.get_phase_color('_ND_')
            amount = self.minmap.get_phase_amount('_ND_')
            self.legend.addClass('_ND_', color, amount)

    
    def replaceRemovedPixels(self, mask: np.ndarray) -> np.ndarray:
        '''
        Rename pixels removed from advanced refinement operations. Removed 
        pixels must be highlighted by 'mask'. 

        Parameters
        ----------
        mask : numpy ndarray
            Boolean array that highlights pixels that have been removed from
            the current mineral map and need renaming. It must have the same
            shape of the current mineral map.

        Returns
        -------
        minmap : numpy ndarray
            A modified version of the current mineral map array.

        '''
    # Get a copy of the current mineral map array and the renaming strategy
        minmap = self.minmap.minmap.copy()
        strategy = self.del_pixels_combox.currentText()

    # Nearest class strategy
        if strategy == 'Nearest class':
            minmap = iatools.replace_with_nearest(minmap, mask)
    # _ND_ strategy
        else:
            minmap[mask] = '_ND_'

        return minmap


    def computeBasicRefinement(self) -> np.ndarray:
        '''
        Return a refined encoded version of the current mineral map after the
        application of basic refinement operations. 

        Returns
        -------
        refined : numpy ndarray
            Refined encoded mineral map.

        '''
    # Construct kernel structure
        kern_shape = self.kern_shape_btns.get_checked().objectName()
        kern_size = self.kern_size_spbox.value()
        kernel = iatools.construct_kernel_filter(kern_shape, kern_size)

    # Apply max frequency (mode) filter
        mmap_enc = self.minmap.minmap_encoded
        nan_phase = self.nd_combox.currentText()
        nan_id = None if nan_phase == '' else self.minmap.as_id(nan_phase)
        nan_thr = self.nd_thresh_spbox.value()
        refined = iatools.apply_mode_filter(mmap_enc, kernel, nan_id, nan_thr)

    # Preserve borders if required
        if self.skip_border_cbox.isChecked():
            refined = self.preserveMapBorders(refined, mmap_enc, kern_size//2)

        return refined


    def computeAdvancedRefinement(self, phasemap: np.ndarray) -> np.ndarray:
        '''
        Return a refined version of the input 'phasemap' after the application
        of advanced refinement operations.

        Parameters
        ----------
        phasemap : numpy ndarray
            Boolean map of the currently selected phase.

        Returns
        -------
        refined : numpy ndarray
            Refined phase map.

        '''

    # Construct kernel structure
        kern_shape = self.kern_shape_btns.get_checked().objectName()
        kern_size = self.kern_size_spbox.value()
        kernel = iatools.construct_kernel_filter(kern_shape, kern_size)
        
    # Get ROI mask if required ('Fill Holes' always ignores the ROI)
        algm = cast(iatools.BinaryMorphologyAlgorithms, self.algm_combox.currentText())
        roi = self.constructRoiMask() if algm != 'Fill Holes' else None
        if self.invert_roi_cbox.isChecked() and roi is not None:
            roi = np.invert(roi)

    # Phase mask inversion (if required) only affects morphology operation. The
    # result must then be inverted back. 
        if self.invert_mask_cbox.isChecked():
            refined = np.invert(iatools.apply_binary_morph(
                np.invert(phasemap), algm, kernel, roi))
        else:
            refined = iatools.apply_binary_morph(phasemap, algm, kernel, roi)

    # Preserve borders if required
        if self.skip_border_cbox.isChecked():
            refined = self.preserveMapBorders(refined, phasemap, kern_size//2)

        return refined

    
    def preserveMapBorders( # maybe should be moved to image_analysis_tools.py
        self,
        refined: np.ndarray,
        original: np.ndarray, 
        thickness: int
    ) -> np.ndarray:
        '''
        Return a modified version of 'refined' mineral map, where the pixels
        at the border are replaced by the corresponding ones in the 'original'
        mineral map. The border thickness is controlled by 'thickness'.

        Parameters
        ----------
        refined : numpy ndarray
            Refined mineral map. Must have the same shape of 'original'.
        original : numpy ndarray
            Original mineral map. Must have the same shape of 'refined'.
        thickness : int
            Border thickness.

        Returns
        -------
        numpy ndarray
            Refined mineral map with preserved borders.

        '''
        if thickness <= 0:
            return refined
        
        border_mask = np.zeros(original.shape)
        border_mask[thickness:-thickness, thickness:-thickness] = 1
        return np.where(border_mask, refined, original)
    

    def toggleRoiSelector(self, toggled: bool) -> None:
        '''
        Toggle on/off the ROI rectangle selector.

        Parameters
        ----------
        toggled : bool
            Toggle the ROI selector on (True) or off (False).

        '''
    # Toggle on/off and update the ROI rectangle selector widget
        self.roi_sel.set_active(toggled)
        self.roi_sel.set_visible(toggled)
        self.roi_sel.update()

    # Also update the advanced preview if a valid ROI is/was set
        if self._roi_extents is not None:
            self.updateAdvancedPreview()


    def onRoiDrawn(
        self,
        eclick: plots.mpl_backend_bases.MouseEvent,
        erelease: plots.mpl_backend_bases.MouseEvent
    ) -> None:
        '''
        Callback function for the ROI rectangle selector. It is triggered when
        selection is performed by the user (left mouse button click-release).

        Parameters
        ----------
        eclick : Matplotlib MouseEvent
            Mouse click event.
        erelease : Matplotlib MouseEvent
            Mouse release event.

        '''
    # Do nothing if the advanced preview canvas is empty
        if self.prev_canvas_advan.is_empty():
            return
    
    # Update the ROI extents attribute and the advanced preview
        rows, cols = self.minmap.shape
        self._roi_extents = self.roi_sel.fixed_extents((rows, cols))
        self.updateAdvancedPreview()
    

    def constructRoiMask(self) -> np.ndarray | None:
        '''
        Construct a ROI mask from the user-drawn ROI. The mask is used for the
        advanced refiner operations.

        Returns
        -------
        numpy ndarray or None
            The ROI mask or None if no valid ROI is drawn.

        '''
    # ROI mask is invalid if ROI extents are invalid or ROI selector is off 
        if self._roi_extents is None or not self.roi_sel.active:
            return None

    # Construct the ROI mask array from the current ROI extents
        r0, r1, c0, c1 = self._roi_extents
        roi_mask = np.zeros(self.minmap.shape, dtype=bool)
        roi_mask[r0:r1, c0:c1] = True
        return roi_mask


    def resetPhase(self) -> None:
        '''
        Cancel all refinement operations made on the currently selected phase.
        This include added and removed (renamed) pixels.

        '''
    # Do nothing if currently selected phase is invalid
        if self._phase is None:
            return

    # Wherever the phase is present (original or current map), reset to original
        orig_minmap = self.minmap_backup.minmap
        curr_minmap = self.minmap.minmap
        mask = (orig_minmap == self._phase) | (curr_minmap == self._phase)
        restored = np.where(mask, orig_minmap, curr_minmap)
        self.minmap.edit_minmap(restored, alter_probmap=False)
    
    # Also restore the original probability map values, since those are altered
    # when phases are refined (see 'alter_probmap' argument of 'edit_minmap'). 
        self.minmap.probmap[mask] = self.minmap_backup.probmap[mask]

    # Render advanced refiner widgets
        self.renderAdvanced()

    # Force edit legend amounts
        self.updateLegendAmounts()


    def resetMap(self) -> None:
        '''
        Cancel all the refinement operations made on the current mineral map. 

        '''
    # Do nothing if the current mineral map has not been edited
        if self.minmap == self.minmap_backup:
            return
    
    # Ask for user confirm
        choice = CW.MsgBox(self, 'Quest', 'Discard all refinements?')
        if choice.no():
            return
    
    # Reset map and render refiner widgets
        self.set_minmap(self.minmap_backup.copy())
        if self._refiner_mode == 0:
            self.renderBasic()
        else:
            self.renderAdvanced()


    def saveMap(self) -> None:
        '''
        Save current refined mineral map to disk.

        '''
    # Ask for output path
        ftype = 'Mineral map (*.mmp)'
        outpath = CW.FileDialog(self, 'save', 'Save Map', ftype).get_path()
        if not outpath:
            return
        
    # Send success or error message
        try:
            self.minmap.save(outpath)
            CW.MsgBox(self, 'Info', 'Map saved successfully.')
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save map.', str(e))

    # deprecated
    # def killReferences(self) -> None:
    #     '''
    #     Reimplementation of the 'killReferences' method from the parent class.
    #     It also deletes the ROI selector, thus disconnecting its signals.

    #     '''
    #     del self.roi_sel
    #     super().killReferences()

    



class DataViewer(QW.QWidget):

    def __init__(self) -> None:
        '''
        The main tool of X-Min Learn, that allows the visualization of imported
        data. Unlike any other tool, it cannot be moved or closed by users.

        '''
        super().__init__()

    # Set tool title and icon
        self.setWindowTitle('Data Viewer')
        self.setWindowIcon(style.getIcon('DATA_VIEWER'))

    # Set main attributes
        self._displayed: CW.DataObject | None = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Maps Canvas (Image Canvas)
        self.canvas = plots.ImageCanvas()
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.enable_picking(True)
        self.canvas.setMinimumWidth(500)

    # Navigation Toolbar (Navigation Toolbar)
        self.navtbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # Pixel Finder (Coordinate Finder) [-> Navigation Toolbar widget]
        self.pixel_finder = CW.CoordinatesFinder()
        self.navtbar.addSeparator()
        self.navtbar.addWidget(self.pixel_finder)

    # Current showed map path (Path Label)
        self.curr_path = CW.PathLabel()

    # Adjust Window Layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navtbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.curr_path)
        self.setLayout(main_layout)

    
    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.pixel_finder.coordinatesRequested.connect(self.canvas.zoom_to)


    def showContextMenu(self, point: QC.QPoint) -> None:
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



# [ -------------------- !!! CURRENTLY NOT IMPLEMENTED ---------------------- ]
# class PixelEditor(QW.QWidget): 
#     def __init__(self, XMapsInfo, originalData, edits, parent):
#         self.parent = parent
#         self.XMapsPath, self.XMapsData = XMapsInfo
#         super(PixelEditor, self).__init__()
#         self.setWindowTitle('Pixel Editor')
#         self.setWindowIcon(QG.QIcon('Icons/edit.png'))
#         self.setAttribute(QC.Qt.WA_QuitOnClose, False)
#         self.setAttribute(QC.Qt.WA_DeleteOnClose, True)

#         self.editsDict = edits
#         self.originalData = originalData
#         self.editedData = self.originalData.copy()
#         for (x,y), val in self.editsDict.items():
#             self.editedData[x,y] = val
#         self.tempResult = None

#         self.init_ui()
#         self.adjustSize()
#         self.update_preview()

#     def init_ui(self):

#         # Edited Pixels Preview Area
#         self.editPreview = CW.HeatMapCanvas(size=(6, 4.5), binary=True, tight=True,
#                                               cbar=False, wheel_zoom=False)
#         self.editPreview.setMinimumSize(100, 100)

#         # Progress bar
#         self.progBar = QW.QProgressBar()

#         # Current edited value combo box
#         self.currEdit_combox = QW.QComboBox()
#         self.currEdit_combox.addItems(set(self.editsDict.values()))
#         self.currEdit_combox.currentTextChanged.connect(self.current_preview)

#         # Preview button
#         self.showPreview_btn = QW.QPushButton('Preview')
#         self.showPreview_btn.setToolTip('Show preview (it may require some time).')
#         self.showPreview_btn.clicked.connect(self.update_preview)

#         # Save Button
#         self.save_btn = CW.IconButton('Icons/save.png', 'Save Edits')
#         self.save_btn.clicked.connect(self.save_edits)

#         # Training Mode Preferences
#         self.tol_spbox = QW.QSpinBox()
#         self.tol_spbox.setToolTip('Select the pixel tolerance for computation.')
#         self.tol_spbox.setValue(15)
#         self.tol_spbox.setRange(0, 10000)
#         self.tol_spbox.valueChanged.connect(self.reset_tempResult)
#         tolerance = QW.QFormLayout()
#         tolerance.addRow('Set Tolerance', self.tol_spbox)

#         self.proximity_cbox = QW.QCheckBox('Evaluate Proximity')
#         self.proximity_cbox.setToolTip('Include pixel proximity effect in computation.')
#         self.proximity_cbox.setChecked(True)
#         self.proximity_cbox.clicked.connect(self.reset_tempResult)

#         trainBox = QW.QVBoxLayout()
#         trainBox.addLayout(tolerance)
#         trainBox.addWidget(self.proximity_cbox)
#         self.trainGA = CW.GroupArea(trainBox, 'Training Mode', checkable=True)
#         self.trainGA.setChecked(False)
#         self.trainGA.toggled.connect(self.toggleTrainMode)
#         # self.trainGA.setMaximumWidth(180)

#         # Input Maps Checkbox
#         self.CboxMaps = CW.CBoxMapLayout(self.XMapsPath)
#         for cbox in self.CboxMaps.Cbox_list:
#             cbox.clicked.connect(self.reset_tempResult)
#         MapsGSA = CW.GroupScrollArea(self.CboxMaps, 'Input Maps')
#         # MapsGSA.setMaximumWidth(180)

#         # Adjust Layout
#         rightVbox = QW.QVBoxLayout()
#         rightVbox.addWidget(self.currEdit_combox)
#         rightVbox.addWidget(self.showPreview_btn)
#         rightVbox.addWidget(self.save_btn)
#         rightVbox.addWidget(self.progBar)
#         rightVbox.addStretch()
#         rightVbox.addWidget(self.trainGA)

#         layout = QW.QHBoxLayout()
#         layout.addWidget(MapsGSA, 1)
#         layout.addWidget(self.editPreview, 2)
#         layout.addLayout(rightVbox, 1)
#         self.setLayout(layout)


#     def reset_tempResult(self):
#         self.tempResult = None

#     def save_edits(self):
#         if self.tempResult is None:
#             self.update_preview()

#         editedMap = self.tempResult
#         outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save edited map',
#                                                     settings.get_dir('out'),
#                                                     '''Compressed ASCII file (*.gz)
#                                                        ASCII file (*.txt)''')
#         if outpath:
#             settings.set_dir('out', os.path.dirname(outpath))
#             np.savetxt(outpath, editedMap, fmt='%s')
#             QW.QMessageBox.information(self, 'X-Min Learn',
#                                        'Edited map saved with success.')

#     def mapsFitting(self, maps):
#         # Check for maps fitting errors
#         shapes = [arr.shape for arr in maps]
#         for shp in shapes:
#             if shp != self.originalData.shape: #there are different map shapes
#                 QW.QMessageBox.critical(self, 'X-Min Learn',
#                                         'The selected maps have different shapes.')
#                 return (False, None)
#         else:
#             return (True, self.originalData.shape)


#     def calc_pixAffinity(self):
#     # Get affinity parameters
#         tol = self.tol_spbox.value()
#         prox = self.proximity_cbox.isChecked()

#     # Get input maps data
#         xmaps_cbox = filter(lambda cbox: cbox.isChecked(), self.CboxMaps.Cbox_list)
#         xmaps_data = [self.XMapsData[int(cbox.objectName())] for cbox in xmaps_cbox]
#         n_maps = len(xmaps_data)

#     # Check that input maps have same shape
#         mapsfit, shape2D = self.mapsFitting(xmaps_data)
#         if mapsfit:
#             self.progBar.setMaximum(len(self.editsDict))
#         # Build a 3D maps data stacked array (-> shape = (n_rows x n_cols x n_maps))
#             xmaps_stack = np.dstack(xmaps_data)
#         # Build a boolean mask highlighting pixels edited by user
#             edit_mask = self.editedData != self.originalData
#         # Initialize the output map as a copy of the original mineral map
#             outData = self.originalData.copy()
#         # Initialize the minimum variance 2D matrix with the Maximum Accepted Tolerance (MAT) + 1
#         # The MAT is equal to the tolerance value set by user multiplied by the number of input maps
#             min_variance = np.ones(outData.shape) * ((tol*n_maps)+1)

#         # START ITERATION for each edited pixel (r and c are the i-th pixel indices (row and column))
#             for n, (r, c) in enumerate(zip(*np.where(edit_mask)), start=1):
#             # Get the edited pixel value through its indices
#                 value = self.editedData[r, c]
#             # Get the rows and columns indices (both 2D matrix (n_rows x n_cols)) of a single input map
#             # in the stack (they are the same for each map, because maps have the same shape)
#                 rows_idx, cols_idx = np.indices(xmaps_stack.shape[:2])
#             # Confront each row and column index with those of the i-th edited pixel in the iteration.
#             # 'near_idx' will be a 2D matrix (n_rows x n_cols) storing the proximity values of every
#             # pixel with respect to the i-th edited pixel in the iteration. 'prox' is a boolean value
#             # referred to user's request to include proximity calculation. If False it will set the
#             # near_idx to 0 otherwise it will not affect the result (it multiplies it by 1)
#                 near_idx = prox * tol/(((rows_idx - r)**2 + (cols_idx - c)**2)**0.5 + 1)
#             # Copy and paste the same near_idx matrix for each map, since the maps are aligned.
#             # 'near_idx3D' will be a 3D matrix of shape (n_rows x n_cols x n_maps)
#                 near_idx3D = np.repeat(near_idx[:,:,np.newaxis], n_maps, axis=2)
#             # The variance value is computed as the absolute difference between the 3D input maps
#             # stack (n_rows x n_cols x n_maps) and the voxel (1 x 1 x n_maps) referred to the i-th
#             # edited pixel in the iteration. The near_idx value will then be subtracted from it.
#                 variance = np.abs(xmaps_stack - xmaps_stack[r, c, :]) - near_idx3D
#             # Fitting is a boolean 2D matrix (n_rows x n_cols) that indicates if for all input maps (-> axis=2)
#             # a variance voxel satisfy the condition: variance <= tolerance threshold set by user.
#                 fitting = np.all(variance <= tol, axis=2)
#             # Transform the fitting matrix in a 3D matrix as done for the 'near_idx' matrix.
#             # The 'fitting3D' matrix has the following shape: (n_rows x n_cols x n_maps)
#                 fitting3D = np.repeat(fitting[:, :, np.newaxis], n_maps, axis=2)
#             # Sum the variance values along the voxels that actually satisfy the condition imposed by the
#             # fitting3D matrix (i.e., the variance value for each map is <= the tolerance threshold)
#                 tot_variance = np.where(fitting3D, variance, np.nan).sum(axis=2)
#             # Edit the output result with the i-th pixel value only where the variance sum is minor
#             # than the minimum founded variance.
#                 outData = np.where(tot_variance < min_variance, value, outData)
#             # Refresh the minimum variance
#                 min_variance = np.fmin(min_variance, tot_variance)

#                 self.progBar.setValue(n)

#         else:
#             outData = self.editedData.copy()

#         self.progBar.reset()
#         return outData

#     def calc_boolMap(self, current):
#         isCurrent = self.tempResult == current
#         isEdited = self.tempResult != self.originalData
#         boolMap = isCurrent & isEdited
#         return boolMap

#     def toggleTrainMode(self, state):
#         if state and len(self.CboxMaps.Cbox_list) == 0:
#              QW.QMessageBox.warning(self, 'X-Min Learn',
#    "Train mode cannot be enabled if no x-ray maps are loaded. Please load the maps in the 'X-Ray Maps' tab.")
#              self.trainGA.setChecked(False)

#     def update_preview(self):
#         if self.trainGA.isChecked():
#             self.tempResult = self.calc_pixAffinity()
#         else:
#             self.tempResult = self.editedData.copy()

#         curr_edit = self.currEdit_combox.currentText()
#         self.current_preview(curr_edit)

#     def current_preview(self, current):
#         boolMap = self.calc_boolMap(current)
#         title = f'Preview\nN. of {current} pixels = {np.count_nonzero(boolMap)}'
#         self.editPreview.update_canvas(title, boolMap)

#     def closeEvent(self, event):
#         choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                          "Do you really want to close the editor window?",
#                                          QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                          QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:
#             event.accept()
#         else:
#             event.ignore()