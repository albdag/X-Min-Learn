# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:31:39 2023

@author: albdag
"""
import gc

from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW

import conv_functions as CF
import customObjects as cObj
import docks
import preferences as pref
import tools

class MainWindow(QW.QMainWindow):

    def __init__(self):
        '''Constructor.'''
        super(MainWindow, self).__init__()

    # Set main window properties
        self.resize(1600, 900)
        self.setWindowTitle('X-Min Learn - Alpha version')
        self.setWindowIcon(QG.QIcon(r'Icons/XML_logo.png'))
        self.setDockOptions(self.AnimatedDocks | self.AllowTabbedDocks)
        self.statusBar()
        self.setStyleSheet(pref.SS_mainWindow)

    # Initialize GUI
        self._init_ui()
    
    # Connect signals with slots
        self._connect_slots()

    # Show window in full screen
        self.showMaximized()


    def _init_ui(self):
        '''GUI constructor.'''
    # Data Viewer
        self.dataViewer = tools.DataViewer()

    # Central Widget (Tab Widget)
        self.tabWidget = MainTabWidget(self)
        self.tabWidget.addTab(self.dataViewer)

    # Hide the close button from the data viewer tab
        self.tabWidget.tabBar().tabButton(0, QW.QTabBar.RightSide).hide()

    # Set the central widget
        self.setCentralWidget(self.tabWidget)

    # Initialize panes
        self._init_panes()

    # Initialize menu/toolbar actions
        self._init_actions()

    # Initialize toolbars
        self._init_toolbars()

    # Initialize menu
        self._init_menu()


    def _init_panes(self):
        '''Initialize window panes (QDockWidgets).'''
    # Data Manager
        self.dataManager = docks.DataManager()

    # Input Maps Histogram Viewer
        self.histViewer = docks.HistogramViewer(self.dataViewer.canvas)

    # Mineral Maps Mode Viewer
        self.modeViewer = docks.ModeViewer(self.dataViewer.canvas)

    # ROI Editor
        self.roiEditor = docks.RoiEditor(self.dataViewer.canvas)

    # Probability Maps Viewer
        self.pmapViewer = docks.ProbabilityMapViewer(self.dataViewer.canvas)
        # Share pmap and data viewer axis
        # CF.shareAxis(self.pmapViewer.canvas.ax, self.dataViewer.canvas.ax)
        self.dataViewer.canvas.share_axis(self.pmapViewer.canvas.ax)

    # RGBA Composite Maps Viewer
        self.rgbaViewer = docks.RgbaCompositeMapViewer()
        # Share rgba and data viewer axis
        # CF.shareAxis(self.rgbaViewer.canvas.ax, self.dataViewer.canvas.ax)
        self.dataViewer.canvas.share_axis(self.rgbaViewer.canvas.ax)

    # Create panes 
        manager_pane = docks.Pane(self.dataManager, 'Data Manager', 
                                  QG.QIcon(r'Icons/data_manager.png'), False)
        histogram_pane = docks.Pane(self.histViewer, 'Histogram',
                                    QG.QIcon(r'Icons/histogram.png'))
        mode_pane = docks.Pane(self.modeViewer, 'Mode',
                               QG.QIcon(r'Icons/mode.png'))
        roi_pane = docks.Pane(self.roiEditor, 'ROI Editor',
                              QG.QIcon(r'Icons/roi.png'))
        probmap_pane = docks.Pane(self.pmapViewer, 'Probability Map',
                                  QG.QIcon(r'Icons/probmap.png'))
        rgba_pane = docks.Pane(self.rgbaViewer, 'RGBA Map',
                               QG.QIcon(r'Icons/rgba.png'))
        
        self.panes = (manager_pane, histogram_pane, mode_pane, roi_pane, 
                      probmap_pane, rgba_pane)
        
    # Store the panes toggle view actions and set their icons
        self.panes_tva = []
        for p in self.panes:
            action = p.toggleViewAction()
            if p.icon is not None:
                action.setIcon(p.icon)
                action.setIconVisibleInMenu(True)
            self.panes_tva.append(action)

    # Add panes to main window
        self.addPane(QC.Qt.LeftDockWidgetArea, manager_pane)
        self.addPane(QC.Qt.LeftDockWidgetArea, histogram_pane, visible=False)
        self.addPane(QC.Qt.LeftDockWidgetArea, mode_pane, visible=False)
        self.tabifyDockWidget(manager_pane, histogram_pane)
        self.tabifyDockWidget(manager_pane, mode_pane)
        self.addPane(QC.Qt.RightDockWidgetArea, roi_pane, visible=False)
        self.addPane(QC.Qt.RightDockWidgetArea, probmap_pane, visible=False)
        self.addPane(QC.Qt.RightDockWidgetArea, rgba_pane, visible=False)
        self.tabifyDockWidget(roi_pane, probmap_pane)
        self.tabifyDockWidget(roi_pane, rgba_pane)

    # Resize panes
        w = max([p.trueWidget().minimumSizeHint().width() for p in self.panes])
        self.resizeDocks(self.panes, [w] * len(self.panes), QC.Qt.Horizontal)
        
    def _init_actions(self):
        '''Initialize actions shared by menu and toolbars.'''
    # Quit app action
        self.close_action = QW.QAction('&Exit')
        self.close_action.setShortcut('Ctrl+Q')

    # Import X-Ray Maps Action 
        self.load_inmaps_action = QW.QAction(QG.QIcon(r'Icons/inmap.png'),
                                             '&Input Maps')
        self.load_inmaps_action.setShortcut('Ctrl+I')

    # Import Mineral Maps Action
        self.load_minmaps_action = QW.QAction(QG.QIcon(r'Icons/minmap.png'),
                                              '&Mineral Maps')
        self.load_minmaps_action.setShortcut('Ctrl+M')

    # Import Masks Action
        self.load_masks_action = QW.QAction(QG.QIcon(r'Icons/mask.png'),
                                            'Masks')

    # Launch Preferences Action
        self.pref_action = QW.QAction(QG.QIcon('Icons/wrench.png'),
                                      '&Preferences')
        self.pref_action.setShortcut('Ctrl+P')

    # Launch Convert Grayscale to ASCII Action
        self.conv2ascii_action = QW.QAction('Grayscale to Input Map')
        self.conv2ascii_action.setStatusTip('Convert grayscale image to '\
                                            'input map')

    # Launch Convert RGB to Mineral Maps Action
        self.conv2mmap_action = QW.QAction('RGB to Mineral Map')
        self.conv2mmap_action.setStatusTip('Convert RGB image to Mineral Map')

    # Launch Build Dummy Maps Action
        self.dummy_map_action = QW.QAction('Generate &Dummy Maps')
        self.dummy_map_action.setStatusTip('Build placeholder noisy maps')

    # Launch Sub-sample Dataset Action
        self.subsample_ds_action = QW.QAction('&Sub-sample dataset')
        self.subsample_ds_action.setStatusTip('Extract sub-datasets from an '\
                                              'existent dataset')

    # Launch Merge Datasets Action
        self.merge_ds_action = QW.QAction('&Merge datasets')
        self.merge_ds_action.setStatusTip('Merge multiple datasets')

    # Launch Dataset Builder Action
        self.ds_builder_action = QW.QAction(QG.QIcon(r'Icons/compile_dataset.png'),
                                            'Dataset &Builder')
        self.ds_builder_action.setShortcut('Ctrl+Alt+B')
        self.ds_builder_action.setStatusTip('Compile ground truth datasets')

    # Launch Model Learner Action
        self.model_learner_action = QW.QAction(QG.QIcon(r'Icons/merge.png'),
                                               'Model &Learner')
        self.model_learner_action.setShortcut('Ctrl+Alt+L')
        self.model_learner_action.setStatusTip('Build machine learning models')

    # Launch Mineral Classifier Action
        self.classifier_action = QW.QAction(QG.QIcon(r'Icons/classify.png'),
                                            'Mineral &Classifier', self)
        self.classifier_action.setShortcut('Ctrl+Alt+C')
        self.classifier_action.setStatusTip('Predict mineral maps')

    # Launch Phase Refiner Action
        self.refiner_action = QW.QAction(QG.QIcon(r'Icons/refine.png'),
                                         'Phase &Refiner', self)
        self.refiner_action.setShortcut('Ctrl+Alt+R')
        self.refiner_action.setStatusTip('Use morphological image processing '\
                                         'tools to refine mineral maps.')


    def _init_toolbars(self):
        '''Initialize main toolbars.'''
    # Tools toolbar
        self.tools_toolbar = QW.QToolBar('Tools toolbar')
        self.tools_toolbar.setFloatable(False)
        self.tools_toolbar.setStyleSheet(pref.SS_mainToolbar)
        # import data actions (button menu), followed by a separator
        self.tools_toolbar.addActions((self.ds_builder_action, 
                                       self.model_learner_action, 
                                       self.classifier_action, 
                                       self.refiner_action))
        # separator followed by preference dialog
        # other future tools (e.g. map algebra calculator)
    
    # Panes toolbar
        self.panes_toolbar = QW.QToolBar('Panes toolbar')
        self.panes_toolbar.setFloatable(False)
        self.panes_toolbar.setStyleSheet(pref.SS_mainToolbar)
        self.panes_toolbar.addActions(self.panes_tva)

    # Set default toolbars position in the window
        self.addToolBar(QC.Qt.LeftToolBarArea, self.panes_toolbar)
        self.addToolBar(QC.Qt.LeftToolBarArea, self.tools_toolbar)


    def _init_menu(self):
        '''Initialize main menu.'''
    # Set custom stylesheet to menu bar
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(pref.SS_menuBar + pref.SS_menu)

    # File Menu
        file_menu = menu_bar.addMenu('&File')
        import_submenu = file_menu.addMenu(QG.QIcon(r'Icons/import.png'),
                                           '&Import...')
        import_submenu.addActions((self.load_inmaps_action, 
                                   self.load_minmaps_action,
                                   self.load_masks_action))
        file_menu.addActions((self.pref_action, self.close_action))

    # Dataset Menu
        dataset_menu = menu_bar.addMenu('&Dataset')
        dataset_menu.addActions((self.ds_builder_action, 
                                 self.subsample_ds_action, 
                                 self.merge_ds_action))
        
    # Classification Menu
        class_menu = menu_bar.addMenu('&Classification')
        class_menu.addActions((self.classifier_action, 
                               self.model_learner_action))

    # Post-classification Menu
        postclass_menu = menu_bar.addMenu('&Post-classification')
        postclass_menu.addAction(self.refiner_action)

    # Utility Menu
        utility_menu = menu_bar.addMenu('&Utility')
        convert_submenu = utility_menu.addMenu('&Convert')
        convert_submenu.addActions((self.conv2ascii_action, 
                                    self.conv2mmap_action))
        utility_menu.addAction(self.dummy_map_action)

    # View Menu
        view_menu = menu_bar.addMenu('&View')
        panes_submenu = view_menu.addMenu('&Panes')
        panes_submenu.addActions(self.panes_tva)
        view_menu.addSeparator()
        view_menu.addActions((self.panes_toolbar.toggleViewAction(),
                             self.tools_toolbar.toggleViewAction()))
        

    def _connect_slots(self):
        '''Signal-slot connector.'''
    # Connect all panes toggle view actions with a custom slot to force showing
    # them when they are tabified
        for p, a, in zip(self.panes, self.panes_tva):
            a.toggled.connect(lambda t, p=p: self.setPaneVisibility(p, t))

    # Data Manager pane actions 
        self.dataManager.updateSceneRequested.connect(self.update_scene)
        self.dataManager.clearSceneRequested.connect(self.clear_scene)
        self.dataManager.rgbaChannelSet.connect(self.set_rgba_channel)

    # Mode Viewer pane actions
        self.modeViewer.updateSceneRequested.connect(self.update_scene)

    # ROI Editor pane actions
        self.roiEditor.rectangleSelectorUpdated.connect(self.updateHistogram)

        self.roiEditor.autoroi_dial.maps_selector.sampleUpdateRequested.connect(
            lambda: self.roiEditor.autoroi_dial.maps_selector.updateCombox(
                self.dataManager.getAllGroups()))
        
        self.roiEditor.autoroi_dial.maps_selector.mapsUpdateRequested.connect(
            lambda idx: self.roiEditor.autoroi_dial.maps_selector.updateList(
                self.dataManager.topLevelItem(idx)))
        
    # Quit app 
        self.close_action.triggered.connect(self.close)

    # Import X-Ray Maps  
        self.load_inmaps_action.triggered.connect(lambda: self.load('inmaps'))

    # Import Mineral Maps 
        self.load_minmaps_action.triggered.connect(lambda: self.load('minmaps'))

    # Import Masks 
        self.load_masks_action.triggered.connect(lambda: self.load('masks'))

    # Launch Preferences 
        self.pref_action.triggered.connect(
            lambda: self.launch_dialog('Preferences'))

    # Launch Convert Grayscale images to ASCII 
        self.conv2ascii_action.triggered.connect(
            lambda: self.launch_dialog('Grayscale2Ascii'))

    # Launch Convert RGB yo Mineral Maps 
        self.conv2mmap_action.triggered.connect(
            lambda: self.launch_dialog('Rgb2Minmap'))

    # Launch Build Dummy Maps 
        self.dummy_map_action.triggered.connect(
            lambda: self.launch_dialog('DummyMaps'))

    # Launch Sub-sample Dataset 
        self.subsample_ds_action.triggered.connect(
            lambda: self.launch_dialog('SubSampleDataset'))

    # Launch Merge Datasets 
        self.merge_ds_action.triggered.connect(
            lambda: self.launch_dialog('MergeDatasets'))

    # Launch Dataset Builder 
        self.ds_builder_action.triggered.connect(
            lambda: self.launch_tool('DatasetBuilder'))

    # Launch Model Learner 
        self.model_learner_action.triggered.connect(
            lambda: self.launch_tool('ModelLearner'))

    # Launch Mineral Classifier 
        self.classifier_action.triggered.connect(
            lambda: self.launch_tool('MineralClassifier'))

    # Launch Phase Refiner 
        self.refiner_action.triggered.connect(
            lambda: self.launch_tool('PhaseRefiner'))


    def addPane(self, dockWidgetArea, pane, visible=True):
        self.addDockWidget(dockWidgetArea, pane)
        pane.setVisible(visible)

    
    def setPaneVisibility(self, pane: docks.Pane, visible: bool):
        '''
        Set the visibility of a pane. Allows to bring the pane on top if it is 
        already opened but tabified.

        Parameters
        ----------
        pane : docks.Pane
            The pane.
        visible : bool
            Whether the pane should be visible or not.

        '''
    # Set the pane visiblity
        if pane.isVisible() != visible:
            pane.setVisible(visible)
    # Force showing and raising the pane on top if it is tabified
        if visible and self.tabifiedDockWidgets(pane):
            pane.show()
            pane.raise_()


    def createPopupMenu(self):
        popupmenu = super(MainWindow, self).createPopupMenu()
        popupmenu.setStyleSheet(pref.SS_menu)
        return popupmenu
    

    def launch_dialog(self, dialogname):
        # just a placeholder. Probably just use lambda: <dialog>.show() and 
        # discard this function
        print(f'{dialogname} not implemented.')


    def launch_tool(self, toolname, tabbed=True):
    # Force garbage collection every time a new tool instance gets opened
        gc.collect()
        
        if toolname == 'DatasetBuilder':
            tool = tools.DatasetBuilder()

        elif toolname == 'ModelLearner':
            tool = tools.ModelLearner()

        elif toolname == 'MineralClassifier':
            tool = tools.MineralClassifier()
            # Connect this MineralClassifier signals with appropriate slots
            tool.inmaps_selector.sampleUpdateRequested.connect(
                lambda: tool.inmaps_selector.updateCombox(
                    self.dataManager.getAllGroups()))
            tool.inmaps_selector.mapsUpdateRequested.connect(
                lambda idx: tool.inmaps_selector.updateList(
                    self.dataManager.topLevelItem(idx)))
            
        else:
            tool = None

        if isinstance(tool, tools.DraggableTool):
            if tabbed:
                self.tabWidget.addTab(tool)
            else:
                tool.show()

        else:
            print(f'{toolname} not implemented.')


    def load(self, datatype):
    # Get currently active group. If none, build a new one
        current_item = self.dataManager.currentItem()
        group = self.dataManager.getItemParentGroup(current_item)
        if group is None:
            group = self.dataManager.addGroup(return_group=True)

        if datatype == 'inmaps':
            self.dataManager.loadInputMaps(group)

        elif datatype == 'minmaps':
            self.dataManager.loadMineralMaps(group)

        elif datatype == 'masks':
            self.dataManager.loadMasks(group)

        # elif datatype == 'pntdata':
        #     self.dataManager.loadPointData(group)

        else:
            raise NameError(f'{datatype} is not a valid datatype')


    def update_scene(self, item):
    # Re-call this function to refresh the displayed map if item is (sub)group
        if isinstance(item, (cObj.DataGroup, cObj.DataSubGroup)):
            self.update_scene(self.dataViewer._displayedObject)

    # Actions to be performed when item is a data object
        elif isinstance(item, cObj.DataObject):

        # Extract item data, name and parent (sample). Also get mask if present
            i_data, i_name = item.get('data', 'name')
            sample = self.dataManager.getItemParentGroup(item)
            mask = sample.getCompositeMask('checked')
            if mask is not None:
                mask = mask.mask

        # Actions to be performed if the item holds map data in general
            if item.holdsMap():
                title = f'{sample.text(0)} - {i_name}'
                self.dataViewer._displayedObject = item
                self.dataViewer.currPath.setPath(i_data.filepath)

        # Actions to be performed if item holds input map data
            if item.holdsInputMap():
                inmap = i_data.map if mask is None else i_data.get_masked(mask)

                self.dataViewer.canvas.draw_heatmap(inmap, title)
                self.modeViewer.clear_all()
                self.pmapViewer.canvas.clear_canvas()
                self.updateHistogram()
                if self.histViewer.scaler_action.isChecked():
                    self.histViewer.setScalerExtents()
   

        # Actions to be performed if item holds mineral map data
            elif item.holdsMineralMap():
                mmap, enc, col = i_data.get_plot_data()
                if mask is None:
                    pmap = i_data.probmap
                else:
                    _, mmap, pmap = i_data.get_masked(mask)

                self.dataViewer.canvas.draw_discretemap(mmap, enc, col, title)
                self.modeViewer.update(item, title)
                self.pmapViewer.canvas.draw_heatmap(pmap, title)
                if self.pmapViewer.toggle_range_action.isChecked():
                    self.pmapViewer.setViewRange()
                self.histViewer.hideScaler()
                self.histViewer.canvas.clear_canvas()

        # Actions to be performed if item holds mask data
        # We re-call this function using the currently displayed (map) item so
        # so that any visual modification generated by (un)checking a mask is
        # automatically rendered even when the currently selected item in the
        # Data Manager is not a map. This is triggered only within same sample.
            elif item.holdsMask():
                displ_item = self.dataViewer._displayedObject
                displ_sample = self.dataManager.getItemParentGroup(displ_item)
                if sample == displ_sample:
                    self.update_scene(displ_item)


        # Actions to be performed if item holds point data
            # elif currentItem.holdsPointsData(): pass

    # Exit function if item is not a gruop, a subgroup or a data object
        else:
            return


    def clear_scene(self):
        self.dataViewer._displayedObject = None
        self.dataViewer.currPath.clearPath()
        self.dataViewer.canvas.clear_canvas()
        self.modeViewer.clear_all()
        self.pmapViewer.canvas.clear_canvas()
        self.histViewer.hideScaler()
        self.histViewer.canvas.clear_canvas()
        self.rgbaViewer.clear_all()


    def updateHistogram(self):
    # Exit function if the canvas does not display an input map
        if not self.dataViewer.canvas.contains_heatmap(): return

    # Exit function if data is invalid
        data = self.dataViewer.canvas.image.get_array()
        if data is None: return

    # Get the current image title and ROI coordinates. Initialize the ROI mask
        title = self.dataViewer.canvas.ax.get_title()
        roi_coords, roi_mask = self.roiEditor.current_selection, None

    # Populate the ROI mask using ROI coords if they're valid
        if roi_coords is not None and self.roiEditor.rect_sel.active:
            r0,r1, c0,c1 = roi_coords
            roi_mask = data[r0:r1, c0:c1]
            if not len(roi_mask):
                roi_mask = None

    # Update histogram canvas
        self.histViewer.canvas.update_canvas(data, roi_mask, title)


    def set_rgba_channel(self, channel):
    # Get the current item in the data manager. If it is valid (= a DataObject
    # holding an Input Map), send its data to the RGBA Composite Maps Viewer
        item = self.dataManager.currentItem()
        if isinstance(item, cObj.DataObject) and item.holdsInputMap():
            self.rgbaViewer.set_channel(channel, item.get('data'))

        # Force show the RGBA pane to provide feedback
            self.setPaneVisibility(self.panes[5], True)




    # def run_Preferences(self):
    #     self.PrefDialog = dialogs.Preferences(self)
    #     self.PrefDialog.show()

    # def run_DummyMapsBuilder(self):
    #     self.dummyMapsDialog = dialogs.DummyMapsBuilder(self)
    #     self.dummyMapsDialog.show()

    # def run_Convert2xmap(self):
    #     self.convXmapsDialog = dialogs.Image2Ascii(self)
    #     self.convXmapsDialog.show()

    # def run_Convert2minmap(self):
    #     self.convMinmapsDialog = dialogs.Image2Minmap(self)
    #     self.convMinmapsDialog.show()

    # def run_SubSampleDataset(self):
    #     self.subSampleDialog = dialogs.SubSampleDataset(self)
    #     self.subSampleDialog.show()

    # def run_MergeDatasets(self):
    #     self.mergeDSDialog = dialogs.MergeDatasets(self)
    #     self.mergeDSDialog.show()





    # def run_ModelLearner(self):
    #     self.modelLearnerDialog = dialogs.ModelLearner(self)
    #     self.modelLearnerDialog.show()

    # def run_PhaseRefiner(self):
    #     minMap = self._minmapstab.currentMap
    #     if minMap is None:
    #         QW.QMessageBox.critical(self, 'Missing Mineral Map',
    #                                   'Please load a mineral map in the "Classified Mineral Map" Tab first.')
    #     else:
    #         self.phaseRefinerDialog = dialogs.PhaseRefiner(minMap.copy(), self)
    #         self.phaseRefinerDialog.show()

    def closeEvent(self, event):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Do you really want to exit the app?',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class MainTabWidget(QW.QTabWidget):
    '''
    Central widget of the X-Min Learn window. It is a reimplementation of a
    QTabWidget, customized to accept drag and drop events of its own tabs. Such
    tabs are the major X-Min Learn windows, that can be attached to this widget
    or detached and visualized as separated windows. See the "DraggableTool"
    class in "tools.py" module for more details.

    '''

    def __init__(self, parent):
        '''
        MainTabWidget class constructor.

        Parameters
        ----------
        parent : QWidget
            The GUI parent of this widget.

        '''
        super(MainTabWidget, self).__init__(parent)

    # Set stylesheet
        self.setStyleSheet(pref.SS_mainTabWidget)

    # Set properties
        self.setAcceptDrops(True)
        self.setMovable(True)
        self.setTabsClosable(True)

    # Connect signals to slots
        self._connect_slots()


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Tab 'X' button pressed --> close the tab
        self.tabCloseRequested.connect(self.closeTab)
    # Tab Double-clicked --> detach the tab
        self.tabBarDoubleClicked.connect(self.popOut)


    def addTab(self, widget: tools.DraggableTool | QW.QWidget):
        '''
        Reimplementation of the default addTab function. A GroupScrollArea is
        set as the <widget> container. This helps in the visualization of
        complex widgets across different sized screens.

        Parameters
        ----------
        widget : DraggableTool | QWidget
            The widget to be added as tab.

        '''
        icon, title = widget.windowIcon(), widget.windowTitle()
        widget = cObj.GroupScrollArea(widget, frame=False)
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

        '''
    # The tab is closed only if the widget closeEvent is accepted
        closed = self.widget(index).close()
        if closed:
            scroll_area = self.scrollArea(index)
            self.removeTab(index)
            scroll_area.deleteLater()


    def popIn(self, widget: tools.DraggableTool):
        '''
        Re-insert a tab that was displayed as a separate window into the 
        MainTabWidget.

        Parameters
        ----------
        widget : tools.DraggableTool
            The detached widget.
            
        '''
    # Suppress updates temporarily for better performances
        self.setUpdatesEnabled(False)
        self.addTab(widget)
        self.setUpdatesEnabled(True)


    def popOut(self, index):
        '''
        Detach the tab from the MainTabWidget and display it as a separate
        window.

        Parameters
        ----------
        index : int
            The indec of the tab to be detached.

        '''
        wid = self.widget(index)
        scroll_area = self.scrollArea(index)
    # Only pop out draggable tools
        if isinstance(wid, tools.DraggableTool):
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

        '''
        if isinstance(e.source(), tools.DraggableTool):
            e.accept()

    def dropEvent(self, e):
        '''
        Reimplementation of the default dropEvent function. Customized to
        accept only DraggableTool instances.

        Parameters
        ----------
        e : dropEvent
            The dropEvent triggered by the user's drag & drop action.

        '''
        widget = e.source()
        if isinstance(widget, tools.DraggableTool):
            e.setDropAction(QC.Qt.MoveAction)
            e.accept()
            self.popIn(widget)