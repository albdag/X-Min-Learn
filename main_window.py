# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:31:39 2023

@author: albdag
"""
import gc
import json
import os
import pickle
import zipfile

from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW

import convenient_functions as cf
import custom_widgets as CW
import dialogs
import docks
import preferences as pref
import style
import tools

class MainWindow(QW.QMainWindow):

    def __init__(self):
        '''Constructor.'''
        super(MainWindow, self).__init__()

    # Set main window properties
        self.resize(1600, 900)
        self.setWindowTitle('New project[*]') 
        self.setWindowIcon(QG.QIcon(r'Icons/XML_logo.png'))
        self.setDockOptions(self.AllowTabbedDocks)
        self.setAnimated(pref.get_setting('GUI/smooth_animation'))
        self.statusBar()
        self.setStyleSheet(style.SS_MAINWINDOW)

    # Set main window attributes
        self.project_path = None
        self.open_tools = []

    # Initialize GUI
        self._init_ui()
    
    # Connect signals with slots
        self._connect_slots()

    # Restore last window state (toolboxes and panes)
        self.restoreState(pref.get_setting('GUI/window_state'), version=0)


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
        '''
        Initialize window panes (QDockWidget).

        '''
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
        self.dataViewer.canvas.share_axis(self.pmapViewer.canvas.ax)

    # RGBA Composite Maps Viewer
        self.rgbaViewer = docks.RgbaCompositeMapViewer()
        # Share rgba and data viewer axis
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
        
    # Store the panes toggle view actions, set their icons and their objectName
        self.panes_tva = []
        for p in self.panes:
            action = p.toggleViewAction()
            p.setObjectName(p.title) # required for saving pane state
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
        '''
        Initialize actions shared by menu and toolbars.
        
        '''
    # New project action
        self.new_project_action = QW.QAction(
            QG.QIcon(r'Icons/empty.png'), '&New project')
        self.new_project_action.setShortcut('Ctrl+N')

    # Open project action
        self.open_project_action = QW.QAction(
            QG.QIcon(r'Icons/open.png'), '&Open project')
        self.open_project_action.setShortcut('Ctrl+O')

    # Save project action
        self.save_project_action = QW.QAction(
            QG.QIcon(r'Icons/save.png'), '&Save project')
        self.save_project_action.setShortcut('Ctrl+S')

    # Save project as... action
        self.save_project_as_action = QW.QAction(
            QG.QIcon(r'Icons/save_as.png'), '&Save project as...')
        self.save_project_as_action.setShortcut('Ctrl+Shift+S')

    # Quit app action
        self.close_action = QW.QAction('&Exit')

    # Import Input Maps Action 
        self.load_inmaps_action = QW.QAction(
            QG.QIcon(r'Icons/inmap.png'), '&Input Maps')
        self.load_inmaps_action.setShortcut('Ctrl+I')

    # Import Mineral Maps Action
        self.load_minmaps_action = QW.QAction(
            QG.QIcon(r'Icons/minmap.png'), '&Mineral Maps')
        self.load_minmaps_action.setShortcut('Ctrl+M')

    # Import Masks Action
        self.load_masks_action = QW.QAction(
            QG.QIcon(r'Icons/mask.png'), 'Mas&ks')
        self.load_masks_action.setShortcut('Ctrl+K')

    # Launch Preferences Action
        self.pref_action = QW.QAction(
            QG.QIcon(r'Icons/wrench.png'), 'Preferences')

    # Launch Convert Image to Input Map Action
        self.conv2inmap_action = QW.QAction('Image to Input Map')
        self.conv2inmap_action.setStatusTip('Convert image to Input Map')

    # Launch Convert Image to Mineral Maps Action
        self.conv2minmap_action = QW.QAction('Image to Mineral Map')
        self.conv2minmap_action.setStatusTip('Convert image to Mineral Map')

    # Launch Generate Dummy Maps Action
        self.dummymap_action = QW.QAction('Generate &Dummy Maps')
        self.dummymap_action.setStatusTip('Build placeholder noisy maps')

    # Launch Sub-sample Dataset Action
        self.subsample_ds_action = QW.QAction('&Sub-sample Dataset')
        self.subsample_ds_action.setStatusTip('Extract sub-sets from dataset')

    # Launch Merge Datasets Action
        self.merge_ds_action = QW.QAction('&Merge Datasets')
        self.merge_ds_action.setStatusTip('Merge multiple datasets into one')

    # Launch Dataset Builder Action
        self.ds_builder_action = QW.QAction(
            QG.QIcon(r'Icons/compile_dataset.png'), 'Dataset &Builder')
        self.ds_builder_action.setShortcut('Ctrl+Alt+B')
        self.ds_builder_action.setStatusTip('Compile ground truth datasets')

    # Launch Model Learner Action
        self.model_learner_action = QW.QAction(
            QG.QIcon(r'Icons/merge.png'), 'Model &Learner')
        self.model_learner_action.setShortcut('Ctrl+Alt+L')
        self.model_learner_action.setStatusTip('Build machine learning models')

    # Launch Mineral Classifier Action
        self.classifier_action = QW.QAction(
            QG.QIcon(r'Icons/classify.png'), 'Mineral &Classifier')
        self.classifier_action.setShortcut('Ctrl+Alt+C')
        self.classifier_action.setStatusTip('Predict mineral maps')

    # Launch Phase Refiner Action
        self.refiner_action = QW.QAction(
            QG.QIcon(r'Icons/refine.png'), 'Phase &Refiner')
        self.refiner_action.setShortcut('Ctrl+Alt+R')
        self.refiner_action.setStatusTip('Refine mineral maps')

    # About X-Min Learn Action
        self.about_action = QW.QAction('About X-Min Learn')
        self.about_action.setStatusTip('About X-Min Learn app')

    # About Qt Action
        self.aboutqt_action = QW.QAction('About Qt')
        self.aboutqt_action.setStatusTip('About Qt toolkit')


    def _init_toolbars(self):
        '''
        Initialize main toolbars.

        '''
    # Tools toolbar
        self.tools_toolbar = QW.QToolBar('Tools toolbar')
        self.tools_toolbar.setObjectName('ToolsToolbar') # to save its state
        self.tools_toolbar.setFloatable(False)
        self.tools_toolbar.setStyleSheet(style.SS_MAINTOOLBAR)
        self.tools_toolbar.addActions(
            (self.ds_builder_action, self.model_learner_action,
             self.classifier_action, self.refiner_action))
        # separator followed by preference dialog
        # other future tools (e.g. map algebra calculator)
    
    # Panes toolbar
        self.panes_toolbar = QW.QToolBar('Panes toolbar')
        self.panes_toolbar.setObjectName('PanesToolbar') # to save its state
        self.panes_toolbar.setFloatable(False)
        self.panes_toolbar.setStyleSheet(style.SS_MAINTOOLBAR)
        self.panes_toolbar.addActions(self.panes_tva)

    # Set default toolbars position in the window
        self.addToolBar(QC.Qt.LeftToolBarArea, self.panes_toolbar)
        self.addToolBar(QC.Qt.LeftToolBarArea, self.tools_toolbar)


    def _init_menu(self):
        '''
        Initialize main menu.
        
        '''
    # Set custom stylesheet to menu bar
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(style.SS_MENUBAR + style.SS_MENU)

    # File Menu
        file_menu = menu_bar.addMenu('&File')
        file_menu.addActions(
            (self.new_project_action, self.open_project_action, 
             self.save_project_action, self.save_project_as_action))
        file_menu.addSeparator()

        import_submenu = file_menu.addMenu(
            QG.QIcon(r'Icons/import.png'), '&Import...')
        import_submenu.addActions(
            (self.load_inmaps_action, self.load_minmaps_action,
              self.load_masks_action))
        
        file_menu.addActions((self.pref_action, self.close_action))

    # Dataset Menu
        dataset_menu = menu_bar.addMenu('&Dataset')
        dataset_menu.addActions(
            (self.ds_builder_action, self.subsample_ds_action, 
             self.merge_ds_action))
        
    # Classification Menu
        class_menu = menu_bar.addMenu('&Classification')
        class_menu.addActions(
            (self.classifier_action, self.model_learner_action))

    # Post-classification Menu
        postclass_menu = menu_bar.addMenu('&Post-classification')
        postclass_menu.addAction(self.refiner_action)

    # Utility Menu
        utility_menu = menu_bar.addMenu('&Utility')
        convert_submenu = utility_menu.addMenu('&Convert')
        convert_submenu.addActions(
            (self.conv2inmap_action, self.conv2minmap_action))
        utility_menu.addAction(self.dummymap_action)

    # View Menu
        view_menu = menu_bar.addMenu('&View')
        panes_submenu = view_menu.addMenu('&Panes')
        panes_submenu.addActions(self.panes_tva)
        view_menu.addSeparator()
        view_menu.addActions((self.panes_toolbar.toggleViewAction(),
                              self.tools_toolbar.toggleViewAction()))
        
    # Info menu
        info_menu = menu_bar.addMenu('&?')
        info_menu.addActions((self.about_action, self.aboutqt_action))
        # Separator
        # Help (user-guide?)
        # Separator
        # Check updates (link to github page?)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Detach tool from main tabWidget when double clicked on tab
        self.tabWidget.tabBarDoubleClicked.connect(self.popOutTool)

    # Connect all panes toggle view actions with a custom slot to force showing
    # them when they are tabified
        for p, a, in zip(self.panes, self.panes_tva):
            a.triggered.connect(lambda t, p=p: self.setPaneVisibility(p, t))

    # Data Manager pane actions 
        self.dataManager.updateSceneRequested.connect(self.updateScene)
        self.dataManager.clearSceneRequested.connect(self.clearScene)
        self.dataManager.rgbaChannelSet.connect(self.setRgbaChannel)
        self.dataManager.model().dataChanged.connect(
            lambda: self.setWindowModified(True))
        self.dataManager.model().rowsRemoved.connect(
            lambda: self.setWindowModified(True))
        
    # Histogram Viewer pane actions
        self.histViewer.scalerRangeChanged.connect(
            lambda: self.setWindowModified(True))

    # Mode Viewer pane actions
        self.modeViewer.updateSceneRequested.connect(self.updateScene)

    # ROI Editor pane actions
        self.roiEditor.rectangleSelectorUpdated.connect(self.updateHistogram)
        self.roiEditor.rectangleSelectorUpdated.connect(
            lambda: self.setWindowModified(True))

        self.roiEditor.autoroi_dial.maps_selector.sampleUpdateRequested.connect(
            lambda: self.roiEditor.autoroi_dial.maps_selector.updateCombox(
                self.dataManager.getAllGroups()))
        
        self.roiEditor.autoroi_dial.maps_selector.mapsUpdateRequested.connect(
            lambda idx: self.roiEditor.autoroi_dial.maps_selector.updateList(
                self.dataManager.topLevelItem(idx)))
        
        self.roiEditor.table.model().dataChanged.connect(
            lambda: self.setWindowModified(True))
        self.roiEditor.table.model().rowsRemoved.connect(
            lambda: self.setWindowModified(True))
        
    # Probability Map Viewer pane actions
        self.pmapViewer.probabilityRangeChanged.connect(
            lambda: self.setWindowModified(True))
        
    # RGBA Viewer pane actions
        self.rgbaViewer.rgbaModified.connect(
            lambda: self.setWindowModified(True))
        
    # Project actions
        self.new_project_action.triggered.connect(self.newProject)
        self.open_project_action.triggered.connect(self.openProject)
        self.save_project_action.triggered.connect(
            lambda: self.saveProject(overwrite=True))
        self.save_project_as_action.triggered.connect(
            lambda: self.saveProject(overwrite=False))
        
    # Quit app 
        self.close_action.triggered.connect(self.close)

    # Import data  
        self.load_inmaps_action.triggered.connect(self.importData)
        self.load_minmaps_action.triggered.connect(self.importData)
        self.load_masks_action.triggered.connect(self.importData)

    # Launch Preferences 
        self.pref_action.triggered.connect(
            lambda: dialogs.Preferences(self).show())

    # Launch Convert Images to Input Maps 
        self.conv2inmap_action.triggered.connect(
            lambda: dialogs.ImageToInputMap(self).show())

    # Launch Convert Image to Mineral Maps 
        self.conv2minmap_action.triggered.connect(
            lambda: dialogs.ImageToMineralMap(self).show())

    # Launch Generate Dummy Maps 
        self.dummymap_action.triggered.connect(
            lambda: dialogs.DummyMapsBuilder(self).show())

    # Launch Sub-sample Dataset 
        self.subsample_ds_action.triggered.connect(
            lambda: dialogs.SubSampleDataset(self).show())

    # Launch Merge Datasets 
        self.merge_ds_action.triggered.connect(
            lambda: dialogs.MergeDatasets(self).show())
        
    # Launch about X-Min Learn dialog
        self.about_action.triggered.connect(self.about)
        
    # Launch about Qt dialog
        self.aboutqt_action.triggered.connect(QW.qApp.aboutQt)

    # Launch Dataset Builder 
        self.ds_builder_action.triggered.connect(
            lambda: self.openTool('DatasetBuilder'))

    # Launch Model Learner 
        self.model_learner_action.triggered.connect(
            lambda: self.openTool('ModelLearner'))

    # Launch Mineral Classifier 
        self.classifier_action.triggered.connect(
            lambda: self.openTool('MineralClassifier'))

    # Launch Phase Refiner 
        self.refiner_action.triggered.connect(
            lambda: self.openTool('PhaseRefiner'))


    def about(self):
        '''
        Show the about X-Min Learn dialog.

        '''
        html = f'''
        <p><span style="font-size: 12pt; font-weight: bold;">About X-Min Learn</span></p>
        <p>Currently used version: <span style="font-weight: bold;">{QW.qApp.applicationVersion()}</span></p>
        <p>Author: Dr. Alberto D'Agostino (Ph.D.) - University of Catania<br>
        Contacts: <a href='mailto:dagostino.alberto@hotmail.it'>dagostino.alberto@hotmail.it</a> | <a href='mailto:alberto.dagostino@unict.it'>alberto.dagostino@unict.it</a></p>
        <p><br>X-Min Learn is an open-source project (<a href='https://www.gnu.org/licenses/gpl-3.0.html'>GPLv3</a>).
        Most of the app icons are provided for free by <a href='https://icons8.it/'>Icons8</a>.
        For more info check the <a href='https://github.com/albdag/X-Min-Learn'>project page</a>.</p>
        '''
        QW.QMessageBox.about(self, 'About X-Min Learn', html)


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
        popupmenu.setStyleSheet(style.SS_MENU)
        return popupmenu
    

    def launch_dialog(self, dialogname):
        # just a placeholder. Probably just use lambda: <dialog>.show() and 
        # discard this function
        print(f'{dialogname} not implemented.')


    def openTool(self, toolname, tabbed=True):
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
            
        elif toolname == 'PhaseRefiner':
            tool = tools.PhaseRefiner()
            # Connect this PhaseRefiner signals with appropriate slots
            tool.minmap_selector.sampleUpdateRequested.connect(
                lambda: tool.minmap_selector.updateCombox(
                    self.dataManager.getAllGroups()))
            tool.minmap_selector.mapsUpdateRequested.connect(
                lambda idx: tool.minmap_selector.updateList(
                    self.dataManager.topLevelItem(idx)))
            
        else:
            print(f'{toolname} not implemented.')
            return

    # Show tool as a new tab in the tab widget or as a floating window
        if isinstance(tool, tools.DraggableTool):
            tool.closed.connect(lambda: self.closeTool(tool))
            self.open_tools.append(tool)

            if tabbed:
                self.tabWidget.addTab(tool)
            else:
                tool.show()


    def popOutTool(self, index: int):
        '''
        Detach the tab at index 'index' from the main TabWidget and display it
        as a separate floating window. The tab must contain a DraggableTool for
        it to be detached.

        Parameters
        ----------
        index : int
            The index of the tab to be detached.

        '''
        tool = self.tabWidget.widget(index)
        point = self.tabWidget.tabBar().tabRect(index).center()

        if isinstance(tool, tools.DraggableTool):
        # Set tool as floating (parent=None) and remove it from tab widget
            tool.setParent(None)
            self.tabWidget.closeTab(index)
        # Move tool close to where it was originally tabbed
            parent_point = self.tabWidget.tabBar().mapToParent(point)
            QC.QTimer.singleShot(0, lambda: tool.move(parent_point))
            tool.setVisible(True)
            

    def closeTool(self, tool: tools.DraggableTool):
    # Safety: do nothing if tool is not a DraggableTool
        if not isinstance(tool, tools.DraggableTool): 
            return
        
        if not tool.isFloating():
            tab_index = self.tabWidget.indexOf(tool)
            self.tabWidget.closeTab(tab_index)
        
        tool.disconnect()
        tool.killReferences()
        self.open_tools.remove(tool)


    def importData(self):
        '''
        Wrapper function to load data in the Data Manager. The type of data to
        be loaded is automatically retrieved from the sender action.

        '''
    # Get currently active group. If none, build a new one
        current_item = self.dataManager.currentItem()
        group = self.dataManager.getItemParentGroup(current_item)
        if group is None:
            group = self.dataManager.addGroup()

    # Import data in the proper subgroup depending on loading data type
        sender_action = self.sender()

        if sender_action == self.load_inmaps_action:
            self.dataManager.loadData(group.inmaps)

        elif sender_action == self.load_minmaps_action:
            self.dataManager.loadData(group.minmaps)

        elif sender_action == self.load_masks_action:
            self.dataManager.loadData(group.masks)

        else: # safety
            return


    def updateScene(self, item: CW.DataObject|CW.DataSubGroup|CW.DataGroup):
        '''
        Main control function for rendering data in the Data Viewer and in the 
        panes. 

        Parameters
        ----------
        item : DataObject or DataSubGroup or DataGroup
            Item to be rendered. Canonically, it should be a DataObject; if a
            DataSubGroup or a DataGroup is passed, this function attempts to
            re-render the currently displayed data object.

        '''
    # Re-call this function to refresh the displayed map if item is (sub)group
        if isinstance(item, (CW.DataGroup, CW.DataSubGroup)):
            self.updateScene(self.dataViewer._displayedObject)

    # Actions to be performed when item is a data object
        elif isinstance(item, CW.DataObject):

        # Extract item data, filepath, name and parent group (sample)
            i_data, i_path, i_name = item.get('data', 'filepath', 'name')
            sample = self.dataManager.getItemParentGroup(item)
        # Exit function and clear the scene if the sample is None
            if sample is None:
                return self.clearScene()
        # Check for source file existence
            if not item.filepathValid():
                item.setNotFound(True)
        # Also get mask if present
            mask = sample.getCompositeMask('checked')
            if mask is not None:
                mask = mask.mask

        # Actions to be performed if the item holds map data in general
            if item.holdsMap():
                title = f'{sample.text(0)} - {i_name}'
                self.dataViewer._displayedObject = item
                self.dataViewer.currPath.setPath(i_path)

        # Actions to be performed if item holds input map data
            if item.holdsInputMap():
                inmap = i_data.map if mask is None else i_data.get_masked(mask)

                self.dataViewer.canvas.draw_heatmap(inmap, title)
                self.modeViewer.clearAll()
                self.pmapViewer.canvas.clear_canvas()
                self.updateHistogram()
                if self.histViewer.scaler_action.isChecked():
                    self.histViewer.updateScalerExtents()
   

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
                    self.pmapViewer.updateViewRange()
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
                    self.updateScene(displ_item)


        # Actions to be performed if item holds point data
            # elif currentItem.holdsPointsData(): pass

    # Exit function if item is not a group, a subgroup or a data object
        else:
            return


    def clearScene(self):
        '''
        Clear the current view of the Data Viewer and the panes.

        '''
        self.dataViewer._displayedObject = None
        self.dataViewer.currPath.clearPath()
        self.dataViewer.canvas.clear_canvas()
        self.modeViewer.clearAll()
        self.pmapViewer.canvas.clear_canvas()
        self.histViewer.hideScaler()
        self.histViewer.canvas.clear_canvas()
        self.rgbaViewer.clearAll()


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


    def setRgbaChannel(self, channel: str):
        '''
        Set current item in the Data Manager as channel 'channel' in the RGBA
        Map Viewer pane.

        Parameters
        ----------
        channel : str
            RGBA channel. Must be one of 'R', 'G', 'B', 'A'.

        '''
    # Get the current item in the data manager. If it is valid (= a DataObject
    # holding an Input Map), send its data to the RGBA Composite Maps Viewer
        item = self.dataManager.currentItem()
        if isinstance(item, CW.DataObject) and item.holdsInputMap():
            self.rgbaViewer.setChannel(channel, item.get('data'))

        # Force show the RGBA pane to provide feedback
            self.setPaneVisibility(self.panes[5], True)


    def saveProject(self, overwrite=True) -> bool:
        '''
        Save current project to disk.

        Parameters
        ----------
        overwrite : bool, optional
           Whether the current project path should be overwritten. If False,
           user is prompted to select a new file destination (save as). The
           default is True.

        Returns
        -------
        bool
            Whether the project was saved or not.

        '''
    # Do not save the project if the filepath is invalid
        path = self.project_path
        if not overwrite or path is None:   
            path, _ = QW.QFileDialog.getSaveFileName(self, 'Save project',
                                                     pref.get_dir('out'),
                                                     'X-Min Learn project (*.xmj)')
            if not path:
                return False

    # Send warning and ask for user's choice if Data Manager has unsaved data
        if self.dataManager.hasUnsavedData():
            text = 'Unsaved edits in the Data Manager will not be saved '\
                   'automatically. Proceed anyway?'
            choice = CW.MsgBox(self, 'QuestWarn', text)  
            if choice.no():
                return False      

    # Collect project data
        project_info = {
            'Name': cf.path2filename(path),
            'Version': QW.qApp.applicationVersion()
        }
        
        current_view = {
            'Path': self.dataViewer.currPath.fullpath,
            'Zoom': [
                self.dataViewer.canvas.ax.get_xlim(),
                self.dataViewer.canvas.ax.get_ylim()
            ]
        }
        
        panes_data = {
            p.objectName(): p.trueWidget().getConfig() for p in self.panes
        }
        
        tools_data = {} # TODO

    # Save project data to file
        try:
            with zipfile.ZipFile(path, 'w') as xmj:
                xmj.writestr('ProjectInfo', json.dumps(project_info, indent=4))
                xmj.writestr('CurrentView', json.dumps(current_view, indent=4))
                xmj.writestr('PanesData', json.dumps(panes_data, indent=4))
                xmj.writestr('ToolsData', pickle.dumps(tools_data))
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save project.', str(e))
            return False

    # Set the app default output directory to 'path' only after the project has 
    # been successfully saved
        pref.set_dir('out', os.path.dirname(path))

    # Update window attributes 
        self.project_path = path
        self.setWindowTitle(f'{project_info['Name']}[*]')
        self.setWindowModified(False)
        return True


    def projectSafeToClose(self) -> bool:
        '''
        Check if the current project can be closed without data loss.

        Returns
        -------
        bool
            Whether the project is safe to close.

        '''
    # Project is safe to close if no tool is open and window is not modified
        if not self.isWindowModified() and not len(self.open_tools):
            return True

    # Otherwise, ask for user's choice to save the current project
        btns = QW.QMessageBox.Yes | QW.QMessageBox.No | QW.QMessageBox.Cancel
        choice = CW.MsgBox(self, 'QuestWarn', 'Save current project?', 
                           btns=btns, def_btn=QW.QMessageBox.Yes)
    
    # User wants to save -> launch 'saveProject' function and return its output
        if choice.yes(): 
            return self.saveProject()
    
    # User does not want to save -> the project is safe to close
        elif choice.no():
            return True
    
    # User canceled the choice dialog -> the project is not safe to close
        else:
            return False


    def newProject(self):
        '''
        Initialize a new project.

        '''
    # Deny creating a new project if the current project is not safe to close
        if not self.projectSafeToClose():
            return
        
    # Clear scene
        self.clearScene()

    # Reset panes GUI state
        for pane in self.panes:
            pane.trueWidget().resetConfig()
        
    # Kill floating and tabbed tools
        for tool in reversed(self.open_tools):
            self.closeTool(tool)
   
    # Update window attributes and properties
        self.project_path = None
        self.setWindowTitle('New project[*]')
        self.setWindowModified(False)       


    def openProject(self):
        '''
        Load and open a project from disk.

        '''
    # Deny opening a project if the current project is not safe to close
        if not self.projectSafeToClose():
            return
            
    # Do nothing if project path is invalid (file dialog is canceled)
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Open project',
                                                 pref.get_dir('in'),
                                                 'X-Min Learn project (*.xmj)')
        if not path:
            return
        
    # Retrive data from project file
        try: 
            with zipfile.ZipFile(path, 'r') as xmj:
                project_info = json.loads(xmj.read('ProjectInfo'))
                current_view = json.loads(xmj.read('CurrentView'))
                panes_data = json.loads(xmj.read('PanesData'))
                tools_data = pickle.loads(xmj.read('ToolsData'))
        except Exception as e:
            return CW.MsgBox(self, 'Crit', 'Failed to open project.', str(e))
        
    # Ask for user's choice if the project and the app version do not coincide
        if (vers := project_info['Version']) != QW.qApp.applicationVersion():
            choice = CW.MsgBox(self, 'QuestWarn', 'This project was saved in '\
                               f'a different version: {vers}. Proceed anyway?')
            if choice.no():
                return

    # Initialize a list to store any loading error
        loading_errors = []

    # Clear scene
        self.clearScene()

    # Load current view limits (zoom)
        xlim, ylim = current_view['Zoom']
        self.dataViewer.canvas.ax.set(xlim=xlim, ylim=ylim)

    # Load panes states
        for pane in self.panes:
            pane_name = pane.objectName()
            try:
                pane.trueWidget().loadConfig(panes_data[pane_name])
            except Exception as e:
                loading_errors.append((pane_name, e))

    # Load current view object
        for item in self.dataManager.getAllDataObjects():
            if item.get('filepath') == current_view['Path']:
                self.dataManager.setCurrentItem(item)
                self.updateScene(item)
                break

    # Close currently opened tools and load project's tools states 
    # TODO...

    # Set the app defualt input directory to 'path' after project is loaded to 
    # avoid it being replaced by the paths of data loaded within the individual
    # panes and tools 
        pref.set_dir('in', os.path.dirname(path))

    # Update window attributes and properties
        self.project_path = path
        self.setWindowTitle(f'{project_info['Name']}[*]')
        self.setWindowModified(False)   

    # Send any catched loading error via warning
        if len(loading_errors):
            CW.MsgBox(self, 'Warn', 'Found corrupted data in the project.',
                      '\n\n'.join((f'{k} -> {e}' for k, e in loading_errors)))


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


    def closeEvent(self, event: QG.QCloseEvent):
        '''
        Reimplementation of the closeEvent. It shows a dialog to save the
        project if it has been modified. The current state of the window is
        automatically saved on exit.

        Parameters
        ----------
        event : QG.QCloseEvent
            The close event.

        '''
        if self.projectSafeToClose():
        # Avoid to manually have to close floating tools if main app is closed
            for tool in reversed(self.open_tools):
                self.closeTool(tool)

        # Save the current state of main window's panes and toolbars
            pref.edit_setting('GUI/window_state', self.saveState(version=0))
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
        self.setStyleSheet(style.SS_MAINTABWIDGET)

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
    # Tab 'X' button pressed --> triggers the closeEvent of the widget
        self.tabCloseRequested.connect(lambda idx: self.widget(idx).close())


    def addTab(self, widget: tools.DraggableTool | QW.QWidget):
        '''
        Reimplementation of the default addTab function. A GroupScrollArea is
        set as the 'widget' container. This helps in the visualization of
        complex widgets across different sized screens.

        Parameters
        ----------
        widget : DraggableTool or QWidget
            The widget to be added as tab.

        '''
    # Insert widget in a scroll area, add it and set it as the current tab
        icon, title = widget.windowIcon(), widget.windowTitle()
        scroll = CW.GroupScrollArea(widget, frame=False)
        super(MainTabWidget, self).addTab(scroll, icon, title)
        self.setCurrentWidget(scroll)

    # Send 'tabified' signal if a draggable tool is added
        if isinstance(widget, tools.DraggableTool):
            widget.tabified.emit()


    def widget(self, index: int):
        '''
        Reimplementation of the default 'widget' function. Returns the widget
        held by the tab at 'index' by passing the GroupScrollArea widget that
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


    def scrollArea(self, index: int):
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
          

    def closeTab(self, index: int):
        '''
        Close the tab at index 'index'.

        Parameters
        ----------
        index : int
            The tab index in the tab bar.

        '''
    # 'removeTab' just removes the tab from the TabBar but not the widget page,
    # so we manually do it by deleting the scroll area that wraps the tool
        scroll_area = self.scrollArea(index)
        self.removeTab(index) 
        scroll_area.deleteLater()


    def indexOf(self, widget: tools.DraggableTool | QW.QWidget) -> int:
        '''
        Reimplementation of the default 'indexOf' function. Returns the index
        of the tab that contains 'widget'. If the widget is not found, the
        index will be -1. Please note that this function expects 'widget' to be
        the actual displayed widget and not the scroll area that wraps it (see
        also indexOfScrollArea function). 

        Parameters
        ----------
        widget : tools.DraggableTool | QW.QWidget
            Widget. 

        Returns
        -------
        idx : int
            Index of the widget.

        '''
        widgets = self.getAllWidgets()
        idx = -1 if widget not in widgets else widgets.index(widget)
        return idx
    

    def indexOfScrollArea(self, scroll_area: CW.GroupScrollArea) -> int:
        '''
        Return the index of the tab that contains 'scroll_area'. If the scroll
        area is not fiund, the index will be -1. This function should not be 
        used to get the index of the actual displayed widget, but rather of the
        scroll area that wraps it (see also indexOf function).  

        Parameters
        ----------
        scroll_area : CW.GroupScrollArea
            Scroll area.

        Returns
        -------
        int
            Index of the scroll area.
        '''
        super(MainTabWidget, self).indexOf(scroll_area)
    

    def getAllWidgets(self) -> list[tools.DraggableTool, QW.QWidget]:
        '''
        Return a list of all the widgets (not the scroll areas that wrap them)
        included in the tab widget.


        Returns
        -------
        list[tools.DraggableTool, QW.QWidget]
            List of widgets.

        '''
        return [self.widget(idx) for idx in range(self.count())]


    def dragEnterEvent(self, event: QG.QDragEnterEvent):
        '''
        Reimplementation of the default dragEnterEvent function. Customized to
        accept only DraggableTool instances.

        Parameters
        ----------
        event : QDragEntervent
            The drag enter event triggered by the user's drag action.

        '''
        if isinstance(event.source(), tools.DraggableTool):
            event.accept()


    def dropEvent(self, event: QG.QDropEvent):
        '''
        Reimplementation of the default dropEvent function. Customized to
        accept only DraggableTool instances and add them as new tabs. 

        Parameters
        ----------
        event : QDropEvent
            The drop event triggered by the user's drag & drop action.

        '''
        widget = event.source()
        if isinstance(widget, tools.DraggableTool):
            event.setDropAction(QC.Qt.MoveAction)
            event.accept()
        # Suppress updates temporarily for better performances
            self.setUpdatesEnabled(False)
            self.addTab(widget)
            self.setUpdatesEnabled(True)