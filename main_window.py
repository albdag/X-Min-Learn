# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:31:39 2023

@author: albdag
"""

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
        CF.shareAxis(self.pmapViewer.canvas.ax, self.dataViewer.canvas.ax)

    # RGBA Composite Maps Viewer
        self.rgbaViewer = docks.RgbaCompositeMapViewer()
        # Share rgba and data viewer axis
        CF.shareAxis(self.rgbaViewer.canvas.ax, self.dataViewer.canvas.ax)

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
        p_width = max([p.widget.minimumSizeHint().width() for p in self.panes])
        self.resizeDocks(self.panes, [p_width] * len(self.panes), 
                         QC.Qt.Horizontal)
        
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
        self.tools_toolbar.setIconSize(QC.QSize(32, 32))
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
        self.panes_toolbar.setIconSize(QC.QSize(32, 32))
        self.panes_toolbar.setStyleSheet(pref.SS_mainToolbar)
        self.panes_toolbar.addActions(self.panes_tva)

    # Set default toolbars position in the window
        self.addToolBar(QC.Qt.LeftToolBarArea, self.panes_toolbar)
        self.addToolBar(QC.Qt.RightToolBarArea, self.tools_toolbar)


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
    # Data Manager pane actions 
        self.dataManager.updateSceneRequested.connect(self.update_scene)
        self.dataManager.clearSceneRequested.connect(self.clear_scene)
        self.dataManager.rgbaChannelSet.connect(self.set_rgba_channel)

    # Mode Viewer pane actions
        self.modeViewer.updateSceneRequested.connect(self.update_scene)

    # ROI Editor pane actions
        self.roiEditor.rectangleSelectorUpdated.connect(self.updateHistogram)

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
            lambda: self.launch('Preferences', tabbed=False))

    # Launch Convert Grayscale images to ASCII 
        self.conv2ascii_action.triggered.connect(
            lambda: self.launch('Grayscale2Ascii'))

    # Launch Convert RGB yo Mineral Maps 
        self.conv2mmap_action.triggered.connect(
            lambda: self.launch('Rgb2Minmap'))

    # Launch Build Dummy Maps 
        self.dummy_map_action.triggered.connect(
            lambda: self.launch('DummyMaps'))

    # Launch Sub-sample Dataset 
        self.subsample_ds_action.triggered.connect(
            lambda: self.launch('SubSampleDataset'))

    # Launch Merge Datasets 
        self.merge_ds_action.triggered.connect(
            lambda: self.launch('MergeDatasets'))

    # Launch Dataset Builder 
        self.ds_builder_action.triggered.connect(
            lambda: self.launch('DatasetBuilder'))

    # Launch Model Learner 
        self.model_learner_action.triggered.connect(
            lambda: self.launch('ModelLearner'))

    # Launch Mineral Classifier 
        self.classifier_action.triggered.connect(
            lambda: self.launchMineralClassifier())

    # Launch Phase Refiner 
        self.refiner_action.triggered.connect(
            lambda: self.launch('PhaseRefiner'))


    def addPane(self, dockWidgetArea, pane, visible=True):
        self.addDockWidget(dockWidgetArea, pane)
        pane.setVisible(visible)


    def createPopupMenu(self):
        popupmenu = super(MainWindow, self).createPopupMenu()
        popupmenu.setStyleSheet(pref.SS_menu)
        return popupmenu
    
    def launchMineralClassifier(self, tabbed=True):
        mc = tools.MineralClassifier()
    
    # Connect this MineralClassifier instance's signals with appropriate slots
        mc.sample_combox.clicked.connect(
            lambda: mc.updateSamples(self.dataManager.getAllGroups()))
        mc.sample_combox.activated.connect(
            lambda idx: mc.updateInmapsList(
                self.dataManager.topLevelItem(idx).inmaps))
    
    # Show the Mineral Classifier as a tab or as a stand-alone window
        if tabbed:
            self.tabWidget.addTab(mc)
        else:
            mc.show()

    def launch(self, toolname, tabbed=True):
        tools_dict = {'DatasetBuilder': tools.DatasetBuilder}
        tool = tools_dict.get(toolname, None)

        if tool is not None:
            if tabbed:
                self.tabWidget.addTab(tool(self))
            else:
                tool(self).show()

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
                mmap, enc, col = i_data.get_plotData()
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

        # If the RGBA Viewer is hidden, let's show it to provide feedback
             if not self.rgbaViewer.isVisible():
                 self.panes[5].setVisible(True)




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
        self.setStyleSheet(pref.SS_tabWidget)

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


    def addTab(self, widget):
        '''
        Reimplementation of the default addTab function. A GroupScrollArea is
        set as the <widget> container. This helps in the visualization of
        complex widgets across different sized screens.

        Parameters
        ----------
        widget : QWidget
            The widget to be added as tab.

        '''
        icon, title = widget.windowIcon(), widget.windowTitle()
        widget = cObj.GroupScrollArea(widget)
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
        if isinstance(e.source(), tools.DraggableTool):
            e.setDropAction(QC.Qt.MoveAction)
            e.accept()

        # Suppress updates temporarily for better performances
            wid = e.source()
            self.setUpdatesEnabled(False)
            self.addTab(wid)
            self.setUpdatesEnabled(True)




# # 1ST TAB --> INPUT MAPS VISUALIZER
# class InputMapsTab(cObj.DraggableTool):

#     def __init__(self, parent):
#         self.parent = parent
#         super(InputMapsTab, self).__init__()

#         self.setWindowTitle('Input Maps')
#         self.ROIcoords = None

#         self.init_ui()

#     def init_ui(self):

#     # T.O.C.
#         self.TOC = QW.QListWidget()
#         self.TOC.setHorizontalScrollBar(cObj.StyledScrollBar(Qt.Horizontal))
#         self.TOC.setVerticalScrollBar(cObj.StyledScrollBar(Qt.Vertical))
#         self.TOC.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
#         self.TOC.keyPressEvent = lambda evt: evt.ignore() # ignore key press events to avoid bugs with TOC.currentRow()
#         self.TOC.itemPressed.connect(self.showMap)  # item pressed includes right-click

#     # T.O.C. Option Buttons
#         self.loadMaps_btn = cObj.IconButton('Icons/generic_add.png')
#         self.loadMaps_btn.setToolTip('Load input maps')
#         self.loadMaps_btn.clicked.connect(lambda: self.loadMaps())

#         self.refresh_btn = cObj.IconButton('Icons/refresh.png')
#         self.refresh_btn.setToolTip('Refresh data source')
#         self.refresh_btn.clicked.connect(self.refresh_data)

#         self.delMaps_btn = cObj.IconButton('Icons/generic_del.png')
#         self.delMaps_btn.setToolTip('Remove maps')
#         self.delMaps_btn.clicked.connect(self.delMaps)

#         self.invertMaps_btn = cObj.IconButton('Icons/invert.png')
#         self.invertMaps_btn.setToolTip('Invert pixel values')
#         self.invertMaps_btn.clicked.connect(self.invertMaps)

#         self.scaleCmap_btn = cObj.IconButton('Icons/lockCbar.png')
#         self.scaleCmap_btn.setToolTip('Apply the same colormap range to all maps')
#         self.scaleCmap_btn.setCheckable(True)
#         self.scaleCmap_btn.toggled.connect(self.scaleColormap)

#         self.RGBA_btn = cObj.IconButton('Icons/RGBA.png')
#         self.RGBA_btn.setToolTip('Set a composite RGB(A) image')
#         self.RGBA_btn.clicked.connect(self.calc_RGBAmap)

#     # X-Ray Maps Canvas
#         self.XMapsView = plots.HeatmapCanvas(tight=True)
#         self.XMapsView.setMinimumSize(100, 100)
#         self.XMapsView.mpl_connect('scroll_event', self.scroll_maps)

#     # Rectangle selector widget for X-Ray Maps Canvas
#         self.rectSel = cObj.RectSel(self.XMapsView.ax, self.selectROI, btns=[1])


#     # X-Ray Maps Navigation Toolbar
#         self.XMapsNTbar = plots.NavTbar(self.XMapsView, self)
#         self.XMapsNTbar.fixHomeAction()
#         self.XMapsNTbar.removeToolByIndex([3, 4, 8, 9])

#     # Reset zoom Action in X-Ray Maps NavTbar
#         self.resetZoomAction = QW.QAction(QIcon('Icons/zoom_home.png'),
#                                           'Reset zoom', self.XMapsNTbar)
#         self.resetZoomAction.triggered.connect(self.XMapsView.reset_zoom)

#     # Lock zoom Action in X-Ray Maps NavTbar
#         self.lockZoomAction = QW.QAction(QIcon('Icons/lockZoom.png'),
#                                           'Lock zoom', self.XMapsNTbar)
#         self.lockZoomAction.setCheckable(True)
#         self.lockZoomAction.toggled.connect(lambda tgl: self.XMapsView.toggle_zoomLock(tgl))

#     # ROI selection Action in X-Ray Maps NavTbar
#         self.ROIAction = QW.QAction(QIcon('Icons/ROI_selection.png'),
#                                     'Select Region Of Interest',
#                                     self.XMapsNTbar)
#         self.ROIAction.setCheckable(True)
#         self.ROIAction.setEnabled(False)
#         self.ROIAction.toggled.connect(self.toggle_ROI)

#     # Insert the custom Actions in X-Ray Maps NavTbar
#         self.XMapsNTbar.insertActions(self.XMapsNTbar.findChildren(QW.QAction)[10],
#                                       (self.resetZoomAction, self.lockZoomAction, self.ROIAction))

#     # Go To Pixel Widget in X-Ray Maps NavTbar
#         self.go2Pix = cObj.PixelFinder(self.XMapsView)
#         self.XMapsNTbar.addSeparator()
#         self.XMapsNTbar.addWidget(self.go2Pix)

#     # Current showed map path
#         self.curr_XMapPath = cObj.PathLabel()

#     # Histogram Canvas
#         self.histCanvas = plots.HistogramCanvas(logscale=True)
#         self.histCanvas.setMinimumSize(100, 100)
#         self.histCanvas.ax.yaxis.set_ticks_position('both')

#     # Histogram Navigation Toolbar
#         self.histNTbar = plots.NavTbar(self.histCanvas, self)
#         self.histNTbar.removeToolByIndex([2, 3, 4, 8, 9, 12])

#     # Set bin slider widget in Histogram NavTbar
#         self.bin_slider = QW.QSlider(Qt.Horizontal)
#         self.bin_slider.setSizePolicy(QW.QSizePolicy.MinimumExpanding,
#                                       QW.QSizePolicy.Fixed)
#         self.bin_slider.setMinimum(5)
#         self.bin_slider.setMaximum(100)
#         self.bin_slider.setSingleStep(5)
#         self.bin_slider.setSliderPosition(50)
#         self.bin_slider.valueChanged.connect(self.set_histBins)

#         self.histNTbar.insertWidget(self.histNTbar.findChildren(QW.QAction)[5],
#                                     self.bin_slider)

#     # HeatMap Scaler widget for the Histogram Canvas
#         self.scaler = cObj.HeatmapScaler(self.histCanvas.ax, self.XMapsView)

#     # Composite RGB(A) Image Canvas
#         self.RGBACanvas = plots.HeatmapCanvas(size=(5.0, 6.0), cbar=False)
#         self.RGBACanvas.ax.set_title('RGB(A) composite map')
#         self.RGBACanvas.setMinimumSize(100, 100)
#         CF.shareAxis(self.RGBACanvas.ax, self.XMapsView.ax, True)

#     # Composite RGB(A) Image NavTbar
#         self.RGBANTbar = plots.NavTbar(self.RGBACanvas, self)
#         self.RGBANTbar.fixHomeAction()
#         self.RGBANTbar.removeToolByIndex([3, 4, 8, 9])

#     # R-G-B-A pathlabels for composite RGB(A) image
#         (self.R_lbl,
#          self.G_lbl,
#          self.B_lbl,
#          self.A_lbl) = self.RGBA_lbls = (cObj.PathLabel(),
#                                          cObj.PathLabel(),
#                                          cObj.PathLabel(),
#                                          cObj.PathLabel())

#     # Adjust Window Layout
#         left_grid = QW.QGridLayout()
#         left_grid.addWidget(self.TOC, 0, 0, 1, 3)
#         left_grid.addWidget(self.loadMaps_btn, 1, 0)
#         left_grid.addWidget(self.refresh_btn, 1, 1)
#         left_grid.addWidget(self.delMaps_btn, 1, 2)
#         left_grid.addWidget(self.invertMaps_btn, 2, 0)
#         left_grid.addWidget(self.scaleCmap_btn, 2, 1)
#         left_grid.addWidget(self.RGBA_btn, 2, 2)
#         # left_grid.setRowStretch(0, 1)
#         TOC_group = cObj.GroupArea(left_grid, 'Loaded Maps')

#         Maps_viewBox = QW.QVBoxLayout()
#         Maps_viewBox.addWidget(self.XMapsNTbar)
#         Maps_viewBox.addWidget(self.XMapsView)
#         Maps_viewBox.addWidget(self.curr_XMapPath)

#         hist_viewBox = QW.QVBoxLayout()
#         hist_viewBox.addWidget(self.histNTbar)
#         hist_viewBox.addWidget(self.histCanvas)

#         RGBA_lblGrid = QW.QGridLayout()
#         for col, (title, label) in enumerate(zip(('R','G','B','A'), self.RGBA_lbls)):
#             title = QW.QLabel(title)
#             title.setStyleSheet('''color: black;''')
#             title.setSizePolicy(QW.QSizePolicy.Fixed,
#                                 QW.QSizePolicy.Fixed)
#             label.setAlignment(Qt.AlignCenter)

#             RGBA_lblGrid.addWidget(title, 0, col, alignment=Qt.AlignCenter)
#             RGBA_lblGrid.addWidget(label, 1, col)
#         RGBA_viewBox = QW.QVBoxLayout()
#         RGBA_viewBox.addWidget(self.RGBANTbar, 1)
#         RGBA_viewBox.addWidget(self.RGBACanvas, 2)
#         RGBA_viewBox.addLayout(RGBA_lblGrid)

#         subPlot_Vsplit = cObj.SplitterGroup((hist_viewBox, RGBA_viewBox),
#                                             orient=Qt.Vertical)

#         main_Hsplit = cObj.SplitterGroup((TOC_group, Maps_viewBox, subPlot_Vsplit),
#                                          (1, 3, 2))

#         mainLayout = QW.QHBoxLayout()
#         mainLayout.addWidget(main_Hsplit)
#         self.setLayout(mainLayout)


#     def reset_ui(self):
#         self.TOC.clear()
#         self.XMapsView.clear_canvas()
#         self.ROIAction.setChecked(False)
#         self.ROIAction.setEnabled(False)
#         self.histCanvas.clear_canvas()
#         self.curr_XMapPath.clear_all()

#     def update_TOC(self):
#         self.TOC.clear()
#         self.TOC.addItems([CF.path2fileName(path) for path in _loadedXMapsPath])

#     def _getSelectedMapsIdx(self):
#         selected = list(map(lambda item: self.TOC.row(item), self.TOC.selectedItems()))
#         # selected = [i for i in range(self.TOC.count()) if self.TOC.item(i).isSelected()]
#         return selected

#     def loadMaps(self, paths=None):
#         if paths is None:
#             paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load Maps',
#                                                        pref.get_dirPath('in'),
#                                                        'ASCII maps (*.txt *.gz)')
#         if paths:
#             pref.set_dirPath('in', dirname(paths[0]))
#             progBar = cObj.PopUpProgBar(self, len(paths), 'Loading Maps Data')
#             for n, p in enumerate(paths, start=1):
#                 if not progBar.wasCanceled():
#                     try:
#                         if p not in _loadedXMapsPath:
#                             # populate maps data list
#                             _loadedXMapsData.append(np.loadtxt(p, dtype='int32'))
#                             # populate maps path list
#                             _loadedXMapsPath.append(p)

#                     except Exception as e:
#                         progBar.setWindowModality(Qt.NonModal)
#                         cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Error',
#                                         f'Unexpected ASCII file:\n{p}.',
#                                         detailedText = repr(e))
#                         progBar.setWindowModality(Qt.WindowModal)

#                     finally:
#                         progBar.setValue(n)
#                 else: break

#             self.update_TOC()

#     def refresh_data(self):
#         selected = self._getSelectedMapsIdx()

#         if len(selected):
#             for idx in selected:
#                 path = _loadedXMapsPath[idx]
#                 try:
#                     _loadedXMapsData[idx] = np.loadtxt(path, dtype='int32')
#                 except Exception as e:
#                     cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Bad data source',
#                                     f'Unable to refresh the data source for the following file:\n{path}\n'\
#                                     'The file may have been corrupted, deleted, renamed or moved.',
#                                     detailedText = repr(e))
#         # Show last refreshed element
#             item = self.TOC.item(idx)
#             self.TOC.setCurrentItem(item)
#             self.showMap(item)

#     def delMaps(self):
#         selected = self._getSelectedMapsIdx()

#         if len(selected):
#             choice = QW.QMessageBox.question(self, "Remove Maps", "Remove selected maps?",
#                                              QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                              QW.QMessageBox.No)
#             if choice == QW.QMessageBox.Yes:
#                 for idx in sorted(selected, reverse=True):
#                     del _loadedXMapsPath[idx]
#                     del _loadedXMapsData[idx]
#                 self.update_TOC()

#             # Remove from canvas view the deleted map
#                 if self.TOC.count() > 0:
#                     item = self.TOC.item(idx-1 if idx != 0 else idx)
#                     self.TOC.setCurrentItem(item)
#                     self.showMap(item)
#                 else:
#                     self.reset_ui()

#     def invertMaps(self):
#         selected = self._getSelectedMapsIdx()
#         if len(selected):
#             msg_cbox = QW.QCheckBox('Edit source data file')
#             choice = cObj.RichMsgBox(self, QW.QMessageBox.Question, 'Invert',
#                                      'Invert the selected maps?',
#                                      QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                      QW.QMessageBox.No,
#                                      'If edit source data file is checked, the '\
#                                      'invertion will be permanent.',
#                                      msg_cbox)
#             if choice.clickedButton().text() == '&Yes':
#                 permanent = choice.checkBox().isChecked()
#                 for idx in selected:
#                     inverted = CF.invertArray(_loadedXMapsData[idx])
#                     _loadedXMapsData[idx] = inverted
#                     if permanent:
#                         try:
#                             path = _loadedXMapsPath[idx]
#                             np.savetxt(path, inverted, delimiter=' ', fmt='%d')
#                         except:
#                             QW.QMessageBox.critical(self, 'Error',
#                             f'An error occurred while trying to edit {path} source data.')
#                 self.showMap(self.TOC.currentItem())
#                 QW.QMessageBox.information(self, 'Invertion completed',
#                                            'Maps inverted with success')

#     def scaleColormap(self, enabled):
#         self.XMapsView.scale_clim(enabled, _loadedXMapsData)

#     def calc_RGBAmap(self):
#         selected = self._getSelectedMapsIdx()

#     # More than 4 maps selected -> Raise an error
#         if len(selected) > 4 :
#             return QW.QMessageBox.critical(self, 'Too many maps',
#                                            'Cannot select more than 4 maps.')

#     # No maps are selected -> Ignore event
#         elif len(selected) == 0:
#             return

#     # Correct number of selected maps (1 to 4)
#         else:
#             maps = [_loadedXMapsData[idx] for idx in selected]
#             shape = maps[0].shape
#         # Check that the selected maps share the same shape
#             if np.any([m.shape != shape for m in maps]):
#                 return QW.QMessageBox.critical(self, 'Unfitting maps',
#                                                'The selected maps have different shapes.')
#         # Build the RGBA composite map
#             RGBAmap = CF.composeRGBA(maps, shape)

#         # Populate the R-G-B-A pathlabels with maps paths
#             paths = [_loadedXMapsPath[idx] for idx in selected]
#             for i, lbl in enumerate(self.RGBA_lbls):
#                 try:
#                     lbl.set_fullpath(paths[i], predict_display=True)
#                 except IndexError:
#                     lbl.set_displayName('None')

#         self.RGBACanvas.update_canvas(RGBAmap, 'RGB(A) composite map')

#     def showMap(self, item):
#         if item is not None:
#             name = item.text()
#             idx = self.TOC.currentRow()
#             path = _loadedXMapsPath[idx]
#             data = _loadedXMapsData[idx]
#             try:
#                 self.curr_XMapPath.set_fullpath(path, predict_display=True)
#                 self.XMapsView.update_canvas(data, name)
#                 self.ROIAction.setEnabled(True)
#                 if self.ROIcoords is not None:
#                     r0,r1, c0,c1 = self.ROIcoords
#                     ROIdata = data[r0:r1, c0:c1]
#                 else:
#                     ROIdata = None
#                 self.histCanvas.update_canvas(data, ROIdata, name)
#                 self.scaler.hide()

#             except Exception as e:
#                 cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Error',
#                                 f'Unexpected ASCII file:\n{path}.',
#                                 detailedText = repr(e))
#                 self.curr_XMapPath.clear_all()

#     def scroll_maps(self, evt):
#         if evt.key == 'alt': # substitute with evt.modifiers in MPL v.3.7.1
#             step = evt.step
#             curr_idx = self.TOC.currentRow()

#             if curr_idx != -1:
#                 tot_items = self.TOC.count()
#                 curr_idx = (curr_idx + step) % tot_items

#                 self.TOC.setCurrentRow(curr_idx)
#                 self.showMap(self.TOC.currentItem())

#     def toggle_ROI(self, toggled):
#         self.rectSel.set_active(toggled)
#         self.rectSel.set_visible(toggled)
#         self.XMapsView.enablePicking(toggled)
#         if not toggled:
#             self.ROIcoords = None
#             data = self.XMapsView.image.get_array()
#             name = self.XMapsView.ax.get_title()
#             self.histCanvas.update_canvas(data, title=name)

#     def selectROI(self, eclick, erelease):
#         data = self.XMapsView.image.get_array()
#         mapName = self.XMapsView.ax.get_title()
#         mapShape = data.shape
#         self.ROIcoords = self.rectSel.fixed_extents(mapShape)
#         r0,r1, c0,c1 = self.ROIcoords
#         self.histCanvas.update_canvas(data, data[r0:r1, c0:c1], mapName)


#     def set_histBins(self, value):
#         self.bin_slider.setToolTip(f'Bins = {value}')
#         self.histCanvas.set_nbins(value)
#         if self.histCanvas.hist is not None:
#             mapName = self.XMapsView.ax.get_title()
#             mapData = self.XMapsView.img.get_array()
#             self.histCanvas.update_canvas(mapName, mapData, self.ROIcoords)





# # 2ND TAB --> MINERAL MAPS CLASSIFICATION VISUALIZER
# class MineralMapsTab(cObj.DraggableTool):
#     def __init__(self, parent):
#         self.parent = parent
#         super(MineralMapsTab, self).__init__()

#         self.setWindowTitle('Mineral Maps')
#         self.currentMap = None
#         self.currentPMap = None
#         self.edits = {}

#         self.init_ui()

#     def init_ui(self):
#     # TOC
#         self.TOC = QW.QListWidget()
#         self.TOC.setHorizontalScrollBar(cObj.StyledScrollBar(Qt.Horizontal))
#         self.TOC.setVerticalScrollBar(cObj.StyledScrollBar(Qt.Vertical))
#         self.TOC.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
#         self.TOC.keyPressEvent = lambda evt: evt.ignore() # ignore key press events to avoid bugs with TOC.currentRow()
#         self.TOC.itemPressed.connect(self.show_minMap)  # item pressed includes right-click

#     # TOC Option Buttons
#         self.addMaps_btn = cObj.IconButton('Icons/generic_add.png')
#         self.addMaps_btn.setToolTip('Load mineral maps.')
#         self.addMaps_btn.clicked.connect(lambda: self.loadMaps())

#         self.refresh_btn = cObj.IconButton('Icons/refresh.png')
#         self.refresh_btn.setToolTip('Refresh data source.')
#         self.refresh_btn.clicked.connect(self.refresh_data)

#         self.delMaps_btn = cObj.IconButton('Icons/generic_del.png')
#         self.delMaps_btn.setToolTip('Remove mineral maps.')
#         self.delMaps_btn.clicked.connect(self.delMaps)

#     # Adjust TOC widgets
#         TOC_grid = QW.QGridLayout()
#         TOC_grid.addWidget(self.TOC, 0, 0, 1, 3)
#         TOC_grid.addWidget(self.addMaps_btn, 1, 0)
#         TOC_grid.addWidget(self.refresh_btn, 1, 1)
#         TOC_grid.addWidget(self.delMaps_btn, 1, 2)
#         # TOC_grid.setRowStretch(0, 1)
#         TOC_group = cObj.GroupArea(TOC_grid, 'Loaded Maps')

#     # Mineral Map View Area
#         self.MinMapView = plots.ImageCanvas(tight=True)
#         self.MinMapView.setMinimumSize(100, 100)
#         self.MinMapView.mpl_connect('scroll_event', self.scroll_maps)
#         CF.shareAxis(self.parent._inputmapstab.XMapsView.ax, self.MinMapView.ax,
#                      pref.get_setting('plots/shareaxis', True, type=bool))

#     # Mineral Map Navigation Toolbar
#         self.MinMapNTbar = plots.NavTbar(self.MinMapView, self)
#         self.MinMapNTbar.fixHomeAction()
#         self.MinMapNTbar.removeToolByIndex([3, 4, 8, 9])

#     # Mineral Map path label
#         self.curr_imgPath = cObj.PathLabel()

#     # Draw/Edit Action in MinMap NavTbar
#         self.drawAction = QW.QAction(QIcon('Icons/edit.png'),
#                                     'Edit pixel values\n'\
#                                     'Drag left click to select pixels\n'\
#                                     'Right click to edit pixels',
#                                     self.MinMapNTbar)
#         self.drawAction.setCheckable(True)
#         self.drawAction.setEnabled(False)
#         self.drawAction.toggled.connect(self.toggleEditMode)

#     # Save Edits Action in MinMap NavTbar
#         self.saveEditsAction = QW.QAction(QIcon('Icons/saveEdit.png'),
#                                          'Save edits', self.MinMapNTbar)
#         self.saveEditsAction.setEnabled(False)
#         self.saveEditsAction.triggered.connect(self.saveEdits)

#     # Reset Zoom Action in MinMap NavTbar
#         self.resetZoomAction = QW.QAction(QIcon('Icons/zoom_home.png'),
#                                           'Reset zoom', self.MinMapView)
#         self.resetZoomAction.triggered.connect(self.MinMapView.reset_zoom)

#     # Lock zoom Action in MinMap NavTbar
#         self.lockZoomAction = QW.QAction(QIcon('Icons/lockZoom.png'),
#                                           'Lock zoom', self.MinMapNTbar)
#         self.lockZoomAction.setCheckable(True)
#         self.lockZoomAction.toggled.connect(self.lockMapZoom)

#     # Export Array Action in MinMap NavTbar
#         self.exportAction = QW.QAction(QIcon('Icons/export.png'),
#                                        'Export as numeric array',
#                                        self.MinMapNTbar)
#         self.exportAction.setEnabled(False)
#         self.exportAction.triggered.connect(self.exportArray)

#     # Go To Pixel Widget in MinMap NavTbar
#         self.go2Pix = cObj.PixelFinder(self.MinMapView)

#     # Add custom actions and widgets to MinMap NavTbar
#         self.MinMapNTbar.insertActions(self.MinMapNTbar.findChildren(QW.QAction)[10],
#                                       (self.resetZoomAction, self.lockZoomAction))
#         self.MinMapNTbar.insertActions(self.MinMapNTbar.findChildren(QW.QAction)[11],
#                                        (self.drawAction, self.saveEditsAction))
#         self.MinMapNTbar.insertSeparator(self.MinMapNTbar.findChildren(QW.QAction)[11])
#         self.MinMapNTbar.insertAction(self.MinMapNTbar.findChildren(QW.QAction)[11],
#                                       self.exportAction)

#         self.MinMapNTbar.addSeparator()
#         self.MinMapNTbar.addWidget(self.go2Pix)

#     # Rectangle Selector widget
#         self.rectSel = cObj.RectSel(self.MinMapView.ax, self.editPix, btns=[1, 3])

#     # Mineral Map Legend
#         self.legend = plots.Legend(self.MinMapView)
#         self.legend.itemColorChanged.connect(self.update_modePlot)

#     # Legend Options buttons
#         self.rename_btn = cObj.IconButton('Icons/rename.png')
#         self.rename_btn.setToolTip('Rename selected')
#         self.rename_btn.clicked.connect(self.rename_phase)

#         self.randCol_btn = cObj.IconButton('Icons/resetMapCol.png')
#         self.randCol_btn.setToolTip('Randomize colors')
#         self.randCol_btn.clicked.connect(self.randomize_color)

#         self.savePalette_btn = cObj.IconButton('Icons/savePalette.png')
#         self.savePalette_btn.setToolTip('Save the current palette for this map')
#         self.savePalette_btn.clicked.connect(self.save_palette)

#         self.highlight_btn = cObj.IconButton('Icons/highlight.png')
#         self.highlight_btn.setToolTip('Highlight selected')
#         self.highlight_btn.setCheckable(True)
#         self.highlight_btn.toggled.connect(self.highlight)

#     # Adjust Legend Layout
#         legend_grid = QW.QGridLayout()
#         legend_grid.addWidget(self.legend, 0, 0, 1, 3)
#         legend_grid.addWidget(self.rename_btn, 1, 0)
#         legend_grid.addWidget(self.randCol_btn, 1, 1)
#         legend_grid.addWidget(self.savePalette_btn, 1, 2)
#         legend_grid.addWidget(self.highlight_btn, 2, 0)
#         # legend_grid.setRowStretch(0, 1)
#         legend_group = cObj.GroupArea(legend_grid, 'Legend')

#     # Probability Map View Area
#         self.pMapView = plots.HeatmapCanvas(size=(6.0, 5.0))
#         self.pMapView.setMinimumSize(100, 100)
#         CF.shareAxis(self.pMapView.ax, self.MinMapView.ax, True)

#     # Probability Map Navigation Toolbar
#         self.pMapNTbar = plots.NavTbar(self.pMapView, self)
#         self.pMapNTbar.fixHomeAction()
#         self.pMapNTbar.removeToolByIndex([3, 4, 8, 9])

#     # Load Probability Map Action in PMap NavTbar
#         self.loadPMap_Action = QW.QAction(QIcon('Icons/generic_add_black.png'),
#                                           'Load Probability Map', self.pMapNTbar)
#         self.loadPMap_Action.triggered.connect(lambda: self.show_pMap())
#         self.pMapNTbar.addSeparator()
#         self.pMapNTbar.addAction(self.loadPMap_Action)

#     # Mode Bar Chart Area
#         self.modeView = plots.BarCanvas(size=(6.4, 3.6))
#         self.modeView.setMinimumSize(100, 100)

#     # Mode BarChart NavTbar
#         self.modeNTbar = plots.NavTbar(self.modeView, self)
#         self.modeNTbar.removeToolByIndex(list(range(2,10)) + [12])

#     # Labelize Action in Mode BarChart NavTbar
#         self.labelizeAction = QW.QAction(QIcon('Icons/labelize.png'),
#                                          'Show Amounts', self.modeNTbar)
#         self.labelizeAction.setCheckable(True)
#         self.labelizeAction.toggled.connect(self.modeView.show_amounts)
#         self.modeNTbar.insertAction(self.modeNTbar.findChildren(QW.QAction)[10],
#                                     self.labelizeAction)

#     # Adjust Main Layout
#         left_Vsplit = cObj.SplitterGroup((TOC_group, legend_group),
#                                          orient=Qt.Vertical)

#         mainPlotBox = QW.QVBoxLayout()
#         mainPlotBox.addWidget(self.MinMapNTbar)
#         mainPlotBox.addWidget(self.MinMapView)
#         mainPlotBox.addWidget(self.curr_imgPath)

#         pMapBox = QW.QVBoxLayout()
#         pMapBox.addWidget(self.pMapNTbar)
#         pMapBox.addWidget(self.pMapView)

#         modeBox = QW.QVBoxLayout()
#         modeBox.addWidget(self.modeNTbar)
#         modeBox.addWidget(self.modeView)

#         subPlot_VSplit = cObj.SplitterGroup((modeBox, pMapBox),
#                                             orient=Qt.Vertical)

#         main_Hsplit = cObj.SplitterGroup((left_Vsplit, mainPlotBox, subPlot_VSplit),
#                                          (1, 3, 2))

#         mainLayout = QW.QHBoxLayout()
#         mainLayout.addWidget(main_Hsplit)
#         self.setLayout(mainLayout)

#     def reset_ui(self):
#         self.TOC.clear()
#         self.MinMapView.clear_canvas()
#         self.pMapView.clear_canvas()
#         self.modeView.clear_canvas()
#         self.legend.update()
#         self.curr_imgPath.clear_all()
#         self.currentMap = None
#         self.currentPMap = None
#         self.drawAction.setEnabled(False)
#         self.exportAction.setEnabled(False)

#     def update_TOC(self):
#         self.TOC.clear()
#         self.TOC.addItems([CF.path2fileName(MinMap.filepath) for MinMap in _loadedMinMapsData])

#     def _getSelectedMapsIdx(self):
#         selected = list(map(lambda item: self.TOC.row(item), self.TOC.selectedItems()))
#         # selected = [i for i in range(self.TOC.count()) if self.TOC.item(i).isSelected()]
#         return selected

#     def loadMaps(self, paths=None):
#         if paths is None:
#             paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load mineral maps',
#                                                        pref.get_dirPath('in'),
#                                                       '''Mineral Maps (*mmap)
#                                                          ASCII maps (*.txt *.gz)''')
#         if paths:
#             pref.set_dirPath('in', dirname(paths[0]))
#             progBar = cObj.PopUpProgBar(self, len(paths), 'Loading Maps Data')
#             for n, p in enumerate(paths, start=1):
#                 if not progBar.wasCanceled():
#                     try:
#                         MinMap = MineralMap.load(p)
#                         if MinMap not in _loadedMinMapsData:
#                             _loadedMinMapsData.append(MinMap)
#                     except Exception as e:
#                         progBar.setWindowModality(Qt.NonModal)
#                         cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Error',
#                                         f'Unexpected ASCII file:\n{p}.',
#                                         detailedText = repr(e))
#                         progBar.setWindowModality(Qt.WindowModal)

#                     finally:
#                         progBar.setValue(n)
#                 else: break

#             self.update_TOC()

#     def refresh_data(self):
#         selected = self._getSelectedMapsIdx()

#         if len(selected):
#             for idx in selected:
#                 path = _loadedMinMapsData[idx].filepath
#                 try:
#                     _loadedMinMapsData[idx] = MineralMap.load(path)
#                 except Exception as e:
#                     cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Bad data source',
#                                     f'Unable to refresh the data source for the following file:\n{path}\n'\
#                                     'The file may have been corrupted, deleted, renamed or moved.',
#                                     detailedText = repr(e))
#         # Show last refreshed element
#             item = self.TOC.item(idx)
#             self.TOC.setCurrentItem(item)
#             self.show_minMap(item)

#     def delMaps(self):
#         if self.drawAction.isChecked():
#             QW.QMessageBox.critical(self, 'Error',
#                                     'Cannot remove maps while edit mode is active.')
#             return

#         selected = self._getSelectedMapsIdx()

#         if len(selected):
#             choice = QW.QMessageBox.question(self, "Remove Maps", "Remove selected maps?",
#                                              QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                              QW.QMessageBox.No)
#             if choice == QW.QMessageBox.Yes:
#                 for idx in sorted(selected, reverse=True):
#                     del _loadedMinMapsData[idx]
#                 self.update_TOC()

#             # Remove from canvas view the deleted map
#                 if self.TOC.count() > 0:
#                     item = self.TOC.item(idx-1 if idx != 0 else idx)
#                     self.TOC.setCurrentItem(item)
#                     self.show_minMap(item)
#                 else:
#                     self.reset_ui()


#     def show_minMap(self, item):
#         idx = self.TOC.currentRow()
#         MinMap = _loadedMinMapsData[idx]
#         data = MinMap.minmap_encoded
#         encoder = MinMap.encoder
#         path = MinMap.filepath
#         colors = MinMap.palette.values()
#         name = item.text()

#         try:
#             self.highlight_btn.setChecked(False)
#             self.curr_imgPath.set_fullpath(path, predict_display=True)
#             self.MinMapView.draw_discretemap(data, encoder, colors, name)
#             self.currentMap = MinMap.minmap
#             self.legend.update(MinMap)
#             self.update_modePlot(MinMap)
#             self.drawAction.setEnabled(True)
#             self.exportAction.setEnabled(True)

#         except Exception as e:
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Error',
#                             f'Unexpected ASCII file:\n{path}.',
#                             detailedText = repr(e))
#             self.curr_imgPath.clear_all()
#             return

#         # Try to also show the linked Probability Map
#         probmap = MinMap.probmap
#         if probmap is not None:
#             self.show_pMap(MinMap)
#         else:
#             self.pMapView.clear_canvas()
#             self.currentPMap = None

#     def show_pMap(self, path=None):
#         if path is None:
#             path, _ = QW.QFileDialog.getOpenFileName(self, 'Load probability map',
#                                                      pref.get_dirPath('in'),
#                                                      'ASCII maps (*.txt *.gz)')

#         if isinstance(path, MineralMap):
#             pMap = path.probmap
#             self.pMapView.update_canvas(pMap, 'Probability Map')
#             self.currentPMap = pMap

#         elif path:
#             pref.set_dirPath('in', dirname(path))
#             try:
#                 pMap = np.loadtxt(path) # no need to specify dtype since it's float
#                 self.pMapView.update_canvas('Probability Map', pMap)
#                 self.currentPMap = pMap
#             except Exception as e:
#                 cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'Error',
#                                 f'Unexpected ASCII file:\n{path}.',
#                                 detailedText = repr(e))

#     def update_modePlot(self, MinMap):
#         if MinMap is not None:
#             mode = list(MinMap.mode.values())
#             lbl = [MinMap.asPhase(ID) for ID in MinMap.mode.keys()]
#             col = list(MinMap.palette.values())
#             self.modeView.update_canvas(mode, title='Mode', tickslabels=lbl, colors=col)

#     def rename_phase(self):
#         selected = self.legend.currentItem()
#         if selected is not None:
#             name, ok = QW.QInputDialog.getText(self, 'Rename',
#                                                'Type name (max. 8 ASCII characters):',
#                                                flags=Qt.MSWindowsFixedSizeDialogHint)
#             if ok and name != '':
#                 if len(name) > 8:
#                     QW.QMessageBox.critical(self, 'Invalid name',
#                                             'Please enter max. 8 characters.')
#                 elif not name.isascii():
#                     QW.QMessageBox.critical(self, 'Invalid name',
#                                                 'Only ASCII characters are supported.')
#                 else:
#                     old_name = selected.text().split(' - ')[0]
#                     msg_cbox = QW.QCheckBox('Edit source data file')
#                     choice = cObj.RichMsgBox(self, QW.QMessageBox.Question, 'Rename',
#                                              f'Rename {old_name} phase as {name}?',
#                                              QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                              QW.QMessageBox.Yes,
#                                              'If edit source data file is checked, the '\
#                                              'phase renaming will be permanent.',
#                                              msg_cbox)
#                     if choice.clickedButton().text() == '&Yes':
#                         # Replace old phase name with new entered name
#                         self.currentMap[self.currentMap==old_name] = name

#                         # Get references to the edited mineral map
#                         mapName = self.TOC.currentItem().text()
#                         idx = self.TOC.currentRow()

#                         # Update stored data of the edited mineral map
#                         _loadedMinMapsData[idx].edit_minmap(self.currentMap)
#                         MinMap = _loadedMinMapsData[idx]

#                         # Update mineral map canvas, legend and mode canvas
#                         self.MinMapView.draw_discretemap(MinMap.minmap_encoded, MinMap.encoder,
#                                                       list(MinMap.palette.values()), mapName)
#                         self.legend.update(MinMap)
#                         self.update_modePlot(MinMap)

#                         # Override map data on disk (try for safety measurements)
#                         if choice.checkBox().isChecked():
#                             try:
#                                 MinMap.save(MinMap.filepath)
#                             except:
#                                 QW.QMessageBox.critical(self, 'Error',
#                                 'An error occurred while trying to edit source data.')


#     def randomize_color(self):
#         if self.currentMap is not None:
#             idx = self.TOC.currentRow()
#             MinMap = _loadedMinMapsData[idx]
#             rand_colors = MinMap.rand_colorlist()
#             MinMap.set_palette(rand_colors)

#             self.MinMapView.alter_cmap(rand_colors)
#             self.legend.update(MinMap)
#             self.update_modePlot(MinMap)

#     def save_palette(self):
#         pass

#     def highlight(self, enable): #!!! da sistemare
#         if self.legend.count() > 0 and self.TOC.currentItem() is not None:
#             MinMap = _loadedMinMapsData[self.TOC.currentRow()]
#             if enable:
#                 if self.legend.currentItem() is not None:
#                     phase = self.legend.currentItem().text().split(' - ')[0]
#                     mask = self.currentMap != phase
#                     minmap_ma, minmap_encoded_ma, probmap_ma = MinMap.apply_mask(mask)
#                     title = self.MinMapView.ax.get_title()
#                     self.MinMapView.draw_discretemap(minmap_encoded_ma, MinMap.encoder,
#                                                   list(MinMap.palette.values()), title)
#                     if self.currentPMap is not None:
#                         self.pMapView.update_canvas(probmap_ma, 'Probability Map')
#                 else:
#                     self.highlight_btn.setChecked(False)
#             else:
#                 self.MinMapView.draw_discretemap(MinMap.minmap_encoded, MinMap.encoder,
#                                               list(MinMap.palette.values()), title)

#                 if self.currentPMap is not None:
#                     self.pMapView.update_canvas(MinMap.probmap, 'Probability Map')

#             self.legend.update(MinMap)

#     def lockMapZoom(self, enabled):
#         self.MinMapView.toggle_zoomLock(enabled)
#         self.pMapView.toggle_zoomLock(enabled)

#     def toggleEditMode(self, toggled):
#         if not toggled:
#             choice = QW.QMessageBox.question(self, 'Quit Editing',
#                      'Do you want to quit edit mode? Warning: edits will not be saved.',
#                                              QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                              QW.QMessageBox.No)
#             if choice == QW.QMessageBox.Yes:
#                 self.edits.clear()
#             else:
#                 self.sender().setChecked(True)
#                 toggled = True

#         self.saveEditsAction.setEnabled(toggled)
#         self.rectSel.set_active(toggled)
#         self.rectSel.set_visible(toggled)
#         self.MinMapView.enable_picking(toggled)

#     def editPix(self, eclick, erelease):
#         if erelease.button == MouseButton.RIGHT:
#             mapShape = self.currentMap.shape
#             extents = self.rectSel.fixed_extents(mapShape)
#             if extents is None: return

#             while True:
#                 pixval, ok = QW.QInputDialog.getText(self, 'Edit Name',
#                                                      'Type name (max. 8 ASCII characters):')
#                 if not ok: break
#                 else:
#                     if pixval == '':
#                         QW.QMessageBox.critical(self, 'Invalid name',
#                                                 'Name cannot be empty.')
#                     elif len(pixval) > 8:
#                         QW.QMessageBox.critical(self, 'Invalid name',
#                                                 'Please enter max. 8 characters.')
#                     elif not pixval.isascii():
#                         QW.QMessageBox.critical(self, 'Invalid name',
#                                                 'Only ASCII characters are supported.')
#                     else:
#                     # Compile the edits dictionary
#                         ymin, ymax, xmin, xmax = extents
#                         for col in range(xmin, xmax):
#                             for row in range(ymin, ymax):
#                                 self.edits[(row, col)] = pixval
#                         break



#     def saveEdits(self):
#         choice = QW.QMessageBox.question(self, 'Save Edits', 'Do you want to save edits?',
#                                          QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                          QW.QMessageBox.Yes)
#         if choice == QW.QMessageBox.Yes:
#             self.editorDialog = dialogs.PixelEditor((_loadedXMapsPath, _loadedXMapsData),
#                                                     self.currentMap, self.edits, self)
#             self.editorDialog.show()

#     def exportArray(self):
#         msg_cbox = QW.QCheckBox('Include translation dictionary')
#         choice = cObj.RichMsgBox(self, QW.QMessageBox.Question, 'Export Map',
#                                  'Export map as a numeric array',
#                                  QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                  QW.QMessageBox.Yes,
#                                  'The translation dictionary is a stand-alone '\
#                                  'text file containing the textual references '\
#                                  'to the numerical values ​​of the mineral classes',
#                                  msg_cbox)

#         if choice.clickedButton().text() == '&Yes':
#             outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Export Map',
#                                                         pref.get_dirPath('out'),
#                                                         '''ASCII file (*.txt)''')
#             if outpath:
#                 pref.set_dirPath('out', dirname(outpath))
#                 includeDict = choice.checkBox().isChecked()
#                 self.MinMapView.export_array(outpath, includeDict)

#     def scroll_maps(self, evt):
#         if evt.key == 'alt': # substitute with evt.modifiers in MPL v.3.7.1
#             step = evt.step
#             curr_idx = self.TOC.currentRow()

#             if curr_idx != -1:
#                 tot_items = self.TOC.count()
#                 curr_idx = (curr_idx + step) % tot_items

#                 self.TOC.setCurrentRow(curr_idx)
#                 self.show_minMap(self.TOC.currentItem())






# # palette = QG.QPalette()
# # palette.setColor(QG.QPalette.Window, QG.QColor(53, 53, 53))
# # palette.setColor(QG.QPalette.WindowText, QC.Qt.white)
# # palette.setColor(QG.QPalette.Base, QG.QColor(25, 25, 25))
# # palette.setColor(QG.QPalette.AlternateBase, QG.QColor(53, 53, 53))
# # palette.setColor(QG.QPalette.ToolTipBase, QC.Qt.black)
# # palette.setColor(QG.QPalette.ToolTipText, QC.Qt.white)
# # palette.setColor(QG.QPalette.Text, QC.Qt.white)
# # palette.setColor(QG.QPalette.Button, QG.QColor(53, 53, 53))
# # palette.setColor(QG.QPalette.ButtonText, QC.Qt.white)
# # palette.setColor(QG.QPalette.BrightText, QC.Qt.red)
# # palette.setColor(QG.QPalette.Link, QG.QColor(42, 130, 218))
# # palette.setColor(QG.QPalette.Highlight, QG.QColor(42, 130, 218))
# # palette.setColor(QG.QPalette.HighlightedText, QC.Qt.black)










