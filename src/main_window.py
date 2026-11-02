# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:31:39 2023

@author: albdag
"""

import json
import pickle
import zipfile

from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5 import QtWidgets as QW

from base import InputMap, MineralMap, Mask
import convenient_functions as cf
import custom_widgets as CW
import dialogs
import docks
import settings
import style
import tools


class MainWindow(QW.QMainWindow):

    projectModified = QC.pyqtSignal()

    def __init__(self) -> None:
        '''
        The main window of X-Min Learn.

        '''
        super().__init__()

    # Set main window properties
        self.setWindowTitle('New project[*]') 
        self.setWindowIcon(style.getIcon('LOGO_32'))
        self.setDockOptions(self.DockOption.AllowTabbedDocks)
        self.setAnimated(settings.manager.get('GUI/smooth_animation'))
        self.setStyleSheet(style.SS_MAINWINDOW)

    # Set main window attributes
        self.project_path: str | None = None
        self.open_tools: list[tools.DraggableTool] = []

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.showMaximized()
        self._connect_slots()

    # Restore last window state (toolboxes and panes)
        self.restoreState(settings.manager.get('system/window_state'), version=0)


    def _init_ui(self) -> None:
        '''
        GUI constructor.
        
        '''
    # Data Viewer
        self.dataViewer = tools.DataViewer()

    # Central Widget (Tab Widget)
        self.tabWidget = MainTabWidget(self)
        self.tabWidget.add_tabified_widget(self.dataViewer)
        self.setCentralWidget(self.tabWidget)

    # Hide the close button from the data viewer tab
        btn_pos = QW.QTabBar.ButtonPosition.RightSide
        btn = self.tabWidget.tabbar.tabButton(0, btn_pos)
        if btn is not None:
            btn.hide()

    # Initialize panes
        self._init_panes()

    # Initialize menu/toolbar actions
        self._init_actions()

    # Initialize toolbars
        self._init_toolbars()

    # Initialize menu
        self._init_menu()

    # Add app version label to status bar
        version_lbl = QW.QLabel(QW.qApp.applicationVersion())
        if (statusbar := self.statusBar()) is not None:
            statusbar.addPermanentWidget(version_lbl)


    def _init_panes(self) -> None:
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
        manager_pane = docks.Pane(
            self.dataManager,
            'Data Manager', 
            style.getIcon('DATA_MANAGER'),
            scroll=False,
            tip='Load, view, and manage your project data.'
        )
        histogram_pane = docks.Pane(
            self.histViewer,
            'Histogram',
            style.getIcon('HISTOGRAM_VIEWER'),
            tip='Interact with input map histograms.'
        )
        mode_pane = docks.Pane(
            self.modeViewer,
            'Mode',
            style.getIcon('MODE_VIEWER'),
            tip='Inspect and edit mineral classes.'
        )
        roi_pane = docks.Pane(
            self.roiEditor,
            'ROI Editor',
            style.getIcon('ROI_EDITOR'),
            tip='Define and manage Regions of Interest (ROIs).'
        )
        probmap_pane = docks.Pane(
            self.pmapViewer,
            'Probability Map',
            style.getIcon('PROBABILITY_MAP_VIEWER'),
            tip='Evaluate mineral classifications confidence.'
        )
        rgba_pane = docks.Pane(
            self.rgbaViewer,
            'RGBA Map',
            style.getIcon('RGBA_MAP_VIEWER'),
            tip='Create RGBA composite maps from input maps.'
        )
        
        self.panes = (manager_pane, histogram_pane, mode_pane, roi_pane, 
                      probmap_pane, rgba_pane)
        
    # Store the panes toggle view actions, set their icons and their objectName
        self.panes_tva: list[QW.QAction] = []
        for p in self.panes:
            p.setObjectName(p.title) # required for saving pane state
            action = p.toggleViewAction()
            if action is not None:
                action.setStatusTip(p.tip)
                if p.icon is not None:
                    action.setIcon(p.icon)
                    action.setIconVisibleInMenu(True)
                self.panes_tva.append(action)

    # Add panes to main window
        self.addPane(QC.Qt.DockWidgetArea.LeftDockWidgetArea, manager_pane, True)
        self.addPane(QC.Qt.DockWidgetArea.LeftDockWidgetArea, histogram_pane)
        self.addPane(QC.Qt.DockWidgetArea.LeftDockWidgetArea, mode_pane)
        self.tabifyDockWidget(manager_pane, histogram_pane)
        self.tabifyDockWidget(manager_pane, mode_pane)
        self.addPane(QC.Qt.DockWidgetArea.RightDockWidgetArea, roi_pane)
        self.addPane(QC.Qt.DockWidgetArea.RightDockWidgetArea, probmap_pane)
        self.addPane(QC.Qt.DockWidgetArea.RightDockWidgetArea, rgba_pane)
        self.tabifyDockWidget(roi_pane, probmap_pane)
        self.tabifyDockWidget(roi_pane, rgba_pane)

    # Resize panes
        w = max([p.widget().minimumSizeHint().width() for p in self.panes])
        self.resizeDocks(self.panes, [w] * len(self.panes), style.O_HORIZONTAL)
        

    def _init_actions(self) -> None:
        '''
        Initialize actions shared by menu and toolbars.
        
        '''
    # New project
        self.new_project_action = QW.QAction(
            style.getIcon('FILE_BLANK'), '&New project')
        self.new_project_action.setShortcut('Ctrl+N')

    # Open project
        self.open_project_action = QW.QAction(
            style.getIcon('OPEN'), '&Open project')
        self.open_project_action.setShortcut('Ctrl+O')

    # Save project
        self.save_project_action = QW.QAction(
            style.getIcon('SAVE'), '&Save project')
        self.save_project_action.setShortcut('Ctrl+S')

    # Save project as...
        self.save_project_as_action = QW.QAction(
            style.getIcon('SAVE_AS'), '&Save project as...')
        self.save_project_as_action.setShortcut('Ctrl+Shift+S')

    # Quit app
        self.close_action = QW.QAction('&Exit')

    # Import Input Maps 
        self.load_inmaps_action = QW.QAction(
            style.getIcon('STACK'), '&Input Maps')
        self.load_inmaps_action.setShortcut('Ctrl+I')

    # Import Mineral Maps
        self.load_minmaps_action = QW.QAction(
            style.getIcon('MINERAL'), '&Mineral Maps')
        self.load_minmaps_action.setShortcut('Ctrl+M')

    # Import Masks
        self.load_masks_action = QW.QAction(
            style.getIcon('MASK'), 'Mas&ks')
        self.load_masks_action.setShortcut('Ctrl+K')

    # Launch Preferences
        self.pref_action = QW.QAction(
            style.getIcon('WRENCH'), 'Preferences')

    # Launch Convert Image to Input Map
        self.conv2inmap_action = QW.QAction('Image to Input Map')
        self.conv2inmap_action.setStatusTip('Convert image to Input Map.')

    # Launch Convert Image to Mineral Maps
        self.conv2minmap_action = QW.QAction('Image to Mineral Map')
        self.conv2minmap_action.setStatusTip('Convert image to Mineral Map.')

    # Launch Generate Dummy Maps
        self.dummymap_action = QW.QAction('Generate &Dummy Maps')
        self.dummymap_action.setStatusTip('Build placeholder noisy maps.')

    # Launch Sub-sample Dataset
        self.subsample_ds_action = QW.QAction('&Sub-sample Dataset')
        self.subsample_ds_action.setStatusTip('Extract sub-sets from dataset.')

    # Launch Merge Datasets
        self.merge_ds_action = QW.QAction('&Merge Datasets')
        self.merge_ds_action.setStatusTip('Merge multiple datasets into one.')

    # Launch Dataset Builder
        self.ds_builder_action = QW.QAction(
            style.getIcon('DATASET_BUILDER'), 'Dataset &Builder')
        self.ds_builder_action.setShortcut('Ctrl+Alt+B')
        self.ds_builder_action.setStatusTip('Compile ground truth datasets.')

    # Launch Model Learner
        self.model_learner_action = QW.QAction(
            style.getIcon('MODEL_LEARNER'), 'Model &Learner')
        self.model_learner_action.setShortcut('Ctrl+Alt+L')
        self.model_learner_action.setStatusTip('Build machine learning models.')

    # Launch Mineral Classifier
        self.classifier_action = QW.QAction(
            style.getIcon('MINERAL_CLASSIFIER'), 'Mineral &Classifier')
        self.classifier_action.setShortcut('Ctrl+Alt+C')
        self.classifier_action.setStatusTip('Predict mineral data.')

    # Launch Phase Refiner
        self.refiner_action = QW.QAction(
            style.getIcon('PHASE_REFINER'), 'Phase &Refiner')
        self.refiner_action.setShortcut('Ctrl+Alt+R')
        self.refiner_action.setStatusTip('Refine mineral maps.')

    # About X-Min Learn
        self.about_action = QW.QAction('About X-Min Learn')
        self.about_action.setStatusTip('About X-Min Learn app.')

    # About Qt
        self.aboutqt_action = QW.QAction('About Qt')
        self.aboutqt_action.setStatusTip('About Qt toolkit.')


    def _init_toolbars(self) -> None:
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
        # separator + other future tools (e.g. map algebra calculator)
    
    # Panes toolbar
        self.panes_toolbar = QW.QToolBar('Panes toolbar')
        self.panes_toolbar.setObjectName('PanesToolbar') # to save its state
        self.panes_toolbar.setFloatable(False)
        self.panes_toolbar.setStyleSheet(style.SS_MAINTOOLBAR)
        self.panes_toolbar.addActions(self.panes_tva)

    # Set default toolbars position in the window
        self.addToolBar(QC.Qt.ToolBarArea.LeftToolBarArea, self.panes_toolbar)
        self.addToolBar(QC.Qt.ToolBarArea.LeftToolBarArea, self.tools_toolbar)


    def _init_menu(self) -> None:
        '''
        Initialize main menu.

        Raises
        ------
        RuntimeError
            Raised if the menu bar initialization fails.
        
        '''
    # Set custom stylesheet to menu bar
        if (menu_bar := self.menuBar()) is None:
            raise RuntimeError("Failed to initialize menu bar.")
        menu_bar.setStyleSheet(style.SS_MENUBAR + style.SS_MENU)

    # Initialize menus
        file_menu = QW.QMenu('&File', menu_bar)
        dataset_menu = QW.QMenu('&Dataset', menu_bar)
        class_menu = QW.QMenu('&Classification', menu_bar)
        postclass_menu = QW.QMenu('&Post-classification', menu_bar)
        utility_menu = QW.QMenu('&Utility', menu_bar)
        view_menu = QW.QMenu('&View', menu_bar)
        info_menu = QW.QMenu('&?', menu_bar)

    # File Menu
        file_menu.addActions((
            self.new_project_action,
            self.open_project_action,
            self.save_project_action,
            self.save_project_as_action
        ))
        file_menu.addSeparator()

        # File -> Import sub-menu
        import_submenu = QW.QMenu('&Import...', file_menu)
        import_submenu.setIcon(style.getIcon('IMPORT'))
        import_submenu.addActions((
            self.load_inmaps_action,
            self.load_minmaps_action,
            self.load_masks_action
        ))
        file_menu.addMenu(import_submenu)

        file_menu.addActions((self.pref_action, self.close_action))
        menu_bar.addMenu(file_menu)

    # Dataset Menu
        dataset_menu.addActions((
            self.ds_builder_action,
            self.subsample_ds_action,
            self.merge_ds_action
        ))
        menu_bar.addMenu(dataset_menu)

    # Classification Menu
        class_menu.addActions((
            self.classifier_action, self.model_learner_action))
        menu_bar.addMenu(class_menu)

    # Post-classification Menu
        postclass_menu.addAction(self.refiner_action)
        menu_bar.addMenu(postclass_menu)

    # Utility Menu
        # Utility -> Convert sub-menu
        convert_submenu = QW.QMenu('&Convert', utility_menu)
        convert_submenu.addActions((
            self.conv2inmap_action, self.conv2minmap_action))
        utility_menu.addMenu(convert_submenu)

        utility_menu.addAction(self.dummymap_action)
        menu_bar.addMenu(utility_menu)

    # View Menu
        # View -> Panes sub-menu
        panes_submenu = QW.QMenu('&Panes', view_menu)
        panes_submenu.addActions(self.panes_tva)
        view_menu.addMenu(panes_submenu)

        view_menu.addSeparator()
        view_menu.addAction(self.panes_toolbar.toggleViewAction())
        view_menu.addAction(self.tools_toolbar.toggleViewAction())
        menu_bar.addMenu(view_menu)
        
    # Info menu
        info_menu.addActions((self.about_action, self.aboutqt_action))
        # Separator
        # Help (user-guide?)
        # Separator
        # Check updates (link to github page?)
        menu_bar.addMenu(info_menu)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Project modification-aware signal, forwarded by some child signals
        self.projectModified.connect(self.on_project_modified)

    # Main Tab Widget signals
        self.tabWidget.tabToolDetachRequested.connect(self.popOutTool)
        self.tabWidget.tabToolCloseRequested.connect(self.on_close_tabified_tool)

    # Connect all panes toggle view actions with a custom slot to force showing
    # them when they are tabified
        for p, a, in zip(self.panes, self.panes_tva):
            a.triggered.connect(lambda t, p=p: self.setPaneVisibility(p, t))

    # Data Manager signals 
        self.dataManager.updateSceneRequested.connect(self.updateScene)
        self.dataManager.clearSceneRequested.connect(self.clearScene)
        self.dataManager.rgbaChannelSet.connect(self.setRgbaChannel)
        self.dataManager.dataModified.connect(self.projectModified)
        
    # Histogram Viewer signals
        self.histViewer.scalerRangeChanged.connect(self.projectModified)

    # Mode Viewer signals
        self.modeViewer.updateSceneRequested.connect(self.updateScene)

    # ROI Editor signals
        self.roiEditor.rectangleSelectorUpdated.connect(self.updateHistogram)
        self.roiEditor.rectangleSelectorUpdated.connect(self.projectModified)
        self.roiEditor.roiDataAltered.connect(self.projectModified)
        self.roiEditor.autoroiSamplesRequested.connect(
            lambda: self.roiEditor.updateDetectorSamplesList(
                self.dataManager.getAllGroups()))
        
    # Probability Map Viewer signals
        self.pmapViewer.probabilityRangeChanged.connect(self.projectModified)
        
    # RGBA Viewer signals
        self.rgbaViewer.rgbaModified.connect(self.projectModified)
        
    # Project-related actions (new, open, save) 
        self.new_project_action.triggered.connect(self.newProject)
        self.open_project_action.triggered.connect(self.openProject)
        self.save_project_action.triggered.connect(self._onSaveProject)
        self.save_project_as_action.triggered.connect(self._onSaveProject)
        
    # Quit app 
        self.close_action.triggered.connect(self._onCloseAction)

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
            lambda: self.openTool(
                'DatasetBuilder', settings.manager.get('GUI/tools_tabbed')))

    # Launch Model Learner 
        self.model_learner_action.triggered.connect(
            lambda: self.openTool(
                'ModelLearner', settings.manager.get('GUI/tools_tabbed')))

    # Launch Mineral Classifier 
        self.classifier_action.triggered.connect(
            lambda: self.openTool(
                'MineralClassifier', settings.manager.get('GUI/tools_tabbed')))

    # Launch Phase Refiner 
        self.refiner_action.triggered.connect(
            lambda: self.openTool(
                'PhaseRefiner', settings.manager.get('GUI/tools_tabbed')))


    def on_project_modified(self) -> None:
        '''
        Actions to perform when the current project is modified.

        '''
        self.setWindowModified(True)


    def on_close_tabified_tool(self, idx: int) -> None:
        '''
        Actions to be performed when user requests to close a tab of the main
        tab widget containing a tool.

        Parameters
        ----------
        idx : int
            Index of the tab to be closed.
            
        '''
        # Ask confirm to close the tool contained in the tab
        if isinstance(tool := self.tabWidget.widget(idx), tools.DraggableTool):
            if tool.is_safe_to_close():
                self.unregister_tool(tool)
                self.tabWidget.closeTab(idx)


    def about(self) -> None:
        '''
        Show the about X-Min Learn dialog.

        '''
        title = "About X-Min Learn"
        version = QW.qApp.applicationVersion()
        author = "Dr. Alberto D'Agostino (Ph.D.)"
        affiliation = "University of Catania"
        email1 = "dagostino.alberto@hotmail.it"
        contact1 = f"href='mailto:{email1}' > {email1}"
        email2 = 'alberto.dagostino@unict.it'
        contact2 = f"href='mailto:{email2}' > {email2}"
        license = "https://www.gnu.org/licenses/gpl-3.0.html"
        repo = "https://github.com/albdag/X-Min-Learn"

        html = f'''
        <p><span style="font-size: 12pt; font-weight: bold;">{title}</span></p>
        <p>Currently used version: <span style="font-weight: bold;">{version}</span></p>
        <p><span style="font-weight: bold;">Author</span>: {author}<br>
        {affiliation}</p>
        <p><span style="font-weight: bold;">Contacts</span>:<br>
        <a {contact1}</a><br>
        <a {contact2}</a></p>
        <p><br>X-Min Learn is an open-source project (<a href='{license}'>GPLv3</a>).
        Many of the app icons are provided by <a href='https://icons8.it/'>Icons8</a>.
        For more info check the <a href='{repo}'>project page</a>.</p>
        '''
        
        QW.QMessageBox.about(self, 'About X-Min Learn', html)


    def addPane(
        self,
        area: QC.Qt.DockWidgetArea,
        pane: docks.Pane,
        visible: bool = False
    ) -> None:
        '''
        Add 'pane' to main window as a dock widget, located in the specified
        dock widget area 'area'. 

        Parameters
        ----------
        area : QC.Qt.DockWidgetArea
            Location of the pane in the main window.
        pane : docks.Pane
            Pane to be added.
        visible : bool, optional
            Whether the pane should be visible. The default is False.

        '''
        self.addDockWidget(area, pane)
        pane.setVisible(visible)

    
    def setPaneVisibility(self, pane: docks.Pane, visible: bool) -> None:
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


    def createPopupMenu(self) -> QW.QMenu:
        '''
        Reimplementation of the default 'createPopupMenu' method, that returns
        a popup menu with a custom style.

        Returns
        -------
        QW.QMenu
            Styled popup menu.

        Raises
        -------
        RuntimeError
            Raised if the creation of the popup menu fails.

        '''
        if (popupmenu := super().createPopupMenu()) is None:
            raise RuntimeError("Failed to create popup menu.")
        popupmenu.setStyleSheet(style.SS_MENU)
        return popupmenu


    def openTool(self, toolname: str, tabbed: bool = True) -> None:
        '''
        Show a new instance of a draggable tool.

        Parameters
        ----------
        toolname : str
            Identifier string for the popup tool to be opened. Accepted strings
            are: 'DatasetBuilder', 'ModelLearner', 'MineralClassifier' and
            'PhaseRefiner'.
        tabbed : bool, optional
            Whether the tool should be shown as a tab of the main tab widget 
            (= True) or as a floating window (= False). The default is True.

        '''
    # Force garbage collection every time a new tool instance gets opened
        # gc.collect()
        
    # Dataset Builder instances have no special signal to connect
        if toolname == 'DatasetBuilder':
            tool = tools.DatasetBuilder()

    # Model Learner instances have no special signal to connect
        elif toolname == 'ModelLearner':
            tool = tools.ModelLearner()

    # Mineral Classifier instances have sample selector's signals to be connected
        elif toolname == 'MineralClassifier':
            tool = tools.MineralClassifier()
            tool.inmaps_selector.sampleUpdateRequested.connect(
                lambda: tool.inmaps_selector.updateCombox(
                    self.dataManager.getAllGroups()))
    
    # Phase Refiner instances have sample selector's signals to be connected
        elif toolname == 'PhaseRefiner':
            tool = tools.PhaseRefiner()
            tool.minmap_selector.sampleUpdateRequested.connect(
                lambda: tool.minmap_selector.updateCombox(
                    self.dataManager.getAllGroups()))
    
    # Exit function if 'tool' is not a valid string (safety)
        else:
            return

    # Connect tool's 'closed' signal and add it to the list of opened tools
        tool.closed.connect(lambda: self.unregister_tool(tool))
        self.open_tools.append(tool)

    # Show tool as a new tab in the tab widget or as a floating window
        if tabbed:
            self.tabWidget.add_tabified_widget(tool)
        else:
            tool.show()


    def popOutTool(self, index: int) -> None:
        '''
        Detach the tab at index 'index' from the main tab widget and display it
        as a separate floating window. The tab must contain a DraggableTool for
        it to be detached.

        Parameters
        ----------
        index : int
            The index of the tab to be detached.

        '''   
    # If tool is None, close the tab (safety)
        if (tool := self.tabWidget.widget(index)) is None:
            self.tabWidget.closeTab(index)

    # Set draggable tool as floating (parent=None) and close its tab
        elif isinstance(tool, tools.DraggableTool):
            tool.setParent(None)
            self.tabWidget.closeTab(index)
        # Set tool preferred size and move it to the center of the main window
            center = self.geometry().center()
            w, h = tool.floating_size.width(), tool.floating_size.height()
            x, y = center.x() - w // 2, center.y() - h // 2
            QC.QTimer.singleShot(0, lambda: tool.setGeometry(x, y, w, h))
            tool.setVisible(True)

    # If tool is a QWidget (-> DataViewer) do nothing (safety)
        else:
            return


    def unregister_tool(self, tool: tools.DraggableTool) -> None:
        '''
        Remove tool from the list of tracked opened tools (clear reference).

        Parameters
        ----------
        tool : DraggableTool
            Tool to unregister.

        '''
        if tool in self.open_tools:
            self.open_tools.remove(tool)


    def close_all_tools(self) -> None:
        '''
        Close all opened tool, both floating or tabbed.

        '''
        for tool in reversed(self.open_tools):
            # If tool is floating, force its closure without asking confirm
            if tool.isFloating(): 
                tool.force_close()
            # If tool is tabified, destroy it by closing its parent tab
            else:
                tab_index = self.tabWidget.indexOf(tool)
                self.tabWidget.closeTab(tab_index)

        # Clear the list that tracks opened tools
        self.open_tools.clear()


    def importData(self) -> None:
        '''
        Wrapper method to load data in the Data Manager. The type of data to be
        loaded is automatically retrieved from the sender action.

        '''
    # Build a new group
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


    def updateScene(self, item: docks.DataManagerWidgetItem | None) -> None:
        '''
        Main control method for rendering data in the Data Viewer and in the 
        panes. 

        Parameters
        ----------
        item : DataObject or DataSubGroup or DataGroup or None
            Item to be rendered; canonically, it should be a DataObject. If a
            DataSubGroup or a DataGroup is passed, this method attempts to
            re-render the currently displayed data object. If None is passed,
            this method does nothing.

        '''
    # Re-call this method to refresh the displayed map if item is (sub)group
        if isinstance(item, (CW.DataGroup, CW.DataSubGroup)):
            self.updateScene(self.dataViewer._displayed)

    # Rendering operations require item to be a data object
        elif isinstance(item, CW.DataObject):

        # Exit function and clear the scene if item has been deleted (safety)
            try:
                components = item.getComponents()
                i_data = components['data']
                i_path = components['filepath']
                i_name = components['name']
            except RuntimeError:
                return self.clearScene()

        # Exit function and clear the scene if item has no valid parent group
            sample = self.dataManager.getItemParentGroup(item)
            if sample is None:
                return self.clearScene()

        # Check for source file existence
            if not item.filepathValid():
                item.setNotFound(True)

        # Get mask if present
            mode = settings.manager.get('data/mask_merging_rule')
            mask = sample.getCompositeMask('checked', mode=mode)
            if mask is not None:
                mask = mask.mask

        # Actions to be performed if item contains input map data
            if isinstance(i_data, InputMap):
                inmap = i_data.map if mask is None else i_data.get_masked(mask)
                title = f'{sample.name} - {i_name}'

                self.dataViewer._displayed = item
                self.dataViewer.curr_path.setPath(i_path)
                self.dataViewer.canvas.draw_heatmap(inmap, title)
                self.modeViewer.clearAll()
                self.pmapViewer.canvas.clear_canvas()
                self.updateHistogram()
                if self.histViewer.scaler_action.isChecked():
                    self.histViewer.updateScalerExtents()
   
        # Actions to be performed if item contains mineral map data
            elif isinstance(i_data, MineralMap):
                mmap, enc, col = i_data.get_plot_data()
                title = f'{sample.name} - {i_name}'

                if mask is None:
                    pmap = i_data.probmap
                else:
                    _, mmap, pmap = i_data.get_masked(mask)
                
                self.dataViewer._displayed = item
                self.dataViewer.curr_path.setPath(i_path)
                self.dataViewer.canvas.draw_discretemap(mmap, enc, col, title)
                self.modeViewer.update_viewer(item, title)
                self.pmapViewer.canvas.draw_heatmap(pmap, title)
                if self.pmapViewer.toggle_range_action.isChecked():
                    self.pmapViewer.updateViewRange()
                self.histViewer.hideScaler()
                self.histViewer.canvas.clear_canvas()

        # Actions to be performed if item contains mask data
        # We re-call this method using the currently displayed (map) item so
        # that any visual modification generated by (un)checking a mask is
        # automatically rendered even when the currently selected item in the
        # Data Manager is not a map. This is triggered only within same sample.
            elif isinstance(i_data, Mask):
                displ_item = self.dataViewer._displayed
                displ_sample = self.dataManager.getItemParentGroup(displ_item)
                if sample == displ_sample:
                    self.updateScene(displ_item)

        # Actions to be performed if item contains point data
            # elif isinstance(i_data, PointsAnalysis): pass

    # Exit function if item is not group, subgroup or data object (safety)
        else:
            return


    def clearScene(self) -> None:
        '''
        Clear the current view of the Data Viewer and the panes.

        '''
        self.dataViewer._displayed = None
        self.dataViewer.curr_path.clearPath()
        self.dataViewer.canvas.clear_canvas()
        self.modeViewer.clearAll()
        self.pmapViewer.canvas.clear_canvas()
        self.histViewer.hideScaler()
        self.histViewer.canvas.clear_canvas()


    def updateHistogram(self) -> None:
        '''
        Actions to be performed to refresh the Histogram Viewer pane with data
        displayed in the Data Viewer and the ROI traced with the ROI Editor.

        '''
    # Exit function if the canvas does not display an input map
        if not self.dataViewer.canvas.has_heatmap(): 
            return

    # Exit function if data is invalid
        if (data := self.dataViewer.canvas.get_array()) is None: 
            return
        
    # Populate ROI mask using ROI coords if they're valid
        roi_coords, roi_mask = self.roiEditor.current_selection, None
        if roi_coords is not None and self.roiEditor.rect_sel.active:
            r0,r1, c0,c1 = roi_coords
            roi_mask = data[r0:r1, c0:c1]
            if not len(roi_mask):
                roi_mask = None

    # Update histogram canvas
        title = self.dataViewer.canvas.ax.get_title()
        self.histViewer.canvas.update_canvas(data, roi_mask, title)


    def setRgbaChannel(self, channel: str) -> None:
        '''
        Set current item in the Data Manager as channel 'channel' in the RGBA
        Composite Map Viewer pane.

        Parameters
        ----------
        channel : str
            RGBA channel. Must be one of 'R', 'G', 'B', 'A'.

        '''
    # Get the current item in the data manager. If it is valid (= a DataObject
    # holding an Input Map), send its data to the RGBA Composite Maps Viewer
        item = self.dataManager.currentItem()
        if (
            isinstance(item, CW.DataObject) and 
            (inmap := item.getInputMapData()) is not None
        ):
            self.rgbaViewer.setChannel(channel, inmap)

        # Force show the RGBA pane to provide feedback
            self.setPaneVisibility(self.panes[5], True)


    def _onSaveProject(self) -> None:
        '''
        Handle "Save" and "Save As" action triggers.

        '''
        overwrite = self.sender() == self.save_project_action
        self.saveProject(overwrite)


    def saveProject(self, overwrite: bool = True) -> bool:
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
    # Prompt user to deal with unsaved Data Manager objects
        n_obj = len(unsaved_data_objects := self.dataManager.unsavedData())
        for n in range(n_obj):
            data_obj = unsaved_data_objects.pop(0)
            
        # Skip objects with invalid subgroup or group (safety)
            if (
                (subgr := data_obj.subgroup()) is None or
                (group := subgr.group()) is None
            ):
                continue

            name = data_obj.getComponents()['name']
            displ_name = ' / '.join((group.name, subgr.name, name))
            text = (
                f'Found unsaved data in Data Manager:\n{displ_name}\n'
                f'Save it? ({n_obj - n - 1} more found)'
            )
            btns = QW.QMessageBox.StandardButton(
                QW.QMessageBox.StandardButton.Yes
                | QW.QMessageBox.StandardButton.YesToAll
                | QW.QMessageBox.StandardButton.No
                | QW.QMessageBox.StandardButton.NoToAll
                | QW.QMessageBox.StandardButton.Cancel
            )

            def_btn = QW.QMessageBox.StandardButton.Cancel
            msgbox = CW.MsgBox(self, 'Quest', text, btns=btns, def_btn=def_btn)
            
            if msgbox.is_choice('Yes'):
                self.dataManager.saveData(data_obj)
                continue
            elif msgbox.is_choice('YesToAll'):
                for obj in [data_obj] + unsaved_data_objects:
                    self.dataManager.saveData(obj)
            elif msgbox.is_choice('No'):
                continue
            elif msgbox.is_choice('NoToAll'):
                pass
            elif msgbox.is_choice('Cancel'): # also triggers with "X" or "Esc"
                return False
            break

    # Prompt user to deal with unsaved ROI map data
        if self.roiEditor.isRoiMapUnsaved():
            text = 'Found unsaved ROI data in the current ROI map. Save it?'
            btns = QW.QMessageBox.StandardButton(
                QW.QMessageBox.StandardButton.Yes
                | QW.QMessageBox.StandardButton.No 
                | QW.QMessageBox.StandardButton.Cancel
            )
            def_btn = QW.QMessageBox.StandardButton.Cancel
            choice = CW.MsgBox(self, 'Quest', text, btns=btns, def_btn=def_btn)
            if choice.yes():
                self.roiEditor.saveRoiMap()
            elif choice.no():
                pass
            else: # dialog is canceled
                return False

    # Do not save the project if the filepath is invalid
        path = self.project_path
        if not overwrite or path is None:
            ftype = 'X-Min Learn project (*.xmj)'
            path = CW.FileDialog(self, 'save', 'Save Project', ftype).get_path()
            if not path:
                return False

    # Collect project data
        project_info = {
            'Name': cf.path2filename(path),
            'Version': QW.qApp.applicationVersion()
        }
        
        current_view = {
            'Path': self.dataViewer.curr_path.fullpath,
            'Zoom': [
                self.dataViewer.canvas.ax.get_xlim(),
                self.dataViewer.canvas.ax.get_ylim()
            ]
        }
        
        panes_data = {
            p.objectName(): p.widget().getConfig() for p in self.panes
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
        text = 'Save current project?'
        btns = QW.QMessageBox.StandardButton(
            QW.QMessageBox.StandardButton.Yes 
            | QW.QMessageBox.StandardButton.No 
            | QW.QMessageBox.StandardButton.Cancel
        )
        def_btn = QW.QMessageBox.StandardButton.Yes
        choice = CW.MsgBox(self, 'QuestWarn', text, btns=btns, def_btn=def_btn)
    
    # User wants to save -> launch 'saveProject' method and return its output
        if choice.yes(): 
            return self.saveProject()
    
    # User does not want to save -> the project is safe to close
        elif choice.no():
            return True
    
    # User canceled the choice dialog -> the project is not safe to close
        else:
            return False


    def newProject(self) -> None:
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
            pane.widget().resetConfig()
        
    # Kill floating and tabbed tools
        self.close_all_tools()
   
    # Update window attributes and properties
        self.project_path = None
        self.setWindowTitle('New project[*]')
        self.setWindowModified(False)       


    def openProject(self) -> None:
        '''
        Load and open a project from disk.

        '''
    # Deny opening a project if the current project is not safe to close
        if not self.projectSafeToClose():
            return
            
    # Do nothing if project path is invalid (file dialog is canceled)
        ftype = 'X-Min Learn project (*.xmj)'
        path = CW.FileDialog(self, 'open', 'Open Project', ftype).get_path()
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
            CW.MsgBox(self, 'Crit', 'Failed to open project.', str(e))
            return
        
    # Ask for user's choice if the project and the app version do not coincide
        if (v := project_info['Version']) != QW.qApp.applicationVersion():
            text = f'This project is saved in a different version: {v}. Load?'
            choice = CW.MsgBox(self, 'QuestWarn', text)
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
                pane.widget().loadConfig(panes_data[pane_name])
            except Exception as e:
                loading_errors.append((pane_name, e))

    # Load current view object
        for item in self.dataManager.getAllDataObjects():
            if item.getComponents()['filepath'] == current_view['Path']:
                self.dataManager.setCurrentItem(item)
                self.updateScene(item)
                break

    # Close currently opened tools and load project's tools states 
    # TODO...

    # Update window attributes and properties
        self.project_path = path
        self.setWindowTitle(f'{project_info['Name']}[*]')
        self.setWindowModified(False)   

    # Send any catched loading error via warning
        if len(loading_errors):
            CW.MsgBox(self, 'Warn', 'Found corrupted data in the project.',
                      '\n\n'.join((f'{k} -> {e}' for k, e in loading_errors)))


    def _onCloseAction(self):
        '''
        Handle main window closure action from File menu.

        '''
        self.close()


    def closeEvent(self, a0: QG.QCloseEvent | None) -> None:
        '''
        Reimplementation of the closeEvent. It shows a dialog to save the
        project if it has been modified. The current state of the window is
        automatically saved on exit.

        Parameters
        ----------
        a0 : QCloseEvent or None
            The close event.

        '''
        # Exit function when close event is invalid (safety)
        if a0 is None:
            return
        
        if self.projectSafeToClose():
        # Avoid to manually have to close floating tools if main app is closed
            self.close_all_tools()

        # Save the current state of main window's panes and toolbars
            settings.manager.set('system/window_state', self.saveState(version=0))
            a0.accept()
            
        else:
            a0.ignore()



class MainTabWidget(QW.QTabWidget):

    tabToolDetachRequested = QC.pyqtSignal(int) # index of tab
    tabToolCloseRequested = QC.pyqtSignal(int) # index of tab

    def __init__(self, parent: QW.QWidget) -> None:
        '''
        Central widget of the X-Min Learn window. It is a reimplementation of a
        QTabWidget, customized to accept drag and drop events of its own tabs. 
        Such tabs are the major X-Min Learn windows, that can be attached to 
        this widget or detached and visualized as stand-alone windows. See the 
        "DraggableTool" class in "tools.py" module for more details.

        Parameters
        ----------
        parent : QWidget
            The GUI parent of this widget.

        Raises
        ------
        RuntimeError
            Raised if the main tab bar initialization fails.

        '''
        super().__init__(parent)

    # Set stylesheet
        self.setStyleSheet(style.SS_MAINTABWIDGET)

    # Set properties
        self.setAcceptDrops(True)
        self.setMovable(True)
        self.setTabsClosable(True)

    # Create a tab bar with a custom context menu
        if (tabbar := self.tabBar()) is None:
            raise RuntimeError("Failed to initialize the main tab bar.")
        self.tabbar = tabbar
        self.tabbar.setContextMenuPolicy(QC.Qt.ContextMenuPolicy.CustomContextMenu)

    # Connect signals to slots
        self._connect_slots()


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Tab bar signals
        self.tabCloseRequested.connect(self.tabToolCloseRequested)
        self.tabBarDoubleClicked.connect(self.tabToolDetachRequested)
        self.tabbar.customContextMenuRequested.connect(self.showContextMenu)


    def showContextMenu(self, point: QC.QPoint) -> None:
        '''
        Show a context menu in the tab bar with custom actions.

        Parameters
        ----------
        point : QPoint
            The position of the context menu event that the widget receives.

        ''' 
    # Do nothing if the point is not on a tab
        tab_idx = self.tabbar.tabAt(point)
        if tab_idx == -1:
            return
    
    # Do nothing if the widget at 'tab_idx' is not a DraggableTool
        tool = self.widget(tab_idx)
        if not isinstance(tool, tools.DraggableTool):
            return

    # Define a menu (Menu)
        menu = QW.QMenu()
        menu.setStyleSheet(style.SS_MENU)

    # Detach tool (Action)
        menu.addAction(
            style.getIcon('UNDOCK'), 
            'Detach', 
            lambda: self.tabToolDetachRequested.emit(tab_idx)
        )
        
    # Close tool (Action)
        menu.addAction(
            style.getIcon('CLOSE'),
            'Close',
            lambda: self.tabToolCloseRequested.emit(tab_idx)
        )

    # Show the menu in the same spot where the user triggered the event
        menu.exec(self.tabbar.mapToGlobal(point)) 

    # Force tab bar to update hover state after menu closes (fix visual bug)
        global_pos = QG.QCursor.pos()
        pos = self.tabbar.mapFromGlobal(global_pos)
        event = QG.QMouseEvent(
            QC.QEvent.Type.MouseMove,
            pos,
            global_pos,
            QC.Qt.MouseButton.NoButton,
            QC.Qt.MouseButton.NoButton,
            QC.Qt.KeyboardModifier.NoModifier
        )
        QW.qApp.sendEvent(self.tabbar, event)


    def add_tabified_widget(self, widget: tools.DraggableTool | QW.QWidget) -> None:
        '''
        Add a tabified widget. A GroupScrollArea is set as the container for 
        the provided 'widget'. This helps in the visualization of complex 
        widgets across different sized screens.

        Parameters
        ----------
        widget : DraggableTool or QWidget
            The widget to be added in a new tab.

        '''
    # Insert widget in a scroll area, add it and set it as the current tab
        icon, title = widget.windowIcon(), widget.windowTitle()
        scroll = CW.GroupScrollArea(widget)
        scroll.setAttribute(QC.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.addTab(scroll, icon, title)
        self.setCurrentWidget(scroll)

    # Send 'tabified' signal if a draggable tool is added (currently useless)
        if isinstance(widget, tools.DraggableTool):
            widget.tabified.emit()


    def widget(self, index: int) -> tools.DraggableTool | QW.QWidget | None:
        '''
        Reimplementation of the default 'widget' method. Returns the widget
        held by the tab at 'index' by extracting it from the GroupScrollArea
        that wraps it (see 'add_tabified_widget' method for more details).

        Parameters
        ----------
        index : int
            The position of the widget in the tab bar.

        Returns
        -------
        wid : DraggableTool or QWidget or None
            The widget in the tab page or None if 'index' is out of range.

        '''
        # Return None if index is out of range
        if (scroll_area := self.scrollArea(index)) is None:
            return None
        
        # Return None if wrapped object is not a valid type (safety)
        wid = scroll_area.wrappedObject()
        if not isinstance(wid, (tools.DraggableTool, QW.QWidget)):
            return None
        
        return wid


    def scrollArea(self, index: int) -> CW.GroupScrollArea:
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
        scroll_area = super().widget(index)
        return scroll_area
          

    def closeTab(self, index: int) -> None:
        '''
        Close the tab at index 'index'.

        Parameters
        ----------
        index : int
            The tab index in the tab bar.

        '''
        # 'removeTab' just removes the tab from the tab bar but does not delete
        # the actual widget page. Therefore, we manually do it by closing the
        # scroll area (page of the tab). This also destroys the wrapped tool.
        scroll_area = self.scrollArea(index)
        self.removeTab(index)
        scroll_area.close()


    def indexOf(self, widget: tools.DraggableTool | QW.QWidget) -> int:
        '''
        Reimplementation of the default 'indexOf' method. Returns the index of
        the tab that contains 'widget'. If the widget is not found, the index
        will be -1. Please note that this method expects 'widget' to be the
        actual displayed widget and not the scroll area that wraps it (see also
        'indexOfScrollArea' method). 

        Parameters
        ----------
        widget : DraggableTool or QWidget
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
        area is not found, the index will be -1. This method should not be used
        to get the index of the actual displayed widget, but rather the index
        of the scroll area that wraps it (see also 'indexOf' method).

        Parameters
        ----------
        scroll_area : CW.GroupScrollArea
            Scroll area.

        Returns
        -------
        idx : int
            Index of the scroll area.

        '''
        if (idx := super().indexOf(scroll_area)) is None:
            return -1
        return idx
    

    def getAllWidgets(self) -> list[tools.DraggableTool | QW.QWidget]:
        '''
        Return a list of all the widgets (not the scroll areas that wrap them)
        included in the tab widget.

        Returns
        -------
        list[DraggableTool or QWidget]
            List of widgets.

        '''
        return [self.widget(idx) for idx in range(self.count())]


    def dragEnterEvent(self, a0: QG.QDragEnterEvent | None) -> None:
        '''
        Reimplementation of the default 'dragEnterEvent' method. Customized to
        accept only DraggableTool instances.

        Parameters
        ----------
        a0 : QDragEntervent or None
            The drag enter event triggered by the user's drag action.

        '''
    # Do nothing if event is null (safety)
        if a0 is None:
            return
        
    # Accept only events from a valid source
        if isinstance(a0.source(), tools.DraggableTool):
            a0.accept()


    def dropEvent(self, a0: QG.QDropEvent | None) -> None:
        '''
        Reimplementation of the default 'dropEvent' method. Customized to
        accept only DraggableTool instances and add them as new tabs. 

        Parameters
        ----------
        a0 : QDropEvent or None
            The drop event triggered by the user's drag & drop action.

        '''
    # Do nothing if event is null (safety)
        if a0 is None:
            return

    # Accept only events from a valid source
        if isinstance(widget := a0.source(), tools.DraggableTool):
            a0.setDropAction(QC.Qt.DropAction.MoveAction)
            a0.accept()
        # Suppress updates temporarily for better performances
            self.setUpdatesEnabled(False)
            self.add_tabified_widget(widget)
            self.setUpdatesEnabled(True)