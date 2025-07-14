# -*- coding: utf-8 -*-
"""
Created on Tue May  14 15:03:45 2024

@author: albdag
"""
import os

from PyQt5.QtCore import pyqtSignal, QLocale, QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
import PyQt5.QtWidgets as QW

import numpy as np

from _base import *
import convenient_functions as cf
import custom_widgets as CW
import dataset_tools as dtools
import image_analysis_tools as iatools
import plots
import preferences as pref
import style
import threads


class AutoRoiDetector(QW.QDialog):

    requestRoiMap = pyqtSignal()
    drawingRequested = pyqtSignal(RoiMap)
    npv_encoder = {'Pure': np.max, 'Simple': np.median, 'Smooth': np.mean}

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog designed for the ROI Editor pane, that enables automatic ROI
        identification from input maps of a given sample. The identification is
        based on the analysis of a cumulative Neighborhood Pixel Variance map,
        calculated on input data.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('ROI detector')
        self.setWindowIcon(style.getIcon('ROI_SEARCH'))
        self.setAttribute(Qt.WA_QuitOnClose, False)

    # Set main attributes
        self._current_roimap = None
        self._auto_roimap = None
        self._patches = []
        self._active_thread = None
        self.npv_thread = threads.NpvThread()
        self.roi_detect_thread = threads.RoiDetectionThread()

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.adjustSize()
        self._connect_slots()
        

    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Input maps selector
        self.maps_selector = CW.SampleMapsSelector('inmaps')

    # Number of ROIs selector (SpinBox)
        self.nroi_spbox = CW.StyledSpinBox(min_value=1, max_value=100)
        self.nroi_spbox.setValue(15)

    # ROI size selector (SpinBox)
        self.size_spbox = CW.StyledSpinBox(min_value=3, max_value=99, step=2)
        self.size_spbox.setValue(9)
        self.size_spbox.setToolTip('Define a NxN ROI. N must be an odd number')

    # ROI distance selector
        self.distance_spbox = CW.StyledSpinBox()
        self.distance_spbox.setValue(10)
        self.distance_spbox.setToolTip('Minimum distance between ROI')

    # NPV function types (QButtonLayout)
        self.npv_btns = CW.RadioBtnLayout(('Pure', 'Simple', 'Smooth'))
        tooltips = (
            'Use max function to penalize noisy pixels',
            'Use median function to ignore outliers',
            'Use mean function to smoothen noise'
        )
        for n, t in enumerate(tooltips):
            self.npv_btns.button(n).setToolTip(t)

    # NPV canvas (ImageCanvas)
        self.canvas = plots.ImageCanvas()
    
    # NPV canvas Navigation Toolbar 
        self.navtbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # Search ROIs button
        self.search_btn = CW.StyledButton(
            style.getIcon('ROI_SEARCH'), 'Search ROI')
        self.search_btn.setEnabled(False)

    # Descriptive Progress bar 
        self.progbar = CW.DescriptiveProgressBar()

    # OK button
        self.ok_btn = CW.StyledButton(text='Ok')

    # Cancel button
        self.cancel_btn = CW.StyledButton(text='Cancel')

    # Set layout
        params_form = QW.QFormLayout()
        params_form.addRow('Number of ROI', self.nroi_spbox)
        params_form.addRow('ROI size', self.size_spbox)
        params_form.addRow('ROI distance', self.distance_spbox)

        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(15)
        left_vbox.addWidget(self.maps_selector)
        left_vbox.addLayout(params_form)
        left_vbox.addWidget(CW.GroupArea(self.npv_btns, 'NPV type'))
        left_vbox.addWidget(self.search_btn)
        left_scroll = CW.GroupScrollArea(left_vbox)

        right_grid = QW.QGridLayout()
        right_grid.addWidget(self.navtbar, 0, 0, 1, -1)
        right_grid.addWidget(self.canvas, 1, 0, 1, -1)
        right_grid.addWidget(self.progbar, 2, 0, 1, -1)
        right_grid.addWidget(self.ok_btn, 3, 0)
        right_grid.addWidget(self.cancel_btn, 3, 1)
        right_grid.setRowStretch(1, 1)

        main_layout = QW.QHBoxLayout()
        main_layout.addWidget(left_scroll)
        main_layout.addSpacing(20)
        main_layout.addLayout(right_grid)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Connect signals to slots.

        '''
    # Connect the threads signals
        self.npv_thread.taskInitialized.connect(self.progbar.step)
        self.npv_thread.workFinished.connect(self._parse_npv)
        self.roi_detect_thread.taskInitialized.connect(self.progbar.step)
        self.roi_detect_thread.workFinished.connect(self._parse_roi_detection)

    # Enable search btn and update parameters when input maps list is populated
        self.maps_selector.mapsDataChanged.connect(
            lambda: self.search_btn.setEnabled(True))
        self.maps_selector.mapsDataChanged.connect(self.update_params_range)

    # Fix even size values
        self.size_spbox.valueChanged.connect(self.fix_even_size)

    # Launch the NPV computation when search button is clicked
        self.search_btn.clicked.connect(self.launch_npv_computation)

    # Accept the result when OK button is clicked
        self.ok_btn.clicked.connect(self.accept)

    # Reject the result when Cancel button is clicked
        self.cancel_btn.clicked.connect(self.reject)


    def fix_even_size(self, size: int) -> None:
        '''
        Adjust kernel size so that it is always an odd number.

        Parameters
        ----------
        size : int
            Kernel size to check.

        '''
        if not size % 2:
            self.size_spbox.setValue(size - 1)


    def set_current_roimap(self, roimap: RoiMap) -> None:
        '''
        Set the current roimap. This method is especially useful when called by
        the parent widget (i.e., the ROI Editor pane).

        Parameters
        ----------
        roimap : RoiMap
            New current ROI map.

        '''
        self._current_roimap = roimap


    def _clear_roi_patches(self) -> None:
        '''
        Remove any previous ROI patch displayed in the canvas.

        '''
        for idx in reversed(range(len(self._patches))):
            patch = self._patches.pop(idx)
            patch.remove()


    def update_params_range(self) -> None:
        '''
        Change maximum selectable ROI size and ROI distance based on the shape
        of the input maps.

        '''
    # Get the shape of the first map in the list. If maps have different shapes
    # the dialog will later abort the computation anyway.
        shp = self.maps_selector.maps_list.topLevelItem(0).get('data').shape

    # Set the maximum selectable ROI distance and ROI size based on shape
        max_dist = min(shp) // 10
        max_size = max_dist if not max_dist % 2 else max_dist - 1 # must be ODD
        self.distance_spbox.setMaximum(max_dist)
        self.size_spbox.setMaximum(max_size)


    def launch_npv_computation(self) -> None:
        '''
        Launch the thread that computes the cumulative Neighborhood Pixel 
        Variance (NPV) map.

        '''
    # Exit function if a thread is currentlty active
        if self._active_thread is not None:
            return CW.MsgBox(self, 'Crit', 'A detection process is active.')

    # Disable OK button
        self.ok_btn.setEnabled(False)

    # Get user's parameters
        size = self.size_spbox.value()
        npv_func = self.npv_encoder[self.npv_btns.getChecked().text()]
        
    # Get input map data
        checked_maps = self.maps_selector.getChecked()
        if not len(checked_maps):
            return CW.MsgBox(self, 'Crit', 'Select at least one map.')
        inmaps, names = zip(*[i.get('data', 'name') for i in checked_maps])
        input_stack = InputMapStack(inmaps)

    # Check that input maps share the same shape
        if not input_stack.maps_fit():
            return CW.MsgBox(self, 'C', 'Selected maps have different shapes.')
        
    # Get current ROI map and check its shape
        self.requestRoiMap.emit()
        if self._current_roimap is None:
            self._current_roimap = RoiMap.from_shape(input_stack.maps_shape)
        elif self._current_roimap.shape != input_stack.maps_shape:
            t = 'Loaded ROI map and selected input maps have different shapes.'
            return CW.MsgBox(self, 'Crit', t)

    # Launch the NPV thread
        self.progbar.setMaximum(len(inmaps))
        self._active_thread = self.npv_thread
        self.npv_thread.set_params(input_stack.arrays, names, size, npv_func)
        self.npv_thread.start()


    def _parse_npv(self, thread_result: tuple, success: bool) -> None: 
        '''
        Parse the result of the NPV computation thread. If the computation was
        successful, this method automatically launches the ROI identification
        thread.

        Parameters
        ----------
        thread_result : tuple
            Result of the NPV thread.
        success : bool
            Whether the NPV thread succeeded or not.

        '''
        if success:
        # Display the npv map in canvas and clear old ROI patches and legend
            npv_map, = thread_result
            self._clear_roi_patches()
            self.canvas.figure.legends.clear()
            title = 'Cumulative Neighborhood Pixel Variance (NPV) map'
            self.canvas.draw_heatmap(npv_map, title)

        # Execute ROI detection
            self.launch_roi_detection(npv_map)

        else:
            e, = thread_result
            CW.MsgBox(self, 'Crit', 'NPV calculation failed.', str(e))
            self._end_threaded_session()

    
    def launch_roi_detection(self, npv_map: np.ndarray) -> None:
        '''
        Launch the thread that automatically identifies the best ROIs. This
        method is meant to be called automatically after the NPV computation
        thread successfully returned a cumulative NPV map (see '_parse_npv'
        method for details).

        Parameters
        ----------
        npv_map : np.ndarray
            The cumulative Neighborhood Pixel Variance map.

        '''
    # Get other parameters
        n_roi = self.nroi_spbox.value()
        size = self.size_spbox.value()
        dist = self.distance_spbox.value()

    # Launch the ROI detection thread
        self.progbar.reset()
        self.progbar.setMaximum(n_roi)
        self._active_thread = self.roi_detect_thread
        self.roi_detect_thread.set_params(n_roi, size, dist, npv_map, 
                                            self._current_roimap)
        self.roi_detect_thread.start()


    def _parse_roi_detection(self, thread_result: tuple, success: bool) -> None:
        '''
        Parse the result of the ROI detection thread. If the identification was
        successfull, this method displays the identified ROIs in canvas and
        saves the new (automatic) ROI map.

        Parameters
        ----------
        thread_result : tuple
            The ROI detection thread result.
        success : bool
            Whether the ROI detection thread succeeded or not.

        '''
        if success:
        # Display existent ROIs in black
            for _, bbox in self._current_roimap.roilist:
                patch = plots.RoiPatch(bbox, 'black', filled=False)
                patch.set_label('Existent ROI')
                self.canvas.ax.add_patch(patch)
                self._patches.append(patch)

        # Display new auto detected ROIs in red
            auto_roimap, = thread_result
            for _, bbox in auto_roimap.roilist:
                patch = plots.RoiPatch(bbox, 'red', filled=False)
                patch.set_label('Auto detected ROI')
                self.canvas.ax.add_patch(patch)
                self._patches.append(patch)

        # Show legend and render canvas
            hand, lbls = self.canvas.ax.get_legend_handles_labels()
            entries = dict(zip(lbls, hand))
            self.canvas.figure.legend(
                entries.values(), entries.keys(), loc='lower left')
            self.canvas.draw()

        # Save the auto detected roimap
            self._auto_roimap = auto_roimap

        else:
            e, = thread_result
            CW.MsgBox(self, 'Crit', 'ROI detection failed.', str(e))
            
    # End the threaded session anyway
        self._end_threaded_session()    


    def interrupt_thread(self) -> bool:
        '''
        Stop an external thread.

        Returns
        -------
        bool
            Whether the stop event is confirmed by user or not.

        '''
        choice = CW.MsgBox(self, 'Quest', 'Interrupt automatic ROI detection?')
        if choice.yes():
            self._active_thread.requestInterruption()
            self._end_threaded_session()
            return True
        else:
            return False


    def _end_threaded_session(self) -> None:
        '''
        Internally and visually exit from an external thread session.

        '''
        self._active_thread = None
        self.progbar.reset()
        self.ok_btn.setEnabled(True)


    def accept(self) -> None:
        '''
        Reimplementation of the dialog's 'accept' method. A custom signal is 
        emitted to draw the newly identified ROIs on the input maps.

        '''
        if self._auto_roimap is not None:
            self.drawingRequested.emit(self._auto_roimap)
        super().accept()


    def reject(self) -> None:
        '''
        Reimplementation of the dialog's 'reject' method. If thread is active,
        it requests user to confirm its interruption.

        '''           
        if self._active_thread is not None and not self.interrupt_thread():
            return
        super().reject()
        

    def show(self) -> None:
        '''
        Reimplementation of the dialog's 'show' method. It resets the dialog's
        GUI if it was hidden (closed).

        '''
        if self.isHidden():
            self._auto_roimap = None
            self.set_current_roimap(None)
            self.maps_selector.clear()
            self.search_btn.setEnabled(False)
            self.canvas.clear_canvas()
            self._clear_roi_patches()
            self.canvas.figure.legends.clear()
            super().show()
        else:
            self.activateWindow()



class MergeDatasets(QW.QDialog):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog to create a merged copy of two or more existing ground truth
        datasets.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('Merge Datasets')
        self.setWindowIcon(style.getIcon('GEAR'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # Set main attribute
        self.merged_dataset = None

    # Initialize GUI and connect its signals with slots 
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Decimal character selector for imported datasets
        self.in_csv_decimal = CW.DecimalPointSelector()

    # Import datasets (Styled Button)
        self.import_btn = CW.StyledButton(style.getIcon('IMPORT'), 'Import')

    # Remove datasets (Styled Button)
        self.remove_btn = CW.StyledButton(style.getIcon('REMOVE'), 'Remove')

    # Input datasets paths list (Styled List Widget)
        self.in_path_list = CW.StyledListWidget()
        
    # Decimal point character selector for merged dataset
        self.out_csv_decimal = CW.DecimalPointSelector()

    # Separator character selector for merged dataset
        self.out_csv_separator = CW.SeparatorSymbolSelector()

    # Save merged dataset (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'Save')
        self.save_btn.setEnabled(False)

    # Input datasets preview area (Document Browser)
        self.input_info = CW.DocumentBrowser(readonly=True)

    # Number of dataset preview rows (Styled Spinbox)
        self.nrows_spbox = CW.StyledSpinBox(10, 1000, 10)

    # Merge datasets (Styled Button)
        self.merge_btn = CW.StyledButton(text='Merge', bg=style.OK_GREEN)

    # Merged dataset preview area (Document Browser)
        self.merged_info = CW.DocumentBrowser(readonly=True)

    # Progress bar (Progress Bar)
        self.progbar = QW.QProgressBar()

    # Adjust Layout
        input_form = QW.QFormLayout()
        input_form.addRow('CSV decimal point', self.in_csv_decimal)
        input_form.addRow(self.import_btn)
        input_form.addRow(self.in_path_list)
        input_form.addRow(self.remove_btn)
        input_group = CW.GroupArea(input_form, 'Import datasets')

        output_form = QW.QFormLayout()
        output_form.addRow('CSV decimal point', self.out_csv_decimal)
        output_form.addRow('CSV separator', self.out_csv_separator)
        output_form.addRow(self.save_btn)
        output_group = CW.GroupArea(output_form, 'Export dataset')

        left_vbox = QW.QVBoxLayout()
        left_vbox.addWidget(input_group)
        left_vbox.addWidget(output_group)
        left_vbox.addStretch(1)
        left_scroll = CW.GroupScrollArea(left_vbox, frame=False)

        right_grid = QW.QGridLayout()
        right_grid.setRowMinimumHeight(3, 20)
        right_grid.addWidget(self.input_info, 0, 0, 1, -1)
        right_grid.addWidget(QW.QLabel('Previewed rows'), 1, 0)
        right_grid.addWidget(self.nrows_spbox, 1, 1)
        right_grid.addWidget(self.merge_btn, 2, 0, 1, -1)
        right_grid.addWidget(self.merged_info, 4, 0, 1, -1)
        right_scroll = CW.GroupScrollArea(right_grid, frame=False)

        splitter = CW.SplitterGroup((left_scroll, right_scroll), (0, 1))
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.progbar)
        self.setLayout(main_layout)

    
    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.import_btn.clicked.connect(self.importDatasets)
        self.in_path_list.itemClicked.connect(self.showInputDatasetPreview)
        self.remove_btn.clicked.connect(self.removeDatasets)
        self.merge_btn.clicked.connect(self.mergeDatasets)
        self.save_btn.clicked.connect(self.save)


    def importDatasets(self) -> None:
        '''
        Import ground truth datasets from files. In order to be more memory
        friendly, this method just saves the datasets paths without actually
        loading the entire dataframe.
         
        '''
    # Do nothing if paths are invalid or the file dialog is canceled
        ftype = 'CSV (*.csv)'
        paths = CW.FileDialog(self, 'open', 'Load Datasets', ftype, True).get()
        if not paths:
            return
     
    # Add datasets paths to list but skip those that had already been added
        for p in paths:
            if len(self.in_path_list.findItems(p, Qt.MatchExactly)): 
                continue
            else:
                self.in_path_list.addItem(p)


    def removeDatasets(self) -> None:
        '''
        Remove selected datasets.

        '''
        if len(self.in_path_list.selectedItems()):
            self.in_path_list.removeSelected()
            self.input_info.clear()


    def showInputDatasetPreview(self, item: QW.QListWidgetItem) -> None:
        '''
        Quickly read the input dataset held by 'item' and show a preview of its 
        first rows. This is useful to check all the dataset's columns names.

        Parameters
        ----------
        item : QW.QListWidgetItem
            List item from 'self.in_path_list' holding the dataset path.

        '''
        if item is None:
            return
        
    # Clear info area
        self.input_info.clear()

    # Get dataset preview
        dec = self.in_csv_decimal.currentText()
        nrows = self.nrows_spbox.value()
        preview = dtools.dataframe_preview(item.text(), dec, n_rows=nrows)
    
    # Show preview in the info area
        text = f'DATAFRAME PREVIEW\n(First {nrows} rows)\n\n{repr(preview)}'
        self.input_info.setText(text)


    def showOutputDatasetPreview(self) -> None:
        '''
        Show a preview of the output merged dataset.

        '''
        dataframe = self.merged_dataset.dataframe
        text = f'MERGED DATAFRAME PREVIEW\n\n{repr(dataframe)}'
        self.merged_info.setText(text)
        

    def mergeDatasets(self) -> None:
        '''
        Merge the loaded datasets. This method sends an error message if any of
        the datasets has different unique columns.

        '''
    # Check for at least two imported datasets
        if self.in_path_list.count() < 2:
            return CW.MsgBox(self, 'Crit', 'Import at least two datasets.')
        
    # Check that all paths do still exist
        paths = [item.text() for item in self.in_path_list.getItems()]
        if not all([os.path.exists(p) for p in paths]):
            err_msg = 'Some datasets have been deleted, moved or renamed.'
            return CW.MsgBox(self, 'Crit', err_msg)
        
    # Merge datasets one at the time to build the final merged dataset. The 
    # first path in the list is the starting dataset ('paths.pop(0)')
        dec = self.in_csv_decimal.currentText()
        self.progbar.setRange(0, len(paths))
        self.progbar.setValue(1)
        merged = dtools.GroundTruthDataset.load(paths.pop(0), dec, chunks=True)
        for n, p in enumerate(paths, start=2):
            try:
                self.progbar.setValue(n)
                merged.merge(dtools.GroundTruthDataset.load(p, dec, chunks=True))
            except ValueError:
                self.progbar.reset()
                return CW.MsgBox(self, 'Crit', 'Datasets columns do not fit.')

        self.progbar.reset()

    # Update attribute and save button state. Show dataset preview.
        self.merged_dataset = merged
        self.save_btn.setEnabled(True)
        self.showOutputDatasetPreview()

        
    def save(self) -> None:
        '''
        Save merged datasets to file.

        '''
    # Exit function if outpath is invalid or file dialog is canceled
        ftype = 'CSV (*.csv)'
        outpath = CW.FileDialog(self, 'save', 'Save Dataset', ftype).get()
        if not outpath:
            return

    # Save dataset
        sep = self.out_csv_separator.currentText()
        dec = self.out_csv_decimal.currentText()
        try:
            self.merged_dataset.save(outpath, sep, dec)
            CW.MsgBox(self, 'Info', 'Succesfully saved merged dataset.')
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save merged dataset.', str(e))



class SubSampleDataset(QW.QDialog):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog to create a sub-sampled copy of an existent ground truth
        dataset, allowing the selection of mineral classes to include.

        Parameters
        ----------
        parent : qObject or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('Sub-sample Dataset')
        self.setWindowIcon(style.getIcon('GEAR'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # Define main attributes
        self.reader = dtools.CsvChunkReader(QLocale().decimalPoint())
        self._dataset = None

    # Initialize GUI and connect its signals with slots 
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Import Dataset (Styled Button)
        self.import_btn = CW.StyledButton(
            style.getIcon('IMPORT'), 'Import dataset')

    # Imported dataset path (Path Label)
        self.in_dataset_path = CW.PathLabel(
            full_display=False, placeholder='No dataset loaded')

    # Decimal point character selector for imported dataset
        self.in_csv_decimal = CW.DecimalPointSelector()

    # Decimal point character selector for sub-sampled dataset
        self.out_csv_decimal = CW.DecimalPointSelector()

    # Separator character selector for sub-sampled dataset
        self.out_csv_separator = CW.SeparatorSymbolSelector()

    # Input dataset preview area (Document Browser)
        self.dataset_info = CW.DocumentBrowser(readonly=True)

    # Class selector (Twin List Widgets)
        self.original_classes = CW.StyledListWidget()
        self.subsampled_classes = CW.StyledListWidget()
        for wid in (self.original_classes, self.subsampled_classes):
            wid.setAcceptDrops(True)
            wid.setDragEnabled(True)
            wid.setDefaultDropAction(Qt.MoveAction)

    # Class counter (Label)
        self.counter_lbl = QW.QLabel()

    # Save (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'Save')
        self.save_btn.setEnabled(False)

    # Progress bar (ProgressBar)
        self.progbar = QW.QProgressBar()

    # Adjust layout
        input_form = QW.QFormLayout()
        input_form.addRow('CSV decimal point', self.in_csv_decimal)
        input_form.addRow(self.import_btn)
        input_form.addRow(self.in_dataset_path)
        input_group = CW.GroupArea(input_form, 'Import dataset')

        output_form = QW.QFormLayout()
        output_form.addRow('CSV decimal point', self.out_csv_decimal)
        output_form.addRow('CSV separator', self.out_csv_separator)
        output_form.addRow(self.save_btn)
        output_group = CW.GroupArea(output_form, 'Export dataset')

        left_vbox = QW.QVBoxLayout()
        left_vbox.addWidget(input_group)
        left_vbox.addWidget(output_group)
        left_vbox.addStretch(1)
        left_scroll = CW.GroupScrollArea(left_vbox, frame=False)
        
        right_grid = QW.QGridLayout()
        right_grid.setRowMinimumHeight(1, 20)
        hint = 'Drag & drop to include classes'
        right_grid.addWidget(self.dataset_info, 0, 0, 1, -1)
        right_grid.addWidget(QW.QLabel(hint), 2, 0, 1, -1, Qt.AlignCenter)
        right_grid.addWidget(QW.QLabel('ORIGINAL CLASSES'), 3, 0, Qt.AlignCenter)
        right_grid.addWidget(QW.QLabel('SUB-SAMPLED CLASSES'), 3, 1, Qt.AlignCenter)
        right_grid.addWidget(self.original_classes, 4, 0)
        right_grid.addWidget(self.subsampled_classes, 4, 1)
        right_grid.addWidget(self.counter_lbl, 5, 0, 1, -1)
        right_scroll = CW.GroupScrollArea(right_grid, frame=False)

        splitter = CW.SplitterGroup((left_scroll, right_scroll), (0, 1))
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.progbar)
        self.setLayout(main_layout)

    
    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Chunk reader related signals
        self.in_csv_decimal.currentTextChanged.connect(self.reader.set_decimal)
        self.reader.thread.iterCompleted.connect(self.progbar.setValue)
        self.reader.thread.taskFinished.connect(self._parseReaderResult)

    # Count class amount when clicked
        self.original_classes.itemClicked.connect(self.countClass)
        self.subsampled_classes.itemClicked.connect(self.countClass)

    # Import and save buttons
        self.import_btn.clicked.connect(self.importDataset)
        self.save_btn.clicked.connect(self.save)


    def resetDialog(self) -> None:
        '''
        Reset some GUI widgets to their initial state.

        '''
        self.original_classes.clear()
        self.subsampled_classes.clear()
        self.counter_lbl.clear()
        self.dataset_info.clear()
        self.save_btn.setEnabled(False)


    def importDataset(self) -> None:
        '''
        Import an existent ground truth dataset. This method launches the CSV
        chunk reader thread.

        '''
    # Do nothing if path is invalid or file dialog is canceled
        ftype = 'CSV (*.csv)'
        path = CW.FileDialog(self, 'open', 'Load Dataset', ftype).get()
        if not path:
            return

    # Set current path
        self.in_dataset_path.setPath(path)

    # Load dataset chunks (threaded)
        self.progbar.setRange(0, self.reader.chunks_number(path))
        self.reader.read_threaded(path)


    def _parseReaderResult(self, result: tuple, success: bool) -> None:
        '''
        Parse the result of the chunk reader thread.

        Parameters
        ----------
        result : tuple
            Result of chunk reader thread.
        success : bool
            Whether the chunk reader task finished succesfully or not.

        '''
    # Reset progress bar
        self.progbar.reset()

        if success:
            self.progbar.setRange(0, 3)

        # Compile dataset
            dataframe = self.reader.combine_chunks(result)
            self._dataset = dtools.GroundTruthDataset(dataframe)
            self.progbar.setValue(1)

        # Update GUI
            self.resetDialog()
            self.updateDatasetInfo()
            self.progbar.setValue(2)

        # Split dataset features from targets and update widgets
            try:
                self._dataset.split_features_targets(split_idx=-1)
                self.original_classes.addItems(self._dataset.column_unique(-1))
                self.save_btn.setEnabled(True)
            except Exception as e:
                self._dataset = None
                self.in_dataset_path.clearPath()
                CW.MsgBox(self, 'Crit', 'Invalid dataset.', str(e))

            finally:
                self.progbar.reset()
        
        else:
            self.in_dataset_path.clearPath()
            CW.MsgBox(self, 'Crit', 'Dataset loading failed.', str(result[0]))
            

    def updateDatasetInfo(self) -> None:
        '''
        Populate the dataset info widget with a preview of the imported ground
        truth dataset.

        '''
        text = f'DATAFRAME PREVIEW\n\n{repr(self._dataset.dataframe)}'
        self.dataset_info.setText(text)


    def countClass(self, item: QW.QListWidgetItem) -> None:
        '''
        Show the amount of instances of the selected class in the dataset.

        Parameters
        ----------
        item : QW.QListWidgetItem
            Selected class item.

        '''
        if self._dataset is None: # safety
            return
        class_name = item.text()
        count = np.count_nonzero(self._dataset.targets == class_name)
        self.counter_lbl.setText(f'{class_name} = {count}')


    def save(self) -> None:
        '''
        Save sub-sampled version of the loaded ground truth dataset.

        '''
    # Deny sub-sampling if no classes are selected
        count = self.subsampled_classes.count()
        if count == 0:
            return CW.MsgBox(self, 'Crit', 'Include at least one class.')
        
    # Exit function if outpath is invalid or file dialog is canceled
        ftype = 'CSV (*.csv)'
        outpath = CW.FileDialog(self, 'save', 'Save Dataset', ftype).get()
        if not outpath:
            return
    
    # Get selected class labels
        self.progbar.setRange(0, 3)
        labels = [self.subsampled_classes.item(i).text() for i in range(count)]
        self.progbar.setValue(1)

    # Sub sample dataset
        subsampled = self._dataset.sub_sample(-1, labels)
        self.progbar.setValue(2)

    # Save dataset
        separator_char = self.out_csv_separator.currentText()
        decimal_char = self.out_csv_decimal.currentText()
        try:
            subsampled.save(outpath, separator_char, decimal_char)
            CW.MsgBox(self, 'Info', 'Dataset successfully saved.')

        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save dataset.', str(e))

        finally:
            self.progbar.reset()
        


class ImageToInputMap(QW.QDialog):
    '''
    A dialog to convert images to InputMaps and save them. Multi-channel images
    can be optionally split into 1-channel images and converted as well.
    '''

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog to convert images to InputMaps. Multi-channel images can be
        optionally split into 1-channel images and converted as well.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('Image To Input Map')
        self.setWindowIcon(style.getIcon('GEAR'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # Set main attribute
        self.preview_size = QSize(64, 64)

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Import images (Styled Button)
        self.import_btn = CW.StyledButton(style.getIcon('IMPORT'), 'Import')
        
    # Remove images (Styled Button)
        self.remove_btn = CW.StyledButton(style.getIcon('REMOVE'), 'Remove')
        
    # Images list (Styled List Widget)
        self.img_list = CW.StyledListWidget()
        self.img_list.setIconSize(self.preview_size)

    # Split multichannel images (Check Box)
        self.split_cbox = QW.QCheckBox('Split multi-channel images')
        self.split_cbox.setChecked(True)

    # Output map extension (Styled Combo Box)
        self.map_ext_combox = CW.StyledComboBox()
        self.map_ext_combox.addItems(['.gz', '.txt'])

    # Convert (Styled Button)
        self.convert_btn = CW.StyledButton(text='Convert', bg=style.OK_GREEN)
        
    # Progress bar (Progress Bar)
        self.progbar = QW.QProgressBar()

    # Adjust layout
        left_form = QW.QFormLayout()
        left_form.setVerticalSpacing(15)
        left_form.addRow(self.split_cbox)
        left_form.addRow('Output map format', self.map_ext_combox)
        left_form.addRow(self.convert_btn)
        left_scroll = CW.GroupScrollArea(left_form, 'Options', frame=False)

        right_grid = QW.QGridLayout()
        right_grid.addWidget(self.import_btn, 0, 0)
        right_grid.addWidget(self.remove_btn, 0, 1)
        right_grid.addWidget(self.img_list, 1, 0, 1, -1)
        right_scroll = CW.GroupScrollArea(right_grid, frame=True)

        splitter = CW.SplitterGroup([left_scroll, right_scroll], (0, 1))
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.progbar)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.import_btn.clicked.connect(self.importImages)
        self.remove_btn.clicked.connect(self.img_list.removeSelected)
        self.convert_btn.clicked.connect(self.convertImages)


    def importImages(self) -> None:
        '''
        Import images and show their paths as well as a small preview.

        '''
    # Do nothing if paths are invalid or the file dialog is canceled
        ft = 'TIF (*.tif *.tiff);;BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg)'
        paths = CW.FileDialog(self, 'open', 'Load Images', ft, True).get()
        if not paths:
            return

    # Add images paths and previews (as icon) to list but skip those that had 
    # already been added
        self.progbar.setRange(0, len(paths))
        for n, p in enumerate(paths, start=1):
            self.progbar.setValue(n)
            if len(self.img_list.findItems(p, Qt.MatchExactly)): 
                continue
            try:
                # TIFF file previews may spam warnings on the console (BUG)
                transforms = (Qt.KeepAspectRatio, Qt.SmoothTransformation)
                pixmap = QPixmap(p).scaled(self.preview_size, *transforms)
                self.img_list.addItem(QW.QListWidgetItem(QIcon(pixmap), p))
            except Exception as e:
                self.progbar.reset()
                return CW.MsgBox(self, 'C', 'File import failed.', str(e))

        self.progbar.reset()


    def convertImages(self) -> None:
        '''
        Convert images to InputMaps and save them.

        '''
    # Deny conversion if no image is loaded
        img_count = self.img_list.count()
        if not img_count:
            return CW.MsgBox(self, 'Crit', 'No image loaded.')

    # Do nothing if directory is invalid or the direcotry dialog is canceled
        outdir = CW.FileDialog(self, 'S', 'Output Folder', multifile=True).get()
        if not outdir:
            return
    
    # Try converting images to Input Maps and saving them
        errors_log = []
        ext = str(self.map_ext_combox.currentText())
        self.progbar.setRange(0, img_count)

        for n, i in enumerate(self.img_list.getItems(), start=1):
            try:
                path = i.text()
                fname = cf.path2filename(path) + ext
                self.progbar.setValue(n)
            
            # Convert image to array
                array = iatools.image2array(path, InputMap._DTYPE)

            # Save InputMap. Split multi-channel arrays, if requested
                if array.ndim == 3 and self.split_cbox.isChecked():
                    channels = np.split(array, array.shape[-1], axis=2)
                    for idx, c in enumerate(channels, start=1):
                        fname_channel = cf.extend_filename(fname, f'_ch{idx}')
                        out = os.path.join(outdir, fname_channel)
                        InputMap(np.squeeze(c)).save(out)
                else:
                    out = os.path.join(outdir, fname)
                    InputMap(array).save(out)

            except Exception as e:
                errors_log.append((path, e))
        
        self.progbar.reset()

    # Send message box with success confirm or error warnings
        if len(errors_log):
            fpaths, err = zip(*errors_log)
            return CW.MsgBox(self, 'Warn', f'Failed conversions:\n\n{fpaths}',
                             '\n'.join(str(e) for e in err)) 

        else:
            return CW.MsgBox(self, 'Info', 'Images successfully converted.')
        


class ImageToMineralMap(QW.QDialog):

    def __init__(self, parent: QW.QWidget | None = None):
        '''
        A dialog to convert images to Mineral Maps.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('Image To Mineral Map')
        self.setWindowIcon(style.getIcon('GEAR'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # Set main attribute
        self.image_array = None
        self.minmap = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Import image (Styled Button)
        self.import_btn = CW.StyledButton(style.getIcon('IMPORT'), 'Import')

    # Image path (Path Label)
        self.path_lbl = CW.PathLabel(full_display=False)

    # Pixel color delta (Styled Spin Box)
        self.delta_spbox = CW.StyledSpinBox(1, 255)
        self.delta_spbox.setToolTip('Minimum color variance to split classes')

    # Convert (Styled Button)
        self.convert_btn = CW.StyledButton(text='Convert', bg=style.OK_GREEN)

    # Legend (Legend)
        self.legend = CW.Legend()

    # Save (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'Save')
        self.save_btn.setEnabled(False)

    # Canvas (Image Canvas)
        self.canvas = plots.ImageCanvas(size=(5, 3.75))
        self.canvas.setMinimumSize(300, 300)

    # Navigation Toolbar (Navigation Toolbar)
        self.navtbar = plots.NavTbar.imageCanvasDefault(self.canvas, self)

    # Progress bar (Progress Bar)
        self.progbar = QW.QProgressBar()

    # Adjust layout
        options_vbox = QW.QFormLayout()
        options_vbox.setVerticalSpacing(15)
        options_vbox.addRow('Color delta', self.delta_spbox)
        options_vbox.addRow(self.convert_btn)
        options_vbox.addRow(self.legend)
        options_vbox.addRow(self.save_btn)
        options_group = CW.GroupArea(options_vbox, 'Options')
        
        left_vbox = QW.QVBoxLayout()
        left_vbox.addWidget(self.import_btn)
        left_vbox.addWidget(self.path_lbl)
        left_vbox.addWidget(options_group)
        left_scoll = CW.GroupScrollArea(left_vbox, frame=False)

        right_vbox = QW.QVBoxLayout()
        right_vbox.addWidget(self.navtbar)
        right_vbox.addWidget(self.canvas)
        right_scroll = CW.GroupScrollArea(right_vbox, frame=False)

        splitter = CW.SplitterGroup((left_scoll, right_scroll), (0, 1))
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.progbar)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.import_btn.clicked.connect(self.importImage)
        self.convert_btn.clicked.connect(self.convertImage)
        self.save_btn.clicked.connect(self.saveMineralMap)

    # Legend signals
        self.legend.colorChangeRequested.connect(self.changeColor)
        self.legend.itemRenameRequested.connect(self.renameClass)
        self.legend.itemHighlightRequested.connect(self.highlightClass)


    def importImage(self) -> None:
        '''
        Import image from file and store it as a NumPy array.

        '''
    # Do nothing if image path is invalid or file dialog is canceled 
        ft = 'TIF (*.tif *.tiff);;BMP (*.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg)'
        path = CW.FileDialog(self, 'open', 'Load Image', ft).get()
        if not path:
            return

    # Store the image array
        try:
            self.image_array = iatools.image2array(path, dtype='int32')
            self.path_lbl.setPath(path)
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to import image.', str(e))


    def convertImage(self) -> None:
        '''
        Convert imported image into a mineral map.

        '''
    # Deny conversion if no image is loaded
        array = self.image_array.copy()
        if array is None:
            return CW.MsgBox(self, 'Crit', 'No image loaaded.')
        
    # Get image array shape
        if array.ndim == 3:
            row, col, chan = array.shape
        else:
            row, col = array.shape
            chan = 1

    # Convert greyscale (and binary) arrays to RGB
        if chan == 1:
            if ((array==0) | (array==1)).all():
                array = iatools.binary2greyscale(array)
            array = iatools.greyscale2rgb(array)

    # Convert RGBA array to RGB
        elif chan == 4:
            array = iatools.rgba2rgb(array)

    # Deny conversion if loaded image has 2 or more than 4 channels (possible?)
        elif chan == 2 or chan > 4:
            return CW.MsgBox(self, 'Crit', 'This image cannot be converted.')
        
    # Get unique RGB values. Use color delta to reduce color variance.
        delta = self.delta_spbox.value()
        array = (array//delta) * delta
        array = array.reshape(-1, 3)
        unique = np.unique(array, axis=0)

    # Deny conversion for more than (2**16)/2 = 32768 classes (can be more)
        n_classes = len(unique)
        if n_classes > 32768:
            return CW.MsgBox(self, 'Crit', 'Too many classes found.')

    # Build a flattened minmap
        self.progbar.setRange(0, n_classes)
        minmap = np.empty((row * col), dtype=MineralMap._DTYPE_STR)
        palette = dict()
        for n, rgb in enumerate(unique):
            self.progbar.setValue(n + 1)
            mask = (array == rgb).all(axis=1)
            minmap[mask] = f'{n:05d}' # Ensure correct classes order in legend
            palette[n] = tuple(rgb)

    # Construct mineral map object
        self.minmap = MineralMap(minmap.reshape(row, col), palette_dict=palette)

    # Plot the map and update legend
        title = self.path_lbl.text()
        mmap_enc, encoder, colors = self.minmap.get_plot_data()
        self.canvas.draw_discretemap(mmap_enc, encoder, colors, title)
        self.legend.update(self.minmap)

    # Enable the save button
        self.save_btn.setEnabled(True)

    # Reset progress bar
        self.progbar.reset()


    def saveMineralMap(self) -> None:
        '''
        Save mineral map to file.

        '''
    # Do nothing if output path is invalid or file dialog is canceled
        ftype = 'Mineral map (*.mmp)'
        outpath = CW.FileDialog(self, 'save', 'Save Map', ftype).get()
        if not outpath:
            return
    
    # Save mineral map
        self.minmap.save(outpath)
        CW.MsgBox(self, 'Info', 'Map successfully saved.')
        

    def changeColor(
        self,
        legend_item: QW.QTreeWidgetItem,
        color: tuple[int, int, int]
    ) -> None:
        '''
        Alter the displayed color of a mineral class. This method propagates 
        the changes to mineral map, canvas and legend. The arguments of this 
        method are specifically compatible with the 'colorChangeRequested'
        signal emitted by the legend (see 'Legend' class for more details). 

        Parameters
        ----------
        legend_item : QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple[int, int, int]
            RGB triplet.

        '''
        if self.minmap is None: # safety
            return
        
    # Change color in the mineral map
        class_name = legend_item.text(1)
        self.minmap.set_phase_color(class_name, color)
    
    # Update canvas and legend
        self.canvas.alter_cmap(self.minmap.palette.values())
        self.legend.changeItemColor(legend_item, color)


    def renameClass(
        self,
        legend_item: QW.QTreeWidgetItem,
        new_name: str
    ) -> None:
        '''
        Rename a mineral class. This method propagates the changes to mineral
        map, canvas and legend. The arguments of this method are specifically
        compatible with the 'itemRenameRequested' signal emitted by the legend 
        (see 'Legend' class for more details).

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be renamed
        new_name : str
            New class name.

        '''
        if self.minmap is None: # safety
            return
        
    # Rename class in the mineral map
        old_name = legend_item.text(1)
        self.minmap.rename_phase(old_name, new_name)
    
    # Update canvas and legend
        mmap_enc, encoder, colors = self.minmap.get_plot_data()
        self.canvas.draw_discretemap(mmap_enc, encoder, colors)
        self.legend.renameClass(legend_item, new_name)


    def highlightClass(
        self,
        toggled: bool,
        legend_item: QW.QTreeWidgetItem
    ) -> None:
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
        if self.minmap is None: # safety
            return

        if toggled:
            phase_id = self.minmap.as_id(legend_item.text(1))
            vmin, vmax = phase_id - 0.5, phase_id + 0.5
        else:
            vmin, vmax = None, None

        self.canvas.update_clim(vmin, vmax)
        self.canvas.draw_idle() 



class DummyMapsBuilder(QW.QDialog):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog to generate dummy maps, useful as a placeholder for missing
        input maps when using model-based classifiers.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('Dummy Maps Builder')
        self.setWindowIcon(style.getIcon('GEAR'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # Set main attribute
        self.dummy_map = None

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self.adjustSize()
        self._connect_slots()


    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
    # Tool info (Framed Label)
        info = (
            'Build artificial noisy maps with near-zero values, randomized '
            'through a Gamma distribution function. These maps can be used '
            'as a placeholder for missing maps in a pre-trained model.'
        )
        self.info_lbl = CW.FramedLabel(info)
        self.info_lbl.setWordWrap(True)

    # Map width (Styled Spin Box)
        self.width_spbox = CW.StyledSpinBox(1, 10**5)
        self.width_spbox.setValue(100)

    # Map height (Styled Spin Box)
        self.height_spbox = CW.StyledSpinBox(1, 10**5)
        self.height_spbox.setValue(100)

    # Shape of gamma distribution function (Styled Double Spin Box)
        self.shape_spbox = CW.StyledDoubleSpinBox(0.1, 100., 0.1)
        self.shape_spbox.setValue(1.5)
        self.shape_spbox.setToolTip('Shape of Gamma distribution function')

    # Scale of gamma distribution function (Styled Double Spin Box)
        self.scale_spbox = CW.StyledDoubleSpinBox(0.1, 100., 0.1)
        self.scale_spbox.setValue(1.)
        self.scale_spbox.setToolTip('Scale of Gamma distribution function')

    # Generate random map (Styled Button)
        self.rand_btn = CW.StyledButton(text='Randomize', bg=style.OK_GREEN)

    # Save map (Styled Button)
        self.save_btn = CW.StyledButton(style.getIcon('SAVE'), 'Save')

    # Preview histogram (Histogram Canvas) 
        self.canvas = plots.HistogramCanvas(wheel_pan=False, wheel_zoom=False)
        self.canvas.setMinimumSize(300, 300)
    
    # Navigation Toolbar for preview histogram (Navigation Toolbar)
        self.navtbar = plots.NavTbar.histCanvasDefault(self.canvas, self)

    # Adjust layout
        options_form = QW.QFormLayout()
        options_form.addRow('Map width', self.width_spbox)
        options_form.addRow('Map height', self.height_spbox)
        options_form.addRow('Function shape', self.shape_spbox)
        options_form.addRow('Function scale', self.scale_spbox)
        options_group = CW.GroupArea(options_form, 'Options')

        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(15)
        left_vbox.addWidget(options_group)
        left_vbox.addWidget(self.rand_btn)
        left_vbox.addWidget(self.save_btn)
        left_scroll = CW.GroupScrollArea(left_vbox, tight=True, frame=False)

        right_vbox = QW.QVBoxLayout()
        right_vbox.addWidget(self.navtbar)
        right_vbox.addWidget(self.canvas)
        right_scroll = CW.GroupScrollArea(right_vbox, tight=True, frame=False)
 
        main_layout = CW.SplitterLayout()
        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(right_scroll, 1)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
        self.rand_btn.clicked.connect(self.generateMap)
        self.save_btn.clicked.connect(self.saveMap)


    def generateMap(self) -> None:
        '''
        Generate and render dummy map.

        '''
    # Gather the parameters
        w = self.width_spbox.value()
        h = self.height_spbox.value()
        shp = self.shape_spbox.value()
        scl = self.scale_spbox.value()

    # Generate dummy map
        self.dummy_map = iatools.noisy_array(shp, scl, (h, w), InputMap._DTYPE)

    # Refresh the histogram
        self.canvas.update_canvas(self.dummy_map, title='Dummy map histogram')


    def saveMap(self) -> None:
        '''
        Save dummy map to file.

        '''
    # Deny saving if no map is generated
        if self.dummy_map is None:
            CW.MsgBox(self, 'Crit', 'No map generated.')
        
    # Do nothing if output path is invalid or file dialog is canceled
        ftypes = 'Compressed ASCII file (*.gz);;ASCII file (*.txt)'
        path = CW.FileDialog(self, 'save', 'Save Map', ftypes).get()
        if not path:
            return
    
    # Save map  
        try:
            InputMap(self.dummy_map).save(path)
            CW.MsgBox(self, 'Info', 'Map saved succesfully.')
        except Exception as e:
            CW.MsgBox(self, 'Crit', 'Failed to save map.', str(e))



class Preferences(QW.QDialog):

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        '''
        A dialog to access application preferences.

        Parameters
        ----------
        parent : QWidget or None, optional
            The GUI parent of this dialog. The default is None.

        '''
        super().__init__(parent)

    # Set dialog widget attributes
        self.resize(420, 580)
        self.setWindowTitle('Preferences')
        self.setWindowIcon(style.getIcon('WRENCH'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.readSettings()

    # Initialize GUI and connect its signals with slots
        self._init_ui()
        self._connect_slots()
        

    def _init_ui(self) -> None:
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                           INTERFACE SETTINGS
#  -------------------------------------------------------------------------  #
    # Application font-size (Styled Spin Box)
        self.fontsize_spbox = CW.StyledSpinBox(8, 16)
        self.fontsize_spbox.setValue(self._fontsize)

    # Smooth GUI (Check Box)
        self.smooth_cbox = QW.QCheckBox()
        self.smooth_cbox.setToolTip('Smooth feedback when resizing widgets')
        self.smooth_cbox.setChecked(self._smooth_gui)

    # Interface tab (Group Scroll Area)
        gui_form = QW.QFormLayout()
        gui_form.setSpacing(15)
        gui_form.addRow('Font size', self.fontsize_spbox)
        gui_form.addRow('Smooth resize', self.smooth_cbox)
        gui_scroll = CW.GroupScrollArea(gui_form, tight=True, frame=False)

#  -------------------------------------------------------------------------  #
#                           PLOT STYLE SETTINGS
#  -------------------------------------------------------------------------  #
    # ROI color (Styled Button)
        self.roi_col_btn = CW.StyledButton(CW.ColorIcon(self._roi_col))
        
    # ROI selection color (Styled Button)
        self.roi_selcol_btn = CW.StyledButton(CW.ColorIcon(self._roi_selcol))

    # ROI filled (Check Box)
        self.roi_filled_cbox = QW.QCheckBox('Filled')
        self.roi_filled_cbox.setChecked(self._roi_filled)

    # ROI group (Collapsible Area)
        roi_form = QW.QFormLayout()
        roi_form.setSpacing(15)
        roi_form.addRow('Color', self.roi_col_btn)
        roi_form.addRow('Selection color', self.roi_selcol_btn)
        roi_form.addRow(self.roi_filled_cbox)
        roi_group = CW.CollapsibleArea(roi_form, 'ROI')

    # Plots tab (Group Scroll Area)
        plots_form = QW.QFormLayout()
        plots_form.setSpacing(15)
        plots_form.addRow(roi_group)
        plots_scroll = CW.GroupScrollArea(plots_form, tight=True, frame=False)

#  -------------------------------------------------------------------------  #
#                              DATA SETTINGS
#  -------------------------------------------------------------------------  #
    # Decimal precision (Styled Spin Box)
        self.decimal_spbox = CW.StyledSpinBox(0, 6)
        self.decimal_spbox.setToolTip('Number of displayed decimal places')
        self.decimal_spbox.setValue(self._decimal_prec)

    # Mask merging rule (Radio Buttons Layout)
        btns = ('Intersection', 'Union')
        idx = btns.index(self._mask_merge_type.capitalize())
        tip1 = 'Show pixels if visible through at least one mask'
        tip2 = 'Show pixels only if visible through all masks'
        self.mask_merge_btns = CW.RadioBtnLayout(btns, default=idx, orient='h')
        self.mask_merge_btns.setSpacing(5)
        self.mask_merge_btns.button(0).setToolTip(tip1)
        self.mask_merge_btns.button(1).setToolTip(tip2)

    # Phase count warning (Styled Spin Box)
        self.phase_count_spbox = CW.StyledSpinBox(15, 100)
        tip = 'On mineral map loading, warn user if phases exceed this amount'
        self.phase_count_spbox.setToolTip(tip)
        self.phase_count_spbox.setValue(self._warn_phase_count)

    # Extended log (Check Box)
        self.extlog_cbox = QW.QCheckBox('Extended log')
        self.extlog_cbox.setToolTip('Save advanced info in custom model logs')
        self.extlog_cbox.setChecked(self._extended_log)

    # Mineral map group (Collapsible Area)
        mmap_form = QW.QFormLayout()
        mmap_form.setSpacing(15)
        mmap_form.addRow('Phase count warning', self.phase_count_spbox)
        mmap_group = CW.CollapsibleArea(mmap_form, 'Mineral Map')

    # Mask group (Collapsible Area)
        mask_form = QW.QFormLayout()
        mask_form.setSpacing(15)
        mask_form.addRow('Default merge', self.mask_merge_btns)
        mask_group = CW.CollapsibleArea(mask_form, 'Mask')

    # Model group (Collapsible Area)
        model_form = QW.QFormLayout()
        model_form.setSpacing(15)
        model_form.addRow(self.extlog_cbox)
        model_group = CW.CollapsibleArea(model_form, 'Model')

    # Data tab (Group Scroll Area)
        data_form = QW.QFormLayout()
        data_form.setSpacing(15)
        data_form.addRow('Decimals', self.decimal_spbox)
        data_form.addRow(mmap_group)
        data_form.addRow(mask_group)
        data_form.addRow(model_group)
        data_scroll = CW.GroupScrollArea(data_form, tight=True, frame=False)

#  -------------------------------------------------------------------------  #
#                              DIALOG BUTTONS
#  -------------------------------------------------------------------------  #
    # Ok (Styled Button)
        self.ok_btn = CW.StyledButton(text='Ok')

    # Cancel (Styled Button)
        self.cancel_btn = CW.StyledButton(text='Cancel')

    # Apply (Styled Button)
        self.apply_btn = CW.StyledButton(text='Apply')

    # Default (Styled Button)
        self.default_btn = CW.StyledButton(text='Default')

#  -------------------------------------------------------------------------  #
#                              MAIN LAYOUT
#  -------------------------------------------------------------------------  #
    # Main widget (Styled Tab Widget)
        tabwid = CW.StyledTabWidget()
        tabwid.tabBar().setExpanding(True)
        tabwid.tabBar().setDocumentMode(True)
        tabwid.addTab(gui_scroll, title='Interface')
        tabwid.addTab(plots_scroll, title='Plot Style')
        tabwid.addTab(data_scroll, title='Data')

    # Dialog buttons layout
        btns_hbox = QW.QHBoxLayout()
        btns_hbox.addWidget(self.ok_btn)
        btns_hbox.addWidget(self.cancel_btn)
        btns_hbox.addWidget(self.apply_btn)
        btns_hbox.addWidget(self.default_btn)

    # Adjust main layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(tabwid)
        main_layout.addLayout(btns_hbox)
        self.setLayout(main_layout)


    def _connect_slots(self) -> None:
        '''
        Signals-slots connector.

        '''
    # Smooth GUI
        self.smooth_cbox.stateChanged.connect(
            lambda chk: setattr(self, '_smooth_gui', bool(chk)))
    
    # ROI-related signals
        self.roi_col_btn.clicked.connect(self.changeRoiIconColor)
        self.roi_selcol_btn.clicked.connect(self.changeRoiIconColor)
        self.roi_filled_cbox.stateChanged.connect(
            lambda chk: setattr(self, '_roi_filled', bool(chk)))

    # Decimal precision
        self.decimal_spbox.valueChanged.connect(
            lambda v: setattr(self, '_decimal_prec', v))
    
    # Phase count warning
        self.phase_count_spbox.valueChanged.connect(
            lambda v: setattr(self, '_warn_phase_count', v))
        
    # Mask merging rule
        self.mask_merge_btns.selectionChanged.connect(self.changeMaskMergeRule)
        
    # Extended model logs
        self.extlog_cbox.stateChanged.connect(
            lambda chk: setattr(self, '_extended_log', bool(chk)))

    # Dialog buttons signals
        self.ok_btn.clicked.connect(lambda: self.saveEdits(exit=True))
        self.cancel_btn.clicked.connect(self.close)
        self.apply_btn.clicked.connect(self.saveEdits)
        self.default_btn.clicked.connect(self.resetToDefault)


    def readSettings(self) -> None:
        '''
        Read settings from settings.ini file.

        '''
        self._fontsize = pref.get_setting('GUI/fontsize')
        self._smooth_gui = pref.get_setting('GUI/smooth_animation')
        self._roi_col = pref.get_setting('plots/roi_color')
        self._roi_selcol = pref.get_setting('plots/roi_selcolor')
        self._roi_filled = pref.get_setting('plots/roi_filled')
        self._decimal_prec = pref.get_setting('data/decimal_precision')
        self._extended_log = pref.get_setting('data/extended_model_log')
        self._mask_merge_type = pref.get_setting('data/mask_merging_rule')
        self._warn_phase_count = pref.get_setting('data/warn_phase_count')


    def writeSettings(self) -> None:
        '''
        Write settings to settings.ini file.

        '''
        pref.edit_setting('GUI/fontsize', self._fontsize)
        pref.edit_setting('GUI/smooth_animation', self._smooth_gui)
        pref.edit_setting('plots/roi_color', self._roi_col)
        pref.edit_setting('plots/roi_selcolor', self._roi_selcol)
        pref.edit_setting('plots/roi_filled', self._roi_filled)
        pref.edit_setting('data/decimal_precision', self._decimal_prec)
        pref.edit_setting('data/extended_model_log', self._extended_log)
        pref.edit_setting('data/mask_merging_rule', self._mask_merge_type)
        pref.edit_setting('data/warn_phase_count', self._warn_phase_count)


    def changeRoiIconColor(self) -> None:
        '''
        Set a new ROI (selection) color.

        '''
    # Get the current color as HEX string
        btn = self.sender()
        curr = self._roi_col if btn == self.roi_col_btn else self._roi_selcol

    # Set new current color if new color is valid
        new = QW.QColorDialog.getColor(QColor(curr), self)
        if new.isValid():
            hex = new.name(QColor.HexRgb)
            curr = hex
            btn.setIcon(CW.ColorIcon(hex))


    def changeMaskMergeRule(self, selected_id: int) -> None:
        '''
        Change default mask merging mode based on the selected radio button.

        Parameters
        ----------
        selected_id : int
            Id of the selected radio button.

        '''
        rule = self.mask_merge_btns.button(selected_id).text().lower()
        self._mask_merge_type = rule


    def setAppFontSize(self) -> None:
        '''
        Change app font size. Warning: some widgets needs to be repainted to
        apply the new fontsize.

        '''
        self._fontsize = self.fontsize_spbox.value()
        app = QW.qApp
        font = app.font()
        font.setPointSize(self._fontsize)
        app.setFont(font)


    def resetToDefault(self) -> None:
        '''
        Reset all settings to default values.

        '''
        text = (
            'Reset preferences to default? Some changes may take effect after '
            'restarting the app.'
        )
        choice = CW.MsgBox(self, 'Quest', text)
        if choice.yes():
            pref.clear_settings()
            self.close()


    def saveEdits(self, exit: bool = False) -> None:
        '''
        Save applied changes to settings and exit the dialog if required.

        Parameters
        ----------
        exit : bool, optional
            Whether to close the dialog after saving. The default is False.

        '''
    # Perform changes that can be applied on the fly
        self.setAppFontSize()

    # Save edits in settings.ini file
        self.writeSettings()
    
    # Close dialog if requested
        if exit:
            text = 'Some changes may take effect after restarting the app.'
            CW.MsgBox(self, 'Info', text)
            self.close()