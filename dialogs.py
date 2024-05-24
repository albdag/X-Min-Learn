# -*- coding: utf-8 -*-
"""
Created on Tue May  14 15:03:45 2024

@author: albdag
"""

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon
import PyQt5.QtWidgets as QW

import numpy as np

from _base import RoiMap, InputMapStack
import customObjects as cObj
import plots
import threads




class AutoRoiDetector(QW.QDialog):
    '''
    A dialog for the RoiEditor pane that allows the automatic identification
    of ROIs on the input maps of a given sample. The identification is based
    on the analysis of a cumulative Neighborhood Pixel Variance map, extracted
    from input data.
    '''
    requestRoiMap = pyqtSignal()
    drawingRequested = pyqtSignal(RoiMap)

    def __init__(self, parent=None):
        '''
        AutoRoiDetector class constructor.

        Parameters
        ----------
        parent : qObject | None, optional
            The GUI parent of this dialog. The default is None.
        '''
        super(AutoRoiDetector, self).__init__(parent)

    # Set dialog widget attributes
        self.setWindowTitle('ROI detector')
        self.setWindowIcon(QIcon(r'Icons/roi_detection.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)

    # Set main attributes
        self._current_roimap = None
        self._auto_roimap = None
        self._patches = []
        self._active_thread = None
        self.npv_thread = threads.NpvThread()
        self.roi_detect_thread = threads.RoiDetectionThread()
        self.npv_encoder = {'Pure': np.sum, 
                            'Simple': np.median, 
                            'Smooth': np.mean}

        self._init_ui()
        self._connect_slots()
        

    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Input maps selector
        self.maps_selector = cObj.InputMapsSelector()

    # Number of ROIs selector (SpinBox)
        self.nroi_spbox = cObj.StyledSpinBox(min_value=1, max_value=100)
        self.nroi_spbox.setValue(15)

    # ROI size selector (SpinBox)
        self.size_spbox = cObj.StyledSpinBox(min_value=3, max_value=99, step=2)
        self.size_spbox.setValue(9)
        self.size_spbox.setToolTip('Define a NxN ROI. N must be an odd number')

    # ROI distance selector
        self.distance_spbox = cObj.StyledSpinBox()
        self.distance_spbox.setValue(10)
        self.distance_spbox.setToolTip('Minimum distance between ROI')

    # NPV function types (QButtonLayout)
        self.npv_btns = cObj.RadioBtnLayout(('Pure', 'Simple', 'Smooth'))
        tooltips = ('Use sum function to penalize all noisy pixels',
                    'Use median function to ignore outliers',
                    'Use mean function to smoothen noise')
        for n, t in enumerate(tooltips):
            self.npv_btns.button(n).setToolTip(t)

    # NPV canvas (ImageCanvas)
        self.canvas = plots.ImageCanvas()
    
    # NPV canvas Navigation Toolbar 
        self.navtbar = plots.NavTbar(self.canvas, self)
        self.navtbar.fixHomeAction()
        self.navtbar.removeToolByIndex([3, 4, 8, 9])

    # Search ROIs button
        self.search_btn = cObj.StyledButton(QIcon(r'Icons/roi_detection.png'),
                                            'Search ROI')
        self.search_btn.setEnabled(False)

    # Descriptive Progress bar 
        self.progbar = cObj.DescriptiveProgressBar()

    # OK button
        self.ok_btn = cObj.StyledButton(text='Ok')

    # Cancel button
        self.cancel_btn = cObj.StyledButton(text='Cancel')

    # Set layout
        params_form = QW.QFormLayout()
        params_form.addRow('Number of ROI', self.nroi_spbox)
        params_form.addRow('ROI size', self.size_spbox)
        params_form.addRow('ROI distance', self.distance_spbox)

        radbtn_group = cObj.GroupArea(self.npv_btns, 'NPV type', 
                                      align=Qt.AlignLeft)

        left_vbox = QW.QVBoxLayout()
        left_vbox.addWidget(self.maps_selector)
        left_vbox.addLayout(params_form)
        left_vbox.addWidget(radbtn_group)
        left_vbox.addWidget(self.search_btn)
        left_scroll = cObj.GroupScrollArea(left_vbox)

        right_grid = QW.QGridLayout()
        right_grid.addWidget(self.navtbar, 0, 0, 1, -1)
        right_grid.addWidget(self.canvas, 1, 0, 1, -1)
        right_grid.addWidget(self.progbar, 2, 0, 1, -1)
        right_grid.addWidget(self.ok_btn, 3, 0)
        right_grid.addWidget(self.cancel_btn, 3, 1)
        right_grid.setRowStretch(1, 1)

        main_layout = QW.QHBoxLayout()
        main_layout.addWidget(left_scroll)
        main_layout.addLayout(right_grid)
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        Connect signals to slots.

        '''
    # Connect the threads signals
        self.npv_thread.taskInitialized.connect(self.progbar.step)
        self.npv_thread.workFinished.connect(self._parse_npv)
        self.roi_detect_thread.taskInitialized.connect(self.progbar.step)
        self.roi_detect_thread.workFinished.connect(self._parse_roi_detection)

    # Enable search btn and update parameters when input maps list is populated
        self.maps_selector.inputDataChanged.connect(
            lambda: self.search_btn.setEnabled(True))
        self.maps_selector.inputDataChanged.connect(self.update_params_range)

    # Fix even size values
        self.size_spbox.valueChanged.connect(self.fix_even_size)

    # Launch the NPV computation when search button is clicked
        self.search_btn.clicked.connect(self.launch_npv_computation)

    # Accept the result when OK button is clicked
        self.ok_btn.clicked.connect(self.accept)

    # Reject the result when Cancel button is clicked
        self.cancel_btn.clicked.connect(self.reject)


    def fix_even_size(self, size):
        '''
        Adjust kernel size so that it is always an odd number.

        Parameters
        ----------
        size : _type_
            _description_
        '''
        if not size % 2:
            self.size_spbox.setValue(size - 1)


    def set_current_roimap(self, roimap: RoiMap):
        '''
        Set the current roimap. This function is especially useful when called
        by the parent widget (i.e., the ROI Editor pane).

        Parameters
        ----------
        roimap : RoiMap
            New current ROI map.

        '''
        self._current_roimap = roimap


    def _clear_roi_patches(self):
        '''
        Remove any previous ROI patch displayed in canvas.

        '''
        for idx in reversed(range(len(self._patches))):
            patch = self._patches.pop(idx)
            patch.remove()


    def update_params_range(self):
        '''
        Change maximum selectable ROI size and ROI distance based on the shape
        of the input maps.

        '''
    # Get the shape of the first map in the list. If maps have different shapes
    # the dialog will later abort the computation anyway.
        shp = self.maps_selector.inmaps_list.topLevelItem(0).get('data').shape

    # Set the maximum selectable ROI distance and ROI size based on shape
        max_dist = min(shp) // 10
        max_size = max_dist if not max_dist % 2 else max_dist - 1 # must be ODD
        self.distance_spbox.setMaximum(max_dist)
        self.size_spbox.setMaximum(max_size)


    def launch_npv_computation(self):
        '''
        Launch the thread that computes the cumulative Neighborhood Pixel 
        Variance (NPV) map.

        '''
    # Exit function if a thread is currentlty active
        if self._active_thread is not None:
            info = 'A detection process is currently active.'
            return QW.QMessageBox.information(self, 'X-Min Learn', info)

    # Disable OK button
        self.ok_btn.setEnabled(False)

    # Get user's parameters
        size = self.size_spbox.value()
        npv_func = self.npv_encoder[self.npv_btns.getChecked().text()]
        
    # Get input map data
        checked_maps = self.maps_selector.getChecked()
        if not len(checked_maps):
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'Select at '\
                                           'least one map')
        inmaps, names = zip(*[i.get('data', 'name') for i in checked_maps])
        input_stack = InputMapStack(inmaps)

    # Check that input maps share the same shape
        if not input_stack.maps_fit():
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'The selected '\
                                           'maps have different shapes')
        
    # Get current ROI map and check its shape
        self.requestRoiMap.emit()
        if self._current_roimap is None:
            self._current_roimap = RoiMap.from_shape(input_stack.maps_shape)
        elif self._current_roimap.shape != input_stack.maps_shape:
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'The currently '\
                                           'loaded ROI map and the selected '\
                                           'input maps have different shapes')

    # Launch the NPV thread
        self.progbar.setMaximum(len(inmaps))
        self._active_thread = self.npv_thread
        self.npv_thread.set_params(input_stack.arrays, names, size, npv_func)
        self.npv_thread.start()


    def _parse_npv(self, thread_result: tuple, success: bool): 
        '''
        Parse the result of the NPV computation thread. If the computation was
        successful, this function automatically launches the ROI identification
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
            cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                            'NPV calculation failed', detailedText=repr(e))
            self._end_threaded_session()

    
    def launch_roi_detection(self, npv_map: np.ndarray):
        '''
        Launch the thread that automatically identifies the best ROIs. This
        function is meant to be called automatically after the NPV computation
        thread successfully returned a cumulative NPV map (see _parse_npv 
        function for details).

        Parameters
        ----------
        npv_map : ndarray
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


    def _parse_roi_detection(self, thread_result:tuple, success:bool):
        '''
        Parse the result of the ROI detection thread. If the identification was
        successfull, this function displays the identified ROIs in canvas and
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
                patch = cObj.RoiPatch(bbox, 'black', filled=False)
                patch.set_label('Existent ROI')
                self.canvas.ax.add_patch(patch)
                self._patches.append(patch)

        # Display new auto detected ROIs in red
            auto_roimap, = thread_result
            for _, bbox in auto_roimap.roilist:
                patch = cObj.RoiPatch(bbox, 'red', filled=False)
                patch.set_label('Auto detected ROI')
                self.canvas.ax.add_patch(patch)
                self._patches.append(patch)

        # Show legend and render canvas
            hand, lbls = self.canvas.ax.get_legend_handles_labels()
            entries = dict(zip(lbls, hand))
            self.canvas.figure.legend(entries.values(), entries.keys(),
                                      loc='lower left')
            self.canvas.draw()

        # Save the auto detected roimap
            self._auto_roimap = auto_roimap

        else:
            e, = thread_result
            cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                            'ROI detection failed', detailedText=repr(e))
            
    # End the threaded session anyway
        self._end_threaded_session()    


    def interrupt_thread(self):
        '''
        Stop an external thread.

        Returns
        -------
        bool
            Whether the stop event is confirmed by user or not.

        '''
        quest = 'Do you want to interrupt the automatic ROI detection?'
        choice = QW.QMessageBox.question(self, 'X-Min Learn', quest,
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.Yes)
        if choice == QW.QMessageBox.Yes:
            self._active_thread.requestInterruption()
            self._end_threaded_session()
            return True
        else:
            return False


    def _end_threaded_session(self):
        '''
        Internally and visually exit from an external thread session.

        '''
        self._active_thread = None
        self.progbar.reset()
        self.ok_btn.setEnabled(True)


    def accept(self):
        '''
        Reimplementation of the dialog's accept function. A custom signal is 
        emitted to draw the newly identified ROIs on the input maps.

        '''
        if self._auto_roimap is not None:
            self.drawingRequested.emit(self._auto_roimap)
        super(AutoRoiDetector, self).accept()

    def reject(self):
        '''
        Reimplementation of the dialog's reject function. It just add a way to
        send an exit confirmation message when an external thread is currently
        active.

        '''
        if self._active_thread is not None and self.interrupt_thread():
            super(AutoRoiDetector, self).reject()
        else:
            super(AutoRoiDetector, self).reject()
        

    def show(self):
        '''
        Reimplementation of the dialog's show function. It reset's the dialog
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
            super(AutoRoiDetector, self).show()
        else:
            self.activateWindow()


