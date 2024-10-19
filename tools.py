# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:16:01 2021

@author: albdag
"""
import os
from datetime import datetime
from statistics import mode as math_mode
import webbrowser

import numpy as np
from pandas import DataFrame, concat
from scipy import ndimage as nd

from PyQt5 import QtWidgets as QW
from PyQt5 import QtCore as QC
from PyQt5 import QtGui as QG
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from _base import InputMap, MineralMap, RoiMap, Mask, InputMapStack
import conv_functions as CF
import customObjects as cObj
import ML_tools
import plots
import preferences as pref
import threads


class DraggableTool(QW.QWidget):
    '''
    The main class for all major windows of X-Min Learn, that can be dragged
    and dropped within the main window tab area. See the MainTabWidget class
    for more details.
    '''

    def __init__(self, parent=None):
        '''
        DraggableTool class constructor.

        Parameters
        ----------
        parent : qObject, optional
            The GUI parent widget. The default is None.

        Returns
        -------
        None.

        '''
        super(DraggableTool, self).__init__(parent)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)


    # def dragEnterEvent(self, e): # NOT USEFUL HERE.
    #     '''
    #     Reimplementation of the dragEnterEvent default functions. It just
    #     accepts the event if the tool is not anchored (floating tool).

    #     Parameters
    #     ----------
    #     e : dragEvent
    #         The drag event triggered by the user.

    #     Returns
    #     -------
    #     None.

    #     '''

    #     if self.is_floating():
    #         e.accept()

    def mouseMoveEvent(self, e):
        '''
        Reimplementation of the mouseMoveEvent default function. It provides a
        custom animation for the widget drag & drop action (left-click drag).

        Parameters
        ----------
        e : mouseEvent
            The mouse move event triggered by the user.

        Returns
        -------
        None.

        '''
    # Ignore the event if the tool is already anchored to parent widget
        if not self.isWindow(): return

        if e.buttons() == Qt.LeftButton:
        # Generate mime data and a pixmap for the drag & drop event
            icon = self.windowIcon()
            pixmap = icon.pixmap(icon.actualSize(QC.QSize(32, 32)))
            mimeData = QC.QMimeData()

        # Generate a drag event and execute it
            drag = QG.QDrag(self)
            drag.setMimeData(mimeData)
            drag.setPixmap(pixmap)
            drag.exec_(Qt.MoveAction)


    def closeEvent(self, event):
        '''
        Reimplementation of the closeEvent default function. It adds a question
        dialog to confirm the closing action. It returns True if the event is
        accepted, 0 otherwise.

        Parameters
        ----------
        event : closeEvent
            The close event triggered by the user.

        Returns
        -------
        bool
            Whether or not the close event is accepted.

        '''
        text = f'Close {self.windowTitle()}? Any unsaved data will be lost'
        choice = QW.QMessageBox.question(self, 'X-Min Learn', text,
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
    # accept() or ignore() are returned as boolean output, so that the event is
    # propagated to the parent widget. The parent can be None when closed as a
    # floating window or the MainTabWidget when closed from the tab. In the
    # latter case the MainTabWidget catches the event output and use that info
    # to "choose" to close or not close the corresponding tab.
        if choice == QW.QMessageBox.Yes:
            return event.accept()
        else:
            return event.ignore()


# Application preferences dialog
class Preferences(QW.QWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(Preferences, self).__init__()
        self.setWindowTitle('Preferences')
        self.setWindowIcon(QIcon('Icons/wrench.png'))
        self.resize(200, 500)
        self.setWindowFlags(Qt.Widget | Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.read_settings()
        self.init_ui()

    def read_settings(self):
        self._fontsize = pref.get_setting('main/fontsize', 10, type=int)
        self._dynSplitter = pref.get_setting('main/dynamic_splitter', False, type=bool)
        self._shareaxis = pref.get_setting('plots/shareaxis', True, type=bool)
        self._NTBsize = pref.get_setting('plots/NTBsize', 28, type=int)
        self._legendDec = pref.get_setting('plots/legendDec', 3, type=int)
        self._extendedLog = pref.get_setting('class/extLog', False, type=bool)
        self._trAreasCol = pref.get_setting('class/trAreasCol', (255,0,0), type=tuple)
        self._trAreasFill = pref.get_setting('class/trAreasFill', False, type=bool)
        self._trAreasSel = pref.get_setting('class/trAreasSel', (0,0,255), type=tuple)
        # and so on

    def write_settings(self):
        pref.edit_setting('main/fontsize', self._fontsize)
        pref.edit_setting('main/dynamic_splitter', self._dynSplitter)
        pref.edit_setting('plots/shareaxis', self._shareaxis)
        pref.edit_setting('plots/NTBsize', self._NTBsize)
        pref.edit_setting('plots/legendDec', self._legendDec)
        pref.edit_setting('class/extLog', self._extendedLog)
        pref.edit_setting('class/trAreasCol', self._trAreasCol)
        pref.edit_setting('class/trAreasFill', self._trAreasFill)
        pref.edit_setting('class/trAreasSel', self._trAreasSel)
        # and so on

    def init_ui(self):

        # Selection Color (next work)
        # rectprops = dict(facecolor='red', edgecolor='black',
        #                          alpha=0.2, fill=True)

    # GENERAL SETTINGS

        # Change app font size spin-box
        self.fontsize_spbox = QW.QSpinBox()
        self.fontsize_spbox.setRange(6, 18)
        self.fontsize_spbox.setValue(self._fontsize)

        # Enable dynamic Splitter checkbox
        self.dynSplit_cbox = QW.QCheckBox()
        self.dynSplit_cbox.setToolTip('Get a dynamic feedback when resizing widgets with a handlebar')
        self.dynSplit_cbox.setChecked(self._dynSplitter)

        # General Tab
        general_form = QW.QFormLayout()
        general_form.addRow('Set font size', self.fontsize_spbox)
        general_form.addRow('Dynamic Handlebars', self.dynSplit_cbox)
        self.generalTab = cObj.GroupScrollArea(general_form, 'General settings')

    # PLOTS SETTINGS

        # XMapsView and MinMapView shareAxis checkbox
        self.shareaxis_cbox = QW.QCheckBox()
        self.shareaxis_cbox.setToolTip('Zooms applied on X-Ray Maps Tab affect '\
                                       'Mineral Map Tab and vice versa')
        self.shareaxis_cbox.setChecked(self._shareaxis)

        # Navigation Toolbars buttons size slider --> influences the overall size of the NTBar
        self.NTBsize_slider = QW.QSlider(Qt.Horizontal)
        self.NTBsize_slider.setTickPosition(2) # ticks below
        self.NTBsize_slider.setMinimum(10)
        self.NTBsize_slider.setMaximum(50)
        self.NTBsize_slider.setSingleStep(10)
        self.NTBsize_slider.setSliderPosition(self._NTBsize)

        # Legend decimals number
        self.legendDec_spbox = QW.QSpinBox()
        self.legendDec_spbox.setToolTip('Number of decimals shown in legends.')
        self.legendDec_spbox.setRange(0, 5)
        self.legendDec_spbox.setValue(self._legendDec)

        # Plots Tab
        plots_form = QW.QFormLayout()
        plots_form.addRow('Shared zoom', self.shareaxis_cbox)
        plots_form.addRow('Toolbars size', self.NTBsize_slider)
        plots_form.addRow('Legend decimals', self.legendDec_spbox)
        self.plotsTab = cObj.GroupScrollArea(plots_form, 'Plots settings')


    # CLASSIFICATION SETTINGS

        # Extended Log checkbox
        self.extLog_cbox = QW.QCheckBox('Extended Model Log')
        self.extLog_cbox.setToolTip("Include advanced information inside custom model logs.")
        self.extLog_cbox.setChecked(self._extendedLog)

        # Training areas color button
        col_rgb = self._trAreasCol
        self.trAreasCol_btn = QW.QPushButton(cObj.RGBIcon(col_rgb), str(col_rgb))
        self.trAreasCol_btn.clicked.connect(self._changeIconColor)

        # Training areas filled checkbox
        self.trAreasFill_cbox = QW.QCheckBox('Filled')
        self.trAreasFill_cbox.setChecked(self._trAreasFill)

        # Training areas selection color button
        sel_rgb = self._trAreasSel
        self.trAreasSel_btn = QW.QPushButton(cObj.RGBIcon(sel_rgb), str(sel_rgb))
        self.trAreasSel_btn.clicked.connect(self._changeIconColor)


        # Model learner sub-group
        modLearn_group = cObj.GroupArea(self.extLog_cbox, 'Model Learner')

        # Training Areas sub-group
        trAreas_form = QW.QFormLayout()
        trAreas_form.addRow('Color', self.trAreasCol_btn)
        trAreas_form.addRow(self.trAreasFill_cbox)
        trAreas_form.addRow('Selection color', self.trAreasSel_btn)
        trAreas_group = cObj.GroupArea(trAreas_form, 'Training Areas')


        # Model Learner Tab
        class_vbox = QW.QVBoxLayout()
        class_vbox.addWidget(modLearn_group)
        class_vbox.addWidget(trAreas_group)
        self.classTab = cObj.GroupScrollArea(class_vbox, 'Classification settings')


    # TAB WIDGET
        self.TabWid = QW.QTabWidget()
        self.TabWid.addTab(self.generalTab, 'General')
        self.TabWid.addTab(self.plotsTab, 'Plots')
        self.TabWid.addTab(self.classTab, 'Classification')
        self.setStyleSheet('QTabBar {color: black;}')


    # PREFERENCES DIALOG BUTTONS

        # Save button
        self.save_btn = QW.QPushButton('OK')
        self.save_btn.clicked.connect(lambda: self.savePref(exitPref=True))

        # Cancel button
        self.cancel_btn = QW.QPushButton('Cancel')
        self.cancel_btn.clicked.connect(self.close)

        # Apply button
        self.apply_btn = QW.QPushButton('Apply')
        self.apply_btn.clicked.connect(self.savePref)

        # Default button
        self.default_btn = QW.QPushButton('Default')
        self.default_btn.clicked.connect(self.defaultPref)

        # Buttons Layout
        dialogBtns = QW.QHBoxLayout()
        dialogBtns.addWidget(self.save_btn, alignment=Qt.AlignLeft)
        dialogBtns.addWidget(self.cancel_btn, alignment=Qt.AlignLeft)
        dialogBtns.addWidget(self.apply_btn, alignment=Qt.AlignLeft)
        dialogBtns.addWidget(self.default_btn, alignment=Qt.AlignRight)

    # MAIN LAYOUT
        mainLayout = QW.QVBoxLayout()
        mainLayout.addWidget(self.TabWid)
        mainLayout.addLayout(dialogBtns)
        self.setLayout(mainLayout)






    # def enable_persRect(self, state):
    #     # Tab2 is a reference to the ClassImgTab
    #     Tab2 = self.parent.main_win.tabWidget.Tabs.widget(1)
    #     Tab2.rectSel.persistent = state

    def _changeIconColor(self):
        col = QW.QColorDialog.getColor()
        if col.isValid():
            rgb = tuple(col.getRgb()[:-1])
            self.sender().setIcon(cObj.RGBIcon(rgb))
            self.sender().setText(str(rgb))

    def set_fontSize(self):
        self._fontsize = self.fontsize_spbox.value()
        pref.setAppFont(self._fontsize)

    def dynamicSplitter(self):
        self._dynSplitter = self.dynSplit_cbox.isChecked()
        for i, spt in enumerate(cObj.SplitterGroup.instances):
            try:
                spt.setOpaqueResize(self._dynSplitter)
            except (ReferenceError, RuntimeError):
                del cObj.SplitterGroup.instances[i]

    def shareAxis(self):
        self._shareaxis = self.shareaxis_cbox.isChecked()
        xmapsview = self.parent._xmapstab.XMapsView
        minmapview = self.parent._minmaptab.MinMapView
        CF.shareAxis(xmapsview.ax, minmapview.ax, self._shareaxis)

    def set_NTbarSize(self): # deprecated! make it change after app restart
        self._NTBsize = self.NTBsize_slider.value()
        for i, ntb in enumerate(cObj.NavTbar.instances):
            try:
                ntb.setIconSize(QC.QSize(self._NTBsize, self._NTBsize))
            except (ReferenceError, RuntimeError):
                del cObj.NavTbar.instances[i]

    def set_legendDec(self):
        self._legendDec = self.legendDec_spbox.value()
        for i, leg in enumerate(cObj.CanvasLegend.instances):
            try:
                leg.setPrecision(self._legendDec)
                leg.update()
            except (ReferenceError, RuntimeError):
                del cObj.CanvasLegend.instances[i]


    def extendedLog(self):
        self._extendedLog = self.extLog_cbox.isChecked()

    def set_trAreasCol(self):
        rgb_str = self.trAreasCol_btn.text()
        rgb = tuple(map(int, rgb_str[1:-1].split(',')))
        self._trAreasCol = rgb

    def set_trAreasFill(self):
        self._trAreasFill = self.trAreasFill_cbox.isChecked()

    def set_trAreasSel(self):
        rgb_str = self.trAreasSel_btn.text()
        rgb = tuple(map(int, rgb_str[1:-1].split(',')))
        self._trAreasSel = rgb




    def defaultPref(self):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Do you want to reset all preferences to default? '\
                                         'Note: the changes will only take effect after restarting the app.',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            pref.clear_settings()
            self.close()

    def savePref(self, exitPref=False):
        #apply edits in current running app
        self.set_fontSize()
        self.dynamicSplitter()
        self.shareAxis()
        self.set_NTbarSize()
        self.set_legendDec()
        self.extendedLog()
        self.set_trAreasCol()
        self.set_trAreasFill()
        self.set_trAreasSel()


        # save edits in settings.ini file
        self.write_settings()
        if exitPref:
            self.close()



class Image2Ascii(QW.QWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(Image2Ascii, self).__init__()
        self.setWindowTitle('Grayscale to ASCII')
        self.setWindowIcon(QIcon('Icons/gear.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.loadedImgs = []

        self.init_ui()
        self.adjustSize()

    def init_ui(self):
    # Images import button
        self.import_btn = QW.QPushButton('Import Images')
        self.import_btn.clicked.connect(self.importImgs)

    # Images delete button
        self.del_btn = QW.QPushButton('Remove Images')
        self.del_btn.setStyleSheet('''background-color: red;
                                      font-weight: bold;
                                      color: white''')
        self.del_btn.clicked.connect(self.removeImgs)

    # Loaded images visualizer list
        self.loadedList = cObj.StyledListWidget()

    # Auto-load result checkbox
        self.autoLoad_cbox = QW.QCheckBox('Auto-load')
        self.autoLoad_cbox.setToolTip('Automatically load files after conversion.')
        self.autoLoad_cbox.setChecked(True)

    # Split multichannel images checkbox
        self.splitChannels_cbox = QW.QCheckBox('Split multi-channel images')
        self.splitChannels_cbox.setChecked(True)

    # Output extension combobox
        self.ext_combox = QW.QComboBox()
        self.ext_combox.addItems(['.gz', '.txt'])

    # Convert button
        self.convert_btn = QW.QPushButton('Convert')
        self.convert_btn.setStyleSheet('''background-color: rgb(50,205,50);
                                          font-weight: bold''')
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.convert)

    # Progress bar
        self.progBar = QW.QProgressBar()

    # Adjust Options Area Layout
        options_form = QW.QFormLayout()
        options_form.addRow(self.autoLoad_cbox)
        options_form.addRow(self.splitChannels_cbox)
        options_form.addRow('Output File Format', self.ext_combox)
        options_form.addRow(self.convert_btn)
        options_group = cObj.GroupArea(options_form, 'Options')

    # Adjust Main Layout
        main_grid = QW.QGridLayout()
        main_grid.addWidget(self.import_btn, 0, 0, 1, 1, alignment=Qt.AlignLeft)
        main_grid.addWidget(self.del_btn, 0, 1, 1, 1, alignment=Qt.AlignRight)
        main_grid.addWidget(options_group, 0, 2, 2, 1)
        main_grid.addWidget(self.loadedList, 1, 0, 1, 2)
        main_grid.addWidget(self.progBar, 2, 0, 1, 3)
        self.setLayout(main_grid)

    def importImgs(self):
        imgs, _ = QW.QFileDialog.getOpenFileNames(self, 'Import maps images',
                                                  pref.get_dirPath('in'),
                                                 '''TIF (*.tif; *.tiff)
                                                    BMP (*.bmp)
                                                    PNG (*.png)
                                                    JPEG (*.jpg; *.jpeg)''')
        if imgs:
            pref.set_dirPath('in', os.path.dirname(imgs[0]))
            for i in imgs:
                if i not in self.loadedImgs:
                    self.loadedList.addItem(i)
                    self.loadedImgs.append(i)
            self.convert_btn.setEnabled(True)

    def removeImgs(self):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Remove selected images?',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            selected = self.loadedList.selectedItems()
            for item in sorted(selected, reverse=True):
                row = self.loadedList.row(item)
                del self.loadedImgs[row]
                self.loadedList.takeItem(row)
                del item

            if len(self.loadedImgs) == 0:
                self.convert_btn.setEnabled(False)

    def convert(self):
        outdir = QW.QFileDialog.getExistingDirectory(self, "Select output directory")
        if outdir:
            self.convert_btn.setEnabled(False)
            self.progBar.setRange(0, len(self.loadedImgs))
            ext = str(self.ext_combox.currentText())
            errLog, outfiles = [], []

            for n, p in enumerate(self.loadedImgs):
                try:
                    arr = CF.map2ASCII(p)
                    if arr.ndim == 3 and self.splitChannels_cbox.isChecked():
                        channels = np.split(arr, arr.shape[-1], axis=2)
                        fname = CF.path2filename(p)
                        for n_ch, ch in enumerate(channels, start=1):
                            outpath = os.path.join(outdir, CF.extend_filename(fname, f'_ch{n_ch}', ext))
                            np.savetxt(outpath, np.squeeze(ch), delimiter=' ', fmt='%d')
                            outfiles.append(outpath) # prepare the out file to autoload
                    else:
                        fname = CF.path2filename(p)
                        outpath = os.path.join(outdir, fname + ext)
                        np.savetxt(outpath, arr, delimiter=' ', fmt='%d')
                        outfiles.append(outpath) # prepare the out file to autoload
                except Exception as e:
                    errLog.append((p,e))
                    os.remove(outpath)
                finally:
                    self.progBar.setValue(n+1)

            if len(errLog) == 0:
                 QW.QMessageBox.information(self, 'X-Min Learn',
                                           'All images were converted with success.')
            else:
                failed, errMsg = zip(*errLog)
                cObj.RichMsgBox(self, QW.QMessageBox.Warning, 'X-Min Learn',
                                f'The following images failed the convertion:\n\n{failed}',
                                detailedText='\n'.join(err.args[0] for err in errMsg))

            self.progBar.reset()
            self.convert_btn.setEnabled(True)
            if self.autoLoad_cbox.isChecked():
                self.parent._xmapstab.loadMaps(outfiles)


class Image2Minmap(QW.QWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(Image2Minmap, self).__init__()
        self.setWindowTitle('RGB to Mineral Map')
        self.setWindowIcon(QIcon('Icons/gear.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.minmap = None

        self.init_ui()
        self.adjustSize()

    def init_ui(self):
    # Image import button
        self.import_btn = QW.QPushButton('Load')
        self.import_btn.setToolTip('Load RGB image')
        self.import_btn.clicked.connect(self.importImg)

    # Image pathlabel
        self.imgPath = cObj.PathLabel()

    # Convert button
        self.convert_btn = QW.QPushButton('Convert')
        self.convert_btn.setStyleSheet('''background-color: rgb(50,205,50);
                                          font-weight: bold''')
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.convert)

    # Progress bar
        self.progBar = QW.QProgressBar()

    # Image visualizer canvas
        self.imgView = cObj.DiscreteClassCanvas(self, size=(5, 3.75))
        self.imgView.setMinimumSize(100,100)

    # Image visualizer NavTbar
        self.imgViewNtbar = cObj.NavTbar(self.imgView, self)
        self.imgViewNtbar.removeToolByIndex([3, 4, 8, 9, 10, 11])
        self.imgViewNtbar.fixHomeAction()

    # Image legend
        self.legend = cObj.CanvasLegend(self.imgView)

    # Auto-load result checkbox
        self.autoLoad_cbox = QW.QCheckBox('Auto-load')
        self.autoLoad_cbox.setToolTip('Automatically load map after conversion.')
        self.autoLoad_cbox.setChecked(True)

    # Save button
        self.save_btn = cObj.IconButton(r'Icons\save.png', 'Save')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_minmap)


    # Adjust Options Area Layout
        options_grid = QW.QGridLayout()
        options_grid.addWidget(self.import_btn, 0, 0, 1, 1, alignment=Qt.AlignLeft)
        options_grid.addWidget(self.convert_btn, 0, 1, 1, 1, alignment=Qt.AlignRight)
        options_grid.addWidget(self.imgPath, 1, 0, 1, 2)
        options_grid.addWidget(self.progBar, 2, 0, 1, 2)
        options_grid.addWidget(self.legend, 3, 0, 1, 2)
        options_grid.addWidget(self.autoLoad_cbox, 4, 0, 1, 1)
        options_grid.addWidget(self.save_btn, 4, 1, 1, 1)

    # Adjust View Area Layout
        view_vbox = QW.QVBoxLayout()
        view_vbox.addWidget(self.imgViewNtbar)
        view_vbox.addWidget(self.imgView)
        view_group = cObj.GroupArea(view_vbox)

    # Adjust main Layout
        main_hsplit = cObj.SplitterGroup((options_grid, view_group), (1, 3)) # use splitter layout instead
        mainLayout = QW.QHBoxLayout()
        mainLayout.addWidget(main_hsplit)
        self.setLayout(mainLayout)



    def importImg(self):

        if self.imgPath.get_fullpath() != '':
            choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         "Do you want to load a different image?",
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
            if choice == QW.QMessageBox.No : return

        img, _ = QW.QFileDialog.getOpenFileName(self, 'Import image',
                                                  pref.get_dirPath('in'),
                                                 '''TIF (*.tif; *.tiff)
                                                    BMP (*.bmp)
                                                    PNG (*.png)
                                                    JPEG (*.jpg; *.jpeg)''')
        if img:
            pref.set_dirPath('in', os.path.dirname(img))
            self.imgPath.set_fullpath(img, predict_display=True)
            self.convert_btn.setEnabled(True)


    def convert(self):
        img_path = self.imgPath.get_fullpath()
        if img_path == '' : return

    # Convert image to array
        arr = CF.map2ASCII(img_path)

    # Get array shape
        row, col, chan = arr.shape
    # Raise error if loaded image is not RGB(A)
        if chan < 3 or chan > 4:
            return QW.QMessageError.critical(self, 'X-Min Learn',
                                             'The selected image cannot be converted.')

    # Convert RGBA array to RGB
        elif chan == 4:
            arr = CF.rgba2rgb(arr)

    # Reshape the array to get unique RGB values
        arr = arr.reshape(-1, 3)

    # Get unique RGB values
        unq = np.unique(arr, axis=0)

    # Raise error if there are  more than (2**16)/2 = 32768 classes (uint16) # (can be enhanced)
        n_classes = len(unq)
        if n_classes > 32768:
            return QW.QMessageBox.critical(self, 'X-Min Learn'
                                           'The loaded image contains too many classes (more than 32768)')

    # Build a flattened minmap
        self.progBar.setRange(0, n_classes)
        self.minmap = np.empty((row*col), dtype='U8')
        for n, rgb in enumerate(unq):
            mask = (arr == rgb).all(axis=1)
            self.minmap[mask] = f'{n:05d}' # legend has to be ORDERED properly to match colors
            self.progBar.setValue(n+1)

    # Reshape the minmap to the orignal shape
        self.minmap = self.minmap.reshape(row, col)

    # Plot the map and update legend
        self.imgView.update_canvas(self.imgPath.get_displayName(), self.minmap,
                                   colors = unq.tolist())
        self.legend.update()

    # Enable the save button
        self.save_btn.setEnabled(True)

    # Reset progress bar
        self.progBar.reset()


    def save_minmap(self):

        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Export Mineral Map',
                                                        pref.get_dirPath('out'),
                                                        '''Compressed ASCII file (*.gz)
                                                           ASCII file (*.txt)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            np.savetxt(outpath, self.minmap, fmt='%s')

            if self.autoLoad_cbox.isChecked():
                self.parent._minmaptab.loadMaps((outpath,))

            QW.QMessageBox.information(self, 'X-Min Learn',
                                       'Mineral Map was converted and saved with success.')






class DummyMapsBuilder(QW.QWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super(DummyMapsBuilder, self).__init__()
        self.setWindowTitle('Dummy Maps Builder')
        self.setWindowIcon(QIcon('Icons/gear.png'))
        self.resize(600, 750)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.dummy_map = None

        self.init_ui()


    def init_ui(self):

    # Information about this tool label
        self.toolInfo_lbl = QW.QLabel('This tool allows to build artificial noisy maps '\
                                      'featuring a near-zero value on all of their pixels. '\
                                      'The values are randomized through a Gamma distribution '\
                                      'function. Such maps can be used as a placeholder for '\
                                      'missing mandatory maps when applying a pre-trained model.')
        self.toolInfo_lbl.setWordWrap(True)
        self.toolInfo_lbl.setSizePolicy(QW.QSizePolicy.Expanding,
                                        QW.QSizePolicy.Fixed)

    # Map width spinbox
        self.mapWidth_spbox = QW.QSpinBox()
        self.mapWidth_spbox.setRange(1, 10e8)
        self.mapWidth_spbox.setValue(100)

    # Map height spinbox
        self.mapHeight_spbox = QW.QSpinBox()
        self.mapHeight_spbox.setRange(1, 10e8)
        self.mapHeight_spbox.setValue(100)

    # Shape of gamma distribution spinbox
        self.shape_spbox = QW.QDoubleSpinBox()
        self.shape_spbox.setRange(0.1, 100.)
        self.shape_spbox.setDecimals(2)
        self.shape_spbox.setSingleStep(0.1)
        self.shape_spbox.setValue(1.5)
        self.shape_spbox.setToolTip('The shape of the gamma distribution function.')

    # Scale of gamma distribution spinbox
        self.scale_spbox = QW.QDoubleSpinBox()
        self.scale_spbox.setRange(0.1, 100.)
        self.scale_spbox.setDecimals(2)
        self.scale_spbox.setSingleStep(0.1)
        self.scale_spbox.setValue(1.)
        self.scale_spbox.setToolTip('The scale of the gamma distribution function.')

    # Generate button
        self.generate_btn = QW.QPushButton('Generate')
        self.generate_btn.clicked.connect(self.generate_dummyMap)

    # Save button
        self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'Save')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_dummyMap)

    # Preview histogram canvas + Navigation Toolbar
        self.histChart = cObj.HistogramCanvas(self, size=(5.6, 7.5))
        self.histChart.setMinimumSize(100, 100)
        self.histChart_NTbar = cObj.NavTbar(self.histChart, self)
        self.histChart_NTbar.removeToolByIndex([3, 4, 8, 9])

    # Adjust main layout
        plot_vbox = QW.QVBoxLayout()
        plot_vbox.addWidget(self.histChart_NTbar)
        plot_vbox.addWidget(self.histChart)

        gridLayout = QW.QGridLayout()
        gridLayout.addWidget(QW.QLabel('Map Width'), 0, 0)
        gridLayout.addWidget(self.mapWidth_spbox, 0, 1)
        gridLayout.addWidget(QW.QLabel('Map Height'), 1, 0)
        gridLayout.addWidget(self.mapHeight_spbox, 1, 1)
        gridLayout.addWidget(QW.QLabel('Function shape'), 2, 0)
        gridLayout.addWidget(self.shape_spbox, 2, 1)
        gridLayout.addWidget(QW.QLabel('Function scale'), 3, 0)
        gridLayout.addWidget(self.scale_spbox, 3, 1)
        gridLayout.addWidget(self.generate_btn, 4, 0)
        gridLayout.addWidget(self.save_btn, 4, 1)
        gridLayout.addLayout(plot_vbox, 0, 2, 5, 1)
        gridLayout.setColumnStretch(2, 1)

        mainLayout = QW.QVBoxLayout()
        mainLayout.addWidget(self.toolInfo_lbl)
        mainLayout.addWidget(cObj.GroupArea(gridLayout))
        self.setLayout(mainLayout)


    def generate_dummyMap(self):
    # Gather the parameters
        w = self.mapWidth_spbox.value()
        h = self.mapHeight_spbox.value()
        shp = self.shape_spbox.value()
        scl = self.scale_spbox.value()
    # Generate the map
        self.dummy_map = np.random.gamma(shp, scl, size=(h, w)).round()
    # Refresh the histogram
        self.histChart.update_canvas('', self.dummy_map)
    # Enable save button
        self.save_btn.setEnabled(True)

    def save_dummyMap(self):
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save Map',
                                                    pref.get_dirPath('out'),
                                                    '''Compressed ASCII file (*.gz)
                                                       ASCII file (*.txt)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            np.savetxt(outpath, self.dummy_map, fmt='%d')






class SubSampleDataset(QW.QWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super(SubSampleDataset, self).__init__()
        self.setWindowTitle('Sub-sample Dataset')
        self.setWindowIcon(QIcon('Icons/gear.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.loadedDataset = None
        self.classCounter = dict()

        self.init_ui()
        self.adjustSize()

    def init_ui(self):
        # Import Dataset Button
        self.import_btn = QW.QPushButton(QIcon('Icons/load.png'), 'Import')
        self.import_btn.clicked.connect(self.import_dataset)

        # Original dataset decimal character selector
        self.in_CSVdec = cObj.DecimalPointSelector()
        in_CSVdec_form = QW.QFormLayout()
        in_CSVdec_form.addRow('CSV decimal point', self.in_CSVdec)

        # Original dataset path
        self.dataset_path = cObj.PathLabel('', 'No dataset loaded')

        # Sub-sampled dataset decimal character selector
        self.out_CSVdec = cObj.DecimalPointSelector()
        out_CSVdec_form = QW.QFormLayout()
        out_CSVdec_form.addRow('CSV decimal point', self.out_CSVdec)

        # Sub-sampled dataset separator selector
        self.out_CSVsep = cObj.SeparatorSymbolSelector()
        out_CSVsep_form = QW.QFormLayout()
        out_CSVsep_form.addRow('CSV separator', self.out_CSVsep)

        # Save button
        self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'Save')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_subSample)

        # Imported dataset info area
        self.dataset_info = QW.QTextEdit()
        self.dataset_info.setReadOnly(True)

        # Class Selector Twin List Widgets
        self.orig_class = cObj.StyledListWidget()
        self.subSampled_class = cObj.StyledListWidget()
        for qList in (self.orig_class, self.subSampled_class):
            qList.setAcceptDrops(True)
            qList.setDragEnabled(True)
            qList.setDefaultDropAction(Qt.MoveAction)
            qList.itemClicked.connect(self.show_classCount)

        orig_classGA = cObj.GroupArea(self.orig_class,
                                      'Original Dataset classes')
        subSampled_classGA = cObj.GroupArea(self.subSampled_class,
                                            'Sub-sampled Dataset classes')

        # Help Label
        self.help_lbl = QW.QLabel('Drag & Drop to include/exclude classes')
        # self.help_lbl.setMaximumHeight(20)
        self.help_lbl.setSizePolicy(QW.QSizePolicy.Expanding,
                                    QW.QSizePolicy.Fixed)

        # Class Counter Label
        self.counter_lbl = QW.QLabel()
        # self.counter_lbl.setMaximumHeight(20)
        self.counter_lbl.setSizePolicy(QW.QSizePolicy.Expanding,
                                       QW.QSizePolicy.Fixed)

        # Adjust Layout
        inFile_vbox = QW.QVBoxLayout()
        inFile_vbox.addWidget(self.import_btn)
        inFile_vbox.addLayout(in_CSVdec_form)
        inFile_vbox.addWidget(self.dataset_path)
        inFile_GA = cObj.GroupArea(inFile_vbox, 'Original Dataset')

        outFile_vbox = QW.QVBoxLayout()
        outFile_vbox.addLayout(out_CSVdec_form)
        outFile_vbox.addLayout(out_CSVsep_form)
        outFile_vbox.addWidget(self.save_btn)
        outFile_GA = cObj.GroupArea(outFile_vbox, 'Sub-sampled Dataset')

        upper_hbox = QW.QHBoxLayout()
        upper_hbox.addWidget(inFile_GA, alignment = Qt.AlignLeft)
        upper_hbox.addWidget(outFile_GA, alignment = Qt.AlignRight)

        central_hbox = QW.QHBoxLayout()
        central_hbox.addWidget(orig_classGA, alignment = Qt.AlignLeft)
        central_hbox.addWidget(subSampled_classGA, alignment = Qt.AlignRight)

        lower_hbox = QW.QHBoxLayout()
        lower_hbox.addWidget(self.help_lbl, alignment = Qt.AlignLeft)
        lower_hbox.addWidget(self.counter_lbl, alignment = Qt.AlignRight)

        mainLayout = QW.QVBoxLayout()
        mainLayout.addLayout(upper_hbox, 2)
        mainLayout.addWidget(self.dataset_info, 3)
        mainLayout.addLayout(central_hbox, 3)
        mainLayout.addLayout(lower_hbox, 1)
        self.setLayout(mainLayout)

    def import_dataset(self):
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Import Dataset',
                                                 pref.get_dirPath('in'),
                                                 'Comma Separated Value (*.csv)')
        if path:
            pref.set_dirPath('in', os.path.dirname(path))
            dec = self.in_CSVdec.currentText()
            self.loadedDataset = cObj.CsvChunkReader(dec, pBar=True).read(path)

            ylab = self.loadedDataset.columns[-1]
            self.loadedDataset[ylab] = self.loadedDataset[ylab].astype(str) # be sure that last column is labels
            cnt = self.loadedDataset[ylab].value_counts()
            self.classCounter = dict(zip(cnt.index, cnt))

            self.update_datasetPath(path)
            self.update_datasetInfo()
            self.update_classLists()
            self.counter_lbl.clear()
            self.save_btn.setEnabled(True)

    def update_datasetPath(self, path):
        self.dataset_path.set_displayName(CF.path2filename(path))
        self.dataset_path.set_fullpath(path)

    def update_datasetInfo(self):
        infoText = f'DATAFRAME PREVIEW\n\n{repr(self.loadedDataset)}'
        self.dataset_info.setText(infoText)

    def update_classLists(self):
        self.subSampled_class.clear()
        self.orig_class.clear()
        self.orig_class.addItems(sorted(self.classCounter.keys()))

    def show_classCount(self, item):
        # item = self.sender().currentItem()
        if item is not None:
            current_class = item.text()
            count = self.classCounter[current_class]
            self.counter_lbl.setText(f'{current_class} = {count}')


    def save_subSample(self):
        item_count = self.subSampled_class.count()
        if item_count == 0:
            QW.QMessageBox.critical(self, 'X-Min Learn',
            'No classes selected: the sub-sampled dataset would be empty')
            return

        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save sub-sampled dataset',
                                                    pref.get_dirPath('out'),
                                                    'Comma Separated Values (*.csv)')
        if outpath:
            pref.set_dirPath('in', os.path.dirname(outpath))
            progBar = cObj.PopUpProgBar(self, 1, 'Saving dataset', cancel=False)
            progBar.setValue(0)
            ylab = self.loadedDataset.columns[-1]
            targets = [self.subSampled_class.item(i).text() for i in range(item_count)]
            subSample = self.loadedDataset[self.loadedDataset[ylab].isin(targets)]
            dec = self.out_CSVdec.currentText()
            sep = self.out_CSVsep.currentText()
            subSample.to_csv(outpath, sep=sep, index=False, decimal=dec)
            progBar.setValue(1)
            QW.QMessageBox.information(self, 'X-Min Learn',
                                       'Dataset saved with success')


class MergeDatasets(QW.QWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super(MergeDatasets, self).__init__()
        self.setWindowTitle('Merge Datasets')
        self.setWindowIcon(QIcon('Icons/gear.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.loadedDS = []
        self.merged = None

        self.init_ui()
        self.adjustSize()

    def init_ui(self):

        # Add datasets button
        self.addDS_btn = cObj.IconButton('Icons/generic_add.png')
        self.addDS_btn.clicked.connect(self.add_datasets)

        # Input datasets CSV decimal selector button
        self.in_CSVdec = cObj.DecimalPointSelector()

        # Input datasets list viewer
        self.loadedDS_area = cObj.StyledListWidget(ext_selection=False)
        self.loadedDS_area.itemClicked.connect(self.show_inDSpreview)

        # Remove selected datasets button
        self.delDS_btn = cObj.IconButton('Icons/generic_del.png')
        self.delDS_btn.setToolTip('Remove selected datasets.')
        self.delDS_btn.clicked.connect(self.remove_datasets)

        # Merge datasets button
        self.mergeDS_btn = QW.QPushButton('Merge')
        self.mergeDS_btn.setStyleSheet('''background-color: rgb(50,205,50);
                                          font-weight: bold''')
        self.mergeDS_btn.clicked.connect(self.merge_datasets)

        # Selected datasets preview area
        self.inDS_info = QW.QTextEdit()
        self.inDS_info.setReadOnly(True)

        # Merged dataset preview area
        self.merge_info = QW.QTextEdit()
        self.merge_info.setReadOnly(True)

        # Merged dataset decimal character selector
        self.out_CSVdec = cObj.DecimalPointSelector()

        # Merged dataset separator selector
        self.out_CSVsep = cObj.SeparatorSymbolSelector()

        # Save merged dataset button
        self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'Save')
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_mergedDS)

        # Adjust left vBox Layout
        importDS_form = QW.QFormLayout()
        importDS_form.addRow('Add datasets', self.addDS_btn)
        importDS_form.addRow('CSV decimal point', self.in_CSVdec)
        importDS_form.setFieldGrowthPolicy(importDS_form.FieldsStayAtSizeHint)

        delMerge_hbox = QW.QHBoxLayout()
        delMerge_hbox.addWidget(self.delDS_btn, alignment=Qt.AlignLeft)
        delMerge_hbox.addWidget(self.mergeDS_btn, alignment=Qt.AlignRight)

        inDS_vbox = QW.QVBoxLayout()
        inDS_vbox.addLayout(importDS_form)
        inDS_vbox.addWidget(self.loadedDS_area)
        inDS_vbox.addLayout(delMerge_hbox)
        inDS_vbox.addWidget(cObj.GroupArea(self.inDS_info, 'Selected dataset preview'), 2)

        # Adjust right vBox Layout
        outCSV_form = QW.QFormLayout()
        outCSV_form.addRow('CSV decimal point', self.out_CSVdec)
        outCSV_form.addRow('CSV separator', self.out_CSVsep)
        outCSV_form.setFormAlignment(Qt.AlignLeft)
        outCSV_form.setFieldGrowthPolicy(outCSV_form.FieldsStayAtSizeHint)

        outDS_vbox = QW.QVBoxLayout()
        outDS_vbox.addWidget(cObj.GroupArea(self.merge_info, 'Merged dataset preview'), 2)
        outDS_vbox.addLayout(outCSV_form)
        outDS_vbox.addWidget(self.save_btn)

        # Adjust main layout
        mainLayout = QW.QHBoxLayout()
        mainLayout.addLayout(inDS_vbox)
        mainLayout.addLayout(outDS_vbox)
        self.setLayout(mainLayout)

    def read_datasets(self, paths):
        datasets = []
        dec = self.in_CSVdec.currentText()
        one_file = not (len(paths) - 1)

        if not one_file:
            pBar = cObj.PopUpProgBar(self, len(paths), 'Reading datasets')

        for n, p in enumerate(paths, start=1):
            d = cObj.CsvChunkReader(dec, pBar=one_file).read(p)
            datasets.append(d)
            if not one_file: pBar.setValue(n)
        return datasets

    def add_datasets(self):
        paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Import Datasets',
                                                   pref.get_dirPath('in'),
                                                 'Comma Separated Value (*.csv)')
        if paths:
            pref.set_dirPath('in', os.path.dirname(paths[0]))
            for p in paths[:]:
                matching = self.loadedDS_area.findItems(p, Qt.MatchExactly)
                if len(matching): paths.remove(p) # len as boolean
                else: self.loadedDS_area.addItem(p)

            if len(paths):
                datasets = self.read_datasets(paths)
                self.loadedDS.extend(datasets)

    def show_inDSpreview(self, item):
        self.inDS_info.clear()
        if item is not None:
            dataset = self.loadedDS[self.loadedDS_area.row(item)]
            infoText = f'DATAFRAME PREVIEW\n\n{repr(dataset)}'
            self.inDS_info.setText(infoText)

    def remove_datasets(self):
        selected = self.loadedDS_area.selectedItems()
        if len(selected):
            # selectedItems can only return a list of 1 element,
            # so we extract it with '[0]'
            idx = self.loadedDS_area.row(selected[0])
            self.loadedDS_area.takeItem(idx)
            del self.loadedDS[idx]

    def merge_datasets(self):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Do you want to merge the imported datasets?',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.Yes)
        if choice == QW.QMessageBox.Yes:
            # Check that datasets are present
            if len(self.loadedDS) == 0:
                self.merged = None
                self.merge_info.clear()
                self.save_btn.setEnabled(False)
                return

            # Check for datasets column NAME fitting
            # sorted allows to accept same column names with different order
            col_names = sorted(CF.most_frequent([df.columns.to_list() for df in self.loadedDS]))

            for i, ds in enumerate(self.loadedDS):
                if sorted(ds.columns.to_list()) != col_names:
                    fname = self.loadedDS_area.item(i).text()
                    QW.QMessageBox.critical(self, 'X-Min Learn',
                                            'Unable to fit the columns.'\
                                            f'Check the following file:\n{fname}')
                    return

            # concatenate dataframes
            merged = concat(self.loadedDS, ignore_index=True)
            self.merged = merged
            # show merged dataset preview
            infoText = f'MERGED DATAFRAME PREVIEW\n\n{repr(merged)}'
            self.merge_info.setText(infoText)
            # enable save button
            self.save_btn.setEnabled(True)

    def save_mergedDS(self):
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save merged dataset',
                                                    pref.get_dirPath('out'),
                                                   'Comma Separated Values (*.csv)')

        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            sep = self.out_CSVsep.currentText()
            dec = self.out_CSVdec.currentText()
            self.merged.to_csv(outpath, sep=sep, index=False, decimal=dec)
            QW.QMessageBox.information(self, 'X-Min Learn',
                                       'The datasets were succesfully merged.')







class MineralClassifier(DraggableTool):
    '''
    One of the main tools of X-Min Learn, that allows the classification of
    input maps using pre-trained eager ML models, ROI-based lazy ML algorithms
    and/or clustering algorithms. It also allows sub-phase classifications.
    '''

    def __init__(self):
        '''
        Constructor.

        '''
        super(MineralClassifier, self).__init__()

    # Set tool title and icon
        self.setWindowTitle('Mineral Classifier')
        self.setWindowIcon(QIcon(r'Icons/classify.png'))

    # Initialize main attributes
        self._mask = None
        self._nodata_color = [0, 0, 0]

    # Initialize classification state and attributes
        self._isBusyClassifying = False
        self._current_classifier = None

    # Set GUI
        self._init_ui()
        self.adjustSize()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self):
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                                DATA PANEL 
#  -------------------------------------------------------------------------  #
        
    # Input maps selector 
        self.inmaps_selector = cObj.SampleMapsSelector('inmaps')

    # Remove mask button (StyledButton)
        self.del_mask_btn = cObj.StyledButton(QIcon(r'Icons/clear.png'))
        self.del_mask_btn.setToolTip('Remove current mask')

    # Loaded mask (Path Label)
        self.mask_pathlbl = cObj.PathLabel(full_display=False)

    # Load mask from file choice (RadioButton) [Default choice]
        self.mask_radbtn_1 = QW.QRadioButton('Load from file')
        self.mask_radbtn_1.setStyleSheet(pref.SS_radioButton)
        self.mask_radbtn_1.setChecked(True)
    
    # Load mask from file (Styled Button)
        self.load_mask_btn = cObj.StyledButton(QIcon(r'Icons/import.png'), 
                                               'Load')
        
    # Get mask from class choice (RadioButton)
        self.mask_radbtn_2 = QW.QRadioButton('Get from class')
        self.mask_radbtn_2.setStyleSheet(pref.SS_radioButton)

    # Minmap selector to get mask from (Auto Update Combobox)
        self.minmap_combox = cObj.AutoUpdateComboBox()
        self.minmap_combox.setEnabled(False)

    # Mineral Class selector to get mask from (Combobox)
        self.class_combox = cObj.StyledComboBox()
        self.class_combox.setEnabled(False)

    # Mineral maps list (Tree Widget)
        self.minmaps_list = QW.QTreeWidget()
        self.minmaps_list.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self.minmaps_list.setHeaderHidden(True)
        self.minmaps_list.setStyleSheet(pref.SS_menu)
        self.minmaps_list.setMinimumHeight(150)

    # Save mineral map (Styled Button)
        self.save_minmap_btn = cObj.StyledButton(QIcon(r'Icons/save_as.png'))
        self.save_minmap_btn.setToolTip('Save mineral map as...')

    # Delete mineral map (Styled Button)
        self.del_minmap_btn = cObj.StyledButton(QIcon(r'Icons/remove.png'))
        self.del_minmap_btn.setToolTip('Remove mineral map')

    # Mineral maps legend (Legend)
        self.legend = cObj.Legend()
        self.legend.setSelectionMode(QW.QAbstractItemView.SingleSelection)

    # Mineral maps bar canvas (BarCanvas)
        self.barplot = plots.BarCanvas(orientation='h', size=(3.6, 6.4),
                                       wheel_zoom=False, wheel_pan=False)
        self.barplot.setMinimumSize(200, 350)
    
    # Mineral maps bar canvas navigation toolbar (Navigation Toolbar)
        self.barplot_navtbar = plots.NavTbar(self.barplot, self, coords=False)
        self.barplot_navtbar.removeToolByIndex(list(range(2, 10)))

    # Show labels percent in bar canvas Navigation Toolbar (Action)
        self.show_lbl_action = QW.QAction(QIcon(r'Icons/labelize.png'),
                                          'Show amounts', self.barplot_navtbar)
        self.show_lbl_action.setCheckable(True)
        self.barplot_navtbar.insertAction(10, self.show_lbl_action)

    # Silhouette canvas (Silhouette Canvas)
        self.silscore_canvas = plots.SilhouetteCanvas(wheel_zoom=False, 
                                                      wheel_pan=False)
        self.silscore_canvas.setMinimumHeight(450)

    # Silhouette canvas navigation toolbar (Navigation Toolbar)
        self.silscore_navtbar = plots.NavTbar(self.silscore_canvas, self, 
                                              coords=False)
        self.silscore_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Silhouette average score (QLabel)
        self.silscore_lbl = cObj.FramedLabel('None')
        
    # Calinski-Harabasz Index (CHI) score (QLabel)
        self.chiscore_lbl = cObj.FramedLabel('None')

    # Davies-Bouldin Index (DBI) score (QLabel)
        self.dbiscore_lbl = cObj.FramedLabel('None')

#  -------------------------------------------------------------------------  #
#                              CLASSIFIER PANEL 
#  -------------------------------------------------------------------------  #

    # Classifier panel (Styled Tab Widget)
        self.classifier_tabwid = cObj.StyledTabWidget()
        self.classifier_tabwid.tabBar().setDocumentMode(True)
        self.classifier_tabwid.tabBar().setExpanding(True)
        self.pre_train_tab = self.PreTrainedClassifierTab()
        self.roi_based_tab = self.RoiBasedClassifierTab()
        self.unsuperv_tab = self.UnsupervisedClassifierTab()
        self.classifier_tabwid.addTab(self.pre_train_tab, title='Pre-trained')
        self.classifier_tabwid.addTab(self.roi_based_tab, title='ROI-based')
        self.classifier_tabwid.addTab(self.unsuperv_tab, title='Unsupervised')

    # Classification progress bar (Descriptive ProgressBar)
        self.progbar = cObj.DescriptiveProgressBar()

    # Classify button (Styled Button)
        self.classify_btn = cObj.StyledButton(text='CLASSIFY',
                                              bg_color=pref.BTN_GREEN)
        self.classify_btn.setEnabled(False)

    # Interrupt classification process button (StyledButton)
        self.stop_btn = cObj.StyledButton(text='STOP', bg_color=pref.BTN_RED)

#  -------------------------------------------------------------------------  #
#                                VIEWER PANEL 
#  -------------------------------------------------------------------------  #

    # Maps viewer (Image Canvas)
        self.maps_viewer = plots.ImageCanvas()
        self.maps_viewer.setMinimumWidth(250)

    # Viewer Navigation Toolbar (Navigation Toolbar)
        self.viewer_navtbar = plots.NavTbar(self.maps_viewer, self)
        self.viewer_navtbar.fixHomeAction()
        self.viewer_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Confidence value input (SpinBox)
        self.conf_spbox = QW.QSpinBox()
        self.conf_spbox.setToolTip('Confidence')
        self.conf_spbox.setRange(0, 100)
        self.conf_spbox.setSingleStep(1)
        self.conf_spbox.setValue(50)
        self.conf_spbox.setEnabled(False)

    # Confidence Label
        conf_lbl = QW.QLabel('Confidence threshold')
        conf_lbl.setSizePolicy(QW.QSizePolicy.Ignored, QW.QSizePolicy.Fixed)

    # Confidence slider (Slider)
        self.conf_slider = QW.QSlider(Qt.Horizontal)
        self.conf_slider.setSizePolicy(QW.QSizePolicy.MinimumExpanding,
                                       QW.QSizePolicy.Fixed)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setTracking(False)
        self.conf_slider.setSliderPosition(50)
        self.conf_slider.setEnabled(False)

#  -------------------------------------------------------------------------  #
#                                 LAYOUT 
#  -------------------------------------------------------------------------  #
        
    # Data panel layout
        
        # - Input data
        self.inmaps_selector.setContentsMargins(9, 9, 9, 9)

        # - Mask data
        mask_data_grid = QW.QGridLayout()
        mask_data_grid.setAlignment(Qt.Alignment(Qt.AlignLeft | Qt.AlignTop))
        mask_data_grid.setVerticalSpacing(15)
        mask_data_grid.addWidget(self.del_mask_btn, 0, 0)
        mask_data_grid.addWidget(self.mask_pathlbl, 0, 1)
        mask_data_grid.setRowMinimumHeight(1, 20)
        mask_data_grid.addWidget(self.mask_radbtn_1, 2, 0, 1, -1)
        mask_data_grid.addWidget(self.load_mask_btn, 3, 1)
        mask_data_grid.addWidget(self.mask_radbtn_2, 4, 0, 1, -1)
        mask_data_grid.addWidget(QW.QLabel('Select mineral map'), 5, 1)
        mask_data_grid.addWidget(self.minmap_combox, 6, 1)
        mask_data_grid.addWidget(QW.QLabel('Select class'), 7, 1)
        mask_data_grid.addWidget(self.class_combox, 8, 1)
        mask_data_grid.setColumnStretch(1, 2)

        # - Output data
        barplot_vbox = QW.QVBoxLayout()
        barplot_vbox.addWidget(self.barplot_navtbar)
        barplot_vbox.addWidget(self.barplot)

        scores_grid = QW.QGridLayout()
        scores_grid.setVerticalSpacing(10)
        scores_grid.addWidget(self.silscore_navtbar, 0, 0, 1, -1)
        scores_grid.addWidget(self.silscore_canvas, 1, 0, 1, -1)
        scores_grid.setRowMinimumHeight(2, 10)
        scores_grid.addWidget(QW.QLabel('Average silhouette score'), 3, 0)
        scores_grid.addWidget(self.silscore_lbl, 3, 1)
        scores_grid.addWidget(QW.QLabel('Calinski-Harabasz Index'), 4, 0)
        scores_grid.addWidget(self.chiscore_lbl, 4, 1)
        scores_grid.addWidget(QW.QLabel('Davies-Bouldin Index'), 5, 0)
        scores_grid.addWidget(self.dbiscore_lbl, 5, 1)

        graph_tabwid = cObj.StyledTabWidget()
        graph_tabwid.addTab(self.legend, QIcon(r'Icons/legend.png'), None)
        graph_tabwid.addTab(barplot_vbox, QIcon(r'Icons/plot.png'), None)
        graph_tabwid.addTab(scores_grid, QIcon(r'Icons/scores.png'), None)
        graph_tabwid.setTabToolTip(0, 'Legend')
        graph_tabwid.setTabToolTip(1, 'Bar plot')
        graph_tabwid.setTabToolTip(2, 'Clustering scores')

        output_data_grid = QW.QGridLayout()
        output_data_grid.addWidget(self.minmaps_list, 0, 0, 1, -1)
        output_data_grid.addWidget(self.save_minmap_btn, 1, 1)
        output_data_grid.addWidget(self.del_minmap_btn, 1, 2)
        output_data_grid.addWidget(graph_tabwid, 2, 0, 1, -1)
        output_data_grid.setColumnStretch(0, 2)
        output_data_grid.setColumnStretch(1, 1)
        output_data_grid.setColumnStretch(2, 1)
        
        # - Panel layout
        self.data_tabwid = cObj.StyledTabWidget()
        self.data_tabwid.tabBar().setExpanding(True)
        self.data_tabwid.tabBar().setDocumentMode(True)
        self.data_tabwid.addTab(self.inmaps_selector, QIcon(r'Icons/inmap.png'), 
                                title='Input maps')
        self.data_tabwid.addTab(mask_data_grid, QIcon(r'Icons/mask.png'), 
                                title='Mask')
        self.data_tabwid.addTab(output_data_grid, QIcon(r'Icons/minmap.png'), 
                                title='Output maps')
        data_group = cObj.CollapsibleArea(self.data_tabwid, 'Data panel',
                                          collapsed=False)

    # Classifier panel layout
        class_grid = QW.QGridLayout()
        class_grid.addWidget(self.classifier_tabwid, 0, 0, 1, -1)
        class_grid.addWidget(self.progbar, 1, 0, 1, -1)
        class_grid.addWidget(self.classify_btn, 2, 0)
        class_grid.addWidget(self.stop_btn, 2, 1)
        class_group = cObj.CollapsibleArea(class_grid, 'Classifier panel')

    # Viewer panel layout
        viewer_grid = QW.QGridLayout()
        viewer_grid.addWidget(self.viewer_navtbar, 0, 0, 1, -1)
        viewer_grid.addWidget(self.maps_viewer, 1, 0, 1, -1)
        viewer_grid.addWidget(conf_lbl, 2, 0, 1, -1)
        viewer_grid.addWidget(self.conf_spbox, 3, 0)
        viewer_grid.addWidget(self.conf_slider, 3, 1)
        viewer_group = cObj.GroupArea(viewer_grid, 'Viewer panel')

    # Main layout
        left_vbox = QW.QVBoxLayout()
        left_vbox.setSpacing(30)
        left_vbox.addWidget(class_group)
        left_vbox.addWidget(data_group)
        left_vbox.addStretch(1)
        left_scroll = cObj.GroupScrollArea(left_vbox, frame=False)

        main_layout = cObj.SplitterLayout()
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(viewer_group)
        self.setLayout(main_layout)


    def _connect_slots(self):
        '''
        Signals-slots connector.

        '''
    # Reset canvas and enable/disable classify button if input data is updated
        self.inmaps_selector.mapsDataChanged.connect(
            self.maps_viewer.clear_canvas)
        self.inmaps_selector.mapsDataChanged.connect(
            self.updateClassifyButtonState)

    # Actions to be performed when an input map is clicked
        self.inmaps_selector.mapClicked.connect(self.showInputMap)

    # Enable/disable mask radio buttons actions
        self.mask_radbtn_1.toggled.connect(self.onMaskRadioButtonToggled)
        self.mask_radbtn_2.toggled.connect(self.onMaskRadioButtonToggled)

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

    # Save and remove mineral maps button actions
        self.save_minmap_btn.clicked.connect(self.saveMineralMap)
        self.del_minmap_btn.clicked.connect(self.removeMineralMap)

    # Connect legend signals (change item color, rename item, highlight item)
        self.legend.colorChangeRequested.connect(self.changeClassColor)
        self.legend.itemRenameRequested.connect(self.renameClass)
        self.legend.itemHighlightRequested.connect(self.highlightClass)

    # Show classes amounts in the bar plot
        self.show_lbl_action.toggled.connect(self.barplot.show_amounts)

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

    
    def onMaskRadioButtonToggled(self, toggled: bool):
        '''
        Manage the GUI visualization of the input mask options. When one option
        is toggled, the other is disabled.

        Parameters
        ----------
        toggled : bool
            State of radio button.

        '''
        if self.sender() == self.mask_radbtn_1:
            self.load_mask_btn.setEnabled(toggled)
            self.minmap_combox.setEnabled(not toggled)
            self.class_combox.setEnabled(not toggled)
        
        else:
            self.load_mask_btn.setEnabled(not toggled)
            self.minmap_combox.setEnabled(toggled)
            self.class_combox.setEnabled(toggled)


    def updateMineralMapsCombox(self):
        '''
        Populate the combobox that allows the seletion of a mineral map for 
        extracting a mask.

        '''
        count = self.minmaps_list.topLevelItemCount()
        items = [self.minmaps_list.topLevelItem(idx) for idx in range(count)]
        self.minmap_combox.updateItems([i.text(0) for i in items])


    def updateClassesCombox(self, idx:int):
        '''
        Populate the combobox that allows the selection of a mineral phase to 
        use as input mask.

        Parameters
        ----------
        idx : int
            The choosen mineral map index.

        '''
        mmap = self.minmaps_list.topLevelItem(idx)
        classes = mmap.get('data').get_phases()
        self.class_combox.clear()
        self.class_combox.addItems(classes)


    def updateClassifyButtonState(self):
        '''
        Toggle on/off the CLASSIFY button.

        '''
        enabled = self.inmaps_selector.itemCount()
        self.classify_btn.setEnabled(enabled)


    def showMapsViewerContextMenu(self, point: QC.QPoint):
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
        menu.exec(QG.QCursor.pos())


    def showInputMap(self, item: cObj.DataObject | None):
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

    # Get data required to plot the map 
        if self._mask is None:
            array = item.get('data').map
        else:
            array = item.get('data').get_masked(self._mask.mask)
            
        map_name = item.get('name')
        sample_name = self.inmaps_selector.sample_combox.currentText()
        title = f'{sample_name} - {map_name}'

    # Disable confidence spinbox and slider
        self.conf_spbox.setEnabled(False)
        self.conf_slider.setEnabled(False)

    # Plot the map
        self.maps_viewer.draw_heatmap(array, title)


    def loadMaskFromFile(self):
        '''
        Load an input mask from file.

        '''
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Load mask',
                                                 pref.get_dirPath('in'),
                                                 '''Mask (*.msk)
                                                 Text file (*.txt)''')

        if path:
            pref.set_dirPath('in', os.path.dirname(path[0]))
            
            try:
                mask = Mask.load(path)     
            except Exception as e:
                mask = None 
                cObj.RichMsgBox(self, QW.QMessageBox.Critical, 
                                'X-Min Learn', f'Unexpected file:\n{path}.',
                                detailedText=repr(e))
            finally:
                if mask:
                    self.mask_pathlbl.setPath(path)
                    self.setMask(mask)
                else:
                    self.removeMask()



    def getMaskFromClass(self, class_name:str):
        '''
        Extract a mask from the choosen mineral class of the choosen mineral 
        map.

        Parameters
        ----------
        class_name : str
            Name of the choosen class.

        '''
        minmap_name = self.minmap_combox.currentText()
        items = self.minmaps_list.findItems(minmap_name, Qt.MatchExactly, 0)
        if items:
            minmap = items[0].get('data').minmap
            mask = Mask(minmap != class_name)
            self.mask_pathlbl.setPath(f'{minmap_name}/{class_name}')
            self.setMask(mask)
        else:
            self.minmap_combox.clear()
            self.class_combox.clear()
            self.removeMask()
            QW.QMessageBox.critical(self, 'X-Min Learn', 'This mineral map '\
                                    'is no more available')


    def setMask(self, mask: Mask):
        '''
        Set a mask for classification. Masks are only rendered on top of Input 
        Maps and not on top of Mineral Maps.

        Parameters
        ----------
        mask : Mask
            The mask object.

        '''
        self._mask = mask
    
    # We want masks to only rendered on input maps
        if self.maps_viewer.contains_heatmap():
            self.showInputMap(self.inmaps_selector.currentItem())


    def removeMask(self):
        '''
        Remove mask.

        '''
        self._mask = None
        self.mask_pathlbl.clearPath()

    # Masks are only rendered on top of Input Maps
        if self.maps_viewer.contains_heatmap():
            self.showInputMap(self.inmaps_selector.currentItem())


    def addRoiMap(self, roimap: RoiMap):
        '''
        Render ROI map's patches and annotations in the maps canvas.

        Parameters
        ----------
        roimap : RoiMap
            The ROI map to be rendered.

        '''
        color = pref.get_setting('class/trAreasCol', (255,0,0), tuple)
        filled = pref.get_setting('class/trAreasFill', False, bool)

        for name, bbox in roimap.roilist:
            patch = cObj.RoiPatch(bbox, CF.RGB2float([color]), filled)
            text = cObj.RoiAnnotation(name, patch)
            self.maps_viewer.ax.add_patch(patch)
            self.maps_viewer.ax.add_artist(text)

        self.maps_viewer.draw_idle()


    def removeRoiMap(self):
        '''
        Remove all ROI map's patches and annotations from the maps canvas.

        '''
        for child in self.maps_viewer.ax.get_children():
            if isinstance(child, (cObj.RoiPatch, cObj.RoiAnnotation)):
                child.remove()

        self.maps_viewer.draw_idle()


    def showMineralMap(self, item: cObj.DataObject | None):
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
            self.chiscore_lbl.setText('None')
            self.dbiscore_lbl.setText('None')
            return
            
    # Enable confidence spinbox and slider
        self.conf_spbox.setEnabled(True)
        self.conf_slider.setEnabled(True)

    # Alter minmap data with the confidence threshold
        minmap = item.get('data')
        minmap_thr = self._thresholdMineralMap(minmap)

    # Use item name as title
        title = item.get('name')

    # Update the legend
        self.legend.update(minmap_thr)
    
    # Update the bar plot
        lbls, mode = zip(*minmap_thr.get_labeled_mode().items())
        mode_col = [minmap_thr.get_phase_color(lbl) for lbl in lbls]
        self.barplot.update_canvas(mode, lbls, title, mode_col)

    # Update clustering scores using original (non-thresholded) mineral map
        sil_avg, sil_clust, chi, dbi = minmap.get_clustering_scores()
        if sil_avg and sil_clust:
            self.silscore_canvas.update_canvas(sil_clust, sil_avg, title,
                                               minmap.palette)
        else:
            self.silscore_canvas.clear_canvas()

        self.silscore_lbl.setText(str(sil_avg))
        self.chiscore_lbl.setText(str(chi))
        self.dbiscore_lbl.setText(str(dbi))

    # Update the maps viewer
        mmap, enc, col = minmap_thr.get_plot_data()
        self.maps_viewer.draw_discretemap(mmap, enc, col, title)


    def saveMineralMap(self): 
        '''
        Save selected mineral map to file.

        '''
        item = self.minmaps_list.currentItem()
        path, _ = QW.QFileDialog.getSaveFileName(self, 'Save mineral map',
                                                pref.get_dirPath('out'),
                                                'Mineral map (*.mmp)')
        if path:
            pref.set_dirPath('out', os.path.dirname(path))
            minmap = self._thresholdMineralMap(item.get('data'))
            try:
                minmap.save(path)
            except Exception as e:
                text = 'An error occurred while saving the file'
                return cObj.RichMsgBox(self, QW.QMessageBox.Critical, 
                                       'X-Min Learn', text, 
                                       detailedText=repr(e))


    def removeMineralMap(self): 
        '''
        Remove the selected mineral map from the list of classified mineral 
        maps.

        '''
        item = self.minmaps_list.currentItem()
        item_idx = self.minmaps_list.indexOfTopLevelItem(item)
        choice = QW.QMessageBox.warning(self, 'X-Min Learn', 
                                        'Remove selected map?',
                                        QW.QMessageBox.Yes | QW.QMessageBox.No, 
                                        QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            self.minmaps_list.takeTopLevelItem(item_idx)
            new_displayed_item = self.minmaps_list.topLevelItem(item_idx - 1)
            self.minmaps_list.setCurrentItem(new_displayed_item)
            self.showMineralMap(new_displayed_item)


    def setConfidenceThreshold(self, value:int):
        '''
        Change the probability threshold value. This function dinamically 
        redraws the mineral map as well.

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
        self.showMineralMap(self.minmaps_list.currentItem())


    def _thresholdMineralMap(self, minmap:MineralMap):
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
        minmap_thr = MineralMap(minmap.minmap, minmap.probmap, minmap.palette)
    
    # Use confidence threshold to alter the cloned mineral map 
        thresh = self.conf_slider.value() / 100.
        minmap_thr.edit_minmap(minmap_thr._with_nodata(thresh))

    # Set special color to ND data (bugfix for constantly changing _ND_ color)
        if '_ND_' in minmap_thr.get_phases():
            minmap_thr.set_phase_color('_ND_', self._nodata_color)

        return minmap_thr
    

    def changeClassColor(self, legend_item: QW.QTreeWidgetItem, color: tuple):
        '''
        Alter the displayed color of a mineral class. This function propagates 
        the changes to the mineral map, the map canvas, the mode bar plot, the 
        legend and the silhouette plot. The arguments of this function are 
        specifically compatible with the colorChangeRequested signal emitted by 
        the legend (see Legend object for more details). 

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested the color change.
        color : tuple
            RGB triplet.

        '''
    # Get mineral map
        item = self.minmaps_list.currentItem()
        minmap = item.get('data')
        
    # Update the phase color in the mineral map. However, if _ND_ color was 
    # changed, update the _nodata_color attribute instead
        class_name = legend_item.text(1)
        if class_name == '_ND_':
             self._nodata_color = list(color)
        else:
            minmap.set_phase_color(class_name, color)

    # Re-draw mineral map, legend, bar plot and silhouette plot
        self.showMineralMap(item)


    def renameClass(self, legend_item:QW.QTreeWidgetItem, new_name:str):
        '''
        Rename a mineral class. This function propagates the changes to the 
        mineral map, the map canvas, the mode bar plot and the legend. The 
        arguments of this function are specifically compatible with the 
        itemRenameRequested signal emitted by the legend (see Legend object for
        more details).

        Parameters
        ----------
        legend_item : QW.QTreeWidgetItem
            The legend item that requested to be renamed
        new_name : str
            New class name.

        '''
    # Deny renaming as "_ND_"
        if new_name == '_ND_':
            return QW.QMessageBox.critical(self, 'X-Min Learn',
                                           '"_ND_" is a protected name') 

    # Deny renaming the _ND_ class
        old_name = legend_item.text(1)
        if old_name == '_ND_':
            return QW.QMessageBox.critical(self, 'X-Min Learn', 
                                           '"_ND_" class cannot be renamed')
        
    # Deny renaming if name is already taken
        item = self.minmaps_list.currentItem()
        minmap = item.get('data') 
        if new_name in minmap.get_phases():
            return QW.QMessageBox.critical(self, 'X-Min Learn',
                                           f'{new_name} is already taken')

    # If we get here, allow renaming. Re-draw mineral map, legend, barplot and
    # silhouette plot
        minmap.rename_phase(old_name, new_name)
        self.showMineralMap(item)


    def highlightClass(self, toggled:bool, legend_item:QW.QTreeWidgetItem):
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
    # Check that a mineral map is currently displayed in the viewer
        if self.maps_viewer.contains_discretemap():

            if toggled:
            # We need to operate on the thresholded version of the mineral map
                item = self.minmaps_list.currentItem()
                minmap = self._thresholdMineralMap(item.get('data'))
                phase_id = minmap.as_id(legend_item.text(1))
                vmin, vmax = phase_id - 0.5, phase_id + 0.5
            else:
                vmin, vmax = None, None

            self.maps_viewer.update_clim(vmin, vmax)
            self.maps_viewer.draw() 


    def startClassification(self):
        '''
        Launch a classification process.

        '''
    # Do not allow multiple classification processes at once
        if self._isBusyClassifying:
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'Cannot run '\
                                           'multiple classifications at once.')
        
    # Get checked input maps data and their dispayed names
        checked_inmaps = self.inmaps_selector.getChecked()
        inmaps, names = zip(*[i.get('data', 'name') for i in checked_inmaps])

    # Build the input maps stack
        input_stack = InputMapStack(inmaps, self._mask)
    
    # Check for maps perfect overlapping
        if not input_stack.maps_fit():
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'Input maps '\
                                           'have different shape/size')
        
    # Check that mask (if present) has correct shape
        if not input_stack.mask_fit():
            return QW.QMessageBox.critical(self, 'X-Min Learn', 'The selected '\
                                           'mask has an invalid shape')

    # Get the classifier and launch the classification thread
        active_tab = self.classifier_tabwid.currentWidget()
        csf = active_tab.getClassifier(input_stack, names) 

        if csf is not None:
            csf.thread.taskInitialized.connect(self.progbar.step)
            csf.thread.workInterrupted.connect(self._endClassification)
            csf.thread.workFinished.connect(self._parseClassifierResult)

            self.progbar.setRange(0, csf.classification_steps)
            self._current_classifier = csf
            self._isBusyClassifying = True

            csf.startThreadedClassification()


    def _parseClassifierResult(self, result:tuple, success:bool):
        '''
        Parse the result of the classification thread. If the classification 
        was successful, this function shows the mineral map in the canvas.

        Parameters
        ----------
        result : tuple
            Classification thread result.
        success : bool
            Whether the classification thread succeeded or not.

        '''
        if success:  
        # Parse the classification result appropriately
            if self._current_classifier.type == 'Unsupervised':
                pred, prob, sil_avg, sil_clust, chi, dbi = result
                pred = pred.astype('U8')
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
                mmap = np.empty(shape, dtype='U8')
                mmap[:, :] = '_MSK_'
                mmap[rows, cols] = pred
                pmap = np.ones(shape)
                pmap[rows, cols] = prob

        # Create a new item in minmaps list and populate it with a Mineral Map
            mineral_map = MineralMap(mmap, pmap)
            mineral_map.set_clustering_scores(sil_avg, sil_clust, chi, dbi)
            item = cObj.DataObject(mineral_map)
            dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            item.setText(0, f'{self._current_classifier.name} [{dt}]')
            self.minmaps_list.addTopLevelItem(item)

        # Force show the output maps tab and display the mineral map
            self.data_tabwid.setCurrentIndex(2) 
            self.minmaps_list.setCurrentItem(item)
            self.showMineralMap(item)

        else:
            e = result[0]
            cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                            'Classification failed.', detailedText=repr(e))

        self._endClassification(success)


    def stopClassification(self):
        '''
        Interrupt the current classification thread.

        '''
        if self._current_classifier is not None:
            self.progbar.setUndetermined()
            self.progbar.step('Interrupting classification')
            self._current_classifier.thread.requestInterruption()


    def _endClassification(self, success=False):
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
            QW.QMessageBox.information(self, 'X-Min Lern', 
                                       'Classification completed with success')

    def closeEvent(self, event):
        '''
        Reimplementation of the closeEvent function. Requires exit confirm if
        a classification thread is currently active.

        Parameters
        ----------
        event : QEvent
            The close event.

        '''
        if self._isBusyClassifying:
            text = 'A classification process is still active. Close anyway?'
            btns = QW.QMessageBox.Yes | QW.QMessageBox.No
            choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                                            btns, QW.QMessageBox.No)
            if choice == QW.QMessageBox.Yes:
                self.stopClassification()
                event.accept()
            else:
                event.ignore()

        else:
            super(MineralClassifier, self).closeEvent(event)


    class PreTrainedClassifierTab(QW.QWidget):
        '''
        First tab of the Mineral Classifier's tab widget, which allows the 
        selection of a classifier type. This tab allows the loading of a 
        pre-trained ML model.
        '''

        def __init__(self, parent=None):
            '''
            Constructor.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attributes
            self.model = None

            self._init_ui()
            self._connect_slots()

        def _init_ui(self):
            '''
            GUI constructor.

            '''
        # Load Model (Styled Button)
            self.load_btn = cObj.StyledButton(QIcon(r'Icons/import.png'),
                                              'Load model')
            self.load_btn.setToolTip('Load a pre-trained ML model')

        # Loaded model path (Path Label)
            self.model_path = cObj.PathLabel(full_display=False)

        # Loaded model information (Document Browser)
            self.model_info = cObj.DocumentBrowser()
            self.model_info.setDefaultPlaceHolderText(
                'Unable to retrieve model information.')

        # Adjust main layout
            main_layout = QW.QVBoxLayout()
            main_layout.addWidget(self.load_btn)
            main_layout.addWidget(self.model_path)
            main_layout.addSpacing(20)
            main_layout.addWidget(self.model_info)
            self.setLayout(main_layout)


        def _connect_slots(self):
            '''
            Signals-slots connector.

            '''
            self.load_btn.clicked.connect(self.loadModel)


        def loadModel(self):
            '''
            Load a pre-trained model from file.

            '''
            path, _ = QW.QFileDialog.getOpenFileName(self, 'Import model',
                                                     pref.get_dirPath('in'),
                                                     'PyTorch model (*.pth)')
            if path:
                pref.set_dirPath('in', os.path.dirname(path))
                try:
                    self.model = ML_tools.EagerModel.load(path)
                except Exception as e:
                    return cObj.RichMsgBox(self, QW.QMessageBox.Critical,
                                           'X-Min Learn', 'Incompatible model',
                                           detailedText = repr(e))
                self.model_path.setPath(path)
                logpath = self.model.generate_log_path(path)

                # If model log was deleted or moved, ask for rebuilding it
                if not os.path.exists(logpath):
                    quest_text = 'Unable to find model log file. Rebuild it?'
                    btns = QW.QMessageBox.Yes | QW.QMessageBox.No
                    choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                                     quest_text, btns,
                                                     QW.QMessageBox.Yes)
                    if choice == QW.QMessageBox.Yes:
                        ext_log = pref.get_setting('class/extLog', False, bool)
                        self.model.save_log(logpath, extended=ext_log)

                # Load log file. No error will be raised if it does not exist
                self.model_info.setDoc(logpath)


        def getClassifier(self, input_stack: InputMapStack, maps_names: list):
            '''
            Return the classifier stored in the loaded pre-trained model. This
            function also checks if the input maps required by the model are
            present.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.
            maps_names : list
                List of input maps names.

            Returns
            -------
            ModelBasedClassifier or None
                The pre-trained classifier or None if the required input maps
                are not present.

            '''
            if self.model == None:
                QW.QMessageBox.critical(self, 'X-Min Learn', 
                                        'A pre-trained model is required')
                return None

        # Check if all required input maps are present and order them to fit
        # the correct order
        # add a user-friendly popup to link each map to required feat instead (enhancement)
            ordered_indices = []
            for feat in self.model.features:
                if not CF.guessMap(feat, maps_names, caseSens=True):
                    QW.QMessageBox.critical(self, 'X-Min Learn',
                                            f'Unable to identify {feat} map.')
                    return None
                else:
                    ordered_indices.append(maps_names.index(feat))

            input_stack.reorder(ordered_indices)

            return ML_tools.ModelBasedClassifier(input_stack, self.model)


    class RoiBasedClassifierTab(QW.QWidget):
        '''
        Second tab of the Mineral Classifier's tab widget, which allows the 
        selection of a classifier type. This tab allows the selection of a 
        ROI-based classifier.
        '''

        addRoiMapRequested = QC.pyqtSignal(RoiMap)
        removeRoiMapRequested = QC.pyqtSignal()

        def __init__(self, parent=None):
            '''
            Constructor.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attributes
            self._algorithms = ('K-Nearest Neighbors',)
            self._roimap = None

            self._init_ui()
            self._connect_slots()

        def _init_ui(self):
            '''
            GUI constructor.

            '''
        # Load ROI map (Styled Button)
            self.load_btn = cObj.StyledButton(QIcon('Icons/load.png'),
                                              'Load ROI map')
            self.load_btn.setToolTip('Load training ROI data')

        # Remove (unload) ROI map (Styled Button)
            self.unload_btn = cObj.StyledButton(QIcon(r'Icons/clear.png'))
            self.unload_btn.setToolTip('Remove ROI map')
            
        # Loaded model path (Path Label)
            self.roimap_path = cObj.PathLabel(full_display=False)

        # Include pixel proximity (Checkbox)
            self.pixprox_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
            self.pixprox_cbox.setToolTip('Use pixel coords as input features')
            self.pixprox_cbox.setChecked(False)

        # Use parallel computation (Checkbox)
            self.multithread_cbox = QW.QCheckBox('Parallel computation')
            self.multithread_cbox.setToolTip(
                'Distribute computation across multiple processes')
            self.multithread_cbox.setChecked(False)

        # Algorithm selection (Styled ComboBox)
            self.algm_combox = cObj.StyledComboBox()
            self.algm_combox.addItems(self._algorithms)

        # Algorithms Panel (Stacked Widget)
            self.algm_panel = QW.QStackedWidget()


        #----------------K-NEAREST NEIGHBORS ALGORITHM WIDGETS----------------#
        # N. of neighbors  (Styled Spin Box)
            self.knn_nneigh_spbox = cObj.StyledSpinBox(1, 100)
            self.knn_nneigh_spbox.setValue(5)

        # Weight of neighbors (Styled Combo Box)
            self.knn_weight_combox = cObj.StyledComboBox()
            self.knn_weight_combox.addItems(['Uniform', 'Distance'])
            self.knn_weight_combox.setToolTip(
                'Neighbors weight should be uniform or distance-based?')

        # Add KNN widgets to the Algorithm Panel
            knn_layout = QW.QFormLayout()
            knn_layout.addRow('N. of neighbours', self.knn_nneigh_spbox)
            knn_layout.addRow('Neighbors weight', self.knn_weight_combox)
            knn_group = cObj.GroupArea(knn_layout, 'K-Nearest Neighbors')
            
            self.algm_panel.addWidget(knn_group)
        #---------------------------------------------------------------------#

        # Adjust main layout
            main_layout = QW.QGridLayout()
            main_layout.addWidget(self.load_btn, 0, 0, 1, -1)
            main_layout.addWidget(self.unload_btn, 1, 0)
            main_layout.addWidget(self.roimap_path, 1, 1)
            main_layout.setRowMinimumHeight(2, 20)
            main_layout.addWidget(self.pixprox_cbox, 3, 0, 1, -1)
            main_layout.addWidget(self.multithread_cbox, 4, 0, 1, -1)
            main_layout.setRowMinimumHeight(5, 20)
            main_layout.addWidget(QW.QLabel('Select algorithm'), 6, 0, 1, -1)
            main_layout.addWidget(self.algm_combox, 7, 0, 1, -1)
            main_layout.setRowMinimumHeight(8, 10)
            main_layout.addWidget(self.algm_panel, 9, 0, 1, -1)
            main_layout.setColumnStretch(1, 2)
            self.setLayout(main_layout)


        def _connect_slots(self):
            '''
            Signals-slots connector.

            '''
        # Load ROI map from file
            self.load_btn.clicked.connect(self.loadRoiMap)

        # Remove (unload) ROI map
            self.unload_btn.clicked.connect(self.unloadRoiMap)

        # Select a different ROI-based algorithm
            self.algm_combox.currentTextChanged.connect(self.switchAlgorithm)


        def loadRoiMap(self):
            '''
            Load a new ROI map from file and request its rendering in the 
            Mineral Classifier's maps canvas.

            '''
        # Get path to new ROI map
            path, _ = QW.QFileDialog.getOpenFileName(self, 'Load ROI map',
                                                     pref.get_dirPath('in'),
                                                     'ROI maps (*.rmp)')
            if path:
                pref.set_dirPath('in', os.path.dirname(path))
                pbar = cObj.PopUpProgBar(self, 3, 'Loading data', cancel=False)
                pbar.setValue(0)

            # Try loading the ROI map. Exit function if something goes wrong
                try:
                    new_roimap = RoiMap.load(path)
                    pbar.increase()
                except Exception as e:
                    pbar.reset()
                    return cObj.RichMsgBox(self, QW.QMessageBox.Critical,
                                        'X-Min Learn', f'Unexpected file:\n{path}',
                                        detailedText = repr(e))

            # Remove previous ROI map from canvas
                self.unloadRoiMap()
                pbar.increase()

            # Add the new ROI map and populate the canvas with its ROIs
                self._roimap = new_roimap
                self.addRoiMapRequested.emit(new_roimap)
                self.roimap_path.setPath(path)
                pbar.increase()


        def unloadRoiMap(self):
            '''
            Unload current ROI map and request its removal from the Mineral
            Classifier's maps canvas.

            '''
            if self._roimap is not None:
                self._roimap = None
                self.roimap_path.clearPath()
                self.removeRoiMapRequested.emit()
                

        def switchAlgorithm(self, algorithm: str):
            '''
            Swap widget in the algorithms panel based on selected algorithm.

            Parameters
            ----------
            algorithm : str
                Selected algorithm.

            '''
            idx = self._algorithms.index(algorithm)
            self.algm_panel.setCurrentWidget(idx)


        def getClassifier(self, input_stack: InputMapStack, maps_names=None): 
            '''
            Return the ROI-based classifier with the selected parameters. This
            fuction also checks if the extension the ROI map is compatible with
            the extensions of the input maps.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.
            maps_names : None, optional
                This has no use. It is here only for args compatibility with
                ModelBased Classifier. The default is None.

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
                QW.QMessageBox.critical(self, 'X-Min Learn', 
                                        'A ROI map is required')
                return None

        # Check if ROI map extension differs from input maps extensions
            if input_stack.maps_shape != roimap.shape:
                warn_text = 'ROI map extension is different from sample '\
                            'extension. Proceed anyway?'
                btns = QW.QMessageBox.Yes | QW.QMessageBox.No
                choice = QW.QMessageBox.warning(self, 'X-Min Learn', warn_text,
                                                btns, QW.QMessageBox.No)
                if choice == QW.QMessageBox.No:
                    return None

            if algm == 'K-Nearest Neighbors':
                nneigh = self.knn_nneigh_spbox.value()
                weight = self.knn_weight_combox.currentText().lower()
                njobs = -1 if self.multithread_cbox.checkState() else 1
                args = (input_stack, roimap, nneigh, weight)
                kwargs = {'n_jobs': njobs, 'pixel_proximity': prox}
                return ML_tools.KNearestNeighbors(*args, **kwargs)

            else:
                return None


    class UnsupervisedClassifierTab(QW.QWidget):
        ''' 
        Third tab of the Mineral Classifier's tab widget, which allows the 
        selection of a classifier type. This tab allows the selection of an 
        unsupervised classifier.
        '''

        def __init__(self, parent=None):
            '''
            Constructor.

            Parameters
            ----------
            parent : QWidget or None, optional
                GUI parent widget of this tab. The default is None.

            '''
            super().__init__(parent)

        # Set main attribute
            self._algorithms = ('K-Means',)

            self._init_ui()
            self._connect_slots()

        def _init_ui(self):
            '''
            GUI constructor.

            '''
        # Seed generator widget
            self.seed_generator = cObj.RandomSeedGenerator()

        # Compute silhouette score (Checkbox)
            self.silscore_cbox = QW.QCheckBox('Silhouette score')
            self.silscore_cbox.setChecked(True)
            self.silscore_cbox.setToolTip('This may take some time')

        # Silhouette score subset ratio (StyledSpinbox)
            self.silscore_ratio_spbox = cObj.StyledSpinBox()
            self.silscore_ratio_spbox.setSuffix(' %')
            self.silscore_ratio_spbox.setValue(25)
            ratio_ttip = 'Analyze a portion of data to reduce computation time'
            self.silscore_ratio_spbox.setToolTip(ratio_ttip)

        # Compute Calinski-Harabasz Index (Checkbox)
            self.chiscore_cbox = QW.QCheckBox('Calinski-Harabasz Index')
            self.chiscore_cbox.setChecked(True)

        # Compute Davies-Bouldin Index (Checkbox)
            self.dbiscore_cbox = QW.QCheckBox('Davies-Bouldin Index')
            self.dbiscore_cbox.setChecked(True)

        # Include pixel proximity (Checkbox)
            self.pixprox_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
            self.pixprox_cbox.setToolTip('Use pixel coords as input features')
            self.pixprox_cbox.setChecked(False)

        # Use parallel computation (Checkbox)
            self.multithread_cbox = QW.QCheckBox('Parallel computation')
            self.multithread_cbox.setToolTip(
                'Distribute computation across multiple processes')
            self.multithread_cbox.setChecked(False)

        # Algorithm selection (Styled ComboBox)
            self.algm_combox = cObj.StyledComboBox()
            self.algm_combox.addItems(self._algorithms)

        # Algorithms Panel (Stacked Widget)
            self.algm_panel = QW.QStackedWidget()


        #--------------------- K-MEANS ALGORITHM WIDGETS ---------------------#
        # N. of clusters
            self.kmeans_nclust_spbox = cObj.StyledSpinBox(min_value=2)
            self.kmeans_nclust_spbox.setValue(8)

        # Add K-Means widgets to the Algorithm Panel
            kmeans_layout = QW.QFormLayout()
            kmeans_layout.addRow('N. of clusters', self.kmeans_nclust_spbox)
            kmeans_group = cObj.GroupArea(kmeans_layout, 'K-Means')

            self.algm_panel.addWidget(kmeans_group)
        #---------------------------------------------------------------------#

        # Adjust layout
            scores_grid = QW.QGridLayout()
            scores_grid.addWidget(self.silscore_cbox, 0, 0)
            scores_grid.addWidget(QW.QLabel('Data ratio'), 0, 1, Qt.AlignRight)
            scores_grid.addWidget(self.silscore_ratio_spbox, 0, 2)
            scores_grid.addWidget(self.chiscore_cbox, 1, 0, 1, -1)
            scores_grid.addWidget(self.dbiscore_cbox, 2, 0, 1, -1)
            scores_group = cObj.GroupArea(scores_grid, 'Clustering scores',
                                          checkable=True)
            scores_group.setChecked(False)

            main_layout = QW.QVBoxLayout()
            main_layout.addWidget(self.seed_generator)
            main_layout.addSpacing(20)
            main_layout.addWidget(self.pixprox_cbox)
            main_layout.addWidget(self.multithread_cbox)
            main_layout.addSpacing(20)
            main_layout.addWidget(scores_group)
            main_layout.addSpacing(20)
            main_layout.addWidget(QW.QLabel('Select algorithm'))
            main_layout.addWidget(self.algm_combox)
            main_layout.addSpacing(10)
            main_layout.addWidget(self.algm_panel)
            self.setLayout(main_layout)


        def _connect_slots(self):
            '''
            Signals-slots connector.

            '''
        # Disable silhouette data ratio when silhouette score is unchecked
            self.silscore_cbox.stateChanged.connect(
                self.silscore_ratio_spbox.setEnabled)
        # Select a different clustering algorithm
            self.algm_combox.currentTextChanged.connect(self.switchAlgorithm)


        def switchAlgorithm(self, algorithm: str):
            '''
            Swap widget in the algorithms panel based on selected algorithm.

            Parameters
            ----------
            algorithm : str
                Selected algorithm.

            '''
            idx = self._algorithms.index(algorithm)
            self.algm_panel.setCurrentWidget(idx)


        def getClassifier(self, input_stack: InputMapStack, maps_names=None):
            '''
            Return the unsupervised classifier with the selected parameters.

            Parameters
            ----------
            input_stack : InputMapStack
                The stack of input maps.
            maps_names : None, optional
                This has no use. It is here only for args compatibility with
                ModelBased Classifier. The default is None.

            Returns
            -------
            UnsupervisedClassifier
                The unsupervised classifier.

            '''
        # Common clustering parameters
            seed = self.seed_generator.seed
            prox = self.pixprox_cbox.checkState()
            njobs = -1 if self.multithread_cbox.checkState() else 1
            sil = self.silscore_cbox.isEnabled() & self.silscore_cbox.isChecked()
            chi = self.chiscore_cbox.isEnabled() & self.chiscore_cbox.isChecked()
            dbi = self.dbiscore_cbox.isEnabled() & self.dbiscore_cbox.isChecked()
            ratio = self.silscore_ratio_spbox.value() / 100
            kwargs = {'n_jobs': njobs, 'pixel_proximity':prox, 
                      'sil_score': sil, 'chi_score': chi, 'dbi_score': dbi,
                      'sil_ratio': ratio}
        
        # Algorithm-specific parameters
            algm = self.algm_combox.currentText()

            if algm == 'K-Means':
                n_clust = self.kmeans_nclust_spbox.value()
                args = (input_stack, seed, n_clust)
                return ML_tools.KMeans(*args, **kwargs)

            else:
                return None







# class MineralClassifierOLD(QW.QWidget):
#     def __init__(self, XMapsInfo, MinMapsInfo, parent=None):
#         super(MineralClassifierOLD, self).__init__()
#         self.parent = parent

#         self.setWindowTitle('Mineral Classifier')
#         self.setWindowIcon(QIcon('Icons/classify.png'))
#         self.setAttribute(Qt.WA_QuitOnClose, False)
#         self.setAttribute(Qt.WA_DeleteOnClose, True)

#         self.XMapsPath, self.XMapsDataOrig = XMapsInfo
#         self.MinMapsPath, self.MinMapsData = MinMapsInfo

#         self.maskOn = False
#         self.XMapsData = self.XMapsDataOrig.copy()

#         self.algm_rawIn, self.algm_rawOut = None, None
#         self.minMap, self.probMap = None, None

#         # Silhouette score external thread
#         self.silhThread = exthr.SilhouetteThread()
#         self.silhThread.setObjectName('Silhouette scoring')
#         self.silhThread.taskFinished.connect(
#             lambda out: self.plot_silhouetteScore(out[0], out[1]))

#         self.init_ui()
#         self.adjustSize()


#     def init_ui(self):
#     # Input Maps Checkboxes
#         self.CboxMaps = cObj.CBoxMapLayout(self.XMapsPath)
#         # If a threading is running, the next line blocks interactions with cboxes
#         self.CboxMaps.cboxPressed.connect(lambda: self._extThreadRunning())
#         maps_scroll = cObj.GroupScrollArea(self.CboxMaps, 'Input Maps')

#     # Algorithm combo box
#         self.algm_combox = QW.QComboBox()
#         self.algm_combox.addItems(['Pre-trained Model', 'KNN', 'K-Means'])
#         self.algm_combox.currentIndexChanged.connect(self.onAlgmChanged)


#     # ==== ALGORITHMS PREFERENCES WIDGETS STACK ==== #

#     # Pre-trained algorithm preferences group
#         # (Load model button)
#         self.loadModel_btn = QW.QPushButton('Load model')
#         self.loadModel_btn.setToolTip('Select a pre-trained Machine Learning model.')
#         self.loadModel_btn.clicked.connect(self.loadModel)

#         # (Loaded model path)
#         self.model_path = cObj.PathLabel('')

#         # (Adjust Pre-Trained model Group Layout)
#         preTrained_vbox = QW.QVBoxLayout()
#         preTrained_vbox.addWidget(self.loadModel_btn)
#         preTrained_vbox.addWidget(self.model_path)
#         preTrained_group = cObj.GroupArea(preTrained_vbox, 'Algorithm Preferences')

#     # KNN algorithm preferences group
#         # (Number of neighbours spinbox)
#         self.nNeigh_spbox = QW.QSpinBox()
#         self.nNeigh_spbox.setRange(1, 100)
#         self.nNeigh_spbox.setValue(5)

#         # (Weight function combo-box)
#         self.wgtKNN_combox = QW.QComboBox()
#         self.wgtKNN_combox.addItems(['Uniform', 'Distance'])
#         self.wgtKNN_combox.setToolTip('All neighbors are weighted equally [Uniform] '\
#                                       'or closer neighbours have a greater influence [Distance].')

#         # (Include pixel proximity checkbox)
#         self.KNNproximity_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
#         self.KNNproximity_cbox.setToolTip('Include pixel coordinates as input feature')
#         self.KNNproximity_cbox.setChecked(False)

#         # (Adjust KNN Group Layout)
#         KNN_form = QW.QFormLayout()
#         KNN_form.addRow('N. of Neighbours', self.nNeigh_spbox)
#         KNN_form.addRow('Weigths', self.wgtKNN_combox)
#         KNN_form.addRow(self.KNNproximity_cbox)
#         KNN_group = cObj.GroupArea(KNN_form, 'Algorithm Preferences')

#     # K-Means algorithm preferences
#         # (Number of classes spinbox)
#         self.Kclasses_spbox = QW.QSpinBox()
#         self.Kclasses_spbox.setRange(2, 100)
#         self.Kclasses_spbox.setValue(8)

#         # (Random Seed input) -> should it be the same repeated widget for other potential seed-requiring algorithms?
#         self.seedInput = QW.QLineEdit()
#         self.seedInput.setValidator(QIntValidator(0, 10**8))
#         self.seedInput.setText(str(np.random.randint(0, 10**8)))

#         # (Randomize seed button)
#         self.randseed_btn = cObj.IconButton('Icons/dice.png')
#         self.randseed_btn.clicked.connect(
#             lambda: self.seedInput.setText(str(np.random.randint(0, 10**8))))

#         # (Include pixel proximity checkbox)
#         self.kmeansProximity_cbox = QW.QCheckBox('Pixel Proximity (experimental)')
#         self.kmeansProximity_cbox.setToolTip('Include pixel coordinates as input feature')
#         self.kmeansProximity_cbox.setChecked(False)

#         # (Adjust KMeans Group Layout)
#         KMeans_grid = QW.QGridLayout()
#         KMeans_grid.addWidget(QW.QLabel('N. of Classes'), 0, 0)
#         KMeans_grid.addWidget(self.Kclasses_spbox, 0, 1, 1, 2)
#         KMeans_grid.addWidget(QW.QLabel('Seed'), 1, 0)
#         KMeans_grid.addWidget(self.seedInput, 1, 1)
#         KMeans_grid.addWidget(self.randseed_btn, 1, 2)
#         KMeans_grid.addWidget(self.kmeansProximity_cbox, 2, 0, 1, 3)
#         KMeans_group = cObj.GroupArea(KMeans_grid, 'Algorithm Preferences')

#     # Algorithms Preferences Widget (Stacked Widget)
#         self.algmPrefs = QW.QStackedWidget()
#         self.algmPrefs.addWidget(preTrained_group)
#         self.algmPrefs.addWidget(KNN_group)
#         self.algmPrefs.addWidget(KMeans_group)
#         self.algmPrefs.setCurrentIndex(0)

#     # ================================================== #


#     # Mineral Maps combo-box (for sub-phase identification)
#         self.minmaps_combox = QW.QComboBox()
#         self.refresh_minmaps_combox()
#         self.minmaps_combox.currentIndexChanged.connect(self.onMinmapChanged)
#         # Events handling to avoid problems during multi-threading operations
#         self.minmaps_combox.keyPressEvent = lambda evt: evt.ignore()
#         self.minmaps_combox.wheelEvent = lambda evt: evt.ignore()
#         self.minmaps_combox.highlighted.connect(lambda: self._extThreadRunning())

#     # Mineral phase combo-box (mineral phase to use as a mask)
#         self.minPhase_combox = QW.QComboBox()
#         self.minPhase_combox.setEnabled(False)
#         self.minPhase_combox.currentIndexChanged.connect(
#             lambda idx: self.minPhase_combox.setEnabled(idx != -1))
#         self.minPhase_combox.currentTextChanged.connect(self.mask_maps)
#         # Events handling to avoid problems during multi-threading operations
#         self.minPhase_combox.keyPressEvent = lambda evt: evt.ignore()
#         self.minPhase_combox.wheelEvent = lambda evt: evt.ignore()
#         self.minPhase_combox.highlighted.connect(lambda: self._extThreadRunning())

#     # Refresh loaded mineral maps button
#         self.refreshMinMap_btn = cObj.IconButton('Icons/refresh.png')
#         self.refreshMinMap_btn.clicked.connect(self.refresh_minmaps_combox)

#     # Adjust sub-phase identification group
#         subPhase_form = QW.QFormLayout()
#         subPhase_form.addRow('Mineral Map', self.minmaps_combox)
#         subPhase_form.addRow('Phase', self.minPhase_combox)
#         subPhase_form.addRow('Refresh', self.refreshMinMap_btn)
#         subPhase_group = cObj.GroupArea(subPhase_form, 'Sub-phase Identification')


#     # ==== ALGORITHMS PANELS WIDGETS STACK ==== #

#     # Pre-trained Model Information
#         self.modelInfo = cObj.DocumentBrowser()
#         self.modelInfo.set_defaultPlaceHolderText('Unable to retrieve model information.')


#     # Training area selector panel
#         self.trAreaPicker = cObj.TrAreasSelector(self.CboxMaps.Cbox_list, self.XMapsData, self)


#     # CLUSTERING EVALUATION TOOLS

#     # Silhouette score canvas
#         self.silhouetteCanvas = cObj.SilhouetteScoreCanvas(self, tight=True)
#         self.silhouetteCanvas.setMinimumSize(100, 100)

#     # Silhouette Navigation toobar
#         self.silhouetteNTbar = cObj.NavTbar(self.silhouetteCanvas, self)
#         self.silhouetteNTbar.removeToolByIndex([3, 4, 5, 6, 8, 9, 10, 12])

#     # Subset percentage spinbox
#         self.subsetPerc_spbox = QW.QDoubleSpinBox()
#         self.subsetPerc_spbox.setToolTip('Select the percentage of data to be evaluated.')
#         self.subsetPerc_spbox.setRange(0, 1)
#         self.subsetPerc_spbox.setSingleStep(0.01)
#         self.subsetPerc_spbox.setValue(0.3)

#     # Random Seed input
#         self.silhouetteSeed = QW.QLineEdit()
#         self.silhouetteSeed.setValidator(QIntValidator(0, 10**8))
#         self.silhouetteSeed.setText(str(np.random.randint(0, 10**8)))

#     # Silhouette start button
#         self.startSilhouette_btn = QW.QPushButton('START')
#         self.startSilhouette_btn.setStyleSheet('''background-color: rgb(50,205,50);
#                                         font-weight: bold''')
#         self.startSilhouette_btn.clicked.connect(self.start_silhouetteScore)

#     # Silhouette progress bar
#         self.silhouette_pbar = QW.QProgressBar()

#     # Calinski-Harabasz Index button
#         self.CHIscore_btn = QW.QPushButton('Calinski-Harabasz Index')
#         self.CHIscore_btn.setToolTip('Compute the Calinski-Harabasz Index.')
#         self.CHIscore_btn.clicked.connect(self.compute_CHIscore)

#     # Calinski-Harabasz Index label
#         self.CHIscore_label = QW.QLabel('None')
#         self.CHIscore_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

#     # Davies-Bouldin Index button
#         self.DBIscore_btn = QW.QPushButton('Davies-Bouldin Index')
#         self.DBIscore_btn.setToolTip('Compute the Davies-Bouldin Index.')
#         self.DBIscore_btn.clicked.connect(self.compute_DBIscore)

#     # Davies-Bouldin Index label
#         self.DBIscore_label = QW.QLabel('None')
#         self.DBIscore_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

#     # Adjust clustering scores group layout
#         SIL_grid = QW.QGridLayout()
#         SIL_grid.addWidget(QW.QLabel('Subset ratio'), 0, 0)
#         SIL_grid.addWidget(self.subsetPerc_spbox, 0, 1)
#         SIL_grid.addWidget(QW.QLabel('Random seed'), 1, 0)
#         SIL_grid.addWidget(self.silhouetteSeed, 1, 1)
#         SIL_grid.addWidget(self.startSilhouette_btn, 2, 0, alignment=Qt.AlignLeft)
#         SIL_grid.addWidget(self.silhouette_pbar, 3, 0, 1, 2)
#         SIL_group = cObj.GroupArea(SIL_grid, 'Silhouette score')

#         otherScores_vbox = QW.QVBoxLayout()
#         otherScores_vbox.addWidget(self.CHIscore_btn)
#         otherScores_vbox.addWidget(self.CHIscore_label)
#         otherScores_vbox.addWidget(self.DBIscore_btn)
#         otherScores_vbox.addWidget(self.DBIscore_label)
#         otherScores_group = cObj.GroupArea(otherScores_vbox, 'Other scores')

#         clustering_grid = QW.QGridLayout()
#         clustering_grid.setRowStretch(1, 1)
#         clustering_grid.addWidget(self.silhouetteNTbar, 0, 0, 1, 2)
#         clustering_grid.addWidget(self.silhouetteCanvas, 1, 0, 1, 2)
#         clustering_grid.addWidget(cObj.LineSeparator(), 2, 0, 1, 2)
#         clustering_grid.addWidget(SIL_group, 3, 0)
#         clustering_grid.addWidget(otherScores_group, 3, 1)
#         self.clustering_group = QW.QWidget()
#         self.clustering_group.setLayout(clustering_grid)

#     # Algorithms Panels Widget (Stacked Widget)
#         self.algmPanels = QW.QStackedWidget()
#         self.algmPanels.addWidget(self.modelInfo)
#         self.algmPanels.addWidget(self.trAreaPicker)
#         self.algmPanels.addWidget(self.clustering_group)
#         self.algmPanels.setCurrentIndex(0)
#         algmPanels_group = cObj.GroupArea(self.algmPanels, 'Algorithm Panel')

#     # ================================================== #


#     # Result shower canvas
#         self.resultCanvas = cObj.DiscreteClassCanvas(self, size=(10, 10), tight=True)
#         self.resultCanvas.setMinimumSize(100,100)

#     # Result canvas navigation toolbar
#         self.resultNTbar = cObj.NavTbar(self.resultCanvas, self)
#         self.resultNTbar.removeToolByIndex([3, 4, 8, 9])
#         self.resultNTbar.fixHomeAction()

#     # Unclassified pixel shower canvas
#         self.ND_Canvas = cObj.HeatMapCanvas(self, binary=True, cbar=False, tight=True)
#         self.ND_Canvas.setMinimumSize(100, 100)
#         CF.shareAxis(self.ND_Canvas.ax, self.resultCanvas.ax, True)

#     # Unclassified pixel navigation toolbar
#         self.ND_NTbar = cObj.NavTbar(self.ND_Canvas, self)
#         self.ND_NTbar.fixHomeAction()
#         self.ND_NTbar.removeToolByIndex([3, 4, 8, 9, 12])

#     # Result canvas legend
#         self.resultLegend = cObj.CanvasLegend(self.resultCanvas)
#         self.resultLegend.itemColorChanged.connect(self.recolor_plots)

#     # Result bar plot
#         self.resultBars = cObj.BarCanvas(self)
#         self.resultBars.setMinimumSize(100, 100)

#     # Adjust Result group
#         result_grid = QW.QGridLayout()
#         result_grid.addWidget(self.resultNTbar, 0, 0, 1, 2)
#         result_grid.addWidget(self.resultCanvas, 1, 0, 1, 2)
#         result_grid.addWidget(self.ND_NTbar, 0, 2, 1, 2)
#         result_grid.addWidget(self.ND_Canvas, 1, 2, 1, 2)
#         result_grid.addWidget(self.resultLegend, 2, 0, alignment=Qt.AlignHCenter)
#         result_grid.addWidget(self.resultBars, 2, 1, 1, 3)
#         result_grid.setRowStretch(1, 2)
#         result_grid.setRowStretch(2, 1)
#         result_group = cObj.GroupArea(result_grid, 'Classification result')


#     # Classification confidence spinbox
#         self.conf_spbox = QW.QDoubleSpinBox()
#         self.conf_spbox.setToolTip('Set a confidence threshold for the classification.')
#         self.conf_spbox.setRange(0, 1)
#         self.conf_spbox.setSingleStep(0.01)
#         self.conf_spbox.setValue(0.50)

#     # Auto load result checkbox
#         self.autoLoad_cbox = QW.QCheckBox('Auto-load Result')
#         self.autoLoad_cbox.setToolTip('Automatically load the classification result in Classified Mineral Maps Tab.')
#         self.autoLoad_cbox.setChecked(True)

#     # Start button
#         self.start_btn = QW.QPushButton('START')
#         self.start_btn.setStyleSheet('''background-color: rgb(50,205,50);
#                                         font-weight: bold''')
#         self.start_btn.clicked.connect(self.start_classification)

#     # Save button
#         self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'SAVE')
#         self.save_btn.setEnabled(False)
#         self.save_btn.clicked.connect(self.save_result)

#     # Progress bar
#         self.progBar = QW.QProgressBar()

#     # Adjust Preferences group
#         pref_grid = QW.QGridLayout()
#         pref_grid.addWidget(QW.QLabel('Confidence'), 0, 0)
#         pref_grid.addWidget(self.conf_spbox, 0, 1)
#         pref_grid.addWidget(self.autoLoad_cbox, 1, 0, 1, 2)
#         pref_grid.addWidget(self.start_btn, 2, 0)
#         pref_grid.addWidget(self.save_btn, 2, 1)
#         pref_grid.addWidget(self.progBar, 3, 0, 1, 2)
#         pref_group = cObj.GroupArea(pref_grid, 'Preferences')

#     # Adjust Main Layout
#         # (Left_vbox)
#         left_vbox = QW.QVBoxLayout()
#         left_vbox.addWidget(maps_scroll, 1)
#         left_vbox.addWidget(QW.QLabel('Classifier'))
#         left_vbox.addWidget(self.algm_combox)
#         left_vbox.addWidget(self.algmPrefs)
#         left_vbox.addWidget(subPhase_group)
#         left_vbox.addWidget(pref_group)

#         # (Main Layout)
#         main_hsplit = cObj.SplitterGroup((left_vbox, algmPanels_group, result_group),
#                                          (0, 1, 1))
#         mainLayout = QW.QHBoxLayout()
#         mainLayout.addWidget(main_hsplit)
#         self.setLayout(mainLayout)


#     def _extThreadRunning(self):
#         threads = [self.silhThread]
#         for t in threads:
#             if t.isRunning():
#                 _name = t.objectName()
#                 QW.QMessageBox.critical(self, 'X-Min Learn',
#                                         'Cannot perform this action '\
#                                         f'while {_name} operation is running.')
#                 return True
#         return False

#     def onAlgmChanged(self, idx):
#         self.algmPrefs.setCurrentIndex(idx)
#         self.algmPanels.setCurrentIndex(idx)

#     def onMinmapChanged(self, idx):
#         if idx > 0:
#             data = self.MinMapsData[idx - 1]
#             self.minPhase_combox.clear()
#             self.minPhase_combox.addItems(np.unique(data))
#             self.minPhase_combox.setCurrentIndex(0)
#         else:
#             self.minPhase_combox.setCurrentIndex(-1)

#     def refresh_minmaps_combox(self):
#         if not self._extThreadRunning():
#             self.minmaps_combox.clear()
#             self.minmaps_combox.addItem('None')
#             self.minmaps_combox.addItems([CF.path2filename(p) for p in self.MinMapsPath])

#     def mask_maps(self, phase_name):
#         if self.minPhase_combox.currentIndex() != -1:

#         # Compute the condition to be used as mask
#             minmap_idx = self.minmaps_combox.currentIndex() - 1
#             minmap = self.MinMapsData[minmap_idx]
#             mask = minmap != phase_name

#         # Mask the maps data
#             masked_maps = []
#             for idx, m in enumerate(self.XMapsDataOrig):
#                 try:
#                     masked_maps.append(np.ma.masked_where(mask, m))
#                 except IndexError: # raises when xray maps with different shapes are loaded
#                     masked_maps.append(m)
#                     # Uncheck the unfitting maps related checkboxes
#                     self.CboxMaps.Cbox_list[idx].setChecked(False)

#             self.maskOn = True
#             self.XMapsData = masked_maps

#         else:
#             self.maskOn = False
#             self.XMapsData = self.XMapsDataOrig.copy()

#         # Update the training area selector widget
#         self.trAreaPicker.update_mapsData(self.XMapsData)



#     def loadModel(self):
#         path, _ = QW.QFileDialog.getOpenFileName(self, 'Import Supervised Model',
#                                                       pref.get_dirPath('in'),
#                                                       'PyTorch Data File (*.pth)')
#         if path:
#             pref.set_dirPath('in', os.path.dirname(path))
#             self.model_path.set_fullpath(path, predict_display=True)
#             logpath = CF.extend_filename(path, '_log', ext='.txt')

#             # If model log was deleted or moved, ask for rebuilding it
#             if not os.path.exists(logpath):
#                 choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                                  'Unable to find model log file. Rebuild it?',
#                                                  QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                                  QW.QMessageBox.Yes)
#                 if choice == QW.QMessageBox.Yes:
#                     model_vars = ML_tools.loadModel(path)
#                     extendedLog = pref.get_setting('class/extLog', False, bool)
#                     ML_tools.saveModelLog(model_vars, logpath, extendedLog)

#             # Try to load the log file anyways, no error will be raised if file still doesn't exist
#             self.modelInfo.setDoc(logpath)


#     def getSelectedMapsData(self, coord_maps=False):
#         selected_cbox = filter(lambda cbox: cbox.isChecked(), self.CboxMaps.Cbox_list)
#         maps = [self.XMapsData[int(cbox.objectName())] for cbox in selected_cbox]
#         if coord_maps:
#             _shape = maps[0].shape
#             xx, yy = np.meshgrid(np.arange(_shape[1]), np.arange(_shape[0]))
#             if self.maskOn:
#                 mask = maps[0].mask
#                 xx = np.ma.masked_where(mask, xx)
#                 yy = np.ma.masked_where(mask, yy)
#             maps.extend([xx, yy])
#         return maps


#     def getTrainingData(self, pixel_proximity=False, norm_data=True):
#         trAreas = self.trAreaPicker.get_trAreasData()
#         if not len(trAreas):
#             QW.QMessageBox.critical(self, 'X-Min Learn',
#                                     'Please draw training areas first.')
#             return

#     # Merge all selected maps into a single 2D array (nPixels x nMaps).
#     # Also check for same shape. if <pixel_proximity> is True, x and y coords
#     # maps will also be computed.
#         maps = self.getSelectedMapsData(pixel_proximity)
#         X, ok = CF.MergeMaps(maps, mask=self.maskOn)
#         if not ok:
#             QW.QMessageBox.critical(self, 'X-Min Learn',
#                                     'The selected maps have different shapes.')
#             return

#     # Standardize feature (X) data if required
#         if norm_data:
#             if pixel_proximity:
#                 # Pixel coord data is normalized in [0, 1] to reduce their weight on prediction
#                 xmaps, coord = np.split(X, [-2], axis=1)
#                 X[:, :-2] = ML_tools.norm_data(xmaps, return_standards=False, engine='numpy')
#                 X[:, -2:] = coord/coord.max(0)
#             else:
#                 X = ML_tools.norm_data(X, return_standards=False, engine='numpy')

#     # Build an empty (dummy) 2D array of targets (maps shape).
#     # Then fill it with training areas data
#         mapShape = maps[0].shape
#         dummy_Y = np.empty(mapShape, dtype='U8')
#         for (r0,r1,c0,c1), value in trAreas:
#             dummy_Y[r0:r1, c0:c1] = value

#     # If there is a mask, apply it to the targets array and flatten it
#     # to a 1D array (nPixels x 1), excluding the masked indices. Otherwise just flatten it.
#         if self.maskOn:
#             mask = maps[0].mask
#             Y = np.ma.masked_where(mask, dummy_Y).compressed()
#         else:
#             Y = dummy_Y.flatten()

#     # Extract the indices of training pixels from flattened targets array.
#     # Then use that index to get x_train data and y_train data
#         train_idx = (Y != '').nonzero()[0]
#         x_train = X[train_idx, :]
#         y_train = Y[train_idx]

#         return x_train, y_train, X

#     def recolor_plots(self):
#         self.update_resultBars(self.minMap)
#         if self.algm_rawIn is not None: # last algorithm used was a clustering one
#             self.silhouetteCanvas.alterColors(self.resultCanvas.get_colorDict(keys='lbl'))

#     def update_resultBars(self, mapData):
#         lbl, mode = CF.get_mode(mapData, ordered=True)
#         col_dict = CF.sort_dict_by_list(self.resultCanvas.get_colorDict(keys='lbl'), lbl)
#         self.resultBars.update_canvas('Mode', mode, lbl, colors=list(col_dict.values()))

#     def set_confidence(self, conf, pred, prob):
#         return np.where(prob>=conf, pred, '_ND_')

#     def update_resultPlots(self, minMap):
#         self.resultCanvas.update_canvas('Mineral Map', minMap)
#         self.resultLegend.update()
#         self.ND_Canvas.update_canvas('Unclassified Pixels', minMap=='_ND_')
#         self.update_resultBars(minMap)

#     def start_classification(self):
#         if not self._extThreadRunning():
#             selected_cbox = filter(lambda cbox: cbox.isChecked(), self.CboxMaps.Cbox_list)
#             if len(list(selected_cbox)) == 0:
#                 return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                                'Please select at least one map.')
#             algm = self.algm_combox.currentText()

#             if algm == 'Pre-trained Model':
#                 self.run_preTrainedClassifier()

#             elif algm == 'KNN':
#                 self.run_KNNClassfier()

#             elif algm == 'K-Means':
#                 self.run_KMeansClassifier()

#             else : print(f'{algm} not implemented yet')


#     def end_classification(self, success, results, clust_rawInput=None):
#         if success:
#             pred, prob = results

#         # Apply confidence treshold
#             conf = self.conf_spbox.value()
#             pred = self.set_confidence(conf, pred, prob)

#         # Save the algorithm raw output (flattened predictions after confidence treshold) and
#         # raw input (flattened maps stack). This is useful only for clustering results,
#         # which returns also a 'clust_rawInput' arg. Useful for clustering scores analysis
#             self.algm_rawOut = pred
#             self.algm_rawIn = clust_rawInput

#         # Reconstruct the result
#             mapsData = self.getSelectedMapsData()
#             outShape = mapsData[0].shape

#             # If a sub-phase analysis was requested we need to apply the mask to the output.
#             # This is done by creating an empty <outShape> shaped array of '_No{phaseName}' values.
#             # Then the correct indices (rows, cols) where to insert the classified data are taken
#             # from the mask of one of the input maps (the 1st, since they all share the same mask).
#             # For the probability map, the result is simply reshaped to <outShape> and the mask is applied
#             # to it by using the numpy 'masked_where' function.
#             if self.maskOn:
#                 mask = mapsData[0].mask
#                 rows, cols = (mask==0).nonzero()

#                 minmap = np.empty(outShape, dtype='U8')
#                 noPhase = f'_No{self.minPhase_combox.currentText()}'
#                 minmap[:, :] = noPhase
#                 minmap[rows, cols] = pred

#                 pmap = np.zeros(outShape)
#                 pmap[rows, cols] = prob
#                 pmap = np.ma.masked_where(mask, pmap)

#             else:
#                 minmap = pred.reshape(outShape)
#                 pmap = prob.reshape(outShape)

#         # Save the mineral map and the probability map in memory
#             self.minMap = minmap
#             self.probMap = pmap

#         # Update the result widgets
#             self.update_resultPlots(self.minMap)

#             self.save_btn.setEnabled(True)
#             QW.QMessageBox.information(self, 'X-Min Learn',
#                                        'Classification completed succesfully.')
#         else:
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                 'An error occurred during the classification.',
#                 detailedText = repr(results[0]))

#         self.progBar.reset()



#     def save_result(self):
#         if not self._extThreadRunning():
#             outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save Mineral Map',
#                                                 pref.get_dirPath('out'),
#                                                 '''Compressed ASCII file (*.gz)
#                                                    ASCII file (*.txt)''')
#             if outpath:
#                 try:
#                     pref.set_dirPath('out', os.path.dirname(outpath))
#                     pMap_path = CF.extend_filename(outpath, '_probMap')
#                     np.savetxt(outpath, self.minMap, fmt='%s')
#                     np.savetxt(pMap_path, self.probMap, fmt='%.2f')
#                     QW.QMessageBox.information(self, 'File saved',
#                                                'File saved with success.')
#                 except Exception as e:
#                     cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                                     'An error occurred while saving the files.',
#                                     detailedText = repr(e))
#             # Load automatically the result in MinMap Tab if required
#                 if self.autoLoad_cbox.isChecked():
#                     self.parent._minmaptab.loadMaps([outpath])

#                 # Also refresh the MinMap combobox in the sub-phase group
#                     # self.MinMapsPath.append(outpath)
#                     # self.MinMapsData.append(self.minMap)
#                     self.refresh_minmaps_combox()




#     def run_preTrainedClassifier(self):
#         modelPath = self.model_path.get_fullpath()
#         if modelPath == '':
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            'Please load a model first.')

#         self.progBar.setRange(0, 6)
#         cboxList = self.CboxMaps.Cbox_list

#     # Load model variables
#         modelVars = ML_tools.loadModel(modelPath)
#         missing = ML_tools.missingVariables(modelVars)
#         if len(missing):
#             self.progBar.reset()
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            f'The following variables are missing:\n{missing}')
#         requiredMaps = modelVars['ordered_Xfeat']
#         Y_dict = modelVars['Y_dict']
#         self.progBar.setValue(1)

#     # Check for the presence of all required maps
#         checkedMaps = [c.text() for c in cboxList if c.isChecked()]
#         for m in requiredMaps:
#             if not CF.guessMap(m, checkedMaps, caseSens=True):
#                 self.progBar.reset()
#                 return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                                f'Unable to identify {m} map.')
#         self.progBar.setValue(2)

#     # Reorder the input maps to fit the required maps order
#         cboxDict = {c.text() : self.XMapsData[int(c.objectName())] for c in cboxList}
#         orderedMaps = CF.sort_dict_by_list(cboxDict, requiredMaps)
#         self.progBar.setValue(3)

#     # Merge maps into a single 2D array (shape = nPixels x nmaps)
#         maps_data = orderedMaps.values()
#         X, ok = CF.MergeMaps(maps_data, mask=self.maskOn)
#         if not ok:
#             self.progBar.reset()
#             return QW.MessageBox.critical(self, 'Different shapes detected',
#                                           'The selected maps have different shapes.')
#         self.progBar.setValue(4)


#         try:
#         # Run classification
#             prob, predID = ML_tools.applyModel(modelVars, X)
#             self.progBar.setValue(5)

#         # Convert labels IDs to class names
#             pred = CF.decode_labels(predID, Y_dict)
#             self.progBar.setValue(6)

#             success = True
#             results = (pred, prob.detach().numpy())

#         except Exception as e:
#             success = False
#             results = (e,)

#         finally:
#             self.end_classification(success, results)



#     def run_KNNClassfier(self):
#         proximity = self.KNNproximity_cbox.isChecked()
#         train_data = self.getTrainingData(pixel_proximity=proximity)
#         if train_data:
#             self.progBar.setRange(0, 2)

#         # Get required inputs
#             x_train, y_train, X = train_data
#             n_neigh = self.nNeigh_spbox.value()
#             weights = self.wgtKNN_combox.currentText().lower()
#             self.progBar.setValue(1)

#             try:
#             # Run classification
#                 pred, prob = ML_tools.KNN(X, x_train, y_train, n_neigh, weights)
#                 self.progBar.setValue(2)

#                 success = True
#                 results = (pred, prob)

#             except Exception as e:
#                 success = False
#                 results = (e,)

#             finally:
#                 self.end_classification(success, results)


#     def run_KMeansClassifier(self):
#         self.progBar.setRange(0, 4)
#         proximity = self.kmeansProximity_cbox.isChecked()

#     # Merge all selected maps into a single 2D array (nPixels x nMaps).
#     # Also check for same shape.
#         maps = self.getSelectedMapsData(coord_maps=proximity)
#         X, ok = CF.MergeMaps(maps, mask=self.maskOn)
#         self.progBar.setValue(1)
#         if not ok:
#             self.progBar.reset()
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            'The selected maps have different shapes.')
#     # Standardize the data
#         if proximity:
#         # Pixel coord data is normalized in [0, 1] to reduce their weight on prediction
#             xmaps, coord = np.split(X, [-2], axis=1)
#             X[:, :-2] = ML_tools.norm_data(xmaps, return_standards=False, engine='numpy')
#             X[:, -2:] = coord/coord.max(0)
#         else:
#             X = ML_tools.norm_data(X, return_standards=False, engine='numpy')
#             self.progBar.setValue(2)

#     # Get k-means required parameters
#         n_classes = self.Kclasses_spbox.value()
#         seed = int(self.seedInput.text())
#         self.progBar.setValue(3)

#         try:
#         # Run classification
#             pred, prob = ML_tools.K_Means(X, n_classes, seed)
#             self.progBar.setValue(4)

#             success = True
#             results = (pred.astype('U8'), prob)

#         except Exception as e:
#             success = False
#             results = (e,)

#         finally:
#             self.silhouetteCanvas.clear_canvas() # clear the silhouette plot
#             self.end_classification(success, results, clust_rawInput=X)


#     def start_silhouetteScore(self):
#         X = self.algm_rawIn
#         if X is None: # if it is None, the last algorithm used was not a clustering one
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            'Please run a clustering algorithm first')

#         self.silhouette_pbar.setRange(0, 5)
#         self.startSilhouette_btn.setEnabled(False)

#     # Gathering required input parameters
#         pred = self.algm_rawOut
#         subset_size = int(self.subsetPerc_spbox.value() * len(pred))
#         seed = int(self.silhouetteSeed.text())
#         self.silhouette_pbar.setValue(1)

#     # Permute the data and slice it to obtain the subset
#         np.random.seed(seed)
#         subset_idx = np.random.permutation(len(pred))[:subset_size]
#         X = X[subset_idx, :]
#         pred = pred[subset_idx]
#         self.silhouette_pbar.setValue(2)

#     # Compute the silhouette score (thread)
#         self.silhThread.subtaskCompleted.connect(
#             lambda: self.silhouette_pbar.setValue(self.silhouette_pbar.value() + 1))
#         self.silhThread.set_params(X, pred)
#         self.silhThread.start()

#     # # Compute the overall average silhouette score
#     #     mask = pred != '_ND_' # exclude ND data for the average prediction
#     #     sil_avg = ML_tools.silhouette_metric(X[mask, :], pred[mask], type='avg')
#     #     self.silhouette_pbar.setValue(3)

#     # # Compute the silhouette score for each sample
#     #     sil = ML_tools.silhouette_metric(X, pred, type='all')
#     #     self.silhouette_pbar.setValue(4)


#     def plot_silhouetteScore(self, success, results):
#         if success:
#             sil_avg, sil_sam, pred = results
#         # Plot the result
#             colors = self.resultCanvas.get_colorDict(keys='lbl')
#             sil_by_cluster = {}
#             for cluster_name in np.unique(pred):
#                 sil_by_cluster[cluster_name] = sil_sam[pred == cluster_name]
#             self.silhouetteCanvas.update_canvas(sil_by_cluster, sil_avg, colors)
#             self.silhouette_pbar.setValue(5)

#         else:
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                             'Silhouette score computation failed',
#                             detailedText = repr(results[0]))

#         self.silhouette_pbar.reset()
#         self.startSilhouette_btn.setEnabled(True)

#     def compute_CHIscore(self):
#         X = self.algm_rawIn
#         if X is None: # if it is None, the last algorithm used was not a clustering one
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            'Please run a clustering algorithm first')
#         pred = self.algm_rawOut
#         mask = pred != '_ND_' # exclude ND data
#         score = ML_tools.CHIscore(X[mask, :], pred[mask])
#         self.CHIscore_label.setText(str(score))

#     def compute_DBIscore(self):
#         X = self.algm_rawIn
#         if X is None: # if it is None, the last algorithm used was not a clustering one
#             return QW.QMessageBox.critical(self, 'X-Min Learn',
#                                            'Please run a clustering algorithm first')
#         pred = self.algm_rawOut
#         mask = pred != '_ND_' # exclude ND data
#         score = ML_tools.DBIscore(X[mask, :], pred[mask])
#         self.DBIscore_label.setText(str(score))

#     def closeEvent(self, event):
#         choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                          'Do you really want to close the classification window?',
#                                          QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                          QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:
#             event.accept()
#         else:
#             event.ignore()


class DatasetBuilder(DraggableTool):
    '''
    One of the main tools of X-Min Learn, that allows the semi-automated
    compilation of human-readable and machine-friendly ground truth datasets.
    '''
    def __init__(self):
        '''
        Constructor.

        Parameters

        '''
        super(DatasetBuilder, self).__init__()
    # Set tool title and icon
        self.setWindowTitle('Dataset Builder')
        self.setWindowIcon(QIcon('Icons/compile_dataset.png'))
    # Set main attributes
        self._features = set([])
        self.dataset = None
    # Set GUI
        self._init_ui()
        self.adjustSize()

    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Input grid of chemical elements (Grid Layout --> Group scroll area)
        elements = ['A', 'Ac', 'Ag', 'Al', 'As', 'At', 'Au', 'B',
                    'Ba', 'Be', 'Bi', 'Br', 'C', 'Ca', 'Cd', 'Ce',
                    'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu',
                    'F', 'Fe', 'Fr', 'Ga', 'Gd', 'Ge', 'H', 'He',
                    'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'Kr',
                    'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na',
                    'Nb', 'Nd', 'Ne', 'Ni', 'O', 'Os', 'P', 'Pa',
                    'Pb', 'Pd', 'Pm', 'Po', 'Pr', 'Pt', 'Ra', 'Rb',
                    'Re', 'Rh', 'Rn', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                    'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Tc', 'Te',
                    'Th', 'Ti', 'Ti', 'Tm', 'U', 'V', 'W', 'Xe',
                    'Y', 'Yb', 'Zn', 'Zr']

        elem_grid = QW.QGridLayout()
        n_cols = 8
        for x in range(len(elements)):
            elem_btn = cObj.StyledButton(text=elements[x])
            elem_btn.clicked.connect(self.add_inputFeature)
            row, col = x//n_cols, x%n_cols
            elem_grid.addWidget(elem_btn, row, col)
        # Set fixed column width for equal button size
        # fixed_column_width = 50
        # # for col in range(n_cols):
        # #     elem_grid.setColumnMinimumWidth(col, fixed_column_width)
        elem_scroll = cObj.GroupScrollArea(elem_grid)

    # Custom feature entry (Line Edit)
        self.customEntry_lbl = QW.QLineEdit()
        self.customEntry_lbl.setStyleSheet(pref.SS_menu)
        self.customEntry_lbl.setPlaceholderText('Custom feature name')
        self.customEntry_lbl.returnPressed.connect(self.add_inputFeature)

    # Selected elements list (Styled List Widget)
        self.featureList = cObj.StyledListWidget()

    # Elements list option buttons (--> in HBoxLayout)
        self.delElem_btn = cObj.StyledButton(QIcon('Icons/generic_del.png'))
        self.delElem_btn.setToolTip('Remove selected elements')
        self.delElem_btn.clicked.connect(self.del_inputFeature)

        self.refresh_btn = cObj.StyledButton(QIcon('Icons/forward.png'))
        self.refresh_btn.setToolTip('Refresh Dataset Designer')
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self.refresh_designer)

        elemOptions_hbox = QW.QHBoxLayout()
        elemOptions_hbox.addWidget(self.delElem_btn, alignment=Qt.AlignLeft)
        elemOptions_hbox.addWidget(self.refresh_btn, alignment=Qt.AlignRight)

    # Dataset designer 1 (X features) & Dataset designer 2 (Y ground truth labels)
        self.xfeat_designer = self.XDatasetDesigner(self)
        self.ylab_designer = self.YDatasetDesigner(self)
        table_hsplit = cObj.SplitterGroup((self.xfeat_designer, self.ylab_designer),
                                        (6, 1))

    # Dataset Designer option buttons (--> in HBoxLayout)
        self.addRow_btn = cObj.StyledButton(QIcon('Icons/add_tableRow.png'),
                                            'Add row')
        self.addRow_btn.setEnabled(False)
        self.addRow_btn.clicked.connect(self.add_row)

        self.delRow_btn = cObj.StyledButton(QIcon('Icons/del_tableRow.png'),
                                            'Remove row')
        self.delRow_btn.setEnabled(False)
        self.delRow_btn.clicked.connect(self.del_row)

        self.compile_btn = cObj.StyledButton(QIcon('Icons/compile_dataset.png'),
                                             'Compile Dataset')
        self.compile_btn.setEnabled(False)
        self.compile_btn.clicked.connect(self.compile_dataset)

        table_btns = QW.QHBoxLayout()
        table_btns.setAlignment(Qt.AlignRight)
        table_btns.addWidget(self.addRow_btn)
        table_btns.addWidget(self.delRow_btn)
        table_btns.addWidget(self.compile_btn)

    # Class Refinement List (Styled List Widget)
        self.refineList = cObj.StyledListWidget()

    # Class Refinement Option buttons (--> in VBoxLayout)
        self.renameClass_btn = cObj.StyledButton(text='Rename class')
        self.renameClass_btn.clicked.connect(self.rename_class)

        self.deleteClass_btn = cObj.StyledButton(text='Delete class')
        self.deleteClass_btn.clicked.connect(self.delete_class)

        self.mergeClass_btn = cObj.StyledButton(text='Merge classes')
        self.mergeClass_btn.clicked.connect(self.merge_class)

        refine_btns = QW.QVBoxLayout()
        refine_btns.addWidget(self.renameClass_btn)
        refine_btns.addWidget(self.deleteClass_btn)
        refine_btns.addWidget(self.mergeClass_btn)

    # Dataset preview Area (Text Edit)
        self.previewArea = QW.QTextEdit('')
        self.previewArea.setReadOnly(True)
        self.previewArea.setHorizontalScrollBar(
            cObj.StyledScrollBar(Qt.Horizontal))
        self.previewArea.setVerticalScrollBar(
            cObj.StyledScrollBar(Qt.Vertical))

    # CSV decimal point selector (Combo Box)
        self.decimal_combox = cObj.DecimalPointSelector()

    # CSV separator character selector (Combo Box)
        self.separator_combox = cObj.SeparatorSymbolSelector()

    # Split large dataset option (Checkbox)
        self.splitFile_cbox = QW.QCheckBox('Split dataset')
        self.splitFile_cbox.setChecked(False)
        tip = 'Split dataset into multiple CSV files if the number of lines '\
              'exceeds Microsoft Excel rows limit (about 1 million)'
        self.splitFile_cbox.setToolTip(tip)

    # Save dataset button
        self.saveCSV_btn = cObj.StyledButton(QIcon('Icons/save.png'),
                                             'Save dataset')
        self.saveCSV_btn.clicked.connect(self.save_dataset)
        self.saveCSV_btn.setEnabled(False)

    # Adjust main layout
    # Input features group
        infeat_vbox = QW.QVBoxLayout()
        infeat_vbox.addWidget(self.customEntry_lbl)
        infeat_vbox.addWidget(self.featureList)
        infeat_vbox.addLayout(elemOptions_hbox)
        infeat_vsplit = cObj.SplitterGroup((elem_scroll, infeat_vbox), (0, 1),
                                           orient=Qt.Vertical)
        infeat_group = cObj.GroupArea(infeat_vsplit, 'Input features')
    # Dataset designer group
        designer_vbox = QW.QVBoxLayout()
        designer_vbox.addWidget(table_hsplit)
        designer_vbox.addLayout(table_btns)
        designer_group = cObj.GroupArea(designer_vbox, 'Dataset Designer')
    # Dataset refinement group
        refine_hbox = QW.QHBoxLayout()
        refine_hbox.addWidget(self.refineList)
        refine_hbox.addLayout(refine_btns)
        refine_group = cObj.GroupArea(refine_hbox, 'Dataset Refinement')
    # Dataset preview group
        preview_hbox = QW.QHBoxLayout()
        preview_hbox.addWidget(self.previewArea)
        preview_group = cObj.GroupArea(preview_hbox, 'Dataset Info')
    # CSV file preferences group
        csvPref_vbox = QW.QVBoxLayout()
        csvPref_vbox.addWidget(QW.QLabel('CSV decimal point'))
        csvPref_vbox.addWidget(self.decimal_combox)
        csvPref_vbox.addWidget(QW.QLabel('CSV separator'))
        csvPref_vbox.addWidget(self.separator_combox)
        csvPref_vbox.addWidget(self.splitFile_cbox)
        csvPref_vbox.addWidget(self.saveCSV_btn)
        csvPref_group = cObj.GroupArea(csvPref_vbox, 'File preferences')
    # Set main layout
        top_hsplit = cObj.SplitterGroup((infeat_group, designer_group), (1, 2))
        bot_hsplit = cObj.SplitterGroup((refine_group, preview_group,
                                         csvPref_group), (2, 2, 1))
        main_layout = cObj.SplitterLayout(Qt.Vertical)
        main_layout.addWidgets((top_hsplit, bot_hsplit), (2, 1))
        self.setLayout(main_layout)


    def add_inputFeature(self):
        '''
        Add a new input feature.

        '''
        elem = self.sender().text()
        if elem != '':
            self._features.add(elem)
            self._update_featureList()


    def del_inputFeature(self):
        '''
        Remove selected input features.

        '''
        selected = self.featureList.selectedItems()
        for item in selected:
            self._features.remove(item.text())
        self._update_featureList()


    def _update_featureList(self):
        '''
        Update the list of input features.

        '''
        self.featureList.clear()
        self.featureList.addItems(sorted(self._features))
        self.refresh_btn.setEnabled(len(self._features) > 0)


    def refresh_designer(self):
        '''
        Confirm the selected input features and populate the Dataset Designer.

        '''
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         'Do you want to refresh the Dataset '\
                                         'Designer? All previously loaded '\
                                         'maps will be removed.',
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            cols = sorted(self._features)
            self.xfeat_designer.refresh(cols)
            self.ylab_designer.refresh()
            self.addRow_btn.setEnabled(True)
            self.delRow_btn.setEnabled(True)
            self.compile_btn.setEnabled(True)


    def add_row(self):
        '''
        Add one row to the Dataset Designer.

        '''
        self.xfeat_designer.addRow()
        self.ylab_designer.addRow()


    def del_row(self):
        '''
        Remove last row from the Dataset Designer.

        '''
        self.xfeat_designer.delRow()
        self.ylab_designer.delRow()


    def compile_dataset(self):
        '''
        Extract the data from the loaded samples and compile the Ground Truth
        Dataset.

        '''
        checked_samples = []
        tot_rows = self.xfeat_designer.rowCount()
        progBar = cObj.PopUpProgBar(self, tot_rows, 'Compiling Dataset',
                                    cancel=False)

    # Check 0 --> no samples (0 rows)
        if tot_rows == 0:
            progBar.reset()
            return QW.QMessageBox.critical(self, 'X-Min Learn',
                                           'The Dataset Designer is empty.')

        for row in range(tot_rows):
            xfeat = self.xfeat_designer.getRowData(row)
            ylab = self.ylab_designer.getRowData(row)
            sample = xfeat + ylab

        # Check 1 --> missing maps
            if min(map(len, sample)) == 0:
                progBar.reset()
                return QW.QMessageBox.critical(self, 'X-Min Learn',
                                               f'Missing maps in row {row+1}')

        # Check 2 --> wrong maps shape within same sample
            warnings = 0
            shapes = [s.shape for s in sample]
            shp_mode = math_mode(shapes)
            unfitting = [s != shp_mode for s in shapes]
            for col, u in enumerate(unfitting, start=1):
                warnings += u #boolean as 0 or 1
                if col == len(unfitting): # last column is always the Y column
                    self.ylab_designer.cellWidget(row, 0).setWarning(u)
                else:
                    self.xfeat_designer.cellWidget(row, col).setWarning(u)

            if warnings > 0:
                progBar.reset()
                return QW.QMessageBox.warning(self, 'X-Min Learn',
                                              f'A total of {warnings} maps '\
                                              'do not fit the shape of the '\
                                              f'sample in row {row+1}. '\
                                              '\nTip: They are highlighted '\
                                              'with a yellow line.')

        # All checks completed --> add the sample to 'checked_samples'
            checked_samples.append(sample)

    # Prepare the dataset for compilation
        headers = self.xfeat_designer.col_lbls[1:] + ['Class']
        self.dataset = DataFrame(columns = headers)

    # For each sample link to every column (key) a flattened 1D array (value)
    # and then merge all samples into a singular dataset (concat)
        for n, sam in enumerate(checked_samples, start=1):
            sample_dict = dict(zip(headers, [arr.flatten() for arr in sam]))
            self.dataset = concat([self.dataset, DataFrame(sample_dict)],
                                   ignore_index=True)
            # self.dataset = self.dataset.append(DataFrame(sample_dict), ignore_index=True)
            progBar.setValue(n)

        self._update_unique_classes()
        self.saveCSV_btn.setEnabled(True)
        return QW.QMessageBox.information(self, 'X-Min Learn',
                                          'Dataset compiled with success.')


    def _update_unique_classes(self):
        '''
        Update the list of unique classes found in the dataset.

        '''
        unq = self.dataset.Class.unique()
        self.refineList.clear()
        self.refineList.addItems(sorted(unq))
        self._update_dataset_preview()


    def _update_dataset_preview(self):
        '''
        Update the Dataset Preview.

        '''
        preview = repr(self.dataset)
        class_count = repr(self.dataset.Class.value_counts())
        text = f'DATAFRAME PREVIEW\n\n{preview}\n\n\n'\
               f'PER-CLASS DATA COUNT\n\n{class_count}'
        self.previewArea.setText(text)


    def isClassNameAvailable(self, name):
        '''
        Check if a class name is available or if it is already present in the
        dataset. If not available, user can still choose to overwrite it.

        Parameters
        ----------
        name : str
            The class name.

        Returns
        -------
        available : bool
            Wether or not the class name is available.

        '''
        available = name not in self.dataset.Class.values
        if not available:
            choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                             f'{name} already exists in the '\
                                            'dataset. Overwrite it?',
                                            QW.QMessageBox.Yes | QW.QMessageBox.No,
                                            QW.QMessageBox.No)
            available = choice == QW.QMessageBox.Yes
        return available


    def rename_class(self):
        '''
        Rename a class in the dataset.

        Returns
        -------
        None.

        '''
        selected = self.refineList.selectedItems()
    # Just exit function if no classes are selected
        if len(selected) == 0:
            return
    # Only one class can be renamed at a time (send warning)
        elif len(selected) > 1:
            return QW.QMessageBox.warning(self, 'X-Min Learn',
                                          'Rename one class at a time')
    # Length of <selected> is one, so <item> is the first element in the list
        else:
            item = selected[0]
            new_name, ok = QW.QInputDialog.getText(self, 'X-Min Learn',
                                                  f'Rename {item.text()}:',
                                                  flags=Qt.MSWindowsFixedSizeDialogHint)
            if ok and self.isClassNameAvailable(new_name):
                progBar = cObj.PopUpProgBar(self, 1, cancel=False)
                progBar.setValue(0)
                self.dataset.replace(item.text(), new_name, inplace=True)
                self._update_unique_classes()
                progBar.setValue(1)


    def delete_class(self):
        '''
        Remove data with the selected classes from the dataset.

        '''
        selected = self.refineList.selectedItems()
        if len(selected) > 0:
            choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                             'Remove the selected classes?',
                                             QW.QMessageBox.Yes | QW.QMessageBox.No,
                                             QW.QMessageBox.No)
            if choice == QW.QMessageBox.Yes:
                targets = [item.text() for item in selected]
                progBar = cObj.PopUpProgBar(self, 1, cancel=False)
                progBar.setValue(0)
                self.dataset = self.dataset[~self.dataset.Class.isin(targets)]
                self.dataset.reset_index(drop=True, inplace=True)
                self._update_unique_classes()
                progBar.setValue(1)


    def merge_class(self):
        '''
        Unify two or more classes in the dataset under a new name.

        '''
        selected = self.refineList.selectedItems()
    # Just exit function if no classes are selected
        if len(selected) == 0:
            return
    # At least two classes must be selected (send warning)
        elif len(selected) == 1:
            return QW.QMessageBox.warning(self, 'X-Min Learn',
                                   'Select at least two classes to merge')
        else:
            targets = [item.text() for item in selected]
            name, ok = QW.QInputDialog.getText(self, 'Merge classes',
                                                  f'Merge {targets} and '\
                                                  'rename them as:',
                                                  flags=Qt.MSWindowsFixedSizeDialogHint)
            if ok and self.isClassNameAvailable(name):
                progBar = cObj.PopUpProgBar(self, 1, cancel=False)
                progBar.setValue(0)
                self.dataset.replace(targets, [name]*len(targets), inplace=True)
                self._update_unique_classes()
                progBar.setValue(1)


    def save_dataset(self):
        '''
        Save the ground truth dataset as one or multiple CSV file(s).

        '''
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save new dataset',
                                                    pref.get_dirPath('out'),
                                                    'CSV file (*.csv)')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            dec = self.decimal_combox.currentText()
            sep = self.separator_combox.currentText()

        # Set the number of output csv files (subsets) if requested
            nfiles = 1
            if self.splitFile_cbox.isChecked():
                batch_size = 2**20 # Excel maximum number of rows
                nfiles += len(self.dataset) // batch_size

        # Disable the save button during saving operations, then re-enable it
            self.sender().setEnabled(False)
            subsets = np.array_split(self.dataset, nfiles)
            progBar = cObj.PopUpProgBar(self, len(subsets), 'Saving Dataset')

            for idx, ss in enumerate(subsets, start=1):
                if progBar.wasCanceled(): 
                    break
                elif (nfiles - 1): # if there is more than one file to save
                    path = CF.extend_filename(outpath, str(idx)) 
                else:
                    path = outpath

                ss.to_csv(path, sep=sep, index=False, decimal=dec)
                progBar.setValue(idx)

            self.sender().setEnabled(True)

            return QW.QMessageBox.information(self, 'X-Min Learn',
                                              'Dataset saved succesfully.')


    class XDatasetDesigner(cObj.StyledTable):
        '''
        Table widget for the Dataset Builder tool. It allows the user to import
        input maps data for the automatic compilation of ground truth datasets.
        '''

        def __init__(self, parent=None):
            '''
            Constructor.

            Parameters
            ----------
            parent : QWidget or None, optional
                The GUI parent of this widget. The default is None.

            '''
        # Call the constructor of the parent class
            self.parent = parent
            super().__init__(1, 1, parent=self.parent)

        # Initialize the horizontal header
            self.horizontalHeader().setSectionResizeMode(1) # Stretch
            self.col_lbls = ['Input Maps']
            self.setHorizontalHeaderLabels(self.col_lbls)


        def init_firstCol(self, row):
            '''
            Initialize the first column of the table to include a special button
            that allows user to import the entire row of data.

            Parameters
            ----------
            row : int
                Index of row.

            '''
        # Set resize mode to ResizeToContent(3) only for first column(0)
            self.horizontalHeader().setSectionResizeMode(0, 3)

        # Special button to fill the entire row of data
            fillRow_btn = cObj.StyledButton(QIcon('Icons/generic_add_blue.png'))
            fillRow_btn.setObjectName(str(row)) # to easily access the row
            fillRow_btn.setToolTip('Fill entire row')
            fillRow_btn.clicked.connect(self.smart_fillRow)
            self.setCellWidget(row, 0, fillRow_btn)


        def init_rowWidgets(self, row):
            '''
            Initialize the widgets (see DatasetDesignerWidget class) held by each
            cell in the given row.

            Parameters
            ----------
            row : int
                Index of row.

            '''
        # Initialize the DatasetDesignerWidgets in every column except the 
        # first, because it holds the special "fill row" button (for details  
        # see init_firstCol function)
            for col in range(1, self.columnCount()):
                col_lbl = self.horizontalHeaderItem(col).text()
                wid = self.parent.DatasetDesignerWidget('input', col_lbl, self)
                self.setCellWidget(row, col, wid)

        # Set resize mode of vertical header to ResizeToContent(3)
            self.verticalHeader().setSectionResizeMode(row, 3)


        def refresh(self, columns):
            '''
            Reset and re-initialize the table with new column names.

            Parameters
            ----------
            columns : list
                List of column names.

            '''
        # Clear all and set new column names
            self.clear()
            self.setColumnCount(len(columns) + 1)
            self.col_lbls = [''] + columns
            self.setHorizontalHeaderLabels(self.col_lbls)

        # Initialize the table
            for row in range(self.rowCount()):
                self.init_firstCol(row)
                self.init_rowWidgets(row)


        def addRow(self):
            '''
            Add a row to the table.

            '''
            row = self.rowCount()
            self.insertRow(row)
            self.init_firstCol(row)
            self.init_rowWidgets(row)


        def delRow(self):
            '''
            Remove last row from the table.

            '''
            row = self.rowCount() - 1
            if row >= 0:
                self.removeRow(row)


        def smart_fillRow(self):
            '''
            Try to fill an entire row of the table by automatically importing 
            and ordering multiple input maps based on their filename.

            '''
            paths, _ = QW.QFileDialog.getOpenFileNames(self, 'Load input maps',
                                                        pref.get_dirPath('in'),
                                                        'ASCII maps (*.txt *.gz)')
            if paths:
                pref.set_dirPath('in', os.path.dirname(paths[0]))
                progBar = cObj.PopUpProgBar(self, len(paths), 'Loading Maps')
                for n, p in enumerate(paths, start=1):
                    if progBar.wasCanceled(): break
                    try:
                        name = CF.path2filename(p)
                        matching_col = CF.guessMap(name, self.col_lbls) # !!! find a more elegant solution
                    # If a filename matches with a column, then add it
                        if matching_col:
                            row = int(self.sender().objectName())
                            col = self.col_lbls.index(matching_col)
                            self.cellWidget(row, col).addMap(p)

                    except Exception as e:
                        progBar.setWindowModality(Qt.NonModal)
                        cObj.RichMsgBox(self, QW.QMessageBox.Critical,
                                        'X-Min Learn',
                                        f'Unexpected ASCII file:\n{p}.',
                                        detailedText = repr(e))
                        progBar.setWindowModality(Qt.WindowModal)

                    finally:
                        progBar.setValue(n)


        def getRowData(self, row):
            '''
            Extract map data from the given row.

            Parameters
            ----------
            row : int
                Index of row.

            Returns
            -------
            rowData : list
                List of map data in the form of numpy arrays.

            '''
            rowData = []
            for col in range(self.columnCount()):
                if col != 0:
                    rowData.append(self.cellWidget(row, col).data)
            return rowData


    class YDatasetDesigner(cObj.StyledTable):
        '''
        Table widget for the Dataset Builder tool. It allows the user to import
        labels in the form of already classified mineral maps data in order to
        automatically compile a ground truth dataset.
        '''

        def __init__(self, parent=None):
            '''
            YDatasetDesigner class constructor

            Parameters
            ----------
            parent : QWidget or None, optional
                The GUI parent of the widget. The default is None.

            '''
        # Call the constructor of the parent class
            self.parent = parent
            super().__init__(1, 1, parent=self.parent)

        # Initialize the horizontal header
            self.horizontalHeader().setSectionResizeMode(1) # Stretch
            self.col_lbl = ['Mineral Map']
            self.setHorizontalHeaderLabels(self.col_lbl)


        def init_rowWidget(self, row):
            '''
            Initialize the widgets (see DatasetDesignerWidget class) held by the
            cell in the given row.

            Parameters
            ----------
            row : int
                Index of row.

            '''
        # Initialize the DatasetDesignerWidget in the one and only cell
            wid = self.parent.DatasetDesignerWidget('output', self.col_lbl, self)
            self.setCellWidget(row, 0, wid)

        # Set resize mode of vertical header to ResizeToContent(3)
            self.verticalHeader().setSectionResizeMode(row, 3)


        def refresh(self):
            '''
            Reset and re-initialize the table.

            '''
            self.clear()
            self.setHorizontalHeaderLabels(self.col_lbl)
            for row in range(self.rowCount()):
                self.init_rowWidget(row)


        def addRow(self):
            '''
            Add a row to the table.

            '''
            row = self.rowCount()
            self.insertRow(row)
            self.init_rowWidget(row)


        def delRow(self):
            '''
            Remove last row from the table.

            '''
            row = self.rowCount() - 1
            if row >= 0:
                self.removeRow(row)


        def getRowData(self, row):
            '''
            Extract map data from the given row.

            Parameters
            ----------
            row : int
                Index of row.

            Returns
            -------
            rowData : list
                List (of one element) of map data in the form of numpy arrays.

            '''
            rowData = [self.cellWidget(row, 0).data]
            return rowData


    class DatasetDesignerWidget(QW.QWidget):
        '''
        Base widget used by the Dataset Designer tool to load and manage map
        data. See XDatasetDesigner and YDatasetDesigner classes for details.
        '''

        def __init__(self, mapType, mapName, parent=None):
            '''
            Constructor.

            Parameters
            ----------
            mapType : str
                Describer for the role of map data. Must be 'input' or 'output'.
            mapName : str
                Name of the map.
            parent : QTableWidget or None, optional
                The table that holds this widget. The default is None.

            '''
        # Raise error if mapType is not valid
            assert mapType in ('input', 'output')

        # Call the constructor of the parent class
            self.parent = parent
            super().__init__(parent)

        # Define main attributes
            self.mapType = mapType
            self.mapName = mapName
            self.data = np.array([])

        # Add file button
            self.add_btn = cObj.StyledButton(QIcon('Icons/smooth_add.png'))
            self.add_btn.setFlat(True)
            self.add_btn.clicked.connect(lambda: self.addMap())

        # Remove file button
            self.del_btn = cObj.StyledButton(QIcon('Icons/smooth_del.png'))
            self.del_btn.setFlat(True)
            self.del_btn.setEnabled(False)
            self.del_btn.clicked.connect(self.delMap)

        # Info color line (Frame)
            self.infoFrame = QW.QFrame()
            self.infoFrame.setFrameStyle(QW.QFrame.HLine | QW.QFrame.Plain)
            self.infoFrame.setLineWidth(3)
            self.infoFrame.setMidLineWidth(3)
            self.infoFrame.setStyleSheet("color: red;")

        # Adjust Layout
            layout = QW.QVBoxLayout()
            layout.addWidget(self.add_btn, alignment=Qt.AlignHCenter)
            layout.addWidget(self.del_btn, alignment=Qt.AlignHCenter)
            layout.addWidget(self.infoFrame)
            self.setLayout(layout)


        def addMap(self, path=None):
            '''
            Load map data from file.

            Parameters
            ----------
            path : Path-like or None, optional
                Map filepath. If None, it can be selected interactively. The
                default is None.

            '''
        # If path is missing, prompt user to load a file, defining the expected
        # file extension based on map role
            if path == None:
                if self.mapType == 'input':
                    filext_filter = 'ASCII file (*.txt *.gz)'
                else:
                    filext_filter = '''Mineral maps (*.mmp)
                                       Legacy mineral maps (*.txt *.gz)'''
                path, _ = QW.QFileDialog.getOpenFileName(self, f'Load {self.mapName} Map',
                                                         pref.get_dirPath('in'),
                                                         filext_filter)
        # If path is valid, load the data
            if path:
                pref.set_dirPath('in', os.path.dirname(path))
                try:
                    if self.mapType == 'input':
                        self.data = InputMap.load(path).map
                    else:
                        self.data = MineralMap.load(path).minmap

                    self.add_btn.setEnabled(False)
                    self.del_btn.setEnabled(True)
                    self.infoFrame.setStyleSheet("color: lightgreen;")
                    self.setToolTip(path)

                except ValueError:
                    return QW.QMessageBox.critical(self.parent.parent,
                                                    'X-Min Learn',
                                                    f'Unexpected file:\n{path}.')


        def delMap(self):
            '''
            Remove loaded map data.

            '''
            self.add_btn.setEnabled(True)
            self.del_btn.setEnabled(False)
            self.data = np.array([])
            self.infoFrame.setStyleSheet("color: red;")
            self.setToolTip('')


        def setWarning(self, show):
            '''
            Show/hide a warning in the form of a yellow info line (Frame).

            Parameters
            ----------
            show : bool
                Show/hide the warning.

            '''
            color = 'yellow' if show else 'lightgreen'
            self.infoFrame.setStyleSheet("color: %s;" %(color))



class PixelEditor(QW.QWidget):
    def __init__(self, XMapsInfo, originalData, edits, parent):
        self.parent = parent
        self.XMapsPath, self.XMapsData = XMapsInfo
        super(PixelEditor, self).__init__()
        self.setWindowTitle('Pixel Editor')
        self.setWindowIcon(QIcon('Icons/edit.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.editsDict = edits
        self.originalData = originalData
        self.editedData = self.originalData.copy()
        for (x,y), val in self.editsDict.items():
            self.editedData[x,y] = val
        self.tempResult = None

        self.init_ui()
        self.adjustSize()
        self.update_preview()

    def init_ui(self):

        # Edited Pixels Preview Area
        self.editPreview = cObj.HeatMapCanvas(size=(6, 4.5), binary=True, tight=True,
                                              cbar=False, wheel_zoom=False)
        self.editPreview.setMinimumSize(100, 100)

        # Progress bar
        self.progBar = QW.QProgressBar()

        # Current edited value combo box
        self.currEdit_combox = QW.QComboBox()
        self.currEdit_combox.addItems(set(self.editsDict.values()))
        self.currEdit_combox.currentTextChanged.connect(self.current_preview)

        # Preview button
        self.showPreview_btn = QW.QPushButton('Preview')
        self.showPreview_btn.setToolTip('Show preview (it may require some time).')
        self.showPreview_btn.clicked.connect(self.update_preview)

        # Save Button
        self.save_btn = cObj.IconButton('Icons/save.png', 'Save Edits')
        self.save_btn.clicked.connect(self.save_edits)

        # Training Mode Preferences
        self.tol_spbox = QW.QSpinBox()
        self.tol_spbox.setToolTip('Select the pixel tolerance for computation.')
        self.tol_spbox.setValue(15)
        self.tol_spbox.setRange(0, 10000)
        self.tol_spbox.valueChanged.connect(self.reset_tempResult)
        tolerance = QW.QFormLayout()
        tolerance.addRow('Set Tolerance', self.tol_spbox)

        self.proximity_cbox = QW.QCheckBox('Evaluate Proximity')
        self.proximity_cbox.setToolTip('Include pixel proximity effect in computation.')
        self.proximity_cbox.setChecked(True)
        self.proximity_cbox.clicked.connect(self.reset_tempResult)

        trainBox = QW.QVBoxLayout()
        trainBox.addLayout(tolerance)
        trainBox.addWidget(self.proximity_cbox)
        self.trainGA = cObj.GroupArea(trainBox, 'Training Mode', checkable=True)
        self.trainGA.setChecked(False)
        self.trainGA.toggled.connect(self.toggleTrainMode)
        # self.trainGA.setMaximumWidth(180)

        # Input Maps Checkbox
        self.CboxMaps = cObj.CBoxMapLayout(self.XMapsPath)
        for cbox in self.CboxMaps.Cbox_list:
            cbox.clicked.connect(self.reset_tempResult)
        MapsGSA = cObj.GroupScrollArea(self.CboxMaps, 'Input Maps')
        # MapsGSA.setMaximumWidth(180)

        # Adjust Layout
        rightVbox = QW.QVBoxLayout()
        rightVbox.addWidget(self.currEdit_combox)
        rightVbox.addWidget(self.showPreview_btn)
        rightVbox.addWidget(self.save_btn)
        rightVbox.addWidget(self.progBar)
        rightVbox.addStretch()
        rightVbox.addWidget(self.trainGA)

        layout = QW.QHBoxLayout()
        layout.addWidget(MapsGSA, 1)
        layout.addWidget(self.editPreview, 2)
        layout.addLayout(rightVbox, 1)
        self.setLayout(layout)


    def reset_tempResult(self):
        self.tempResult = None

    def save_edits(self):
        if self.tempResult is None:
            self.update_preview()

        editedMap = self.tempResult
        outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save edited map',
                                                    pref.get_dirPath('out'),
                                                    '''Compressed ASCII file (*.gz)
                                                       ASCII file (*.txt)''')
        if outpath:
            pref.set_dirPath('out', os.path.dirname(outpath))
            np.savetxt(outpath, editedMap, fmt='%s')
            QW.QMessageBox.information(self, 'X-Min Learn',
                                       'Edited map saved with success.')

    def mapsFitting(self, maps):
        # Check for maps fitting errors
        shapes = [arr.shape for arr in maps]
        for shp in shapes:
            if shp != self.originalData.shape: #there are different map shapes
                QW.QMessageBox.critical(self, 'X-Min Learn',
                                        'The selected maps have different shapes.')
                return (False, None)
        else:
            return (True, self.originalData.shape)

    # def calc_pixAffinity(self):
    #     tol = self.tol_spbox.value()
    #     prox = self.proximity_cbox.isChecked()
    #     xmaps_cbox = filter(lambda cbox: cbox.isChecked(), self.CboxMaps.Cbox_list)
    #     xmaps_data = [self.XMapsData[int(cbox.objectName())] for cbox in xmaps_cbox]

    #     mapsfit, shape2D = self.mapsFitting(xmaps_data)
    #     if mapsfit:
    #         progBar = cObj.PopUpProgBar(self, shape2D[0], 'Computing pixel editing')
    #         shape3D = (len(xmaps_data), shape2D[0], shape2D[1])
    #         xmaps_arr = np.array(xmaps_data).reshape(shape3D)
    #         signature = {((x,y), val): xmaps_arr[:, x, y] for (x, y), val in self.editsDict.items()}
    #         outData = self.originalData.copy()

    #         for x in range (shape3D[1]):
    #             if not progBar.wasCanceled():
    #                 for y in range (shape3D[2]):
    #                     voxel = xmaps_arr[:, x, y]

    #                     fitting = []
    #                     for (coord, val), sign in signature.items():
    #                         near_idx = prox * tol/(((x - coord[0])**2 + (y - coord[1])**2)**0.5 + 1)
    #                         variance = np.abs(voxel - sign) - near_idx
    #                         if np.all(variance <= tol):
    #                             fitting.append((variance.sum(), val))
    #                     if len(fitting) > 0:
    #                         best_fit, new_lbl = min(fitting)
    #                         outData[x, y] = new_lbl

    #                 progBar.setValue(x+1)
    #             else:
    #                 outData = self.editedData.copy()
    #                 break


    #     else:
    #         outData = self.editedData.copy()

    #     return outData

    def calc_pixAffinity(self):
    # Get affinity parameters
        tol = self.tol_spbox.value()
        prox = self.proximity_cbox.isChecked()

    # Get input maps data
        xmaps_cbox = filter(lambda cbox: cbox.isChecked(), self.CboxMaps.Cbox_list)
        xmaps_data = [self.XMapsData[int(cbox.objectName())] for cbox in xmaps_cbox]
        n_maps = len(xmaps_data)

    # Check that input maps have same shape
        mapsfit, shape2D = self.mapsFitting(xmaps_data)
        if mapsfit:
            self.progBar.setMaximum(len(self.editsDict))
        # Build a 3D maps data stacked array (-> shape = (n_rows x n_cols x n_maps))
            xmaps_stack = np.dstack(xmaps_data)
        # Build a boolean mask highlighting pixels edited by user
            edit_mask = self.editedData != self.originalData
        # Initialize the output map as a copy of the original mineral map
            outData = self.originalData.copy()
        # Initialize the minimum variance 2D matrix with the Maximum Accepted Tolerance (MAT) + 1
        # The MAT is equal to the tolerance value set by user multiplied by the number of input maps
            min_variance = np.ones(outData.shape) * ((tol*n_maps)+1)

        # START ITERATION for each edited pixel (r and c are the i-th pixel indices (row and column))
            for n, (r, c) in enumerate(zip(*np.where(edit_mask)), start=1):
            # Get the edited pixel value through its indices
                value = self.editedData[r, c]
            # Get the rows and columns indices (both 2D matrix (n_rows x n_cols)) of a single input map
            # in the stack (they are the same for each map, because maps have the same shape)
                rows_idx, cols_idx = np.indices(xmaps_stack.shape[:2])
            # Confront each row and column index with those of the i-th edited pixel in the iteration.
            # 'near_idx' will be a 2D matrix (n_rows x n_cols) storing the proximity values of every
            # pixel with respect to the i-th edited pixel in the iteration. 'prox' is a boolean value
            # referred to user's request to include proximity calculation. If False it will set the
            # near_idx to 0 otherwise it will not affect the result (it multiplies it by 1)
                near_idx = prox * tol/(((rows_idx - r)**2 + (cols_idx - c)**2)**0.5 + 1)
            # Copy and paste the same near_idx matrix for each map, since the maps are aligned.
            # 'near_idx3D' will be a 3D matrix of shape (n_rows x n_cols x n_maps)
                near_idx3D = np.repeat(near_idx[:,:,np.newaxis], n_maps, axis=2)
            # The variance value is computed as the absolute difference between the 3D input maps
            # stack (n_rows x n_cols x n_maps) and the voxel (1 x 1 x n_maps) referred to the i-th
            # edited pixel in the iteration. The near_idx value will then be subtracted from it.
                variance = np.abs(xmaps_stack - xmaps_stack[r, c, :]) - near_idx3D
            # Fitting is a boolean 2D matrix (n_rows x n_cols) that indicates if for all input maps (-> axis=2)
            # a variance voxel satisfy the condition: variance <= tolerance threshold set by user.
                fitting = np.all(variance <= tol, axis=2)
            # Transform the fitting matrix in a 3D matrix as done for the 'near_idx' matrix.
            # The 'fitting3D' matrix has the following shape: (n_rows x n_cols x n_maps)
                fitting3D = np.repeat(fitting[:, :, np.newaxis], n_maps, axis=2)
            # Sum the variance values along the voxels that actually satisfy the condition imposed by the
            # fitting3D matrix (i.e., the variance value for each map is <= the tolerance threshold)
                tot_variance = np.where(fitting3D, variance, np.nan).sum(axis=2)
            # Edit the output result with the i-th pixel value only where the variance sum is minor
            # than the minimum founded variance.
                outData = np.where(tot_variance < min_variance, value, outData)
            # Refresh the minimum variance
                min_variance = np.fmin(min_variance, tot_variance)

                self.progBar.setValue(n)

        else:
            outData = self.editedData.copy()

        self.progBar.reset()
        return outData

    def calc_boolMap(self, current):
        isCurrent = self.tempResult == current
        isEdited = self.tempResult != self.originalData
        boolMap = isCurrent & isEdited
        return boolMap

    def toggleTrainMode(self, state):
        if state and len(self.CboxMaps.Cbox_list) == 0:
             QW.QMessageBox.warning(self, 'X-Min Learn',
   "Train mode cannot be enabled if no x-ray maps are loaded. Please load the maps in the 'X-Ray Maps' tab.")
             self.trainGA.setChecked(False)

    def update_preview(self):
        if self.trainGA.isChecked():
            self.tempResult = self.calc_pixAffinity()
        else:
            self.tempResult = self.editedData.copy()

        curr_edit = self.currEdit_combox.currentText()
        self.current_preview(curr_edit)

    def current_preview(self, current):
        boolMap = self.calc_boolMap(current)
        title = f'Preview\nN. of {current} pixels = {np.count_nonzero(boolMap)}'
        self.editPreview.update_canvas(title, boolMap)

    def closeEvent(self, event):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         "Do you really want to close the editor window?",
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()



class ModelLearner(DraggableTool):
    '''
    One of the main tools of X-Min Learn, that allows the supported developing 
    of machine learning eager supervised models.
    '''

    def __init__(self):
        '''
        Constructor.

        '''
        super(ModelLearner, self).__init__()

    # Set widget attributes
        self.setWindowTitle('Model Learner')
        self.setWindowIcon(QIcon(r'Icons/merge.png'))

# Main attributes -------------------------------------------------------------

    # Ground-truth dataset
        self.dataset = None

    # Available over-sampling and under-sampling algorithms
        self.os_list = ('None', 'SMOTE', 'BorderlineSMOTE', 'ADASYN')
        self.us_list = ('None', 'RandUS', 'NearMiss', 'TomekLinks', 'ENN-all', 
                        'ENN-mode', 'NCR')
        
    # Balancing strategy list
        self.balancing_strategies = ('Current', 'Min', 'Max', 'Mean', 'Median',
                                     'Custom value', 'Custom multi-value')

    # Train set balancing operation tracker
        self.balancing_info = []

    # Balancing external thread
        self.balancing_thread = threads.BalancingThread()
        self.balancing_thread.setObjectName('Balancing')

    # Learning external thread
        self.learning_thread = threads.LearningThread()
        self.learning_thread.setObjectName('Learning')

    # Machine learning model, network architecture and optimizer
        self.model = None 
        self.network = None 
        self.optimizer = None 

# End of main attributes ------------------------------------------------------

    # Initialize the GUI
        self._init_ui()

    # Connect signals to slots
        self._connect_slots()


    def _init_ui(self): 
        '''
        GUI constructor.

        '''
#  -------------------------------------------------------------------------  #
#                       RANDOM SEED GENERATOR WIDGET
#  -------------------------------------------------------------------------  #
    # Random seed generator group (Group Area)
        self.seed_generator = cObj.RandomSeedGenerator()
        seed_group = cObj.GroupArea(self.seed_generator, 
                                    'Random seed generator')

#  -------------------------------------------------------------------------  #
#                       GROUND TRUTH DATASET WIDGETS 
#  -------------------------------------------------------------------------  #
    # Load dataset button (Styled Button)
        self.load_dataset_btn = cObj.StyledButton(QIcon(r'Icons/import.png'),
                                                  'Load dataset')
    # CSV decimal character selector
        self.csv_dec_selector = cObj.DecimalPointSelector()

    # Loaded dataset path (Path Label)
        self.dataset_path_lbl = cObj.PathLabel(full_display=False,
                                               placeholder='No dataset loaded')

    # Loaded dataset preview area (Document Browser)
        self.dataset_preview = cObj.DocumentBrowser(read_only=True)
        self.dataset_preview.setMinimumHeight(350)

    # Ground truth dataset group (Group Area)
        dataset_grid = QW.QGridLayout()
        dataset_grid.setSpacing(20)
        dataset_grid.addWidget(self.load_dataset_btn, 0, 0, 1, 2)
        dataset_grid.addWidget(QW.QLabel('CSV decimal point'), 0, 2)
        dataset_grid.addWidget(self.csv_dec_selector, 0, 3)
        dataset_grid.addWidget(self.dataset_path_lbl, 1, 0, 1, -1)
        dataset_grid.addWidget(self.dataset_preview, 2, 0, 1, -1)
        dataset_group = cObj.CollapsibleArea(dataset_grid, collapsed=False,
                                             title='Ground truth dataset')
        
#  -------------------------------------------------------------------------  #
#                            PARENT MODEL WIDGETS 
#  -------------------------------------------------------------------------  #

    # Load parent model button (Styled Button)
        self.load_pmodel_btn = cObj.StyledButton(QIcon(r'Icons/import.png'),
                                                 'Load model')
        self.load_pmodel_btn.setEnabled(False)

    # Remove loaded parent model button (Styled Button)
        self.unload_pmodel_btn = cObj.StyledButton(QIcon(r'Icons/clear.png'))

    # Parent model path (Path Label)
        self.pmodel_path = cObj.PathLabel(full_display=False, 
                                          placeholder='No model loaded')

    # Parent model group (Group Area)
        pmodel_grid = QW.QGridLayout()
        pmodel_grid.setSpacing(20)
        pmodel_grid.addWidget(self.load_pmodel_btn, 0, 0, 1, -1)
        pmodel_grid.addWidget(self.unload_pmodel_btn, 1, 0)
        pmodel_grid.addWidget(self.pmodel_path, 1, 1)
        pmodel_grid.setColumnStretch(1, 2)
        pmodel_group = cObj.CollapsibleArea(pmodel_grid, 
                                            title='Update existent model')

#  -------------------------------------------------------------------------  #
#                        HYPERPARAMETERS INPUT WIDGETS 
#  -------------------------------------------------------------------------  #

    # Learning rate input (Styled DoubleSpinBox)
        self.lr_spbox = cObj.StyledDoubleSpinBox(min_value=0.00001, 
                                                 max_value=10., step=0.01,
                                                 decimals=5)
        self.lr_spbox.setValue(0.01)

    # Weight decay input (Styled DoubleSpinBox)
        self.wd_spbox = cObj.StyledDoubleSpinBox(max_value=10., step=0.01,
                                                 decimals=5)
        self.wd_spbox.setValue(0.)

    # Momentum input (Styled DoubleSpinBox)
        self.mtm_spbox = cObj.StyledDoubleSpinBox(max_value=10., step=0.01,
                                                  decimals=5)
        self.mtm_spbox.setValue(0.)

    # Epochs input (Styled SpinBox)
        self.epochs_spbox = cObj.StyledSpinBox(max_value=10**8)
        self.epochs_spbox.setValue(100)

    # Batch size (editable Styled Combobox)
        combox_tooltip = 'A batch size of 0 means a single batch'
        default_batch_sizes = map(lambda n: str(2 ** n), range(5, 13))
        self.batch_combox = cObj.StyledComboBox(tooltip=combox_tooltip)
        self.batch_combox.addItem('0')
        self.batch_combox.addItems(default_batch_sizes)
        self.batch_combox.setCurrentIndex(0)
        self.batch_combox.setEditable(True)
        self.batch_combox.setValidator(QG.QIntValidator(0, 2**24))
        self.batch_combox.setInsertPolicy(QW.QComboBox.InsertPolicy.NoInsert)

    # Hyperparameters group (Group Area)
        hparam_form = QW.QFormLayout()
        hparam_form.setSpacing(10)
        hparam_form.addRow('Learning Rate', self.lr_spbox)
        hparam_form.addRow('Weight Decay', self.wd_spbox)
        hparam_form.addRow('Momentum', self.mtm_spbox)
        hparam_form.addRow('Epochs', self.epochs_spbox)
        hparam_form.addRow('Batch size', self.batch_combox)
        hparam_group = cObj.CollapsibleArea(hparam_form, 
                                            title='Hyperparameters')
        
#  -------------------------------------------------------------------------  #
#                      LEARNING PREFERENCES INPUT WIDGETS 
#  -------------------------------------------------------------------------  #

    # Feature mapping (Checkbox)
        self.feat_mapping_cbox = QW.QCheckBox('Polynomial feature mapping')

    # Polynomial mapping degree (Styled SpinBox)
        self.poly_deg_spbox = cObj.StyledSpinBox(min_value=2, max_value=5)
        self.poly_deg_spbox.setValue(2)
        self.poly_deg_spbox.setEnabled(False)
        self.poly_deg_spbox.setToolTip('Polynomial degree')

    # Algorithm (Styled ComboBox)
        algm_list = ['Softmax Regression']
        self.algm_combox = cObj.StyledComboBox()
        self.algm_combox.addItems(algm_list)

    # Optimization function (Styled ComboBox)
        optim_list = ['SGD']
        self.optim_combox = cObj.StyledComboBox()
        self.optim_combox.addItems(optim_list)

    # Learning graphic update rate (Line Edit)
        self.plots_update_rate = QW.QLineEdit()
        self.plots_update_rate.setValidator(QG.QIntValidator(1, 10**8))
        self.plots_update_rate.setText('10')
        self.plots_update_rate.setStyleSheet(pref.SS_menu)

    # Number of workers (Styled Spinbox)
        max_workers = ML_tools.num_cores() // 2 # safety
        self.workers_spbox = cObj.StyledSpinBox(max_value=max_workers)
        self.workers_spbox.setValue(0)
        self.workers_spbox.setToolTip(
            '0 workers means no parallel computation. Used if batch size > 0.')

    # Use GPU acceleration (Checkbox)
        self.cuda_cbox = QW.QCheckBox('Use GPU acceleration')
        self.cuda_cbox.setChecked(ML_tools.cuda_available())
        self.cuda_cbox.setEnabled(ML_tools.cuda_available())

    # Learning preferences group (Group Area)
        pref_form = QW.QFormLayout()
        pref_form.setSpacing(10)
        pref_form.addRow(self.feat_mapping_cbox, self.poly_deg_spbox)
        pref_form.addRow('Algorithm', self.algm_combox)
        pref_form.addRow('Optimization function', self.optim_combox)
        pref_form.addRow('Graphics refresh rate', self.plots_update_rate)
        pref_form.addRow('Number of workers', self.workers_spbox)
        pref_form.addRow(self.cuda_cbox)
        pref_group = cObj.CollapsibleArea(pref_form,
                                          title='Learning preferences')

#  -------------------------------------------------------------------------  #
#                   START, STOP, TEST, SAVE, PROGBAR WIDGETS 
#  -------------------------------------------------------------------------  #

    # Start learning button (Styled Button)
        self.start_learn_btn = cObj.StyledButton(text='LEARN', 
                                                 bg_color=pref.BTN_GREEN)
        self.start_learn_btn.setToolTip('Start learning session')
        self.start_learn_btn.setEnabled(False)

    # Stop learning button (Styled Button)
        self.stop_learn_btn = cObj.StyledButton(text='STOP', 
                                                 bg_color=pref.BTN_RED)
        self.stop_learn_btn.setToolTip('Stop learning session')
        self.stop_learn_btn.setEnabled(False)

    # Test model button (Styled Button)
        self.test_model_btn = cObj.StyledButton(QIcon(r'Icons/test.png'), 
                                                'TEST MODEL')
        self.test_model_btn.setEnabled(False)

    # Save model button (Styled Button)
        self.save_model_btn = cObj.StyledButton(QIcon(r'Icons/save.png'), 
                                                'SAVE MODEL')
        self.save_model_btn.setEnabled(False)

    # Progress bar
        self.learning_pbar = QW.QProgressBar()

#  -------------------------------------------------------------------------  #
#                       TRAIN, VALIDATION, TEST WIDGETS 
#  -------------------------------------------------------------------------  #

    # Train set ratio selector (Styled SpinBox)
        self.train_ratio_spbox = cObj.StyledSpinBox(1, 98)
        self.train_ratio_spbox.setSuffix(' %')
        self.train_ratio_spbox.setValue(50)

    # Validation set ratio selector (Styled SpinBox)
        self.valid_ratio_spbox = cObj.StyledSpinBox(1, 98)
        self.valid_ratio_spbox.setSuffix(' %')
        self.valid_ratio_spbox.setValue(25)

    # Test set ratio selector (Styled SpinBox)
        self.test_ratio_spbox = cObj.StyledSpinBox(1, 98)
        self.test_ratio_spbox.setSuffix(' %')
        self.test_ratio_spbox.setValue(25)

    # Split ground truth dataset button (Styled Button)
        self.split_dataset_btn = cObj.StyledButton(text='SPLIT', 
                                                   bg_color=pref.BTN_GREEN)
        self.split_dataset_btn.setToolTip('Split ground truth dataset into '\
                                          'train, validation and test sets')
        self.split_dataset_btn.setEnabled(False)

    # Train, Validation and Test sets PieChart (PieCanvas)
        self.subsets_pie = plots.PieCanvas(wheel_pan=False, wheel_zoom=False)
        self.subsets_pie.fig.patch.set(facecolor=pref.CASPER_LIGHT, 
                                       edgecolor=pref.BLACK_PEARL, lw=2)
        self.subsets_pie.setMinimumSize(200, 200)

    # Train, Validation & Test sets barchart (BarCanvas)
        self.subsets_barplot = plots.BarCanvas(wheel_pan=False, 
                                               wheel_zoom=False)
        self.subsets_barplot.setMinimumSize(200, 300)

    # Train set class visualizer (StyledListWidget)
        self.train_class_list = cObj.StyledListWidget(ext_selection=False)
        self.train_class_list.setMinimumWidth(100)

    # Train set per-class current and total counter labels (FramedLabels)
        self.train_curr_count_lbl = cObj.FramedLabel('')
        self.train_tot_count_lbl = cObj.FramedLabel('')

    # Validation set class visualizer (StyledListWidget)
        self.valid_class_list = cObj.StyledListWidget(ext_selection=False)
        self.valid_class_list.setMinimumWidth(100)

    # Validation set per-class current and total counter labels (FramedLabels)
        self.valid_curr_count_lbl = cObj.FramedLabel('')
        self.valid_tot_count_lbl = cObj.FramedLabel('')

    # Test set class visualizer (StyledListWidget)
        self.test_class_list = cObj.StyledListWidget(ext_selection=False)
        self.test_class_list.setMinimumWidth(100)

    # Test set per-class current and total counter labels (FramedLabels)
        self.test_curr_count_lbl = cObj.FramedLabel('')
        self.test_tot_count_lbl = cObj.FramedLabel('')

    # Split Train, validation, test subset widgets group
        counters_grid = QW.QGridLayout()
        counters_grid.setContentsMargins(20, 5, 0, 0)
        counters_grid.setHorizontalSpacing(20)
        counters_grid.setVerticalSpacing(5)
        counters_grid.addWidget(QW.QLabel('Train'), 0, 0, Qt.AlignHCenter)
        counters_grid.addWidget(QW.QLabel('Validation'), 0, 1, Qt.AlignHCenter)
        counters_grid.addWidget(QW.QLabel('Test'), 0, 2, Qt.AlignHCenter)
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
        split_group = cObj.GroupArea(split_grid, 'Split dataset')

#  -------------------------------------------------------------------------  #
#                       BALANCING OPERATIONS WIDGETS 
#  -------------------------------------------------------------------------  #

    # Balancing help button (StyledButton)
        self.balancing_help_btn = cObj.StyledButton(QIcon(r'Icons/info.png'))
        self.balancing_help_btn.setMaximumSize(30, 30)
        self.balancing_help_btn.setFlat(True)
        self.balancing_help_btn.setToolTip('More info about dataset balancing')

    # Oversampling strategy warning icon (QLabel)
        os_warn = QG.QPixmap(r'Icons/warnIcon.png')
        self.os_warn_icon = QW.QLabel()
        self.os_warn_icon.setPixmap(os_warn.scaled(16, 16, Qt.KeepAspectRatio))
        self.os_warn_icon.setToolTip(
            'The selected Over-Sampling algorithm generates an amount of '\
            'samples that may differ from the requested one')
        self.os_warn_icon.hide()

    # Oversampling algorithm selector (Styled ComboBox)
        self.os_combox = cObj.StyledComboBox()
        self.os_combox.addItems(self.os_list)

    # Oversampling K-neighbours selector (Styled SpinBox)
        self.k_neigh_spbox = cObj.StyledSpinBox()
        self.k_neigh_spbox.setValue(5)
        self.k_neigh_spbox.setToolTip(
            'Nearest neighbours to construct synthetic samples')

    # Oversampling M-neighbours selector (Styled SpinBox)
        self.m_neigh_spbox = cObj.StyledSpinBox()
        self.m_neigh_spbox.setValue(10)
        self.m_neigh_spbox.setToolTip(
            'Nearest neighbours to determine if a minority sample is in danger')
        
    # Undersampling strategy warning icon (QLabel)
        us_warn = QG.QPixmap(r'Icons/warnIcon.png')
        self.us_warn_icon = QW.QLabel()
        self.us_warn_icon.setPixmap(us_warn.scaled(16, 16, Qt.KeepAspectRatio))
        self.us_warn_icon.setToolTip(
            'The selected Under-Sampling algorithm removes an amount of '\
            'samples that may differ from the requested one')
        self.us_warn_icon.hide()

    # Undersampling algorithm selector (Styled ComboBox)
        self.us_combox = cObj.StyledComboBox()
        self.us_combox.addItems(self.us_list)

    # Undersampling N-neighbours selector (Styled SpinBox)
        self.n_neigh_spbox = cObj.StyledSpinBox()
        self.n_neigh_spbox.setValue(3)
        self.n_neigh_spbox.setToolTip('Size of the neighbourhood to consider')

    # Strategy selector (Styled ComboBox)
        self.strategy_combox = cObj.StyledComboBox()
        self.strategy_combox.addItems(self.balancing_strategies)

    # Strategy value (LineEdit)
        self.strategy_value = QW.QLineEdit()
        regex = QC.QRegularExpression(r"^(?:[1-9]\d{0,8}|1000000000)$") # 1 - 10^9
        self.strategy_value.setValidator(QG.QRegularExpressionValidator(regex))
        self.strategy_value.setStyleSheet(pref.SS_menu)

    # Strategy percentage (Line edit)
        self.strategy_percent = QW.QLineEdit()
        regex = QC.QRegularExpression(r"^(?:[1-9]|\d{2,3}|1000)$") # 1 - 1000
        self.strategy_percent.setValidator(QG.QRegularExpressionValidator(regex))
        self.strategy_percent.setText('100')
        self.strategy_percent.setStyleSheet(pref.SS_menu)

    # Use parallel computation (Checkbox)
        self.balancing_multicore_cbox = QW.QCheckBox('Parallel computation')
        self.balancing_multicore_cbox.setToolTip(
            'Distribute computation across multiple processes')
        self.balancing_multicore_cbox.setChecked(False)

    # Balancing preview table (Styled Table)
        hl = ('Class', 'Original', 'Current', 'After balancing')
        self.balancing_table = cObj.StyledTable(0, 4)
        # ResizeModes: 0=Interactive, 1=Stretch, 2=Fixed, 3=ResizeToContent
        self.balancing_table.horizontalHeader().setSectionResizeMode(3)
        self.balancing_table.horizontalHeader().setStretchLastSection(True)
        self.balancing_table.verticalHeader().setSectionResizeMode(3)
        self.balancing_table.setHorizontalHeaderLabels(hl)

    # Start balancing button (Styled Button)
        self.start_balancing_btn = cObj.StyledButton(text='Start', 
                                                     bg_color=pref.BTN_GREEN)
        self.start_balancing_btn.setToolTip('Start balancing session')

    # Stop balancing button (Styled Button)
        self.stop_balancing_btn = cObj.StyledButton(text='Stop',
                                                    bg_color=pref.BTN_RED)
        self.stop_balancing_btn.setToolTip('Stop balancing session')

    # Cancel balancing button (Styled Button)
        self.canc_balancing_btn = cObj.StyledButton(QIcon(r'Icons/clear.png'),
                                                    'Cancel')
        self.canc_balancing_btn.setToolTip('Cancel balancing results')

    # Balancing operations progress bar (Descriptive Progress Bar)
        self.balancing_pbar = cObj.DescriptiveProgressBar()
        self.balancing_pbar.setMaximum(5)

    # Balancing group 
        balancing_grid = QW.QGridLayout()
        balancing_grid.setSpacing(10)
        balancing_grid.addWidget(self.balancing_help_btn, 0, 0, Qt.AlignLeft)
        balancing_grid.addWidget(QW.QLabel('Over-Sampling'), 1, 0, 1, 2)
        balancing_grid.addWidget(self.os_warn_icon, 1, 2, Qt.AlignRight)
        balancing_grid.addWidget(cObj.LineSeparator(lw=2), 2, 0, 1, 3)
        balancing_grid.addWidget(self.os_combox, 3, 0)
        balancing_grid.addWidget(QW.QLabel('K neigh.'), 3, 1, Qt.AlignRight)
        balancing_grid.addWidget(self.k_neigh_spbox, 3, 2)
        balancing_grid.addWidget(QW.QLabel('M neigh.'), 4, 1, Qt.AlignRight)
        balancing_grid.addWidget(self.m_neigh_spbox, 4, 2)
        balancing_grid.addWidget(QW.QLabel('Under-Sampling'), 6, 0, 1, 2)
        balancing_grid.addWidget(self.us_warn_icon, 6, 2, Qt.AlignRight)
        balancing_grid.addWidget(cObj.LineSeparator(lw=2), 7, 0, 1, 3)
        balancing_grid.addWidget(self.us_combox, 8, 0)
        balancing_grid.addWidget(QW.QLabel('N neigh.'), 8, 1, Qt.AlignRight)
        balancing_grid.addWidget(self.n_neigh_spbox, 8, 2)
        balancing_grid.addWidget(QW.QLabel('Strategy'), 10, 0, 1, 3)
        balancing_grid.addWidget(cObj.LineSeparator(lw=2), 11, 0, 1, 3)
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

        self.balancing_group = cObj.GroupArea(balancing_grid, checkable=True,
                                              title='Balance train set')
        self.balancing_group.setChecked(False)
        self.balancing_group.setEnabled(False)

#  -------------------------------------------------------------------------  #
#                         LEARNING PROGRESSION PLOTS 
#  -------------------------------------------------------------------------  #

    # Loss curves canvas (CurveCanvas)
        self.loss_plot = plots.CurveCanvas('Loss plot', 'Epochs', 'Loss', 
                                           layout='tight', wheel_pan=False,
                                           wheel_zoom=False)
        self.loss_plot.setMinimumSize(500, 400)

    # Loss curves Navigation Toolbar (NavTbar)
        self.loss_navtbar = plots.NavTbar(self.loss_plot, self)
        self.loss_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.loss_navtbar.fixHomeAction()

    # Loss info labels (FramedLabels)
        self.train_loss_lbl = cObj.FramedLabel('')
        self.valid_loss_lbl = cObj.FramedLabel('')

    # Accuracy curves canvas (CurveCanvas)
        self.accuracy_plot = plots.CurveCanvas('Accuracy plot', 'Epochs',
                                               'Accuracy', layout='tight',
                                               wheel_pan=False,
                                               wheel_zoom=False)
        self.accuracy_plot.setMinimumSize(500, 400)

    # Accuracy curves Navigation Toolbar (NavTbar)
        self.accuracy_navtbar = plots.NavTbar(self.accuracy_plot, self)
        self.accuracy_navtbar.removeToolByIndex([3, 4, 8, 9])
        self.accuracy_navtbar.fixHomeAction()

    # Accuracy info labels (FramedLabels)
        self.train_accuracy_lbl = cObj.FramedLabel('')
        self.valid_accuracy_lbl = cObj.FramedLabel('')

    # Learning progression plots group
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

        learn_plot_tabwid = cObj.StyledTabWidget()
        learn_plot_tabwid.tabBar().setDocumentMode(True)
        learn_plot_tabwid.tabBar().setExpanding(True)
        learn_plot_tabwid.addTab(loss_grid, QIcon(r'Icons/loss.png'),
                                 'Loss plot')
        learn_plot_tabwid.addTab(accuracy_grid, QIcon(r'Icons/accuracy.png'),
                                 'Accuracy plot')
        learn_group = cObj.GroupArea(learn_plot_tabwid, 'Learning progression')

#  -------------------------------------------------------------------------  #
#                        CONFUSION MATRICES WIDGETS 
#  -------------------------------------------------------------------------  #

    # Train Confusion Matrix canvas (ConfMatCanvas)
        self.train_confmat = plots.ConfMatCanvas('Train set', layout='tight',
                                                 wheel_pan=False,
                                                 wheel_zoom=False)
        self.train_confmat.setMinimumSize(600, 600)

    # Train Confusion Matrix Navigation Toolbar (NavTbar)
        self.train_cm_navtbar = plots.NavTbar(self.train_confmat, self,
                                              coords=False)
        self.train_cm_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Annotation as percentage action [--> Train CM Navigation ToolBar]
        self.train_cm_perc_action = QW.QAction(QIcon(r'Icons/labelize.png'),
                                               'Annotations as percentage',
                                               self.train_cm_navtbar)
        self.train_cm_perc_action.setCheckable(True)
        self.train_cm_perc_action.setChecked(True)
        self.train_cm_navtbar.insertAction(2, self.train_cm_perc_action)
        self.train_cm_navtbar.insertSeparator(2)


    # Train F1 scores labels (FramedLabels)
        self.train_f1_macro_lbl = cObj.FramedLabel('')
        self.train_f1_weight_lbl = cObj.FramedLabel('')

    # Validation Confusion Matrix area (ConfMatCanvas)
        self.valid_confmat = plots.ConfMatCanvas('Validation set',
                                                 layout='tight',
                                                 wheel_pan=False,
                                                 wheel_zoom=False)
        self.valid_confmat.setMinimumSize(600, 600)

    # Validation Confusion Matrix Navigation Toolbar (NavTbar)
        self.valid_cm_navtbar = plots.NavTbar(self.valid_confmat, self,
                                              coords=False)
        self.valid_cm_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Annotation as percentage action [--> Validation CM NavigationToolBar]
        self.valid_cm_perc_action = QW.QAction(QIcon(r'Icons/labelize.png'),
                                               'Annotations as percentage',
                                               self.valid_cm_navtbar)
        self.valid_cm_perc_action.setCheckable(True)
        self.valid_cm_perc_action.setChecked(True)
        self.valid_cm_navtbar.insertAction(2, self.valid_cm_perc_action)
        self.valid_cm_navtbar.insertSeparator(2)

    # Validation F1 scores label (FramedLabels)
        self.valid_f1_macro_lbl = cObj.FramedLabel('')
        self.valid_f1_weight_lbl = cObj.FramedLabel('')

    # Test Confusion Matrix (ConfMatCanvas)
        self.test_confmat = plots.ConfMatCanvas('Test set', layout='tight',
                                                wheel_pan=False, 
                                                wheel_zoom=False)
        self.test_confmat.setMinimumSize(600, 600)

    # Test Confusion Matrix Navigation Toolbar (NavTbar)
        self.test_cm_navtbar = plots.NavTbar(self.test_confmat, self, 
                                             coords=False)
        self.test_cm_navtbar.removeToolByIndex([3, 4, 8, 9])

    # Annotation as percentage action [--> Test CM NavigationToolBar]
        self.test_cm_perc_action = QW.QAction(QIcon(r'Icons/labelize.png'),
                                              'Annotations as percentage',
                                              self.test_cm_navtbar)
        self.test_cm_perc_action.setCheckable(True)
        self.test_cm_perc_action.setChecked(True)
        self.test_cm_navtbar.insertAction(2, self.test_cm_perc_action)
        self.test_cm_navtbar.insertSeparator(2)

    # Test score labels (FramedLabels)
        self.test_accuracy_lbl = cObj.FramedLabel('')
        self.test_f1_macro_lbl = cObj.FramedLabel('')
        self.test_f1_weight_lbl = cObj.FramedLabel('')

    # Confusion matrix group 
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

        confmat_tabwid = cObj.StyledTabWidget()
        confmat_tabwid.tabBar().setDocumentMode(True)
        confmat_tabwid.tabBar().setExpanding(True)
        confmat_tabwid.addTab(tr_cm_grid, QIcon(r'Icons/train_set.png'), 
                         'Train set')
        confmat_tabwid.addTab(vd_cm_grid, QIcon(r'Icons/valid_set.png'), 
                         'Validation set')
        confmat_tabwid.addTab(ts_cm_grid, QIcon(r'Icons/test_set.png'), 
                         'Test set')
        confmat_group = cObj.GroupArea(confmat_tabwid, 'Model evaluation')

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
        left_vbox.addLayout(main_ctrl_grid)
        left_vbox.addStretch(1)
        left_scroll = cObj.GroupScrollArea(left_vbox, frame=False)

    # Adjust learning panels layout (right Group Scroll Area)
        right_vbox = QW.QVBoxLayout()
        right_vbox.setSpacing(30)
        right_vbox.addWidget(split_group)
        right_vbox.addWidget(self.balancing_group)
        right_vbox.addWidget(learn_group)
        right_vbox.addWidget(confmat_group)
        right_scroll = cObj.GroupScrollArea(right_vbox, frame=False)

    # Main Layout
        main_layout = cObj.SplitterLayout()
        main_layout.addWidgets((left_scroll, right_scroll))
        self.setLayout(main_layout)
        

    def _connect_slots(self):
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

    # Load ground truth dataset
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
        link = 'https://imbalanced-learn.org/stable/user_guide.html#user-guide'
        self.balancing_help_btn.clicked.connect(lambda: webbrowser.open(link))

    # Show/hide OS warning icon based on selected OS algorithm
        self.os_combox.currentTextChanged.connect(
            lambda t: self.os_warn_icon.setHidden(t != 'ADASYN'))

    # Enable m-neighbors spinbox only when OS algorithm is BorderlineSMOTE
        self.os_combox.currentTextChanged.connect(
            lambda t: self.m_neigh_spbox.setEnabled(t=='BorderlineSMOTE'))
    
    # Show/hide US warning icon based on selected US algorithm
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
        

    def _reset(self):
        '''
        Restore the Model Learner to its original state. Only the loaded ground
        truth dataset and the random seed are preserved.

        '''
    # Reset dataset proterties
        self.dataset.reset()
    
    # Reset model 
        self.model = None 

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


    def _threadRunning(self):
        '''
        Check if an external thread is currently running.

        Returns
        -------
        tuple -> (bool, str | None) 
            (True, thread name) if a thread is running, otherwise (True, None).

        '''
        for thread in (self.balancing_thread, self.learning_thread):
            if thread.isRunning():
                return (True, thread.objectName())
        else:
            return (False, None)
        

    def onSeedChanged(self, old_seed: int, new_seed: int):
        '''
        Determine whether or not a random seed change should be allowed based
        on the current state of the Model Learner. 

        Parameters
        ----------
        old_seed : int
            Random seed before seed change event.
        new_seed : int
            Random seed after seed change event.

        '''
    # Prevent changing seed when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            self.seed_generator.blockSignals(True)
            self.seed_generator.seed_input.setText(str(old_seed))
            self.seed_generator.blockSignals(False)
            text = f'Cannot change seed while a {thr_name} Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)
    
    # Ask confirmation if a dataset was already processed
        if self.dataset and self.dataset.are_subsets_split():
            text = 'Current progress will be lost if seed is changed. Confirm?'
            choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                     QW.QMessageBox.Yes | QW.QMessageBox.No,
                     QW.QMessageBox.No)
            if choice == QW.QMessageBox.No:
                self.seed_generator.blockSignals(True)
                self.seed_generator.seed_input.setText(str(old_seed))
                self.seed_generator.blockSignals(False)
                return
            else:
                self._reset()


    def loadGroundTruthDataset(self):
        '''
        Load ground truth dataset from file.

        '''
    # Prevent loading dataset when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot load dataset while a {thr_name} Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)
        
        path, _ = QW.QFileDialog.getOpenFileName(self, 'Load dataset',
                                                 pref.get_dirPath('in'),
                                                 'Comma Separated Values (*.csv)')
        if path:
            pref.set_dirPath('in', os.path.dirname(path))
        
        # Ask confirmation if a dataset was already processed
            if self.dataset:
                text = 'Current progress will be lost. Load a new dataset?'
                choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                                                QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                QW.QMessageBox.No)
                if choice == QW.QMessageBox.No:
                    return
                else:
                    self._reset()

        # Load ground truth dataset
            dec = self.csv_dec_selector.currentText()
            dataframe = cObj.CsvChunkReader(dec, pBar=True).read(path)
            self.dataset = ML_tools.GroundTruthDataset(dataframe, path)
            self.dataset_path_lbl.setPath(path)
        # Update dataset preview
            text = f'DATASET PREVIEW\n\n{repr(self.dataset.dataframe)}'
            self.dataset_preview.setText(text)
        # Enable "Split dataset" and "Load previous model" buttons
            self.load_pmodel_btn.setEnabled(True)
            self.split_dataset_btn.setEnabled(True)
    

    def _fixSubsetsRatios(self, new_value: int):
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


    def splitDataset(self):
        '''
        Split dataset into train, validation and test sets.

        '''
    # Prevent splitting dataset when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot split dataset while a {thr_name} Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)

    # Ask for overwriting previous balancing results
        if self.balancing_info:
            text = 'Any existent balancing result will be discarded. Continue?'
            choice = QW.QMessageBox.question(self, 'X-Min Learn', text,
                                             QW.QMessageBox.Yes | QW.QMessageBox.No,
                                             QW.QMessageBox.No)
            if choice == QW.QMessageBox.No: 
                return
            
    # Reset balancing info
        self.balancing_info.clear()

    # Reset dataset properties if split was already performed
        if self.dataset.are_subsets_split():
            self.dataset.reset()

    # Split features [X] from targets [Y] expressed as labels
        pbar = cObj.PopUpProgBar(self, 4, 'Splitting dataset', cancel=False)
        try:
            self.dataset.split_features_targets(xtype='int32')
            pbar.setValue(1)

    # Update encoder. Inherit from parent model encoder if present.    
            parent_enc = {}
            if (pmodel_path := self.pmodel_path.fullpath) != '':
                parent_enc = ML_tools.EagerModel.load(pmodel_path).encoder
            self.dataset.update_encoder(parent_enc)
            pbar.setValue(2)

    # Split dataset into train, validation and test sets
            tr_ratio = self.train_ratio_spbox.value() / 100
            vd_ratio = self.valid_ratio_spbox.value() / 100
            ts_ratio = self.test_ratio_spbox.value() / 100
            seed = self.seed_generator.seed
            self.dataset.split_subsets(tr_ratio, vd_ratio, ts_ratio, seed) 
            pbar.setValue(3)

        except Exception as e:
            pbar.reset()
            return cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                   'Unexpected data type', detailedText=repr(e))

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
        pbar.setValue(4)


    def updateSubsetCounterWidget(self, subset: str):
        '''
        Update list widget of a subset class counter.

        Parameters
        ----------
        subset : str
            Name of the subset. Must be one of ('Train', 'Validation', 'Test').

        '''
        if subset == 'Train':
            tr_classes, tr_counts = zip(*self.dataset.train_counter.items())
            self.train_class_list.clear()
            self.train_class_list.addItems(sorted(tr_classes))
            self.train_curr_count_lbl.setText('')
            self.train_tot_count_lbl.setText(f'Tot = {sum(tr_counts)}')
        
        elif subset == 'Validation':
            vd_classes, vd_counts = zip(*self.dataset.valid_counter.items())
            self.valid_class_list.clear()
            self.valid_class_list.addItems(sorted(vd_classes))
            self.valid_curr_count_lbl.setText('')
            self.valid_tot_count_lbl.setText(f'Tot = {sum(vd_counts)}')

        elif subset == 'Test':
            ts_classes, ts_counts = zip(*self.dataset.test_counter.items())
            self.test_class_list.clear()
            self.test_class_list.addItems(sorted(ts_classes))
            self.test_curr_count_lbl.setText('')
            self.test_tot_count_lbl.setText(f'Tot = {sum(ts_counts)}')

        else: 
            return


    def updateSubsetsPlots(self):
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
        bar_data = [list(cnt.values()) for cnt in self.dataset.counters()]
        tickslbl = list(self.dataset.encoder.keys())
        self.subsets_barplot.update_canvas(bar_data, tickslbl, multibars=True)


    def countClassInstances(self, item: QW.QListWidgetItem):
        '''
        Count instances of selected class. The target subset is automatically
        identified using sender().

        Parameters
        ----------
        item : QW.QListWidgetItem
            Subset list's item containing the class name as text.

        '''
        subset = self.sender()

        if subset == self.train_class_list:
            counter = self.dataset.train_counter
            label = self.train_curr_count_lbl

        elif subset == self.valid_class_list:
            counter = self.dataset.valid_counter
            label = self.valid_curr_count_lbl

        elif subset == self.test_class_list:
            counter = self.dataset.test_counter
            label = self.test_curr_count_lbl

        else: 
            return

        count = counter[item.text()]
        label.setText(f'{item.text()} = {count}')


    def cancelBalancingResults(self): 
        '''
        Cancel all balancing operation and restore original train, validation
        and test subsets.

        '''
    # Exit if dataset is not split (safety only, should be impossible) or not 
    # balanced 
        if not self.dataset.are_subsets_split() or not self.balancing_info:
            return
        
    # Ask for user confirm
        text = 'Any existent balancing result will be discarded. Continue?'
        choice = QW.QMessageBox.question(self, 'X-Min Learn', text,
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.No: 
            return
        
    # Discard balancing results
        self.dataset.discard_balancing()
        self.balancing_info.clear()

    # Set ratios of subsets spin-boxes to original values 
        train_ratio, valid_ratio, test_ratio = self.dataset.orig_subsets_ratios
        self.train_ratio_spbox.setValue(round(train_ratio * 100))
        self.valid_ratio_spbox.setValue(round(valid_ratio * 100))
        self.test_ratio_spbox.setValue(round(test_ratio * 100))

    # Update only the train subset counter, as balancing only affects train set
        self.updateSubsetCounterWidget('Train')

    # Update other GUI elements
        self.updateSubsetsPlots()
        self.resetBalancingGroup()
        self._initBalancingTable()

    
    def stopBalancingSession(self):
        '''
        Interrupt balancing session. This is visually shown by setting the 
        balancing progress bar to an undetermined state.

        '''
        if self.balancing_thread.isRunning():
            self.balancing_pbar.setUndetermined()
            self.balancing_pbar.step('Interrupting session')
            self.balancing_thread.requestInterruption()


    def startBalancingSession(self):
        '''
        Start a new Balancing Session. This function launches the balancing 
        external thread.

        '''
    # Prevent starting a new balancing session when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot start a new Balancing Session while a {thr_name} '\
                    'Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)
        
    # Get cell (widgets) data from balancing table 
        classes = [c.text() for c in self.getBalancingTableCellsByColumn(0)]
        values = [cw.value() for cw in self.getBalancingTableCellsByColumn(3)]
        ratios = [cw.ratio(1) for cw in self.getBalancingTableCellsByColumn(3)]

    # Get over-sampling and under-sampling parameters
        strategy = dict(zip(classes, values))
        ovs = self.os_combox.currentText()
        uds = self.us_combox.currentText()
        seed = self.seed_generator.seed
        kos = self.k_neigh_spbox.value()
        mos = self.m_neigh_spbox.value()
        nus = self.n_neigh_spbox.value()
        njobs = (-1) ** self.balancing_multicore_cbox.isChecked()

    # Ask for user confirm
        i = []
        if ovs == 'None':
            ovs = None
            i.append('Over-Sampling not selected. Increased classes ignored.')
        elif max(ratios) > 5:
            i.append('High after-balancing value(s) detected (>500 %). '\
                     'Aggressive over-sampling can cause model overfitting!')
        if uds == 'None':
            uds = None
            i.append('Under-Sampling not selected. Decreased classes ignored.')
        elif min(ratios) < 0.2:
            i.append('Low after-balancing value(s) detected (<20 %). '\
                     'Aggressive under-sampling can cause information loss!')

        icon = QW.QMessageBox.Warning if i else QW.QMessageBox.NoIcon
        note = '\n\n'.join(i) if i else 'No additional info.'
        text = f'You have {len(i)} warning(s). Click "Show Details" for more '\
                'info. Launch Balancing Session?'
        msg = cObj.RichMsgBox(self, icon, 'X-Min Learn', text,
                              QW.QMessageBox.Yes | QW.QMessageBox.No, 
                              QW.QMessageBox.No, detailedText=note)
        
    # Run the balancing session
        if msg.clickedButton().text() == '&Yes':
            self.canc_balancing_btn.setEnabled(False)
            self.balancing_pbar.reset()
            self.balancing_pbar.setUndetermined()

            f1 = self.dataset.parse_balancing_strategy
            f2 = self.dataset.oversample
            f3 = self.dataset.undersample
            f4 = self.dataset.shuffle
            pipeline = (f1, f2, f3, f4)
            params = (strategy, seed, ovs, uds, kos, mos, nus, njobs)

            self.balancing_thread.set_pipeline(pipeline, params)
            self.balancing_thread.start()


    def _parseBalancingResult(self, result: tuple, success: bool):
        '''
        Parse the result of the balancing thread. If the Balancing Session was
        successful, this function applies the balancing to the train set.

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
            self.dataset.apply_balancing(x_bal, y_bal)

        # Update train, validation and test sets ratios spin boxes
            self.balancing_pbar.step('Updating subsets')
            tr_ratio, vd_ratio, ts_ratio = self.dataset.current_subsets_ratios()
            self.train_ratio_spbox.setValue(round(tr_ratio * 100))
            self.valid_ratio_spbox.setValue(round(vd_ratio * 100))
            self.test_ratio_spbox.setValue(round(ts_ratio * 100))
        
        # Update only the train counter widget, as balancing only affects train
            self.updateSubsetCounterWidget('Train')

        # Add new balancing info to the balancing operation tracker
            self.balancing_pbar.step('Tracking balancing info')
            info['New TVT ratios'] = (round(tr_ratio, 2), 
                                      round(vd_ratio, 2), 
                                      round(ts_ratio, 2))
            self.balancing_info.append(info)

        # Update pie chart and bar chart plots
            self.balancing_pbar.step('Updating plots')
            self.updateSubsetsPlots()

        # Re-initialize the balancing table
            self.balancing_pbar.step('Updating table')
            self._initBalancingTable()
        
        else:
        # Forward error if balancing operations failed
            cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                            'Balancing failed', detailedText=repr(result[0]))
            
    # End balancing session in any case
        self._endBalancingSession(success)


    def _endBalancingSession(self, success=False):
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
            text = 'Balancing session concluded succesfully'
            QW.QMessageBox.information(self, 'X-Min Learn', text)


    def _initBalancingTable(self):
        '''
        Initialize the balancing table by populating it with current train set 
        data. 

        '''
    # Get train set data and set the appropriate number of rows in the table
        names, curr_counts = zip(*self.dataset.train_counter.items())
        n_rows = len(names)
        self.balancing_table.setRowCount(n_rows)

        for row in range(n_rows):
        # 1st col item: class name 
            name = names[row]
            i0 = QW.QTableWidgetItem(name)
        # 2nd col item: original class count before any balancing operation
            orig_count = self.dataset.orig_train_counter[name]
            i1 = QW.QTableWidgetItem(str(orig_count)) 
        # 3rd col item: current class count 
            curr_count = curr_counts[row]
            i2 = QW.QTableWidgetItem(str(curr_count))
        # Set such items as not editable (only selectable and enabled)
            for col, i in enumerate((i0, i1, i2)):
                i.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.balancing_table.setItem(row, col, i)
        # 4th col item: expected class count after next balancing operation.
        # This item is a PercentLineEdit connected with a function that sets 
        # the strategy value to 'Custom multi-value' when the item is edited. 
        # The starting value of this item is initialized as the current class 
        # count (= 3rd column value).
            i3 = cObj.PercentLineEdit(curr_count, min_perc=1, max_perc=1000)
            i3.valueEdited.connect(
                lambda: self.strategy_combox.setCurrentText('Custom multi-value'))
            self.balancing_table.setCellWidget(row, 3, i3)


    def resetBalancingGroup(self):
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


    def getBalancingTableCellsByColumn(self, column: int):
        '''
        Convenient function to get all cells in a given column of the balancing
        table. 

        Parameters
        ----------
        column : int
            Column index.

        '''
        rrange = range(self.balancing_table.rowCount())
    # "After Balancing" column
        if column == 3:
            return [self.balancing_table.cellWidget(r, column) for r in rrange]
    # All other columns
        else:
            return [self.balancing_table.item(r, column) for r in rrange]


    def parseStrategy(self, strategy: str):
        '''
        Apply different visual modification to the widgets of the balancing
        group depending on the strategy that has been selected. This is mainly
        done to provide feedback and/or information about the choosen strategy.

        Parameters
        ----------
        strategy : str
            Selected balancing strategy.

        '''
    # Set the strategy percent to 100 % if strategy is 'Current'
        if strategy == 'Current':
            self.strategy_percent.setText('100')
            self.strategy_value.clear()

    # Set the strategy value to minimum counter value if strategy is 'Min'
        elif strategy == 'Min':
            n = min(self.dataset.train_counter.values())
            self.strategy_value.setText(str(n))
            self.strategy_percent.clear()

    # Set the strategy value to maximum counter value if strategy is 'Max'
        elif strategy == 'Max':
            n = max(self.dataset.train_counter.values())
            self.strategy_value.setText(str(n))
            self.strategy_percent.clear()

    # Set the strategy value to average counter value if strategy is 'Mean'
        elif strategy == 'Mean':
            n = np.mean(list(self.dataset.train_counter.values()))
            self.strategy_value.setText(str(round(n)))
            self.strategy_percent.clear()

    # Set the strategy value to median counter value if strategy is 'Median'
        elif strategy == 'Median':
            n = np.median(list(self.dataset.train_counter.values()))
            self.strategy_value.setText(str(round(n)))
            self.strategy_percent.clear()
        
    # Clear out strategy percent if strategy is 'Custom value'
        elif strategy == 'Custom value':
            self.strategy_percent.clear()
    
    # Clear out strategy value if strategy is 'Custom multi-value'
        elif strategy == 'Custom multi-value':
            self.strategy_value.clear()

        else:
            return
        

    def fillAfterBalancingColumn(self, value: str|None):
        '''
        Fill the last column ('After Balancing') of the balancing table with
        a unique balancing strategy custom value or percentage. 

        Parameters
        ----------
        value : str | None
            The balancing strategy value or percentage. If None this function 
            does nothing.

        '''
        if not value:
            return
    # Determine how to fill column based on which widget is the sender
        if self.sender() == self.strategy_value:
            for row in range(self.balancing_table.rowCount()):
                self.balancing_table.cellWidget(row, 3).setValue(int(value))
        else:
            for row in range(self.balancing_table.rowCount()):
                self.balancing_table.cellWidget(row, 3).setPercent(int(value))


    def loadParentModel(self):
        '''
        Load parent ML model, allowing users to train a new model starting from
        its pre-trained parameters.

        '''
    # Prevent loading parent model when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot load model while a {thr_name} Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)

    # Load parent model
        path, _  = QW.QFileDialog.getOpenFileName(self, 'Load parent model',
                                                  pref.get_dirPath('in'),
                                                  'PyTorch Data File (*.pth)')
        if path:
            pref.set_dirPath('in', os.path.dirname(path))

        # Ask confirmation if a dataset was already processed
            if self.dataset.are_subsets_split():
                text = 'Current progress will be lost if model is loaded. Confirm?'
                choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                                                QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                QW.QMessageBox.No)
                if choice == QW.QMessageBox.No:
                    return
                else:
                    self._reset()
           
            pbar = cObj.PopUpProgBar(self, 3, 'Loading Model')
            
            try:
            # Import parent model
                pmodel = ML_tools.EagerModel.load(path)
                pbar.setValue(1)

            # Check if parent model shares the same features of loaded dataset
                if pmodel.features != self.dataset.features_names():
                    raise ValueError('The model and the loaded dataset do '\
                                     'not share the same features.')
                
            # Check if parent model shares the same targets of loaded dataset
                if pmodel.targets != self.dataset.targets_names():
                    raise ValueError('The model and the loaded dataset do '\
                                     'not share the same target classes.')
                pbar.setValue(2)

            # Update GUI
                lr, wd, mtm, _ = pmodel.hyperparameters
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
                pbar.setValue(3)
                QW.QMessageBox.information(self, 'X-Min Learn', 'Model loaded.')
            
            except Exception as e:
                pbar.reset()
                self.removeParentModel()
                cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                'The selected model produced an error.',
                                detailedText=repr(e))


    def removeParentModel(self):
        '''
        Remove loaded parent model, allowing users to train a new model without
        pre-trained model parameters.

        '''
        self.pmodel_path.clearPath()
        # User can select again polynomial feature mapping and algorithm
        self.feat_mapping_cbox.setEnabled(True)
        self.feat_mapping_cbox.setChecked(False)
        self.algm_combox.setEnabled(True)


    def updateScoreLabel(self, subset: str, metric: str, score: float|None):
        '''
        Update a label that displays a specific learning score.

        Parameters
        ----------
        subset : str
            Name of the subset. Must be one of ('Train', 'Validation', 'Test').
        metric : str
            Name of the score. Must be one of ('loss', 'accuracy', 'F1_macro', 
            'F1_weighted').
        score : float | None
            Value of the score. If None the label is cleared.

        '''
        names = ('Train', 'Validation', 'Test')
        metrics = ('loss', 'accuracy', 'F1_macro', 'F1_weighted')

        labels = (
            (self.train_loss_lbl, self.valid_loss_lbl, None),                             # loss
            (self.train_accuracy_lbl, self.valid_accuracy_lbl, self.test_accuracy_lbl),   # accuracy
            (self.train_f1_macro_lbl, self.valid_f1_macro_lbl, self.test_f1_macro_lbl),   # f1 macro
            (self.train_f1_weight_lbl, self.valid_f1_weight_lbl, self.test_f1_weight_lbl) # f1 weighted
            )
        col = names.index(subset)
        row = metrics.index(metric)
        widget = labels[row][col]

        if widget:
            text = '' if score is None else '{:.9f}'.format(score)
            widget.setText(text)

    
    def stopLearningSession(self):
        '''
        Interrupt learning session. This is visually shown by setting the 
        learning progress bar to an undetermined state.

        '''
        if self.learning_thread.isRunning():
            self.learning_pbar.setRange(0, 0)
            self.learning_thread.requestInterruption()


    def startLearningSession(self):
        '''
        Start a new Learning Session. This function launches the learning 
        external thread.

        '''
    # Prevent starting a new learning session when a thread is active
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'Cannot start a new Learning Session while a {thr_name} '\
                    'Session is active'
            return QW.QMessageBox.critical(self, 'X-Min Learn', text)
        
    # Check that subsets contain at least one instance of each target class
        tr_cls = {k for k, v in self.dataset.train_counter.items() if v}
        vd_cls = {k for k, v in self.dataset.valid_counter.items() if v}
        ts_cls = {k for k, v in self.dataset.test_counter.items() if v}

        if xor := ((tr_cls ^ vd_cls) | (tr_cls ^ ts_cls) | (vd_cls ^ ts_cls)): 
            err = 'Some classes have 0 instances in one or more subsets.'
            return cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                   err, detailedText=', '.join(xor))

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

    # If a parent model is provided, update only some of its variables
        if parent_path:
            self.model = ML_tools.EagerModel.load(parent_path)
            var_dict = self.model.variables
            var_dict['optimizer'] = optim_name
            var_dict['device'] = device
            var_dict['parent_model_path'] = parent_path
            var_dict['dataset_path'] = self.dataset.filepath
            var_dict['tvt_ratios'] = self.dataset.orig_subsets_ratios
            var_dict['balancing_info'] = self.balancing_info
            var_dict['learning_rate'] = lr
            var_dict['weight_decay'] = wd
            var_dict['momentum'] = mtm
            var_dict['batch_size'] = batch_size

    # If a parent model is not provided, initialize a new model from scratch
        else:
            self.model = ML_tools.EagerModel.initialize_empty()
            var_dict = self.model.variables
            var_dict['algorithm'] = algm_name
            var_dict['optimizer'] = optim_name
            var_dict['input_features'] = self.dataset.features_names()
            var_dict['class_encoder'] = self.dataset.encoder
            var_dict['device'] = device
            var_dict['seed'] = self.seed_generator.seed
            var_dict['parent_model_path'] = parent_path
            var_dict['dataset_path'] = self.dataset.filepath
            var_dict['tvt_ratios'] = self.dataset.orig_subsets_ratios
            var_dict['balancing_info'] = self.balancing_info
            var_dict['polynomial_degree'] = poly_deg
            var_dict['learning_rate'] = lr
            var_dict['weight_decay'] = wd
            var_dict['momentum'] = mtm
            var_dict['batch_size'] = batch_size
            var_dict['accuracy_list'] = ([], [])
            var_dict['loss_list'] = ([], [])
       
        self.learning_pbar.setValue(1)

    # Map features from linear to polynomial (get original data if degree=1)
        x_train, x_valid = self.dataset.x_train, self.dataset.x_valid
        degree = self.model.poly_degree
        x_train = ML_tools.map_polinomial_features(x_train, degree)
        x_valid = ML_tools.map_polinomial_features(x_valid, degree)

        self.learning_pbar.setValue(2)

    # Convert train and validation features and targets to torch Tensors
        x_train = ML_tools.array2tensor(x_train, 'float32')
        y_train = ML_tools.array2tensor(self.dataset.y_train, 'uint8')
        x_valid = ML_tools.array2tensor(x_valid, 'float32')
        y_valid = ML_tools.array2tensor(self.dataset.y_valid, 'uint8')

    # Normalize feature data. If standards exist in model variables it means
    # that they are derived from a parent model and therefore they must not be
    # changed. Otherwise, a new model is being constructed and standards must
    # be calculated.
        if standards := var_dict['standards']:
            x_mean, x_stdev = standards
            x_train_norm = ML_tools.norm_data(x_train, x_mean, x_stdev,
                                              return_standards=False)
        else:
            x_train_norm, x_mean, x_stdev = ML_tools.norm_data(x_train)
            var_dict['standards'] = (x_mean, x_stdev)

        x_valid_norm = ML_tools.norm_data(x_valid, x_mean, x_stdev,
                                          return_standards=False)
        
        self.learning_pbar.setValue(3)

    # Prepare network and optimizer. If a parent model is provided, load its 
    # network and optimizer state dicts
        self.network = self.model.get_network_architecture()
        self.network.to(device)
        self.optimizer = self.model.get_optimizer(self.network)
        var_dict['loss'] = self.network._loss

        if parent_network_state_dict := var_dict['model_state_dict']:
            self.network.load_state_dict(parent_network_state_dict)
        if parent_optimizer_state_dict := var_dict['optimizer_state_dict']:
            self.optimizer.load_state_dict(parent_optimizer_state_dict)

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
        if batch_size:
            train_loader = ML_tools.DataLoader(x_train_norm, y_train, 
                                               batch_size, workers) 
            valid_loader = ML_tools.DataLoader(x_valid_norm, y_valid, 
                                               batch_size, workers)
            task = lambda: self.network.batch_learn(train_loader, valid_loader,
                                                    self.optimizer, device)
        else:
            task = lambda: self.network.learn(x_train_norm, y_train, 
                                              x_valid_norm, y_valid, 
                                              self.optimizer, device)

        self.learning_thread.set_task(task)
        self.learning_thread.set_params(e_min, e_max, uprate, *loss_lists, 
                                        *acc_lists)
        self.learning_thread.start()


    def _parseLearningResult(self, result: tuple, success: bool):
        '''
        Parse the result of the learning thread. If the Learning Session was
        successful, this function populates the train and validation confusion
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
        if success: 
            tr_loss_list, vd_loss_list, tr_acc_list, vd_acc_list = result
            self.learning_pbar.setRange(0, 2)
            self.learning_pbar.setTextVisible(False)

            d = self.model.poly_degree
            x_mean, x_stdev = self.model.variables['standards']
            device = self.model.variables['device']
            x_train = ML_tools.map_polinomial_features(self.dataset.x_train, d)
            x_valid = ML_tools.map_polinomial_features(self.dataset.x_valid, d)
            x_train = ML_tools.array2tensor(x_train, 'float32')
            x_valid = ML_tools.array2tensor(x_valid, 'float32')
            x_train = ML_tools.norm_data(x_train, x_mean, x_stdev, False)
            x_valid = ML_tools.norm_data(x_valid, x_mean, x_stdev, False)
            train_pred = self.network.predict(x_train.to(device))[1].cpu()
            valid_pred = self.network.predict(x_valid.to(device))[1].cpu()

        # Update train and validation F1 scores
            #                  macro  |  weighted
            # F1 train                |
            # F1 validation           |
            # F1 test                 |
            f1_scores = np.empty((3, 2))
            for n, avg in enumerate(('macro', 'weighted')):
                tr_args = (self.dataset.y_train, train_pred, avg)
                vd_args = (self.dataset.y_valid, valid_pred, avg)
                f1_tr = f1_scores[0, n] = ML_tools.f1_score(*tr_args)
                f1_vd = f1_scores[1, n] = ML_tools.f1_score(*vd_args)
                self.updateScoreLabel('Train', f'F1_{avg}', f1_tr)
                self.updateScoreLabel('Validation', f'F1_{avg}', f1_vd)

            self.learning_pbar.setValue(1)

        # Complete populating model variables
            var_dict = self.model.variables
            var_dict['epochs'] = len(tr_loss_list)
            var_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            var_dict['model_state_dict'] = self.network.state_dict()
            var_dict['accuracy_list'] = (tr_acc_list, vd_acc_list)
            var_dict['loss_list'] = (tr_loss_list, vd_loss_list)
            var_dict['accuracy'] = [tr_acc_list[-1], vd_acc_list[-1], None]
            var_dict['loss'] = [tr_loss_list[-1], vd_loss_list[-1], None]
            var_dict['f1_scores'] = f1_scores

        # Update Confusion Matrices
            self.updateConfusionMatrix('Train', train_pred)
            self.updateConfusionMatrix('Validation', valid_pred)

            self.learning_pbar.setValue(2)

        # Update GUI
            self.test_model_btn.setEnabled(True)
            self.learning_pbar.reset()
            self.learning_pbar.setTextVisible(True)
            text = 'Learning Session completed.'
            QW.QMessageBox.information(self, 'X-Min Learn', text)

    # Forward error if learning session failed
        else:
            e = result[0]
            text = 'Learning Session failed.'
            cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn', text,
                            detailedText=repr(e))

    # Update Start-Stop-Learn buttons on exit in any case
        self.start_learn_btn.setEnabled(True)
        self.stop_learn_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)



    # def _updateLearningScores(self, scores: tuple):
    #     '''
    #     Update the loss and accuracy list with new values. This function should
    #     be called every time a new learning epoch is completed.

    #     Parameters
    #     ----------
    #     scores : tuple
    #         Learning scores. In order: epoch, train and validation losses,
    #         train and validation accuracies.

    #     '''
    #     e, (tr_loss, vd_loss), (tr_acc, vd_acc) = scores
    # # Store new loss and accuracy values
    #     self.train_loss_list.append(tr_loss)
    #     self.valid_loss_list.append(vd_loss)
    #     self.train_accuracy_list.append(tr_acc)
    #     self.valid_accuracy_list.append(vd_acc)
    # # Update progress bar
    #     self.learning_pbar.setValue(e)


    def _drawLearningScores(self, scores: tuple):
        '''
        Update loss and accuracy curves, as well as score labels.

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


    def updateLearningCurve(self, plot: str, y_train: list, y_valid: list):
        '''
        Update loss or accuracy plot.

        Parameters
        ----------
        plot : str
            Plot to update. Must be 'loss' or 'accuracy'.
        y_train : list
            Loss/accuracy values for train set.
        y_valid : list
            Loss/accuracy values for validation set.

        Raises
        ------
        ValueError
            plot is not 'loss' or 'accuracy'.

        '''
        if plot == 'loss':
            canvas = self.loss_plot
        elif plot == 'accuracy':
            canvas = self.accuracy_plot
        else:
            raise ValueError(f'{plot} is not a valid key argument for plot.')
        
        x_data = range(len(y_train))
        curves = [(x_data, y_train), (x_data, y_valid)]  
        canvas.update_canvas(curves, [f'Train {plot}', f'Validation {plot}'])


    def updateConfusionMatrix(self, subset: str, preds: np.ndarray):
        '''
        Populate confusion matrix of the given subset.

        Parameters
        ----------
        subset : str
            Name of the subset. Must be one of ('Train', 'Validation', 'Test').

        '''
        if subset == 'Train':
            true = self.dataset.y_train
            perc = self.train_cm_perc_action.isChecked()
            canvas = self.train_confmat

        elif subset == 'Validation':
            true = self.dataset.y_valid
            perc = self.valid_cm_perc_action.isChecked()
            canvas = self.valid_confmat

        elif subset == 'Test':
            true = self.dataset.y_test
            perc = self.test_cm_perc_action.isChecked()
            canvas = self.test_confmat

        else: 
            return

        if preds is not None:
            lbls, ids = zip(*self.dataset.encoder.items())
            confmat = ML_tools.confusion_matrix(true, preds, ids)
            canvas.update_canvas(confmat, perc)
            canvas.set_ticks(lbls)


    def toggleConfMatPercent(self, toggled: bool):
        '''
        Toggle on/off confusion matrix values as percentages. The matrix canvas
        is automatically identified using sender().

        Parameters
        ----------
        toggled : bool
            Whether to show values as percentages.

        '''
    # Get the matrix canvas from the NavTbar whose action called this function
        canvas = self.sender().parent().canvas
        if canvas.matplot:
            canvas.remove_annotations()
            canvas.annotate(toggled)
            canvas.draw_idle()


    def testModel(self):
        '''
        Test the trained model on test set.

        '''
    # Fixing potential bug were save model button is active when it should not (? - legacy fix, test)
        if self.model is None:
            self.save_model_btn.setEnabled(False)
            return

    # Ask for user confirmation
        text = 'After testing, models should not be further trained on the '\
               'same train set. Proceed?' 
        choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                                        QW.QMessageBox.Yes | QW.QMessageBox.No,
                                        QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            self.learning_pbar.setRange(0, 4)

        # Predict targets on test set
            d = self.model.poly_degree
            device = self.model.variables['device']
            x_mean, x_stdev = self.model.variables['standards']

            x_test = ML_tools.map_polinomial_features(self.dataset.x_test, d)
            x_test = ML_tools.array2tensor(x_test, 'float32')
            x_test = ML_tools.norm_data(x_test, x_mean, x_stdev, False)
            test_pred = self.network.predict(x_test.to(device))[1].cpu()

            self.learning_pbar.setValue(1)

        # Compute test accuracy
            test_acc = ML_tools.accuracy_score(self.dataset.y_test, test_pred)
            self.model.variables['accuracy'][2] = test_acc
            self.updateScoreLabel('Test', 'accuracy', test_acc)
            
            self.learning_pbar.setValue(2)

        # Compute test F1 scores 
            for n, avg in enumerate(('macro', 'weighted')):
                f1_ts = ML_tools.f1_score(self.dataset.y_test, test_pred, avg)
                self.updateScoreLabel('Test', f'F1_{avg}', f1_ts)
                self.model.variables['f1_scores'][2, n] = f1_ts
            
            self.learning_pbar.setValue(3)

        # Update test Confusion Matrix 
            self.updateConfusionMatrix('Test', test_pred)
            self.learning_pbar.setValue(4)

        # Update GUI
            self.save_model_btn.setEnabled(True)
            
        # Reset progress bar and end testing session with success
            self.learning_pbar.reset()
            text = 'Your model has been tested succesfully.'
            QW.QMessageBox.information(self, 'X-Min Learn', text)


    def saveModel(self):
        '''
        Save model to file. Also saves a model log.

        '''
        path, _  = QW.QFileDialog.getSaveFileName(self, 'Save model',
                                                  pref.get_dirPath('out'),
                                                  'PyTorch Data File (*.pth)')
        if path:
            pref.set_dirPath('out', os.path.dirname(path))
            try:
                log_path = self.model.generate_log_path(path)
                extended_log = pref.get_setting('class/extLog', False, bool)
                self.model.save(path, log_path, extended_log)
                QW.QMessageBox.information(self, 'X-Min Learn', 'Model saved.')
            except Exception as e:
                cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
                                'An error occurred while saving the model.',
                                 detailedText=repr(e))
                

    def closeEvent(self, event: QG.QCloseEvent):
        '''
        Reimplementation of the closeEvent function. Requires exit confirm if
        a learning thread or a balancing thread is currently active.

        Parameters
        ----------
        event : QCloseEvent
            The close event.

        '''
        thr_active, thr_name = self._threadRunning()
        if thr_active:
            text = f'A {thr_name} Session is still active. Close anyway?'
            choice = QW.QMessageBox.warning(self, 'X-Min Learn', text,
                                            QW.QMessageBox.Yes | QW.QMessageBox.No,
                                            QW.QMessageBox.No)
            if choice == QW.QMessageBox.Yes:
                self.stopBalancingSession()
                self.stopLearningSession()
                event.accept()
            else:
                event.ignore()

        else:
            super(ModelLearner, self).closeEvent(event)



# class ModelLearner_OLD(DraggableTool):

#     def __init__(self, parent):
#         self.parent = parent
#         super(ModelLearner_OLD, self).__init__()
#         self.setWindowTitle('Model Learner')
#         self.setWindowIcon(QIcon('Icons/merge.png'))

#     # Overall Random Seed
#         self.seed = None

#     # Ground-truth dataset
#         self.dataset = None

#     # Label encoder dict
#         self.Y_dict = dict()

#     # Original Train, Validation & Test rateos (before balancing operations)
#         self.orig_TVTrateos = (None, None, None)

#     # Original Train set targets (before balancing operations)
#         self.origY_train = None

#     # Train, Validation & Test sets
#         self.X_train, self.X_valid, self.X_test = None, None, None
#         self.Y_train, self.Y_valid, self.Y_test = None, None, None

#     # Class counters for Train, Validation & Test sets List_Widgets
#         self.trClassCounter = dict()
#         self.vdClassCounter = dict()
#         self.tsClassCounter = dict()

#     # Train set balancing operation tracer
#         self.balanceInfo = False

#     # Balancing external thread
#         self.balThread = threads.BalanceThread()
#         self.balThread.setObjectName('Balancing')
#         self.balThread.taskFinished.connect(lambda out: self.finalize_balancing(out))

#     # Learning external thread
#         self.learnThread = threads.LearningThread()
#         self.learnThread.setObjectName('Learning')
#         self.learnThread.epochCompleted.connect(lambda out: self.update_learningScores(out))
#         self.learnThread.updateRequested.connect(self.update_learningScoresView)
#         self.learnThread.taskFinished.connect(lambda out: self.finalize_learning(out))

#     # New model instance and parameters dictionary
#         self.model = None
#         self.model_vars = dict()

#     # Train, Validation and Test Model predictions
#         self.tr_preds, self.vd_preds, self.ts_preds = None, None, None

#     # Train & Validation loss and accuracy evolutions lists (for graphics)
#         self.tr_losses, self.vd_losses = [], []
#         self.tr_acc, self.vd_acc = [], []


#         self.init_ui()
#         self.randomize_seed()
#         self.adjustSize()
#         self.showMaximized()


#     def init_ui(self):

# # ==== GROUND TRUTH DATASET WIDGETS ==== #

#     # Load dataset button
#         self.loadDS_btn = QW.QPushButton(QIcon('Icons/load.png'), 'Import')
#         self.loadDS_btn.clicked.connect(self.load_dataset)

#     # CSV decimal character selector
#         self.CSVdec = cObj.DecimalPointSelector()
#         CSVdec_form = QW.QFormLayout()
#         CSVdec_form.addRow('CSV decimal point', self.CSVdec)

#     # Loaded dataset path
#         self.DS_path = cObj.PathLabel('', 'No dataset loaded')

#     # Loaded dataset preview Area
#         self.DS_previewArea = QW.QTextEdit()
#         self.DS_previewArea.setReadOnly(True)
#         self.DS_previewArea.setMinimumHeight(350)

#     # Adjust Load GT-dataset group layout
#         loadDS_grid = QW.QGridLayout()
#         loadDS_grid.addWidget(self.loadDS_btn, 0, 0)
#         loadDS_grid.addLayout(CSVdec_form, 0, 1)
#         loadDS_grid.addWidget(self.DS_path, 1, 0, 1, 2)
#         loadDS_grid.addWidget(self.DS_previewArea, 2, 0, 1, 2)
#         loadDS_grid.setRowStretch(2, 1)
#         loadDS_group = cObj.GroupArea(loadDS_grid, 'Ground-truth dataset')


# # ==== RANDOM SEED GENERATOR WIDGETS==== #

#     # Random seed line edit
#         self.seedInput = QW.QLineEdit()
#         self.seedInput.setValidator(QG.QIntValidator(0, 10**8))
#         self.seedInput.textChanged.connect(self.change_seed)

#     # Randomize seed button
#         self.randseed_btn = cObj.IconButton('Icons/dice.png')
#         self.randseed_btn.clicked.connect(self.randomize_seed)

#     # Adjust random seed group layout
#         seed_hbox = QW.QHBoxLayout()
#         seed_hbox.addWidget(self.seedInput, 1)
#         seed_hbox.addWidget(self.randseed_btn, alignment = Qt.AlignRight)
#         seed_group = cObj.GroupArea(seed_hbox, 'Random seed generator')


# # ==== PREVIOUS ML MODEL LOADING WIDGETS ==== #

#     # Load previous model button
#         self.loadModel_btn = QW.QPushButton('Load')
#         self.loadModel_btn.clicked.connect(self.load_parentModel)
#         self.loadModel_btn.setEnabled(False)

#     # Remove loaded model button
#         self.removeModel_btn = QW.QPushButton('Remove')
#         self.removeModel_btn.clicked.connect(self.remove_parentModel)

#     # Previous model path
#         self.parentModel_path = cObj.PathLabel('', 'No model loaded')

#     # Adjust previous model group
#         parentModel_form = QW.QFormLayout()
#         parentModel_form.addRow('Update model', self.loadModel_btn)
#         parentModel_form.addRow(self.parentModel_path, self.removeModel_btn)
#         parentModel_group = cObj.GroupArea(parentModel_form, 'Load previous model')


# # ==== HYPER-PARAMETERS INPUT WIDGETS ==== #

#     # Learning rate input
#         self.lr_spbox = QW.QDoubleSpinBox()
#         self.lr_spbox.setDecimals(5)
#         self.lr_spbox.setRange(0, 10)
#         self.lr_spbox.setSingleStep(0.01)
#         self.lr_spbox.setValue(0.01)

#     # Weight decay input
#         self.wd_spbox = QW.QDoubleSpinBox()
#         self.wd_spbox.setDecimals(5)
#         self.wd_spbox.setRange(0, 10)
#         self.wd_spbox.setSingleStep(0.01)
#         self.wd_spbox.setValue(0)

#     # Momentum input
#         self.mtm_spbox = QW.QDoubleSpinBox()
#         self.mtm_spbox.setDecimals(5)
#         self.mtm_spbox.setRange(0, 10)
#         self.mtm_spbox.setSingleStep(0.01)
#         self.mtm_spbox.setValue(0)

#     # Epochs input
#         self.epochs_spbox = QW.QSpinBox()
#         self.epochs_spbox.setRange(0, 10**8)
#         self.epochs_spbox.setValue(100)
#         self.epochs_spbox.valueChanged.connect(
#             lambda x: self.graph_updateRate.setText(str(x//10)))

#     # Adjust hyper-parameters group layout
#         hyperparam_form = QW.QFormLayout()
#         hyperparam_form.addRow('Learning Rate', self.lr_spbox)
#         hyperparam_form.addRow('Weight Decay', self.wd_spbox)
#         hyperparam_form.addRow('Momentum', self.mtm_spbox)
#         hyperparam_form.addRow('Epochs', self.epochs_spbox)
#         hyperparam_group = cObj.GroupArea(hyperparam_form, 'Hyperparameters')


# # ==== LEARNING PREFERENCES INPUT WIDGETS ==== #

#     # Regressor type combo-box
#         self.regrType_combox = QW.QComboBox()
#         self.regrType_combox.addItems(['Linear', 'Polynomial'])
#         self.regrType_combox.currentTextChanged.connect(
#             lambda txt: self.polyDeg_spbox.setEnabled(txt=='Polynomial'))
#         self.regrType_combox.currentTextChanged.connect(
#             lambda txt: self.polyDeg_spbox.setValue(1 + (txt=='Polynomial')))

#     # Polynomial degree spin-box
#         self.polyDeg_spbox = QW.QSpinBox()
#         self.polyDeg_spbox.setMinimum(1)
#         self.polyDeg_spbox.setValue(1)
#         self.polyDeg_spbox.setEnabled(False)
#         self.polyDeg_spbox.setToolTip('Select the degree of the polynomial regressor.')

#     # Algorithm combo-box
#         algm_list = ['Softmax Regression']
#         self.algm_combox = QW.QComboBox()
#         self.algm_combox.addItems(algm_list)

#     # Optimization function combo-box
#         optim_list = ['SGD']
#         self.optim_combox = QW.QComboBox()
#         self.optim_combox.addItems(optim_list)

#     # Use GPU acceleration checkbox
#         self.cuda_cbox = QW.QCheckBox('Use GPU acceleration')
#         self.cuda_cbox.setChecked(True)

#     # Learning graphic update rate selector
#         self.graph_updateRate = QW.QLineEdit()
#         self.graph_updateRate.setValidator(QG.QIntValidator(1, 10**8))
#         self.graph_updateRate.setToolTip('Graphics update rate during learnign iterations')
#         self.graph_updateRate.setText('10')

#     # Adjust learnig preferences group layout
#         preferences_form = QW.QFormLayout()
#         preferences_form.addRow('Regressor type', self.regrType_combox)
#         preferences_form.addRow('Degree', self.polyDeg_spbox)
#         preferences_form.addRow('Algorithm', self.algm_combox)
#         preferences_form.addRow('Optimization function', self.optim_combox)
#         preferences_form.addRow(self.cuda_cbox)
#         preferences_form.addRow('Graphics update rate', self.graph_updateRate)
#         preferences_group = cObj.GroupArea(preferences_form, 'Learning preferences')


# # ==== START - STOP - TEST - SAVE - PROGBAR WIDGETS ==== #

#     # Start learning button
#         self.startLearn_btn = QW.QPushButton('START')
#         self.startLearn_btn.setStyleSheet('''background-color: rgb(50,205,50);
#                                               font-weight: bold''')
#         self.startLearn_btn.setToolTip('Start learning session')
#         self.startLearn_btn.clicked.connect(self.initialize_learning)
#         self.startLearn_btn.setEnabled(False)

#     # Stop learning button
#         self.stopLearn_btn = QW.QPushButton('STOP')
#         self.stopLearn_btn.setStyleSheet('''background-color: red;
#                                             font-weight: bold''')
#         self.stopLearn_btn.setToolTip('Stop current learning session')
#         self.stopLearn_btn.clicked.connect(self.stop_learning)
#         self.stopLearn_btn.setEnabled(False)

#     # Test model button
#         self.testModel_btn = QW.QPushButton(QIcon('Icons/test.png'), 'TEST MODEL')
#         self.testModel_btn.clicked.connect(self.test_model)
#         self.testModel_btn.setEnabled(False)

#     # Save model button
#         self.saveModel_btn = QW.QPushButton(QIcon('Icons/save.png'), 'SAVE MODEL')
#         self.saveModel_btn.clicked.connect(self.save_model)
#         self.saveModel_btn.setEnabled(False)

#     # Progress bar
#         self.progBar = QW.QProgressBar()

#     # Adjust start-stop-test-save-pbar layout
#         mainActions_grid = QW.QGridLayout()
#         mainActions_grid.addWidget(self.startLearn_btn, 0, 0)
#         mainActions_grid.addWidget(self.stopLearn_btn, 0, 1)
#         mainActions_grid.addWidget(self.testModel_btn, 1, 0)
#         mainActions_grid.addWidget(self.saveModel_btn, 1, 1)
#         mainActions_grid.addWidget(self.progBar, 2, 0, 1, 2)


# # ==== TRAIN - VALIDATION - TEST SPLIT WIDGETS ==== #

#     # Train, Validation & Test rateo selectors
#         self.trRateo_spbox = QW.QSpinBox()
#         self.vdRateo_spbox = QW.QSpinBox()
#         self.tsRateo_spbox = QW.QSpinBox()
#         for spbox, val in ((self.trRateo_spbox, 50),
#                            (self.vdRateo_spbox, 25),
#                            (self.tsRateo_spbox, 25)):
#             spbox.setRange(1, 98)
#             spbox.setValue(val)
#             spbox.setSuffix(' %')
#             spbox.valueChanged.connect(self.adjust_splitRateos)

#     # Split GT dataset button
#         self.splitDS_btn = QW.QPushButton('SPLIT')
#         self.splitDS_btn.setStyleSheet('''background-color: rgb(50,205,50);
#                                           font-weight: bold''')
#         self.splitDS_btn.clicked.connect(self.split_dataset)
#         self.splitDS_btn.setEnabled(False)

#     # Train, Validation & Test sets PieChart
#         self.tvt_pie = cObj.PieCanvas(self, perc=None)
#         self.tvt_pie.setMinimumSize(150, 150)

#     # Train, Validation & Test sets barchart
#         self.tvt_bar =  cObj.BarCanvas(self)
#         self.tvt_bar.fig.patch.set(facecolor=CF.RGB2float([(169, 185, 188)]), linewidth=0)
#         self.tvt_bar.setMinimumSize(100, 300)

#     # Train set class visualizer
#         self.trSet_list = cObj.StyledListWidget(ext_selection=False)
#         self.trSet_list.itemClicked.connect(lambda i: self.count_target(i, 'Train'))
#         self.trSet_currCount = QW.QLabel('')
#         self.trSet_totCount  = QW.QLabel('Tot = 0')

#     # Validation set class visualizer
#         self.vdSet_list = cObj.StyledListWidget(ext_selection=False)
#         self.vdSet_list.itemClicked.connect(lambda i: self.count_target(i, 'Validation'))
#         self.vdSet_currCount = QW.QLabel('')
#         self.vdSet_totCount  = QW.QLabel('Tot = 0')

#     # Test set class visualizer
#         self.tsSet_list = cObj.StyledListWidget(ext_selection=False)
#         self.tsSet_list.itemClicked.connect(lambda i: self.count_target(i, 'Test'))
#         self.tsSet_currCount = QW.QLabel('')
#         self.tsSet_totCount  = QW.QLabel('Tot = 0')

#     # Adjust Train-Validation-Test split group layout
#         # (Split visualization group)
#         split_grid = QW.QGridLayout()
#         for row, (lbl, sb) in enumerate((('Train set', self.trRateo_spbox),
#                                          ('Validation set', self.vdRateo_spbox),
#                                          ('Test set', self.tsRateo_spbox))):
#             split_grid.addWidget(QW.QLabel(lbl), row, 0)
#             split_grid.addWidget(sb, row, 1)
#         split_grid.addWidget(self.splitDS_btn, row+1, 0, 1, 2)
#         split_grid.addWidget(self.tvt_pie, 0, 2, 4, 1)
#         split_grid.addWidget(self.tvt_bar, 4, 0, 1, 3)
#         split_grid.setColumnStretch(2, 1)

#         # (Train-Validation-Test lists group)
#         trSet_vbox = QW.QVBoxLayout()
#         trSet_vbox.addWidget(self.trSet_list, 1)
#         trSet_vbox.addWidget(self.trSet_currCount)
#         trSet_vbox.addWidget(self.trSet_totCount)
#         trSet_group = cObj.GroupArea(trSet_vbox, 'Train Set')

#         vdSet_vbox = QW.QVBoxLayout()
#         vdSet_vbox.addWidget(self.vdSet_list, 1)
#         vdSet_vbox.addWidget(self.vdSet_currCount)
#         vdSet_vbox.addWidget(self.vdSet_totCount)
#         vdSet_group = cObj.GroupArea(vdSet_vbox, 'Validation Set')

#         tsSet_vbox = QW.QVBoxLayout()
#         tsSet_vbox.addWidget(self.tsSet_list, 1)
#         tsSet_vbox.addWidget(self.tsSet_currCount)
#         tsSet_vbox.addWidget(self.tsSet_totCount)
#         tsSet_group = cObj.GroupArea(tsSet_vbox, 'Test Set')

#         sets_hbox = QW.QHBoxLayout()
#         sets_hbox.addWidget(trSet_group)
#         sets_hbox.addWidget(vdSet_group)
#         sets_hbox.addWidget(tsSet_group)

#         # (T-V-T group Layout)
#         tvt_hsplit = cObj.SplitterGroup((split_grid, sets_hbox))
#         tvt_group = cObj.GroupArea(tvt_hsplit, 'Split Dataset')


# # ==== BALANCING OPERATIONS WIDGETS ==== #

#     # Balancing help button
#         self.balanceHelp_btn = cObj.IconButton('Icons/info.png')
#         self.balanceHelp_btn.setMaximumSize(30, 30)
#         self.balanceHelp_btn.setFlat(True)
#         self.balanceHelp_btn.setToolTip('Click for more info about dataset balancing algorithms')
#         self.balanceHelp_btn.clicked.connect(lambda: webbrowser.open(
#             'https://imbalanced-learn.org/stable/user_guide.html#user-guide'))

#     # Over_Sampling algorithm selector
#         OS_list = ['None', 'SMOTE', 'BorderlineSMOTE', 'ADASYN']
#         self.OS_combox = QW.QComboBox()
#         self.OS_combox.addItems(OS_list)
#         self.OS_combox.currentTextChanged.connect(self.enable_mOS)

#     # Over_Sampling K-neighbours selector
#         self.kOS_spbox = QW.QSpinBox()
#         self.kOS_spbox.setValue(5)
#         self.kOS_spbox.setToolTip('Number of nearest neighbours to construct synthetic samples')

#     # Over_Sampling M-neighbours selector
#         self.mOS_spbox = QW.QSpinBox()
#         self.mOS_spbox.setValue(10)
#         self.mOS_spbox.setToolTip('Number of nearest neighbours to determine if a minority sample is in danger')

#     # Under_Sampling algorithm selector
#         US_list = ['None', 'RandUS', 'NearMiss', 'TomekLinks',
#                     'ENN-all', 'ENN-mode', 'NCR-all', 'NCR-mode']
#         self.US_combox = QW.QComboBox()
#         self.US_combox.addItems(US_list)
#         self.US_combox.currentTextChanged.connect(self.update_USwarning)
#         self.US_combox.currentTextChanged.connect(self.enable_nUS)

#     # Under_Sampling strategy warning icon
#         self.US_warnIcon = QW.QLabel()
#         w_icon = QG.QPixmap('Icons/warnIcon.png').scaled(30, 30, Qt.KeepAspectRatio)
#         self.US_warnIcon.setPixmap(w_icon)
#         self.US_warnIcon.setToolTip('Warning: the selected Under-Sampling algorithm '\
#                                     'ignores the strategy required by the user.')
#         self.US_warnIcon.hide()

#     # Under_Sampling N-neighbours selector
#         self.nUS_spbox = QW.QSpinBox()
#         self.nUS_spbox.setValue(3)
#         self.nUS_spbox.setToolTip('Size of the neighbourhood to consider')

#     # Strategy selector
#         strat_list = ['Current', 'Min', 'Max', 'Mean', 'Median',
#                       'Custom value', 'Custom multi-value']
#         self.strategy_combox = QW.QComboBox()
#         self.strategy_combox.addItems(strat_list)
#         self.strategy_combox.currentTextChanged.connect(self.update_strategyValue)

#     # Strategy value
#         self.strat_value = QW.QLineEdit()
#         self.strat_value.setValidator(QG.QIntValidator(0, 10**10))
#         self.strat_value.textEdited.connect(
#             lambda: self.strategy_combox.setCurrentText('Custom value'))
#         self.strat_value.textChanged.connect(self.autoFill_balanceTable)

#     # Balancing preview table
#         self.balance_table = QW.QTableWidget(0, 4)
#         self.balance_table.horizontalHeader().setSectionResizeMode(1) # Stretch
#         self.balance_table.verticalHeader().setSectionResizeMode(3) # ResizeToContent
#         self.balance_table.setHorizontalHeaderLabels(['Class name', 'Original size',
#                                                       'Current size', 'After balancing'])

#     # Run balancing button
#         self.runBalance_btn = QW.QPushButton('Start Balancing')
#         self.runBalance_btn.setStyleSheet('''background-color: rgb(50,205,50);
#                                           font-weight: bold''')
#         self.runBalance_btn.clicked.connect(self.initialize_balancing)

#     # Cancel balancing button
#         self.cancBalance_btn = QW.QPushButton('Cancel Balancing')
#         self.cancBalance_btn.setStyleSheet('''background-color: red;
#                                               font-weight: bold''')
#         self.cancBalance_btn.clicked.connect(self.canc_balancing)

#     # Balancing operations ProgressBar
#         self.balance_pBar = QW.QProgressBar()
#         self.balance_pBar.setMaximum(4)

#     # Adjust Balance group layout
#         # (Start-stop balancing hbox)
#         balActions_hbox = QW.QHBoxLayout()
#         balActions_hbox.setSpacing(5)
#         balActions_hbox.addWidget(self.runBalance_btn)
#         balActions_hbox.addWidget(self.cancBalance_btn)

#         # (Balancing options)
#         balOptions_grid = QW.QGridLayout()
#         balOptions_grid.setVerticalSpacing(10)
#         balOptions_grid.addWidget(self.balanceHelp_btn, 0, 0, alignment = Qt.AlignLeft)
#         balOptions_grid.addWidget(self.US_warnIcon, 0, 2, alignment = Qt.AlignRight)
#         balOptions_grid.addWidget(QW.QLabel('Over-Sampling algorithm'), 1, 0)
#         balOptions_grid.addWidget(cObj.LineSeparator(), 1, 1, 1, 2)
#         balOptions_grid.addWidget(self.OS_combox, 2, 0)
#         balOptions_grid.addWidget(QW.QLabel('k-neighbours'), 2, 1, alignment = Qt.AlignRight)
#         balOptions_grid.addWidget(self.kOS_spbox, 2, 2)
#         balOptions_grid.addWidget(QW.QLabel('m-neighbours'), 3, 1, alignment = Qt.AlignRight)
#         balOptions_grid.addWidget(self.mOS_spbox, 3, 2)
#         balOptions_grid.addWidget(QW.QLabel('Under-Sampling algorithm'), 4, 0)
#         balOptions_grid.addWidget(cObj.LineSeparator(), 4, 1, 1, 2)
#         balOptions_grid.addWidget(self.US_combox, 5, 0)
#         balOptions_grid.addWidget(QW.QLabel('n-neighbours'), 5, 1, alignment = Qt.AlignRight)
#         balOptions_grid.addWidget(self.nUS_spbox, 5, 2)
#         balOptions_grid.addWidget(cObj.LineSeparator(), 6, 0, 1, 3)
#         balOptions_grid.addWidget(QW.QLabel('Strategy'), 7, 0)
#         balOptions_grid.addWidget(self.strategy_combox, 7, 1, 1, 2)
#         balOptions_grid.addWidget(QW.QLabel('Unique value'), 8, 0)
#         balOptions_grid.addWidget(self.strat_value, 8, 1, 1, 2)
#         balOptions_grid.addWidget(cObj.LineSeparator(), 9, 0, 1, 3)
#         balOptions_grid.addLayout(balActions_hbox, 10, 0, 1, 3)
#         balOptions_grid.addWidget(self.balance_pBar, 11, 0, 1, 3)

#         # (Balance group main layout)
#         balance_hsplit = cObj.SplitterGroup((balOptions_grid, self.balance_table), (0, 1))
#         self.balance_group = cObj.GroupArea(balance_hsplit, 'Balance train set',
#                                             checkable=True)
#         self.balance_group.setAlignment(Qt.AlignLeft)
#         self.balance_group.setChecked(False)
#         self.balance_group.setEnabled(False)


# # ==== LEARNING SESSION PLOTS ==== #

#     # Loss curves canvas
#         self.lossView = cObj.CurvePlotCanvas(self, title='Loss curves', tight=True,
#                                              xlab='Epochs', ylab='Loss')
#         self.lossView.setMinimumSize(500, 400)

#     # Loss curves Navigation Toolbar
#         self.lossNTbar = cObj.NavTbar(self.lossView, self)
#         self.lossNTbar.removeToolByIndex([3, 4, 8, 9])
#         self.lossNTbar.fixHomeAction()

#     # Loss info labels
#         self.trLoss_lbl = QW.QLabel('None')
#         self.vdLoss_lbl = QW.QLabel('None')

#     # Accuracy curves canvas
#         self.accuracyView = cObj.CurvePlotCanvas(self, title='Accuracy curves', tight=True,
#                                                  xlab='Epochs', ylab='Accuracy')
#         self.accuracyView.setMinimumSize(500, 400)

#     # Accuracy curves Navigation Toolbar
#         self.accuracyNTbar = cObj.NavTbar(self.accuracyView, self)
#         self.accuracyNTbar.removeToolByIndex([3, 4, 8, 9])
#         self.accuracyNTbar.fixHomeAction()

#     # Accuracy info labels
#         self.trAcc_lbl = QW.QLabel('None')
#         self.vdAcc_lbl = QW.QLabel('None')

#     # Train Confusion Matrix canvas
#         self.CMtrainView = cObj.MatrixCanvas(self,
#                                              title='Train set Confusion Matrix',
#                                              xlab='Predicted classes',
#                                              ylab='True Classes')
#         self.CMtrainView.setMinimumSize(600, 600)

#     # Train Confusion Matrix Navigation Toolbar
#         self.CMtrainNTbar = cObj.NavTbar(self.CMtrainView, self)
#         self.CMtrainNTbar.removeToolByIndex([3, 4, 8, 9, -1])

#     # Show annotation as percentage action [--> Train CM NavTBar]
#         self.CMtrainPercAction = QW.QAction(QIcon('Icons/CM_annotate.png'),
#                                             'Toggle annotations as percentage',
#                                             self.CMtrainNTbar)
#         self.CMtrainPercAction.setCheckable(True)
#         self.CMtrainPercAction.setChecked(True)
#         self.CMtrainPercAction.triggered.connect(lambda: self.update_ConfusionMatrix('Train'))
#         self.CMtrainNTbar.insertAction(5, self.CMtrainPercAction)

#     # Train F1 scores labels
#         self.trF1Micro_lbl = QW.QLabel('None')
#         self.trF1Macro_lbl = QW.QLabel('None')
#         self.trF1Weight_lbl = QW.QLabel('None')

#     # Validation Confusion Matrix area
#         self.CMvalidView = cObj.MatrixCanvas(self,
#                                             title='Validation set Confusion Matrix',
#                                             xlab='Predicted classes',
#                                             ylab='True Classes')
#         self.CMvalidView.setMinimumSize(600, 600)

#     # Validation Confusion Matrix Navigation Toolbar
#         self.CMvalidNTbar = cObj.NavTbar(self.CMvalidView, self)
#         self.CMvalidNTbar.removeToolByIndex([3, 4, 8, 9, -1])

#     # Show annotation as percentage action [--> Validation CM NavTBar]
#         self.CMvalidPercAction = QW.QAction(QIcon('Icons/CM_annotate.png'),
#                                            'Toggle annotations as percentage',
#                                             self.CMvalidNTbar)
#         self.CMvalidPercAction.setCheckable(True)
#         self.CMvalidPercAction.setChecked(True)
#         self.CMvalidPercAction.triggered.connect(lambda: self.update_ConfusionMatrix('Validation'))
#         self.CMvalidNTbar.insertAction(5, self.CMvalidPercAction)

#     # Validation F1 scores label
#         self.vdF1Micro_lbl = QW.QLabel('None')
#         self.vdF1Macro_lbl = QW.QLabel('None')
#         self.vdF1Weight_lbl = QW.QLabel('None')

#     # Adjust Learning Session plots group layout
#         # (Loss Form Layout)
#         loss_form = QW.QFormLayout()
#         loss_form.addRow('Train loss ', self.trLoss_lbl)
#         loss_form.addRow('Validation loss ', self.vdLoss_lbl)

#         # (Accuracy Form Layout)
#         acc_form = QW.QFormLayout()
#         acc_form.addRow('Train accuracy ', self.trAcc_lbl)
#         acc_form.addRow('Validation accuracy ', self.vdAcc_lbl)

#         # (Train F1 scores Form Layout)
#         trF1_form = QW.QFormLayout()
#         trF1_form.addRow('Train F1 score (Micro AVG) ', self.trF1Micro_lbl)
#         trF1_form.addRow('Train F1 score (Macro AVG) ', self.trF1Macro_lbl)
#         trF1_form.addRow('Train F1 score (Weighted AVG) ', self.trF1Weight_lbl)

#         # (Validation F1 scores Form Layout)
#         vdF1_form = QW.QFormLayout()
#         vdF1_form.addRow('Validation F1 score (Micro AVG) ', self.vdF1Micro_lbl)
#         vdF1_form.addRow('Validation F1 score (Macro AVG) ', self.vdF1Macro_lbl)
#         vdF1_form.addRow('Validation F1 score (Weighted AVG) ', self.vdF1Weight_lbl)

#         # (Main group layout)
#         learnPlots_grid = QW.QGridLayout()
#         for row, (l, r) in enumerate(((self.lossNTbar   , self.accuracyNTbar),
#                                       (self.lossView    , self.accuracyView ),
#                                       (loss_form        , acc_form          ),
#                                       (self.CMtrainNTbar, self.CMvalidNTbar ),
#                                       (self.CMtrainView , self.CMvalidView  ),
#                                       (trF1_form        , vdF1_form         ))):
#             if row in (2, 5):
#                 learnPlots_grid.addLayout(l, row, 0)
#                 learnPlots_grid.addLayout(r, row, 1)
#             else:
#                 learnPlots_grid.addWidget(l, row, 0)
#                 learnPlots_grid.addWidget(r, row, 1)
#         learnPlots_group = cObj.GroupArea(learnPlots_grid, 'Learning Evaluation')


# # ==== FINAL TESTING PLOTS ==== #

#     # Test Confusion Matrix
#         self.CMtestView = cObj.MatrixCanvas(self,
#                                             title='Test set Confusion Matrix',
#                                             xlab='Predicted classes',
#                                             ylab='True Classes')
#         self.CMtestView.setMinimumSize(600, 600)

#     # Test Confusion Matrix Navigation Toolbar
#         self.CMtestNTbar = cObj.NavTbar(self.CMtestView, self)
#         self.CMtestNTbar.removeToolByIndex([3, 4, 8, 9, -1])

#     # Show annotation as percentage action [--> Test CM NavTBar]
#         self.CMtestPercAction = QW.QAction(QIcon('Icons/CM_annotate.png'),
#                                             'Toggle annotations as percentage',
#                                             self.CMtestNTbar)
#         self.CMtestPercAction.setCheckable(True)
#         self.CMtestPercAction.setChecked(True)
#         self.CMtestPercAction.triggered.connect(lambda: self.update_ConfusionMatrix('Test'))
#         self.CMtestNTbar.insertAction(5, self.CMtestPercAction)

#     # Test score labels
#         self.tsLoss_lbl = QW.QLabel('None')
#         self.tsAcc_lbl = QW.QLabel('None')
#         self.tsF1Micro_lbl = QW.QLabel('None')
#         self.tsF1Macro_lbl = QW.QLabel('None')
#         self.tsF1Weight_lbl = QW.QLabel('None')

#     # Model info preview area
#         self.modelPreview = QW.QTextEdit()
#         self.modelPreview.setReadOnly(True)

#     # Adjust final testing group layout
#         # (Test scores form layout)
#         tsScores_form = QW.QFormLayout()
#         # tsScores_form.addRow('Test Loss', self.tsLoss_lbl)
#         tsScores_form.addRow('Test Accuracy ', self.tsAcc_lbl)
#         tsScores_form.addRow('Test F1 score (Micro AVG) ', self.tsF1Micro_lbl)
#         tsScores_form.addRow('Test F1 score (Macro AVG) ', self.tsF1Macro_lbl)
#         tsScores_form.addRow('Test F1 score (Weighted AVG) ', self.tsF1Weight_lbl)

#         # (Main group layout)
#         test_grid = QW.QGridLayout()
#         test_grid.addWidget(self.CMtestNTbar, 0, 0)
#         test_grid.addWidget(QW.QLabel('Model Variables Preview'), 0, 1, alignment=Qt.AlignHCenter)
#         test_grid.addWidget(self.CMtestView, 1, 0)
#         test_grid.addWidget(self.modelPreview, 1, 1, 3, 1)
#         test_grid.addLayout(tsScores_form, 2, 0)
#         test_group = cObj.GroupArea(test_grid, 'Model testing')


# # ==== ADJUST MAIN LAYOUT ==== #

#     # Adjust left area
#         left_vbox = QW.QVBoxLayout()
#         left_vbox.addWidget(loadDS_group, 1)
#         left_vbox.addWidget(seed_group)
#         left_vbox.addWidget(parentModel_group)
#         left_vbox.addWidget(hyperparam_group)
#         left_vbox.addWidget(preferences_group)
#         left_vbox.addLayout(mainActions_grid)
#         left_scroll = cObj.GroupScrollArea(left_vbox)
#         left_scroll.setStyleSheet('QWidget {""}') # set the default stylesheet

#     # Adjust right area
#         right_vbox = QW.QVBoxLayout()
#         right_vbox.setSpacing(50)
#         right_vbox.addWidget(tvt_group)
#         right_vbox.addWidget(self.balance_group)
#         right_vbox.addWidget(learnPlots_group)
#         right_vbox.addWidget(test_group)
#         right_scroll = cObj.GroupScrollArea(right_vbox)

#     # Adjust final layout
#         main_hsplit = cObj.SplitterGroup((left_scroll, right_scroll), (0, 1)) # use SplitterLayout
#         mainLayout = QW.QGridLayout()
#         mainLayout.addWidget(main_hsplit)
#         self.setLayout(mainLayout)

#     def reset_UI(self):
#     # Reset parent model
#         self.remove_parentModel()
#     # Reset split dataset operations
#         self.trSet_list.clear()
#         self.vdSet_list.clear()
#         self.tsSet_list.clear()
#         self.tvt_pie.clear_canvas()
#         self.tvt_bar.clear_canvas()
#     # Reset balancing features
#         self.reset_balanceGroup()
#     # Reset Learning session features
#         self.lossView.clear_canvas()
#         self.accuracyView.clear_canvas()
#         self.CMtrainView.clear_canvas()
#         self.CMvalidView.clear_canvas()
#     # Reset Testing session features
#         self.CMtestView.clear_canvas()
#         self.modelPreview.clear()
#     # Reset all scores labels
#         for metric in ('loss', 'accuracy', 'F1_micro', 'F1_macro', 'F1_weighted'):
#             for name in ('Train', 'Validation', 'Test'):
#                 self.update_scoreLabel(name, metric, None)
#     # Disable "start", "stop", "test" and "save" model operations
#         self.startLearn_btn.setEnabled(False)
#         self.stopLearn_btn.setEnabled(False)
#         self.testModel_btn.setEnabled(False)
#         self.saveModel_btn.setEnabled(False)

#     def load_dataset(self, autosplit=True):
#         path, _ = QW.QFileDialog.getOpenFileName(self, 'Load Ground-truth dataset',
#                                                   pref.get_dirPath('in'),
#                                                   'Comma Separated Values (*.csv)')
#         if path:
#             pref.set_dirPath('in', os.path.dirname(path))
#             if self._splitDatasetPerformed():
#                 choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                                  'The entire learning procedure '\
#                                                  'must be performed again. Load a new dataset?.',
#                                                  QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                                  QW.QMessageBox.No)
#                 if choice == QW.QMessageBox.No:
#                     return
#                 else:
#                     self.reset_UI()

#         # Get decimal point character
#             dec = self.CSVdec.currentText()
#         # Load GT dataset
#             self.dataset = cObj.CsvChunkReader(dec, pBar=True).read(path)
#         # Update GT dataset path
#             self.DS_path.set_fullpath(path, predict_display=True)
#         # Update dataset preview
#             self.DS_previewArea.setText(f'DATASET PREVIEW\n\n{repr(self.dataset)}')
#         # Enable "Split dataset" and "Load previous model"
#             self.loadModel_btn.setEnabled(True)
#             self.splitDS_btn.setEnabled(True)


#     def adjust_splitRateos(self, value):
#         spboxes = [self.trRateo_spbox, self.vdRateo_spbox, self.tsRateo_spbox]
#         tot = lambda: sum([spb.value() for spb in spboxes])
#         if tot != 100:
#             # Reverse <spboxes> when the sender is the 2nd element, to achieve a GUI friendlier behaviour
#             curr_idx = spboxes.index(self.sender())
#             if curr_idx == 1: spboxes.reverse()
#             for idx, spb in enumerate(spboxes):
#                 if not idx == curr_idx:
#                     # Temporarily block valueChanged signal when changing value (avoid loops).
#                     spb.blockSignals(True)
#                     # Increment/decrement the selected spinbox value
#                     spb.setValue(spb.value() + 100 - tot())
#                     # Unblock valueChanged signal
#                     spb.blockSignals(False)

#     def _splitDatasetPerformed(self):
#         set_sum = sum([l.count() for l in (self.trSet_list, self.vdSet_list, self.tsSet_list)])
#         return set_sum > 0

#     def split_dataset(self):
#         if self._extThreadRunning():
#             return

#     # Check if balancing operations have been performed already
#         if self.balanceInfo:
#             choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                              'Any previous balancing operation on '\
#                                              'train set will be discarded. Continue?',
#                                              QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                              QW.QMessageBox.No)
#             if choice == QW.QMessageBox.No: return

#     # Split features [X] from targets [Y (as labels)]
#         pbar = cObj.PopUpProgBar(self, 5, 'Splitting dataset', cancel=False)
#         try:
#             X, Y_lbl = ML_tools.splitXFeat_YTarget(self.dataset.to_numpy(),
#                                                    xtype='int32', ytype='U8')
#             pbar.setValue(1)
#         except Exception as e:
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                             'Unexpected data type', detailedText = repr(e))
#             pbar.reset()
#             return

#     # Update Y_dict and encode Y labels through it
#         self.update_Ydict(Y_lbl)
#         Y = CF.encode_labels(Y_lbl, self.Y_dict)
#         pbar.setValue(2)

#     # Store original Train, Validation & Test sets rateos (before balancing operations)
#         trRateo = self.trRateo_spbox.value()/100
#         vdRateo = self.vdRateo_spbox.value()/100
#         tsRateo = self.tsRateo_spbox.value()/100
#         self.orig_TVTrateos = (trRateo, vdRateo, tsRateo)
#         pbar.setValue(3)

#     # Split X features and Y targets into Train, Validation & Test sets
#         seed = self.seed
#         X_split, Y_split = ML_tools.splitTrainValidTest(X, Y, trRateo, vdRateo, seed)
#         self.X_train, self.X_valid, self.X_test = X_split
#         self.Y_train, self.Y_valid, self.Y_test = Y_split
#         pbar.setValue(4)

#     # Store original Train targets (before balancing operations)
#         self.origY_train = self.Y_train.copy()
#         pbar.setValue(5)

#     # Update Train, Validation & Test sets class counters and viewers
#         for _set, _Y in (('Train',      self.Y_train),
#                          ('Validation', self.Y_valid),
#                          ('Test',       self.Y_test)):
#             self.update_setClassCounter(_set, _Y)
#             self.update_setClassViewer(_set)

#     # Update TVT pie chart & bar chart
#         self.update_tvtPlots()

#     # Reset balancing features
#         self.reset_balanceGroup()
#         self.init_balanceTable()
#         self.balance_group.setEnabled(True)

#     # Update Main Actions buttons (START-TEST-LEARN)
#         self.startLearn_btn.setEnabled(True)
#         self.testModel_btn.setEnabled(False)
#         self.saveModel_btn.setEnabled(False)


#     def update_tvtPlots(self):
#     # Update Pie chart
#         trRateo = self.trRateo_spbox.value()
#         vdRateo = self.vdRateo_spbox.value()
#         tsRateo = self.tsRateo_spbox.value()
#         lbls = (f'Train ({trRateo} %)', f'Validation ({vdRateo} %)', f'Test ({tsRateo} %)')
#         self.tvt_pie.update_canvas((trRateo, vdRateo, tsRateo), lbls)
#     # Update Bar chart
#         barData = [list(cnt.values()) for cnt in (self.trClassCounter,
#                                                   self.vdClassCounter,
#                                                   self.tsClassCounter)]
#         self.tvt_bar.update_canvas('', barData, list(self.Y_dict.keys()),
#                                    multibars=True)


#     def update_setClassCounter(self, set_name, Yset):
#         Yset = CF.decode_labels(Yset, self.Y_dict)
#         counter = dict()
#         for k in self.Y_dict.keys():
#             counter[k] = np.count_nonzero(Yset==k)

#         if set_name == 'Train':
#             self.trClassCounter = counter
#         elif set_name == 'Validation':
#             self.vdClassCounter  = counter
#         elif set_name == 'Test':
#             self.tsClassCounter = counter
#         else : return

#     def update_setClassViewer(self, set_name):
#         if set_name == 'Train':
#             listWid = self.trSet_list
#             curr_lbl = self.trSet_currCount
#             tot_lbl = self.trSet_totCount
#             counter = self.trClassCounter
#         elif set_name == 'Validation':
#             listWid = self.vdSet_list
#             curr_lbl = self.vdSet_currCount
#             tot_lbl = self.vdSet_totCount
#             counter = self.vdClassCounter
#         elif set_name == 'Test':
#             listWid = self.tsSet_list
#             curr_lbl = self.tsSet_currCount
#             tot_lbl = self.tsSet_totCount
#             counter = self.tsClassCounter
#         else : return

#         listWid.clear()
#         listWid.addItems(sorted(counter.keys()))
#         curr_lbl.setText('')
#         tot_lbl.setText(f'Tot = {sum(counter.values())}')


#     def count_target(self, item, set_name):
#         if set_name == 'Train':
#             counter = self.trClassCounter
#             curr_lbl = self.trSet_currCount
#         elif set_name == 'Validation':
#             counter = self.vdClassCounter
#             curr_lbl = self.vdSet_currCount
#         elif set_name == 'Test':
#             counter = self.tsClassCounter
#             curr_lbl = self.tsSet_currCount
#         else : return

#         count = counter[item.text()]
#         curr_lbl.setText(f'{item.text()} = {count}')

#     def update_Ydict(self, Y_lbl):
#         # Checking for parent model Y_dict
#         self.Y_dict = {}
#         if self.parentModel_path.get_fullpath() != '':
#             self.Y_dict = ML_tools.loadModel(
#                 self.parentModel_path.get_fullpath())['Y_dict']

#         # Update Y_dict {label: ID}
#         for u in np.unique(Y_lbl):
#             if not u in self.Y_dict.keys():
#                 self.Y_dict[u] = len(self.Y_dict)

#     def randomize_seed(self):
#         seed = np.random.randint(0, 10**8)
#         self.seedInput.setText(str(seed))

#     def change_seed(self, text):
#         if self._extThreadRunning():
#             return
#         if self._splitDatasetPerformed():
#             choice = QW.QMessageBox.warning(self, 'X-Min Learn',
#                      'Warning: the entire learnig procedure must be performed '\
#                      'again if the seed is changed now. Confirm?',
#                      QW.QMessageBox.Yes | QW.QMessageBox.No,
#                      QW.QMessageBox.No)
#             if choice == QW.QMessageBox.No:
#                 self.seedInput.blockSignals(True)
#                 self.seedInput.setText(str(self.seed))
#                 self.seedInput.blockSignals(False)
#                 return
#             else:
#                 self.reset_UI()

#         if text != '':
#             self.seed = int(text)

#     def _getOrigTrClassCount(self, target_lbl):
#         target_ID = self.Y_dict[target_lbl]
#         count = np.count_nonzero(self.origY_train == target_ID)
#         return count

#     def reset_balanceGroup(self):
#         self.balanceInfo = False
#         self.balance_group.setEnabled(False)
#         self.balance_table.clearContents()
#         self.strategy_combox.setCurrentText('Current')
#         self.strat_value.clear()

#     def init_balanceTable(self):
#         names, curr_cnt = zip(*self.trClassCounter.items())
#         n_rows = len(names)
#         self.balance_table.setRowCount(n_rows)
#         for row in range(n_rows):
#         # compile and set first three columns items as locked
#             i0 = QW.QTableWidgetItem(names[row])
#             i1 = QW.QTableWidgetItem(str(self._getOrigTrClassCount(names[row])))
#             i2 = QW.QTableWidgetItem(str(curr_cnt[row]))
#             for col, i in enumerate((i0, i1, i2)):
#                 i.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
#                 self.balance_table.setItem(row, col, i)
#         # set a lineEdit with integer validator to fourth column
#             i3 = QW.QLineEdit()
#             i3.setValidator(QG.QIntValidator(bottom=0))
#             i3.textEdited.connect(
#                 lambda: self.strategy_combox.setCurrentText('Custom multi-value'))
#             self.balance_table.setCellWidget(row, 3, i3)
#         # set current class counts in fourth column
#             self.balance_table.cellWidget(row, 3).setText(str(curr_cnt[row]))


#     def autoFill_balanceTable(self, value):
#         n_rows = self.balance_table.rowCount()
#         for row in range(n_rows):
#             self.balance_table.cellWidget(row, 3).setText(value)

#     def get_balanceTableColumnCells(self, col):
#         n_rows = self.balance_table.rowCount()
#         if col == 3:
#             cells = [self.balance_table.cellWidget(r, col).text() for r in range(n_rows)]
#         else:
#             cells = [self.balance_table.item(r, col).text() for r in range(n_rows)]
#         return cells

#     def enable_mOS(self, text):
#         enabled = text == 'BorderlineSMOTE'
#         self.mOS_spbox.setEnabled(enabled)

#     def enable_nUS(self, text):
#         enabled = text not in ('RandUS', 'TomekLinks')
#         self.nUS_spbox.setEnabled(enabled)

#     def update_USwarning(self, text):
#         if text in ('TomekLinks', 'ENN-all', 'ENN-mode', 'NCR-all', 'NCR-mode'):
#             self.US_warnIcon.show()
#         else:
#             self.US_warnIcon.hide()

#     def update_strategyValue(self, text):
#         counts = list(self.trClassCounter.values())
#         num = None

#         if text == 'Current':
#             self.init_balanceTable()
#         elif text == 'Min':
#             num = min(counts)
#         elif text == 'Max':
#             num = max(counts)
#         elif text == 'Mean':
#             num = int(np.mean(counts))
#         elif text == 'Median':
#             num = int(np.median(counts))

#         if num is not None:
#             self.strat_value.setText(str(num))


#     def _extThreadRunning(self):
#         threads = [self.balThread, self.learnThread]
#         for t in threads:
#             if t.isRunning():
#                 _name = t.objectName()
#                 QW.QMessageBox.critical(self, 'Error', 'Cannot perform this action '\
#                                         f'while {_name} operation is running.')
#                 return True
#         return False

#     def initialize_balancing(self):
#         if self._extThreadRunning():
#             return
#     # Ask user confirm + balance note
#         note = "Note: The number of samples won't be actually over-sampled/"\
#                 "under-sampled if no over-sample/under-sample algorithm has been selected."
#         choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                   f'Do you want to run the train set balancing operation?\n\n{note}',
#                   QW.QMessageBox.Yes | QW.QMessageBox.No, QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:

#         # Check for empty cells under the "After balancing" column
#             cells = self.get_balanceTableColumnCells(3)
#             if '' in cells:
#                 QW.QMessageBox.critical(self, 'X-Min Learn',
#                 'Some cells under the "Balanced" column are empty.')
#                 return

#         # Retreive input parameters for the balancing function
#             seed = self.seed
#             OS = self.OS_combox.currentText()
#             US = self.US_combox.currentText()
#             if OS == 'None': OS = None
#             if US == 'None': US = None
#             if OS == US == None: # check for algorithms absence
#                 QW.QMessageBox.critical(self, 'X-Min Learn',
#                 'Please select at least one sampling algorithm')
#                 return
#             kOS = self.kOS_spbox.value()
#             mOS = self.mOS_spbox.value()
#             nUS = self.nUS_spbox.value()
#             strategy = self.strategy_combox.currentText()
#             if strategy == 'Custom value':
#                 strategy = int(self.strat_value.text())
#             elif strategy in ('Custom multi-value', 'Current'):
#                 class_names = self.get_balanceTableColumnCells(0)
#                 custom_values = [int(c) for c in cells]
#                 strategy = dict(zip(class_names, custom_values))

#         # Disable Start and Cancel balancing buttons
#             self.runBalance_btn.setEnabled(False)
#             self.cancBalance_btn.setEnabled(False)

#         # Run the balancing thread
#             self.balance_pBar.setRange(0, 0)
#             task = lambda: ML_tools.balance_TrainSet(self.X_train, self.Y_train,
#                                                       strategy, OS, US, kOS, mOS,
#                                                       nUS, seed)
#             self.balThread.set_func(task)
#             self.balThread.start()


#     def finalize_balancing(self, thread_out):
#         self.balance_pBar.setRange(0, 4)

#     # Check the result from the balance external thread
#         if len(thread_out) == 1: # it means that the balancing operation has raised an exception
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                             'Balancing failed.', detailedText=repr(thread_out[0]))

#         else: # it means that the balancing operation was completed succesfully
#             self.X_train, self.Y_train, balInfo = thread_out

#         # Add new balancing info to the balancing operation tracer
#             if not self.balanceInfo:
#                 self.balanceInfo = []
#             self.balanceInfo.append(balInfo)
#             self.balance_pBar.setValue(1)

#         # Update Train set class counter and viewer
#             self.update_setClassCounter('Train', self.Y_train)
#             self.update_setClassViewer('Train')
#             self.balance_pBar.setValue(2)

#         # Recalc Train, Validation and Test sets rateos
#             tr_size = sum(self.trClassCounter.values())
#             vd_size = sum(self.vdClassCounter.values())
#             ts_size = sum(self.tsClassCounter.values())
#             tot_size = tr_size + vd_size + ts_size
#             trRateo = tr_size/tot_size
#             vdRateo = vd_size/tot_size
#             tsRateo = ts_size/tot_size

#         # Update Train, Validation & Test spin-boxes
#             self.trRateo_spbox.setValue(round(trRateo*100))
#             self.vdRateo_spbox.setValue(round(vdRateo*100))
#             self.tsRateo_spbox.setValue(round(tsRateo*100))

#         # Update pie chart and bar chart plots
#             self.update_tvtPlots()
#             self.balance_pBar.setValue(3)

#         # Add new rateos info into self.balaceInfo dictionary
#             self.balanceInfo[-1]['New TVT rateos'] = (round(trRateo, 2),
#                                                       round(vdRateo, 2),
#                                                       round(tsRateo, 2))

#         # Re-initialize the balance table
#             self.init_balanceTable()
#             self.balance_pBar.setValue(4)

#         # Exit balancing operations with success
#             QW.QMessageBox.information(self, 'X-Min Learn',
#                                         'Balancing operations concluded succesfully')

#     # Reset pbar and enable Start and Cancel buttons in any case
#         self.balance_pBar.reset()
#         self.runBalance_btn.setEnabled(True)
#         self.cancBalance_btn.setEnabled(True)


#     def canc_balancing(self):
#         choice = QW.QMessageBox.question(self, 'X-Min Learn',
#         'Warning: Any previous balancing operation on the train set will be discarded.',
#                                           QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                           QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:
#         # Delete all balancing info (to avoid MessageBox at the beginning of self.split_dataset())
#             self.balanceInfo = False
#         # Set Train, Validation & Test spin-boxes values with original rateos
#             trRateo, vdRateo, tsRateo = self.orig_TVTrateos
#             self.trRateo_spbox.setValue(round(trRateo*100))
#             self.vdRateo_spbox.setValue(round(vdRateo*100))
#             self.tsRateo_spbox.setValue(round(tsRateo*100))
#         # Re-launch split dataset function
#             self.split_dataset()

#     def load_parentModel(self):
#         if self._extThreadRunning():
#             return
#         if self._splitDatasetPerformed():
#             choice = QW.QMessageBox.warning(self, 'X-Min Learn', 'Warning: '\
#                      'The entire learnig procedure must be performed again. Continue?',
#                      QW.QMessageBox.Yes | QW.QMessageBox.No,
#                      QW.QMessageBox.No)
#             if choice == QW.QMessageBox.No:
#                 return
#             else:
#                 self.reset_UI()

#         path, _  = QW.QFileDialog.getOpenFileName(self, 'Load parent model',
#                                                   pref.get_dirPath('in'),
#                                                   'PyTorch Data File (*.pth)')
#         if path:
#             pref.set_dirPath('in', os.path.dirname(path))
#             pbar = cObj.PopUpProgBar(self, 4, 'Loading Model')
#             try:
#             # Import parent model variables
#                 modelVars = ML_tools.loadModel(path)
#                 pbar.setValue(1)

#             # Check parent model variables
#                 missing = ML_tools.missingVariables(modelVars)
#                 if len(missing):
#                     raise KeyError(f'Missing required variables: {missing}')
#                 pbar.setValue(2)

#             # Check that parent model shares the same X features of loaded GT dataset
#                 parent_xF = modelVars['ordered_Xfeat']
#                 dataset_xF = self.dataset.columns.to_list()[:-1]
#                 if parent_xF != dataset_xF:
#                     raise ValueError('The model and the ground-truth dataset '\
#                                      'do not share the same features.')
                
#                 ## add check for same y classes
#                 pbar.setValue(3)

#             # Update hyper-parameters and preferences
#                 self.parentModel_path.set_fullpath(path, predict_display=True)
#                 self.seedInput.setText(str(modelVars['seed']))
#                 self.lr_spbox.setValue(modelVars['lr'])
#                 self.wd_spbox.setValue(modelVars['wd'])
#                 self.mtm_spbox.setValue(modelVars['mtm'])
#                 self.regrType_combox.setCurrentText('Polynomial' if modelVars['regressorDegree']>1 else 'Linear')
#                 self.polyDeg_spbox.setValue(modelVars['regressorDegree']) # if it is 1, this will be ignored(?)
#                 # User must not change the regressor type and degree in update mode
#                 self.regrType_combox.setEnabled(False)
#                 self.polyDeg_spbox.setEnabled(False)
#                 self.algm_combox.setCurrentText(modelVars['algm_name'])
#                 self.optim_combox.setCurrentText(modelVars['optim_name'])
#                 pbar.setValue(4)
#                 QW.QMessageBox.information(self, 'X-Min Learn',
#                                            'Model loaded with success.')
#             except Exception as e:
#                 pbar.reset()
#                 cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                                 'The selected model produced an error.',
#                                 detailedText = repr(e))
#                 self.remove_parentModel()

#     def remove_parentModel(self):
#         self.parentModel_path.set_fullpath('')
#         self.parentModel_path.set_displayName('No model loaded')
#         # Allow the user to interact with the regressor type and degree when update mode is off
#         self.regrType_combox.setEnabled(True)
#         self.regrType_combox.setCurrentText('Linear')
#         self.polyDeg_spbox.setEnabled(False)
#         self.polyDeg_spbox.setValue(1)

#     def update_curve(self, graph, y_data):
#         y_train, y_valid = y_data
#         x_data = range(len(y_train))

#         if graph == 'loss':
#             canvas = self.lossView
#         elif graph == 'accuracy':
#             canvas = self.accuracyView
#         else:
#             raise ValueError(f'{graph} is not a valid key argument for graph')

#         if canvas.has_curves():
#             canvas.update_canvas([(x_data, y_train),
#                                   (x_data, y_valid)])
#         else:
#             canvas.add_curve(x_data, y_train, f'Train {graph}', 'b')
#             canvas.add_curve(x_data, y_valid, f'Validation {graph}', 'r')
#             canvas.update_canvas()

#     def update_ConfusionMatrix(self, set_name):
#         if set_name == 'Train':
#             true = self.Y_train
#             preds = self.tr_preds
#             asPerc = self.CMtrainPercAction.isChecked()
#             canvas = self.CMtrainView
#         elif set_name == 'Validation':
#             true = self.Y_valid
#             preds = self.vd_preds
#             asPerc = self.CMvalidPercAction.isChecked()
#             canvas = self.CMvalidView
#         elif set_name == 'Test':
#             true = self.Y_test
#             preds = self.ts_preds
#             asPerc = self.CMtestPercAction.isChecked()
#             canvas = self.CMtestView
#         else: return

#         if preds is None:
#             return

#         lbls, IDs = zip(*self.Y_dict.items())
#         CM = ML_tools.confusionMatrix(true, preds, IDs, percent=asPerc)
#         canvas.update_canvas(CM)
#         canvas.set_ticks(lbls)


#     def update_scoreLabel(self, set_name, metric, score):
#         names = ('Train', 'Validation', 'Test')
#         metrics = ('loss', 'accuracy', 'F1_micro', 'F1_macro', 'F1_weighted')
#         labels = np.array([[self.trLoss_lbl    , self.vdLoss_lbl    , None               ],
#                            [self.trAcc_lbl     , self.vdAcc_lbl     , self.tsAcc_lbl     ],
#                            [self.trF1Micro_lbl , self.vdF1Micro_lbl , self.tsF1Micro_lbl ],
#                            [self.trF1Macro_lbl , self.vdF1Macro_lbl , self.tsF1Macro_lbl ],
#                            [self.trF1Weight_lbl, self.vdF1Weight_lbl, self.tsF1Weight_lbl]])
#         col = names.index(set_name)
#         row = metrics.index(metric)
#         lbl = labels[row, col]
#         if lbl is not None:
#             labels[row, col].setText(str(score))

# #     def start_learning(self):
# #         if self._extThreadRunning():
# #             return
# #     # Check that Train, Validation and Test sets share the same unique classes
# #         tr_unq, vd_unq, ts_unq = [np.unique(arr) for arr in (self.Y_train, self.Y_valid, self.Y_test)]
# #         if not np.array_equal(tr_unq, vd_unq) or not np.array_equal(tr_unq, ts_unq):
# #             QW.QMessageBox.critical(self, 'Datasets inconsistency',
# #                                     "Train, validation and test sets do not share the same classes.")
# #             return

# #     # Update Main Actions buttons (START-STOP-TEST-LEARN)
# #         self.startLearn_btn.setEnabled(False)
# #         self.stopLearn_btn.setEnabled(True)
# #         self.testModel_btn.setEnabled(False)
# #         self.saveModel_btn.setEnabled(False)

# #     # Check if it is required to update a previous model
# #         parent_path = self.parentModel_path.get_fullpath()
# #         update_model = parent_path != ''
# #         if update_model:
# #             parent_vars = ML_tools.loadModel(parent_path)
# #         else: parent_vars = None

# #     # Convert Train and Validation features and targets to torch Tensors
# #         X_train = ML_tools.array2Tensor(self.X_train)
# #         Y_train = ML_tools.array2Tensor(self.Y_train, 'uint8')
# #         X_valid = ML_tools.array2Tensor(self.X_valid)
# #         Y_valid = ML_tools.array2Tensor(self.Y_valid, 'uint8')

# #     # Normalize feature data
# #         if update_model:
# #             X_mean, X_std = parent_vars['standards']
# #             X_train_norm = ML_tools.norm_data(X_train, X_mean, X_std,
# #                                               return_standards=False)
# #         else:
# #             X_train_norm, X_mean, X_std = ML_tools.norm_data(X_train)

# #         X_valid_norm = ML_tools.norm_data(X_valid, X_mean, X_std,
# #                                           return_standards=False)

# #     # Get Hyper-parameters and seed
# #         seed = self.seed
# #         lr = self.lr_spbox.value()
# #         wd = self.wd_spbox.value()
# #         mtm = self.mtm_spbox.value()
# #         epochs = self.epochs_spbox.value()

# #     # Get learning preferences
# #         algm_name = self.algm_combox.currentText()
# #         optm_name = self.optim_combox.currentText()

# #     # Initialize model
# #         self.model = ML_tools.getModel(algm_name,
# #                                        self.X_train.shape[1],
# #                                        len(self.Y_dict), # Y_dict includes parent model classes
# #                                        seed=seed)
# #         if update_model:
# #             ML_tools.embed_modelParameters(parent_vars['model_state_dict'], self.model)

# #     # Set device
# #         useGPU = self.cuda_cbox.isChecked()
# #         device = "cuda" if (useGPU & ML_tools.cuda_available()) else "cpu"
# #         self.model.to(device)

# #     # Set optimizer function
# #         optimizer = ML_tools.getOptimizer(optm_name, self.model, lr, mtm, wd)
# #         # PROBABLY NOT NEEDED BECAUSE GETS THE PARAMETERS FROM MODEL (already updated),
# #         # AND HYPER-PARAMETERS CAN BE CHANGED BY USER
# #         # if update_model:
# #         #     optimizer.load_state_dict(parent_vars['optim_state_dict'])
# #         #     # after loading the parent state dict refresh the hyper-parameters
# #         #     for g in optimizer.param_groups:
# #         #         g['lr'] = lr
# #         #         g['momentum'] = mtm
# #         #         g['weight_decay'] = wd


# #     # Set progress bar range
# #         e_min = parent_vars['epochs'] if update_model else 0
# #         e_max = e_min + epochs
# #         self.progBar.setRange(e_min, e_max)

# # # ---------------------- START LEARNING SESSION ---------------------- #

# #     # Initialize losses and accuracy lists
# #         if update_model:
# #             self.tr_losses, self.vd_losses = parent_vars['loss_list']
# #             self.tr_acc, self.vd_acc = parent_vars['accuracy_list']
# #         else:
# #             self.tr_losses, self.vd_losses = [], []
# #             self.tr_acc, self.vd_acc = [], []

# #     # Adjust graphics update rate
# #         upRate = int(self.graph_updateRate.text())
# #         if upRate == 0: upRate = e_max #prevent ZeroDivisionError when update graphics

# #     # Clear curve plots
# #         for plot in (self.lossView, self.accuracyView):
# #             plot.clear_canvas()

# #     # Start iterating through epochs
# #         for e in range(e_min, e_max):

# #             # (Check for user cancel request)
# #             if self.cancel_learning:
# #                 self.cancel_learning = False
# #                 e -= 1
# #                 break

# #             # (Learn)
# #             (train_loss,
# #              valid_loss,
# #              tr_preds,
# #              vd_preds) = self.model.learn(X_train_norm, Y_train,
# #                                           X_valid_norm, Y_valid,
# #                                           optimizer, device)

# #             # (Compute accuracy)
# #             train_acc = ML_tools.accuracy(self.Y_train, tr_preds)
# #             valid_acc = ML_tools.accuracy(self.Y_valid, vd_preds)

# #             # (Store new loss and accuracy values)
# #             self.tr_losses.append(train_loss)
# #             self.vd_losses.append(valid_loss)
# #             self.tr_acc.append(train_acc)
# #             self.vd_acc.append(valid_acc)

# #             # (Update loss and accuracy plots and labels)
# #             if (e+1) % upRate == 0:
# #                 self.update_curve('loss', e+1, (self.tr_losses, self.vd_losses))
# #                 self.update_curve('accuracy', e+1, (self.tr_losses, self.vd_losses))
# #                 self.update_scoreLabel('Train', 'loss', self.tr_losses[-1])
# #                 self.update_scoreLabel('Validation', 'loss', self.vd_losses[-1])
# #                 self.update_scoreLabel('Train', 'accuracy', self.tr_acc[-1])
# #                 self.update_scoreLabel('Validation', 'accuracy', self.vd_acc[-1])

# #             # (Update progress bar)
# #             self.progBar.setValue(e)

# #     # Update loss and accuracy plots and labels on exit
# #         self.update_curve('loss', e+1, (self.tr_losses, self.vd_losses))
# #         self.update_curve('accuracy', e+1, (self.tr_losses, self.vd_losses))
# #         self.update_scoreLabel('Train', 'loss', self.tr_losses[-1])
# #         self.update_scoreLabel('Validation', 'loss', self.vd_losses[-1])
# #         self.update_scoreLabel('Train', 'accuracy', self.tr_acc[-1])
# #         self.update_scoreLabel('Validation', 'accuracy', self.vd_acc[-1])

# #     # Update Confusion Matrices
# #         self.tr_preds = self.model.predict(X_train_norm.to(device))[1].cpu()
# #         self.vd_preds = self.model.predict(X_valid_norm.to(device))[1].cpu()
# #         self.update_ConfusionMatrix('Train')
# #         self.update_ConfusionMatrix('Validation')

# #     # Update Train and Validation F1 scores

# #         #               micro  |  macro  |  weighted
# #         # F1 train             |         |
# #         # F1 validation        |         |
# #         # F1 test              |         |

# #         f1_scores = np.empty((3,3))
# #         for n, avg in enumerate(('micro', 'macro', 'weighted')):
# #             f1_scores[0, n] = _tr = ML_tools.f1score(self.Y_train, self.tr_preds, avg)
# #             f1_scores[1, n] = _vd = ML_tools.f1score(self.Y_valid,  self.vd_preds, avg)
# #             self.update_scoreLabel('Train', f'F1_{avg}', _tr)
# #             self.update_scoreLabel('Validation', f'F1_{avg}', _vd)

# #     # Save model variables in memory
# #         self.model_vars = {'algm_name'        : self.model._name,
# #                            'loss_name'        : self.model._loss,
# #                            'optim_name'       : optm_name,
# #                            'optim_state_dict' : optimizer.state_dict(),
# #                            'model_state_dict' : self.model.state_dict(),
# #                            'standards'        : (X_mean, X_std),
# #                            'parentModel_path' : parent_path,
# #                            'seed'             : seed,
# #                            'epochs'           : len(self.tr_losses),
# #                            'lr'               : lr,
# #                            'wd'               : wd,
# #                            'mtm'              : mtm,
# #                            'GT_dataset_path'  : self.DS_path.get_fullpath(),
# #                            'TVT_rateos'       : self.orig_TVTrateos,
# #                            'balancing_info'   : self.balanceInfo,
# #                            'ordered_Xfeat'    : self.dataset.columns.to_list()[:-1],
# #                            'Y_dict'           : self.Y_dict,
# #                            'F1_scores'        : f1_scores,
# #                            'accuracy'         : [train_acc, valid_acc],
# #                            'loss'             : [train_loss, valid_loss],
# #                            'accuracy_list'    : (self.tr_acc, self.vd_acc),
# #                            'loss_list'        : (self.tr_losses, self.vd_losses)}

# #     # Update Main Actions buttons (START-STOP-TEST-LEARN) on exit
# #         self.startLearn_btn.setEnabled(True)
# #         self.stopLearn_btn.setEnabled(False)
# #         self.testModel_btn.setEnabled(True)
# #         self.saveModel_btn.setEnabled(False)

# #     # Reset progress bar and end learning session with success
# #         self.progBar.reset()
# #         QW.QMessageBox.information(self, 'Learning completed',
# #                                     'Learning session completed succesfully.')

# #     def stop_learning(self):
# #         self.cancel_learning = True

#     def initialize_learning(self):
#         if self._extThreadRunning():
#             return
#     # Check that Train, Validation and Test sets share the same unique classes
#         tr_unq, vd_unq, ts_unq = [np.unique(arr) for arr in (self.Y_train, self.Y_valid, self.Y_test)]
#         if not np.array_equal(tr_unq, vd_unq) or not np.array_equal(tr_unq, ts_unq):
#             QW.QMessageBox.critical(self, 'X-Min Learn',
#                                     "Train, validation and test sets do not share the same classes.")
#             return

#     # Set the progress bar temporarily
#         self.progBar.setRange(0, 4)
#         self.progBar.setTextVisible(False)

#     # Update Main Actions buttons (START-STOP-TEST-LEARN)
#         self.startLearn_btn.setEnabled(False)
#         self.stopLearn_btn.setEnabled(True)
#         self.testModel_btn.setEnabled(False)
#         self.saveModel_btn.setEnabled(False)

#     # Check if it is required to update a previous model
#         parent_path = self.parentModel_path.get_fullpath()
#         update_model = parent_path != ''
#         parent_vars = ML_tools.loadModel(parent_path) if update_model else None # EagerModel.load(parent_path)
#         self.progBar.setValue(1)

#     # Map features from linear to polynomial if required
#         regrDegree = self.polyDeg_spbox.value()
#         if regrDegree > 1:
#             X_train = ML_tools.map_polinomial_features(self.X_train, regrDegree)
#             X_valid = ML_tools.map_polinomial_features(self.X_valid, regrDegree)
#         else:
#             X_train = self.X_train.copy()
#             X_valid = self.X_valid.copy()
#         self.progBar.setValue(2)

#     # Convert Train and Validation features and targets to torch Tensors
#         X_train = ML_tools.array2tensor(X_train, 'float32')
#         Y_train = ML_tools.array2tensor(self.Y_train, 'uint8')
#         X_valid = ML_tools.array2tensor(X_valid, 'float32')
#         Y_valid = ML_tools.array2tensor(self.Y_valid, 'uint8')
#         self.progBar.setValue(3)

#     # Normalize feature data
#         if update_model:
#             X_mean, X_std = parent_vars['standards']
#             X_train_norm = ML_tools.norm_data(X_train, X_mean, X_std,
#                                               return_standards=False)
#         else:
#             X_train_norm, X_mean, X_std = ML_tools.norm_data(X_train)

#         X_valid_norm = ML_tools.norm_data(X_valid, X_mean, X_std,
#                                           return_standards=False)
#         self.progBar.setValue(4)


#     # Get Hyper-parameters and seed
#         seed = self.seed
#         lr = self.lr_spbox.value()
#         wd = self.wd_spbox.value()
#         mtm = self.mtm_spbox.value()
#         epochs = self.epochs_spbox.value()

#     # Get learning preferences
#         algm_name = self.algm_combox.currentText()
#         optm_name = self.optim_combox.currentText()

#     # Initialize model
#         self.model = ML_tools.getModel(algm_name,
#                                        X_train.shape[1],
#                                        len(self.Y_dict), # Y_dict includes parent model classes
#                                        seed=seed)
#         if update_model:
#             ML_tools.embed_modelParameters(parent_vars['model_state_dict'], self.model)

#     # Set device
#         useGPU = self.cuda_cbox.isChecked()
#         device = "cuda" if (useGPU & ML_tools.cuda_available()) else "cpu"
#         self.model.to(device)

#     # Set optimizer function
#         self.optimizer = ML_tools.getOptimizer(optm_name, self.model, lr, mtm, wd)

#         # PROBABLY NOT NEEDED BECAUSE GETS THE PARAMETERS FROM MODEL (already updated),
#         # AND HYPER-PARAMETERS CAN BE CHANGED BY USER
#         # if update_model:
#         #     optimizer.load_state_dict(parent_vars['optim_state_dict'])
#         #     # after loading the parent state dict refresh the hyper-parameters
#         #     for g in optimizer.param_groups:
#         #         g['lr'] = lr
#         #         g['momentum'] = mtm
#         #         g['weight_decay'] = wd


#     # Reset progress bar range
#         e_min = parent_vars['epochs'] if update_model else 0
#         e_max = e_min + epochs
#         self.progBar.reset()
#         self.progBar.setRange(e_min, e_max)
#         self.progBar.setTextVisible(True)

#     # Initialize losses and accuracy lists
#         if update_model:
#             self.tr_losses, self.vd_losses = parent_vars['loss_list']
#             self.tr_acc, self.vd_acc = parent_vars['accuracy_list']
#         else:
#             self.tr_losses, self.vd_losses = [], []
#             self.tr_acc, self.vd_acc = [], []

#     # Adjust graphics update rate
#         upRate = int(self.graph_updateRate.text())
#         if upRate == 0:
#             upRate = e_max # prevent ZeroDivisionError when update graphics
#         elif upRate / epochs < 0.02:
#             upRate = epochs * 0.02 # prevent too fast updates that crashes the thread(?)

#     # Clear curve plots
#         for plot in (self.lossView, self.accuracyView):
#             plot.clear_canvas()

#     # Initialize model variables in order. Some variables will be provided in finalize_learning() func
#         self.model_vars = {'algm_name'        : self.model._name,
#                            'loss_name'        : self.model._loss,
#                            'optim_name'       : optm_name,
#                            'optim_state_dict' : None,
#                            'model_state_dict' : None,
#                            'regressorDegree'  : regrDegree,
#                            'standards'        : (X_mean, X_std),
#                            'parentModel_path' : parent_path,
#                            'GT_dataset_path'  : self.DS_path.get_fullpath(),
#                            'TVT_rateos'       : self.orig_TVTrateos,
#                            'balancing_info'   : self.balanceInfo,
#                            'device'           : device,
#                            'seed'             : seed,
#                            'epochs'           : None,
#                            'lr'               : lr,
#                            'wd'               : wd,
#                            'mtm'              : mtm,
#                            'accuracy_list'    : None,
#                            'loss_list'        : None,
#                            'accuracy'         : None,
#                            'loss'             : None,
#                            'F1_scores'        : None,
#                            'ordered_Xfeat'    : self.dataset.columns.to_list()[:-1],
#                            'Y_dict'           : self.Y_dict}

#     # Start the learning thread
#         task = lambda: self.model.learn(X_train_norm, Y_train,
#                                         X_valid_norm, Y_valid,
#                                         self.optimizer, device)
#         self.learnThread.setParameters(task, (self.Y_train, self.Y_valid),
#                                        (e_min, e_max), upRate)
#         self.learnThread.start()



#     def update_learningScores(self, thread_out):
#     # Extract objects from thread output
#         e, (tr_loss, vd_loss), (tr_acc, vd_acc) = thread_out
#     # Store new loss and accuracy values
#         self.tr_losses.append(tr_loss)
#         self.vd_losses.append(vd_loss)
#         self.tr_acc.append(tr_acc)
#         self.vd_acc.append(vd_acc)
#     # Update progress bar
#         self.progBar.setValue(e)

#     def update_learningScoresView(self):
#         self.update_curve('loss', (self.tr_losses, self.vd_losses))
#         self.update_curve('accuracy', (self.tr_acc, self.vd_acc))
#         self.update_scoreLabel('Train', 'loss', self.tr_losses[-1])
#         self.update_scoreLabel('Validation', 'loss', self.vd_losses[-1])
#         self.update_scoreLabel('Train', 'accuracy', self.tr_acc[-1])
#         self.update_scoreLabel('Validation', 'accuracy', self.vd_acc[-1])

#     def finalize_learning(self, thread_out):
#     # Reset the progress bar
#         self.progBar.reset()

#     # Check the result from the learning external thread:
#         completed, e = thread_out
#         if not completed: # it means that the learning operation has raised an exception
#             cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                             'Learning failed.', detailedText=repr(e))

#         else:
#         # Set the Progress Bar temporarily
#             self.progBar.setRange(0, 4)
#             self.progBar.setTextVisible(False)

#         # Update loss and accuracy plots and labels on exit
#             self.update_learningScoresView()
#             self.progBar.setValue(1)

#         # Populate missing model variables in memory (except F1 scores)
#             self.model_vars['epochs']           = len(self.tr_losses)
#             self.model_vars['optim_state_dict'] = self.optimizer.state_dict()
#             self.model_vars['model_state_dict'] = self.model.state_dict()
#             self.model_vars['accuracy_list']    = (self.tr_acc, self.vd_acc)
#             self.model_vars['loss_list']        = (self.tr_losses, self.vd_losses)
#             self.model_vars['accuracy']         = [self.tr_acc[-1], self.vd_acc[-1]]
#             self.model_vars['loss']             = [self.tr_losses[-1], self.vd_losses[-1]]
#             self.progBar.setValue(2)

#         # Apply the model one more time to get final predictions
#         # Also we apply the predict function of the model in order to compute the softmax function
#         # This is better than using the predictions from forward() because they're based on 'pure' linear outputs
#             # regrDegree = self.model_vars['regressorDegree']
#             # if regrDegree > 1:
#             #     X_train = ML_tools.map2Polynomial(self.X_train, regrDegree)

#             # X_train_norm = ML_tools.norm_data(ML_tools.array2Tensor(self.X_train),
#             #                                   *self.model_vars['standards'], False)
#             # X_valid_norm = ML_tools.norm_data(ML_tools.array2Tensor(self.X_valid),
#             #                                   *self.model_vars['standards'], False)
#             # self.tr_preds = self.model.predict(X_train_norm.to(self.model_vars['device']))[1].cpu()
#             # self.vd_preds = self.model.predict(X_valid_norm.to(self.model_vars['device']))[1].cpu()
#             self.tr_preds = ML_tools.applyModel(self.model_vars, self.X_train)[1]
#             self.vd_preds = ML_tools.applyModel(self.model_vars, self.X_valid)[1]
#             self.progBar.setValue(3)

#         # Update Train and Validation F1 scores and save them in memory

#             #               micro  |  macro  |  weighted
#             # F1 train             |         |
#             # F1 validation        |         |
#             # F1 test              |         |

#             f1_scores = np.empty((3,3))
#             for n, avg in enumerate(('micro', 'macro', 'weighted')):
#                 f1_scores[0, n] = _tr = ML_tools.f1score(self.Y_train, self.tr_preds, avg)
#                 f1_scores[1, n] = _vd = ML_tools.f1score(self.Y_valid, self.vd_preds, avg)
#                 self.update_scoreLabel('Train', f'F1_{avg}', _tr)
#                 self.update_scoreLabel('Validation', f'F1_{avg}', _vd)
#             self.model_vars['F1_scores'] = f1_scores
#             self.progBar.setValue(4)

#         # Update Confusion Matrices
#             self.update_ConfusionMatrix('Train')
#             self.update_ConfusionMatrix('Validation')

#         # Enable testing and end learning session with success
#             self.testModel_btn.setEnabled(True)
#             self.progBar.reset()
#             self.progBar.setTextVisible(True)
#             QW.QMessageBox.information(self, 'X-Min Learn',
#                                        'Learning session completed succesfully.')

#     # Update Main Actions buttons (START-STOP-LEARN) on exit
#         self.startLearn_btn.setEnabled(True)
#         self.stopLearn_btn.setEnabled(False)
#         self.saveModel_btn.setEnabled(False)


#     def stop_learning(self):
#         self.learnThread.requestInterruption()

#     def test_model(self):
#         # Handling of potential GUI bugs were save model button is active when it should not
#         if self.model is None:
#             self.saveModel_btn.setEnabled(False)
#             return

#         choice = QW.QMessageBox.warning(self, 'X-Min Learn', 'Are you sure to '\
#                 'test your model?\nWarning: once the model is tested on Test set '\
#                 'it should not be further trained with the same Train set.',
#                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                         QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:
#             self.progBar.setRange(0, 5)

#         # Predict targets on Test set
#             self.ts_preds = ML_tools.applyModel(self.model_vars, self.X_test)[1]
#             self.progBar.setValue(1)

#         # Compute Test accuracy and update label and model variable
#             test_acc = ML_tools.accuracy(self.Y_test, self.ts_preds)
#             self.update_scoreLabel('Test', 'accuracy', test_acc)
#             self.model_vars['accuracy'].append(test_acc)
#             self.progBar.setValue(2)

#         # Compute Test F1 scores and update label and model variable
#             for n, avg in enumerate(('micro', 'macro', 'weighted')):
#                 _ts = ML_tools.f1score(self.Y_test, self.ts_preds, avg)
#                 self.update_scoreLabel('Test', f'F1_{avg}', _ts)
#                 self.model_vars['F1_scores'][2, n] = _ts
#             self.progBar.setValue(3)

#         # Update Test Confusion Matrix
#             self.update_ConfusionMatrix('Test')
#             self.progBar.setValue(4)

#         # Populate model preview
#             text = ''
#             for k, v in self.model_vars.items():
#                 text += f'{k.upper()} = {repr(v)}\n\n'
#             self.modelPreview.clear()
#             self.modelPreview.setText(text)
#             self.progBar.setValue(5)

#         # Enable Save model button on exit
#             self.saveModel_btn.setEnabled(True)

#         # Reset progress bar and end testing session with success
#             self.progBar.reset()
#             QW.QMessageBox.information(self, 'X-Min Learn',
#                                         'Your model has been tested succesfully.')


#     def save_model(self):
#         path, _  = QW.QFileDialog.getSaveFileName(self, 'Save model',
#                                                   pref.get_dirPath('out'),
#                                                   'PyTorch Data File (*.pth)')
#         if path:
#             pref.set_dirPath('out', os.path.dirname(path))
#             try:
#                 log_path = CF.extend_filename(path, '_log', ext='.txt')
#                 extendedLog = pref.get_setting('class/extLog', False, bool)
#                 ML_tools.saveModel(self.model_vars, path, log_path, extendedLog)
#                 QW.QMessageBox.information(self, 'Model saved',
#                                            'The model was saved succesfully.')
#             except Exception as e:
#                 cObj.RichMsgBox(self, QW.QMessageBox.Critical, 'X-Min Learn',
#                                 'An error was raised while saving the model.',
#                                  detailedText=repr(e))


#     def closeEvent(self, event):
#         choice = QW.QMessageBox.question(self, 'X-Min Learn',
#                                           "Do you really want to close the Model Learner?",
#                                           QW.QMessageBox.Yes | QW.QMessageBox.No,
#                                           QW.QMessageBox.No)
#         if choice == QW.QMessageBox.Yes:
#             event.accept()
#         else:
#             event.ignore()




class PhaseRefiner(QW.QWidget):

    def __init__(self, minmap_array, parent):
        self.parent = parent
        super(PhaseRefiner, self).__init__()
        self.setWindowTitle('Phase Refiner')
        self.setWindowIcon(QIcon('Icons/refine.png'))
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.minmap = minmap_array
        self.phases = np.unique(self.minmap)
        self._backup = self.minmap.copy()

    # Build Tab Widget
        self._basicTab = self.BasicRefiner(self)
        self._advancedTab = self.AdvancedRefiner(self)

        self.tabWidget = QW.QTabWidget()
        self.setStyleSheet('''QTabBar {color: black;}'''
                           '''QTabWidget {background-color: rgb(169, 185, 188);}''')
        self.tabWidget.addTab(self._basicTab, 'Basic')
        self.tabWidget.addTab(self._advancedTab, 'Advanced')

    # Adjust Main Layout
        mainLayout = QW.QHBoxLayout()
        mainLayout.addWidget(self.tabWidget)
        self.setLayout(mainLayout)

        self.adjustSize()

    def closeEvent(self, event):
        choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                         "Do you really want to close the Phase Refiner?",
                                         QW.QMessageBox.Yes | QW.QMessageBox.No,
                                         QW.QMessageBox.No)
        if choice == QW.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


    class BasicRefiner(QW.QWidget):

        def __init__(self, parent):

            self.parent = parent
            super(parent.BasicRefiner, self).__init__()

            self.phases = self.parent.phases
            self.minmap = self.parent.minmap
            self.refined = self.minmap.copy()

            self.originalModeDict = dict(zip(*CF.get_mode(self.minmap, ordered=True)))
            self.refinedModeDict = self.originalModeDict.copy()

            self.mainParent = self.parent.parent

            self.init_ui()

        def init_ui(self):

        # Kernel size selector widget
            self.kernelSel = cObj.KernelSelector()

        # Preserve image borders checkbox
            self.psvBorders_cbox = QW.QCheckBox('Preserve borders')
            self.psvBorders_cbox.setToolTip('The borders thickness will be choosen according to the kernel size')

        # Nan value selector combobox
            self.nanVal_combox = QW.QComboBox()
            self.nanVal_combox.addItem('None')
            self.nanVal_combox.addItems(self.phases)
            self.nanVal_combox.setToolTip('Treat the selected phase as No Data')
            self.nanVal_combox.currentTextChanged.connect(
                lambda txt: self.nanTol_spbox.setEnabled(txt!='None'))

        # Nan tolerance selector double spinbox
            self.nanTol_spbox = QW.QDoubleSpinBox()
            self.nanTol_spbox.setDecimals(2)
            self.nanTol_spbox.setValue(0.5)
            self.nanTol_spbox.setRange(0, 0.99)
            self.nanTol_spbox.setSingleStep(0.1)
            self.nanTol_spbox.setToolTip('If the amount of NaN pixels in Kernel are bigger than this value, '\
                                         'the output will be forced to be the NaN value')
            self.nanTol_spbox.setEnabled(False)

        # Apply button
            self.apply_btn = QW.QPushButton('Apply')
            self.apply_btn.setStyleSheet('''background-color: rgb(50,205,50);
                                             font-weight: bold''')
            self.apply_btn.clicked.connect(self.applyFilter)

        # Save button
            self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'Save')
            self.save_btn.setEnabled(False)
            self.save_btn.clicked.connect(self.save)

        # Adjust preferences group
            nan_form = QW.QFormLayout()
            nan_form.addRow('NaN value', self.nanVal_combox)
            nan_form.addRow('NaN tolerance percentage', self.nanTol_spbox)

            btn_hbox = QW.QHBoxLayout()
            btn_hbox.addWidget(self.apply_btn)
            btn_hbox.addWidget(self.save_btn)

            pref_vbox = QW.QVBoxLayout()
            pref_vbox.addWidget(self.kernelSel)
            pref_vbox.addWidget(self.psvBorders_cbox)
            pref_vbox.addLayout(nan_form)
            pref_vbox.addLayout(btn_hbox)
            pref_group = cObj.GroupArea(pref_vbox, 'Preferences')

        # Original map canvas + NavTbar
            self.origCanvas = cObj.DiscreteClassCanvas(self, tight=True)
            self.origCanvas.update_canvas('Original', self.minmap,
                                          colors = self.mainParent._minmaptab.MinMapView.curr_colors)
            self.origCanvas.setMinimumSize(100, 100)
            self.origNTbar = cObj.NavTbar(self.origCanvas, self)
            self.origNTbar.fixHomeAction()
            self.origNTbar.removeToolByIndex([3, 4, 8, 9])

        # Original map Mode Bar Chart
            self.origMode = cObj.BarCanvas(self, size=(5.0, 3.5))
            self._updateModePlot('Original')

        # Refined map canvas + NavTbar
            self.refCanvas = cObj.DiscreteClassCanvas(self, tight=True)
            self.refCanvas.setMinimumSize(100, 100)
            self.refCanvas.update_canvas('Refined', self.minmap,
                                          colors = self.origCanvas.curr_colors)
            CF.shareAxis(self.origCanvas.ax, self.refCanvas.ax, True)
            self.refNTbar = cObj.NavTbar(self.refCanvas, self)
            self.refNTbar.fixHomeAction()
            self.refNTbar.removeToolByIndex([3, 4, 8, 9])

        # Refined map Mode Bar Chart
            self.refMode = cObj.BarCanvas(self, size=(5.0, 3.5))
            self._updateModePlot('Refined')

        # Legend List widget
            self.legend = cObj.CanvasLegend(self.origCanvas, amounts=False,
                                            childrenCanvas=[self.refCanvas])
            self.legend.itemColorChanged.connect(
                lambda: self._updateModePlot('Original'))
            self.legend.itemColorChanged.connect(
                lambda: self._updateModePlot('Refined'))

        # Adjust Main Layout
            left_area = QW.QVBoxLayout()
            left_area.addWidget(self.legend)
            left_area.addWidget(pref_group)

            plot_area = QW.QGridLayout()
            for row, (w_orig, w_ref) in enumerate((
                (self.origNTbar,  self.refNTbar),
                (self.origCanvas, self.refCanvas),
                (self.origMode,   self.refMode))):
                plot_area.addWidget(w_orig, row, 0)
                plot_area.addWidget(w_ref,  row, 1)
            plot_area.setRowStretch(1, 2)
            plot_area.setRowStretch(2, 1)

            main_split = cObj.SplitterGroup((left_area, plot_area), (1, 2)) # use SplitterLayout

            mainLayout = QW.QHBoxLayout()
            mainLayout.addWidget(main_split)
            self.setLayout(mainLayout)


        def _updateRefinedCanvas(self, lbl_arr):
            self.refCanvas.update_canvas('Refined', lbl_arr)
            self.legend._transferColors()
            self._updateModePlot('Refined')
            # lbl, mode = CF.get_mode(lbl_arr, ordered=True)
            # self.refMode.update_canvas('Refined Mineral Mode', mode, lbl)

        def _updateModePlot(self, plot):
            if plot == 'Refined':
                canvas = self.refMode
                modeDict = self.refinedModeDict
            elif plot == 'Original':
                canvas = self.origMode
                modeDict = self.originalModeDict
            else : raise NameError(f'{plot}')

            lbl, mode = zip(*modeDict.items())
            # Use the current colors of original canvas, which includes always all the classes
            col_dict = dict(zip(self.phases, self.origCanvas.curr_colors))
            # The sort_dict_by_list func excludes from col_dict the keys not in lbl (see documentation)
            col_dict = CF.sort_dict_by_list(col_dict, lbl)
            canvas.update_canvas(f'{plot} Mineral Mode', mode, lbl,
                                 colors=list(col_dict.values()))

        # def get_arrayCenter(self, arr):
        #     size = len(arr)
        #     idx = size//2
        #     # if size % 2 == 0:
        #     #     idx += (size**0.5)//2
        #     center = arr[int(idx)]
        #     return center

        def preserve_borders(self, orig, filt, tck):
            # tck = border thickness
            if tck <= 0:
                return filt
            out_mask = np.zeros(orig.shape)
            out_mask[tck:-tck, tck:-tck] = 1
            out = np.where(out_mask, filt, orig)
            return out

        def max_freq(self, arr, nan=None, nan_tolerance=0.5):
            arr = arr.astype(int)
            nan_cnt = np.count_nonzero(arr==nan)
            if nan_cnt > len(arr) * nan_tolerance:
                return nan

            f = np.bincount(arr[arr!=nan])
            out = f.argmax()
            # max_f = np.where(f == f.max())[0]
            # if len(max_f) > 1:
            #     center = self.get_arrayCenter(arr)
            #     if center in max_f:
            #         out = center

            return out

        def applyFilter(self):
            pbar = cObj.PopUpProgBar(self, 5, 'Applying filter',
                                     cancel=False, forceShow=True)
            arr = self.origCanvas.data
            transDict = self.origCanvas.dataDict
            kernel = self.kernelSel.get_structure()
            excl_borders = self.psvBorders_cbox.isChecked()
            nan_str = self.nanVal_combox.currentText()
            nan = None if nan_str=='None' else transDict[nan_str]
            nan_tol = self.nanTol_spbox.value()
            pbar.setValue(1)

            freq = nd.generic_filter(arr, function=self.max_freq, footprint=kernel, mode='nearest',
                                     extra_keywords={'nan': nan, 'nan_tolerance': nan_tol})
            if excl_borders:
                border_tck = kernel.shape[0] // 2
                freq = self.preserve_borders(arr, freq, border_tck)
            pbar.setValue(2)

            self.refined = CF.decode_labels(freq, transDict)
            pbar.setValue(3)

            self.refinedModeDict = dict(zip(*CF.get_mode(self.refined, ordered=True)))
            pbar.setValue(4)

            self._updateRefinedCanvas(self.refined)
            self.save_btn.setEnabled(True)
            pbar.setValue(5)

        def save(self):
            outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save Map',
                                                        pref.get_dirPath('out'),
                                                        '''Compressed ASCII file (*.gz)
                                                           ASCII file (*.txt)''')
            if outpath:
                pref.set_dirPath('out', os.path.dirname(outpath))
                np.savetxt(outpath, self.refined, fmt='%s')
                autoload = QW.QMessageBox.question(self, 'X-Min Learn',
                                                   'Do you want to load the result in the "Mineral Map Tab"?',
                                                   QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                   QW.QMessageBox.Yes)
                if autoload == QW.QMessageBox.Yes:
                    self.mainParent._minmaptab.loadMaps([outpath])


    class AdvancedRefiner(QW.QWidget):

        def __init__(self, parent):

            self.parent = parent
            super(parent.AdvancedRefiner, self).__init__()

            self.minmap = self.parent.minmap
            self.phases = self.parent.phases
            self.mainParent = self.parent.parent

            self.algmDict = {'Erosion + Reconstruction' : lambda x, s, mask: nd.binary_propagation(
                                                                             nd.binary_erosion(x, s, mask=mask),
                                                                             structure=s, mask=x),
                             'Opening (Erosion + Dilation)' : nd.binary_opening,
                             'Closing (Dilation + Erosion)' : nd.binary_closing,
                             'Erosion' :    nd.binary_erosion,
                             'Dilation' :   nd.binary_dilation,
                             'Fill holes' : nd.binary_fill_holes}

            self.varianceRGB = {-1 : (255,0,0),      # -> -1 new background pixel
                                 0 : (0,0,0),        # ->  0 same background pixel
                                 1 : (255,255,255),  # ->  1 same phase pixel
                                 2 : (0,255,0)}      # ->  2 new phase pixel

            self.roi = None

            self.init_ui()
            self.update_phaseView()


        def init_ui(self):

        # Mineral phases TOC
            self.TOC = cObj.RadioBtnLayout(self.phases)
            for btn in self.TOC.get_buttonList():
                btn.clicked.connect(self.update_phaseView)
            TOC_group = cObj.GroupScrollArea(self.TOC, 'Mineral Phases')
            TOC_group.setStyleSheet('border: 0px;')

        # Reset original phase button
            self.resetPhase_btn = QW.QPushButton('Reset')
            self.resetPhase_btn.setStyleSheet('''background-color: red;
                                                 font-weight: bold;
                                                 color: white''')
            self.resetPhase_btn.setToolTip('Discard all refinement operations for this phase')
            self.resetPhase_btn.clicked.connect(self.resetPhase)

        # Original phase preview + NavTbar
            self.origCanvas = cObj.HeatMapCanvas(self, binary=True, cbar=False, tight=True)
            self.origCanvas.fig.suptitle('Original', size='x-large')
            self.origCanvas.setMinimumSize(100, 100)

            self.origNTbar = cObj.NavTbar(self.origCanvas, self)
            self.origNTbar.fixHomeAction()
            self.origNTbar.removeToolByIndex([3, 4, 8, 9, 12])

        # Refine phase button
            self.refinePhase_btn = QW.QPushButton('Apply')
            self.refinePhase_btn.setStyleSheet('''background-color: rgb(50,205,50);
                                                  font-weight: bold''')
            self.refinePhase_btn.setToolTip('Refine the phase')
            self.refinePhase_btn.clicked.connect(self.refinePhase)

        # Refined phase preview + NavTbar
            self.refCanvas = cObj.DiscreteClassCanvas(self, tight=True)
            self.refCanvas.fig.suptitle('Refined', size='x-large')
            self.refCanvas.setMinimumSize(100, 100)
            CF.shareAxis(self.origCanvas.ax, self.refCanvas.ax, True)

            self.refNTbar = cObj.NavTbar(self.refCanvas, self)
            self.refNTbar.fixHomeAction()
            self.refNTbar.removeToolByIndex([3, 4, 8, 9, 12])

        # Region Of Interest Selector + Zoom Lock Actions in refined phase NavTbar
            self.roiSelAction = QW.QAction(QIcon('Icons/ROI_selection.png'),
                                          'Select Region Of Interest', self.refNTbar)
            self.roiSelAction.setCheckable(True)
            self.roiSelAction.toggled.connect(self.toggle_roiSel)

            self.zoomLockAction = QW.QAction(QIcon('Icons/lockZoom.png'),
                                            'Lock zoom', self.refNTbar)
            self.zoomLockAction.setCheckable(True)
            self.zoomLockAction.triggered.connect(self.lock_zoom)

            self.refNTbar.insertActions(10, (self.zoomLockAction, 
                                             self.roiSelAction))

        # ROI warning icon
            self.algmWarn = QW.QLabel()
            self.algmWarn.setPixmap(QG.QPixmap('Icons/warnIcon.png').scaled(30, 30,
                                                                         Qt.KeepAspectRatio))
            self.algmWarn.setSizePolicy(QW.QSizePolicy.Maximum,
                                        QW.QSizePolicy.Maximum)
            self.algmWarn.setToolTip('Warning: the selected algorithm ignores the Region Of Interest.')
            self.algmWarn.hide()

        # Rectangle selector widget
            self.rectSel = cObj.RectSel(self.refCanvas.ax, self.select_roi, btns=[1])

        # Preview Area group
            origTbar_hbox = QW.QHBoxLayout()
            origTbar_hbox.addWidget(self.resetPhase_btn, alignment=Qt.AlignLeft)
            origTbar_hbox.addWidget(self.origNTbar, alignment=Qt.AlignLeft)
            origTbar_hbox.addStretch()
            orig_vbox = QW.QVBoxLayout()
            orig_vbox.addLayout(origTbar_hbox)
            orig_vbox.addWidget(self.origCanvas)

            refTbar_hbox = QW.QHBoxLayout()
            refTbar_hbox.addWidget(self.refinePhase_btn, alignment=Qt.AlignLeft)
            refTbar_hbox.addWidget(self.refNTbar, alignment=Qt.AlignLeft)
            refTbar_hbox.addStretch()
            refTbar_hbox.addWidget(self.algmWarn, alignment=Qt.AlignRight)
            ref_vbox = QW.QVBoxLayout()
            ref_vbox.addLayout(refTbar_hbox)
            ref_vbox.addWidget(self.refCanvas)

            preview_hbox = QW.QHBoxLayout()
            preview_hbox.addLayout(orig_vbox)
            preview_hbox.addLayout(ref_vbox)
            preview_group = cObj.GroupArea(preview_hbox, 'Preview')

        # Morphological tool algorithms combobox
            self.algm_combox = QW.QComboBox()
            self.algm_combox.addItems(self.algmDict.keys())
            self.algm_combox.currentTextChanged.connect(self.update_phaseView)
            algm_form = QW.QFormLayout()
            algm_form.addRow('Algorithm', self.algm_combox)

        # Invert mask checkbox
            self.invertMask_cbox = QW.QCheckBox('Invert mask')
            self.invertMask_cbox.stateChanged.connect(self.update_phaseView)

        # Invert ROI checkbox
            self.invertROI_cbox = QW.QCheckBox('Invert ROI')
            self.invertROI_cbox.stateChanged.connect(self.update_phaseView)

        # Kernel size selector widget
            self.kernelSel = cObj.KernelSelector()
            self.kernelSel.structureChanged.connect(self.update_phaseView)

        # Removed pixels behaviour combobox
            self.delPixAs_combox = QW.QComboBox()
            self.delPixAs_combox.addItems(['Nearest',
                                           'ND'])
            self.delPixAs_combox.setToolTip('Indicates how to reclassify pixels deleted by the refinement algorithm.')
            delPixAs_form = QW.QFormLayout()
            delPixAs_form.addRow('Removed Pixels as:', self.delPixAs_combox)

        # Reset all edits button
            self.resetAll_btn = QW.QPushButton(QIcon('Icons/generic_del.png'),
                                               'Reset All')
            self.resetAll_btn.setToolTip('Discard all edits')
            self.resetAll_btn.clicked.connect(self.resetAll)

        # Save edits button
            self.save_btn = QW.QPushButton(QIcon('Icons/save.png'), 'Save')
            self.save_btn.clicked.connect(self.saveEdits)

        # Adjust options group
            saveReset_hbox = QW.QHBoxLayout()
            saveReset_hbox.addWidget(self.resetAll_btn)
            saveReset_hbox.addWidget(self.save_btn)

            options_vbox = QW.QVBoxLayout()
            options_vbox.addLayout(algm_form)
            options_vbox.addWidget(self.invertMask_cbox)
            options_vbox.addWidget(self.invertROI_cbox)
            options_vbox.addWidget(self.kernelSel)
            options_vbox.addLayout(delPixAs_form)
            options_vbox.addLayout(saveReset_hbox)
            options_group = cObj.GroupArea(options_vbox, 'Preferences')


        # Adjust Main Layout
            left_box = QW.QVBoxLayout()
            left_box.addWidget(TOC_group)
            left_box.addWidget(options_group)

            mainSplit = cObj.SplitterGroup((left_box, preview_group), # use SplitterLayout
                                           (1, 2))
            mainLayout = QW.QHBoxLayout()
            mainLayout.addWidget(mainSplit)
            self.setLayout(mainLayout)



        def lock_zoom(self):
            lock = self.sender().isChecked()
            self.origCanvas.toggle_zoomLock(lock)
            self.refCanvas.toggle_zoomLock(lock)

        def toggle_roiSel(self, toggled):
            if not toggled: self.roi = None
            self.rectSel.set_active(toggled)
            self.rectSel.set_visible(toggled)
            self.refCanvas.enablePicking(toggled)
            self.update_phaseView()

        def select_roi (self, eclick, erelease):
            mapShape = self.minmap.shape
            extents = self.rectSel.fixed_extents(mapShape)
            if extents is None : return
            self.roi = extents
            self.update_phaseView()
            self.rectSel.updateCursor()

        def check_ROIwarn(self):
            algm = self.algm_combox.currentText()
            if algm == 'Fill holes' and self.rectSel.active:
                self.algmWarn.show()
            else:
                self.algmWarn.hide()



        def replace_excluded(self, array, strategy):
            mask = array == 'nan' # np.nan becomes 'nan' in a string array

            if strategy == 'Nearest':
                i = nd.distance_transform_edt(mask,
                                              return_distances=False,
                                              return_indices=True)
                array = array[tuple(i)]

            elif strategy == 'ND':
                array[mask] = '_ND_'

            return array

        def refinePhase(self):
            phase = self.TOC.get_checked().text()
            mask = self.minmap == phase
            edits, _ = self._varianceMatrix(self._refine(mask), mask)
            removedAs = self.delPixAs_combox.currentText()
        # Get rid of excluded pixels
            self.minmap[edits == -1] = np.nan
            self.minmap = self.replace_excluded(self.minmap, removedAs)
        # Add the new pixels
            self.minmap[edits == 2] = phase

            self.update_phaseView()

        def resetPhase(self):
            phase = self.TOC.get_checked().text()
            bkp = self.parent._backup
            mask = (bkp == phase) | (self.minmap == phase)
            self.minmap = np.where(mask, bkp, self.minmap)
            self.update_phaseView()

        def resetAll(self):
            choice = QW.QMessageBox.question(self, 'X-Min Learn',
                                                   'Do you really want to discard all the edits?',
                                                   QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                   QW.QMessageBox.No)
            if choice== QW.QMessageBox.Yes:
                self.minmap = self.parent._backup.copy()
                self.update_phaseView()

        def _compute_ROImask(self, mapShape):
            if self.roi is None:
                r0,c0 = 0, 0
                r1,c1 = mapShape
            else:
                r0,r1,c0,c1 = self.roi
            roi_mask = np.zeros(mapShape, dtype=bool)
            roi_mask[r0:r1, c0:c1] = True
            # Invert ROI if required
            if self.invertROI_cbox.isChecked():
                roi_mask = np.invert(roi_mask)
            return roi_mask

        def _refine(self, ph_mask):
        # Get all of the morphological parameters
            algm = self.algm_combox.currentText()
            invert = self.invertMask_cbox.isChecked()
            struct = self.kernelSel.get_structure()
            roi_mask = self._compute_ROImask(ph_mask.shape)
            if invert: ph_mask = np.invert(ph_mask)
        # 'Fill holes' algorithm do not support ROI mask
            if algm == 'Fill holes':
                refined = self.algmDict[algm](ph_mask, struct)
            else:
                refined = self.algmDict[algm](ph_mask, struct, mask=roi_mask)

        # The inversion is only for refinement sake, therefore we invert back the result
            if invert : refined = np.invert(refined)
            return refined

        def _varianceMatrix(self, new, old):
            new = new.astype('int8')
            old = old.astype('int8')
            res = new - old + new
            colors = [v for k,v in self.varianceRGB.items() if k in np.unique(res)]
            return res, colors

        def update_phaseView(self):
            phase = self.TOC.get_checked().text()
            mask = self.minmap == phase
            edits, colors = self._varianceMatrix(self._refine(mask), mask)
            self.check_ROIwarn()

            nPix_orig = np.count_nonzero(mask)
            nPix_edits = np.count_nonzero(edits>0)
            self.origCanvas.update_canvas(f'{phase} = {nPix_orig}', mask)
            self.refCanvas.update_canvas(f'{phase} = {nPix_edits}',
                                         edits, colors = colors)


        def saveEdits(self):
            outpath, _ = QW.QFileDialog.getSaveFileName(self, 'Save Map',
                                                        pref.get_dirPath('out'),
                                                        '''Compressed ASCII file (*.gz)
                                                           ASCII file (*.txt)''')
            if outpath:
                pref.set_dirPath('out', os.path.dirname(outpath))
                np.savetxt(outpath, self.minmap, fmt='%s')
                autoload = QW.QMessageBox.question(self, 'X-Min Learn',
                                                   'Do you want to load the result in the "Mineral Map Tab"?',
                                                   QW.QMessageBox.Yes | QW.QMessageBox.No,
                                                   QW.QMessageBox.Yes)
                if autoload == QW.QMessageBox.Yes:
                    self.mainParent._minmaptab.loadMaps([outpath])





class DataViewer(QW.QWidget):
    '''
    The main tool of X-Min Learn, that allows the visualization of imported
    data. Unlike any other tool, it cannot be moved or closed by users.
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super(DataViewer, self).__init__()

    # Set tool title and icon
        self.setWindowTitle('Data Viewer')
        self.setWindowIcon(QIcon(r'Icons/data_viewer.png'))

    # Set main attributes
        self._displayedObject = None

    # Set GUI
        self._init_ui()
        self.adjustSize()

    def _init_ui(self):
        '''
        GUI constructor.

        '''
    # Maps Canvas
        self.canvas = plots.ImageCanvas()
        self.canvas.customContextMenuRequested.connect(self.showContextMenu)
        self.canvas.enable_picking(True)
        self.canvas.setMinimumWidth(500)

    # Navigation Toolbar
        self.navTbar = plots.NavTbar(self.canvas, self)
        self.navTbar.fixHomeAction()
        self.navTbar.removeToolByIndex([3, 4, 8, 9])

    # Go To Pixel Widget
        self.go2Pix = cObj.PixelFinder(self.canvas)
        self.navTbar.addSeparator()
        self.navTbar.addWidget(self.go2Pix)

    # Current showed map path
        self.currPath = cObj.PathLabel()

    # Adjust Window Layout
        main_layout = QW.QVBoxLayout()
        main_layout.addWidget(self.navTbar)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.currPath)

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
    # Show the menu in the same spot where the user triggered the event
        menu.exec(QG.QCursor.pos())






































