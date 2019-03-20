#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import traceback
import numpy as np
from pathlib import PurePath

from skimage import io
from PyQt5 import QtCore
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QFileSystemModel, QDialogButtonBox
import pyqtgraph as pg
from ui.KairoSightMainMDI import Ui_MDIMainWindow
from ui.KairoSightWidgetTIFFpyqtgraph import Ui_WidgetTiff
from ui.KairoSightWidgetFolderTree import Ui_WidgetFolderTree
from ui.KairoSightWidgetImagePrep import Ui_WidgetImagePrep
from ui.KairoSightWidgetIsolate import Ui_WidgetIsolate
from ui.KairoSightWidgetAnalyze import Ui_WidgetAnalyze
from algorithms import tifopen


class DesignerMainWindow(QMainWindow, Ui_MDIMainWindow):
    """Customization for Qt Designer created window"""

    def __init__(self, parent=None):
        # initialization of the superclass
        super(DesignerMainWindow, self).__init__(parent)
        # setup the GUI --> function generated by pyuic4
        self.setupUi(self)
        # connect the signals with the slots
        # self.actionLoad.triggered.connect(self.open_tiff)
        # self.actionClose.triggered.connect(self.close)
        self.actionTIFF.triggered.connect(self.open_tiff)
        self.actionFolder.triggered.connect(self.open_folder)
        self.actionStart_ImagePrep.triggered.connect(self.image_prep)
        self.actionStart_Isolate.triggered.connect(self.isolate)
        self.actionStart_Analyze.triggered.connect(self.analyze)

    def open_tiff(self, file=None):
        """Open a SubWindow with a TIFF stack in the main MDI area"""
        if file:
            print('Opening tiff with passed filepath: ' + file)
        else:
            # Use a QFileDialog to get filepath if none provided
            file, mask = QFileDialog.getOpenFileName(self, 'Open a .tif/.tiff stack')

        if file:
            self.statusBar().showMessage('Opening ' + file + ' ...')
            f_purepath = PurePath(file)
            f_path = str(f_purepath.parent) + '\\'
            f_name = f_purepath.stem
            f_ext = f_purepath.suffix
            f_display = f_path + ' ' + f_name + ' ' + f_ext
            print('file (path name ext): ' + f_display)
            if f_ext is '.tif' or '.tiff':
                print('TIFF chosen')
                # Create QMdiSubWindow with Ui_WidgetTiff
                try:
                    sub = DesignerSubWindowTiff(f_path=f_path, f_name=f_name, f_ext=f_ext)
                    print('DesignerSubWindowTiff "sub" created')
                    sub.setObjectName(str(file))
                    sub.setWindowTitle('TIFF View: ' + f_display)
                    # Add and connect QMdiSubWindow to MDI
                    self.mdiArea.addSubWindow(sub)
                    print('"sub" added to MDI')
                    sub.show()
                    self.statusBar().showMessage('Opened ' + file)
                except Exception:
                    traceback.print_exc()
                    self.statusBar().showMessage('Failed to open, ' + file)
        else:
            print('path is None')
            self.statusBar().showMessage('Open cancelled')

    def open_folder(self):
        """Open a SubWindow with a folder tree view in the main MDI area"""
        folder_path = QFileDialog.getExistingDirectory(self, 'Choose a folder to view')
        print('Folder chosen! path: ' + folder_path)
        # Create QMdiSubWindow with Ui_WidgetTiff
        sub = DesignerSubWindowFolder(root=folder_path)
        print('DesignerSubWindowFolder "sub" created')
        print('Set "sub" widget to "Ui_WidgetFolderTree"')
        sub.setWindowTitle('Folder View: ' + folder_path)
        # Add and connect QMdiSubWindow to MDI
        self.mdiArea.addSubWindow(sub)
        sub.pushButtonOpen.released.connect(lambda: self.open_tiff(sub.currentFilePath))
        print('"sub" added to MDI')
        sub.show()

    def image_prep(self):
        """Open the Image Process SubWindow"""
        tiff_windows = []
        # Create a list of all open TIFF subwindows
        for sub in self.mdiArea.subWindowList():
            # print('**' + str(type(sub.widget())) + ', ' + sub.widget().objectName() + ' is a tiff? ')
            if type(sub.widget()) is DesignerSubWindowTiff:
                tiff_windows.append(sub)
        if tiff_windows:
            sub_process = DesignerSubWindowImagePrep(w_list=tiff_windows)
            self.mdiArea.addSubWindow(sub_process)
            sub_process.show()
        else:
            self.statusBar().showMessage('No open videos to process!')

    def isolate(self):
        """Open the Isolate SubWindow"""
        tiff_windows = []
        # Create a list of all open TIFF subwindows
        for sub in self.mdiArea.subWindowList():
            # print('**' + str(type(sub.widget())) + ', ' + sub.widget().objectName() + ' is a tiff? ')
            if type(sub.widget()) is DesignerSubWindowTiff:
                if sub.widget().Preps:
                    tiff_windows.append(sub)
        if tiff_windows:
            sub_iso = DesignerSubWindowIsolate(w_list=tiff_windows)
            self.mdiArea.addSubWindow(sub_iso)
            sub_iso.show()
        else:
            self.statusBar().showMessage('No processed videos to isolate!')

    def analyze(self):
        """Open the Analyze SubWindow"""
        tiff_windows = []
        # Create a list of all open TIFF subwindows
        for sub in self.mdiArea.subWindowList():
            # print('**' + str(type(sub.widget())) + ', ' + sub.widget().objectName() + ' is a tiff? ')
            if type(sub.widget()) is DesignerSubWindowTiff:
                if sub.widget().ROIs:
                    tiff_windows.append(sub)
        if tiff_windows:
            sub_analyze = DesignerSubWindowAnalyze(w_list=tiff_windows)
            self.mdiArea.addSubWindow(sub_analyze)
            sub_analyze.show()
        else:
            self.statusBar().showMessage('No processed videos with ROIs to analyze!')


class DesignerSubWindowTiff(QWidget, Ui_WidgetTiff):
    """Customization for WidgetTiff subwindow for an MDI"""
    # TODO Build a better data tree for Preps, ROIs, and Analysis
    INDEX_P, TRANSFORM, BACKGROUND = range(3)
    INDEX_R, PREP, TYPE, POSITION, SIZE, TIME = range(6)
    INDEX_A, ROI, TYPE, ROI_CALC, PEAKS, PROCESS = range(6)

    def __init__(self, parent=None, f_path=None, f_name=None, f_ext=None):
        # Initialization of the superclass
        super(DesignerSubWindowTiff, self).__init__(parent)
        # Setup the GUI
        self.setupUi(self)
        pg.setConfigOptions(background=pg.mkColor(0.1))
        pg.setConfigOptions(foreground=pg.mkColor(0.3))
        # Preserve plot area's aspect ration so image always scales correctly
        self.graphicsView.p1.setAspectLocked(True)
        # Connect the scrollbar's value signal to trigger a video update
        self.horizontalScrollBar.valueChanged['int'].connect(self.updateVideo)
        # Load the video file
        self.video_path = f_path
        self.video_name = f_name
        self.video_ext = f_ext
        self.video_file, self.dt = tifopen.tifopen(self.video_path, self.video_name + self.video_ext)
        print('tifopen finished')
        # get video properties
        self.video_shape = self.video_file.shape
        if len(self.video_shape) < 3:
            raise Exception('TIFF has less than 3 dimensions')
        self.frames = self.video_shape[0]

        # Transpose second and third axes (y, x) to correct orientation (x, y)
        self.video_data = np.transpose(self.video_file, (0, 2, 1))
        # Flip each frame in the left/right direction, expected to be up/down
        for i in range(self.frames):
            self.video_data[i] = np.fliplr(self.video_data[i])

        self.fps = 1000 / self.dt
        self.duration = self.fps * (self.frames + 1)
        self.width, self.height = self.video_shape[2], self.video_shape[1]
        print('video shape:         ', self.video_shape)
        print('Width x Height:      ', self.width, self.height)
        print('# of Frames:         ', self.frames)
        print('Frame Period (ms):   ', self.dt)
        print('FPS:                 ', self.fps)
        print('Duration (ms):       ', self.duration)
        self.SizeLabelEdit.setText(str(self.width) + ' X ' + str(self.height))
        if not np.isnan(self.dt):
            self.framePeriodMsLineEdit.setText(str(self.dt))
            self.framePeriodMsLineEdit.setEnabled(False)
            self.frameRateLineEdit.setText(str(self.fps))
            self.frameRateLineEdit.setEnabled(False)
            self.durationMsLineEdit.setText(str(self.duration))
            self.durationMsLineEdit.setEnabled(False)

        # Setup Preps, ROIs, and Anlysis variables
        self.Preps = []  # A list of prep dictionaries
        self.prep_default = {'transform': 'NaN', 'background': 'NaN'}
        self.ROIs = []  # A list of pg.ROI objects
        self.Analysis = []  # A list of Analysis results dictionaries
        self.analysis_default = {'ROI': 'NaN', 'INDEX_A': 'NaN', 'TYPE': 'NaN',
                                 'ROI_CALC': 'NaN', 'PEAKS': 'NaN', 'PROCESS': 'NaN'}
        # Set scroll bar maximum to number of frames
        self.horizontalScrollBar.setMinimum(1)
        self.horizontalScrollBar.setMaximum(self.frames)
        self.frame_current = 0
        # Set histogram to image levels and use a manual range
        self.graphicsView.hist.setLevels(self.video_data.min(), self.video_data.max())
        self.graphicsView.hist.setHistogramRange(self.video_data.min(), self.video_data.max())

        # Setup data treeviews
        self.treeViewPreps.setAlternatingRowColors(True)
        self.treeViewROIs.setAlternatingRowColors(True)
        self.treeViewAnalysis.setAlternatingRowColors(True)
        # Preps model
        self.modelPrep = QStandardItemModel(0, 3)
        self.modelPrep.setHeaderData(self.INDEX_P, Qt.Horizontal, "#")
        self.modelPrep.setHeaderData(self.TRANSFORM, Qt.Horizontal, "Transform")
        self.modelPrep.setHeaderData(self.BACKGROUND, Qt.Horizontal, "Background")
        self.treeViewPreps.setModel(self.modelPrep)
        # ROI model
        self.modelRoi = QStandardItemModel(0, 6)
        self.modelRoi.setHeaderData(self.INDEX_R, Qt.Horizontal, "#")
        self.modelRoi.setHeaderData(self.PREP, Qt.Horizontal, "Prep#")
        self.modelRoi.setHeaderData(self.TYPE, Qt.Horizontal, "Type")
        self.modelRoi.setHeaderData(self.POSITION, Qt.Horizontal, "Position (X,Y)")
        self.modelRoi.setHeaderData(self.SIZE, Qt.Horizontal, "Size (px)")
        self.modelRoi.setHeaderData(self.TIME, Qt.Horizontal, "Time (frames)")
        self.treeViewROIs.setModel(self.modelRoi)
        # Analysis model
        self.modelAnalysis = QStandardItemModel(0, 6)
        self.modelAnalysis.setHeaderData(self.INDEX_A, Qt.Horizontal, "#")
        self.modelAnalysis.setHeaderData(self.ROI, Qt.Horizontal, "ROI#")
        self.modelAnalysis.setHeaderData(self.TYPE, Qt.Horizontal, "Type")
        self.modelAnalysis.setHeaderData(self.ROI_CALC, Qt.Horizontal, "ROI Calc.")
        self.modelAnalysis.setHeaderData(self.PEAKS, Qt.Horizontal, "Peak Det.")
        self.modelAnalysis.setHeaderData(self.PROCESS, Qt.Horizontal, "Results")
        self.treeViewAnalysis.setModel(self.modelAnalysis)
        # TODO Use double-click to view analysis results

        # Add default prep, with no transform or background removal
        self.addPrep(self.prep_default)
        print('WidgetTiff ready')

    def updateVideo(self, frame=0):
        """Updates the video frame drawn to the canvas"""
        # print('Updating video plot in a subWindow with:')
        print('*** Showing ' + self.video_name + '[' + str(frame) + ']')
        # Update ImageItem with a frame in stack
        self.frame_current = frame
        self.graphicsView.img.setImage(self.video_data[frame - 1])
        # Notify histogram item of image change
        self.graphicsView.hist.regionChanged()
        try:
            if self.ROIs:
                # Draw ROIs
                for roi in self.ROIs:
                    self.graphicsView.p1.addItem(roi)
        except Exception:
            traceback.print_exc()

    def getRoiPreview(self, roi):
        data = self.video_data[self.frame_current]
        data_img = self.graphicsView.img
        data_preview = roi.getArrayRegion(data, data_img)
        return data_preview

    def addPrep(self, prep=None):
        if prep:
            print('** Adding passed prep: ', prep)
            self.Preps.append(prep)
            transform = prep['transform']
            background = prep['background']

            length = self.modelPrep.rowCount()
            self.modelPrep.insertRow(length)
            self.modelPrep.setData(self.modelPrep.index(length, self.INDEX_P), length)
            self.modelPrep.setData(self.modelPrep.index(length, self.TRANSFORM), transform)
            self.modelPrep.setData(self.modelPrep.index(length, self.BACKGROUND), background)
            print('** Preps are now: ', self.Preps)
        else:
            print('** No Prep to add!')

    def removePrep(self, prep=None):
        if prep:
            print('** Removing passed prep: ', prep)
        else:
            print('** No Prep to remove!')

    def addROI(self, prep=0, roi=None, time=None):
        # TODO change TIME to FRAMES for clarity
        if roi:
            print('** Adding passed ROI: ', roi)
            roi.translatable = False
            roi_state = roi.getState()
            x, y = str(int(roi_state['pos'].x())), str(int(roi_state['pos'].y()))
            position = x + ',' + y
            r = int(roi_state['size'][0])
            if not time:
                time = '0-' + str(self.frames)
            else:
                time = str(time[0]) + '-' + str(time[0])
            roi_new = pg.CircleROI([x, y], [r, r], pen=(2, 9), movable=False)
            roi_new.setPen(color='54FF00')
            self.graphicsView.p1.addItem(roi_new)
            self.ROIs.append(roi_new)
            roi_new.removeHandle(0)
            length = self.modelRoi.rowCount()
            self.modelRoi.insertRow(length)
            self.modelRoi.setData(self.modelRoi.index(length, self.INDEX_R), str(length))
            self.modelRoi.setData(self.modelRoi.index(length, self.PREP), str(prep))
            self.modelRoi.setData(self.modelRoi.index(length, self.TYPE), 'Circle')
            self.modelRoi.setData(self.modelRoi.index(length, self.POSITION), position)
            self.modelRoi.setData(self.modelRoi.index(length, self.SIZE), r)
            self.modelRoi.setData(self.modelRoi.index(length, self.TIME), time)
            # print('** ROIs are now: ', self.ROIs)
        else:
            print('** No ROI to add!')

    def addAnalysis(self, analysis=None):
        if analysis:
            print('** Adding passed Analysis: ', analysis)
            # roi_state = roi.getState()
            # x, y = str(int(roi_state['pos'].x())), str(int(roi_state['pos'].y()))
            # position = x + ',' + y
            # r = int(roi_state['size'][0])
            # if not time:
            #     time = '0-' + str(self.frames)
            # else:
            #     time = str(time[0]) + '-' + str(time[0])
            # roi_new = pg.CircleROI([x, y], [r, r], pen=(2, 9), movable=False)
            # self.Analysis.append(analysis)
            # self.graphicsView.p1.addItem(roi_new)
            length = self.modelAnalysis.rowCount()
            self.modelAnalysis.insertRow(length)
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.INDEX_A), str(length))
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.ROI), 'ROI#')
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.TYPE), 'Vm or Ca')
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.ROI_CALC), 'MEAN')
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.PEAKS), 'THR,LOT')
            self.modelAnalysis.setData(self.modelAnalysis.index(length, self.PROCESS), 'ALL')
        else:
            print('** No Prep to add!')


class DesignerSubWindowFolder(QWidget, Ui_WidgetFolderTree):
    """Customization for WidgetFolderTree subwindow for an MDI"""

    def __init__(self, parent=None, root=None):
        # initialization of the superclass
        super(DesignerSubWindowFolder, self).__init__(parent)
        self.dir = QDir(root)
        self.currentFileName = ''
        self.currentFilePath = ''
        # setup the GUI
        self.setupUi(self)
        print('WidgetFolderTree UI setup')
        self.model = QFileSystemModel()
        self.model.setRootPath(root)
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index(root))
        print('treeView ready')

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_treeView_clicked(self, index):
        index_item = self.model.index(index.row(), 0, index.parent())
        self.currentFileName = self.model.fileName(index_item)
        self.currentFilePath = self.model.filePath(index_item)
        print('Clicked: ' + self.currentFilePath + ' ' + self.currentFileName)


class DesignerSubWindowImagePrep(QWidget, Ui_WidgetImagePrep):
    """Customization for WidgetFolderTree subwindow for an MDI"""
    # TODO Add preps with entered parameters
    # TODO Add functionality to Discard button

    def __init__(self, parent=None, w_list=None):
        # initialization of the superclass
        super(DesignerSubWindowImagePrep, self).__init__(parent)
        print('Creating WidgetImagePrep')
        self.windowList = w_list
        self.currentFileName = ''
        self.currentFilePath = ''
        self.windowDict = {}  # Dictionary with structure "window_name_short": "window"
        name_limit = 50
        for w in self.windowList:
            w_name = w.widget().objectName()
            w_name_short = '..' + w_name[-name_limit:] if len(w_name) > name_limit else w_name
            # Populate dictionary
            self.windowDict[w_name_short] = w.widget()
        self.currentWindow = None
        self.currentPlot = None
        self.currentPreps = []
        self.prep_preview = {'transform': 'NeW', 'background': 'NeW'}
        # setup the GUI
        self.setupUi(self)
        print('WidgetImagePrep UI setup...')
        self.comboBoxSource.addItems(self.windowDict.keys())
        self.comboBoxSource.currentIndexChanged['int'].connect(self.selectionMadeSource)
        self.comboBoxPreps.currentIndexChanged['int'].connect(self.selectionMadePrep)

        self.rotateComboBox.setEnabled(False)
        self.thresholdSpinBox.setEnabled(False)
        self.minSizeSpinBox.setEnabled(False)
        self.checkBoxApplyTransform.stateChanged.connect(self.checkBoxChangedTransform)
        self.checkBoxApplyRemoveBackground.stateChanged.connect(self.checkBoxChangedRemoveBackground)

        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(self.applyPrep)
        self.buttonBox.button(QDialogButtonBox.Discard).clicked.connect(self.discardPrep)
        # self.checkBoxPreview.stateChanged.connect(self.checkBoxChangedPreview)
        self.selectionMadeSource(0)
        # self.listWidgetOpenTiffs.addItems(self.windowListNames)
        print('WidgetImagePrep ready')

    def selectionMadeSource(self, i):
        print('** selection made in a ', type(self))
        print('*Current: ', self.comboBoxSource.currentText())
        self.currentWindow = self.windowDict[self.comboBoxSource.currentText()]
        self.currentPlot = self.currentWindow.graphicsView.p1
        self.currentPreps = self.currentWindow.Preps

        self.currentPreps = self.currentWindow.Preps
        self.comboBoxPreps.clear()
        self.comboBoxPreps.addItem('*New*')
        for idx, prep in enumerate(self.currentPreps):
            self.comboBoxPreps.addItem('#' + str(idx) + ': ' + str(prep))
            print('Listing Prep #', idx, ': ', prep)

        print('*Window: ', str(self.currentWindow))
        print('*W x H: ', str(self.currentWindow.width), ' X ', str(self.currentWindow.height))
        print('Preps: ', str(self.currentPreps))


    def selectionMadePrep(self, i):
        """Slot for comboBoxPreps.currentIndexChanged"""
        try:
            print('** selection made in a ', type(self))
            print('* Current Prep: ', self.comboBoxPreps.currentText())
            # for prep in self.currentWindow.Preps:

            index_current = self.comboBoxPreps.currentIndex()
            prep_current = self.currentWindow.Preps[index_current - 1]
            print('* Prep: ', str(prep_current))
            # prep_current.setPen(color='FF000A')
            # self.updateParameters(prep_current)
        except Exception:
            traceback.print_exc()


    def loadDefaults(self):
        """Populate Prep parameter inputs with default values"""
        self.rotateComboBox.setEnabled(False)

        self.thresholdSpinBox.setEnabled(False)
        self.minSizeSpinBox.setEnabled(False)

    def checkBoxChangedTransform(self):
        """Enable or disable Transform parameter entry"""
        # print('*Preview checkbox changed to: ', self.checkBoxPreview.isChecked())
        if self.checkBoxApplyTransform.isChecked():
            self.rotateComboBox.setEnabled(True)
        else:
            self.rotateComboBox.setEnabled(False)

    def checkBoxChangedRemoveBackground(self):
        """Enable or disable Remove Background parameter entry"""
        # print('*Preview checkbox changed to: ', self.checkBoxPreview.isChecked())
        if self.checkBoxApplyRemoveBackground.isChecked():
            self.thresholdSpinBox.setEnabled(True)
            self.minSizeSpinBox.setEnabled(True)
        else:
            self.thresholdSpinBox.setEnabled(False)
            self.minSizeSpinBox.setEnabled(False)

    def applyPrep(self):
        # self.prep_preview['transform'] =
        # self.prep_preview['backgroud'] =
        self.currentWindow.addPrep(self.prep_preview)
        self.selectionMadeSource(0)
        self.prep_preview = {'transform': 'NeW', 'background': 'NeW'}

    def discardPrep(self):
        if len(self.currentPreps) < 2:
            print('Cannot discard default prep!')
        else:
            self.currentWindow.removePrep(self.prep_preview)


class DesignerSubWindowIsolate(QWidget, Ui_WidgetIsolate):
    """Customization for WidgetFolderTree subwindow for an MDI"""
    # TODO Add functionality to Discard button
    # TODO Populate currentROIs based on selected Prep

    def __init__(self, parent=None, w_list=None):
        # initialization of the superclass
        super(DesignerSubWindowIsolate, self).__init__(parent)
        self.windowList = w_list
        self.windowDict = {}  # Dictionary with structure "window_name_short": "window"
        name_limit = 50
        for w in self.windowList:
            w_name = w.widget().objectName()
            w_name_short = '..' + w_name[-name_limit:] if len(w_name) > name_limit else w_name
            # Populate dictionary
            self.windowDict[w_name_short] = w.widget()
        self.currentWindow = None
        self.currentPlot = None
        self.currentPreps = []
        self.currentROIs = []
        self.roi_preview = None
        # setup the GUI
        self.setupUi(self)
        # Setup preview plot in isolate subwindow
        # TODO remove border around preview, maybe switch to:
        #         self.rawImg = RawImageWidget(QWidget())
        w_preview = self.widgetPreview.addLayout(row=0, col=0)
        self.v_preview = w_preview.addViewBox(lockAspect=True)
        self.img_preview = pg.ImageItem()

        self.v_preview.addItem(self.img_preview)
        self.v_preview.disableAutoRange('xy')
        self.v_preview.autoRange()
        print('WidgetIsolate UI setup...')
        self.comboBoxSource.addItems(self.windowDict.keys())
        self.comboBoxSource.currentIndexChanged['int'].connect(self.selectionMadeSource)
        self.comboBoxPreps.currentIndexChanged['int'].connect(self.selectionMadePrep)
        self.comboBoxROIs.currentIndexChanged['int'].connect(self.selectionMadeROI)

        self.originXLineEdit.textEdited.connect(self.checkBoxChangedPreview)
        self.originYLineEdit.textEdited.connect(self.checkBoxChangedPreview)
        self.radiusLineEdit.valueChanged.connect(self.checkBoxChangedPreview)

        self.buttonBox.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.loadDefaults)
        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(self.applyROI)
        self.checkBoxPreview.stateChanged.connect(self.checkBoxChangedPreview)
        self.selectionMadeSource(0)
        print('WidgetIsolate ready')

    def closeEvent(self, event):
        """Reimplementation of QWidget.closeEvent

        Parameters
        ----------
        event : PySide2.QtGui.QCloseEvent
            Event when Qt receives a window close request for a top-level widget from the window system
        """
        self.checkBoxPreview.setChecked(False)
        event.accept()

    def selectionMadeSource(self, i):
        """Slot for comboBoxSource.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current source: ', self.comboBoxSource.currentText())
        self.currentWindow = self.windowDict[self.comboBoxSource.currentText()]
        self.currentPlot = self.currentWindow.graphicsView.p1
        self.currentPreps = self.currentWindow.Preps
        self.comboBoxPreps.clear()
        for idx, prep in enumerate(self.currentPreps):
            self.comboBoxPreps.addItem('#' + str(idx) + ': ' + str(prep))
            print('Listing Prep #', idx, ': ', prep)

        self.currentROIs = self.currentWindow.ROIs
        self.comboBoxROIs.clear()
        self.comboBoxROIs.addItem('*New*')
        for idx, roi in enumerate(self.currentROIs):
            self.comboBoxROIs.addItem(str(idx) + ': ' + str(roi.saveState()))
            print('Listing ROI #', idx, ': ', roi)

        print('* Window: ', str(self.currentWindow))
        print('* W x H: ', str(self.currentWindow.width), ' X ', str(self.currentWindow.height))
        print('* Preps: ', str(self.currentPreps))
        print('* ROIs: ', str(self.currentROIs))
        self.loadDefaults()

    def selectionMadePrep(self, i):
        """Slot for comboBoxPreps.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current Prep: ', self.comboBoxPreps.currentText())
        # for prep in self.currentWindow.Preps:

        index_current = self.comboBoxPreps.currentIndex()
        prep_current = self.currentWindow.Preps[index_current]
        print('* Prep: ', str(prep_current))
        # prep_current.setPen(color='FF000A')
        # self.updateParameters(prep_current)

    def selectionMadeROI(self, i):
        """Slot for comboBoxROIs.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current ROI: ', self.comboBoxROIs.currentText())
        for roi in self.currentWindow.ROIs:
            roi.setPen(color='54FF00')
        index_current = self.comboBoxROIs.currentIndex()
        if index_current > 0:
            # An existing ROI has been selected
            roi_current = self.currentWindow.ROIs[index_current - 1]
            print('* ROI: ', str(roi_current))
            roi_current.setPen(color='FF000A')
            self.updateParameters(roi_current)
        else:
            # *NEW* has been selected
            self.loadDefaults()

    def loadDefaults(self):
        """Populate ROI parameter inputs with default values"""
        default_r = 15
        if self.currentWindow:
            try:
                # Populate fields with default values
                self.originXLineEdit.setText(str(int(self.currentWindow.width / 2)))
                self.originYLineEdit.setText(str(int(self.currentWindow.height / 2)))
                self.radiusLineEdit.setValue(default_r)
            except Exception:
                traceback.print_exc()
        else:
            print('No tiff windows available')

    def updateParameters(self, roi):
        """Populate ROI parameter inputs with an existing ROI's parameters"""
        roi_state = roi.getState()
        x, y = str(int(roi_state['pos'].x())), str(int(roi_state['pos'].y()))
        r = int(roi_state['size'][0])
        print("Updating region with: ", x, ' ', y, ' ', r)
        # Populate fields with passed values
        self.originXLineEdit.setText(x)
        self.originYLineEdit.setText(y)
        self.radiusLineEdit.setValue(r)

    def checkBoxChangedPreview(self):
        """Create or destroy preview ROI based on the Preview checkbox"""
        # print('*Preview checkbox changed to: ', self.checkBoxPreview.isChecked())
        if self.checkBoxPreview.isChecked():
            if not self.roi_preview:
                # Get current ROI values
                x, y = int(self.originXLineEdit.text()), int(self.originYLineEdit.text())
                r = self.radiusLineEdit.value()
                # Create preview ROI if it doesn't exist
                self.roi_preview = pg.CircleROI([x, y], [r, r], pen=(2, 9), scaleSnap=True, translateSnap=True)
                self.roi_preview.setPen(color='FF8700')
                # Draw region on current tiff window's plot
                # print('Adding roi_preview')
                self.currentPlot.addItem(self.roi_preview)
                # self.currentROIs.append(self.roi_preview)
                self.roi_preview.sigRegionChangeFinished.connect(lambda: self.updateParameters(self.roi_preview))
                self.roi_preview.sigRegionChanged.connect(self.updatePreview)
                self.updatePreview()
            else:
                # ROI Preview exists, update the params
                # Get current ROI values
                x, y = int(self.originXLineEdit.text()), int(self.originYLineEdit.text())
                r = self.radiusLineEdit.value()
                self.roi_preview.setPos((x, y))
                self.roi_preview.setSize(r)
        else:
            # Remove preview ROI
            # print('Removing roi_preview')
            self.currentPlot.removeItem(self.roi_preview)
            self.roi_preview = None

    def updatePreview(self):
        """Updates ROI preview image in Isolate subwindow"""
        if self.roi_preview:
            # Get current video frame data and preview ROI data
            data_frame = self.currentWindow.video_data[self.currentWindow.frame_current]
            data_preview = self.currentWindow.getRoiPreview(self.roi_preview)

            # self.roi_preview.setParentItem(img_preview)
            # Draw preview data in isolate subwindow
            self.img_preview.setImage(data_preview, levels=(0, data_frame.max()))
            self.v_preview.autoRange()
        else:
            print('No ROI preview to update!')

    def applyROI(self):
        """Adds an ROI to a TIFF or applies changes to an existing ROI"""
        if self.comboBoxROIs.currentIndex() is 0:
            if not self.roi_preview:
                print('No ROI to add')
                return
            # Add the preview ROI to the current TIFF window
            prepIndex = self.comboBoxPreps.currentIndex()
            prepText = self.comboBoxPreps.currentText()
            prepData = self.comboBoxPreps.currentData()
            print("*Adding ROI with prepIndex " + str(prepIndex))
            print("*Adding ROI with prepText " + prepText)
            print("*Adding ROI with prepData " + str(prepData))
            self.currentWindow.addROI(prepIndex, self.roi_preview)
        else:
            # Set state of the chosen ROI (current list index - 1, due to *NEW* at index 0)
            roi_current = self.currentROIs[self.comboBoxROIs.currentIndex() - 1]
            roi_current.setState(self.roi_preview.saveState())
            roi_current.setPen(color='54FF00')
        self.checkBoxPreview.setChecked(False)
        self.selectionMadeSource(0)


class DesignerSubWindowAnalyze(QWidget, Ui_WidgetAnalyze):
    """Customization for WidgetFolderTree subwindow for an MDI"""

    # TODO Plot data at each step of analysis
    # TODO Add functionality to Discard button
    def __init__(self, parent=None, w_list=None):
        # initialization of the superclass
        super(DesignerSubWindowAnalyze, self).__init__(parent)
        print('Creating WidgetAnalyze')
        self.windowList = w_list
        self.currentFileName = ''
        self.currentFilePath = ''
        self.windowDict = {}  # Dictionary with structure "window_name_short": "window"
        name_limit = 50
        for w in self.windowList:
            w_name = w.widget().objectName()
            w_name_short = '..' + w_name[-name_limit:] if len(w_name) > name_limit else w_name
            # Populate dictionary
            self.windowDict[w_name_short] = w.widget()
        self.currentWindow = None
        self.currentPlot = None
        self.currentPreps = []
        self.currentROIs = []
        self.currentAnalysis = []
        self.analysis_preview = None
        # setup the GUI
        self.setupUi(self)
        print('WidgetAnalyze UI setup...')
        self.tabPeakDetect.setEnabled(False)
        self.tabProcess.setEnabled(False)

        self.comboBoxSource.addItems(self.windowDict.keys())
        self.comboBoxSource.currentIndexChanged['int'].connect(self.selectionMadeSource)
        self.comboBoxPreps.currentIndexChanged['int'].connect(self.selectionMadePrep)
        self.comboBoxROIs.currentIndexChanged['int'].connect(self.selectionMadeROI)
        self.comboBoxAnalysis.currentIndexChanged['int'].connect(self.selectionMadeAnalysis)

        self.buttonBoxCondition.button(QDialogButtonBox.Apply).clicked.connect(self.applyCondition)
        self.buttonBoxPeakDetect.button(QDialogButtonBox.Apply).clicked.connect(self.applyPeakDetect)
        self.buttonBoxAnalyze.button(QDialogButtonBox.Ok).clicked.connect(self.applyAnalyze)
        # self.checkBoxPreview.stateChanged.connect(self.checkBoxChangedPreview)
        self.progressBar.setValue(0)
        self.selectionMadeSource(0)
        # self.listWidgetOpenTiffs.addItems(self.windowListNames)
        print('WidgetAnalyze ready')

    def selectionMadeSource(self, i):
        """Slot for comboBoxSource.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current source: ', self.comboBoxSource.currentText())
        self.currentWindow = self.windowDict[self.comboBoxSource.currentText()]
        self.currentPlot = self.currentWindow.graphicsView.p1
        self.currentPreps = self.currentWindow.Preps
        self.comboBoxPreps.clear()
        for idx, prep in enumerate(self.currentPreps):
            self.comboBoxPreps.addItem(str(prep))
            self.comboBoxPreps.setItemText(idx, '#' + str(idx))
            print('Listing Prep #', idx, ': ', prep)

        self.comboBoxROIs.clear()
        self.currentROIs = self.currentWindow.ROIs
        for idx, roi in enumerate(self.currentROIs):
            self.comboBoxROIs.addItem(str(idx) + ': ' + str(roi.saveState()))
            print('Listing ROI #', idx, ': ', prep)

        self.comboBoxAnalysis.clear()
        self.currentAnalysis = self.currentWindow.Analysis
        self.comboBoxAnalysis.addItem('*New*')
        for idx, analysis in enumerate(self.currentAnalysis):
            self.comboBoxAnalysis.addItem(str(idx) + ': ' + str(analysis))
            print('Listing Analysis #', idx, ': ', analysis)

        print('* Window: ', str(self.currentWindow))
        print('* W x H: ', str(self.currentWindow.width), ' X ', str(self.currentWindow.height))
        print('* Preps: ', str(self.currentPreps))
        print('* ROIs: ', str(self.currentROIs))
        print('* Analysis: ', str(self.currentAnalysis))
        # self.loadDefaults()

    def selectionMadePrep(self, i):
        """Slot for comboBoxPreps.currentIndexChanged"""
        try:
            print('** selection made in a ', type(self))
            print('* Current Prep: ', self.comboBoxPreps.currentText())
            # for prep in self.currentWindow.Preps:

            index_current = self.comboBoxPreps.currentIndex()
            prep_current = self.currentWindow.Preps[index_current]
            print('* Prep: ', str(prep_current))
            # prep_current.setPen(color='FF000A')
            # self.updateParameters(prep_current)
        except Exception:
            traceback.print_exc()

    def selectionMadeROI(self, i):
        """Slot for comboBoxROIs.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current ROI: ', self.comboBoxROIs.currentText())
        for roi in self.currentWindow.ROIs:
            roi.setPen(color='54FF00')
        index_current = self.comboBoxROIs.currentIndex()
        # An existing ROI has been selected
        roi_current = self.currentWindow.ROIs[index_current]
        print('* ROI: ', str(roi_current))
        roi_current.setPen(color='FF000A')
        self.progressBar.setValue(20)

    def selectionMadeAnalysis(self, i):
        """Slot for comboBoxAnalysis.currentIndexChanged"""
        print('** selection made in a ', type(self))
        print('* Current Analysis: ', self.comboBoxROIs.currentText())
        index_current = self.comboBoxROIs.currentIndex()
        # An existing ROI has been selected
        self.analysis_preview = self.currentWindow.ROIs[index_current]
        print('* Analysis: ', str(self.analysis_preview))

    def applyCondition(self):
        # TODO Use parameters to plot conditioned ROI data
        if self.tabProcess.isEnabled():
            self.tabProcess.setEnabled(False)
        self.tabPeakDetect.setEnabled(True)
        self.tabWidgetAnalysisSteps.setCurrentWidget(self.tabPeakDetect)
        self.progressBar.setValue(60)

    def applyPeakDetect(self):
        # TODO Use parameters to add peaks to ROI data plot
        self.tabProcess.setEnabled(True)
        self.progressBar.setValue(100)
        self.tabWidgetAnalysisSteps.setCurrentWidget(self.tabProcess)

    def applyAnalyze(self):
        # TODO Pass an analysis dict to addAnalysis
        self.currentWindow.addAnalysis(self.currentWindow.analysis_default)
        # self.currentWindow.addAnalysis(self.analysis_preview)
        self.selectionMadeSource(0)
        if self.tabProcess.isEnabled():
            self.tabProcess.setEnabled(False)
        if self.tabPeakDetect.isEnabled():
            self.tabPeakDetect.setEnabled(False)
        self.progressBar.setValue(20)


# create the GUI application
app = QApplication(sys.argv)
# instantiate the main window
dmw = DesignerMainWindow()
# show it
dmw.show()
# start the Qt main loop execution, exiting from this script
# with the same return code as the Qt application
sys.exit(app.exec_())
