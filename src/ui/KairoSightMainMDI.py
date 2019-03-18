# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightMainMDI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MDIMainWindow(object):
    def setupUi(self, MDIMainWindow):
        MDIMainWindow.setObjectName("MDIMainWindow")
        MDIMainWindow.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MDIMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName("mdiArea")
        self.verticalLayout.addWidget(self.mdiArea)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MDIMainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MDIMainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuImagePrerp = QtWidgets.QMenu(self.menubar)
        self.menuImagePrerp.setObjectName("menuImagePrerp")
        self.menuIsolateSignals = QtWidgets.QMenu(self.menubar)
        self.menuIsolateSignals.setObjectName("menuIsolateSignals")
        self.menuAnalyze = QtWidgets.QMenu(self.menubar)
        self.menuAnalyze.setObjectName("menuAnalyze")
        self.menuExport = QtWidgets.QMenu(self.menubar)
        self.menuExport.setObjectName("menuExport")
        MDIMainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MDIMainWindow)
        self.statusbar.setObjectName("statusbar")
        MDIMainWindow.setStatusBar(self.statusbar)
        self.actionTIFF = QtWidgets.QAction(MDIMainWindow)
        self.actionTIFF.setObjectName("actionTIFF")
        self.actionClose = QtWidgets.QAction(MDIMainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionFolder = QtWidgets.QAction(MDIMainWindow)
        self.actionFolder.setObjectName("actionFolder")
        self.actionStart_ImagePrep = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_ImagePrep.setObjectName("actionStart_ImagePrep")
        self.actionStart_Isolate = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_Isolate.setObjectName("actionStart_Isolate")
        self.actionStart_Analyze = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_Analyze.setObjectName("actionStart_Analyze")
        self.menuOpen.addAction(self.actionTIFF)
        self.menuOpen.addAction(self.actionFolder)
        self.menuOpen.addSeparator()
        self.menuImagePrerp.addAction(self.actionStart_ImagePrep)
        self.menuIsolateSignals.addAction(self.actionStart_Isolate)
        self.menuAnalyze.addAction(self.actionStart_Analyze)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuImagePrerp.menuAction())
        self.menubar.addAction(self.menuIsolateSignals.menuAction())
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())

        self.retranslateUi(MDIMainWindow)
        QtCore.QMetaObject.connectSlotsByName(MDIMainWindow)

    def retranslateUi(self, MDIMainWindow):
        _translate = QtCore.QCoreApplication.translate
        MDIMainWindow.setWindowTitle(_translate("MDIMainWindow", "MVP1"))
        self.menuOpen.setTitle(_translate("MDIMainWindow", "1) Open"))
        self.menuImagePrerp.setTitle(_translate("MDIMainWindow", "2) Image Prep"))
        self.menuIsolateSignals.setTitle(_translate("MDIMainWindow", "3) Isolate Signals"))
        self.menuAnalyze.setTitle(_translate("MDIMainWindow", "4) Analyze"))
        self.menuExport.setTitle(_translate("MDIMainWindow", "Export"))
        self.actionTIFF.setText(_translate("MDIMainWindow", "TIFF"))
        self.actionTIFF.setToolTip(_translate("MDIMainWindow", ".tiff, .tif"))
        self.actionClose.setText(_translate("MDIMainWindow", "Close"))
        self.actionFolder.setText(_translate("MDIMainWindow", "Folder"))
        self.actionStart_ImagePrep.setText(_translate("MDIMainWindow", "Start wizard"))
        self.actionStart_Isolate.setText(_translate("MDIMainWindow", "Start wizard"))
        self.actionStart_Analyze.setText(_translate("MDIMainWindow", "Start wizard"))

