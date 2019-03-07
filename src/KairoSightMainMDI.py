# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightMainMDI.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MDIMainWindow(object):
    def setupUi(self, MDIMainWindow):
        MDIMainWindow.setObjectName("MDIMainWindow")
        MDIMainWindow.resize(894, 691)
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 894, 21))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuPrepare = QtWidgets.QMenu(self.menubar)
        self.menuPrepare.setObjectName("menuPrepare")
        self.menuIsolate = QtWidgets.QMenu(self.menubar)
        self.menuIsolate.setObjectName("menuIsolate")
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
        self.actionStart_wizard = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_wizard.setObjectName("actionStart_wizard")
        self.actionStart_wizard_2 = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_wizard_2.setObjectName("actionStart_wizard_2")
        self.actionStart_wizard_3 = QtWidgets.QAction(MDIMainWindow)
        self.actionStart_wizard_3.setObjectName("actionStart_wizard_3")
        self.menuOpen.addAction(self.actionTIFF)
        self.menuOpen.addAction(self.actionFolder)
        self.menuOpen.addSeparator()
        self.menuPrepare.addAction(self.actionStart_wizard)
        self.menuIsolate.addAction(self.actionStart_wizard_2)
        self.menuAnalyze.addAction(self.actionStart_wizard_3)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.menubar.addAction(self.menuPrepare.menuAction())
        self.menubar.addAction(self.menuIsolate.menuAction())
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuExport.menuAction())

        self.retranslateUi(MDIMainWindow)
        QtCore.QMetaObject.connectSlotsByName(MDIMainWindow)

    def retranslateUi(self, MDIMainWindow):
        _translate = QtCore.QCoreApplication.translate
        MDIMainWindow.setWindowTitle(_translate("MDIMainWindow", "MVP1"))
        self.menuOpen.setTitle(_translate("MDIMainWindow", "Open"))
        self.menuPrepare.setTitle(_translate("MDIMainWindow", "Prepare"))
        self.menuIsolate.setTitle(_translate("MDIMainWindow", "Isolate"))
        self.menuAnalyze.setTitle(_translate("MDIMainWindow", "Analyze"))
        self.menuExport.setTitle(_translate("MDIMainWindow", "Export"))
        self.actionTIFF.setText(_translate("MDIMainWindow", "TIFF"))
        self.actionTIFF.setToolTip(_translate("MDIMainWindow", ".tiff, .tif"))
        self.actionClose.setText(_translate("MDIMainWindow", "Close"))
        self.actionFolder.setText(_translate("MDIMainWindow", "Folder"))
        self.actionStart_wizard.setText(_translate("MDIMainWindow", "Start wizard"))
        self.actionStart_wizard_2.setText(_translate("MDIMainWindow", "Start wizard"))
        self.actionStart_wizard_3.setText(_translate("MDIMainWindow", "Start wizard"))


