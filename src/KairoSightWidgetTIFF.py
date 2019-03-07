# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightWidgetTIFF.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_WidgetTiff(object):
    def setupUi(self, WidgetTiff):
        WidgetTiff.setObjectName("WidgetTiff")
        WidgetTiff.resize(566, 438)
        self.verticalLayout = QtWidgets.QVBoxLayout(WidgetTiff)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mpl = MplWidget(WidgetTiff)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl.sizePolicy().hasHeightForWidth())
        self.mpl.setSizePolicy(sizePolicy)
        self.mpl.setObjectName("mpl")
        self.verticalLayout.addWidget(self.mpl)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalScrollBar = QtWidgets.QScrollBar(WidgetTiff)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalLayout.addWidget(self.horizontalScrollBar)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lcdNumber = QtWidgets.QLCDNumber(WidgetTiff)
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setObjectName("lcdNumber")
        self.horizontalLayout.addWidget(self.lcdNumber)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(WidgetTiff)
        self.horizontalScrollBar.valueChanged['int'].connect(self.lcdNumber.display)
        QtCore.QMetaObject.connectSlotsByName(WidgetTiff)

    def retranslateUi(self, WidgetTiff):
        _translate = QtCore.QCoreApplication.translate
        WidgetTiff.setWindowTitle(_translate("WidgetTiff", "TIFF Viewer"))


from mplwidget import MplWidget
