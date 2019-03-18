# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightWidgetTIFFpyqtgraph.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WidgetTiff(object):
    def setupUi(self, WidgetTiff):
        WidgetTiff.setObjectName("WidgetTiff")
        WidgetTiff.resize(750, 428)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(WidgetTiff)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.graphicsView = GraphicsWidget(WidgetTiff)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_2.addWidget(self.graphicsView)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalScrollBar = QtWidgets.QScrollBar(WidgetTiff)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setInvertedAppearance(False)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalLayout.addWidget(self.horizontalScrollBar)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.lcdNumber = QtWidgets.QLCDNumber(WidgetTiff)
        self.lcdNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.lcdNumber.setObjectName("lcdNumber")
        self.horizontalLayout.addWidget(self.lcdNumber)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBoxProperties = QtWidgets.QGroupBox(WidgetTiff)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxProperties.sizePolicy().hasHeightForWidth())
        self.groupBoxProperties.setSizePolicy(sizePolicy)
        self.groupBoxProperties.setObjectName("groupBoxProperties")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBoxProperties)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.formLayoutProperties = QtWidgets.QFormLayout()
        self.formLayoutProperties.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayoutProperties.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayoutProperties.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayoutProperties.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayoutProperties.setObjectName("formLayoutProperties")
        self.SizeLabel = QtWidgets.QLabel(self.groupBoxProperties)
        self.SizeLabel.setObjectName("SizeLabel")
        self.formLayoutProperties.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.SizeLabel)
        self.SizeLabelEdit = QtWidgets.QLabel(self.groupBoxProperties)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SizeLabelEdit.sizePolicy().hasHeightForWidth())
        self.SizeLabelEdit.setSizePolicy(sizePolicy)
        self.SizeLabelEdit.setMinimumSize(QtCore.QSize(90, 20))
        self.SizeLabelEdit.setObjectName("SizeLabelEdit")
        self.formLayoutProperties.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.SizeLabelEdit)
        self.durationMsLabel = QtWidgets.QLabel(self.groupBoxProperties)
        self.durationMsLabel.setObjectName("durationMsLabel")
        self.formLayoutProperties.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.durationMsLabel)
        self.framePeriodMsLabel = QtWidgets.QLabel(self.groupBoxProperties)
        self.framePeriodMsLabel.setObjectName("framePeriodMsLabel")
        self.formLayoutProperties.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.framePeriodMsLabel)
        self.framePeriodMsLineEdit = QtWidgets.QLineEdit(self.groupBoxProperties)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.framePeriodMsLineEdit.sizePolicy().hasHeightForWidth())
        self.framePeriodMsLineEdit.setSizePolicy(sizePolicy)
        self.framePeriodMsLineEdit.setMinimumSize(QtCore.QSize(150, 20))
        self.framePeriodMsLineEdit.setObjectName("framePeriodMsLineEdit")
        self.formLayoutProperties.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.framePeriodMsLineEdit)
        self.frameRateLabel = QtWidgets.QLabel(self.groupBoxProperties)
        self.frameRateLabel.setObjectName("frameRateLabel")
        self.formLayoutProperties.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.frameRateLabel)
        self.frameRateLineEdit = QtWidgets.QLineEdit(self.groupBoxProperties)
        self.frameRateLineEdit.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frameRateLineEdit.sizePolicy().hasHeightForWidth())
        self.frameRateLineEdit.setSizePolicy(sizePolicy)
        self.frameRateLineEdit.setMinimumSize(QtCore.QSize(150, 20))
        self.frameRateLineEdit.setBaseSize(QtCore.QSize(0, 0))
        self.frameRateLineEdit.setClearButtonEnabled(False)
        self.frameRateLineEdit.setObjectName("frameRateLineEdit")
        self.formLayoutProperties.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.frameRateLineEdit)
        self.durationMsLineEdit = QtWidgets.QLineEdit(self.groupBoxProperties)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.durationMsLineEdit.sizePolicy().hasHeightForWidth())
        self.durationMsLineEdit.setSizePolicy(sizePolicy)
        self.durationMsLineEdit.setMinimumSize(QtCore.QSize(150, 20))
        self.durationMsLineEdit.setObjectName("durationMsLineEdit")
        self.formLayoutProperties.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.durationMsLineEdit)
        self.resolutionLineEdit = QtWidgets.QLineEdit(self.groupBoxProperties)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.resolutionLineEdit.sizePolicy().hasHeightForWidth())
        self.resolutionLineEdit.setSizePolicy(sizePolicy)
        self.resolutionLineEdit.setMinimumSize(QtCore.QSize(150, 20))
        self.resolutionLineEdit.setObjectName("resolutionLineEdit")
        self.formLayoutProperties.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.resolutionLineEdit)
        self.resolutionLabel = QtWidgets.QLabel(self.groupBoxProperties)
        self.resolutionLabel.setObjectName("resolutionLabel")
        self.formLayoutProperties.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.resolutionLabel)
        self.horizontalLayout_4.addLayout(self.formLayoutProperties)
        self.verticalLayout.addWidget(self.groupBoxProperties)
        self.groupBoxPreps = QtWidgets.QGroupBox(WidgetTiff)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxPreps.sizePolicy().hasHeightForWidth())
        self.groupBoxPreps.setSizePolicy(sizePolicy)
        self.groupBoxPreps.setObjectName("groupBoxPreps")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBoxPreps)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.treeViewPreps = QtWidgets.QTreeView(self.groupBoxPreps)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeViewPreps.sizePolicy().hasHeightForWidth())
        self.treeViewPreps.setSizePolicy(sizePolicy)
        self.treeViewPreps.setMinimumSize(QtCore.QSize(280, 50))
        self.treeViewPreps.setMaximumSize(QtCore.QSize(16777215, 80))
        self.treeViewPreps.setIndentation(5)
        self.treeViewPreps.setObjectName("treeViewPreps")
        self.treeViewPreps.header().setCascadingSectionResizes(True)
        self.treeViewPreps.header().setDefaultSectionSize(40)
        self.treeViewPreps.header().setMinimumSectionSize(20)
        self.verticalLayout_3.addWidget(self.treeViewPreps)
        self.verticalLayout.addWidget(self.groupBoxPreps)
        self.groupBoxROIs = QtWidgets.QGroupBox(WidgetTiff)
        self.groupBoxROIs.setObjectName("groupBoxROIs")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBoxROIs)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.treeViewROIs = QtWidgets.QTreeView(self.groupBoxROIs)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeViewROIs.sizePolicy().hasHeightForWidth())
        self.treeViewROIs.setSizePolicy(sizePolicy)
        self.treeViewROIs.setMinimumSize(QtCore.QSize(280, 50))
        self.treeViewROIs.setMaximumSize(QtCore.QSize(16777215, 80))
        self.treeViewROIs.setIndentation(5)
        self.treeViewROIs.setObjectName("treeViewROIs")
        self.treeViewROIs.header().setCascadingSectionResizes(True)
        self.treeViewROIs.header().setDefaultSectionSize(40)
        self.treeViewROIs.header().setMinimumSectionSize(20)
        self.verticalLayout_5.addWidget(self.treeViewROIs)
        self.verticalLayout.addWidget(self.groupBoxROIs)
        self.groupBoxAnalysis = QtWidgets.QGroupBox(WidgetTiff)
        self.groupBoxAnalysis.setObjectName("groupBoxAnalysis")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBoxAnalysis)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.treeViewAnalysis = QtWidgets.QTreeView(self.groupBoxAnalysis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeViewAnalysis.sizePolicy().hasHeightForWidth())
        self.treeViewAnalysis.setSizePolicy(sizePolicy)
        self.treeViewAnalysis.setMinimumSize(QtCore.QSize(280, 50))
        self.treeViewAnalysis.setMaximumSize(QtCore.QSize(16777215, 80))
        self.treeViewAnalysis.setIndentation(5)
        self.treeViewAnalysis.setObjectName("treeViewAnalysis")
        self.treeViewAnalysis.header().setCascadingSectionResizes(True)
        self.treeViewAnalysis.header().setDefaultSectionSize(40)
        self.treeViewAnalysis.header().setMinimumSectionSize(20)
        self.verticalLayout_4.addWidget(self.treeViewAnalysis)
        self.verticalLayout.addWidget(self.groupBoxAnalysis)
        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.retranslateUi(WidgetTiff)
        self.horizontalScrollBar.valueChanged['int'].connect(self.lcdNumber.display)
        QtCore.QMetaObject.connectSlotsByName(WidgetTiff)

    def retranslateUi(self, WidgetTiff):
        _translate = QtCore.QCoreApplication.translate
        WidgetTiff.setWindowTitle(_translate("WidgetTiff", "TIFF Viewer"))
        self.groupBoxProperties.setTitle(_translate("WidgetTiff", "Properties"))
        self.SizeLabel.setText(_translate("WidgetTiff", "Width x Height (px)"))
        self.SizeLabelEdit.setText(_translate("WidgetTiff", "---- X ----"))
        self.durationMsLabel.setText(_translate("WidgetTiff", "Duration (ms)"))
        self.framePeriodMsLabel.setText(_translate("WidgetTiff", "Frame Period (ms)"))
        self.framePeriodMsLineEdit.setPlaceholderText(_translate("WidgetTiff", "----"))
        self.frameRateLabel.setText(_translate("WidgetTiff", "Frame Rate (fps)"))
        self.frameRateLineEdit.setPlaceholderText(_translate("WidgetTiff", "----"))
        self.durationMsLineEdit.setPlaceholderText(_translate("WidgetTiff", "----"))
        self.resolutionLineEdit.setPlaceholderText(_translate("WidgetTiff", "----"))
        self.resolutionLabel.setText(_translate("WidgetTiff", "Resolution (cm/px)"))
        self.groupBoxPreps.setTitle(_translate("WidgetTiff", "Preps"))
        self.groupBoxROIs.setTitle(_translate("WidgetTiff", "ROIs"))
        self.groupBoxAnalysis.setTitle(_translate("WidgetTiff", "Analysis"))

from ui.graphicswidget import GraphicsWidget
