# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightWidgetExportCopyPaste.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WidgetExportCopyPaste(object):
    def setupUi(self, WidgetExportCopyPaste):
        WidgetExportCopyPaste.setObjectName("WidgetExportCopyPaste")
        WidgetExportCopyPaste.resize(600, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WidgetExportCopyPaste.sizePolicy().hasHeightForWidth())
        WidgetExportCopyPaste.setSizePolicy(sizePolicy)
        WidgetExportCopyPaste.setMinimumSize(QtCore.QSize(600, 300))
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(WidgetExportCopyPaste)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.labelSource = QtWidgets.QLabel(WidgetExportCopyPaste)
        self.labelSource.setObjectName("labelSource")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelSource)
        self.comboBoxSource = QtWidgets.QComboBox(WidgetExportCopyPaste)
        self.comboBoxSource.setObjectName("comboBoxSource")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBoxSource)
        self.labelAnalysis = QtWidgets.QLabel(WidgetExportCopyPaste)
        self.labelAnalysis.setObjectName("labelAnalysis")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.labelAnalysis)
        self.comboBoxAnalysis = QtWidgets.QComboBox(WidgetExportCopyPaste)
        self.comboBoxAnalysis.setObjectName("comboBoxAnalysis")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBoxAnalysis)
        self.horizontalLayout.addLayout(self.formLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioButton = QtWidgets.QRadioButton(WidgetExportCopyPaste)
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(WidgetExportCopyPaste)
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.horizontalLayout.addLayout(self.verticalLayout)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.radioButtonAllResults = QtWidgets.QRadioButton(WidgetExportCopyPaste)
        self.radioButtonAllResults.setObjectName("radioButtonAllResults")
        self.verticalLayout_3.addWidget(self.radioButtonAllResults)
        self.checkBoxAPDs = QtWidgets.QCheckBox(WidgetExportCopyPaste)
        self.checkBoxAPDs.setObjectName("checkBoxAPDs")
        self.verticalLayout_3.addWidget(self.checkBoxAPDs)
        self.checkBoxPeriods = QtWidgets.QCheckBox(WidgetExportCopyPaste)
        self.checkBoxPeriods.setObjectName("checkBoxPeriods")
        self.verticalLayout_3.addWidget(self.checkBoxPeriods)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.tableViewResults = QtWidgets.QTableView(WidgetExportCopyPaste)
        self.tableViewResults.setObjectName("tableViewResults")
        self.verticalLayout_2.addWidget(self.tableViewResults)

        self.retranslateUi(WidgetExportCopyPaste)
        QtCore.QMetaObject.connectSlotsByName(WidgetExportCopyPaste)

    def retranslateUi(self, WidgetExportCopyPaste):
        _translate = QtCore.QCoreApplication.translate
        WidgetExportCopyPaste.setWindowTitle(_translate("WidgetExportCopyPaste", "Export for Copy + Paste"))
        self.labelSource.setText(_translate("WidgetExportCopyPaste", "Source:"))
        self.labelAnalysis.setText(_translate("WidgetExportCopyPaste", "Analysis"))
        self.radioButton.setText(_translate("WidgetExportCopyPaste", "Individual"))
        self.radioButton_2.setText(_translate("WidgetExportCopyPaste", "Mean"))
        self.radioButtonAllResults.setText(_translate("WidgetExportCopyPaste", "All Results"))
        self.checkBoxAPDs.setText(_translate("WidgetExportCopyPaste", "APDs"))
        self.checkBoxPeriods.setText(_translate("WidgetExportCopyPaste", "Periods"))

