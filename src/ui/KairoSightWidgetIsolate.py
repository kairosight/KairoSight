# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KairoSightWidgetIsolate.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_WidgetIsolate(object):
    def setupUi(self, WidgetIsolate):
        WidgetIsolate.setObjectName("WidgetIsolate")
        WidgetIsolate.resize(350, 443)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(WidgetIsolate.sizePolicy().hasHeightForWidth())
        WidgetIsolate.setSizePolicy(sizePolicy)
        WidgetIsolate.setMaximumSize(QtCore.QSize(350, 16777215))
        self.verticalLayout = QtWidgets.QVBoxLayout(WidgetIsolate)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelSource = QtWidgets.QLabel(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelSource.sizePolicy().hasHeightForWidth())
        self.labelSource.setSizePolicy(sizePolicy)
        self.labelSource.setTextFormat(QtCore.Qt.PlainText)
        self.labelSource.setObjectName("labelSource")
        self.horizontalLayout_2.addWidget(self.labelSource)
        self.comboBoxSource = QtWidgets.QComboBox(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxSource.sizePolicy().hasHeightForWidth())
        self.comboBoxSource.setSizePolicy(sizePolicy)
        self.comboBoxSource.setObjectName("comboBoxSource")
        self.horizontalLayout_2.addWidget(self.comboBoxSource)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.labelPreps = QtWidgets.QLabel(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelPreps.sizePolicy().hasHeightForWidth())
        self.labelPreps.setSizePolicy(sizePolicy)
        self.labelPreps.setTextFormat(QtCore.Qt.PlainText)
        self.labelPreps.setObjectName("labelPreps")
        self.horizontalLayout_7.addWidget(self.labelPreps)
        self.comboBoxPreps = QtWidgets.QComboBox(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxPreps.sizePolicy().hasHeightForWidth())
        self.comboBoxPreps.setSizePolicy(sizePolicy)
        self.comboBoxPreps.setObjectName("comboBoxPreps")
        self.horizontalLayout_7.addWidget(self.comboBoxPreps)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.labelROIs = QtWidgets.QLabel(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.labelROIs.sizePolicy().hasHeightForWidth())
        self.labelROIs.setSizePolicy(sizePolicy)
        self.labelROIs.setTextFormat(QtCore.Qt.PlainText)
        self.labelROIs.setObjectName("labelROIs")
        self.horizontalLayout_5.addWidget(self.labelROIs)
        self.comboBoxROIs = QtWidgets.QComboBox(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxROIs.sizePolicy().hasHeightForWidth())
        self.comboBoxROIs.setSizePolicy(sizePolicy)
        self.comboBoxROIs.setObjectName("comboBoxROIs")
        self.comboBoxROIs.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBoxROIs)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.tabWidgetRoiTypes = QtWidgets.QTabWidget(WidgetIsolate)
        self.tabWidgetRoiTypes.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidgetRoiTypes.sizePolicy().hasHeightForWidth())
        self.tabWidgetRoiTypes.setSizePolicy(sizePolicy)
        self.tabWidgetRoiTypes.setMaximumSize(QtCore.QSize(350, 16777215))
        self.tabWidgetRoiTypes.setTabBarAutoHide(False)
        self.tabWidgetRoiTypes.setObjectName("tabWidgetRoiTypes")
        self.tabCircle = QtWidgets.QWidget()
        self.tabCircle.setObjectName("tabCircle")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tabCircle)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.formLayoutCenter = QtWidgets.QFormLayout()
        self.formLayoutCenter.setObjectName("formLayoutCenter")
        self.originXLabel = QtWidgets.QLabel(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.originXLabel.sizePolicy().hasHeightForWidth())
        self.originXLabel.setSizePolicy(sizePolicy)
        self.originXLabel.setObjectName("originXLabel")
        self.formLayoutCenter.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.originXLabel)
        self.originXLineEdit = QtWidgets.QLineEdit(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.originXLineEdit.sizePolicy().hasHeightForWidth())
        self.originXLineEdit.setSizePolicy(sizePolicy)
        self.originXLineEdit.setMinimumSize(QtCore.QSize(40, 0))
        self.originXLineEdit.setObjectName("originXLineEdit")
        self.formLayoutCenter.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.originXLineEdit)
        self.originYLabel = QtWidgets.QLabel(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.originYLabel.sizePolicy().hasHeightForWidth())
        self.originYLabel.setSizePolicy(sizePolicy)
        self.originYLabel.setObjectName("originYLabel")
        self.formLayoutCenter.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.originYLabel)
        self.originYLineEdit = QtWidgets.QLineEdit(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.originYLineEdit.sizePolicy().hasHeightForWidth())
        self.originYLineEdit.setSizePolicy(sizePolicy)
        self.originYLineEdit.setMinimumSize(QtCore.QSize(40, 0))
        self.originYLineEdit.setObjectName("originYLineEdit")
        self.formLayoutCenter.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.originYLineEdit)
        self.horizontalLayout_3.addLayout(self.formLayoutCenter)
        self.formLayoutRadius = QtWidgets.QFormLayout()
        self.formLayoutRadius.setObjectName("formLayoutRadius")
        self.radiusLabel = QtWidgets.QLabel(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radiusLabel.sizePolicy().hasHeightForWidth())
        self.radiusLabel.setSizePolicy(sizePolicy)
        self.radiusLabel.setObjectName("radiusLabel")
        self.formLayoutRadius.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.radiusLabel)
        self.radiusSpinBox = QtWidgets.QSpinBox(self.tabCircle)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radiusSpinBox.sizePolicy().hasHeightForWidth())
        self.radiusSpinBox.setSizePolicy(sizePolicy)
        self.radiusSpinBox.setMinimumSize(QtCore.QSize(50, 0))
        self.radiusSpinBox.setObjectName("radiusSpinBox")
        self.formLayoutRadius.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.radiusSpinBox)
        self.horizontalLayout_3.addLayout(self.formLayoutRadius)
        self.widgetPreview = GraphicsLayoutWidget(self.tabCircle)
        self.widgetPreview.setMinimumSize(QtCore.QSize(70, 70))
        self.widgetPreview.setMaximumSize(QtCore.QSize(70, 70))
        self.widgetPreview.setObjectName("widgetPreview")
        self.horizontalLayout_3.addWidget(self.widgetPreview)
        self.tabWidgetRoiTypes.addTab(self.tabCircle, "")
        self.tabPoly = QtWidgets.QWidget()
        self.tabPoly.setObjectName("tabPoly")
        self.tabWidgetRoiTypes.addTab(self.tabPoly, "")
        self.verticalLayout.addWidget(self.tabWidgetRoiTypes)
        self.groupBoxTime = QtWidgets.QGroupBox(WidgetIsolate)
        self.groupBoxTime.setObjectName("groupBoxTime")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBoxTime)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.checkBoxTimeAll = QtWidgets.QCheckBox(self.groupBoxTime)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBoxTimeAll.sizePolicy().hasHeightForWidth())
        self.checkBoxTimeAll.setSizePolicy(sizePolicy)
        self.checkBoxTimeAll.setChecked(False)
        self.checkBoxTimeAll.setObjectName("checkBoxTimeAll")
        self.horizontalLayout_4.addWidget(self.checkBoxTimeAll)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.startLabel = QtWidgets.QLabel(self.groupBoxTime)
        self.startLabel.setObjectName("startLabel")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.startLabel)
        self.startSpinBox = QtWidgets.QSpinBox(self.groupBoxTime)
        self.startSpinBox.setEnabled(True)
        self.startSpinBox.setMinimumSize(QtCore.QSize(60, 0))
        self.startSpinBox.setMinimum(1)
        self.startSpinBox.setMaximum(999999)
        self.startSpinBox.setObjectName("startSpinBox")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.startSpinBox)
        self.endSpinBox = QtWidgets.QSpinBox(self.groupBoxTime)
        self.endSpinBox.setMinimumSize(QtCore.QSize(60, 0))
        self.endSpinBox.setMinimum(1)
        self.endSpinBox.setMaximum(999999)
        self.endSpinBox.setObjectName("endSpinBox")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.endSpinBox)
        self.endLabel = QtWidgets.QLabel(self.groupBoxTime)
        self.endLabel.setObjectName("endLabel")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.endLabel)
        self.horizontalLayout_4.addLayout(self.formLayout_3)
        self.label = QtWidgets.QLabel(self.groupBoxTime)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout.addWidget(self.groupBoxTime)
        self.widgetPreview_2 = GraphicsLayoutWidget(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetPreview_2.sizePolicy().hasHeightForWidth())
        self.widgetPreview_2.setSizePolicy(sizePolicy)
        self.widgetPreview_2.setMinimumSize(QtCore.QSize(70, 100))
        self.widgetPreview_2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.widgetPreview_2.setObjectName("widgetPreview_2")
        self.verticalLayout.addWidget(self.widgetPreview_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.checkBoxPreview = QtWidgets.QCheckBox(WidgetIsolate)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.checkBoxPreview.setFont(font)
        self.checkBoxPreview.setObjectName("checkBoxPreview")
        self.horizontalLayout.addWidget(self.checkBoxPreview)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.buttonBox = QtWidgets.QDialogButtonBox(WidgetIsolate)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.buttonBox.sizePolicy().hasHeightForWidth())
        self.buttonBox.setSizePolicy(sizePolicy)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply|QtWidgets.QDialogButtonBox.Discard|QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(WidgetIsolate)
        self.tabWidgetRoiTypes.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(WidgetIsolate)

    def retranslateUi(self, WidgetIsolate):
        _translate = QtCore.QCoreApplication.translate
        WidgetIsolate.setWindowTitle(_translate("WidgetIsolate", "Isolate"))
        self.labelSource.setText(_translate("WidgetIsolate", "Source:"))
        self.labelPreps.setText(_translate("WidgetIsolate", "Prep:"))
        self.labelROIs.setText(_translate("WidgetIsolate", "ROI:"))
        self.comboBoxROIs.setItemText(0, _translate("WidgetIsolate", "*New*"))
        self.originXLabel.setText(_translate("WidgetIsolate", "Origin, X (px)"))
        self.originXLineEdit.setText(_translate("WidgetIsolate", "0"))
        self.originYLabel.setText(_translate("WidgetIsolate", "Origin, Y (px)"))
        self.originYLineEdit.setText(_translate("WidgetIsolate", "0"))
        self.radiusLabel.setText(_translate("WidgetIsolate", "Radius (px)"))
        self.tabWidgetRoiTypes.setTabText(self.tabWidgetRoiTypes.indexOf(self.tabCircle), _translate("WidgetIsolate", "Circle Region"))
        self.tabWidgetRoiTypes.setTabText(self.tabWidgetRoiTypes.indexOf(self.tabPoly), _translate("WidgetIsolate", "Poly Region"))
        self.groupBoxTime.setTitle(_translate("WidgetIsolate", "Time Span"))
        self.checkBoxTimeAll.setText(_translate("WidgetIsolate", "Use All"))
        self.startLabel.setText(_translate("WidgetIsolate", "Start"))
        self.endLabel.setText(_translate("WidgetIsolate", "End"))
        self.label.setText(_translate("WidgetIsolate", "frames"))
        self.checkBoxPreview.setText(_translate("WidgetIsolate", "Preview"))

from pyqtgraph.widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
