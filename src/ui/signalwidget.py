

from PyQt5 import QtCore, QtGui, QtWidgets


class SignalWidget(QtWidgets.QWidget):
    """Widget defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        super(SignalWidget, self).__init__(parent)

        self.widget = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("SignalWidget")
        self.horizontalLayout_SignalWidget = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_SignalWidget.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_SignalWidget.setObjectName("horizontalLayout_SignalWidget")
        self.comboBoxSignal = QtWidgets.QComboBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBoxSignal.sizePolicy().hasHeightForWidth())
        self.comboBoxSignal.setSizePolicy(sizePolicy)
        self.comboBoxSignal.setObjectName("comboBoxSignal")
        self.horizontalLayout_SignalWidget.addWidget(self.comboBoxSignal)
        self.checkBoxSignal = QtWidgets.QCheckBox(self.SignalWidget)
        self.checkBoxSignal.setText("")
        self.checkBoxSignal.setCheckable(True)
        self.checkBoxSignal.setChecked(True)
        self.checkBoxSignal.setObjectName("checkBoxSignal")
        self.horizontalLayout_SignalWidget.addWidget(self.checkBoxSignal)
        self.formLayoutSignals.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.widget)
