#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from ui.KairoSight_WindowMain import Ui_WindowMain
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow


class WindowMain(QWidget, Ui_WindowMain):
    """Customization for Ui_WindowMain"""

    def __init__(self, parent=None):
        super(WindowMain, self).__init__(parent)    # initialization of the superclass
        self.setupUi(self)  # setup the UI


if __name__ == '__main__':
    # create the GUI application
    app = QApplication(sys.argv)
    # instantiate and show the main window
    dmw = WindowMain()
    dmw.show()
    # start the Qt main loop execution, exiting from this script
    # with the same return code as the Qt application
    sys.exit(app.exec_())
