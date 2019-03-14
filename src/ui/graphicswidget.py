

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget

from pyqtgraph.widgets.GraphicsLayoutWidget import GraphicsLayoutWidget
from pyqtgraph import ImageItem, HistogramLUTItem


class GraphicsWidget(QWidget):
    """Widget defined in Qt Designer"""

    def __init__(self, parent=None):
        # initialization of Qt MainWindow widget
        super(GraphicsWidget, self).__init__(parent)

        # Create a central Graphics Layout Widget
        self.widget = GraphicsLayoutWidget()

        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.widget.addPlot()
        # Item for displaying image data
        self.img = ImageItem()
        self.p1.addItem(self.img)

        # create a vertical box layout
        self.vbl = QVBoxLayout()
        # add widget to vertical box
        self.vbl.addWidget(self.widget)
        # set the layout to the vertical box
        self.setLayout(self.vbl)

        # Levels/color control with a histogram
        # TODO try with a HistogramLUTWidget
        self.hist = HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.widget.addItem(self.hist)
        self.hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
