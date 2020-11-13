import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl


class Viewer:

    def __init__(self) -> None:

        self.app = pg.QtGui.QApplication([])
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('PyFDS Viewer')
        self.window.setGeometry(0, 0, 1000, 500)

        grid = gl.GLGridItem()
        grid.setSize(2000, 2000, 2000)
        grid.setSpacing(20, 20, 20)

        self.window.addItem(grid)
        self.window.setCameraPosition(distance=200)
        self.window.setBackgroundColor('k')

        self.window.show()
        self.window.raise_()

        self.plot_initialised = False

    def update(self, state: np.ndarray):
        pass
