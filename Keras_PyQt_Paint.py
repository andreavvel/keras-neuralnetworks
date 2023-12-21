import sys
from typing import Callable
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtGui import QImage, QPainter, QPen, QBrush, QPixmap, QMouseEvent
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QFormLayout, QGridLayout, QMessageBox, QPushButton, QLineEdit, QPlainTextEdit, QSpinBox
from PyQt6.QtGui import QPainter, QColor, QFont, QColorConstants

### "Paint" implementation


# The size and color of the pen
PEN_WDTH = 3
PEN_COLOR = QColorConstants.White

# The size of the drawing field PIXMAP_SIZE x PIXMAP_SIZE
PIXMAP_SIZE = 256

# Simple Paint implementation
class Paint(QtWidgets.QMainWindow):

    def __init__(self, predict_function):
        """
        predict_function - function called when drawing is finished.
         Should return the value (number) that was returned by the neural network
        """
        super().__init__()

        # The main widget that stores the layout
        self.window = QWidget()

        # Creating a window in which it will be possible to draw
        self.paint = QtWidgets.QLabel()
        self.paint.setFixedWidth(PIXMAP_SIZE)
        self.paint.setFixedHeight(PIXMAP_SIZE)
        self.image = QImage(PIXMAP_SIZE, PIXMAP_SIZE, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.black)

        # Creating a storing layout: 
        # - drawing window 
        # - image clearing button 
        # - window displaying the neural network response
        self.layout = QGridLayout()
        self.prediction = QLineEdit()
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)   

        self.layout.addWidget(self.prediction,1,0) 
        self.layout.addWidget(self.clear_button,1,1) 
        self.layout.addWidget(self.paint,0,0) 

        self.prediction.setDisabled(1)
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)

        # Variables that hold the last position of the mouse 
        self.lastPoint = QPoint()

        self.predict_function = predict_function

    def clear(self):
        """
        A function that clears the drawing area
        """
        self.image.fill(QColorConstants.Black)
        self.prediction.clear()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        """
        Function called when the mouse button is pressed.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            painter = QPainter(self.image)  
            painter.setPen(QPen(PEN_COLOR, PEN_WDTH))
            painter.drawPoint(event.pos())
            self.drawing = True  
            self.lastPoint = event.pos()
            self.update()


    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Function called when the mouse is moved with the mouse button pressed.
        """
        if (event.buttons() & Qt.MouseButton.LeftButton) == Qt.MouseButton.LeftButton and self.drawing:
            painter = QPainter(self.image) 
            painter.setPen(QPen(PEN_COLOR, PEN_WDTH))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Function called when the mouse button is released.
        """
        if event.button == Qt.MouseButton.LeftButton:
            self.drawing = False

        if self.predict_function:
            #self.prediction.setText(str(self.predict_function(self.image)))
            prediction_conv = self.predict_function(self.image)
            self.prediction.setText(f"Prediction: {prediction_conv}")
        

    def paintEvent(self, event: QtGui.QPaintEvent):
        """
        Function called when the window is updating
        """
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.paint.rect(), self.image, self.image.rect())

   


import Keras_PyQt_Paint_Model as kppm

app = QtWidgets.QApplication(sys.argv)
window = Paint(kppm.predict)
window.show()
app.exec()