
# Implementation of prediction and model loading support

from keras.models import model_from_json
from keras.models import load_model
import numpy, cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from PyQt6.QtWidgets import QWidget, QFileDialog
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QPoint

from PyQt6.QtGui import QImage

def qimage_to_array(image: QImage):
    """
    A function that converts a QImage object to a numpy array
    """
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    image = image.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)

    # Convert QImage to numpy array
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    numpy_array = np.array(ptr).reshape(28, 28, 1)

    # Normalize pixel values
    numpy_array = numpy_array / 255.0  # Assuming normalization to [0, 1] range
    # using the OpenCV library to display the image after conversion
    #cv2.imshow('Check if the function works!', numpy_array)
    return numpy_array
    



def predict(image: QImage):
    """
    A function that uses the loaded neural network model to predict the sign in the image

    Appropriate code to handle the loaded model should be added here
    """

    numpy_array = qimage_to_array(image)

    # use of the OpenCV library to resize the image to the size of the images used in the MNIST file
    """numpy_array = cv2.resize(numpy_array, (28,28))
    numpy_array = numpy_array.reshape((1, 28* 28)).astype('float32') / 255.0  # Normalize
    """
    # Reshape the image to match the expected input shape of the model
    numpy_array = numpy_array.reshape((-1, 28, 28, 1))  # Reshape to (batch_size, height, width, channels)
    prediction = get_model().predict(numpy_array)

    # using the OpenCV library to display the image after conversion
    #cv2.imshow('Check if the function works!!', numpy_array.reshape(28, 28))

    return np.argmax(prediction)


def get_model():
    """
    Function that loads the learned model of the neural network

     You should add the appropriate code for loading of the model and weights
    """ 
    return tf.keras.models.load_model('mnistModel2.model')  