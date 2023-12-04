from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
from matplotlib import pyplot as plt
import tensorflow as tf

NUMBERS_CLASSIF= [0,1,2,3,4,5,6,7,8,9]

#loading the model again ok
new_model = tf.keras.models.load_model('mnistModel.model')

#making predictions based on the imported model
predictions = new_model.predict([X_test])