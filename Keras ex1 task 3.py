from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

#make sure to pre process the image when uploading it
def preprocess_input_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")  # Load the image and resize to 28x28 grayscale
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28 * 28)).astype('float32') 
    return img_array

def preprocess_input_image2(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255.0  # Reshape and normalize
    return img_array

img_path = r'C:\Users\HP\Documents\mnist_tests\seven.jpg'  
processed_image = preprocess_input_image2(img_path)
#loading the model again ok
new_model = tf.keras.models.load_model('mnistModel2.model')

#making predictions based on the imported model
prediction = new_model.predict([processed_image])

print(np.argmax(prediction))