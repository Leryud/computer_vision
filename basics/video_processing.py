import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions

from IPython.display import display as ipydisplay, Image, clear_output, HTML # for interacting with the notebook better

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt # (optional) for plotting and showing images inline

import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
print('Keras image data format: {}'.format(K.image_data_format()))

dir = os.getcwd()
IMAGES_FOLDER = os.path.join(dir, 'images') # images for visuals

MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history


def bgrtorgb(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)


def show_image(name):
    fname = os.path.join(IMAGES_FOLDER, name)
    ipydisplay(Image(fname))


def plot_image(image, figsize=(8, 8), recolour=False):

    if recolour:
        image = bgrtorgb(image)

    plt.figure(figsize=figsize)

    if image.shape[-1] == 3:
        plt.imshow(image)
    elif image.shape[-1] == 1 or len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        raise Exception('Invalid image shape.')

def screenshot():
        fname = input("File name : ")
        cv2.imwrite(os.path.join(IMAGES_FOLDER, f'{fname}.jpg'), frame)
        print(f"Screenshot saved in {IMAGES_FOLDER}")

def print_key(k):
    if k != 255:
        print(k)

video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xff
    print_key(k)
    if k == 27:  #escape pressed
        break
    elif k == 115: #s is pressed
        screenshot()

# * The following code is useful to get out of a loop error when capturing video
cv2.destroyAllWindows()
video.release()