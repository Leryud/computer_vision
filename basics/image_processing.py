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

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


print('Keras image data format: {}'.format(K.image_data_format()))

IMAGES_FOLDER = os.path.join('./images') # images for visuals

MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history

earth_fname = os.path.join(IMAGES_FOLDER, 'earth.jpg')
earth_img = cv2.imread(earth_fname)
# comment out the line below to see the colour difference
earth_img = cv2.cvtColor(earth_img, cv2.COLOR_BGR2RGB)


def img_blur(img, boxsize=41):
    box_blur_img = img.copy()
    box_blur_img = cv2.blur(box_blur_img, (boxsize, boxsize))

    return box_blur_img


def gaussian_blur(img, boxsize=41, kernel_size=10):
    blur_img = img.copy()
    blur_img = cv2.GaussianBlur(blur_img, (boxsize, boxsize), kernel_size)
    return blur_img


def dilate_img(img, kernel_size=10):
    dilated_img = img.copy()
    dilated_img = cv2. dilate(dilated_img,
                              np.ones((kernel_size, kernel_size),
                                      dtype=np.uint8),
                              iterations=1)
    return dilated_img


def erode_img(img, kernel_size=10):
    eroded_img = img.copy()
    eroded_img = cv2.erode(eroded_img,
                           np.ones((kernel_size, kernel_size),
                                    dtype=np.uint8),
                            iterations=1)
    return eroded_img


def canny_img(img, kernel):
    canny_img = img.copy()
    canny_img = erode_img(canny_img, kernel+2)
    thresh = 75
    edges = cv2.Canny(canny_img, thresh, thresh)
    #couldnt figure out how to add subplot with color map gray on plotly subplots
    plt.imshow(edges.astype(np.uint8), cmap='gray')
    plt.show()

    return edges

def threshold_img(img, down_limit=80, up_limit=255):
    threshold_img = img.copy()
    threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(threshold_img, down_limit, up_limit), cv2.THRESH_BINARY)
    plt.imshow(thresh, cmap='gray')
    plt.show()


# plotting images

earth_blurred  = img_blur(earth_img, boxsize=11)
earth_gaussian = gaussian_blur(earth_img, boxsize=11)
earth_dilated  = dilate_img(earth_img, kernel_size=10)
earth_eroded   = erode_img(earth_img, kernel_size=10)

fig = make_subplots(rows=2, cols=4)

fig.add_trace(go.Image(z=earth_blurred), row=1, col=1)
fig.add_trace(go.Image(z=earth_gaussian), row=1, col=2)
fig.add_trace(go.Image(z=earth_dilated), row=1, col=3)
fig.add_trace(go.Image(z=earth_eroded), row=1, col=4)
canny_img(earth_img, kernel=10)
threshold_img(earth_img, 80, 255)
fig.show()