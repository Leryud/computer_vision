# %%
# import sys # system functions (ie. exiting the program)
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

# %%

IMAGES_FOLDER = '/Users/leo/Documents/dev/computer_vision/images'

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


def static_BST(bg, current):

    bg_img_path = os.path.join(IMAGES_FOLDER, f"{bg}.jpg")
    current_img_path = os.path.join(IMAGES_FOLDER, f"{current}.jpg")

    bg_img = cv2.imread(bg_img_path)
    current_img = cv2.imread(current_img_path)

    diff = cv2.absdiff(bg_img, current_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_blur = cv2.GaussianBlur(mask_thresh, (3, 3), 10)
    mask_erosion = cv2.erode(mask_blur, np.ones((5, 5), dtype=np.uint8), iterations=1)

    mask_indexes = mask_erosion > 0

    foreground =np.zeros_like(current_img, dtype=np.uint8)
    for i, row in enumerate(mask_indexes):
        foreground[i, row] = current_img[i, row]

    plot_image(bg_img, recolour=True)
    plot_image(current_img, recolour=True)
    plot_image(diff, recolour=True)
    plot_image(mask)
    plot_image(mask_erosion)
    plot_image(foreground, recolour=True)

# %%
static_BST("bg", "fg")
# %%

def motion_BST():

    ERODE = True
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg = cv2.createBackgroundSubtractorKNN()

    video = cv2.VideoCapture(os.path.join(IMAGES_FOLDER, 'bg_substract_movement.mp4'))

    while True:
        time.sleep(0.025)

        timer = cv2.getTickCount()

        success, frame = video.read()
        if not success:
            break

        fgmask = fgbg.apply(frame) 

        if ERODE:
            fgmask = cv2.erode(fgmask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(fgmask, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("fgmask", fgmask)

        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    cv2.destroyAllWindows()
    video.release()


motion_BST()