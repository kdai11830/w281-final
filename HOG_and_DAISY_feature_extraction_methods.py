from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
from skimage import feature
import colorsys
import os
import random
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


def extract_RGB_or_HSV_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
    RGB_or_HSV_or_gray="gray",
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    if RGB_or_HSV_or_gray == "R":
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "G":
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "B":
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "H":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "S":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "V":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("specify a channel or grayscale...")

    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_DAISY_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
    RGB_or_HSV_or_gray="gray",
):
    """
    DAISY on RGB, HSV or grayscale channels.
    """
    if RGB_or_HSV_or_gray == "R":
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "G":
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "B":
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "H":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 0]
    elif RGB_or_HSV_or_gray == "S":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 1]
    elif RGB_or_HSV_or_gray == "V":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]
    elif RGB_or_HSV_or_gray == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print("specify a channel or grayscale...")

    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Grayscale   # # #


def extract_small_gray_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_gray_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_gray_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Red   # # #


def extract_small_red_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = img[:, :, 2]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_red_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = img[:, :, 2]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_red_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = img[:, :, 2]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Green   # # #


def extract_small_green_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = img[:, :, 1]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_green_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = img[:, :, 1]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_green_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = img[:, :, 1]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Blue   # # #


def extract_small_blue_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = img[:, :, 0]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_blue_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = img[:, :, 0]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_blue_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = img[:, :, 0]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Hue   # # #


def extract_small_hue_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_hue_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_hue_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Saturation   # # #


def extract_small_saturation_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_saturation_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_saturation_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()


# # #   Brightness   # # #


def extract_small_brightness_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(20, 20),
    cells_per_block=(1, 1),
):
    """ """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_large_brightness_hog_features(
    img,
    orientations=9,
    pixels_per_cell=(60, 60),
    cells_per_block=(1, 1),
):
    """
    HOG on RGB, HSV or grayscale channels.

    from skimage import data, exposure
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    hog = feature.hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )
    return hog


def extract_brightness_daisy_features(
    img,
    step=220,
    radius=70,
    rings=3,
    histograms=8,
    orientations=8,
):
    """ """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    daisy = feature.daisy(
        img,
        step=step,
        radius=radius,
        rings=rings,
        histograms=histograms,
        orientations=orientations,
    )
    return daisy.flatten()
